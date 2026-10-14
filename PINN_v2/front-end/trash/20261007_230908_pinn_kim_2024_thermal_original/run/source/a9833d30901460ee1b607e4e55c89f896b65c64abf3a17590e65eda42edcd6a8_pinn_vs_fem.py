"""Phase 6 study: PINN against the converged FEniCSx field on V3-V5.

The FEM temperature at the finest mesh of a phase6 fem_ladder study is passed to evaluation only
(reference_used_in_loss is False, fem_labels_used is 0). The study refuses a case whose FEM did not converge
unless allow_unconverged is set, and it refuses a FEM file whose evaluation points differ from the problem's.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pinn_v2.cases.definition import CaseDefinition
from pinn_v2.experiments.fem_ladder import LADDER, V2_DIR, is_definition
from pinn_v2.experiments.pinn_baselines import split_spec, write_report
from pinn_v2.manifest.run_manifest import build_manifest, write_manifest, write_json_atomic
from pinn_v2.solvers.pinn import trainer
from pinn_v2.solvers.pinn.problem import load_problem

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = dict(adam_steps=6000, lbfgs_steps=100, lr=3e-3, width=32, depth=3)
HEAVY = dict(points=dict(n_interior=256, n_layer=64, n_face=64, n_boundary=64), adam_steps=3000, lbfgs_steps=0)


def source_of(case):
    return CaseDefinition.load(V2_DIR.joinpath(f'{case}.json')) if is_definition(case) else case


def fem_reference(fem_root, case, problem):
    block = json.loads(Path(fem_root).joinpath('summary.json').read_text(encoding='utf-8'))['cases'][case]
    level = block['records'][-1]['level']
    data = np.load(Path(fem_root).joinpath(case, 'base', f'fem_l{level}.npz'))
    xy, _, _ = problem.evaluation()
    if data['xy'].shape != xy.shape or not np.allclose(data['xy'], xy, atol=1e-12):
        raise ValueError(f'FEM evaluation points of {case} differ from the PINN problem; no reference is built')
    return np.asarray(data['T']).ravel(), block['converged'], level


def run_study(study_id, fem_study, cases=None, architectures=('M1',), seeds=(0,), root=ROOT, allow_unconverged=False, adam_steps=None, lbfgs_steps=None, heavy_regions=30):
    output = Path(root).joinpath('results', 'phase6', study_id)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new study id')
    fem_root = Path(root).joinpath('results', 'phase6', fem_study)
    output.mkdir(parents=True)
    rows = []
    for level, names in LADDER.items():
        for case in names:
            if cases is not None and case not in cases:
                continue
            problem = load_problem(source_of(case))
            reference, converged, fem_level = fem_reference(fem_root, case, problem)
            if not converged and not allow_unconverged:
                print(case, 'skipped: FEM mesh study did not converge', flush=True)
                rows.append(dict(level=level, case=case, architecture='-', seed=None, status='skipped', error='FEM mesh study did not converge'))
                continue
            for spec in architectures:
                architecture, attention = split_spec(spec)
                for seed in seeds:
                    cfg = dict(SETTINGS)
                    if len(problem.regions) > heavy_regions:
                        cfg.update(HEAVY)
                    if attention:
                        cfg['lbfgs_steps'] = 0
                    if adam_steps is not None:
                        cfg['adam_steps'] = adam_steps
                    if lbfgs_steps is not None and not attention:
                        cfg['lbfgs_steps'] = lbfgs_steps
                    result = trainer.run(source_of(case), architecture, output.joinpath(case, f'{spec}_s{seed}'), init_seed=seed, sampling_seed=11 + 1000 * seed,
                        study=study_id, log_every=200, attention=attention, reference=reference, reference_label=f'FEniCSx level {fem_level} ({fem_study}), evaluation only', **cfg)
                    e = result.get('evaluation', {})
                    rows.append(dict(level=level, case=case, architecture=spec, seed=seed, status=result['status'], parameters=result.get('parameters'),
                        final_loss=result.get('final_loss'), elapsed_s=result['elapsed_s'], nrmse_pct=e.get('nrmse_pct'), error_rise_pct=e.get('error_rise_pct'),
                        error_Tmax_abs_K=e.get('error_Tmax_abs_K'), max_error_K=e.get('max_error_K'),
                        worst_region_rmse_rise_pct=max(e['region_rmse_rise_pct'].values()) if e.get('region_rmse_rise_pct') else None,
                        balance_global_pct=e.get('balance', {}).get('global_derived_pct'),
                        interface_T_jump_max_K=max([i['T_jump_max_K'] for i in e.get('interfaces', [])], default=None),
                        interface_flux_jump_rms_pct=max([i['flux_jump_rms_pct'] for i in e.get('interfaces', [])], default=None),
                        acceptance=e.get('acceptance', {}).get('state'), gates=e.get('acceptance', {}).get('gates'), fem_converged=converged,
                        error=(result.get('error') or {}).get('message')))
                    print(case, spec, seed, rows[-1]['status'], rows[-1]['nrmse_pct'], rows[-1]['acceptance'], flush=True)
    summary = dict(study_id=study_id, fem_study=fem_study, seeds=list(seeds), architectures=list(architectures), rows=rows, criteria_status='proposed, not confirmed by the user')
    write_json_atomic(output.joinpath('summary.json'), summary)
    manifest = build_manifest('phase6_pinn_vs_fem', [sys.executable, '-m', 'pinn_v2.experiments.pinn_vs_fem', '--study-id', study_id, '--fem-study', fem_study],
        seeds=dict(init=list(seeds), sampling='11 + 1000*seed', audit=909091), extra=dict(fem_labels_used_for_training=False, fem_study=fem_study, rows=len(rows)))
    write_manifest(output, manifest)
    write_report(output, summary, 'phase6')
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-id', required=True)
    parser.add_argument('--fem-study', required=True)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--architectures', nargs='*', default=['M1'])
    parser.add_argument('--seeds', nargs='*', type=int, default=[0])
    parser.add_argument('--adam-steps', type=int)
    parser.add_argument('--lbfgs-steps', type=int)
    parser.add_argument('--allow-unconverged', action='store_true')
    a = parser.parse_args()
    out, _ = run_study(a.study_id, a.fem_study, a.cases, tuple(a.architectures), tuple(a.seeds), allow_unconverged=a.allow_unconverged, adam_steps=a.adam_steps, lbfgs_steps=a.lbfgs_steps)
    print(out)


if __name__ == '__main__':
    main()
