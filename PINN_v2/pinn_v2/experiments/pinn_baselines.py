"""Phase 4 study: M0, M1 and M2 on V0-V2 (manufactured and concentric cases with exact references).

Writes only to a new directory results/phase4/<study_id>. No FEM field and no exact value enters a loss;
the exact solution is used after training to score. Every run, failures included, is kept.
"""
import argparse
import sys
from pathlib import Path

from pinn_v2.manifest.run_manifest import build_manifest, write_manifest, write_json_atomic
from pinn_v2.solvers.pinn import trainer

ROOT = Path(__file__).resolve().parents[2]
LADDER = dict(V0=('mms_constant', 'mms_robin'), V1=('mms_interface',), V2=('coaxial_full', 'coaxial_robin', 'coaxial_bc_dirichlet'))
ARCHITECTURES = ('M0', 'M1', 'M2')
DEFAULT = dict(adam_steps=4000, lbfgs_steps=60, lr=3e-3, width=32, depth=3, weights={}, points={})
# Chosen from short tuning runs (2000-6000 Adam steps on coaxial_full/coaxial_robin); not tuned against any acceptance gate per case.
OVERRIDES = {('coaxial_full', 'M2'): dict(adam_steps=6000, lbfgs_steps=100), ('coaxial_robin', 'M2'): dict(adam_steps=6000, lbfgs_steps=100),
    ('coaxial_bc_dirichlet', 'M2'): dict(adam_steps=6000, lbfgs_steps=100)}


def split_spec(spec):
    """'M1+rba' -> ('M1', 'rba'); a plain name has no attention."""
    name, _, attention = spec.partition('+')
    return name, (attention or None)


def settings(case, architecture):
    return {**DEFAULT, **OVERRIDES.get(case, {}), **OVERRIDES.get((case, architecture), {})}


def run_study(study_id, cases=None, architectures=ARCHITECTURES, seeds=(0,), root=ROOT, adam_steps=None, lbfgs_steps=None, phase='phase4'):
    output = Path(root).joinpath('results', phase, study_id)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new study id')
    output.mkdir(parents=True)
    selected = [(level, case) for level, names in LADDER.items() for case in names if cases is None or case in cases]
    rows = []
    for level, case in selected:
        for spec in architectures:
            architecture, attention = split_spec(spec)
            for seed in seeds:
                cfg = settings(case, architecture)
                if attention:
                    cfg['lbfgs_steps'] = 0
                if adam_steps is not None:
                    cfg['adam_steps'] = adam_steps
                if lbfgs_steps is not None and not attention:
                    cfg['lbfgs_steps'] = lbfgs_steps
                result = trainer.run(case, architecture, output.joinpath(case, f'{spec}_s{seed}'), init_seed=seed,
                    sampling_seed=11 + 1000 * seed, study=study_id, log_every=200, attention=attention, **cfg)
                e = result.get('evaluation', {})
                rows.append(dict(level=level, case=case, architecture=spec, seed=seed, status=result['status'], parameters=result.get('parameters'),
                    final_loss=result.get('final_loss'), elapsed_s=result['elapsed_s'], nrmse_pct=e.get('nrmse_pct'), error_rise_pct=e.get('error_rise_pct'),
                    error_Tmax_abs_K=e.get('error_Tmax_abs_K'), max_error_K=e.get('max_error_K'),
                    worst_region_rmse_rise_pct=max(e['region_rmse_rise_pct'].values()) if e.get('region_rmse_rise_pct') else None,
                    balance_global_pct=e.get('balance', {}).get('global_derived_pct'),
                    interface_T_jump_max_K=max([i['T_jump_max_K'] for i in e.get('interfaces', [])], default=None),
                    interface_flux_jump_rms_pct=max([i['flux_jump_rms_pct'] for i in e.get('interfaces', [])], default=None),
                    acceptance=e.get('acceptance', {}).get('state'), gates=e.get('acceptance', {}).get('gates'),
                    error=(result.get('error') or {}).get('message')))
                print(rows[-1]['case'], spec, seed, rows[-1]['status'], rows[-1]['nrmse_pct'], rows[-1]['acceptance'], flush=True)
    summary = dict(study_id=study_id, seeds=list(seeds), architectures=list(architectures), rows=rows, criteria_status='proposed, not confirmed by the user')
    write_json_atomic(output.joinpath('summary.json'), summary)
    manifest = build_manifest(f'{phase}_pinn_study', [sys.executable, '-m', 'pinn_v2.experiments.pinn_baselines', '--study-id', study_id, '--phase', phase],
        seeds=dict(init=list(seeds), sampling='11 + 1000*seed', audit=909091), extra=dict(fem_labels_used_for_training=False, rows=len(rows)))
    write_manifest(output, manifest)
    write_report(output, summary, phase)
    return output, summary


def _fmt(value, spec='.3g'):
    return '' if value is None else format(value, spec)


def write_report(output, summary, phase='phase4'):
    lines = [f"# {phase} PINN study, study {summary['study_id']}", '', f"Acceptance criteria: {summary['criteria_status']}.", '',
        '| level | case | arch | seed | status | NRMSE % | dT error % | Tmax err K | worst region % | balance % | [T] max K | [q.n] rms % | verdict |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in summary['rows']:
        lines.append(f"| {r['level']} | {r['case']} | {r['architecture']} | {r['seed']} | {r['status']} | {_fmt(r['nrmse_pct'])} | {_fmt(r['error_rise_pct'])} | "
            f"{_fmt(r['error_Tmax_abs_K'])} | {_fmt(r['worst_region_rmse_rise_pct'])} | {_fmt(r['balance_global_pct'])} | {_fmt(r['interface_T_jump_max_K'])} | "
            f"{_fmt(r['interface_flux_jump_rms_pct'])} | {r['acceptance'] or r['error']} |")
    path = Path(output).joinpath('report.md')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-id', required=True)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--architectures', nargs='*', default=list(ARCHITECTURES))
    parser.add_argument('--seeds', nargs='*', type=int, default=[0])
    parser.add_argument('--adam-steps', type=int)
    parser.add_argument('--lbfgs-steps', type=int)
    parser.add_argument('--phase', default='phase4')
    a = parser.parse_args()
    out, _ = run_study(a.study_id, a.cases, tuple(a.architectures), tuple(a.seeds), adam_steps=a.adam_steps, lbfgs_steps=a.lbfgs_steps, phase=a.phase)
    print(out)


if __name__ == '__main__':
    main()
