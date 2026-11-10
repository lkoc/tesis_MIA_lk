"""Adaptive all-case campaign: retained models, three seeds, staged Adam/L-BFGS."""
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.trainer import run
from pinn_v3.solvers.fem.adapter import run_problem_fem

MODELS = ('M1', 'M1_EXACT', 'CUSPR', 'M0R', 'M2')
SEEDS = (0, 1, 2)
ADAM_BLOCKS = (500, 500, 500)
LBFGS_BLOCKS = (50, 50, 50, 50)
RESULTS = ROOT / 'results' / 'adaptive_all_cases_20261027_final'
MANUAL_STOP_KEYS = {'annulus:CUSPR:seed2'}
WEIGHTS = dict(pde=1.0, energy=100.0)
# V2 stop thresholds and the requested two-times relaxation for stage decisions.
# Official thesis qualification thresholds.
STRICT = dict(error_Tmax_abs_K=0.05, nrmse_pct=0.25, balance_pct=0.5)
RELAXED = {name: 2 * value for name, value in STRICT.items()}


def load_reference(problem, case_id):
    """Return a reference array and label, or None when this case has no reference."""
    xy, _, _ = problem.evaluation()
    analytic = problem.reference(xy)
    if analytic is not None and np.isfinite(analytic).all():
        return np.asarray(analytic, dtype=float), 'analytic reference'
    fem_path = ROOT / 'results' / 'fem_generic' / case_id / 'fem_l1.npz'
    if fem_path.exists():
        fem = np.load(fem_path)
        values = np.asarray(LinearNDInterpolator(fem['dof_xy'], fem['dof_T'], fill_value=np.nan)(xy), dtype=float)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = NearestNDInterpolator(fem['dof_xy'], fem['dof_T'])(xy[missing])
        return values, 'FEM interpolated reference'
    return None, 'physics_only_no_reference'


def ensure_fem_reference(case_id, problem_path):
    """Create the V3 FEM level-1 reference inside V3 when it is absent."""
    output = ROOT / 'results' / 'fem_generic' / case_id
    record_path = output / 'fem_l1.json'
    field_path = output / 'fem_l1.npz'
    if record_path.exists() and field_path.exists():
        return dict(status='reused', elapsed_s=0.0, record=str(record_path))
    started = time.perf_counter()
    try:
        run_problem_fem(problem_path, output, levels=(1,))
    except RuntimeError as error:
        raise RuntimeError(f'FEM reference failed for {case_id}: {error}') from error
    return dict(status='generated', elapsed_s=time.perf_counter() - started, record=str(record_path))


def metrics(result):
    evaluation = result.get('evaluation') or {}
    values = dict(error_Tmax_abs_K=evaluation.get('error_Tmax_abs_K'), nrmse_pct=evaluation.get('nrmse_pct'),
                  balance_pct=(evaluation.get('balance') or {}).get('global_derived_pct'))
    return values


def passes_relaxed(values):
    return all(values[name] is not None and np.isfinite(values[name]) and values[name] < RELAXED[name] for name in STRICT)


def passes_strict(values):
    return all(values[name] is not None and np.isfinite(values[name]) and values[name] < STRICT[name] for name in STRICT)


def run_block(problem, architecture, root, seed, block_name, adam_steps, lbfgs_steps, state, reference, reference_label):
    output = root / block_name
    return run(problem, architecture, output, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
        width=32, depth=3, lr=0.002, weights=WEIGHTS, init_seed=seed, sampling_seed=11 + seed,
        sampling='mixed', focused=True, focus_criterion='residual', focus_surfaces=False,
        resample_every=100, reference=reference, reference_label=reference_label,
        full_audit=True, initial_state=state, extra_lbfgs_steps=0)


def state_from(output):
    return torch.load(output / 'latest.pt', map_location='cpu')


def recover_stage(run_root, name, steps):
    output = run_root / name
    result_path = output / 'result.json'
    if not result_path.exists() or not (output / 'latest.pt').exists():
        return None
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result.get('status') != 'completed':
        return None
    return dict(name=name, steps=steps, metrics=metrics(result),
                final_loss=result.get('final_loss'), status=result.get('status'),
                elapsed_s=result.get('elapsed_s'))


def quarantine_incomplete_stage(output):
    if not output.exists():
        return
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = output.with_name(f'{output.name}_incomplete_{stamp}')
    suffix = 1
    while backup.exists():
        backup = output.with_name(f'{output.name}_incomplete_{stamp}_{suffix}')
        suffix += 1
    shutil.move(str(output), str(backup))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    case_paths = sorted((ROOT / 'problems').glob('*/problem.json'))
    for case_path in case_paths:
        case_id = case_path.parent.name
        fem_info = None
        try:
            fem_info = ensure_fem_reference(case_id, case_path)
        except Exception:
            fem_info = dict(status='failed', error=traceback.format_exc())
        for architecture in MODELS:
            for seed in SEEDS:
                key = f'{case_id}:{architecture}:seed{seed}'
                previous = summary.get(key, {})
                if key in MANUAL_STOP_KEYS:
                    if previous.get('status') == 'running':
                        previous.update(status='stopped_manual', stop_reason='Stopped after lbfgs_3 by campaign operator',
                                        last_completed_stage='lbfgs_3')
                        summary[key] = previous
                        summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
                    continue
                valid_previous = previous.get('status') in ('completed', 'physics_only_completed') and \
                    previous.get('fem_reference', {}).get('status') != 'failed' and \
                    all(stage.get('status') == 'completed' for stage in previous.get('stages', []))
                if valid_previous:
                    continue
                run_root = RESULTS / case_id / architecture / f'seed_{seed}'
                recovered_stages = []
                for index, block in enumerate(ADAM_BLOCKS, 1):
                    stage = recover_stage(run_root, f'adam_{index}', block)
                    if stage is not None:
                        recovered_stages.append(stage)
                for index, block in enumerate(LBFGS_BLOCKS, 1):
                    stage = recover_stage(run_root, f'lbfgs_{index}', block)
                    if stage is not None:
                        recovered_stages.append(stage)
                record = dict(case=case_id, architecture=architecture, seed=seed, status='running',
                              stages=recovered_stages)
                summary[key] = record
                summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
                try:
                    if fem_info.get('status') == 'failed':
                        raise RuntimeError(f'FEM reference unavailable for {case_id}: {fem_info.get("error", "unknown error")}')
                    started = time.perf_counter()
                    problem = GenericProblem(load_problem(case_path))
                    reference, reference_label = load_reference(problem, case_id)
                    completed_names = {stage['name'] for stage in record['stages']}
                    state = state_from(run_root / record['stages'][-1]['name']) if record['stages'] else None
                    relaxed = reference is None or any(
                        passes_relaxed(stage['metrics']) for stage in record['stages']
                        if stage['name'].startswith('adam_'))
                    adam_total = sum(stage['steps'] for stage in record['stages'] if stage['name'].startswith('adam_'))
                    for index, block in enumerate(ADAM_BLOCKS, 1):
                        stage_name = f'adam_{index}'
                        if stage_name in completed_names:
                            continue
                        if reference is not None and relaxed:
                            break
                        quarantine_incomplete_stage(run_root / stage_name)
                        result = run_block(problem, architecture, run_root, seed, stage_name, block, 0,
                                           state, reference, reference_label)
                        state = state_from(run_root / stage_name)
                        values = metrics(result)
                        adam_total += block
                        record['stages'].append(dict(name=stage_name, steps=block, metrics=values,
                                                      final_loss=result.get('final_loss'), status=result.get('status'),
                                                      elapsed_s=result.get('elapsed_s')))
                        completed_names.add(stage_name)
                        summary[key] = record
                        summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
                        if reference is not None and passes_relaxed(values):
                            relaxed = True
                            break
                    for index, block in enumerate(LBFGS_BLOCKS, 1):
                        stage_name = f'lbfgs_{index}'
                        if stage_name in completed_names:
                            continue
                        quarantine_incomplete_stage(run_root / stage_name)
                        result = run_block(problem, architecture, run_root, seed, stage_name, 0, block,
                                           state, reference, reference_label)
                        state = state_from(run_root / stage_name)
                        values = metrics(result)
                        record['stages'].append(dict(name=stage_name, steps=block, metrics=values,
                                                      final_loss=result.get('final_loss'), status=result.get('status'),
                                                      elapsed_s=result.get('elapsed_s')))
                        completed_names.add(stage_name)
                        summary[key] = record
                        summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
                        if reference is not None and passes_strict(values):
                            break
                    final = record['stages'][-1]
                    stages_ok = bool(record['stages']) and all(stage.get('status') == 'completed' for stage in record['stages'])
                    record.update(status=('completed' if reference is not None else 'physics_only_completed')
                                  if stages_ok else 'failed',
                                  reference_label=reference_label, strict_converged=passes_strict(final['metrics']) if reference is not None else None,
                                  relaxed_reached=relaxed, total_adam_steps=adam_total,
                                  elapsed_s=time.perf_counter() - started,
                                  elapsed_min=(time.perf_counter() - started) / 60,
                                  fem_reference=fem_info)
                except Exception:
                    record.update(status='failed', error=traceback.format_exc(),
                                  elapsed_s=time.perf_counter() - started,
                                  elapsed_min=(time.perf_counter() - started) / 60,
                                  fem_reference=fem_info)
                summary[key] = record
                summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=float), flush=True)


if __name__ == '__main__':
    main()
