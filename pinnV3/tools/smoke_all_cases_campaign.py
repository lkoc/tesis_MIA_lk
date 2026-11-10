"""Short all-case smoke campaign with local float32 PINNs and cached FEM references."""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.manifest.run_manifest import replace_json_atomic
from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.trainer import run
from pinn_v3.solvers.fem.adapter import run_problem_fem

MODELS = ('M1', 'M1_EXACT', 'CUSPR', 'M0R', 'M2')
SEEDS = (0, 1, 2)
ADAM_STEPS = 100
LBFGS_STEPS = 10
LBFGS_ITERATIONS = 1
LEARNING_RATE = 0.003
DTYPE = 'float32'
POINTS = dict(n_interior=1024, n_layer=512, n_face=256, n_boundary=256)
WEIGHTS = dict(pde=1.0, energy=100.0)
RESULTS = ROOT / 'results' / 'smoke_float32_adam100_lbfgs10_x2pts_20261028'


def load_reference(problem, case_id):
    xy, _, _ = problem.evaluation()
    exact = problem.reference(xy)
    if exact is not None and np.isfinite(exact).all():
        return np.asarray(exact, dtype=float), 'analytic reference'
    field_path = ROOT / 'results' / 'fem_generic' / case_id / 'fem_l1.npz'
    if not field_path.exists():
        return None, 'physics_only_no_reference'
    fem = np.load(field_path)
    values = np.asarray(LinearNDInterpolator(
        fem['dof_xy'], fem['dof_T'], fill_value=np.nan)(xy), dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = NearestNDInterpolator(fem['dof_xy'], fem['dof_T'])(xy[missing])
    return values, 'FEM interpolated reference'


def ensure_fem_reference(case_id, canonical, problem_path):
    output = ROOT / 'results' / 'fem_generic' / case_id
    record_path = output / 'fem_l1.json'
    field_path = output / 'fem_l1.npz'
    if record_path.exists() and field_path.exists():
        return dict(status='reused', elapsed_s=0.0, record=str(record_path))
    started = time.perf_counter()
    try:
        run_problem_fem(problem_path, output, levels=(1,))
    except RuntimeError as error:
        elapsed = time.perf_counter() - started
        if canonical.doc.get('reference', {}).get('exact') is None:
            raise RuntimeError(f'FEM reference failed for {case_id}: {error}') from error
        return dict(status='failed_but_analytic_available', elapsed_s=elapsed, error=str(error), record=None)
    return dict(status='generated', elapsed_s=time.perf_counter() - started, record=str(record_path))


def metrics(result):
    evaluation = result.get('evaluation') or {}
    return dict(error_Tmax_abs_K=evaluation.get('error_Tmax_abs_K'),
                nrmse_pct=evaluation.get('nrmse_pct'),
                balance_pct=(evaluation.get('balance') or {}).get('global_derived_pct'))


def save_summary(path, summary):
    replace_json_atomic(path, summary)


def next_resume_directory(previous_run_dir):
    index = 1
    while True:
        candidate = previous_run_dir.with_name(f'{previous_run_dir.name}_resume_{index}')
        if not candidate.exists():
            return candidate
        index += 1


def continuation_target(original_run_dir, previous_record):
    previous_run_dir = Path(previous_record.get('run_dir', original_run_dir))
    checkpoint = previous_run_dir / 'model.pt'
    if checkpoint.exists():
        return next_resume_directory(previous_run_dir), checkpoint
    if original_run_dir.exists():
        return next_resume_directory(original_run_dir), None
    return original_run_dir, None


def main():
    summary_path = RESULTS / 'summary.json'
    if RESULTS.exists():
        if not summary_path.exists():
            raise FileNotFoundError(f'Cannot resume without campaign summary: {summary_path}')
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
    else:
        RESULTS.mkdir(parents=True)
        summary = {}
    save_summary(summary_path, summary)

    case_paths = sorted((ROOT / 'problems').glob('*/problem.json'))
    for case_path in case_paths:
        case_id = case_path.parent.name
        case_keys = [f'{case_id}:{architecture}:seed{seed}'
            for architecture in MODELS for seed in SEEDS]
        if all(summary.get(key, {}).get('status') == 'completed' for key in case_keys):
            continue
        try:
            canonical = load_problem(case_path)
            problem = GenericProblem(canonical)
            fem_info = ensure_fem_reference(case_id, canonical, case_path)
            reference, reference_label = load_reference(problem, case_id)
        except Exception:
            error = traceback.format_exc()
            for architecture in MODELS:
                for seed in SEEDS:
                    key = f'{case_id}:{architecture}:seed{seed}'
                    if summary.get(key, {}).get('status') == 'completed':
                        continue
                    summary[key] = dict(case=case_id, architecture=architecture, seed=seed,
                                        status='failed_preflight', error=error)
            save_summary(summary_path, summary)
            continue

        for architecture in MODELS:
            for seed in SEEDS:
                key = f'{case_id}:{architecture}:seed{seed}'
                previous_record = summary.get(key, {})
                if previous_record.get('status') == 'completed':
                    continue
                original_run_dir = RESULTS / case_id / architecture / f'seed_{seed}'
                run_root, resume_from = continuation_target(original_run_dir, previous_record)
                record = dict(case=case_id, architecture=architecture, seed=seed,
                              status='running', fem_reference=fem_info, run_dir=str(run_root))
                if resume_from is not None:
                    record['resume_from'] = str(resume_from)
                summary[key] = record
                save_summary(summary_path, summary)
                started = time.perf_counter()
                try:
                    result = run(
                        problem, architecture, run_root,
                        adam_steps=ADAM_STEPS, lbfgs_steps=LBFGS_STEPS,
                        lbfgs_iterations=LBFGS_ITERATIONS, lr=LEARNING_RATE,
                        width=32, depth=3, weights=WEIGHTS, init_seed=seed,
                        sampling_seed=11 + seed, sampling='mixed', focused=True,
                        focus_criterion='residual', focus_surfaces=False,
                        resample_every=100, reference=reference,
                        reference_label=reference_label, full_audit=True,
                        dtype=DTYPE, points=POINTS, extra_lbfgs_steps=0,
                        resume_from=resume_from)
                    record.update(status=result.get('status', 'failed'),
                                  metrics=metrics(result), result_status=result.get('status'),
                                  elapsed_s=result.get('elapsed_s'), device=result.get('device'),
                                  reference_label=reference_label,
                                  smoke_only=True, official_qualification=False)
                except Exception:
                    record.update(status='failed', error=traceback.format_exc(),
                                  elapsed_s=time.perf_counter() - started,
                                  reference_label=reference_label, smoke_only=True,
                                  official_qualification=False)
                summary[key] = record
                save_summary(summary_path, summary)
                print(json.dumps({key: record}, default=float), flush=True)

    print(json.dumps(dict(output=str(RESULTS), entries=len(summary)), flush=True))


if __name__ == '__main__':
    main()