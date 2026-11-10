"""Medium smoke: FEM readout copy (lr=0) versus no initializer, 400 Adam and 25 L-BFGS steps, with accuracy-based early stop."""
import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.trainer import run
from tools.smoke_all_cases_campaign import ensure_fem_reference, load_reference

FEM_INIT = dict(source='fem', fem_node_multiplier=1.0, mode='extrapolate', pretrain_steps=0,
                fem_transfer='readout_gls', fem_gradient_weight=1.0, fem_ridge=0.0)


def cases_from(cases, start_case=None):
    cases = tuple(cases)
    if start_case is None:
        return cases
    try:
        return cases[cases.index(start_case):]
    except ValueError as exc:
        raise ValueError(f'Start case {start_case!r} is not in the campaign case list') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=('mms_sine', 'cable_disk'))
    parser.add_argument('--start-case', help='Resume at this case while preserving the full campaign scope')
    parser.add_argument('--seeds', type=int, nargs='+', default=(0, 1, 2))
    parser.add_argument('--architectures', nargs='+', default=('M1',))
    parser.add_argument('--inits', nargs='+', choices=('none', 'fem'), default=('fem',))
    parser.add_argument('--adam-steps', type=int, default=400)
    parser.add_argument('--lbfgs-steps', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--n-interior', type=int, default=200)
    parser.add_argument('--n-layer', type=int, default=100)
    parser.add_argument('--n-face', type=int, default=24)
    parser.add_argument('--n-boundary', type=int, default=24)
    parser.add_argument('--cycle-lbfgs-chunk', type=int, default=0)
    parser.add_argument('--extend-steps', type=int, default=0)
    parser.add_argument('--extend-times', type=int, default=0)
    parser.add_argument('--extend-factor', type=float, default=1.5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--skip-architectures', nargs='*', default=(),
                        help='Skip pending runs for selected architectures when resuming a campaign')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_cases = cases_from(args.cases, args.start_case)
    unknown_skips = set(args.skip_architectures) - set(args.architectures)
    if unknown_skips:
        parser.error(f'skipped architectures must be included in --architectures: {sorted(unknown_skips)}')
    output = args.output.resolve()
    adam_steps, lbfgs_steps = args.adam_steps, args.lbfgs_steps
    planned = adam_steps * (1 + args.extend_times) + lbfgs_steps
    points = dict(n_interior=args.n_interior, n_layer=args.n_layer,
                  n_face=args.n_face, n_boundary=args.n_boundary)
    campaign_settings = dict(adam_steps=adam_steps, lbfgs_steps=lbfgs_steps, lbfgs_iterations=1,
        extra_lbfgs_steps=0, width=12, depth=2, lr=args.lr, cycle_lbfgs_chunk=args.cycle_lbfgs_chunk,
        points=points, inits=list(args.inits), architectures=list(args.architectures),
        extend=dict(steps=args.extend_steps, times=args.extend_times, factor=args.extend_factor))
    summary_path = output / 'summary.json'
    if args.resume:
        if not summary_path.exists():
            raise FileNotFoundError(f'Cannot resume without campaign summary: {summary_path}')
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        mismatches = [key for key, value in campaign_settings.items() if summary.get(key) != value]
        if mismatches:
            raise ValueError(f'Resume settings differ from existing campaign: {mismatches}')
        summary['cases'] = list(args.cases)
        summary['seeds'] = list(args.seeds)
        for label, record in summary.get('runs', {}).items():
            if record.get('case') not in args.cases or record.get('architecture') not in args.architectures or \
                    record.get('seed') not in args.seeds or record.get('init') not in args.inits:
                raise ValueError(f'Existing run {label} is outside the requested resume scope')
    else:
        output.mkdir(parents=True, exist_ok=False)
        summary = dict(campaign_settings, smoke_only=True, official_qualification=False, runs={},
                       planned_adam_max_steps=adam_steps * (1 + args.extend_times),
                       fem_references={}, case_errors={}, cases=list(args.cases), seeds=list(args.seeds))
    summary['planned_runs'] = len(args.cases) * len(args.seeds) * len(args.architectures) * len(args.inits)
    summary['excluded_architectures'] = sorted(set(summary.get('excluded_architectures', ())) |
        set(args.skip_architectures))
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    for case in run_cases:
        problem_path = ROOT / 'problems' / case / 'problem.json'
        try:
            canonical = load_problem(problem_path)
            problem = GenericProblem(canonical)
            reference, reference_label = load_reference(problem, case)
            if reference is None:
                summary['fem_references'][case] = ensure_fem_reference(case, canonical, problem_path)
                reference, reference_label = load_reference(problem, case)
                if reference is None:
                    raise RuntimeError(f'No analytic or FEM score reference available for {case}')
        except Exception as exc:
            summary['case_errors'][case] = f'{type(exc).__name__}: {exc}'
            for seed in args.seeds:
                for arch, init in itertools.product(args.architectures, args.inits):
                    label = f'{case}_{arch}_s{seed}_{init}'
                    summary['runs'][label] = dict(status='failed_preflight', error=summary['case_errors'][case],
                        case=case, architecture=arch, seed=seed, init=init)
            (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps({case: summary['case_errors'][case]}, flush=True))
            continue
        rise = float(np.max(reference) - problem.T_level)
        stop = dict(error_Tmax_abs_K=0.05 * rise, nrmse_pct=5.0, balance_pct=2.0, every=25)
        for seed in args.seeds:
            for arch, init in itertools.product(args.architectures, args.inits):
                label = f'{case}_{arch}_s{seed}_{init}'
                run_output = output / label
                existing = summary['runs'].get(label, {})
                if arch in summary['excluded_architectures']:
                    if existing.get('status') not in ('completed', 'failed', 'failed_preflight'):
                        record = dict(status='skipped_duration',
                            reason='excluded from remaining campaign due to excessive runtime',
                            case=case, architecture=arch, seed=seed, init=init)
                        summary['runs'][label] = record
                        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
                        print(json.dumps({label: record}, allow_nan=False), flush=True)
                    continue
                if existing.get('status') == 'completed' and (run_output / 'result.json').exists():
                    continue
                if existing.get('status') in ('failed', 'failed_preflight') and not args.retry_failed:
                    continue
                if run_output.exists():
                    archive = output / '_interrupted_runs'
                    archive.mkdir(exist_ok=True)
                    archived_run = archive / f'{label}_{time.time_ns()}'
                    shutil.move(str(run_output), str(archived_run))
                started = time.perf_counter()
                try:
                    result = run(problem, arch, run_output, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
                        lbfgs_iterations=1, extra_lbfgs_steps=0, lr=args.lr, width=12, depth=2,
                        init_seed=seed, sampling_seed=11 + seed,
                        points=points,
                        weights=dict(pde=1., energy=0.), device='cpu', dtype='float32',
                        reference=reference, reference_label=reference_label, full_audit=True,
                        accuracy_stop=stop, rbf_initialization=FEM_INIT if init == 'fem' else None, log_every=10,
                        cycle_lbfgs_chunk=args.cycle_lbfgs_chunk,
                        cycle_extension=dict(steps=args.extend_steps, times=args.extend_times, factor=args.extend_factor) if args.extend_times else None)
                    evaluation = result.get('evaluation') or {}
                    history = result.get('history', [])
                    last_step = int(history[-1]['step']) if history else -1
                    record = dict(status=result.get('status'), error=result.get('error'),
                        wall_s=time.perf_counter() - started, elapsed_s=result.get('elapsed_s'),
                        last_step=last_step, rows_planned=planned,
                        adam_rows=sum(row['phase'] == 'adam' for row in history),
                        lbfgs_rows=sum(row['phase'] == 'lbfgs' for row in history),
                        relaxations=len((result.get('accuracy_stop') or {}).get('relaxations', [])),
                        early_stop=bool(history) and last_step + 1 < planned,
                        nrmse_pct=evaluation.get('nrmse_pct'), error_Tmax_abs_K=evaluation.get('error_Tmax_abs_K'),
                        error_Tmax_site_K=evaluation.get('error_Tmax_site_K'), error_rise_site_pct=evaluation.get('error_rise_site_pct'),
                        balance_pct=(evaluation.get('balance') or {}).get('global_derived_pct'),
                        accepted=(evaluation.get('acceptance') or {}).get('accepted'),
                        termination_reason=(result.get('accuracy_stop') or {}).get('termination_reason'),
                        lbfgs_failure=(result.get('accuracy_stop') or {}).get('lbfgs_failure'),
                        initialization=result.get('rbf_initialization'), stop_threshold_K=stop['error_Tmax_abs_K'])
                except Exception as exc:
                    record = dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                                  wall_s=time.perf_counter() - started)
                record.update(case=case, architecture=arch, seed=seed, init=init)
                if args.cycle_lbfgs_chunk and 'accepted' in record:
                    record['early_stop'] = bool(record['accepted'])
                summary['runs'][label] = record
                summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
                print(json.dumps({label: record}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
