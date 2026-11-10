"""Compare fixed collocation with progressive points under the smoke recipe."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.trainer import run
from tools.smoke_all_cases_campaign import DTYPE, POINTS, WEIGHTS, load_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='cable_disk')
    parser.add_argument('--architecture', default='M1_EXACT')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--device', choices=('xpu', 'cpu'), default='xpu')
    parser.add_argument('--order', choices=('fixed_first', 'progressive_first'), default='fixed_first')
    parser.add_argument('--point-start', type=float, default=0.5)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'results' / 'point_schedule_ab_20261028')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Benchmark output already exists: {output}')
    output.mkdir(parents=True)

    case_path = ROOT / 'problems' / args.case / 'problem.json'
    canonical = load_problem(case_path)
    problem = GenericProblem(canonical)
    reference, reference_label = load_reference(problem, args.case)
    records = {}
    order = (('fixed', 0), ('progressive', 10))
    if args.order == 'progressive_first':
        order = tuple(reversed(order))
    for label, stages in order:
        started = time.perf_counter()
        result = run(
            problem, args.architecture, output / label,
            adam_steps=100, lbfgs_steps=10, lbfgs_iterations=1, lr=0.003,
            width=32, depth=3, weights=WEIGHTS, init_seed=args.seed,
            sampling_seed=11 + args.seed, sampling='mixed', focused=True,
            focus_criterion='residual', focus_surfaces=False, resample_every=100,
            reference=reference, reference_label=reference_label, full_audit=True,
            dtype=DTYPE, points=POINTS, extra_lbfgs_steps=0, device=args.device,
            point_schedule_stages=stages, point_schedule_start=args.point_start)
        evaluation = result.get('evaluation') or {}
        records[label] = dict(
            status=result.get('status'),
            elapsed_s=result.get('elapsed_s'),
            training_elapsed_s=result.get('training_elapsed_s'),
            wall_s=time.perf_counter() - started,
            device=result.get('device'),
            steps=dict(adam=100, lbfgs=10),
            point_schedule_stages=stages,
            point_schedule_start=args.point_start if stages else None,
            metrics=dict(
                error_Tmax_abs_K=evaluation.get('error_Tmax_abs_K'),
                nrmse_pct=evaluation.get('nrmse_pct'),
                balance_pct=(evaluation.get('balance') or {}).get('global_derived_pct')))

    fixed_s = records['fixed']['elapsed_s']
    progressive_s = records['progressive']['elapsed_s']
    records['comparison'] = dict(
        speedup=fixed_s / progressive_s if fixed_s and progressive_s else None,
        saved_pct=100 * (fixed_s - progressive_s) / fixed_s if fixed_s else None,
        adam_speedup=(records['fixed'].get('training_elapsed_s', {}).get('adam', 0.) /
            records['progressive'].get('training_elapsed_s', {}).get('adam', 0.)
            if records['progressive'].get('training_elapsed_s', {}).get('adam', 0.) else None))
    report = dict(case=args.case, architecture=args.architecture, seed=args.seed,
        device=args.device, dtype=DTYPE, max_points=POINTS,
        point_schedule_start=args.point_start, records=records)
    report_path = output / 'comparison.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()