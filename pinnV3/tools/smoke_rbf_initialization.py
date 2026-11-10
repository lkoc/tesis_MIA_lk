"""Compare optional RBF-FD PINN initialization strategies on one cable case."""
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
from tools.smoke_all_cases_campaign import load_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='cable_disk')
    parser.add_argument('--architecture', default='M1')
    parser.add_argument('--device', choices=('cpu', 'xpu'), default='cpu')
    parser.add_argument('--fractions', type=float, nargs='+', default=(0.01, 0.05, 0.10))
    parser.add_argument('--adam-steps', type=int, default=20)
    parser.add_argument('--lbfgs-steps', type=int, default=0)
    parser.add_argument('--fem-multipliers', type=float, nargs='+', default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument('--fem-pretrain-steps', type=int, nargs='+', default=(20,))
    parser.add_argument('--fem-transfer', nargs='+', choices=('pretrain', 'readout', 'readout_gls'), default=('pretrain',))
    parser.add_argument('--n-interior', type=int, nargs='+', default=(200,))
    parser.add_argument('--gradient-weights', type=float, nargs='+', default=(1.0,))
    parser.add_argument('--widths', type=int, nargs='+', default=(12,))
    parser.add_argument('--ridges', type=float, nargs='+', default=(0.0,))
    parser.add_argument('--rbf-transfers', nargs='*', choices=('readout', 'readout_gls'), default=())
    parser.add_argument('--rbf-fractions', type=float, nargs='+', default=(1.0,))
    parser.add_argument('--skip-rbf', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    problem = GenericProblem(load_problem(ROOT / 'problems' / args.case / 'problem.json'))
    reference, reference_label = load_reference(problem, args.case)
    experiments = []
    for width in args.widths:
        for n_interior in args.n_interior:
            suffix = f'_w{width}_n{n_interior}'
            experiments.append((f'baseline{suffix}', None, n_interior, width))
            for transfer in args.rbf_transfers:
                for fraction in args.rbf_fractions:
                    experiments.append((f'rbf_{transfer}_f{int(round(fraction * 100)):03d}{suffix}',
                        dict(source='rbf_readout', rbf_transfer=transfer, rbf_anchor_fraction=fraction, mode='extrapolate'), n_interior, width))
            if not args.skip_rbf:
                experiments.extend((f'{mode}_{int(fraction * 100):02d}{suffix}',
                    dict(mode=mode, fraction=fraction, pretrain_steps=20, surface_count=24, growth_stages=10), n_interior, width)
                    for mode in ('extrapolate', 'incremental') for fraction in args.fractions)
            for multiplier in args.fem_multipliers:
                for transfer in args.fem_transfer:
                    for steps in (args.fem_pretrain_steps if transfer == 'pretrain' else (0,)):
                        for weight in (args.gradient_weights if transfer == 'readout_gls' else (1.0,)):
                            for ridge in (args.ridges if transfer in ('readout', 'readout_gls') else (0.0,)):
                                label = f'fem_m{multiplier:g}_{transfer}_pt{steps}_gw{weight:g}_r{ridge:g}{suffix}'
                                experiments.append((label.replace('.', 'p'),
                                    dict(source='fem', fem_node_multiplier=multiplier, mode='extrapolate', pretrain_steps=steps,
                                         fem_transfer=transfer, fem_gradient_weight=weight, fem_ridge=ridge), n_interior, width))
    summary = dict(case=args.case, architecture=args.architecture, device=args.device,
        seed=0, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps, lbfgs_iterations=1,
        smoke_only=True, official_qualification=False, experiments={})
    for label, settings, n_interior, width in experiments:
        started = time.perf_counter()
        try:
            result = run(problem, args.architecture, output / label, adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps, lbfgs_iterations=1,
                width=width, depth=2, lr=0.002, init_seed=0, sampling_seed=11,
                points=dict(n_interior=n_interior, n_layer=100, n_face=24, n_boundary=24),
                weights=dict(pde=1., energy=100.), device=args.device, dtype='float32',
                reference=reference, reference_label=reference_label, full_audit=True,
                rbf_initialization=settings, log_every=10)
            evaluation = result.get('evaluation') or {}
            record = dict(status=result.get('status'), error=result.get('error'),
                wall_s=time.perf_counter() - started, elapsed_s=result.get('elapsed_s'),
                training_elapsed_s=result.get('training_elapsed_s'),
                lbfgs_calls_completed=sum(row.get('phase') == 'lbfgs' for row in result.get('history', [])),
                initialization=result.get('rbf_initialization'),
                error_Tmax_abs_K=evaluation.get('error_Tmax_abs_K'),
                nrmse_pct=evaluation.get('nrmse_pct'),
                balance_pct=(evaluation.get('balance') or {}).get('global_derived_pct'),
                accepted=(evaluation.get('acceptance') or {}).get('accepted'))
        except Exception as exc:
            record = dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                          wall_s=time.perf_counter() - started)
        record['n_interior'] = n_interior
        record['width'] = width
        summary['experiments'][label] = record
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({label: record}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()