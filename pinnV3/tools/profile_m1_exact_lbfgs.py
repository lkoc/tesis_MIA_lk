"""Profile one native M1_EXACT L-BFGS call from a saved smoke checkpoint."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_RUN = (ROOT / 'results' / 'smoke_float32_adam100_lbfgs10_x2pts_20261028'
               / 'aras_flat' / 'M1_EXACT' / 'seed_0')

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.objective import Objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--device', choices=('xpu', 'cpu'), default='xpu')
    parser.add_argument('--trace', action='store_true',
        help='Collect the detailed operator trace; adds significant profiler overhead.')
    parser.add_argument('--report', type=Path,
        default=ROOT / 'results' / 'smoke_profile_m1_exact_lbfgs_20261028.txt')
    args = parser.parse_args()
    run_dir = args.run.resolve()
    metadata = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
    config = metadata['config']
    if config['architecture'] != 'M1_EXACT':
        raise ValueError(f"Expected M1_EXACT checkpoint, got {config['architecture']!r}")
    if args.device == 'xpu' and not torch.xpu.is_available():
        raise RuntimeError('XPU was requested but PyTorch reports no XPU device')

    problem_path = ROOT / 'problems' / metadata['case'] / 'problem.json'
    problem = GenericProblem(load_problem(problem_path))
    model = build_model(problem, config['architecture'], config['width'], config['depth'],
        config['activation'], input_normalization=config.get('input_normalization'),
        trace_parameterization=config.get('trace_parameterization', 'standard'),
        soil_envelope_gain=config.get('soil_envelope_gain', 1.))
    torch_dtype = torch.float32 if config['dtype'] == 'float32' else torch.float64
    model = model.to(device=args.device, dtype=torch_dtype)
    state = torch.load(run_dir / 'latest.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state)

    points = dict(config['points'])
    objective = Objective(model, seed=metadata['seeds']['sampling'], sampling=config['sampling'],
        weights=config['weights'], focused='resampling_policy' in config,
        focus_criterion=config.get('focus_criterion', 'residual'),
        focus_surfaces=config.get('focus_surfaces', False), **points)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1,
        history_size=50, line_search_fn='strong_wolfe')
    closures = 0

    def closure():
        nonlocal closures
        closures += 1
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective()
        loss.backward()
        return loss

    profile = None
    if args.trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == 'xpu' and hasattr(torch.profiler.ProfilerActivity, 'XPU'):
            activities.append(torch.profiler.ProfilerActivity.XPU)
        profile = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True)
    if args.device == 'xpu':
        torch.xpu.synchronize()
    started = time.perf_counter()
    if profile is not None:
        profile.__enter__()
    try:
        optimizer.step(closure)
    finally:
        if profile is not None:
            profile.__exit__(None, None, None)
    if args.device == 'xpu':
        torch.xpu.synchronize()
    elapsed = time.perf_counter() - started

    header = json.dumps(dict(case=metadata['case'], architecture=config['architecture'],
        device=str(next(model.parameters()).device), dtype=config['dtype'],
        parameters=model.parameter_count(), lbfgs_outer_calls=1, max_iter=1,
        closure_calls=closures, elapsed_s=elapsed,
        seconds_per_closure=elapsed / max(closures, 1)), indent=2)
    report = header
    if profile is not None:
        table = profile.key_averages().table(sort_by='self_cpu_time_total', row_limit=25)
        report = f'{header}\n\n{table}\n'
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding='utf-8')
    print(report, flush=True)


if __name__ == '__main__':
    main()