"""Profile one L-BFGS step (strong Wolfe) for M1_EXACT on a generic case: closure count and time per call."""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.objective import DEFAULT_WEIGHTS, Objective

MAX_CALLS = 60
POINTS = dict(n_interior=200, n_layer=100, n_face=24, n_boundary=24)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'aras_single'
    output = ROOT / 'results' / f'profile_m1exact_lbfgs_{case}_20261031.txt'
    lines = []
    def log(text):
        lines.append(text)
        print(text, flush=True)

    problem = GenericProblem(load_problem(ROOT / 'problems' / case / 'problem.json'))
    torch.manual_seed(0)
    model = build_model(problem, 'M1_EXACT', 12, 2, 'tanh', input_normalization='unit_box_v1').to(dtype=torch.float32)
    requested = dict(DEFAULT_WEIGHTS, pde=1., energy=100.)
    kept = ('pde', 'constitutive', 'interface_temperature', 'interface_flux')
    weights = {name: requested[name] if name in kept else 0. for name in DEFAULT_WEIGHTS}
    objective = Objective(model, seed=11, sampling='mixed', weights=weights, attention=None,
                          exact_energy=False, focused=False, focus_criterion='residual',
                          focus_surfaces=True, initial_fraction=None, **POINTS)
    log(f'case={case} regions={len(problem.regions)} params={sum(p.numel() for p in model.parameters())}')

    started = time.perf_counter()
    total, parts, _ = objective()
    total.backward()
    log(f'one forward+backward: {time.perf_counter() - started:.3f} s, loss={float(total):.4e}')
    log('parts: ' + ', '.join(f'{k}={float(v):.3e}' for k, v in parts.items()))

    profiler = cProfile.Profile()
    profiler.enable()
    model.zero_grad()
    total, _, _ = objective()
    total.backward()
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(18)
    log(stream.getvalue())

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1, history_size=50, line_search_fn='strong_wolfe')
    calls = []

    def closure():
        if len(calls) >= MAX_CALLS:
            raise RuntimeError(f'closure budget of {MAX_CALLS} calls exhausted')
        began = time.perf_counter()
        lbfgs.zero_grad()
        value, _, _ = objective()
        value.backward()
        calls.append((time.perf_counter() - began, float(value)))
        print(f'  closure {len(calls)}: {calls[-1][0]:.3f} s loss={calls[-1][1]:.6e}', flush=True)
        return value

    began = time.perf_counter()
    try:
        lbfgs.step(closure)
        log(f'one L-BFGS step: {time.perf_counter() - began:.2f} s, closure calls={len(calls)}')
    except (FloatingPointError, RuntimeError) as exc:
        log(f'L-BFGS step stopped after {len(calls)} closure calls: {type(exc).__name__}: {exc}')
    if calls:
        log(f'mean closure time {sum(c[0] for c in calls) / len(calls):.3f} s')
    output.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
