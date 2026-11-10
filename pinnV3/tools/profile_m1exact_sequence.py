"""Profile M1_EXACT along the smoke sequence: FEM copy, Adam, the acceptance audit, then one L-BFGS step."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.evaluation.audit import evaluate_model
from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.fem_initializer import fem_initialization
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.objective import DEFAULT_WEIGHTS, Objective

MAX_CALLS = 40
ADAM_STEPS = 30
POINTS = dict(n_interior=200, n_layer=100, n_face=24, n_boundary=24)
FEM_SETTINGS = dict(source='fem', fem_node_multiplier=1.0, mode='extrapolate', pretrain_steps=0,
                    fem_transfer='readout_gls', fem_gradient_weight=1.0,
                    fem_ridge=float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'none' else 0.0)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'aras_single'
    output = ROOT / 'results' / 'profile_m1exact_sequence_20261031'
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

    began = time.perf_counter()
    if len(sys.argv) > 2 and sys.argv[2] == 'none':
        log('no FEM copy')
    else:
        report = fem_initialization(model, problem, objective, FEM_SETTINGS, output)
        log(f'FEM copy: {time.perf_counter() - began:.2f} s, rms={report["rms_K"]:.3f} K, weight_norm={report["weight_norm"]:.2f}')
    with torch.no_grad():
        total, _, _ = objective()
    log(f'loss after copy: {float(total):.4e}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    began = time.perf_counter()
    for step in range(ADAM_STEPS):
        optimizer.zero_grad()
        total, _, _ = objective()
        total.backward()
        optimizer.step()
        objective.update_attention()
    log(f'{ADAM_STEPS} Adam steps: {time.perf_counter() - began:.2f} s, loss={float(total.detach()):.4e}')

    began = time.perf_counter()
    report, _ = evaluate_model(model, None, 'profile', include_audit=True)
    log(f'evaluate_model with audit: {time.perf_counter() - began:.2f} s')

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1, history_size=50, line_search_fn='strong_wolfe')
    calls = []

    def closure():
        if len(calls) >= MAX_CALLS:
            raise RuntimeError(f'closure budget of {MAX_CALLS} calls exhausted')
        started = time.perf_counter()
        lbfgs.zero_grad()
        value, _, _ = objective()
        value.backward()
        calls.append((time.perf_counter() - started, float(value.detach())))
        print(f'  closure {len(calls)}: {calls[-1][0]:.3f} s loss={calls[-1][1]:.6e}', flush=True)
        return value

    began = time.perf_counter()
    try:
        lbfgs.step(closure)
        log(f'one L-BFGS step: {time.perf_counter() - began:.2f} s, closures={len(calls)}')
    except (FloatingPointError, RuntimeError) as exc:
        log(f'L-BFGS stopped after {len(calls)} closures in {time.perf_counter() - began:.2f} s: {type(exc).__name__}: {exc}')
    if calls:
        finite = [c[1] for c in calls if np.isfinite(c[1])]
        log(f'mean closure {np.mean([c[0] for c in calls]):.3f} s, finite losses {len(finite)}/{len(calls)}')
    (ROOT / 'results' / 'profile_m1exact_sequence_20261031.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
