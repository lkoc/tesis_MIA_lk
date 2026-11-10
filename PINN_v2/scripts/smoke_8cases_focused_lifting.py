"""Smoke test: {M1, DCSNN} x {BC as loss, exact Dirichlet lifting} x {uniform, focused sampling} on aras_flat.

Focused = worst-residual points from uniform + log-radial candidates, plus fresh log-radial points.
Run from PINN_v2: python scripts/smoke_8cases_focused_lifting.py
"""
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.experiments.exact_cable_transfer_campaign import case_definitions
from pinn_v2.physics.residuals import heat_residual
from pinn_v2.solvers.pinn import trainer
from pinn_v2.solvers.pinn.networks import build_model, install_dirichlet_lifting
from pinn_v2.solvers.pinn.objective import Objective
from pinn_v2.solvers.pinn.problem import load_problem

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'phase7' / 'p7_smoke_8cases_800_20261023'
OUT.mkdir(parents=True, exist_ok=True)
problem = load_problem(case_definitions()['aras_flat'])
reference = load_reference_npz(ROOT / 'results' / 'phase7' / 'p7_exact_cable_transfer_20261021' / 'aras_flat' / 'fem' / 'fem_l1.npz', problem)
ADAM_STEPS = 800
LBFGS_CALLS = 10
LBFGS_ITER = 10
REFRESH = 100
ENERGY = 100.
LABEL = 'FEM P2 level 1, aras_flat (evaluation only)'
SOIL = problem.regions[0]
CENTERS = np.asarray(problem.case['cables'], dtype=float)
R_CABLE = float(problem.case['radius'])
R_MAX = 4.0


def uniform_candidates(n, rng):
    x0, x1, y0, y1 = problem.bounds
    return rng.uniform([x0, y0], [x1, y1], (n, 2))


def log_radial_candidates(n, rng):
    index = rng.integers(0, len(CENTERS), n)
    angle = rng.uniform(0., 2 * np.pi, n)
    radius = R_CABLE * np.exp(rng.uniform(0., np.log(R_MAX / R_CABLE), n))
    return CENTERS[index] + np.c_[radius * np.cos(angle), radius * np.sin(angle)]


def pde_score(model, points):
    xy = torch.as_tensor(points, dtype=torch.float64).requires_grad_(True)
    temperature, _ = model.field(SOIL, xy)
    residual = heat_residual(temperature, xy, problem.conductivity(SOIL, xy), problem.source(SOIL, xy).detach())
    return (residual / problem.scales(SOIL)['divergence']).detach().abs().ravel().numpy()


def refresh(objective, model, rng):
    pool = np.vstack([uniform_candidates(4096, rng), log_radial_candidates(4096, rng)])
    pool = pool[problem.contains_points(SOIL, pool)]
    worst = pool[np.argsort(-pde_score(model, pool))[:256]]
    fresh = log_radial_candidates(8192, rng)
    fresh = fresh[problem.contains_points(SOIL, fresh)][:512 - len(worst)]
    points = np.vstack([worst, fresh])
    objective.interior[SOIL.id] = torch.as_tensor(points, dtype=torch.float64).requires_grad_(True)
    objective.sources[SOIL.id] = problem.source(SOIL, objective.interior[SOIL.id]).detach()


def run(architecture, boundary, sampling):
    torch.manual_seed(0)
    rng = np.random.default_rng(123)
    if architecture == 'M1':
        model = build_model(problem, 'M1', width=32, depth=3, activation='tanh',
                            input_normalization='unit_box_v1', trace_parameterization='legacy', soil_envelope_gain=1.)
    else:
        model = build_model(problem, architecture, width=32, depth=3, activation='tanh',
                            input_normalization=None, trace_parameterization='legacy', soil_envelope_gain=1.)
    weights = dict(energy=ENERGY)
    if boundary == 'exact':
        install_dirichlet_lifting(model, problem)
        weights['dirichlet'] = 0.
    objective = Objective(model, n_interior=512, n_layer=256, n_face=128, n_boundary=128, seed=11,
                          sampling='uniform' if sampling == 'uniform' else 'mixed', weights=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    started = time.perf_counter()
    for step in range(ADAM_STEPS):
        if sampling == 'focused' and step > 0 and step % REFRESH == 0:
            refresh(objective, model, rng)
        optimizer.zero_grad()
        total, _, _ = objective()
        total.backward()
        optimizer.step()
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=LBFGS_ITER, history_size=50, line_search_fn='strong_wolfe')

    def closure():
        lbfgs.zero_grad()
        total, _, _ = objective()
        total.backward()
        return total

    for _ in range(LBFGS_CALLS):
        lbfgs.step(closure)
    final, _ = trainer.evaluate_model(model, reference, LABEL, include_audit=True)
    interfaces = final.get('interfaces', [])
    return dict(elapsed_s=round(time.perf_counter() - started, 1), Tmax_pred_C=final.get('Tmax_pred_C'),
        error_Tmax_abs_K=final.get('error_Tmax_abs_K'), nrmse_pct=final.get('nrmse_pct'),
        balance_mismatch_pct=final.get('balance', {}).get('global_derived_pct'),
        max_interface_T_jump_K=max((item.get('T_jump_max_K', 0.) for item in interfaces), default=None),
        max_interface_flux_jump_pct=max((item.get('flux_jump_max_pct', 0.) for item in interfaces), default=None))


CASES = [
    ('M1_bcloss_uniform', 'M1', 'loss', 'uniform'),
    ('M1_bcloss_focused', 'M1', 'loss', 'focused'),
    ('M1_bcexact_uniform', 'M1', 'exact', 'uniform'),
    ('M1_bcexact_focused', 'M1', 'exact', 'focused'),
    ('DCSNN_bcloss_uniform', 'DCSNN', 'loss', 'uniform'),
    ('DCSNN_bcloss_focused', 'DCSNN', 'loss', 'focused'),
    ('DCSNN_bcexact_uniform', 'DCSNN', 'exact', 'uniform'),
    ('DCSNN_bcexact_focused', 'DCSNN', 'exact', 'focused'),
]

if __name__ == '__main__':
    results = {}
    for name, architecture, boundary, sampling in CASES:
        try:
            results[name] = run(architecture, boundary, sampling)
        except Exception as exc:
            results[name] = dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-800:])
        print(name, json.dumps(results[name]), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('DONE', OUT, flush=True)