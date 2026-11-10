"""Smoke test on the two-layer slab: M0 and DCSNN with the V2 recipe, compared with FEM nodal values.

Recipe: boundary losses (no exact lifting for mixed boundaries), PDE + energy losses, residual-focused
resampling every 100 steps, 800 Adam steps and 20 L-BFGS steps at the end. Finished runs are reused.
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.build import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.trainer import run

PROBLEM = ROOT / 'problems' / 'two_layer_slab' / 'problem.json'
FEM = ROOT / 'results' / 'fem_problems' / 'two_layer_slab' / 'fem_l1.npz'
RESULTS = ROOT / 'results' / 'pinn_smoke'
ARCHITECTURES = ('M0', 'DCSNN')
ADAM_STEPS, LBFGS_STEPS, RESAMPLE_EVERY = 800, 20, 100
WEIGHTS = dict(pde=1., energy=100., dirichlet=100., neumann=10., robin=10.)
WIDTH, DEPTH = 32, 3


def predict(problem, model, xy):
    """Temperature at xy from the network of the owning region."""
    owner = problem.owner_index(xy)
    values = np.full(len(xy), np.nan)
    for index, region in enumerate(problem.regions):
        mine = owner == index
        if not mine.any():
            continue
        points = torch.as_tensor(xy[mine], dtype=torch.float64)
        with torch.no_grad():
            output = model.field(region, points)
        temperature = output[0] if isinstance(output, tuple) else output
        values[mine] = temperature.detach().cpu().numpy().ravel()
    return values


def train(problem, architecture, run_dir):
    """Train once; a finished run directory is reused as it is."""
    if run_dir.joinpath('result.json').exists():
        return json.loads(run_dir.joinpath('result.json').read_text(encoding='utf-8'))
    return run(problem, architecture, run_dir, adam_steps=ADAM_STEPS, lbfgs_steps=LBFGS_STEPS,
        width=WIDTH, depth=DEPTH, weights=WEIGHTS, sampling='mixed', resample_every=RESAMPLE_EVERY, focused=True)


def compare(problem, architecture, run_dir):
    model = build_model(problem, architecture, WIDTH, DEPTH, 'tanh').double()
    model.load_state_dict(torch.load(run_dir / 'latest.pt', map_location='cpu'))
    model.eval()
    fem = np.load(FEM)
    xy, t_fem = fem['dof_xy'], fem['dof_T']
    t_pinn = predict(problem, model, xy)
    error = t_pinn - t_fem
    return dict(points=int(len(xy)), rmse_K=float(np.sqrt(np.nanmean(error ** 2))),
        max_abs_error_K=float(np.nanmax(np.abs(error))), fem_Tmax_C=float(t_fem.max()),
        pinn_Tmax_C=float(np.nanmax(t_pinn)), nan_points=int(np.isnan(t_pinn).sum()))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for architecture in ARCHITECTURES:
        run_dir = RESULTS / f'two_layer_slab_{architecture}_local'
        try:
            problem = GenericProblem(load_problem(PROBLEM).doc)
            result = train(problem, architecture, run_dir)
            summary[architecture] = dict(status=result.get('status'), training_error=result.get('error'),
                **compare(problem, architecture, run_dir))
        except Exception:
            summary[architecture] = dict(error=traceback.format_exc())
    RESULTS.joinpath('smoke_summary_local.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
