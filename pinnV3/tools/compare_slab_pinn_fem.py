"""Train M0 on the region problem and compare its temperature with the FEM nodal values at the same points."""
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
RESULTS = ROOT / 'results' / 'pinn_problems'
ADAM_STEPS = 800
WIDTH, DEPTH = 32, 3


def predict(problem, model, xy):
    """Temperature of the model at xy, evaluated with the network of the owning region."""
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


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f'two_layer_slab_m0_{ADAM_STEPS}_r2'
    report = {}
    try:
        problem = GenericProblem(load_problem(PROBLEM).doc)
        run_dir = RESULTS / name
        if run_dir.joinpath('result.json').exists():
            # A finished run is reused; training is never repeated for the comparison.
            result = json.loads(run_dir.joinpath('result.json').read_text(encoding='utf-8'))
        else:
            result = run(problem, 'M0', run_dir, adam_steps=ADAM_STEPS, lbfgs_steps=0, width=WIDTH, depth=DEPTH)
        model = build_model(problem, 'M0', WIDTH, DEPTH, 'tanh').double()
        model.load_state_dict(torch.load(RESULTS / name / 'latest.pt', map_location='cpu'))
        model.eval()
        fem = np.load(FEM)
        xy, t_fem = fem['dof_xy'], fem['dof_T']
        t_pinn = predict(problem, model, xy)
        error = t_pinn - t_fem
        report = dict(run=name, status=result.get('status'), points=int(len(xy)),
            max_abs_error_K=float(np.nanmax(np.abs(error))), rmse_K=float(np.sqrt(np.nanmean(error ** 2))),
            fem_Tmin_C=float(t_fem.min()), fem_Tmax_C=float(t_fem.max()),
            pinn_Tmin_C=float(np.nanmin(t_pinn)), pinn_Tmax_C=float(np.nanmax(t_pinn)),
            nan_points=int(np.isnan(t_pinn).sum()))
    except Exception:
        report = dict(run=name, error=traceback.format_exc())
    RESULTS.joinpath(f'{name}_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
