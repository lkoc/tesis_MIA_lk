"""Save an inference snapshot from any completed or running exact-BC checkpoint."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model

CASE = 'mms_sine'
ARCHITECTURE = 'M1_EXACT_BC'
RUN = ROOT / 'results' / 'exact_bc_long' / CASE / ARCHITECTURE
FEM = ROOT / 'results' / 'fem_generic' / CASE / 'fem_l1.npz'
OUTPUT = ROOT / 'results' / 'exact_bc_long' / CASE / f'{ARCHITECTURE}_intermediate.json'


def main():
    problem = GenericProblem(load_problem(ROOT / 'problems' / CASE / 'problem.json').doc)
    model = build_model(problem, ARCHITECTURE, 32, 3, 'tanh').double()
    model.load_state_dict(torch.load(RUN / 'latest.pt', map_location='cpu'))
    model.eval()
    fem = np.load(FEM)
    owner = problem.owner_index(fem['dof_xy'])
    prediction = np.full(len(owner), np.nan)
    for region in problem.regions:
        selected = owner == region.id
        if not selected.any():
            continue
        points = torch.as_tensor(fem['dof_xy'][selected], dtype=torch.float64)
        with torch.no_grad():
            output = model.field(region, points)
        temperature = output[0] if isinstance(output, tuple) else output
        prediction[selected] = temperature.detach().cpu().numpy().ravel()
    error = prediction - fem['dof_T']
    result = dict(case=CASE, architecture=ARCHITECTURE, checkpoint=str(RUN / 'latest.pt'),
        fem_points=int(len(prediction)), nan_points=int(np.isnan(prediction).sum()),
        rmse_K=float(np.sqrt(np.nanmean(error ** 2))), max_abs_error_K=float(np.nanmax(np.abs(error))),
        fem_Tmax_C=float(np.nanmax(fem['dof_T'])), pinn_Tmax_C=float(np.nanmax(prediction)),
        pinn_Tmin_C=float(np.nanmin(prediction)), status='intermediate_inference')
    OUTPUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
