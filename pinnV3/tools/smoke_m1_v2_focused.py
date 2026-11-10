"""V2-compatible M1 smoke: soft boundaries, PDE focus and fixed surfaces."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.trainer import run

CASE = 'cable_disk'
ARCHITECTURE = 'M1'
OUT = ROOT / 'results' / 'm1_v2_focused_smoke_v2' / ARCHITECTURE
FEM = ROOT / 'results' / 'fem_generic' / CASE / 'fem_l1.npz'


def main():
    problem = GenericProblem(load_problem(ROOT / 'problems' / CASE / 'problem.json').doc)
    try:
        result = run(problem, ARCHITECTURE, OUT, adam_steps=800, lbfgs_steps=20,
            width=32, depth=3, weights={'pde': 1.0, 'energy': 100.0}, sampling='mixed',
            focused=True, focus_criterion='residual', focus_surfaces=False, resample_every=100)
        model = build_model(problem, ARCHITECTURE, 32, 3, 'tanh').double()
        model.load_state_dict(torch.load(OUT / 'latest.pt', map_location='cpu'))
        model.eval()
        fem = np.load(FEM)
        owner = problem.owner_index(fem['dof_xy'])
        prediction = np.full(len(owner), np.nan)
        for region in problem.regions:
            selected = owner == region.id
            if selected.any():
                points = torch.as_tensor(fem['dof_xy'][selected], dtype=torch.float64)
                with torch.no_grad():
                    prediction[selected] = model.field(region, points)[0].detach().cpu().numpy().ravel()
        error = prediction - fem['dof_T']
        report = dict(training_status=result.get('status'), final_loss=result.get('final_loss'),
            final_parts=result.get('final_parts'), points=int(len(error)), nan_points=int(np.isnan(prediction).sum()),
            rmse_K=float(np.sqrt(np.nanmean(error ** 2))), max_abs_error_K=float(np.nanmax(np.abs(error))),
            fem_Tmax_C=float(np.nanmax(fem['dof_T'])), pinn_Tmax_C=float(np.nanmax(prediction)))
    except Exception:
        report = dict(status='failed', error=traceback.format_exc())
    (OUT.parent / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()