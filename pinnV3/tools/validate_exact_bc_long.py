"""Long generic V3 validation of exact-boundary PINNs against FEM nodal results."""
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

CASES = ('mms_sine', 'cable_disk')
ARCHITECTURES = ('M1_EXACT_BC', 'DCSNN_EXACT_BC')
STEPS, LBFGS, REFRESH = 8000, 20, 100
WEIGHTS = dict(pde=1.0, energy=100.0)
RESULTS = ROOT / 'results' / 'exact_bc_long'


def predict(problem, model, xy):
    """Evaluate the region network at the same FEM points."""
    owner = problem.owner_index(xy)
    values = np.full(len(xy), np.nan)
    for region in problem.regions:
        selected = owner == region.id
        if not selected.any():
            continue
        points = torch.as_tensor(xy[selected], dtype=torch.float64)
        with torch.no_grad():
            output = model.field(region, points)
        temperature = output[0] if isinstance(output, tuple) else output
        values[selected] = temperature.detach().cpu().numpy().ravel()
    return values


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for case_id in CASES:
        source = ROOT / 'problems' / case_id / 'problem.json'
        fem_path = ROOT / 'results' / 'fem_generic' / case_id / 'fem_l1.npz'
        for architecture in ARCHITECTURES:
            key = f'{case_id}:{architecture}'
            run_dir = RESULTS / case_id / architecture
            try:
                document = load_problem(source).doc
                problem = GenericProblem(document)
                if run_dir.joinpath('result.json').exists():
                    result = json.loads(run_dir.joinpath('result.json').read_text(encoding='utf-8'))
                else:
                    result = run(problem, architecture, run_dir, adam_steps=STEPS, lbfgs_steps=LBFGS,
                        width=32, depth=3, weights=WEIGHTS, sampling='mixed', focused=True,
                        focus_criterion='mixed', resample_every=REFRESH, exact_energy=True)
                model = build_model(problem, architecture, 32, 3, 'tanh').double()
                model.load_state_dict(torch.load(run_dir / 'latest.pt', map_location='cpu'))
                model.eval()
                fem = np.load(fem_path)
                prediction = predict(problem, model, fem['dof_xy'])
                error = prediction - fem['dof_T']
                reference = problem.reference(fem['dof_xy'])
                summary[key] = dict(status=result.get('status'), training_error=result.get('error'),
                    steps=STEPS, lbfgs_steps=LBFGS, resample_every=REFRESH, focus_criterion='mixed',
                    points=int(len(error)), nan_points=int(np.isnan(prediction).sum()),
                    rmse_K=float(np.sqrt(np.nanmean(error ** 2))),
                    max_abs_error_K=float(np.nanmax(np.abs(error))),
                    fem_Tmax_C=float(fem['dof_T'].max()), pinn_Tmax_C=float(np.nanmax(prediction)),
                    reference_rmse_K=(float(np.sqrt(np.nanmean((prediction - reference) ** 2)))
                        if reference is not None else None))
            except Exception:
                summary[key] = dict(error=traceback.format_exc())
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
