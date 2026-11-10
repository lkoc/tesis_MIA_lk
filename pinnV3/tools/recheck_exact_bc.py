"""Re-evaluate saved exact-boundary checkpoints after a parametrization fix."""
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

CASES = ('mms_sine', 'cable_disk')
ARCHITECTURES = ('M1_EXACT_BC', 'DCSNN_EXACT_BC')
OUTPUT = ROOT / 'results' / 'exact_bc_long' / 'corrected_parametrization_summary.json'


def predict(problem, model, xy):
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
    report = {}
    for case_id in CASES:
        problem_path = ROOT / 'problems' / case_id / 'problem.json'
        fem_path = ROOT / 'results' / 'fem_generic' / case_id / 'fem_l1.npz'
        problem = GenericProblem(load_problem(problem_path).doc)
        fem = np.load(fem_path)
        for architecture in ARCHITECTURES:
            run_dir = ROOT / 'results' / 'exact_bc_long' / case_id / architecture
            model = build_model(problem, architecture, 32, 3, 'tanh').double()
            model.load_state_dict(torch.load(run_dir / 'latest.pt', map_location='cpu'))
            model.eval()
            prediction = predict(problem, model, fem['dof_xy'])
            error = prediction - fem['dof_T']
            key = f'{case_id}:{architecture}'
            report[key] = dict(status='inference_only_corrected_parameterization', points=int(len(error)),
                nan_points=int(np.isnan(prediction).sum()), rmse_K=float(np.sqrt(np.nanmean(error ** 2))),
                max_abs_error_K=float(np.nanmax(np.abs(error))), fem_Tmax_C=float(np.nanmax(fem['dof_T'])),
                pinn_Tmax_C=float(np.nanmax(prediction)), pinn_Tmin_C=float(np.nanmin(prediction)),
                checkpoint=str(run_dir / 'latest.pt'))
    OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
