"""Per-region breakdown of the PDE residual of a trained model: is the source balanced or ignored?"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.physics.residuals import heat_residual
from pinn_v3.problem.build import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model

PROBLEM = ROOT / 'problems' / 'two_layer_slab' / 'problem.json'
RUN = ROOT / 'results' / 'pinn_smoke' / 'two_layer_slab_M0_scaled'
OUTPUT = ROOT / 'results' / 'pinn_smoke' / 'pde_breakdown_M0_scaled.json'
POINTS, SEED = 2000, 5


def main():
    problem = GenericProblem(load_problem(PROBLEM).doc)
    model = build_model(problem, 'M0', 32, 3, 'tanh').double()
    model.load_state_dict(torch.load(RUN / 'latest.pt', map_location='cpu'))
    model.eval()
    report = {}
    for region in problem.regions:
        xy_np = problem.sample_interior(region, POINTS, seed=SEED)
        xy = torch.as_tensor(xy_np, dtype=torch.float64).requires_grad_(True)
        output = model.field(region, xy)
        T = output[0] if isinstance(output, tuple) else output
        k = problem.conductivity(region, xy)
        q = problem.source(region, xy).detach()
        residual = heat_residual(T, xy, k, q)
        # residual = -div(k grad T) - q, so div(k grad T) = -(residual + q).
        divergence = -(residual.detach() + q)
        scales = problem.scales(region)
        report[problem.doc['regions'][region.id]['id']] = dict(
            rms_source_W_m3=float(torch.sqrt(torch.mean(q ** 2))),
            rms_divergence_W_m3=float(torch.sqrt(torch.mean(divergence ** 2))),
            rms_residual_W_m3=float(torch.sqrt(torch.mean(residual.detach() ** 2))),
            normalized_pde_term=float(torch.mean((residual.detach() / scales['divergence']) ** 2)),
            divergence_scale=float(scales['divergence']))
    OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
