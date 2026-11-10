"""Inspect the trained cable checkpoint by region and interface."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.physics.residuals import heat_residual
from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.objective import flux_fields

CASE = 'cable_disk'
ARCHITECTURE = 'M1_EXACT_BC'
RUN = ROOT / 'results' / 'cable_interface_fix_smoke_v3' / ARCHITECTURE
OUTPUT = ROOT / 'results' / 'cable_interface_fix_smoke_v3' / 'diagnostic.json'


def main():
    problem = GenericProblem(load_problem(ROOT / 'problems' / CASE / 'problem.json').doc)
    model = build_model(problem, ARCHITECTURE, 32, 3, 'tanh').double()
    model.load_state_dict(torch.load(RUN / 'latest.pt', map_location='cpu'))
    model.eval()
    report = {'regions': {}, 'interfaces': []}
    for region in problem.regions:
        xy_np = problem.sample_interior(region, 4000, 20261025 + region.id)
        xy = torch.as_tensor(xy_np, dtype=torch.float64).requires_grad_(True)
        temperature, _ = model.field(region, xy)
        k = problem.conductivity(region, xy)
        source = problem.source(region, xy).detach()
        residual = heat_residual(temperature, xy, k, source)
        report['regions'][region.name] = dict(
            temperature_min_C=float(temperature.detach().min()),
            temperature_max_C=float(temperature.detach().max()),
            temperature_mean_C=float(temperature.detach().mean()),
            source_min_W_m3=float(source.min()),
            source_max_W_m3=float(source.max()),
            residual_rms=float(torch.sqrt(torch.mean(residual.detach() ** 2))),
            region_power_W_m=float(problem.region_power[region.id]),
            sample_points=len(xy_np))
    for face in problem.faces(512):
        left, right = problem.regions[face.left], problem.regions[face.right]
        xy = torch.as_tensor(face.xy, dtype=torch.float64).requires_grad_(True)
        normal = torch.as_tensor(face.normal, dtype=torch.float64)
        tl, ql, _, _ = flux_fields(model, left, xy)
        tr, qr, _, _ = flux_fields(model, right, xy)
        report['interfaces'].append(dict(
            name=face.name, left=left.name, right=right.name,
            temperature_jump_rms=float(torch.sqrt(torch.mean((tl - tr).detach() ** 2))),
            flux_jump_rms=float(torch.sqrt(torch.mean((((ql - qr) * normal).sum(1, keepdim=True)).detach() ** 2))),
            left_temperature_max=float(tl.detach().max()), right_temperature_max=float(tr.detach().max())))
    OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
