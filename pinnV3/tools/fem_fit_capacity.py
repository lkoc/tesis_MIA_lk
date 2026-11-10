"""Capacity check: fit the M1 PINN network to the cached FEM level-1 field at collocation points, measuring error on all reference nodes."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.fem_initializer import points_result
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.rbf_fd import pretrain_pinn

REFERENCE = ROOT / 'results' / 'fem_generic' / 'cable_disk' / 'fem_l1.npz'
OUTPUT = ROOT / 'results' / 'fem_fit_capacity_20261030.json'
CONFIGS = ((12, 2, 20, 1e-3), (12, 2, 300, 1e-3), (12, 2, 3000, 1e-3), (12, 2, 3000, 1e-2),
           (32, 3, 300, 1e-3), (32, 3, 3000, 1e-3))
POINTS_PER_REGION = 150


def evaluate(model, definition, dof_xy, dof_T, owners):
    errors = []
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for position, region in enumerate(model.problem.regions):
            select = owners == position
            xy = torch.as_tensor(dof_xy[select], dtype=dtype)
            predicted = model.field(region, xy)[0].numpy().reshape(-1)
            errors.append(predicted - dof_T[select])
    error = np.concatenate(errors)
    return dict(rms_K=float(np.sqrt(np.mean(error ** 2))), max_K=float(np.abs(error).max()))


def main():
    generic = GenericProblem(load_problem(ROOT / 'problems' / 'cable_disk' / 'problem.json'))
    definition = generic.definition
    rng = np.random.default_rng(3)
    collocation = {position: definition.sample_region_interior(position, POINTS_PER_REGION, rng)
                   for position in range(len(definition.regions))}
    reference = np.load(REFERENCE)
    dof_xy, dof_T = reference['dof_xy'], reference['dof_T']
    result = points_result(definition, dof_xy, dof_T, collocation)
    owners = definition.owner(dof_xy[:, 0], dof_xy[:, 1])
    report = dict(points=int(len(result['xy'])), runs=[])
    for width, depth, steps, learning_rate in CONFIGS:
        torch.manual_seed(0)
        model = build_model(generic, 'M1', width, depth, 'tanh')
        started = time.perf_counter()
        pretrain = pretrain_pinn(model, result, steps=steps, learning_rate=learning_rate, target_points=None)
        elapsed = time.perf_counter() - started
        fit = evaluate(model, definition, dof_xy, dof_T, owners)
        report['runs'].append(dict(width=width, depth=depth, steps=steps, learning_rate=learning_rate,
            parameters=model.parameter_count(), normalized_loss_initial=pretrain['initial_loss'],
            normalized_loss_final=pretrain['final_loss'], pretrain_s=elapsed, **fit))
        print(json.dumps(report['runs'][-1]), flush=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
