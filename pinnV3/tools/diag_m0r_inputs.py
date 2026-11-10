"""Diagnose input sizes of RegionScaledGlobal (M0R) on a generic case."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'aras_single'
    problem = GenericProblem(load_problem(ROOT / 'problems' / case / 'problem.json'))
    region = problem.regions[0]
    print('regions', len(problem.regions), 'source_disks', len(problem.source_disks))
    print('input_size global', problem.input_size(region, 'global'))
    xy = torch.rand(5, 2, dtype=torch.float64) * 0.5
    labels = [r.id for r in problem.regions]
    model = build_model(problem, 'M0R', 12, 2, 'tanh').double()
    print('first linear in_features', model.networks[0][0].in_features)
    print('coordinates dim', problem.coordinates(region, xy, 'global').shape[1])
    try:
        T, _ = model.field(region, xy)
        print('field ok', tuple(T.shape))
    except Exception as exc:
        print('field FAILED', type(exc).__name__, str(exc)[:200])


if __name__ == '__main__':
    main()
