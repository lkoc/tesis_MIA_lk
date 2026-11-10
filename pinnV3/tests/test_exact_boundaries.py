"""Strong boundary construction for generic pinn-v3 problems."""
import numpy as np
import pytest
import torch

from pinn_v3.problem.build import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model

PROBLEM = 'pinnV3/problems/two_layer_slab/problem.json'
MODELS = ('M0_EXACT_BC', 'M1_EXACT_BC', 'DCSNN_EXACT_BC')
SIDES = {
    'bottom': (np.array([[1.0, 0.0]]), np.array([[0.0, -1.0]])),
    'top': (np.array([[1.0, 1.0]]), np.array([[0.0, 1.0]])),
    'left': (np.array([[0.0, 0.2]]), np.array([[-1.0, 0.0]])),
    'right': (np.array([[2.0, 0.2]]), np.array([[1.0, 0.0]])),
}


@pytest.fixture(scope='module')
def problem():
    return GenericProblem(load_problem(PROBLEM).doc)


def boundary_residuals(problem, model, side):
    xy_np, normal_np = SIDES[side]
    region_id = int(problem.owner_index(xy_np)[0])
    region = problem.regions[region_id]
    xy = torch.as_tensor(xy_np, dtype=torch.float64).requires_grad_(True)
    temperature, _ = model.field(region, xy)
    gradient = torch.autograd.grad(temperature.sum(), xy, create_graph=True)[0]
    normal = torch.as_tensor(normal_np, dtype=torch.float64)
    flux = -problem.conductivity(region, xy) * (gradient * normal).sum(dim=1, keepdim=True)
    condition = problem.doc['boundaries'][side]
    if condition['type'] == 'dirichlet':
        target = problem._bc(condition)['value'](xy)
        return (temperature - target).detach().item()
    if condition['type'] == 'neumann':
        target = problem._bc(condition)['value'](xy)
    else:
        ambient = problem._bc(condition)['T_inf'](xy)
        target = condition['h_W_m2K'] * (temperature - ambient)
    return (flux - target).detach().item()


@pytest.mark.parametrize('architecture', MODELS)
def test_model_builds_with_strong_boundary_wrapper(problem, architecture):
    model = build_model(problem, architecture, width=16, depth=2, activation='tanh').double()
    assert model.exact_boundary
    for side in SIDES:
        assert abs(boundary_residuals(problem, model, side)) < 1e-8
