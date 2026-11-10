"""Checks for generic V3 region scales and cable topology."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_v3.evaluation.metrics import temperature_summary
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.generic_problem import GenericProblem


def minimal(size=1.0, k=2.0, t_scale=10.0):
    """Single-region square with constant conductivity and uniform Dirichlet sides."""
    return {'schema': 'pinn-v3-problem-1', 'id': 'dimension_check',
            'domain': {'shape': 'rectangle', 'bounds': [0.0, size, 0.0, size]},
            'regions': [{'id': 'soil', 'priority': 0,
                         'shape': {'type': 'rectangle', 'bounds': [0.0, size, 0.0, size]},
                         'conductivity': k}],
            'boundaries': {side: {'type': 'dirichlet', 'value': 20.0} for side in ('bottom', 'top', 'left', 'right')},
            'temperature_scale_K': t_scale}


def scales(doc):
    problem = GenericProblem(doc)
    return problem.scales(problem.regions[0])


def test_length_doubling_halves_flux_and_quarters_divergence():
    small, large = scales(minimal(size=1.0)), scales(minimal(size=2.0))
    assert large['flux'] / small['flux'] == pytest.approx(0.5, rel=1e-6)
    assert large['divergence'] / small['divergence'] == pytest.approx(0.25, rel=1e-6)
    assert large['temperature'] == small['temperature']


def test_conductivity_scales_flux_and_divergence_linearly():
    base, doubled = scales(minimal(k=2.0)), scales(minimal(k=4.0))
    assert doubled['flux'] / base['flux'] == pytest.approx(2.0, rel=1e-9)
    assert doubled['divergence'] / base['divergence'] == pytest.approx(2.0, rel=1e-9)


def test_temperature_scale_scales_flux_and_divergence_linearly():
    base, doubled = scales(minimal(t_scale=10.0)), scales(minimal(t_scale=20.0))
    assert doubled['flux'] / base['flux'] == pytest.approx(2.0, rel=1e-9)
    assert doubled['divergence'] / base['divergence'] == pytest.approx(2.0, rel=1e-9)


def test_temperature_summary_handles_missing_conductor_points_and_weights():
    from pinn_v3.evaluation.metrics import temperature_summary

    summary = temperature_summary(np.array([20.0, 21.0]), np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([0, 0]), np.ones(2), ['soil', 'thin_conductor'], [1], np.ones(2, dtype=bool))
    zero_weight_summary = temperature_summary(np.array([20.0, 21.0, 23.0]),
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), np.array([0, 1, 1]),
        np.array([1.0, 0.0, 0.0]), ['soil', 'thin_conductor'], [1], np.ones(3, dtype=bool))

    assert summary['conductor_mean_C'] == {}
    assert zero_weight_summary['conductor_mean_C']['thin_conductor'] == pytest.approx(22.0)


def test_generic_cable_topology_builds_m1_exact():
    root = Path(__file__).resolve().parents[1]
    from pinn_v3.problem.repository import load_problem
    definition = load_problem(root / 'problems' / 'aras_flat' / 'problem.json')
    problem = GenericProblem(definition)

    assert problem.definition is definition
    assert len(problem.cable_layout) == 3
    assert all(region.kind == 'conductor' for region in problem.regions[1:])
    assert len(problem.circular_interfaces) == 12
    assert {interface.cable_index for interface in problem.circular_interfaces} == {0, 1, 2}
    assert all(problem.areas[region.id] > 0 for region in problem.regions)

    xy, labels, weights = problem.evaluation(per_region=4, seed=7)
    summary = temperature_summary(np.zeros(len(xy)), xy, labels, weights,
        [region.name for region in problem.regions], problem.conductors(), np.ones(len(xy), dtype=bool))
    assert len(summary['conductor_mean_C']) == 12

    model = build_model(problem, 'M1_EXACT', width=8, depth=2, activation='tanh')
    assert model.architecture == 'M1_EXACT'
    assert model.kind == 'exact_interfaces'


def test_generic_cable_exact_focused_objective_supports_backward():
    from pinn_v3.solvers.pinn.objective import Objective

    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / 'problems' / 'aras_flat' / 'problem.json').read_text(encoding='utf-8'))
    problem = GenericProblem(document)
    torch.manual_seed(0)
    model = build_model(problem, 'M1_EXACT', width=8, depth=2, activation='tanh',
                        input_normalization='unit_box_v1').double()
    objective = Objective(model, n_interior=4, n_layer=4, n_face=4, n_boundary=4,
                          seed=11, focused=True, focus_criterion='residual', focus_surfaces=False)

    assert all(points.shape == (4, 2) and torch.isfinite(points).all()
               for points in objective.interior.values())
    total, _, _ = objective()
    assert torch.isfinite(total)
    total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_vectorized_exact_disk_lifting_matches_mode_recurrence():
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / 'problems' / 'aras_flat' / 'problem.json').read_text(encoding='utf-8'))
    problem = GenericProblem(document)
    model = build_model(problem, 'M1_EXACT', width=8, depth=2, activation='tanh').float()
    interface = model._interfaces_by_cable[0][0]
    region = problem.regions[interface.left]
    xy = torch.as_tensor(problem.sample_interior(region, 16, seed=13), dtype=torch.float32)
    with torch.no_grad():
        model.trace_coefficients.uniform_(-0.02, 0.02)

    center = model._interface_center(interface)
    delta = xy[:, :2] - center
    radius = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    cos_theta, sin_theta = delta[:, :1] / radius, delta[:, 1:2] / radius
    reference_basis = [torch.ones_like(radius)]
    cos_n, sin_n = torch.ones_like(radius), torch.zeros_like(radius)
    for _ in range(model.fourier_order):
        cos_n, sin_n = cos_n * cos_theta - sin_n * sin_theta, sin_n * cos_theta + cos_n * sin_theta
        reference_basis.extend((cos_n, sin_n))
    torch.testing.assert_close(model._angular_basis(xy, center), torch.cat(reference_basis, dim=1),
                               rtol=2e-5, atol=2e-5)

    normalized = (xy[:, :2] - model._interface_center(interface)) / interface.radius
    radius_squared = normalized.square().sum(dim=1, keepdim=True)
    real, imaginary = torch.ones_like(radius_squared), torch.zeros_like(radius_squared)
    harmonics = [real]
    for _ in range(model.fourier_order):
        real, imaginary = (real * normalized[:, :1] - imaginary * normalized[:, 1:2],
                            real * normalized[:, 1:2] + imaginary * normalized[:, :1])
        harmonics.extend((real, imaginary))
    index = model.interface_index[interface.name]
    temperature = model._temperature_coefficients(interface).clone()
    temperature[0] += model.outer_temperature
    flux = model._flux_coefficients(interface)
    conductivity = problem.conductivity(region, xy)
    expected = torch.zeros_like(radius_squared)
    for mode in range(2 * model.fourier_order + 1):
        order = 0 if mode == 0 else (mode + 1) // 2
        derivative = -flux[mode] / conductivity
        correction = (interface.radius * derivative - order * temperature[mode]) / 2
        expected += harmonics[mode] * (temperature[mode] + correction * (radius_squared - 1))

    actual = model._disk_lifting(interface, xy, region)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    global_points = torch.as_tensor(problem.sample_interior(problem.regions[1], 3, seed=10), dtype=torch.float32)
    global_features = problem.coordinates(problem.regions[1], global_points, 'global')
    assert global_features.shape[1] == problem.input_size(problem.regions[1], 'global')
    assert torch.all(global_features[:, 5 + problem.regions[1].id] == 1)


def test_generic_annulus_builds_m1_exact():
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / 'problems' / 'annulus' / 'problem.json').read_text(encoding='utf-8'))
    problem = GenericProblem(document)

    model = build_model(problem, 'M1_EXACT', width=8, depth=2, activation='tanh')
    assert model.architecture == 'M1_EXACT'
    assert model.boundary_mode == 'annulus'
    xy = torch.as_tensor(problem.sample_interior(problem.regions[0], 4, seed=7), dtype=torch.float32)
    temperature, flux = model.field(problem.regions[0], xy)
    assert temperature.shape == (4, 1)
    assert flux is None
