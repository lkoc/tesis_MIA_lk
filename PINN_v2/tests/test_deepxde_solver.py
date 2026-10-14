from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_v2.solvers.pinn import deepxde_solver
from pinn_v2.solvers.pinn.problem import load_problem
from pinn_v2.solvers.pinn.trainer import run

ROOT = Path(__file__).resolve().parents[1]
SMALL_POINTS = dict(n_interior=12, n_layer=8, n_face=8, n_boundary=8)


class _ConstantNetwork(torch.nn.Module):
    def __init__(self, region_count):
        super().__init__()
        self.outputs = torch.nn.Parameter(torch.zeros(region_count, dtype=torch.float64))
        self.temperature_offsets = torch.nn.Parameter(torch.zeros(region_count, dtype=torch.float64))

    def forward(self, inputs):
        return self.outputs.expand(len(inputs), -1)


def test_deepxde_temperature_has_trainable_regional_level_and_loads_legacy_state():
    problem = load_problem(ROOT / 'pinn_v2' / 'cases' / 'v2' / 'references' / 'cables' / 'aras_flat.json')
    network = _ConstantNetwork(len(problem.regions))
    model = deepxde_solver.DeepXDEModel(problem, network, 'FNN')
    xy = torch.zeros((len(problem.regions), 2), dtype=torch.float64)
    labels = torch.arange(len(problem.regions))
    offsets = torch.arange(len(problem.regions), dtype=torch.float64) / 10
    with torch.no_grad():
        network.temperature_offsets.copy_(offsets)

    expected = problem.T_ref + problem.T_scale * offsets
    assert torch.allclose(model.temperature(xy, labels).ravel(), expected)
    for region in problem.regions:
        value, _ = model.field(region, xy[:1])
        assert torch.allclose(value.squeeze(), expected[region.id])

    model.load_state_dict({'outputs': network.outputs.detach().clone()})
    assert torch.equal(network.temperature_offsets, torch.zeros_like(network.temperature_offsets))


@pytest.mark.parametrize('sampler', ('mixed', 'pseudo', 'LHS', 'Halton', 'Hammersley', 'Sobol'))
def test_deepxde_samplers_produce_reproducible_valid_per_region_anchors(sampler):
    pytest.importorskip('deepxde')
    problem = load_problem(ROOT / 'pinn_v2' / 'cases' / 'v2' / 'references' / 'cables' / 'aras_flat.json')
    import deepxde as dde

    for region in problem.regions:
        first = deepxde_solver.sample_deepxde_interior(dde, problem, region, 12, 729 + region.id, sampler)
        second = deepxde_solver.sample_deepxde_interior(dde, problem, region, 12, 729 + region.id, sampler)

        assert first.shape == (12, 2)
        assert np.isfinite(first).all()
        assert np.array_equal(first, second)
        assert problem.domain.contains(region, first).all(), region.name


def test_cable_region_features_stay_finite_at_cable_centers():
    problem = load_problem(ROOT / 'pinn_v2' / 'cases' / 'v2' / 'references' / 'cables' / 'aras_flat.json')
    centers = torch.tensor(problem.case['cables'], dtype=torch.float64)

    for region in problem.regions:
        features = problem.coordinates(region, centers, 'region')
        assert torch.isfinite(features).all(), region.name


@pytest.mark.parametrize('internal_model', ('PFNN', 'FNN'))
def test_deepxde_internal_networks_train_and_reload(tmp_path, internal_model):
    pytest.importorskip('deepxde')
    case = ROOT / 'pinn_v2' / 'cases' / 'data' / 'mms_interface.json'
    output = tmp_path / internal_model
    result = run(case, 'DeepXDE', output, deepxde_model=internal_model, adam_steps=2,
                 lbfgs_steps=1 if internal_model == 'PFNN' else 0, width=8, depth=2,
                 points=SMALL_POINTS, log_every=1)

    assert result['status'] == 'completed', result.get('error')
    assert result['config']['deepxde_model'] == internal_model
    assert np.isfinite(result['final_loss'])
    assert result['reference_used_in_loss'] is False

    model = deepxde_solver.load_model(output, load_problem(case))
    value = model.temperature(torch.tensor([[0.25, 0.5]], dtype=torch.float64), torch.tensor([0]))
    assert torch.isfinite(value).all()


def test_deepxde_pfnn_trains_shared_cable_case(tmp_path):
    pytest.importorskip('deepxde')
    case = ROOT / 'pinn_v2' / 'cases' / 'v2' / 'references' / 'cables' / 'aras_flat.json'
    result = run(case, 'DeepXDE', tmp_path / 'cable', deepxde_model='PFNN', adam_steps=1,
                 width=8, depth=1, points=dict(n_interior=8, n_layer=6, n_face=4, n_boundary=4),
                 log_every=1, deepxde_sampling='Sobol')

    assert result['status'] == 'completed', result.get('error')
    assert np.isfinite(result['final_loss'])
    assert result['config']['deepxde_sampling'] == 'Sobol'
    assert result['physics_sha256']
    assert result['fem_labels_used'] == 0

    problem = load_problem(case)
    model = deepxde_solver.load_model(tmp_path / 'cable', problem)
    coordinates = torch.zeros((len(problem.regions), 2), dtype=torch.float64)
    region_ids = torch.arange(len(problem.regions))
    temperatures = model.temperature(coordinates, region_ids)
    assert temperatures.numel() == len(problem.regions)
    assert torch.isfinite(temperatures).all()
    assert torch.isfinite(model.network.temperature_offsets).all()
    assert torch.any(model.network.temperature_offsets != 0)

    cable_centers = torch.tensor(problem.case['cables'], dtype=torch.float64)
    conductor_ids = torch.tensor([region.id for region in problem.regions if region.kind == 'conductor'])
    conductor_temperatures = model.temperature(cable_centers, conductor_ids)
    assert torch.isfinite(conductor_temperatures).all()


@pytest.mark.skipif(not hasattr(torch, 'xpu') or not torch.xpu.is_available(), reason='PyTorch XPU is unavailable')
def test_deepxde_can_train_on_xpu_and_restore_default_device(tmp_path):
    pytest.importorskip('deepxde')
    case = ROOT / 'pinn_v2' / 'cases' / 'data' / 'mms_interface.json'
    default_device = torch.get_default_device()
    result = run(case, 'DeepXDE', tmp_path / 'xpu', deepxde_model='PFNN', adam_steps=2,
                 width=8, depth=2, points=SMALL_POINTS, log_every=1, device='xpu', full_audit=False)

    assert result['status'] == 'completed', result.get('error')
    assert result['device'].startswith('xpu')
    assert torch.get_default_device() == default_device