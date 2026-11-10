from pathlib import Path

import numpy as np

from pinn_v3.problem.build import load_problem
from pinn_v3.solvers.pinn.fem_initializer import fem_gradients, points_result, readout_transfer


def test_linear_fem_field_is_reproduced_at_collocation_points_inside_region():
    problem = load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json')
    x0, x1, y0, y1 = problem.bounds
    grid_x, grid_y = np.meshgrid(np.linspace(x0, x1, 21), np.linspace(y0, y1, 21))
    dof_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    dof_temperature = 2 * dof_xy[:, 0] + 3 * dof_xy[:, 1] + 1
    rng = np.random.default_rng(3)
    collocation = {0: np.column_stack((rng.uniform(x0 + 0.05, x1 - 0.05, 40), rng.uniform(y0 + 0.05, y1 - 0.05, 40)))}
    result = points_result(problem, dof_xy, dof_temperature, collocation)
    np.testing.assert_allclose(result['xy'], collocation[0])
    np.testing.assert_allclose(result['temperature'], 2 * result['xy'][:, 0] + 3 * result['xy'][:, 1] + 1, atol=1e-9)
    assert result['anchors'][0].tolist() == list(range(40))
    assert set(result['labels'].tolist()) == {0}


def test_fem_reference_cache_reuses_only_matching_problem_and_multiplier(monkeypatch, tmp_path):
    from pinn_v3.solvers.fem import adapter
    from pinn_v3.solvers.pinn import fem_initializer

    definition = load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json')
    fem_initializer._FEM_REFERENCE_CACHE.clear()
    calls = []

    def fake_run(source, output, levels, node_multipliers):
        calls.append((source['id'], tuple(levels), tuple(node_multipliers)))
        output.mkdir(parents=True, exist_ok=True)
        stem = f"fem_l0{fem_initializer.fem_token(node_multipliers[0])}"
        np.savez(output / f'{stem}.npz', dof_xy=np.array([[0., 0.], [1., 1.]]), dof_T=np.array([20., 21.]))
        return [dict(id=source['id'], ndofs=2)]

    monkeypatch.setattr(adapter, 'run_problem_fem', fake_run)
    first = fem_initializer.run_fem_reference(definition, tmp_path / 'first', 1.0)
    second = fem_initializer.run_fem_reference(definition, tmp_path / 'second', 1.0)
    third = fem_initializer.run_fem_reference(definition, tmp_path / 'third', 0.5)

    assert len(calls) == 2
    assert not first['cache_hit']
    assert second['cache_hit'] and second['wall_s'] == 0.
    assert not third['cache_hit']
    np.testing.assert_array_equal(second['dof_xy'], first['dof_xy'])
    np.testing.assert_array_equal(second['dof_T'], first['dof_T'])


def test_surface_points_are_assigned_to_both_interface_regions_and_boundary_owner():
    from pinn_v3.solvers.pinn.fem_initializer import _region_surface_points
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model
    from pinn_v3.solvers.pinn.objective import Objective

    problem = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'cable_disk' / 'problem.json'))
    objective = Objective(build_model(problem, 'M1', 4, 1, 'tanh'), n_interior=4, n_layer=4, n_face=6, n_boundary=6)
    for region in problem.regions:
        actual = _region_surface_points(objective, region.id)
        expected = []
        for face, xy, _normal, xy_right in objective.faces:
            if region.id == face.left:
                expected.append(xy)
            if region.id == face.right:
                expected.append(xy_right)
        expected += [xy for piece, xy, _ in objective.pieces if piece.region == region.id]
        assert len(actual) == len(expected)
        for points, tensor in zip(actual, expected):
            np.testing.assert_array_equal(points, tensor.detach().cpu().numpy())
    assert objective.faces
    assert all(len(_region_surface_points(objective, region.id)) for region in problem.regions)


def test_shared_interface_nodes_are_available_on_both_sides_without_neighbor_interior():
    from pinn_v3.solvers.pinn.fem_initializer import _region_node_masks

    problem = load_problem(Path(__file__).parents[1] / 'problems' / 'cable_disk' / 'problem.json')
    points = np.array([[0.1, 0.0], [0.05, 0.0], [0.2, 0.0]])
    masks = _region_node_masks(problem, points, include_shared=True)
    assert masks[0].tolist() == [True, False, True]
    assert masks[1].tolist() == [True, True, False]


def test_fem_initialization_includes_surfaces_for_exact_and_other_models(monkeypatch, tmp_path):
    import torch
    from pinn_v3.solvers.pinn import fem_initializer
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model
    from pinn_v3.solvers.pinn.objective import Objective

    generic = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'cable_disk' / 'problem.json'))
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41))
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    temperature = 20 + points[:, 0] + 2 * points[:, 1]
    monkeypatch.setattr(fem_initializer, 'run_fem_reference', lambda *args: dict(
        dof_xy=points, dof_T=temperature, record=dict(ndofs=len(points)), wall_s=0.))
    settings = dict(fem_node_multiplier=1., fem_transfer='readout_gls', fem_gradient_weight=1.,
                    fem_ridge=1e-2, mode='extrapolate', fem_surfaces=True)
    for architecture in ('M1_EXACT', 'M1', 'M0', 'M2', 'CUSP'):
        torch.manual_seed(4)
        model = build_model(generic, architecture, 6, 2, 'tanh')
        objective = Objective(model, n_interior=12, n_layer=12, n_face=4, n_boundary=4)
        report = fem_initializer.fem_initialization(model, generic, objective, settings, tmp_path)
        assert report['steps'] == 0
        assert report['fem_surfaces'] is True
        assert report['surface_anchor_count'] > 0
        assert report['anchor_count'] > sum(len(points) for points in objective.interior.values())
        assert np.isfinite(report['surface_rms_K'])
    settings['fem_surfaces'] = False
    report = fem_initializer.fem_initialization(model, generic, objective, settings, tmp_path)
    assert report['surface_anchor_count'] == 0
    assert report['anchor_count'] == sum(len(points) for points in objective.interior.values())


def test_readout_transfer_recovers_values_representable_by_output_layer():
    import torch
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model
    torch.manual_seed(0)
    generic = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json'))
    model = build_model(generic, 'M1', 12, 2, 'tanh')
    region = generic.regions[0]
    xy = generic.definition.sample_region_interior(0, 60, np.random.default_rng(5))
    with torch.no_grad():
        network = model.networks[0]
        network[-1].weight.add_(torch.randn_like(network[-1].weight))
        network[-1].bias.add_(torch.randn_like(network[-1].bias))
        target = model.field(region, torch.as_tensor(xy, dtype=torch.float32))[0].numpy().reshape(-1)
    result = dict(xy=xy, labels=np.zeros(60, dtype=int), temperature=target, anchors={0: np.arange(60)})
    assert readout_transfer(model, result)['rms_K'] < 1e-3


def test_fem_gradients_are_exact_for_quadratic_field():
    problem = load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json')
    x0, x1, y0, y1 = problem.bounds
    rng = np.random.default_rng(7)
    dof_xy = np.column_stack((rng.uniform(x0, x1, 400), rng.uniform(y0, y1, 400)))
    x, y = dof_xy.T
    dof_temperature = 1 + 2 * x - 3 * y + x ** 2 + x * y
    points = np.column_stack((rng.uniform(x0 + 0.1, x1 - 0.1, 20), rng.uniform(y0 + 0.1, y1 - 0.1, 20)))
    gradient = fem_gradients(problem, dof_xy, dof_temperature, {0: points})[0]
    expected = np.column_stack((2 + 2 * points[:, 0] + points[:, 1], -3 + points[:, 0]))
    np.testing.assert_allclose(gradient, expected, atol=1e-8)


def test_readout_with_gradients_recovers_values_and_gradients_of_representable_field():
    import torch
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model
    torch.manual_seed(0)
    generic = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json'))
    model = build_model(generic, 'M1', 12, 2, 'tanh')
    region = generic.regions[0]
    xy = generic.definition.sample_region_interior(0, 60, np.random.default_rng(5))
    with torch.no_grad():
        network = model.networks[0]
        network[-1].weight.add_(torch.randn_like(network[-1].weight))
        network[-1].bias.add_(torch.randn_like(network[-1].bias))
    points = torch.as_tensor(xy, dtype=torch.float32).requires_grad_(True)
    temperature = model.field(region, points)[0]
    gradient = torch.autograd.grad(temperature.sum(), points)[0].numpy()
    result = dict(xy=xy, labels=np.zeros(60, dtype=int), temperature=temperature.detach().numpy().reshape(-1),
                  anchors={0: np.arange(60)})
    report = readout_transfer(model, result, gradients={0: gradient})
    assert report['rms_K'] < 1e-3
    assert report['gradient_rms_K'] < 1e-3


def test_ridge_shrinks_output_weights_of_the_readout():
    import torch
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model
    torch.manual_seed(0)
    generic = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json'))
    model = build_model(generic, 'M1', 12, 2, 'tanh')
    region = generic.regions[0]
    xy = generic.definition.sample_region_interior(0, 60, np.random.default_rng(5))
    with torch.no_grad():
        network = model.networks[0]
        network[-1].weight.add_(torch.randn_like(network[-1].weight))
        network[-1].bias.add_(torch.randn_like(network[-1].bias))
    target = model.field(region, torch.as_tensor(xy, dtype=torch.float32))[0].detach().numpy().reshape(-1)
    result = dict(xy=xy, labels=np.zeros(60, dtype=int), temperature=target, anchors={0: np.arange(60)})
    free = readout_transfer(model, result)
    shrunk = readout_transfer(model, result, ridge=1e3)
    assert shrunk['weight_norm'] < free['weight_norm']
    assert shrunk['rms_K'] >= free['rms_K']


def _multi_region_result(model, generic, seed=5, count=40):
    import torch
    rng = np.random.default_rng(seed)
    xy_blocks, value_blocks, label_blocks, anchors, cursor = [], [], [], {}, 0
    for region in generic.regions:
        xy = generic.definition.sample_region_interior(region.id, count, rng)
        with torch.no_grad():
            value = model.field(region, torch.as_tensor(xy, dtype=torch.float32))[0].numpy().reshape(-1)
        xy_blocks.append(xy)
        value_blocks.append(value)
        label_blocks.append(np.full(count, region.id))
        anchors[region.id] = np.arange(cursor, cursor + count)
        cursor += count
    return dict(xy=np.vstack(xy_blocks), labels=np.concatenate(label_blocks),
                temperature=np.concatenate(value_blocks), anchors=anchors)


def test_readout_recovers_representable_field_for_global_mixed_and_cusp_models():
    import torch
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model
    generic = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'cable_disk' / 'problem.json'))
    for architecture in ('M0', 'M2', 'CUSP'):
        torch.manual_seed(0)
        model = build_model(generic, architecture, 12, 2, 'tanh')
        with torch.no_grad():
            for network in model.networks:
                network[-1].weight[0].add_(torch.randn_like(network[-1].weight[0]))
                network[-1].bias[0].add_(torch.randn_like(network[-1].bias[0]))
        result = _multi_region_result(model, generic)
        assert readout_transfer(model, result)['rms_K'] < 1e-3, architecture


def test_readout_recovers_representable_field_through_exact_mms_boundary_wrapper():
    import torch
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model

    torch.manual_seed(3)
    generic = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'mms_sine' / 'problem.json'))
    model = build_model(generic, 'M1_EXACT', 8, 2, 'tanh')
    region = generic.regions[0]
    xy = generic.definition.sample_region_interior(0, 80, np.random.default_rng(12))
    with torch.no_grad():
        network = model.base_model.networks[0]
        network[-1].weight.add_(torch.randn_like(network[-1].weight))
        network[-1].bias.add_(torch.randn_like(network[-1].bias))
        target = model.field(region, torch.as_tensor(xy, dtype=torch.float32))[0].numpy().reshape(-1)
    result = dict(xy=xy, labels=np.zeros(len(xy), dtype=int), temperature=target,
                  anchors={0: np.arange(len(xy))})

    report = readout_transfer(model, result)

    assert report['rms_K'] < 1e-3


def test_exact_readout_gradient_rms_matches_actual_field_gradient():
    import torch
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model

    torch.manual_seed(7)
    generic = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'cable_disk' / 'problem.json'))
    model = build_model(generic, 'M1_EXACT', 8, 2, 'tanh', input_normalization='unit_box_v1')
    result = _multi_region_result(model, generic, count=20)
    gradients = {}
    for region in generic.regions:
        points = torch.tensor(result['xy'][result['anchors'][region.id]], dtype=torch.float32, requires_grad=True)
        temperature = model.field(region, points)[0]
        gradients[region.id] = torch.autograd.grad(temperature.sum(), points)[0].detach().numpy()
    report = readout_transfer(model, result, gradients=gradients, ridge=1e-3)
    residuals = []
    for region in generic.regions:
        points = torch.tensor(result['xy'][result['anchors'][region.id]], dtype=torch.float32, requires_grad=True)
        temperature = model.field(region, points)[0]
        actual = torch.autograd.grad(temperature.sum(), points)[0].detach().numpy()
        residuals.append(np.sqrt(generic.areas[region.id]) * (actual - gradients[region.id]).T.reshape(-1))
    actual_rms = np.sqrt(np.mean(np.concatenate(residuals) ** 2))
    np.testing.assert_allclose(report['gradient_rms_K'], actual_rms, rtol=2e-2, atol=1e-3)


def test_exact_robin_readout_handles_multiple_output_layer_evaluations():
    import torch
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model

    torch.manual_seed(13)
    generic = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'mms_robin' / 'problem.json'))
    model = build_model(generic, 'M1_EXACT', 8, 2, 'tanh', input_normalization='unit_box_v1')
    region = generic.regions[0]
    xy = generic.definition.sample_region_interior(0, 60, np.random.default_rng(23))
    network = model.base_model.networks[0]
    with torch.no_grad():
        network[-1].weight.add_(torch.randn_like(network[-1].weight))
        network[-1].bias.add_(torch.randn_like(network[-1].bias))

    points = torch.as_tensor(xy, dtype=torch.float32).requires_grad_(True)
    temperature = model.field(region, points)[0]
    gradient = torch.autograd.grad(temperature.sum(), points)[0].detach().numpy()
    result = dict(xy=xy, labels=np.zeros(len(xy), dtype=int),
                  temperature=temperature.detach().numpy().reshape(-1),
                  anchors={0: np.arange(len(xy))})

    report = readout_transfer(model, result, gradients={0: gradient})

    assert report['rms_K'] < 1e-3
    assert report['gradient_rms_K'] < 1e-3


def test_local_quadratic_fit_recovers_values_of_quadratic_field():
    from pinn_v3.solvers.pinn.fem_initializer import local_quadratic_fit
    problem = load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json')
    x0, x1, y0, y1 = problem.bounds
    rng = np.random.default_rng(9)
    node_xy = np.column_stack((rng.uniform(x0, x1, 400), rng.uniform(y0, y1, 400)))
    x, y = node_xy.T
    node_temperature = 1 + 2 * x - 3 * y + x ** 2 + x * y
    points = np.column_stack((rng.uniform(x0 + 0.1, x1 - 0.1, 20), rng.uniform(y0 + 0.1, y1 - 0.1, 20)))
    values, _ = local_quadratic_fit(problem, node_xy, node_temperature, np.zeros(400, dtype=int), {0: points})[0]
    px, py = points.T
    np.testing.assert_allclose(values, 1 + 2 * px - 3 * py + px ** 2 + px * py, atol=1e-8)
