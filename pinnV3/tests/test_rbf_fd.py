import numpy as np
import pytest

from pinn_v3.problem.build import build_problem, load_problem
from pinn_v3.solvers.pinn.rbf_fd import build_region_nodes, check_maximum_principle, derivative_weights, pretrain_pinn, solve_regions, solve_single_rectangle


@pytest.mark.parametrize('scale', [1., 0.001])
def test_weights_reproduce_quadratic_derivatives(scale):
    rng = np.random.default_rng(42)
    center = np.array([2., -3.])
    nodes = center + scale * rng.uniform(-1., 1., size=(21, 2))
    offsets = nodes - center
    x, y = offsets.T
    values = 2 + 3*x - 4*y + 5*x*x + 6*x*y + 7*y*y
    weights = derivative_weights(center, nodes)
    np.testing.assert_allclose(values @ weights, [3., -4., 24.], rtol=1e-8, atol=1e-7)


def test_degenerate_stencil_is_rejected():
    nodes = np.column_stack((np.arange(10), np.zeros(10)))
    with pytest.raises(ValueError, match='quadratic'):
        derivative_weights([0., 0.], nodes)


def test_invalid_temperature_violating_maximum_principle_is_rejected():
    from pathlib import Path
    problem = load_problem(Path(__file__).parents[1] / 'problems' / 'cable_disk' / 'problem.json')
    with pytest.raises(ValueError, match='lower temperature bound'):
        check_maximum_principle(problem, [20., -1000.])


def test_single_rectangle_solves_variable_k_and_mixed_boundary_conditions():
    doc = {
        'schema': 'pinn-v3-problem-1', 'id': 'rbf_quadratic',
        'domain': {'shape': 'rectangle', 'bounds': [0., 1., 0., 1.]},
        'regions': [{'id': 'material', 'priority': 0,
            'shape': {'type': 'rectangle', 'bounds': [0., 1., 0., 1.]},
            'conductivity': {'expression': '1 + x'},
            'source': {'expression': '-2 - 4*x'}}],
        'boundaries': {
            'left': {'type': 'dirichlet', 'value': 2.},
            'right': {'type': 'neumann', 'q_n_W_m2': -4.},
            'bottom': {'type': 'neumann', 'q_n_W_m2': 0.},
            'top': {'type': 'robin', 'h_W_m2K': 2.,
                'T_inf_C': {'expression': '2 + x**2'}},
        },
    }
    result = solve_single_rectangle(build_problem(doc), interior_per_axis=8,
        boundary_per_side=10, stencil_size=21)
    expected = 2 + result['xy'][:, 0]**2
    np.testing.assert_allclose(result['temperature'], expected, rtol=1e-7, atol=1e-7)
    assert result['scaled_residual_inf'] < 1e-10


def test_nodes_have_paired_interfaces_and_four_thin_annulus_layers():
    from pathlib import Path

    path = Path(__file__).parents[1] / 'problems' / 'xlpe_discrete_layers' / 'problem.json'
    problem = load_problem(path)
    nodes = build_region_nodes(problem, interior_per_region=24, surface_count=16)
    assert nodes['interfaces']
    for left, right, normal in nodes['interfaces']:
        np.testing.assert_array_equal(nodes['xy'][left], nodes['xy'][right])
        assert np.all(nodes['labels'][left] != nodes['labels'][right])
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.)
    for region_id, region in enumerate(problem.regions):
        shape = region['shape']
        if shape['type'] != 'annulus':
            continue
        points = nodes['xy'][(nodes['labels'] == region_id) & (nodes['kinds'] == 'interior')]
        radii = np.linalg.norm(points - shape['center'], axis=1)
        for layer in range(1, 5):
            expected = shape['inner_radius_m'] + (shape['outer_radius_m'] - shape['inner_radius_m']) * layer / 5
            assert np.count_nonzero(np.isclose(radii, expected, rtol=0., atol=1e-10)) >= 16


def test_initializer_reuses_supplied_pinn_collocation_points():
    from pathlib import Path

    problem = load_problem(Path(__file__).parents[1] / 'problems' / 'cable_disk' / 'problem.json')
    rng = np.random.default_rng(42)
    selected = {region_id: problem.sample_region_interior(region_id, 12, rng)
                for region_id in range(len(problem.regions))}
    nodes = build_region_nodes(problem, interior_per_region=12, surface_count=12,
                               interior_points=selected)
    offset = 0
    for region_id in range(len(problem.regions)):
        np.testing.assert_array_equal(nodes['xy'][offset:offset + 12], selected[region_id])
        offset += 12


@pytest.mark.parametrize('architecture', ['M1', 'M1_EXACT', 'CUSPR', 'M0R', 'M2'])
def test_pretraining_supports_campaign_pinn_models(architecture):
    from pathlib import Path
    import torch
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.networks import build_model

    problem = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json'))
    model = build_model(problem, architecture, width=8, depth=2).double()
    xy = problem.sample_interior(problem.regions[0], 24, 42)
    result = dict(xy=xy, labels=np.zeros(len(xy), dtype=int), anchors={0: np.arange(len(xy))},
                  temperature=20. + xy[:, 0])
    before = [parameter.detach().clone() for parameter in model.parameters()]
    report = pretrain_pinn(model, result, steps=2)
    assert report['points'] == 24
    assert np.isfinite(report['final_loss'])
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


@pytest.mark.parametrize('mode', ['extrapolate', 'incremental'])
@pytest.mark.parametrize('fraction', [0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
def test_optional_initializer_runs_inside_native_trainer(tmp_path, mode, fraction):
    from pathlib import Path
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.trainer import run

    problem = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json'))
    settings = dict(fraction=fraction, mode=mode, pretrain_steps=2, surface_count=12, growth_stages=10)
    result = run(problem, 'M1', tmp_path / mode, adam_steps=10, lbfgs_steps=0, width=8, depth=2,
        points=dict(n_interior=40, n_layer=40, n_face=8, n_boundary=8),
        device='cpu', full_audit=False, rbf_initialization=settings)
    assert result['status'] == 'completed', result.get('error')
    assert result['rbf_initialization']['anchor_count'] == int(np.ceil(40 * fraction))
    assert result['rbf_initialization']['mode'] == mode
    assert result['config']['rbf_initialization']['fraction'] == fraction
    assert result['config']['adam_steps'] == 10
    assert (tmp_path / mode / 'rbf_initialization.npz').exists()


def test_incremental_initializer_can_resume_from_checkpoint(tmp_path):
    from pathlib import Path
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.trainer import resume_training, run

    problem = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json'))
    settings = dict(fraction=0.1, mode='incremental', pretrain_steps=2, surface_count=12, growth_stages=10)
    result = run(problem, 'M1', tmp_path / 'interrupted', adam_steps=10, lbfgs_steps=0, width=8, depth=2,
        points=dict(n_interior=40, n_layer=40, n_face=8, n_boundary=8),
        device='cpu', full_audit=False, rbf_initialization=settings, log_every=1,
        stop_callback=lambda row, model: row['step'] == 1)
    assert result['status'] == 'completed', result.get('error')
    resumed = resume_training(tmp_path / 'interrupted' / 'model.pt', tmp_path / 'resumed',
        device='cpu', full_audit=False)
    assert resumed['status'] == 'completed', resumed.get('error')
    assert resumed['config']['rbf_initialization'] == result['config']['rbf_initialization']
    assert resumed['history'][-1]['step'] == 9


def test_finished_initializer_training_can_continue_with_new_budget(tmp_path):
    from pathlib import Path
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    from pinn_v3.solvers.pinn.trainer import continue_training, run

    problem = GenericProblem(load_problem(Path(__file__).parents[1] / 'problems' / 'mms_constant' / 'problem.json'))
    settings = dict(fraction=0.1, mode='incremental', pretrain_steps=2, surface_count=12, growth_stages=10)
    result = run(problem, 'M1', tmp_path / 'initial', adam_steps=10, lbfgs_steps=0, width=8, depth=2,
        points=dict(n_interior=40, n_layer=40, n_face=8, n_boundary=8),
        device='cpu', full_audit=False, rbf_initialization=settings)
    assert result['status'] == 'completed', result.get('error')
    continued = continue_training(tmp_path / 'initial' / 'model.pt', tmp_path / 'continued',
        adam_steps=2, device='cpu', full_audit=False)
    assert continued['status'] == 'completed', continued.get('error')
    assert continued['config']['initialization_policy'] == 'weights_only_warm_start'
    assert 'rbf_initialization' not in continued['config']
    assert continued['history'][-1]['step'] == 1


def test_two_material_interface_conserves_flux_and_temperature():
    doc = {
        'schema': 'pinn-v3-problem-1', 'id': 'rbf_two_materials',
        'domain': {'shape': 'rectangle', 'bounds': [0., 1., 0., 1.]},
        'regions': [
            {'id': 'left', 'priority': 0, 'shape': {'type': 'rectangle', 'bounds': [0., 1., 0., 1.]},
             'conductivity': 1.},
            {'id': 'right', 'priority': 1, 'shape': {'type': 'rectangle', 'bounds': [0.5, 1., 0., 1.]},
             'conductivity': 2.},
        ],
        'boundaries': {
            'left': {'type': 'dirichlet', 'value': 0.},
            'right': {'type': 'dirichlet', 'value': 1.},
            'bottom': {'type': 'neumann', 'q_n_W_m2': 0.},
            'top': {'type': 'neumann', 'q_n_W_m2': 0.},
        },
    }
    result = solve_regions(build_problem(doc), interior_per_region=70, surface_count=14, seed=5)
    x = result['xy'][:, 0]
    expected = np.where(result['labels'] == 0, 4*x/3, 2*x/3 + 1/3)
    np.testing.assert_allclose(result['temperature'], expected, rtol=1e-7, atol=1e-7)
    assert result['scaled_residual_inf'] < 1e-10
    for left, right, _ in result['interfaces']:
        np.testing.assert_allclose(result['temperature'][left], result['temperature'][right], atol=1e-10)


def test_concentric_interface_matches_radial_analytical_solution():
    doc = {
        'schema': 'pinn-v3-problem-1', 'id': 'rbf_concentric',
        'domain': {'shape': 'annulus', 'center': [0., 0.], 'inner_radius_m': 0.1,
                   'outer_radius_m': 1., 'bounds': [-1., 1., -1., 1.]},
        'regions': [
            {'id': 'inner', 'priority': 0,
             'shape': {'type': 'annulus', 'center': [0., 0.], 'inner_radius_m': 0.1, 'outer_radius_m': 0.5},
             'conductivity': 1.},
            {'id': 'outer', 'priority': 1,
             'shape': {'type': 'annulus', 'center': [0., 0.], 'inner_radius_m': 0.5, 'outer_radius_m': 1.},
             'conductivity': 20.},
        ],
        'boundaries': {'inner': {'type': 'dirichlet', 'value': 1.},
                       'outer': {'type': 'dirichlet', 'value': 0.}},
    }
    result = solve_regions(build_problem(doc), interior_per_region=300, surface_count=48, seed=9)
    radius = np.linalg.norm(result['xy'], axis=1)
    resistance = np.log(5.) + np.log(2.) / 20.
    drop = np.where(result['labels'] == 0, np.log(radius / 0.1),
                    np.log(5.) + np.log(radius / 0.5) / 20.)
    expected = 1. - drop / resistance
    assert np.max(np.abs(result['temperature'] - expected)) < 0.03
    assert result['scaled_residual_inf'] < 1e-10


def test_thin_annulus_and_high_contrast_match_radial_solution():
    radii = [0.1, 0.12, 0.121, 1.]
    conductivities = [380., 0.29, 1.]
    regions = []
    for index, conductivity in enumerate(conductivities):
        regions.append({'id': f'layer_{index}', 'priority': index,
            'shape': {'type': 'annulus', 'center': [0., 0.],
                      'inner_radius_m': radii[index], 'outer_radius_m': radii[index + 1]},
            'conductivity': conductivity})
    doc = {'schema': 'pinn-v3-problem-1', 'id': 'rbf_thin_annulus',
        'domain': {'shape': 'annulus', 'center': [0., 0.], 'inner_radius_m': radii[0],
                   'outer_radius_m': radii[-1], 'bounds': [-1., 1., -1., 1.]},
        'regions': regions,
        'boundaries': {'inner': {'type': 'dirichlet', 'value': 1.},
                       'outer': {'type': 'dirichlet', 'value': 0.}}}
    result = solve_regions(build_problem(doc), interior_per_region=200, surface_count=48, seed=9)
    resistances = np.log(np.array(radii[1:]) / np.array(radii[:-1])) / conductivities
    radius = np.linalg.norm(result['xy'], axis=1)
    region_ids = result['labels']
    accumulated = np.r_[0., np.cumsum(resistances)]
    drop = accumulated[region_ids] + np.log(radius / np.array(radii)[region_ids]) / np.array(conductivities)[region_ids]
    expected = 1. - drop / sum(resistances)
    assert np.max(np.abs(result['temperature'] - expected)) < 0.05
    assert result['scaled_residual_inf'] < 1e-9