"""Region problem format: valid example, field and ownership semantics, and rejection of invalid data."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pinn_v3.problem.build import build_problem, load_problem
from pinn_v3.problem.shapes import boundary_distance

EXAMPLE = Path(__file__).resolve().parents[1] / 'problems' / 'two_layer_slab' / 'problem.json'


def example():
    return json.loads(EXAMPLE.read_text(encoding='utf-8'))


def test_example_builds_and_regions_are_ordered():
    problem = load_problem(EXAMPLE)
    assert [region['id'] for region in problem.regions] == ['soil', 'slab', 'conductor']


def test_later_region_overrides_earlier_one():
    problem = load_problem(EXAMPLE)
    owner = problem.owner(np.array([0.5, 0.5, 1.0]), np.array([0.1, 0.5, 0.8]))
    assert owner.tolist() == [0, 1, 2]


def test_fields_come_from_the_owning_region():
    problem = load_problem(EXAMPLE)
    k = problem.field('conductivity', np.array([0.5, 0.5]), np.array([0.1, 0.5]))
    assert k[0] == pytest.approx(1.0)
    assert k[1] == pytest.approx(2.0 + np.sin(np.pi * 0.5))
    q = problem.field('source', np.array([1.0, 0.5]), np.array([0.8, 0.5]))
    assert q.tolist() == [10000.0, 0.0]


def test_table_interpolates_linearly_along_its_axis():
    doc = example()
    doc['regions'][0]['conductivity'] = {'table': {'axis': 'y', 'coordinates': [0.0, 1.0], 'values': [1.0, 3.0]}}
    problem = build_problem(doc)
    # y = 0.1 lies outside the slab, so the soil table is the owner's law there.
    assert problem.field('conductivity', np.array([0.2]), np.array([0.1]))[0] == pytest.approx(1.2)


def test_sampling_is_labelled_by_owner():
    problem = load_problem(EXAMPLE)
    xy, owner = problem.sample_interior(40, seed=3)
    assert np.bincount(owner).tolist() == [40, 40, 40]
    assert np.array_equal(problem.owner(xy[:, 0], xy[:, 1]), owner)
    boundary = problem.sample_boundary(16)
    assert set(boundary) == {'bottom', 'top', 'left', 'right'}
    assert all(np.all(side['owner'] >= 0) for side in boundary.values())


def test_interior_sampling_rejects_region_boundaries_and_tolerates_roundoff():
    problem = load_problem(EXAMPLE)
    xy, owner = problem.sample_interior(80, seed=19)
    for region_id, region in enumerate(problem.regions):
        selected = owner == region_id
        shape = region['shape']
        assert np.all(boundary_distance(shape, xy[selected, 0], xy[selected, 1]) > 1e-7)
    near_edge = np.array([[-1e-10, 0.5], [2.0 + 1e-10, 0.5]])
    assert np.all(problem.owner(near_edge[:, 0], near_edge[:, 1]) >= 0)


def test_local_coordinates_sample_thin_region_independently_of_domain_scale():
    doc = {
        'schema': 'pinn-v3-problem-1',
        'id': 'thin_region_sampling',
        'domain': {'shape': 'rectangle', 'bounds': [-45.5, 45.5, -45.5, 0.0]},
        'temperature_scale_K': 20.0,
        'regions': [
            {'id': 'background', 'priority': 0,
             'shape': {'type': 'rectangle', 'bounds': [-45.5, 45.5, -45.5, 0.0]},
             'conductivity': 1.0},
            {'id': 'thin_strip', 'priority': 1,
             'shape': {'type': 'rectangle', 'bounds': [0.0, 0.002, -1.7, -1.5]},
             'conductivity': 2.0},
        ],
        'boundaries': {side: {'type': 'dirichlet', 'value': 20.0}
                       for side in ('bottom', 'top', 'left', 'right')},
    }
    problem = build_problem(doc)

    xy, owner = problem.sample_interior(256, seed=29)
    assert np.bincount(owner).tolist() == [256, 256]
    local = problem.domain_to_local(1, xy[owner == 1])
    np.testing.assert_allclose(problem.local_to_domain(1, local), xy[owner == 1], atol=1e-12)
    assert np.all((local > 0) & (local < 1))

    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    pinn_problem = GenericProblem(doc)
    np.testing.assert_array_equal(
        pinn_problem.sample_interior(pinn_problem.regions[1], 256, seed=29),
        problem.sample_region_interior(1, 256, np.random.default_rng(29)))


def test_boundary_sampling_preserves_a_thin_region_segment():
    doc = {
        'schema': 'pinn-v3-problem-1',
        'id': 'thin_boundary_sampling',
        'domain': {'shape': 'rectangle', 'bounds': [-45.5, 45.5, -45.5, 0.0]},
        'temperature_scale_K': 20.0,
        'regions': [
            {'id': 'background', 'priority': 0,
             'shape': {'type': 'rectangle', 'bounds': [-45.5, 45.5, -45.5, 0.0]},
             'conductivity': 1.0},
            {'id': 'thin_strip', 'priority': 1,
             'shape': {'type': 'rectangle', 'bounds': [0.0, 0.002, -45.5, -45.498]},
             'conductivity': 2.0},
        ],
        'boundaries': {side: {'type': 'dirichlet', 'value': 20.0}
                       for side in ('bottom', 'top', 'left', 'right')},
    }
    problem = build_problem(doc)

    local_segments = [piece for piece in problem.sample_boundary_regions(16)
                      if piece['side'] == 'bottom' and piece['region'] == 1]
    assert len(local_segments) == 1
    assert local_segments[0]['xy'].shape == (32, 2)  # two rows: the boundary and one row inside the region
    assert local_segments[0]['length'] == pytest.approx(0.002)
    assert np.all(problem.owner(local_segments[0]['xy'][:, 0], local_segments[0]['xy'][:, 1]) == 1)


def mutate(change):
    doc = example()
    change(doc)
    return doc


CASES = {
    'missing side': lambda d: d['boundaries'].pop('top'),
    'unknown boundary type': lambda d: d['boundaries']['top'].update(type='mixed'),
    'robin with h = 0': lambda d: d['boundaries']['top'].update(h_W_m2K=0.0),
    'uncovered domain': lambda d: d['regions'].pop(0),
    'disk leaves the domain': lambda d: d['regions'][2]['shape'].update(center=[1.0, 0.95]),
    'table not increasing': lambda d: d['regions'][0].update(conductivity={'table': {'axis': 'y', 'coordinates': [1.0, 0.0], 'values': [1.0, 2.0]}}),
    'unknown region key': lambda d: d['regions'][0].update(color='red'),
    'invalid expression': lambda d: d['regions'][1].update(conductivity={'expression': 'sin(x'}),
    'non-positive conductivity': lambda d: d['regions'][0].update(conductivity=-1.0),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_invalid_documents_are_rejected(name):
    with pytest.raises(ValueError):
        build_problem(mutate(CASES[name]))


def test_non_positive_temperature_scale_is_rejected():
    with pytest.raises(ValueError):
        build_problem(mutate(lambda d: d.update(temperature_scale_K=0.0)))


def test_declared_temperature_scale_reaches_the_pinn_adapter():
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    doc = example()
    doc['temperature_scale_K'] = 80.0
    assert GenericProblem(doc).T_scale == pytest.approx(80.0)


def scaled_document(doc, factor):
    """Every length multiplied by factor; conductivities and temperatures unchanged."""
    out = copy.deepcopy(doc)
    out['domain']['bounds'] = [v * factor for v in out['domain']['bounds']]
    for region in out['regions']:
        shape = region['shape']
        if shape['type'] == 'rectangle':
            shape['bounds'] = [v * factor for v in shape['bounds']]
        elif shape['type'] == 'disk':
            shape['center'] = [c * factor for c in shape['center']]
            shape['radius_m'] *= factor
        else:
            shape['center'] = [c * factor for c in shape['center']]
            shape['inner_radius_m'] *= factor
            shape['outer_radius_m'] *= factor
    return out


def test_scales_follow_length_dimensions():
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    doc = example()
    doc['temperature_scale_K'] = 80.0
    doc['regions'][1]['conductivity'] = 2.0
    base, big = GenericProblem(doc), GenericProblem(scaled_document(doc, 2.0))
    for region in base.regions:
        a, b = base.scales(region), big.scales(region)
        assert b['temperature'] == pytest.approx(a['temperature'])
        if base.source_abs[region.id] > 0:
            # Sourced region: local drop q L^2 / k grows as L^2, so flux grows as L and divergence stays q.
            assert b['flux'] == pytest.approx(2 * a['flux'], rel=2e-2)
            assert b['divergence'] == pytest.approx(a['divergence'], rel=2e-2)
        else:
            # Soil keeps the global temperature scale: flux ~ 1/L and divergence ~ 1/L^2.
            assert b['flux'] == pytest.approx(a['flux'] / 2, rel=2e-2)
            assert b['divergence'] == pytest.approx(a['divergence'] / 4, rel=2e-2)


def test_scales_are_linear_in_conductivity():
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    doc = example()
    doc['temperature_scale_K'] = 80.0
    doc['regions'][1]['conductivity'] = 2.0
    tripled = copy.deepcopy(doc)
    for region in tripled['regions']:
        region['conductivity'] = 3.0 * region['conductivity'] if isinstance(region['conductivity'], float) else region['conductivity']
    tripled['regions'][1]['conductivity'] = 6.0
    base, scaled = GenericProblem(doc), GenericProblem(tripled)
    for region in base.regions:
        a, b = base.scales(region), scaled.scales(region)
        if base.source_abs[region.id] > 0:
            # Sourced region: the local drop scales as 1/k, so flux and divergence do not change.
            assert b['flux'] == pytest.approx(a['flux'], rel=1e-9)
            assert b['divergence'] == pytest.approx(a['divergence'], rel=1e-9)
        else:
            assert b['flux'] == pytest.approx(3 * a['flux'], rel=1e-9)
            assert b['divergence'] == pytest.approx(3 * a['divergence'], rel=1e-9)


def test_conductor_divergence_scale_equals_its_source():
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem
    doc = example()
    doc['temperature_scale_K'] = 80.0
    doc['regions'][1]['conductivity'] = 2.0
    conductor = GenericProblem(doc).regions[2]
    assert GenericProblem(doc).scales(conductor)['divergence'] == pytest.approx(10000.0, rel=1e-9)
