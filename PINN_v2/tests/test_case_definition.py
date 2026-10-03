import copy
import json
from pathlib import Path
import numpy as np
import pytest
from pinn_v2.cases.catalog import cases, field_k
from pinn_v2.cases.definition import CaseDefinition, SCHEMA_VERSION, load_all
from pinn_v2.cases.errors import DimensionalError, MissingDataError
from pinn_v2.cases.units import conductivity_from_resistivity, quantity
from pinn_v2.geometry.domain import FullDomain

BCS = dict(top=dict(type='robin', h=10.0, T_inf=20.0), bottom=dict(type='dirichlet', temperature=15.0),
           left=dict(type='dirichlet', temperature=15.0), right=dict(type='dirichlet', temperature=15.0))


def base():
    legacy = cases()['xlpe_single']
    data = CaseDefinition.from_legacy(legacy, 'src', 'unit-test source').to_dict()
    data['boundary_conditions'] = copy.deepcopy(BCS)
    data['data_status']['boundary_conditions'] = dict(status='supuesto', source='src', note='test boundary data')
    return data


def test_every_case_loads_with_a_hash_and_legacy_fingerprints_are_preserved():
    definitions = load_all()
    legacy = cases()
    root = Path(__file__).resolve().parents[1] / 'pinn_v2' / 'cases' / 'v2'
    catalog_ids = {CaseDefinition.load(path).id for path in root.rglob('*.json')}
    assert set(definitions) == catalog_ids
    assert 'kim_v5_summer' in definitions and 'kim_2024_thermal_original' in definitions
    assert sum(definition.physical['kind'] == 'cable' for definition in definitions.values()) == 13
    assert sum(definition.physical['kind'] != 'cable' for definition in definitions.values()) == 8
    for name, definition in definitions.items():
        assert len(definition.sha256) == 64 and definition.schema_version == SCHEMA_VERSION
        if name in legacy and legacy[name]['kind'] == 'cable':
            assert definition.domain().fingerprint == FullDomain(legacy[name]).fingerprint


def test_analytic_verification_cases_are_active_v2_definitions():
    definitions = load_all()
    verification = {name: definition for name, definition in definitions.items() if definition.physical['kind'] != 'cable'}
    assert len(verification) == 8
    assert verification['annulus'].physical['kind'] == 'annulus'
    assert all(definition.physical['kind'].startswith('mms') for name, definition in verification.items() if name != 'annulus')


def test_hash_is_stable_roundtrips_and_reacts_to_any_change():
    definition = CaseDefinition.from_dict(base())
    again = CaseDefinition.from_dict(json.loads(json.dumps(definition.to_dict())))
    assert definition.sha256 == again.sha256
    changed = base(); changed['data_status']['source']['note'] = 'different explanation'
    assert CaseDefinition.from_dict(changed).sha256 != definition.sha256
    changed = base(); changed['boundary_conditions']['top']['h'] = 11.0
    other = CaseDefinition.from_dict(changed)
    assert other.sha256 != definition.sha256 and other.domain().fingerprint != definition.domain().fingerprint


def test_integral_of_q_equals_conductor_power_for_every_cable_case():
    for definition in load_all().values():
        if definition.physical['kind'] != 'cable':
            continue
        domain = definition.domain()
        for region in domain.conductors:
            from pinn_v2.physics.skin_effect import prescribed_power
            assert domain.source_integral(region) == pytest.approx(prescribed_power(domain.case), rel=1e-9)


def test_kim_v5_summer_uses_published_data_and_the_copper_alpha():
    definition = load_all()['kim_v5_summer']
    layers = definition.physical['layers']
    assert len(layers) == 9 and layers[7][2] == 2.15 and layers[8][2] == 0.2857
    assert definition.physical['alpha'] == 0.00393 and definition.physical['current'] == 1026.0
    bc = definition.domain().side_conditions()
    assert bc['top'] == dict(type='robin', h_W_m2K=14.605, T_inf_C=27.2)
    assert all(bc[s] == dict(type='dirichlet', T_C=15.2) for s in ('bottom', 'left', 'right'))
    flagged = {item['field']: item for item in definition.inconsistencies}
    assert flagged['alpha']['values'] == dict(paper=0.0393, used=0.00393)
    assert flagged['winter.T_air']['resolution'].endswith('not invented')
    assert definition.data_status['boundary_conditions']['status'] == 'extraido'


def test_kim_thermal_original_preserves_fixed_power_profiles_and_patch():
    definition = load_all()['kim_2024_thermal_original']
    domain = definition.domain()
    conductor = domain.conductors[0]
    assert domain.source_integral(conductor) == pytest.approx(27.97)
    assert domain.specification()['thermal_source']['power_per_cable_W_m'] == pytest.approx(27.97)
    points = np.array([[45.5, 0.0], [45.5, -0.25], [45.5, -45.5]])
    assert np.allclose(domain.dirichlet_value('right', points).ravel(), [26.1, 24.89, 15.2])
    x = np.array([[0.0], [2.0]])
    y = np.array([[-1.4], [-1.4]])
    assert np.allclose(field_k(domain.case, x, y).ravel(), [2.094, 1.351], atol=2e-4)


def test_missing_or_flagged_data_are_rejected():
    for mutate, error in [
        (lambda d: d['data_status'].pop('materials'), MissingDataError),
        (lambda d: d['data_status']['source'].update(status='faltante'), MissingDataError),
        (lambda d: d['data_status']['source'].update(source='nobody'), MissingDataError),
        (lambda d: d['data_status']['source'].pop('note'), MissingDataError),
        (lambda d: d['boundary_conditions'].pop('left'), MissingDataError),
        (lambda d: d['boundary_conditions']['top'].pop('T_inf'), MissingDataError),
        (lambda d: d.pop('physical'), MissingDataError),
        (lambda d: d['provenance'][0].pop('citation'), MissingDataError),
        (lambda d: d['data_status']['source'].update(status='guess'), ValueError),
        (lambda d: d.update(schema_version='pinn-v2-case-0'), ValueError),
        (lambda d: d.update(unexpected=1), ValueError),
        (lambda d: d['physical'].update(id='other'), ValueError),
    ]:
        data = base(); mutate(data)
        with pytest.raises(error):
            CaseDefinition.from_dict(data)


def test_dimensional_validation_and_k_is_never_a_resistivity():
    assert quantity(dict(value=2.0, unit='W/(m·K)'.replace(chr(183), ' ')), 'conductivity', 'k') == 2.0
    assert quantity(dict(value=14.6, unit='W/(m2 K)'), 'heat_transfer_coefficient', 'h') == 14.6
    with pytest.raises(DimensionalError, match='resistivity'):
        quantity(dict(value=0.5, unit='K m/W'), 'conductivity', 'k')
    with pytest.raises(DimensionalError):
        quantity(dict(value=10.0, unit='W/m2'), 'heat_transfer_coefficient', 'h')
    with pytest.raises(DimensionalError):
        quantity(float('nan'), 'temperature', 'T')
    assert conductivity_from_resistivity(2.0) == 0.5
    data = base(); data['boundary_conditions']['top']['h'] = dict(value=10.0, unit='W/m2')
    with pytest.raises(DimensionalError):
        CaseDefinition.from_dict(data)


def test_boundary_condition_rules():
    for mutate in [lambda b: b['top'].update(h=0.0), lambda b: b['top'].update(h=-1.0),
                   lambda b: b.update(extra=dict(type='dirichlet', temperature=1.0)),
                   lambda b: b['left'].update(type='mixed')]:
        data = base(); mutate(data['boundary_conditions'])
        with pytest.raises(ValueError):
            CaseDefinition.from_dict(data)
    data = base()
    data['boundary_conditions'] = {s: dict(type='neumann', heat_flux_out=0.0) for s in ('bottom', 'top', 'left', 'right')}
    with pytest.raises(ValueError, match='undetermined'):
        CaseDefinition.from_dict(data)
    data = base(); data['physical']['outer_gradient_K_m'] = 1.0
    with pytest.raises(ValueError):
        CaseDefinition.from_dict(data)


def test_robin_is_kept_as_robin_and_sides_are_oriented():
    domain = load_all()['kim_v5_summer'].domain()
    assert not domain.all_dirichlet
    x0, x1, y0, y1 = domain.case['bounds']
    sides = {s['name']: s for s in domain.boundary_sides(16)}
    assert [s['tag'] for s in sides.values()] == [1001, 1002, 1003, 1004]
    for name, coord, axis, normal in [('bottom', y0, 1, (0, -1)), ('top', y1, 1, (0, 1)), ('left', x0, 0, (-1, 0)), ('right', x1, 0, (1, 0))]:
        side = sides[name]
        assert np.allclose(side['xy'][:, axis], coord) and np.allclose(side['normal'], normal)
    assert sum(s['length'] for s in sides.values()) == pytest.approx(2 * (x1 - x0) + 2 * (y1 - y0))
    assert sides['top']['bc']['type'] == 'robin'
    assert np.allclose(domain.dirichlet_value('bottom', sides['bottom']['xy']), 15.2)
    with pytest.raises(ValueError):
        domain.dirichlet_value('top', sides['top']['xy'])


def test_legacy_cases_keep_the_legacy_dirichlet_boundary():
    domain = FullDomain(cases()['xlpe_single'])
    assert domain.all_dirichlet and set(domain.side_conditions()) == {'bottom', 'top', 'left', 'right'}
    xy = np.array([[0., 0.], [1., 0.]])
    assert np.allclose(domain.dirichlet_value('top', xy), domain.case['T0'])
    assert FullDomain(cases()['xlpe_single']).fingerprint == domain.fingerprint


def test_explicit_default_dirichlet_preserves_legacy_physics_fingerprint():
    legacy = cases()['xlpe_single']
    data = CaseDefinition.from_legacy(legacy, 'src', 'unit-test source').to_dict()
    data['boundary_conditions'] = {side: dict(type='dirichlet', temperature=dict(value=legacy['T0'], unit='degC'))
                                   for side in ('bottom', 'top', 'left', 'right')}
    data['data_status']['boundary_conditions'] = dict(status='supuesto', source='src', note='Explicit legacy boundary')
    explicit = CaseDefinition.from_dict(data).domain()
    assert explicit.fingerprint == FullDomain(legacy).fingerprint


def test_imperfect_contact_requires_a_documented_resistance_and_changes_the_physics():
    data = base()
    data['contact'] = [dict(interface='cable_0_interface_1', R=dict(value=2e-3, unit='m2 K/W'), status='supuesto', source='src')]
    data['data_status']['contact'] = dict(status='supuesto', source='src', note='test value, not a measurement')
    definition = CaseDefinition.from_dict(data)
    domain, perfect = definition.domain(), CaseDefinition.from_dict(base()).domain()
    assert domain.has_imperfect_contact and domain.contact == {'cable_0_interface_1': 2e-3}
    assert domain.fingerprint != perfect.fingerprint
    assert 'perfect thermal contact' in perfect.specification()['assumptions']
    assert 'perfect thermal contact except documented interfaces' in domain.specification()['assumptions']
    for mutate in [lambda c: c[0].update(interface='nowhere'), lambda c: c[0].pop('source'), lambda c: c[0].update(R=0.0),
                   lambda c: c[0].update(R=dict(value=1.0, unit='W/(m2 K)')), lambda c: c.append(copy.deepcopy(c[0]))]:
        bad = copy.deepcopy(data); mutate(bad['contact'])
        with pytest.raises(ValueError):
            CaseDefinition.from_dict(bad)
    missing = copy.deepcopy(data); missing['data_status'].pop('contact')
    with pytest.raises(MissingDataError):
        CaseDefinition.from_dict(missing)