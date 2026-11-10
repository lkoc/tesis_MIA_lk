"""Validation of pinn-v3-problem-1 documents; every error names the field that is wrong."""
import math
import numpy as np
from pinn_v3.problem.fields import evaluate_field, validate_field
from pinn_v3.problem.shapes import SHAPES, contains, inside_bounds

SCHEMA = 'pinn-v3-problem-1'
SIDES = ('bottom', 'top', 'left', 'right')
ANNULUS_SIDES = ('inner', 'outer')
BOUNDARY_TYPES = ('dirichlet', 'neumann', 'robin')
REGION_KEYS = {'id', 'kind', 'priority', 'shape', 'conductivity', 'source', 'heat_capacity'}
COVERAGE_GRID = 201


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_shape(shape, bounds, where):
    _require(isinstance(shape, dict) and shape.get('type') in SHAPES, f'{where}: shape.type must be one of {SHAPES}')
    kind = shape['type']
    if kind == 'rectangle':
        _require(set(shape) == {'type', 'bounds'} and len(shape['bounds']) == 4, f'{where}: rectangle needs bounds only')
        x0, x1, y0, y1 = shape['bounds']
        _require(x0 < x1 and y0 < y1, f'{where}: rectangle is empty')
    elif kind == 'disk':
        _require(set(shape) == {'type', 'center', 'radius_m'}, f'{where}: disk needs center and radius_m only')
        _require(shape['radius_m'] > 0, f'{where}: disk radius must be positive')
    else:
        _require(set(shape) == {'type', 'center', 'inner_radius_m', 'outer_radius_m'}, f'{where}: annulus needs center, inner and outer radius')
        _require(0 < shape['inner_radius_m'] < shape['outer_radius_m'], f'{where}: annulus needs 0 < inner < outer')
    if kind != 'rectangle':
        _require(len(shape['center']) == 2, f'{where}: center must be [x, y]')
    _require(inside_bounds(shape, bounds), f'{where}: shape leaves the domain')


def _validate_boundary(side, bc):
    where = f'boundaries.{side}'
    _require(isinstance(bc, dict) and bc.get('type') in BOUNDARY_TYPES, f'{where}.type must be one of {BOUNDARY_TYPES}')
    if bc['type'] == 'dirichlet':
        _require(set(bc) == {'type', 'value'}, f'{where}: dirichlet needs value only')
        validate_field(bc['value'], f'{where}.value')
    elif bc['type'] == 'neumann':
        _require(set(bc) == {'type', 'q_n_W_m2'}, f'{where}: neumann needs q_n_W_m2 only')
        validate_field(bc['q_n_W_m2'], f'{where}.q_n_W_m2')
    else:
        _require(set(bc) == {'type', 'h_W_m2K', 'T_inf_C'}, f'{where}: robin needs h_W_m2K and T_inf_C only')
        _require(isinstance(bc['h_W_m2K'], (int, float)) and bc['h_W_m2K'] > 0, f'{where}: robin h must be positive')
        validate_field(bc['T_inf_C'], f'{where}.T_inf_C')


def _check_coverage_and_conductivity(doc):
    """Every domain point must have exactly one owner; each owner's conductivity must be positive."""
    domain = doc['domain']
    if domain['shape'] == 'annulus':
        angles = np.linspace(0, 2 * math.pi, COVERAGE_GRID, endpoint=False)
        radii = np.linspace(domain['inner_radius_m'], domain['outer_radius_m'], COVERAGE_GRID)
        radius, angle = np.meshgrid(radii, angles)
        x = (domain['center'][0] + radius * np.cos(angle)).ravel()
        y = (domain['center'][1] + radius * np.sin(angle)).ravel()
    else:
        x0, x1, y0, y1 = domain['bounds']
        x, y = np.meshgrid(np.linspace(x0, x1, COVERAGE_GRID), np.linspace(y0, y1, COVERAGE_GRID))
        x, y = x.ravel(), y.ravel()
    owner = np.full(x.shape, -1)
    span = max(domain['outer_radius_m'] * 2 if domain['shape'] == 'annulus'
               else max(domain['bounds'][1] - domain['bounds'][0], domain['bounds'][3] - domain['bounds'][2]), 1.0)
    tolerance = 1e-8 * span
    for index in sorted(range(len(doc['regions'])), key=lambda item: doc['regions'][item]['priority']):
        region = doc['regions'][index]
        owner[contains(region['shape'], x, y, tolerance)] = index
    _require(np.all(owner >= 0), f'{np.count_nonzero(owner < 0)} grid points of the domain belong to no region')
    for index, region in enumerate(doc['regions']):
        mine = owner == index
        values = evaluate_field(region['conductivity'], x[mine], y[mine])
        _require(np.all(np.isfinite(values)) and np.all(values > 0),
                 f"region {region['id']!r}: conductivity must be positive and finite")


def validate_problem(doc):
    """Return the validated document; raise ValueError naming the first wrong field."""
    _require(isinstance(doc, dict) and doc.get('schema') == SCHEMA, f'schema must be {SCHEMA}')
    _require(isinstance(doc.get('id'), str) and doc['id'], 'id must be a non-empty string')
    domain = doc.get('domain', {})
    _require(domain.get('shape') in ('rectangle', 'annulus'), 'domain.shape must be rectangle or annulus')
    if domain['shape'] == 'rectangle':
        _require(len(domain.get('bounds', [])) == 4, 'rectangular domain needs bounds [x0, x1, y0, y1]')
    else:
        _require(set(domain) == {'shape', 'center', 'inner_radius_m', 'outer_radius_m', 'bounds'},
                 'annular domain needs center, radii and bounding bounds')
        _require(0 < domain['inner_radius_m'] < domain['outer_radius_m'], 'annular domain radii are invalid')
    bounds = domain['bounds']
    _require(bounds[0] < bounds[1] and bounds[2] < bounds[3], 'domain bounds are empty')
    if 'temperature_scale_K' in doc:
        scale = doc['temperature_scale_K']
        _require(isinstance(scale, (int, float)) and not isinstance(scale, bool) and math.isfinite(scale) and scale > 0,
                 'temperature_scale_K must be a positive finite number')
    regions = doc.get('regions')
    _require(isinstance(regions, list) and regions, 'regions must be a non-empty list')
    ids = [region.get('id') for region in regions]
    _require(len(set(ids)) == len(ids), 'region ids must be unique')
    for region in regions:
        where = f"region {region.get('id')!r}"
        _require(set(region) <= REGION_KEYS and {'id', 'priority', 'shape', 'conductivity'} <= set(region),
             f'{where}: keys are id, priority, shape, conductivity (required), source and heat_capacity')
        _require(isinstance(region['priority'], int) and not isinstance(region['priority'], bool),
             f'{where}.priority must be an integer')
        _validate_shape(region['shape'], bounds, where)
        if 'kind' in region:
            _require(region['kind'] in ('soil', 'conductor', 'material'), f'{where}.kind is invalid')
        validate_field(region['conductivity'], f'{where}.conductivity')
        validate_field(region.get('source', 0.0), f'{where}.source')
        if 'heat_capacity' in region:
            validate_field(region['heat_capacity'], f'{where}.heat_capacity')
    _require(len({region['priority'] for region in regions}) == len(regions),
             'region priorities must be unique')
    boundaries = doc.get('boundaries')
    boundary_names = ANNULUS_SIDES if domain['shape'] == 'annulus' else SIDES
    _require(isinstance(boundaries, dict) and set(boundaries) == set(boundary_names),
             f'boundaries must declare exactly {boundary_names}')
    for side in boundary_names:
        _validate_boundary(side, boundaries[side])
    reference = doc.get('reference', {})
    _require(set(reference) <= {'exact'}, 'reference may only contain exact')
    if 'exact' in reference:
        validate_field(reference['exact'], 'reference.exact')
    _check_coverage_and_conductivity(doc)
    return doc
