"""Outer boundary conditions per side and imperfect-contact data (numpy-free of torch).

Conventions: n is the outward normal, q = -k grad T, and the outflow q.n is positive
when heat leaves the domain.
  dirichlet: T = T_C
  neumann:   q.n = q_n
  robin:     q.n = h (T - T_inf)      (never converted to a Dirichlet condition)
Rectangular sides are bottom, top, left, right (tags 1001-1004); a circular outer
boundary is the single side "outer" (tag 1000).
"""
import math
from pinn_v2.cases.errors import MissingDataError
from pinn_v2.cases.units import quantity

RECT_SIDES = ('bottom', 'top', 'left', 'right')
SIDE_TAGS = {'outer': 1000, 'bottom': 1001, 'top': 1002, 'left': 1003, 'right': 1004}
SIDE_NORMALS = {'bottom': (0., -1.), 'top': (0., 1.), 'left': (-1., 0.), 'right': (1., 0.)}
_FIELDS = {'dirichlet': {'type', 'temperature'}, 'dirichlet_profile': {'type', 'depth_m', 'temperature_C'},
           'neumann': {'type', 'heat_flux_out'}, 'robin': {'type', 'h', 'T_inf'}}


def _parse_conditions(spec, sides):
    if not isinstance(spec, dict):
        raise ValueError('boundary_conditions must map each side to a condition')
    missing = [s for s in sides if s not in spec]
    if missing:
        raise MissingDataError(f'Boundary condition missing for sides {missing}')
    if set(spec) - set(sides):
        raise ValueError(f'Unknown boundary sides: {sorted(set(spec) - set(sides))}')
    out = {}
    for side in sides:
        item = spec[side]
        kind = item.get('type') if isinstance(item, dict) else None
        if kind not in _FIELDS:
            raise ValueError(f'{side}: type must be dirichlet, neumann or robin')
        if set(item) - _FIELDS[kind]:
            raise ValueError(f'{side}: unknown fields {sorted(set(item) - _FIELDS[kind])}')
        absent = _FIELDS[kind] - set(item)
        if absent:
            raise MissingDataError(f'{side}: missing {sorted(absent)}')
        if kind == 'dirichlet_profile':
            depths, temperatures = item['depth_m'], item['temperature_C']
            if side not in ('left', 'right'):
                raise ValueError(f'{side}: depth profiles are supported only on vertical sides')
            valid = lambda values: isinstance(values, list) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
            if not valid(depths) or not valid(temperatures) or len(depths) < 2 or len(depths) != len(temperatures):
                raise ValueError(f'{side}: invalid temperature profile')
            if depths[0] != 0 or any(a >= b for a, b in zip(depths, depths[1:])):
                raise ValueError(f'{side}: profile depths must increase from the surface')
            out[side] = dict(type=kind, depth_m=list(depths), temperature_C=list(temperatures))
        elif kind == 'dirichlet':
            out[side] = dict(type=kind, T_C=quantity(item['temperature'], 'temperature', side + '.temperature'))
        elif kind == 'neumann':
            out[side] = dict(type=kind, q_n_W_m2=quantity(item['heat_flux_out'], 'heat_flux', side + '.heat_flux_out'))
        else:
            h = quantity(item['h'], 'heat_transfer_coefficient', side + '.h')
            if h <= 0:
                raise ValueError(f'{side}: Robin requires h > 0 (use an explicit Neumann condition otherwise)')
            out[side] = dict(type=kind, h_W_m2K=h, T_inf_C=quantity(item['T_inf'], 'temperature', side + '.T_inf'))
    if not any(b['type'] in ('dirichlet', 'robin') for b in out.values()):
        raise ValueError('All-Neumann boundary data leave the temperature level undetermined')
    return out


def parse_boundary_conditions(spec, circular=False):
    sides = ('outer',) if circular else RECT_SIDES
    return _parse_conditions(spec, sides)


def parse_annulus_boundary_conditions(spec):
    return _parse_conditions(spec, ('inner', 'outer'))


def parse_contact(spec, interface_names):
    """Imperfect thermal contact: [T] = R'' q.n with q.n continuous; R'' needs a documented source."""
    result = {}
    for item in spec or []:
        if set(item) - {'interface', 'R', 'status', 'source'}:
            raise ValueError('Unknown contact fields')
        name = item.get('interface')
        if name not in interface_names:
            raise ValueError(f'Contact refers to an unknown interface: {name}')
        if name in result:
            raise ValueError(f'Duplicate contact for {name}')
        if not item.get('source') or item.get('status') not in ('publicado', 'extraido', 'inferido', 'supuesto'):
            raise MissingDataError(f'Contact on {name} needs a documented source and data status')
        r = quantity(item['R'], 'contact_resistance', name + '.R')
        if r <= 0:
            raise ValueError(f'Contact resistance on {name} must be positive')
        result[name] = r
    return result