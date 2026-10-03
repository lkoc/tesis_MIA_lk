"""Verification case builders for Neumann, Robin and symmetry checks (CaseDefinition v2).

All geometry and material values come from the inherited coaxial verification case; the
boundary data are verification choices flagged as assumptions, never paper data.
"""
import copy
from pinn_v2.cases.definition import CaseDefinition
from pinn_v2.geometry.domain import coaxial_case

SOURCE = dict(id='pinn_v2_verification', citation='PINN_v2 verification builders (inherited coaxial case with chosen boundary data)')


def _status(note):
    return {group: dict(status='supuesto', source=SOURCE['id'], note=note) for group in ('geometry', 'materials', 'source', 'boundary_conditions')}


def _dirichlet(value):
    return dict(type='dirichlet', temperature=dict(value=value, unit='degC'))


def _robin(h, t_inf):
    return dict(type='robin', h=dict(value=h, unit='W/(m2 K)'), T_inf=dict(value=t_inf, unit='degC'))


def _neumann(flux):
    return dict(type='neumann', heat_flux_out=dict(value=flux, unit='W/m2'))


def coaxial_robin(h=15., t_inf=20.):
    """Concentric cable with a Robin outer boundary: exact radial solution exists (radial_exact)."""
    physical = coaxial_case()
    physical['id'] = 'coaxial_robin'
    physical.pop('outer_gradient_K_m', None)
    return CaseDefinition.from_dict(dict(schema_version='pinn-v2-case-1', id='coaxial_robin',
        title='Concentric cable, Robin outer boundary', physical=physical, boundary_conditions=dict(outer=_robin(h, t_inf)),
        provenance=[SOURCE], data_status=_status('Inherited coaxial case; Robin data chosen for verification')))


def coaxial_dirichlet(temperature=20.):
    physical = coaxial_case()
    physical['id'] = 'coaxial_bc_dirichlet'
    return CaseDefinition.from_dict(dict(schema_version='pinn-v2-case-1', id='coaxial_bc_dirichlet',
        title='Concentric cable, explicit Dirichlet outer boundary', physical=physical,
        boundary_conditions=dict(outer=_dirichlet(temperature)),
        provenance=[SOURCE], data_status=_status('Inherited coaxial case; explicit Dirichlet equal to the legacy value')))


def _two_cable_physical(identifier, bounds, cables):
    physical = coaxial_case()
    for key in ('outer_radius', 'outer_gradient_K_m'):
        physical.pop(key, None)
    physical.update(id=identifier, bounds=bounds, cables=cables, scale=1.)
    return physical


def symmetric_pair(half, h=15., t_inf=20., t_ref=15., offset=.5, depth=-1.5, width=4., height=4.):
    """Two cables symmetric about x=0 (full) or one cable with a Neumann q.n=0 symmetry side (half)."""
    top, bottom, side = _robin(h, t_inf), _dirichlet(t_ref), _dirichlet(t_ref)
    if half:
        physical = _two_cable_physical('symmetric_half', [0., width, -height, 0.], [[offset, depth]])
        conditions = dict(top=top, bottom=bottom, left=_neumann(0.), right=side)
        identifier, title = 'symmetric_half', 'Half domain with symmetry Neumann side'
    else:
        physical = _two_cable_physical('symmetric_full', [-width, width, -height, 0.], [[-offset, depth], [offset, depth]])
        conditions = dict(top=top, bottom=bottom, left=side, right=side)
        identifier, title = 'symmetric_full', 'Full domain with two symmetric cables'
    return CaseDefinition.from_dict(dict(schema_version='pinn-v2-case-1', id=identifier, title=title, physical=physical,
        boundary_conditions=conditions, provenance=[SOURCE],
        data_status=_status('Inherited coaxial layers; geometry and boundary data chosen for the symmetry cross-check')))


def all_verification_definitions():
    return [coaxial_robin(), coaxial_dirichlet(), symmetric_pair(False), symmetric_pair(True)]