"""CaseDefinition v2: one declarative, hashed, provenance-carrying description of a case.

The physical block keeps the validated legacy schema (geometry, layers, k, current).
Around it v2 adds: schema version, sources, a data status for every group, per-side
boundary conditions, documented imperfect contacts, known inconsistencies and a hash.
A case is rejected if a required group is missing, flagged as missing, or has no
documented source. Paper values are inputs of the physics or verification data,
never training labels.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from pinn_v2.cases.catalog import validate_case
from pinn_v2.cases.boundary import parse_annulus_boundary_conditions
from pinn_v2.cases.errors import MissingDataError
from pinn_v2.cases.units import quantity
from pinn_v2.physics.skin_effect import prescribed_power

SCHEMA_VERSION = 'pinn-v2-case-1'
STATUSES = ('publicado', 'extraido', 'inferido', 'supuesto', 'faltante')
REQUIRED_GROUPS = ('geometry', 'materials', 'source', 'boundary_conditions')
EXTRA_KEYS = {'outer_radius', 'physics_version', 'heat_capacity_J_m3K', 'outer_gradient_K_m', 'electrical',
              'boundary_conditions', 'contact', 'schema_version', 'source_power_W_m'}
TOP_KEYS = {'schema_version', 'id', 'title', 'physical', 'boundary_conditions', 'contact',
            'provenance', 'data_status', 'inconsistencies'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


@dataclass(frozen=True)
class CaseDefinition:
    schema_version: str
    id: str
    title: str
    physical: dict
    boundary_conditions: dict | None
    contact: tuple
    provenance: tuple
    data_status: dict
    inconsistencies: tuple = ()

    def to_dict(self):
        return dict(schema_version=self.schema_version, id=self.id, title=self.title, physical=copy.deepcopy(self.physical),
                    boundary_conditions=copy.deepcopy(self.boundary_conditions), contact=copy.deepcopy(list(self.contact)),
                    provenance=copy.deepcopy(list(self.provenance)), data_status=copy.deepcopy(self.data_status),
                    inconsistencies=copy.deepcopy(list(self.inconsistencies)))

    @property
    def sha256(self):
        return hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - TOP_KEYS
        if unknown:
            raise ValueError(f'Unknown case-definition fields: {sorted(unknown)}')
        missing = {'schema_version', 'id', 'physical', 'provenance', 'data_status'} - set(data)
        if missing:
            raise MissingDataError(f'Case definition is missing {sorted(missing)}')
        case = cls(schema_version=data['schema_version'], id=data['id'], title=data.get('title', data['id']),
                   physical=copy.deepcopy(data['physical']), boundary_conditions=copy.deepcopy(data.get('boundary_conditions')),
                   contact=tuple(copy.deepcopy(data.get('contact', []))), provenance=tuple(copy.deepcopy(data['provenance'])),
                   data_status=copy.deepcopy(data['data_status']), inconsistencies=tuple(copy.deepcopy(data.get('inconsistencies', []))))
        case.validate()
        return case

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def from_legacy(cls, case, source_id, citation, title=None):
        """Wrap a legacy v1 case. Its Dirichlet boundary is recorded as an inherited assumption."""
        physical = copy.deepcopy(case)
        status = {group: dict(status='supuesto', source=source_id, note='Inherited legacy case; origin not individually audited') for group in REQUIRED_GROUPS}
        status['boundary_conditions']['note'] = 'Legacy contract: Dirichlet T0 on the whole outer boundary'
        return cls.from_dict(dict(schema_version=SCHEMA_VERSION, id=physical['id'], title=title or physical['id'], physical=physical,
                                  provenance=[dict(id=source_id, citation=citation)], data_status=status))

    def solver_case(self):
        """Dictionary consumed by FullDomain; legacy cases keep their legacy fingerprint."""
        c = copy.deepcopy(self.physical)
        if self.boundary_conditions is not None:
            c['boundary_conditions'] = copy.deepcopy(self.boundary_conditions)
        if self.contact:
            c['contact'] = copy.deepcopy(list(self.contact))
        return c

    def domain(self, mode='fixed'):
        from pinn_v2.geometry.domain import FullDomain
        return FullDomain(self.solver_case(), mode)

    def validate(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f'Unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION}')
        if not isinstance(self.id, str) or not self.id or self.id != self.physical.get('id'):
            raise ValueError('Case id must be non-empty and equal to physical.id')
        if {'boundary_conditions', 'contact', 'schema_version'} & set(self.physical):
            raise ValueError('boundary_conditions, contact and schema_version belong outside the physical block')
        sources = {p.get('id') for p in self.provenance}
        if None in sources or len(sources) != len(self.provenance) or not all(p.get('citation') for p in self.provenance):
            raise MissingDataError('Every provenance entry needs a unique id and a citation')
        groups = set(REQUIRED_GROUPS) | ({'contact'} if self.contact else set())
        for group in sorted(groups):
            entry = self.data_status.get(group)
            if entry is None:
                raise MissingDataError(f'No data status for required group {group}')
            status = entry.get('status')
            if status not in STATUSES:
                raise ValueError(f'{group}: unknown data status {status!r}')
            if status == 'faltante':
                raise MissingDataError(f'{group}: required data are flagged as missing; the case is rejected')
            if entry.get('source') not in sources:
                raise MissingDataError(f'{group}: source must reference a provenance id')
            if status != 'publicado' and not entry.get('note'):
                raise MissingDataError(f'{group}: status {status} requires an explanatory note')
        for item in self.inconsistencies:
            if not {'field', 'values', 'resolution'} <= set(item):
                raise ValueError('Each inconsistency needs field, values and resolution')
        physical = {k: v for k, v in self.physical.items() if k not in EXTRA_KEYS}
        validate_case(physical)
        kind = self.physical['kind']
        if kind == 'cable':
            domain = self.domain()
            for region in domain.conductors:
                integral = domain.source_integral(region)
                expected = prescribed_power(domain.case)
                if not math.isclose(integral, expected, rel_tol=1e-6):
                    raise ValueError(f'Integral of Q dA ({integral}) differs from the conductor power ({expected}) in {region.name}')
        elif kind == 'annulus':
            if self.contact:
                raise ValueError('Contact data are not defined for annulus verification cases')
            conditions = parse_annulus_boundary_conditions(self.boundary_conditions)
            if conditions['inner']['type'] != 'neumann' or conditions['outer']['type'] != 'dirichlet':
                raise ValueError('The analytic annulus requires inner Neumann flux and outer Dirichlet temperature')
            expected_flux = -self.physical['power'] / (2 * math.pi * self.physical['ri'])
            if not math.isclose(conditions['inner']['q_n_W_m2'], expected_flux, rel_tol=1e-6):
                raise ValueError('Annulus inner heat flux must integrate to the declared power per unit length')
            if not math.isclose(conditions['outer']['T_C'], self.physical['T0'], abs_tol=1e-12):
                raise ValueError('Annulus outer temperature must equal T0')
        elif self.boundary_conditions is not None or self.contact:
            raise ValueError('Boundary conditions and contact are defined for cable cases only in this schema version')
        return self

def load_all(directory=None):
    """Load every active v2 case recursively; legacy records are not solver inputs."""
    result = {}
    root = Path(directory or Path(__file__).with_name('v2'))
    for path in sorted(root.rglob('*.json')):
        definition = CaseDefinition.load(path)
        if definition.id in result:
            raise ValueError(f'Duplicate case id {definition.id}')
        result[definition.id] = definition
    if not result:
        raise ValueError(f'Empty v2 case catalog: {root}')
    return result
