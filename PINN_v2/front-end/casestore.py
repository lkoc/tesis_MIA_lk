"""Case files for the front-end: list, load, convert, validate and save.

The catalog (pinn_v2/cases/v2) is read only. New or edited cases are written to front-end/user_cases/<folder>
so they never mix with the catalog. Active catalog entries must already use the v2 format.
"""
import copy
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from pinn_v2.cases.definition import CaseDefinition, SCHEMA_VERSION
from pinn_v2.physics.skin_effect import prescribed_power

USER = HERE.joinpath('user_cases')
CASES = HERE.parent.joinpath('pinn_v2', 'cases')
V2_DIR = CASES.joinpath('v2')
LEGACY_DIR = CASES.joinpath('data')
SEGMENT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-]*$')
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-.]*$')


def _kind_of(doc):
    return (doc.get('physical') or doc).get('kind', '?')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def listing():
    """Active v2 catalog entries and user entries, both discovered recursively."""
    catalog, user = [], []
    for path in sorted(V2_DIR.rglob('*.json')):
        doc = _read(path)
        rel = path.relative_to(V2_DIR).with_suffix('')
        catalog.append(dict(ref='catalog:v2:' + rel.as_posix(), name=path.stem, folder='' if str(rel.parent) == '.' else rel.parent.as_posix(),
                            id=doc.get('id'), kind=_kind_of(doc), title=doc.get('title', ''), group='v2'))
    for path in sorted(USER.rglob('*.json')):
        rel = path.relative_to(USER).with_suffix('')
        try:
            doc = _read(path)
            info = dict(id=doc.get('id'), kind=_kind_of(doc), title=doc.get('title', ''), broken=False)
        except Exception as exc:
            info = dict(id=None, kind='?', title=f'JSON ilegible: {exc}', broken=True)
        user.append(dict(ref='user:' + rel.as_posix(), name=rel.name, folder=rel.parent.as_posix() if str(rel.parent) != '.' else '', **info))
    return dict(catalog=catalog, user=user, folders=folders())


def folders():
    return sorted(p.relative_to(USER).as_posix() for p in USER.rglob('*') if p.is_dir())


def _safe_user_path(folder, name):
    parts = [s for s in re.split(r'[\\/]', folder or '') if s]
    if len(parts) > 3 or not all(SEGMENT.match(s) for s in parts):
        raise ValueError('nombre de carpeta no valido (letras, numeros, guion y guion bajo; maximo 3 niveles)')
    if not NAME.match(name or ''):
        raise ValueError('nombre de archivo no valido (letras, numeros, guion, guion bajo, punto)')
    path = USER.joinpath(*parts, name if name.endswith('.json') else name + '.json').resolve()
    if USER.resolve() not in path.parents:
        raise ValueError('ruta fuera de user_cases')
    return path


def _v2_catalog_path(name):
    parts = (name or '').split('/')
    if not parts or not all((SEGMENT if i < len(parts) - 1 else NAME).match(part) for i, part in enumerate(parts)):
        raise ValueError('referencia v2 no valida')
    path = V2_DIR.joinpath(*parts).with_suffix('.json').resolve()
    if V2_DIR.resolve() not in path.parents:
        raise ValueError('ruta fuera del catalogo v2')
    return path


def explicit_dirichlet(physical):
    sides = ('outer',) if physical.get('outer_radius') else ('bottom', 'top', 'left', 'right')
    return {s: dict(type='dirichlet', temperature=dict(value=physical['T0'], unit='degC')) for s in sides}


def from_legacy(flat, source_id='legacy_catalog', citation='Benchmarks case catalog, heredado'):
    """v2 document of a legacy flat case; cable cases get their implicit Dirichlet T0 written explicitly when that stays valid."""
    base = CaseDefinition.from_legacy(flat, source_id, citation).to_dict()
    notice = None
    if flat.get('kind') == 'cable':
        trial = copy.deepcopy(base)
        trial['boundary_conditions'] = explicit_dirichlet(trial['physical'])
        trial['data_status']['boundary_conditions']['note'] = 'Convertido del contrato heredado: Dirichlet T0 en todo el borde, escrito de forma explicita'
        try:
            CaseDefinition.from_dict(trial)
            base = trial
            notice = 'Caso heredado: se convirtio al formato v2 y se escribio la condicion Dirichlet T0 de los bordes de forma explicita.'
        except Exception:
            notice = 'Caso heredado convertido al formato v2; mantiene su condicion de borde heredada (no se pudo explicitar).'
    else:
        notice = 'Caso de verificacion 2D (no es un cable): se conserva como tal y puede ejecutarse con PINN o FEM.'
    return base, notice


def load(ref):
    """Return dict(doc, readonly, notice, ref) for a case reference."""
    kind, _, rest = ref.partition(':')
    if kind == 'catalog':
        group, _, name = rest.partition(':')
        if group == 'v2':
            path = _v2_catalog_path(name)
            if path.is_file():
                return dict(doc=_read(path), readonly=True, notice='Caso del catalogo v2 (solo lectura): duplicalo a Mis casos para editarlo.', ref=ref)
        raise KeyError(f'caso no encontrado: {ref}')
    if kind == 'user':
        folder, _, name = rest.rpartition('/')
        path = _safe_user_path(folder, name)
        if not path.exists():
            raise KeyError(f'caso no encontrado: {ref}')
        return dict(doc=_read(path), readonly=False, notice=None, ref=ref)
    raise KeyError(f'referencia desconocida: {ref}')


def template(kind):
    if kind == 'single':
        doc = _read(V2_DIR.joinpath('references', 'cables', 'xlpe_single.json'))
        doc = copy.deepcopy(doc)
        doc['id'] = doc['physical']['id'] = 'mi_caso'
        doc['title'] = 'Mi caso (desde plantilla de cable unico XLPE)'
        return doc
    physical = dict(id='mi_caso', kind='cable', bounds=[-2., 2., -3., 0.], T0=20., scale=30., source='Caso nuevo', k=1., cables=[[0., -1.]], radius=.02,
        layers=[[0., .02, 400.]], current=500., R20=1e-4, alpha=.00393, power=500. ** 2 * 1e-4, patch=None, bands=[], pair=None)
    status = {g: dict(status='supuesto', source='autor', note='Completar la procedencia de estos datos') for g in ('geometry', 'materials', 'source', 'boundary_conditions')}
    return dict(schema_version=SCHEMA_VERSION, id='mi_caso', title='Mi caso nuevo', physical=physical, boundary_conditions=explicit_dirichlet(physical), contact=[],
        provenance=[dict(id='autor', citation='Definido por el usuario en el front-end')], data_status=status, inconsistencies=[])


def check(doc):
    """Validate a v2 document exactly as the solvers do and add soft warnings and a summary for the UI."""
    out = dict(ok=False, errors=[], warnings=[], summary={}, pinn=dict(supported=False, reason=''), fem=dict(supported=False, reason=''))
    try:
        definition = CaseDefinition.from_dict(copy.deepcopy(doc))
    except Exception as exc:
        out['errors'].append(f'{type(exc).__name__}: {exc}')
        return out
    out['ok'] = True
    phys = definition.physical
    out['summary'] = dict(id=definition.id, kind=phys['kind'], sha256=definition.sha256[:12])
    if phys['kind'] != 'cable':
        supported = phys['kind'] == 'annulus' or phys['kind'].startswith('mms')
        if supported:
            out['pinn']['supported'] = True
            out['fem']['supported'] = True
            out['summary']['verification'] = 'Caso de verificacion general de conduccion 2D'
        else:
            out['pinn']['reason'] = 'Tipo de caso no soportado por el solver PINN.'
            out['fem']['reason'] = 'Tipo de caso no soportado por el solver FEM.'
        return out
    domain = definition.domain()
    power = prescribed_power(domain.case)
    r0 = phys['layers'][0][1]
    b = phys['bounds']
    out['summary'].update(regions=len(domain.regions), cables=len(phys['cables']), power_W_m=power, Q_W_m3=power / (math.pi * r0 ** 2) if r0 > 0 else None,
        width_m=b[1] - b[0], depth_m=b[3] - b[2], interfaces=len(domain.interfaces), imperfect_contact=bool(domain.has_imperfect_contact),
        interface_names=[i.name for i in domain.interfaces], region_names=[r.name for r in domain.regions])
    out['fem']['supported'] = True
    out['pinn']['supported'] = not domain.has_imperfect_contact
    if domain.has_imperfect_contact:
        out['pinn']['reason'] = 'Contacto imperfecto: solo lo resuelve el FEM en esta version.'
    if len(domain.regions) > 30:
        out['warnings'].append(f'{len(domain.regions)} regiones: el PINN sera lento; usa menos puntos o pocos pasos.')
    radius = phys['radius']
    for c in phys['cables']:
        gap = min(c[0] - b[0], b[1] - c[0], c[1] - b[2], b[3] - c[1])
        if gap < 5 * radius:
            out['warnings'].append(f'El cable en ({c[0]:g}, {c[1]:g}) queda a {gap:g} m del borde (menos de 5 radios): el dominio puede ser muy estrecho.')
            break
    bcs = definition.boundary_conditions
    if bcs and all(v.get('type') == 'neumann' for v in bcs.values()):
        out['warnings'].append('Todos los bordes son Neumann: la temperatura queda indeterminada.')
    return out


def save(folder, name, doc, overwrite=False, allow_invalid=False):
    path = _safe_user_path(folder, name)
    verdict = check(doc)
    if not verdict['ok'] and not allow_invalid:
        raise ValueError('El caso no valida; se puede guardar como borrador marcandolo expresamente.\n' + '\n'.join(verdict['errors']))
    if path.exists() and not overwrite:
        raise FileExistsError('ya existe ese archivo; confirma para sobrescribir')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return dict(ref='user:' + path.relative_to(USER).with_suffix('').as_posix(), path=str(path), valid=verdict['ok'], errors=verdict['errors'])


def make_folder(folder):
    parts = [s for s in re.split(r'[\\/]', folder or '') if s]
    if not parts or len(parts) > 3 or not all(SEGMENT.match(s) for s in parts):
        raise ValueError('nombre de carpeta no valido')
    path = USER.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return '/'.join(parts)