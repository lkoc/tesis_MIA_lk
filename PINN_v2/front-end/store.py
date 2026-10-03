"""Solutions the front-end can compare: finished jobs and results already on disk.

A solution id (sid) is job:<job_id>:<variant> for something computed from this front-end, or
res:<path relative to PINN_v2> for an existing run (FEM npz pair or a PINN folder with model.pt).
Existing results are only read, never changed. Grids are cached in front-end/cache.
"""
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

import casestore
import grids

ROOT = casestore.HERE.parent
RUNS = casestore.HERE.joinpath('runs')
CACHE = casestore.HERE.joinpath('cache')
RESULTS = ROOT.joinpath('results')
EXTERNAL_RESULTS = ROOT.parent.joinpath('Benchmarks', 'explicit_study', 'D', 'kim_thermal_shared_20261002', 'fem_reaction_balance')
BUILD_LOCK = threading.Lock()
MEMORY = {}
LISTING = dict(at=0., items=[])


def short_study(path, root):
    parts = Path(path).relative_to(root).parts
    if root == EXTERNAL_RESULTS:
        return 'Benchmarks/D/kim_thermal_shared_20261002'
    return '/'.join(parts[:2]) if len(parts) > 2 else parts[0]


def _result_roots():
    return (RESULTS, EXTERNAL_RESULTS)


def _result_path(sid):
    relative = Path(sid[4:])
    if not relative.parts or relative.is_absolute() or any(part in ('.', '..') for part in relative.parts):
        raise KeyError('ruta de resultado no valida')
    if relative.parts[0] == 'results':
        root, parts = RESULTS, relative.parts[1:]
    elif relative.parts[0] == 'kim_thermal':
        root, parts = EXTERNAL_RESULTS, relative.parts[1:]
    else:
        raise KeyError('ruta de resultado no permitida')
    if not parts:
        raise KeyError('ruta de resultado no valida')
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise KeyError('ruta fuera del arbol de resultados') from exc
    return path


def _is_external_result(path):
    try:
        Path(path).resolve().relative_to(EXTERNAL_RESULTS.resolve())
        return True
    except ValueError:
        return False


def _display_case(case, path):
    display_case = dict(case)
    if _is_external_result(path):
        for key in ('source_power_W_m', 'thermal_source', 'electrical', 'boundary_conditions', 'patch'):
            display_case.pop(key, None)
    return display_case


def _display_domain(spec, path):
    from pinn_v2.geometry.domain import FullDomain
    domain = FullDomain(_display_case(spec['case'], path), spec.get('source_mode', 'fixed'))
    if _is_external_result(path):
        domain.case['patch'] = spec['case'].get('patch')
    return domain


def pinn_source(case_id):
    """Source for load_problem: a v2 definition when one exists, else the catalog id."""
    from pinn_v2.cases.definition import CaseDefinition
    path = casestore.V2_DIR.joinpath(f'{case_id}.json')
    return CaseDefinition.load(path) if path.exists() else case_id


def scan_results():
    """Cheap listing of FEM and PINN results on disk, grouped later by case id."""
    items = []
    for root in _result_roots():
        if not root.exists():
            continue
        for js in root.rglob('fem_l*.json'):
            npz = js.with_suffix('.npz')
            if not npz.exists() or js.is_symlink() or npz.is_symlink():
                continue
            try:
                if root.resolve() not in js.resolve().parents or root.resolve() not in npz.resolve().parents:
                    continue
                rec = json.loads(js.read_text(encoding='utf-8'))
                case = rec['specification']['case']
            except Exception:
                continue
            if case.get('kind') != 'cable':
                continue
            if root == RESULTS:
                sid = 'res:' + js.with_suffix('').relative_to(ROOT).as_posix()
            else:
                sid = 'res:kim_thermal/' + js.with_suffix('').relative_to(root).as_posix()
            smoke = root == EXTERNAL_RESULTS and rec.get('level') == 0
            label = f"FEM nivel {rec.get('level')}" + (' (smoke, no convergido)' if smoke else '')
            items.append(dict(sid=sid, case=case['id'], kind='fem', label=f"{label} [{short_study(js, root)}]", Tmax_C=rec.get('Tmax_C'), ndofs=rec.get('ndofs'),
                sha=rec.get('physics_sha256'), when=js.stat().st_mtime, origin='benchmarks' if root == EXTERNAL_RESULTS else 'resultados'))
    for model in RESULTS.rglob('model.pt'):
        folder = model.parent
        try:
            res = json.loads(folder.joinpath('result.json').read_text(encoding='utf-8'))
        except Exception:
            continue
        if res.get('status') != 'completed' or str(res.get('case', '')).startswith('mms') or str(res.get('case', '')).startswith('annulus'):
            continue
        cfg = res.get('config', {})
        spec = cfg.get('architecture', '?') + (f"+{cfg['attention']}" if cfg.get('attention') else '')
        ev = res.get('evaluation', {})
        items.append(dict(sid='res:' + folder.relative_to(ROOT).as_posix(), case=res['case'], kind='pinn', label=f"PINN {spec} {folder.name} [{short_study(folder, RESULTS)}]",
            Tmax_C=ev.get('Tmax_pred_C'), sha=res.get('physics_sha256'), when=model.stat().st_mtime, origin='resultados', final_loss=res.get('final_loss'),
            steps=cfg.get('adam_steps')))
    return items


def scan_jobs():
    items = []
    for folder in sorted(RUNS.glob('*')):
        try:
            prog = json.loads(folder.joinpath('progress.json').read_text(encoding='utf-8'))
            job = json.loads(folder.joinpath('job.json').read_text(encoding='utf-8'))
        except Exception:
            continue
        for sol in prog.get('solutions', []):
            gfile = folder.joinpath('grids', sol['variant'] + '.npz')
            if gfile.exists():
                items.append(dict(sid=f"job:{folder.name}:{sol['variant']}", case=job['case_id'], kind=job['kind'], label=f"{sol['label']} [{folder.name[:15]}]",
                    Tmax_C=prog.get('summary', {}).get('Tmax_C'), sha=job.get('sha'), when=gfile.stat().st_mtime, origin='front-end'))
    return items


def all_solutions(fresh=False):
    if fresh or time.time() - LISTING['at'] > 30:
        LISTING.update(at=time.time(), items=scan_results())
    return scan_jobs() + LISTING['items']


def by_case(case_id=None):
    items = all_solutions(fresh=case_id is None)
    if case_id is None:
        counts = {}
        for it in items:
            row = counts.setdefault(it['case'], dict(case=it['case'], fem=0, pinn=0))
            row[it['kind']] += 1
        return sorted(counts.values(), key=lambda r: (-(r['fem'] > 0 and r['pinn'] > 0), r['case']))
    return sorted([it for it in items if it['case'] == case_id], key=lambda it: (it['kind'] != 'fem', -it['when']))


def _job_case(job_id):
    from pinn_v2.cases.definition import CaseDefinition
    from pinn_v2.solvers.pinn.problem import load_problem
    doc = json.loads(RUNS.joinpath(job_id, 'case.json').read_text(encoding='utf-8'))
    problem = load_problem(CaseDefinition.from_dict(doc))
    case = dict(problem.case)
    case.setdefault('bounds', list(problem.bounds))
    return case


def _build_res(path):
    """Grids of an existing result, built once and cached on disk."""
    if path.with_suffix('.npz').exists():
        rec = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        spec = rec['specification']
        domain = _display_domain(spec, path)
        pieces = grids.build_fem(path.with_suffix('.npz'), domain, domain.case)
        label = f"FEM nivel {rec.get('level')}"
        if _is_external_result(path) and rec.get('level') == 0:
            label += ' (smoke, no convergido)'
        meta = dict(kind='fem', label=label, physics_sha256=rec.get('physics_sha256'))
    else:
        from pinn_v2.solvers.pinn.problem import load_problem
        res = json.loads(path.joinpath('result.json').read_text(encoding='utf-8'))
        problem = load_problem(pinn_source(res['case']))
        model = grids.load_pinn_model(path, problem)
        pieces = grids.build_pinn(model, problem.domain, problem.case)
        domain = problem.domain
        meta = dict(kind='pinn', label=f"PINN {res['config']['architecture']} {path.name}", physics_sha256=res.get('physics_sha256'))
    return pieces, [r.name for r in domain.regions], meta, domain.case


def get(sid):
    """Solution dict: label, pieces (full and zoom grids), names, meta, case (solver dictionary)."""
    if sid in MEMORY:
        return MEMORY[sid]
    if sid.startswith('job:'):
        _, job_id, variant = sid.split(':', 2)
        folder = RUNS.joinpath(job_id)
        if not folder.resolve().parent == RUNS.resolve():
            raise KeyError('trabajo no valido')
        live = variant == 'live'
        pieces, names, meta = grids.load(folder.joinpath('live_grid.npz' if live else f'grids/{variant}.npz'))
        sol = dict(sid=sid, label=meta.get('label', variant) + (' (en vivo)' if live else ''), pieces=pieces, names=names, meta=meta, case=_job_case(job_id))
        if not live:
            MEMORY[sid] = sol
        return sol
    if sid.startswith('res:'):
        path = _result_path(sid)
        key = hashlib.sha1((sid + str(max((p.stat().st_mtime for p in path.parent.glob(path.name + '*')), default=0.))).encode()).hexdigest()[:16]
        cached = CACHE.joinpath(key + '.npz')
        with BUILD_LOCK:
            if cached.exists():
                pieces, names, meta = grids.load(cached)
                case = _case_of_res(path)
            else:
                pieces, names, meta, case = _build_res(path)
                grids.save(cached, pieces, names, meta)
        result_root = EXTERNAL_RESULTS if _is_external_result(path) else RESULTS
        sol = dict(sid=sid, label=meta['label'] + f' [{short_study(path, result_root)}]', pieces=pieces, names=names, meta=meta, case=case)
        MEMORY[sid] = sol
        if len(MEMORY) > 24:
            MEMORY.pop(next(iter(MEMORY)))
        return sol
    raise KeyError(f'sid desconocido: {sid}')


def _case_of_res(path):
    if path.with_suffix('.npz').exists():
        spec = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['specification']
        return _display_domain(spec, path).case
    from pinn_v2.solvers.pinn.problem import load_problem
    res = json.loads(path.joinpath('result.json').read_text(encoding='utf-8'))
    return load_problem(pinn_source(res['case'])).case


def compatible(ref, cand):
    a, b = ref['pieces']['full'], cand['pieces']['full']
    return a['T'].shape == b['T'].shape and np.allclose(a['window'], b['window']) and ref['pieces']['zoom']['T'].shape == cand['pieces']['zoom']['T'].shape