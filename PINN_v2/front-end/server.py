"""Local web front-end for PINN_v2: edit cases, run FEM or PINN jobs, compare solutions, read the docs.

    python front-end/server.py            ->  http://127.0.0.1:8765

Standard library only (http.server) plus the packages PINN_v2 already needs. Everything runs on this
machine and works offline. Jobs are separate processes (worker.py), so a closed tab or a crash of the
page never kills a training; the page just polls progress.json.
"""
import argparse
import datetime
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

import casestore
import render
import store

STATIC = HERE.joinpath('static')
DOCS = ROOT.joinpath('docs', '_build', 'html')
RUNS = store.RUNS
TRASH = HERE.joinpath('trash')
RUN_SETTINGS = HERE.joinpath('run_settings.json')
MAX_RUNNING = 2
PIDS = {}
PIDS_PROCS = {}
INFO = {}
RUN_SETTINGS_LOCK = threading.Lock()
JOB_ID = re.compile(r'^[A-Za-z0-9_\-]{6,80}$')
NO_WINDOW = 0x08000000 if os.name == 'nt' else 0

ARCHITECTURES = [
    dict(id='M0', title='M0 - una red global', note='Una sola red para todo el dominio; la referencia mas simple.'),
    dict(id='M1', title='M1 - una red por region', note='Redes por capa y por suelo con continuidad en las interfaces. Recomendado.'),
    dict(id='M2', title='M2 - mixta (T y flujo)', note='Predice T y el flujo q; la ley de Fourier entra como residuo.'),
    dict(id='CUSP', title='CUSP', note='Variante con funciones de nivel en las interfaces.'),
    dict(id='DCSNN', title='DCSNN', note='Variante con descomposicion en subdominios.'),
    dict(id='FBPINN', title='FBPINN', note='Subdominios con ventanas solapadas.'),
    dict(id='DeepXDE', title='DeepXDE - PINN fuerte', note='Implementacion DeepXDE con perdidas por region, interfaces y fronteras.')]
PRESETS = dict(
    rapido=dict(title='Rapido (prueba)', adam_steps=400, lbfgs_steps=0, width=24, depth=3, lr=3e-3, points=dict(n_interior=128, n_layer=64, n_face=48, n_boundary=48)),
    estandar=dict(title='Estandar', adam_steps=3000, lbfgs_steps=0, width=32, depth=3, lr=2e-3, points=dict(n_interior=512, n_layer=256, n_face=128, n_boundary=128)),
    detallado=dict(title='Detallado', adam_steps=8000, lbfgs_steps=100, width=48, depth=4, lr=2e-3, points=dict(n_interior=1024, n_layer=384, n_face=192, n_boundary=192)))


def options():
    from pinn_v2.solvers.pinn.objective import DEFAULT_WEIGHTS
    from pinn_v2.solvers.pinn.trainer import POINTS
    return dict(architectures=ARCHITECTURES,
        deepxde_models=[dict(id='PFNN', title='PFNN (subred por region)'), dict(id='FNN', title='FNN (red multisalida)')],
        deepxde_samplers=[dict(id='mixed', title='Mixto (actual)'), dict(id='pseudo', title='Pseudoaleatorio'),
            dict(id='LHS', title='Latin Hypercube'), dict(id='Halton', title='Halton'),
            dict(id='Hammersley', title='Hammersley'), dict(id='Sobol', title='Sobol')],
        attention=[dict(id='', title='Ninguna'), dict(id='rba', title='RBA (pesos residuales adaptativos)'), dict(id='sa', title='SA (autoatencion)')],
        activations=['tanh', 'sin', 'silu', 'gelu'], weights=DEFAULT_WEIGHTS, points=POINTS, presets=PRESETS, fem_levels=[0, 1, 2, 3])


def fem_state():
    if 'fem' not in INFO:
        from pinn_v2.solvers.fem import adapter
        try:
            INFO['fem'] = dict(available=bool(adapter.fem_available()), distro=adapter.DISTRO, python=adapter.PYTHON)
        except Exception as exc:
            INFO['fem'] = dict(available=False, error=str(exc))
    return INFO['fem']


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return default


def alive(pid):
    if not pid:
        return False
    for proc in PIDS_PROCS.values():
        if proc.pid == pid:
            return proc.poll() is None
    try:
        out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'], capture_output=True, text=True, timeout=10).stdout
        return str(pid) in out
    except Exception:
        return False

def job_summary(folder, full=False):
    job = read_json(folder.joinpath('job.json'))
    if job is None:
        return None
    case_doc = read_json(folder.joinpath('case.json'), {}) or {}
    case_kind = job.get('case_kind') or (case_doc.get('physical') or case_doc).get('kind', 'cable')
    prog = read_json(folder.joinpath('progress.json'), {}) or {}
    status = prog.get('status', 'en cola')
    pid = (read_json(folder.joinpath('pid.json'), {}) or {}).get('pid')
    if status == 'running' and not alive(pid):
        status = 'interrupted'
    row = dict(id=folder.name, kind=job['kind'], case=job.get('case_id'), case_kind=case_kind, title=job.get('title'), created=job.get('created'), status=status, phase=prog.get('phase'), step=prog.get('step', 0),
        total_steps=prog.get('total_steps', 0), elapsed_s=prog.get('elapsed_s'), summary=prog.get('summary'), message=prog.get('message'), solutions=prog.get('solutions', []),
        params=job.get('params'), sha=job.get('sha'), threads=job.get('threads', 2), live_every_s=job.get('live_every_s', 4),
        has_checkpoint=folder.joinpath('run', 'model.pt').is_file(), live_step=prog.get('live_step'), live_at=prog.get('live_at'))
    if full:
        history = prog.get('history', [])
        row.update(history=history[-1500:], levels=prog.get('levels', []), traceback=prog.get('traceback'), has_live=folder.joinpath('live_grid.npz').exists())
    else:
        row['last_loss'] = (prog.get('history') or [{}])[-1].get('total')
    return row


def list_jobs():
    rows = [job_summary(f) for f in sorted(RUNS.glob('*'), reverse=True) if f.is_dir()]
    return [r for r in rows if r]


def _trash_folder(job_id):
    if not JOB_ID.match(job_id):
        raise KeyError('trabajo no encontrado en la papelera')
    folder = TRASH.joinpath(job_id)
    if not folder.is_dir() or folder.is_symlink() or folder.resolve().parent != TRASH.resolve():
        raise KeyError('trabajo no encontrado en la papelera')
    return folder


def list_trash():
    rows = [job_summary(f) for f in sorted(TRASH.glob('*'), reverse=True) if f.is_dir() and not f.is_symlink()]
    return [r for r in rows if r]


def archive_job(job_id):
    folder = job_folder(job_id)
    row = job_summary(folder)
    if row is None:
        raise ValueError('La carpeta no contiene un trabajo valido')
    if row['status'] == 'running':
        raise ValueError('Deten el trabajo antes de enviarlo a la papelera')
    TRASH.mkdir(parents=True, exist_ok=True)
    target = TRASH.joinpath(folder.name)
    if target.exists():
        raise FileExistsError('Ya existe un trabajo con ese id en la papelera')
    folder.rename(target)
    for sid in list(store.MEMORY):
        if sid.startswith(f'job:{job_id}:'):
            store.MEMORY.pop(sid, None)
    return dict(ok=True, id=job_id)


def restore_job(job_id):
    folder = _trash_folder(job_id)
    target = RUNS.joinpath(job_id)
    if target.exists():
        raise FileExistsError('Ya existe un trabajo con ese id fuera de la papelera')
    RUNS.mkdir(parents=True, exist_ok=True)
    folder.rename(target)
    return dict(ok=True, id=job_id)


def delete_trash_job(job_id):
    folder = _trash_folder(job_id)
    shutil.rmtree(folder)
    for sid in list(store.MEMORY):
        if sid.startswith(f'job:{job_id}:'):
            store.MEMORY.pop(sid, None)
    return dict(ok=True, id=job_id)


def empty_trash():
    TRASH.mkdir(parents=True, exist_ok=True)
    removed = 0
    for folder in list(TRASH.iterdir()):
        if folder.is_dir() and not folder.is_symlink() and JOB_ID.match(folder.name):
            delete_trash_job(folder.name)
            removed += 1
    return dict(ok=True, deleted=removed)


def running_count():
    return sum(1 for r in list_jobs() if r['status'] == 'running')


def validate_fem_params(params):
    from pinn_v2.solvers.fem.mesh_resolution import validate_node_multiplier

    validated = dict(params)
    levels = validated.get('levels', [0, 1])
    if (not isinstance(levels, list) or not levels
            or any(type(level) is not int or level not in (0, 1, 2, 3) for level in levels)
            or len(set(levels)) != len(levels)):
        raise ValueError('levels debe ser una lista sin duplicados con niveles entre 0 y 3')
    validated['levels'] = levels
    validated['node_multiplier'] = validate_node_multiplier(validated.get('node_multiplier', 1.0))
    preset = validated.get('refinement_preset', 'manual')
    if preset not in ('bajo', 'medio', 'alto', 'manual'):
        raise ValueError('refinement_preset debe ser bajo, medio, alto o manual')
    validated['refinement_preset'] = preset
    return validated


def start_job(body):
    kind = body.get('kind')
    if kind not in ('pinn', 'fem'):
        raise ValueError('kind debe ser pinn o fem')
    params = body.get('params') or {}
    if not isinstance(params, dict):
        raise ValueError('params debe ser un objeto')
    if kind == 'fem':
        params = validate_fem_params(params)
    if body.get('ref'):
        doc = casestore.load(body['ref'])['doc']
    else:
        doc = body['doc']
    verdict = casestore.check(doc)
    if not verdict['ok']:
        raise ValueError('El caso no valida: ' + '; '.join(verdict['errors']))
    if not verdict[kind]['supported']:
        raise ValueError(verdict[kind]['reason'] or f'{kind} no soporta este caso')
    if kind == 'fem' and not fem_state().get('available'):
        raise ValueError('FEniCSx no esta disponible (WSL2). Revisa la pestana Ayuda > Entorno.')
    if running_count() >= MAX_RUNNING:
        raise ValueError(f'Ya hay {MAX_RUNNING} trabajos en marcha; espera o detiene uno.')
    resume_from = body.get('resume_from')
    if resume_from:
        if kind != 'pinn':
            raise ValueError('Solo se puede continuar desde una solucion PINN')
        resume_from = validate_resume_job(resume_from, verdict['summary']['sha256'], body.get('params') or {}, doc)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe = re.sub(r'[^A-Za-z0-9_]+', '_', doc['id'])[:30]
    folder = RUNS.joinpath(f'{stamp}_{kind}_{safe}')
    folder.mkdir(parents=True, exist_ok=False)
    folder.joinpath('case.json').write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    job = dict(kind=kind, case_kind=casestore._kind_of(doc), params=params, threads=int(body.get('threads', 2)), live_every_s=float(body.get('live_every_s', 4)), case_id=doc['id'], title=body.get('title') or doc.get('title') or doc['id'],
        sha=verdict['summary']['sha256'], created=datetime.datetime.now().isoformat(timespec='seconds'), ref=body.get('ref'))
    if resume_from:
        job['resume_from'] = resume_from
    folder.joinpath('job.json').write_text(json.dumps(job, indent=1), encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING='utf-8')
    log = open(folder.joinpath('worker.log'), 'wb')
    proc = subprocess.Popen([sys.executable, str(HERE.joinpath('worker.py')), str(folder)], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
    PIDS[folder.name] = proc.pid
    PIDS_PROCS[folder.name] = proc
    folder.joinpath('pid.json').write_text(json.dumps(dict(pid=proc.pid)), encoding='utf-8')
    return dict(id=folder.name)


def collocation_preview(body):
    import numpy as np
    from pinn_v2.cases.definition import CaseDefinition
    from pinn_v2.solvers.pinn.problem import load_problem
    from pinn_v2.solvers.pinn.trainer import POINTS

    doc = body.get('doc')
    params = body.get('params') or {}
    if not isinstance(doc, dict) or not isinstance(params, dict):
        raise ValueError('Caso o parametros de muestreo invalidos')
    problem = load_problem(CaseDefinition.from_dict(doc))
    counts = dict(POINTS, **(params.get('points') or {}))
    if any(type(value) is not int or value < 1 or value > 100000 for value in counts.values()):
        raise ValueError('Los conteos de puntos deben ser enteros entre 1 y 100000')
    region_counts = {region.id: counts['n_interior'] if region.kind == 'soil' else counts['n_layer']
        for region in problem.regions}
    faces = [face.xy for face in problem.faces(counts['n_face'])]
    pieces = [piece.xy for piece in problem.pieces(counts['n_boundary'])]
    summary = dict(regions=len(problem.regions), soil_regions=sum(r.kind == 'soil' for r in problem.regions),
        layer_regions=sum(r.kind != 'soil' for r in problem.regions), interfaces=len(faces), boundary_pieces=len(pieces),
        interior_points=sum(region_counts.values()),
        interface_points=sum(len(points) for points in faces),
        boundary_points=sum(len(points) for points in pieces))
    summary['total_points'] = sum(summary[key] for key in ('interior_points', 'interface_points', 'boundary_points'))
    if not body.get('include_points'):
        return dict(summary=summary)
    seed = int(params.get('sampling_seed', 11))
    if params.get('architecture') == 'DeepXDE':
        from pinn_v2.solvers.pinn import deepxde_solver
        import deepxde as dde
        sampler = params.get('deepxde_sampling', 'mixed')
        samples = [deepxde_solver.sample_deepxde_interior(dde, problem, region,
            region_counts[region.id], seed + 7919 * region.id, sampler) for region in problem.regions]
    else:
        sampling = params.get('sampling', 'mixed')
        samples = [problem.sample_interior(region, region_counts[region.id],
            seed + 7919 * region.id, sampling) for region in problem.regions]
    def as_list(arrays):
        return [np.asarray(points, dtype=float).tolist() for points in arrays]
    return dict(summary=summary, points=dict(
        interior=[dict(region=region.id, xy=np.asarray(points, dtype=float).tolist())
            for region, points in zip(problem.regions, samples)],
        interfaces=as_list(faces), boundary=as_list(pieces)))


def stop_job(job_id):
    folder = RUNS.joinpath(job_id)
    pid = (read_json(folder.joinpath('pid.json'), {}) or {}).get('pid')
    job = read_json(folder.joinpath('job.json'), {})
    if pid and alive(pid):
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
    if job.get('kind') == 'fem':
        try:
            from pinn_v2.solvers.fem import adapter
            subprocess.run(['wsl', '-d', adapter.DISTRO, '-e', 'pkill', '-f', job_id], capture_output=True, timeout=30)
        except Exception:
            pass
    prog = read_json(folder.joinpath('progress.json'), {}) or {}
    if prog.get('status') == 'running':
        prog.update(status='stopped', phase='detenido por el usuario')
        folder.joinpath('progress.json').write_text(json.dumps(prog), encoding='utf-8')
    return dict(ok=True)


def job_folder(job_id):
    if not JOB_ID.match(job_id) or not RUNS.joinpath(job_id).is_dir():
        raise KeyError('trabajo no encontrado')
    return RUNS.joinpath(job_id)


def validate_resume_job(job_id, case_sha, params, case_doc=None):
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise ValueError('La corrida anterior no es valida')
    folder = job_folder(job_id)
    if folder.is_symlink() or folder.resolve().parent != RUNS.resolve():
        raise ValueError('La corrida anterior no es valida')
    previous = read_json(folder.joinpath('job.json'), {}) or {}
    progress = read_json(folder.joinpath('progress.json'), {}) or {}
    status = progress.get('status')
    if status == 'running':
        pid = (read_json(folder.joinpath('pid.json'), {}) or {}).get('pid')
        if alive(pid):
            raise ValueError('No se puede continuar una corrida PINN que sigue en marcha')
        status = 'interrupted'
    if previous.get('kind') != 'pinn' or status not in ('done', 'failed', 'stopped', 'interrupted'):
        raise ValueError('La corrida anterior debe ser un PINN terminado o detenido')
    same_case = bool(case_sha and previous.get('sha') == case_sha)
    if not same_case and case_doc is not None:
        same_case = read_json(folder.joinpath('case.json')) == case_doc
    if not same_case:
        raise ValueError('La corrida anterior pertenece a otro caso')
    previous_params = previous.get('params') or {}
    signature = ('architecture', 'width', 'depth', 'activation', 'deepxde_model')
    defaults = dict(architecture='M1', width=32, depth=3, activation='tanh', deepxde_model='PFNN')
    if any(previous_params.get(key, defaults[key]) != params.get(key, defaults[key]) for key in signature):
        raise ValueError('La corrida anterior usa una arquitectura o estructura de red incompatible')
    if (previous_params.get('attention') or '') != (params.get('attention') or ''):
        raise ValueError('La corrida anterior usa una atencion incompatible')
    run_dir = folder.joinpath('run')
    checkpoint = run_dir.joinpath('model.pt')
    if run_dir.is_symlink() or checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.resolve().parent != run_dir.resolve():
        raise ValueError('La corrida anterior no tiene un checkpoint PINN valido')
    return job_id


def doc_of(ref=None, doc=None):
    return doc if doc is not None else casestore.load(ref)['doc']


def case_dict(doc):
    from pinn_v2.cases.definition import CaseDefinition
    from pinn_v2.solvers.pinn.problem import load_problem
    problem = load_problem(CaseDefinition.from_dict(doc))
    case = dict(problem.case)
    case.setdefault('bounds', list(problem.bounds))
    return case


def num(q, key, default=None, cast=float):
    value = q.get(key, [''])[0]
    return cast(value) if value not in ('', None) else default


def flag(q, key, default=False):
    value = q.get(key)
    return default if not value else value[0] in ('1', 'true', 'on')


def get_solutions(sids):
    sols = [store.get(s) for s in sids if s]
    return sols


def compare_payload(q):
    ref = store.get(q['ref'][0])
    cands = [store.get(s) for s in q.get('cands', [''])[0].split(',') if s and s != q['ref'][0]]
    ok, skipped = [], []
    for c in cands:
        (ok if store.compatible(ref, c) else skipped).append(c)
    return ref, ok, skipped


def latest_run_settings(ref):
    matches = []
    for folder in RUNS.glob('*'):
        if not folder.is_dir():
            continue
        job = read_json(folder.joinpath('job.json'), {}) or {}
        if job.get('ref') == ref:
            matches.append((str(job.get('created') or ''), folder.name, job))
    if not matches:
        return None
    job = max(matches, key=lambda row: (row[0], row[1]))[2]
    params = job.get('params') or {}
    kind = job.get('kind', 'pinn')
    run_params = {key: value for key, value in params.items() if key not in ('architecture', 'levels')}
    return dict(kind=kind, arch=params.get('architecture', 'M1'), preset='custom',
                p=run_params if kind == 'pinn' else None, levels=params.get('levels', [0, 1]),
                threads=job.get('threads', 2), liveEvery=job.get('live_every_s', 4),
                pointMultiplier=1, deepxdeModel=params.get('deepxde_model', 'PFNN'))


def route_get(path, q):
    if path == '/api/info':
        return dict(fem=fem_state(), root=str(ROOT), user_cases=str(casestore.USER), runs=str(RUNS), python=sys.version.split()[0], max_running=MAX_RUNNING)
    if path == '/api/options':
        return options()
    if path == '/api/materials':
        return read_json(HERE.joinpath('materials.json'), {})
    if path == '/api/cases':
        return dict(items=casestore.listing(), folders=casestore.folders())
    if path == '/api/case':
        return casestore.load(q['ref'][0])
    if path == '/api/run-settings':
        ref = q.get('ref', [''])[0]
        if not ref or len(ref) > 512:
            raise ValueError('referencia de caso no valida')
        settings = read_json(RUN_SETTINGS, {}) or {}
        return dict(settings=settings.get(ref) or latest_run_settings(ref))
    if path == '/api/template':
        return dict(doc=casestore.template(q.get('kind', ['empty'])[0]))
    if path == '/api/jobs':
        return dict(jobs=list_jobs())
    if path == '/api/trash':
        return dict(jobs=list_trash())
    m = re.match(r'^/api/job/([^/]+)/case$', path)
    if m:
        case_doc = read_json(job_folder(m.group(1)).joinpath('case.json'))
        if not isinstance(case_doc, dict):
            raise ValueError('La corrida no tiene una definicion de caso valida')
        return dict(doc=case_doc)
    m = re.match(r'^/api/job/([^/]+)$', path)
    if m:
        row = job_summary(job_folder(m.group(1)), full=True)
        return row
    if path == '/api/solutions':
        case = q.get('case', [None])[0]
        return dict(items=store.by_case(case)) if case else dict(cases=store.by_case(None))
    if path == '/api/help':
        from pinn_v2 import cli
        topic = q.get('topic', [None])[0]
        if topic:
            return dict(topic=topic, text=cli.HELP[topic])
        return dict(topics=list(cli.HELP))
    if path == '/api/docs':
        pages = sorted(p.name for p in DOCS.glob('*.html')) if DOCS.exists() else []
        return dict(available=bool(pages), pages=pages)
    if path == '/api/compare.json':
        ref, cands, skipped = compare_payload(q)
        rows = []
        for c in cands:
            row = dict(sid=c['sid'], **{'label': c['label'], **render.metrics(ref, c)})
            row['same_physics'] = (ref['meta'].get('physics_sha256') == c['meta'].get('physics_sha256')) if ref['meta'].get('physics_sha256') and c['meta'].get('physics_sha256') else None
            rows.append(row)
        return dict(reference=ref['label'], rows=rows, skipped=[c['label'] for c in skipped])
    return None


def route_png(path, q):
    if path == '/api/geometry.png':
        case = case_dict(doc_of(ref=q['ref'][0]))
        return render.geometry_png(case, q.get('view', ['full'])[0])
    m = re.match(r'^/api/job/([^/]+)/(image|geometry)\.png$', path)
    if m:
        folder = job_folder(m.group(1))
        if m.group(2) == 'geometry':
            return render.geometry_png(case_dict(read_json(folder.joinpath('case.json'))), q.get('view', ['full'])[0])
        which = q.get('which', ['live'])[0]
        sol = store.get(f'job:{folder.name}:{which}')
        view = q.get('view', ['zoom'])[0]
        collocation = None
        point_path = folder.joinpath('run', 'collocation.npz')
        if flag(q, 'points') and point_path.is_file():
            import numpy as np
            with np.load(point_path, allow_pickle=False) as data:
                collocation = {kind: [data[key] for key in data.files if key.startswith(prefix)]
                    for kind, prefix in (('interior', 'interior_'), ('interfaces', 'interface_'), ('boundary', 'boundary_'))}
        return render.field_png(sol['pieces'], sol['case'], view, q.get('cmap', ['inferno'])[0], flag(q, 'overlay', True), flag(q, 'iso'), num(q, 'vmin'), num(q, 'vmax'), sol['label'], collocation=collocation)
    if path == '/api/solution.png':
        sol = store.get(q['sid'][0])
        view = q.get('view', ['full'])[0]
        return render.field_png(sol['pieces'], sol['case'], view, q.get('cmap', ['inferno'])[0], flag(q, 'overlay', True), flag(q, 'iso'), num(q, 'vmin'), num(q, 'vmax'), sol['label'])
    if path == '/api/compare.png':
        ref, cands, _ = compare_payload(q)
        if not cands:
            raise ValueError('No hay candidatas comparables con la referencia')
        scale = q.get('scale', [''])[0]
        scale = float(scale) if re.match(r'^[0-9.]+$', scale) else (scale or None)
        return render.compare_png(ref, cands, ref['case'], q.get('view', ['full'])[0], q.get('mode', ['signed'])[0], scale, q.get('cmap', ['inferno'])[0], q.get('error_cmap', [''])[0] or None,
            flag(q, 'overlay', True), flag(q, 'detail'))
    if path == '/api/profile.png':
        ref, cands, _ = compare_png_inputs(q)
        return render.profile_png(ref, cands, q.get('axis', ['x'])[0], num(q, 'at', 0.), q.get('mode', ['signed'])[0])
    return None


def compare_png_inputs(q):
    return compare_payload(q)


def route_post(path, body):
    if path == '/api/validate':
        return casestore.check(body['doc'])
    if path == '/api/save':
        return casestore.save(body['folder'], body['name'], body['doc'], bool(body.get('overwrite')), bool(body.get('allow_invalid')))
    if path == '/api/mkdir':
        return dict(folder=casestore.make_folder(body['folder']))
    if path == '/api/run':
        return start_job(body)
    if path == '/api/collocation-preview':
        return collocation_preview(body)
    if path == '/api/run-settings':
        ref, settings = body.get('ref'), body.get('settings')
        if not isinstance(ref, str) or not ref or len(ref) > 512:
            raise ValueError('referencia de caso no valida')
        if not isinstance(settings, dict):
            raise ValueError('parametros de corrida no validos')
        with RUN_SETTINGS_LOCK:
            saved = read_json(RUN_SETTINGS, {}) or {}
            saved[ref] = settings
            temp = RUN_SETTINGS.with_suffix('.tmp')
            temp.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            temp.replace(RUN_SETTINGS)
        return dict(ok=True)
    m = re.match(r'^/api/job/([^/]+)/stop$', path)
    if m:
        return stop_job(job_folder(m.group(1)).name)
    m = re.match(r'^/api/job/([^/]+)/(trash|restore|delete)$', path)
    if m:
        action = {'trash': archive_job, 'restore': restore_job, 'delete': delete_trash_job}[m.group(2)]
        return action(m.group(1))
    if path == '/api/trash/empty':
        return empty_trash()
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = 'PINNv2Front/1'

    def log_message(self, fmt, *args):
        if os.environ.get('FRONT_END_VERBOSE'):
            super().log_message(fmt, *args)

    def send(self, code, data, ctype='application/json; charset=utf-8', cache=False):
        if isinstance(data, (dict, list)):
            data = json.dumps(data, allow_nan=False, default=str, ensure_ascii=False).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'max-age=60' if cache else 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def fail(self, exc, code=400):
        detail = dict(error=f'{type(exc).__name__}: {exc}')
        if code >= 500:
            detail['trace'] = traceback.format_exc()[-1500:]
        self.send(code, detail)

    def static_file(self, base, rel):
        target = (base / rel).resolve()
        if base.resolve() not in target.parents and target != base.resolve():
            return self.send(403, dict(error='ruta no permitida'))
        if target.is_dir():
            target = target / 'index.html'
        if not target.exists():
            return self.send(404, dict(error='no existe'))
        ctype = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
        if ctype.startswith('text/') or ctype in ('application/javascript', 'application/json'):
            ctype += '; charset=utf-8'
        self.send(200, target.read_bytes(), ctype, cache=base == DOCS)

    def do_GET(self):
        url = urlparse(self.path)
        path, q = unquote(url.path), parse_qs(url.query)
        try:
            if path.startswith('/api/'):
                if path.endswith('.png'):
                    png = route_png(path, q)
                    return self.send(200, png, 'image/png') if png else self.send(404, dict(error='ruta desconocida'))
                data = route_get(path, q)
                return self.send(200, data) if data is not None else self.send(404, dict(error='ruta desconocida'))
            if path == '/docs':
                self.send_response(302)
                self.send_header('Location', '/docs/index.html')
                self.end_headers()
                return
            if path.startswith('/docs/'):
                if not DOCS.exists():
                    return self.send(404, dict(error='Las docs no estan construidas: python docs/build_docs.py'))
                return self.static_file(DOCS, path[6:])
            return self.static_file(STATIC, path.lstrip('/') or 'index.html')
        except (KeyError, FileNotFoundError, FileExistsError, ValueError, NotImplementedError) as exc:
            self.fail(exc, 404 if isinstance(exc, (KeyError, FileNotFoundError)) else 400)
        except Exception as exc:
            self.fail(exc, 500)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            length = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(length) or b'{}')
            data = route_post(path, body)
            return self.send(200, data) if data is not None else self.send(404, dict(error='ruta desconocida'))
        except (KeyError, FileNotFoundError, FileExistsError, ValueError, NotImplementedError) as exc:
            self.fail(exc, 409 if isinstance(exc, FileExistsError) else 400)
        except Exception as exc:
            self.fail(exc, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--open', action='store_true', help='abrir el navegador')
    args = parser.parse_args()
    for folder in (RUNS, TRASH, store.CACHE, casestore.USER):
        folder.mkdir(parents=True, exist_ok=True)
    if not DOCS.exists():
        subprocess.run([sys.executable, str(ROOT.joinpath('docs', 'build_docs.py'))], cwd=str(ROOT), capture_output=True)
    threading.Thread(target=fem_state, daemon=True).start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f'http://{args.host}:{args.port}/'
    print(f'PINN_v2 front-end en {url}  (Ctrl+C para salir)', flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()