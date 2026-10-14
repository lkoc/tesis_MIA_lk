"""One front-end job in its own process: python worker.py <job_folder>.

Reads job.json and case.json from the folder, runs a PINN training or a FEM ladder, and keeps
progress.json (history, step, status) plus live_grid.npz up to date so the browser can show the
solution while it is still being computed. The FEM and the PINN never exchange data.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np

import grids
import store


def swap_in(tmp, final):
    """Atomic replace that tolerates a reader holding the file open (Windows)."""
    for _ in range(40):
        try:
            os.replace(tmp, final)
            return
        except PermissionError:
            time.sleep(0.05)


def write_json(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False, default=str), encoding='utf-8')
    swap_in(tmp, path)


def clean(row):
    return {key: (v if not isinstance(v, float) or np.isfinite(v) else None) for key, v in row.items()}


class Progress:
    def __init__(self, folder, job):
        self.folder, self.job = folder, job
        self.started = time.time()
        self.state = dict(status='running', kind=job['kind'], phase='preparando', step=0, total_steps=0, history=[], elapsed_s=0., message='', solutions=[], levels=[])
        self.last_write = 0.

    def update(self, force=False, **fields):
        self.state.update(fields)
        self.state['elapsed_s'] = time.time() - self.started
        if force or time.time() - self.last_write > 0.4:
            write_json(self.folder.joinpath('progress.json'), self.state)
            self.last_write = time.time()


def run_pinn(folder, job, case_doc, progress):
    import torch
    torch.set_num_threads(int(job.get('threads', 2)))
    from pinn_v2.cases.definition import CaseDefinition
    from pinn_v2.solvers.pinn import trainer
    from pinn_v2.solvers.pinn.problem import load_problem
    definition = CaseDefinition.from_dict(case_doc)
    problem = load_problem(definition)
    names = [r.name for r in problem.regions]
    case = problem.case
    p = job['params']
    adam, lbfgs = int(p.get('adam_steps', 3000)), int(p.get('lbfgs_steps', 0))
    attention = p.get('attention') or None
    if attention:
        lbfgs = 0
    progress.update(phase='entrenando', total_steps=adam + lbfgs, force=True)
    live = dict(last=0.)

    def on_log(row, model):
        history = progress.state['history']
        history.append(clean(row))
        if len(history) > 3000:
            history[:] = history[::2]
        progress.update(step=row['step'], phase='Adam' if row['phase'] == 'adam' else 'L-BFGS')
        now = time.time()
        if now - live['last'] >= float(job.get('live_every_s', 4.)):
            live['last'] = now
            pieces = grids.build_problem_pinn(model, problem, full_n=110, zoom_n=110)
            tmp = folder.joinpath('live_grid.tmp.npz')
            grids.save(tmp, pieces, names, dict(step=row['step'], live=True))
            swap_in(tmp, folder.joinpath('live_grid.npz'))
            progress.update(live_step=row['step'], live_at=now)

    initial_state = None
    resume_from = job.get('resume_from')
    if resume_from:
        if not isinstance(resume_from, str) or Path(resume_from).name != resume_from:
            raise ValueError('Invalid previous run reference')
        previous_folder = store.RUNS.joinpath(resume_from)
        run_dir = previous_folder.joinpath('run')
        checkpoint = run_dir.joinpath('model.pt')
        if previous_folder.is_symlink() or run_dir.is_symlink() or checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError('Previous PINN checkpoint is unavailable')
        initial_state = torch.load(checkpoint, map_location='cpu', weights_only=True)

    result = trainer.run(definition, p.get('architecture', 'M1'), folder.joinpath('run'), adam_steps=adam, lbfgs_steps=lbfgs, lr=float(p.get('lr', 2e-3)),
        width=int(p.get('width', 32)), depth=int(p.get('depth', 3)), activation=p.get('activation', 'tanh'), weights=p.get('weights') or None, points=p.get('points') or None,
        init_seed=int(p.get('init_seed', 0)), sampling_seed=int(p.get('sampling_seed', 11)), log_every=int(p.get('log_every', 20)), attention=attention,
        deepxde_model=p.get('deepxde_model', 'PFNN'),
        deepxde_sampling=p.get('deepxde_sampling', 'mixed'),
        initial_state=initial_state,
        study='front-end', command=['front-end', 'worker', str(folder)], on_log=on_log)
    folder.joinpath('run', 'case.json').write_text(json.dumps(case_doc), encoding='utf-8')
    if result['status'] != 'completed':
        raise RuntimeError((result.get('error') or {}).get('message', 'el entrenamiento fallo'))
    progress.update(phase='generando campo final', force=True)
    model = grids.load_pinn_model(folder.joinpath('run'), problem)
    pieces = grids.build_problem_pinn(model, problem)
    label = p.get('architecture', 'M1') + (f'/{p.get("deepxde_model", "PFNN")}' if p.get('architecture') == 'DeepXDE' else f'+{attention}' if attention else '')
    out = folder.joinpath('grids')
    out.mkdir(exist_ok=True)
    grids.save(out.joinpath('final.npz'), pieces, names, dict(kind='pinn', label=f'PINN {label}', physics_sha256=problem.physics_sha256))
    ev = result.get('evaluation', {})
    summary = dict(final_loss=result.get('final_loss'), parameters=result.get('parameters'), elapsed_s=result.get('elapsed_s'), power_balance_pct=ev.get('balance', {}).get('global_derived_pct'),
        interface_T_jump_max_K=max([i['T_jump_max_K'] for i in ev.get('interfaces', [])], default=None), Tmax_C=float(np.nanmax(pieces['zoom']['T'])))
    progress.update(status='done', phase='terminado', summary=summary, solutions=[dict(variant='final', label=f'PINN {label}')], force=True)


def run_fem(folder, job, case_doc, progress):
    from pinn_v2.cases.definition import CaseDefinition
    from pinn_v2.solvers.fem import adapter
    from pinn_v2.solvers.fem.mesh_resolution import validate_node_multiplier
    definition = CaseDefinition.from_dict(case_doc)
    from pinn_v2.solvers.pinn.problem import load_problem
    problem = load_problem(definition)
    domain = getattr(problem, 'domain', None)
    names = [r.name for r in problem.regions]
    case = problem.case
    verification = domain is None
    case_path = folder.joinpath('case.json')
    levels = [int(v) for v in job['params'].get('levels', [0, 1])]
    node_multiplier = validate_node_multiplier(job['params'].get('node_multiplier', 1.0))
    refinement_preset = job['params'].get('refinement_preset', 'manual')
    progress.update(total_steps=len(levels), phase='FEM (WSL2)', node_multiplier=node_multiplier,
        refinement_preset=refinement_preset, force=True)
    out = folder.joinpath('grids')
    out.mkdir(exist_ok=True)
    rows, solutions = [], []
    for done, level in enumerate(levels):
        progress.update(step=done, phase=f'FEM nivel {level}: mallando y resolviendo', force=True)
        started = time.time()
        if verification:
            record = adapter.run_mms_fem(folder.joinpath('fem'), levels=[level], definition=case_path,
                node_multiplier=node_multiplier)[0]
            npz_path = folder.joinpath('fem', f'{case["id"]}_l{level}.npz')
            pieces = grids.build_verification_fem(npz_path, problem)
        else:
            record = adapter.run_cable_fem(folder.joinpath('fem'), [level], definition=case_path,
                node_multiplier=node_multiplier)[0]
            npz_path = folder.joinpath('fem', f'fem_l{level}.npz')
            pieces = grids.build_fem(npz_path, domain, case)
        grids.save(out.joinpath(f'l{level}.npz'), pieces, names, dict(kind='fem', label=f'FEM nivel {level}', physics_sha256=record.get('physics_sha256')))
        tmp = folder.joinpath('live_grid.tmp.npz')
        grids.save(tmp, pieces, names, dict(step=level, live=True))
        swap_in(tmp, folder.joinpath('live_grid.npz'))
        tmax = float(np.nanmax(pieces['zoom']['T']))
        rows.append(dict(level=level, ndofs=record.get('ndofs'), Tmax_C=record.get('Tmax_C', tmax),
                 l2_error_K=record.get('l2_error_K'), balance_pct=record.get('balance_pct'),
                 node_multiplier=node_multiplier, elapsed_s=time.time() - started))
        solutions.append(dict(variant=f'l{level}', label=f'FEM nivel {level}'))
        progress.update(step=done + 1, levels=rows, solutions=solutions, live_step=level, force=True)
    change = None
    if len(rows) > 1:
        rise = rows[-1]['Tmax_C'] - case['T0']
        change = 100 * abs(rows[-1]['Tmax_C'] - rows[-2]['Tmax_C']) / rise if rise else None
    progress.update(status='done', phase='terminado', summary=dict(Tmax_C=rows[-1]['Tmax_C'],
        ndofs=rows[-1]['ndofs'], node_multiplier=node_multiplier, refinement_preset=refinement_preset,
        last_change_pct_of_rise=change), force=True)


def main():
    folder = Path(sys.argv[1]).resolve()
    job = json.loads(folder.joinpath('job.json').read_text(encoding='utf-8'))
    case_doc = json.loads(folder.joinpath('case.json').read_text(encoding='utf-8'))
    progress = Progress(folder, job)
    progress.update(force=True)
    try:
        (run_pinn if job['kind'] == 'pinn' else run_fem)(folder, job, case_doc, progress)
    except Exception as exc:
        progress.update(status='failed', phase='error', message=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-3000:], force=True)
        sys.exit(1)


if __name__ == '__main__':
    main()