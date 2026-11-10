import json
import sys
from pathlib import Path

import pytest

PINN_V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PINN_V2 / 'front-end'))
import server


def make_completed_pinn(runs, params=None, sha='case-sha', status='done'):
    folder = runs / '20261007_120000_pinn_case_a'
    checkpoint = folder / 'run' / 'model.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    (folder / 'job.json').write_text(json.dumps(dict(kind='pinn', sha=sha, params=params or {
        'architecture': 'M1', 'width': 16, 'depth': 2, 'activation': 'tanh', 'attention': ''})))
    (folder / 'progress.json').write_text(json.dumps(dict(status=status)))
    (folder / 'case.json').write_text(json.dumps(dict(id='case-a', physical=dict(kind='cable'))))
    return folder


def test_resume_requires_completed_same_case_and_compatible_network(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    previous = make_completed_pinn(runs)
    monkeypatch.setattr(server, 'RUNS', runs)
    params = dict(architecture='M1', width=16, depth=2, activation='tanh', attention='')

    assert server.validate_resume_job(previous.name, 'case-sha', params) == previous.name

    with pytest.raises(ValueError, match='otro caso'):
        server.validate_resume_job(previous.name, 'different-sha', params)
    with pytest.raises(ValueError, match='incompatible'):
        server.validate_resume_job(previous.name, 'case-sha', dict(params, width=32))
    with pytest.raises(ValueError, match='no es valida'):
        server.validate_resume_job('../outside', 'case-sha', params)


def test_resume_accepts_legacy_sha_only_when_case_document_matches(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    previous = make_completed_pinn(runs, sha='legacy-sha')
    monkeypatch.setattr(server, 'RUNS', runs)
    params = dict(architecture='M1', width=16, depth=2, activation='tanh', attention='')
    case_doc = dict(id='case-a', physical=dict(kind='cable'))

    assert server.validate_resume_job(previous.name, 'current-sha', params, case_doc) == previous.name

    with pytest.raises(ValueError, match='otro caso'):
        server.validate_resume_job(previous.name, 'current-sha', params, dict(id='other-case'))


def test_resume_rejects_unfinished_jobs_and_missing_checkpoints(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    previous = make_completed_pinn(runs, status='running')
    monkeypatch.setattr(server, 'RUNS', runs)
    monkeypatch.setattr(server, 'alive', lambda pid: True)
    params = dict(architecture='M1', width=16, depth=2, activation='tanh', attention='')

    with pytest.raises(ValueError, match='sigue en marcha'):
        server.validate_resume_job(previous.name, 'case-sha', params)

    monkeypatch.setattr(server, 'alive', lambda pid: False)
    (previous / 'progress.json').write_text('{"status":"done"}')
    (previous / 'run' / 'model.pt').unlink()
    with pytest.raises(ValueError, match='checkpoint'):
        server.validate_resume_job(previous.name, 'case-sha', params)


@pytest.mark.parametrize('status', ['stopped', 'interrupted', 'failed', 'running'])
def test_resume_accepts_incomplete_jobs_with_checkpoint(tmp_path, monkeypatch, status):
    runs = tmp_path / 'runs'
    runs.mkdir()
    previous = make_completed_pinn(runs, status=status)
    monkeypatch.setattr(server, 'RUNS', runs)
    params = dict(architecture='M1', width=16, depth=2, activation='tanh', attention='')

    assert server.validate_resume_job(previous.name, 'case-sha', params) == previous.name


def test_job_case_endpoint_returns_case_used_for_checkpoint(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    previous = make_completed_pinn(runs)
    monkeypatch.setattr(server, 'RUNS', runs)

    result = server.route_get(f'/api/job/{previous.name}/case', {})

    assert result['doc']['id'] == 'case-a'


def test_trainer_initializes_from_previous_solution_weights(tmp_path):
    import torch
    from pinn_v2.solvers.pinn.networks import build_model
    from pinn_v2.solvers.pinn.problem import load_problem
    from pinn_v2.solvers.pinn.trainer import run

    problem = load_problem('mms_constant')
    source_model = build_model(problem, 'M1', width=8, depth=1, activation='tanh').to(dtype=torch.float64)
    initial_state = {name: value.detach().clone() for name, value in source_model.state_dict().items()}
    output = tmp_path / 'continued'

    result = run('mms_constant', 'M1', output, adam_steps=1, width=8, depth=1, points={
        'n_interior': 16, 'n_layer': 8, 'n_face': 8, 'n_boundary': 8}, initial_state=initial_state,
        device='cpu', log_every=1, full_audit=False)

    assert result['status'] == 'completed'
    saved = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
    assert saved['training']['completed_updates'] == 1
    assert any(not torch.equal(saved['model'][name], value) for name, value in initial_state.items())


def test_trainer_keeps_checkpoint_when_logging_fails(tmp_path):
    from pinn_v2.solvers.pinn.trainer import run

    output = tmp_path / 'interrupted'

    def fail_after_checkpoint(row, model):
        raise RuntimeError('simulated interrupted run')

    result = run('mms_constant', 'M1', output, adam_steps=2, width=8, depth=1, points={
        'n_interior': 16, 'n_layer': 8, 'n_face': 8, 'n_boundary': 8}, device='cpu', log_every=1,
        full_audit=False, on_log=fail_after_checkpoint)

    assert result['status'] == 'failed'
    assert (output / 'model.pt').is_file()


def test_portable_checkpoint_rebuilds_model_without_catalog(tmp_path, monkeypatch):
    import shutil
    import numpy as np
    from pinn_v2.solvers.pinn import problem as problems
    from pinn_v2.solvers.pinn.trainer import run

    output = tmp_path / 'trained'
    result = run('mms_interface', 'M1_EXACT', output, adam_steps=2, width=6, depth=1,
        points=dict(n_interior=8, n_layer=8, n_face=8, n_boundary=8),
        device='cpu', log_every=1, full_audit=False)
    assert result['status'] == 'completed'
    isolated = tmp_path / 'isolated'
    isolated.mkdir()
    for name in ('model.pt', 'model.json'):
        shutil.copy2(output / name, isolated / name)

    def forbidden(*args, **kwargs):
        raise AssertionError('Portable loading must not read a catalog')

    monkeypatch.setattr(problems, 'named_case', forbidden)
    from pinn_v2.solvers.pinn.checkpoint import load_checkpoint, infer
    model, specification, bundle = load_checkpoint(isolated / 'model.pt')
    assert specification['physics_sha256'] == result['physics_sha256']
    assert bundle['training']['completed_updates'] == 2
    prediction = infer(model, [[.2, .3], [.8, .3]], flux=True)
    assert prediction['T'].shape == (2,)
    assert prediction['q'].shape == (2, 2)
    assert np.isfinite(prediction['q']).all()
    assert set(path.name for path in isolated.iterdir()) == {'model.pt', 'model.json'}
    import os
    import subprocess
    environment = dict(os.environ, PYTHONPATH=str(PINN_V2))
    executable = sys.executable
    code = ("from pinn_v2.solvers.pinn.checkpoint import load_checkpoint,infer; "
        "model,_,_=load_checkpoint('model.pt'); "
        "print(infer(model,[[.2,.3]],flux=True)['T'][0])")
    completed = subprocess.run([executable, '-c', code], cwd=isolated, env=environment,
        text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize('attention', [None, 'rba', 'sa'])
@pytest.mark.parametrize('resample_every', [0, 2])
def test_portable_resume_matches_uninterrupted_training(tmp_path, attention, resample_every):
    import shutil
    import torch
    from pinn_v2.solvers.pinn.trainer import run, resume_training

    settings = dict(adam_steps=5, width=5, depth=1, lr=.003, attention=attention,
        resample_every=resample_every, points=dict(n_interior=6, n_layer=6, n_face=6, n_boundary=6),
        device='cpu', log_every=1, full_audit=False, validation_every=2)
    continuous = run('mms_interface', 'M1_EXACT', tmp_path / 'continuous', **settings)
    interrupted = run('mms_interface', 'M1_EXACT', tmp_path / 'interrupted', **settings,
        stop_callback=lambda row, model: row['step'] == 2)
    assert continuous['status'] == interrupted['status'] == 'completed'
    isolated = tmp_path / 'isolated'
    isolated.mkdir()
    for name in ('model.pt', 'model.json'):
        shutil.copy2(tmp_path / 'interrupted' / name, isolated / name)
    resumed = resume_training(isolated / 'model.pt', tmp_path / 'resumed', full_audit=False)
    assert resumed['status'] == 'completed'
    expected = torch.load(tmp_path / 'continuous' / 'model.pt', weights_only=True)
    actual = torch.load(tmp_path / 'resumed' / 'model.pt', weights_only=True)
    assert resumed['final_loss'] == continuous['final_loss']
    for name, value in expected['training']['model'].items():
        torch.testing.assert_close(actual['training']['model'][name], value, rtol=0, atol=0)
    assert actual['training']['completed_updates'] == 5
    assert actual['training']['scheduler'] == expected['training']['scheduler']
    for name, value in expected['training']['best_validation_model'].items():
        torch.testing.assert_close(actual['training']['best_validation_model'][name], value, rtol=0, atol=0)


def test_portable_resume_preserves_lbfgs_history(tmp_path):
    import torch
    from pinn_v2.solvers.pinn.trainer import run, resume_training
    settings = dict(adam_steps=2, lbfgs_steps=2, width=4, depth=1,
        points=dict(n_interior=6, n_layer=6, n_face=6, n_boundary=6),
        device='cpu', log_every=1, full_audit=False)
    continuous = run('mms_constant', 'M1_EXACT', tmp_path / 'continuous', **settings)
    interrupted = run('mms_constant', 'M1_EXACT', tmp_path / 'interrupted', **settings,
        stop_callback=lambda row, model: row['phase'] == 'lbfgs')
    assert continuous['status'] == interrupted['status'] == 'completed'
    resumed = resume_training(tmp_path / 'interrupted' / 'model.pt', tmp_path / 'resumed', full_audit=False)
    assert resumed['status'] == 'completed'
    assert resumed['final_loss'] == continuous['final_loss']
    expected = torch.load(tmp_path / 'continuous' / 'model.pt', weights_only=True)
    actual = torch.load(tmp_path / 'resumed' / 'model.pt', weights_only=True)
    for name, value in expected['training']['model'].items():
        torch.testing.assert_close(actual['training']['model'][name], value, rtol=0, atol=0)


def test_portable_pair_rejects_modified_specification(tmp_path):
    from pinn_v2.solvers.pinn.trainer import run
    from pinn_v2.solvers.pinn.checkpoint import load_checkpoint
    output = tmp_path / 'trained'
    result = run('mms_constant', 'M1_EXACT', output, adam_steps=1, width=4, depth=1,
        points=dict(n_interior=4, n_layer=4, n_face=4, n_boundary=4), device='cpu', full_audit=False)
    assert result['status'] == 'completed'
    specification = json.loads((output / 'model.json').read_text())
    specification['config']['dtype'] = 'float32'
    (output / 'model.json').write_text(json.dumps(specification))
    with pytest.raises(ValueError, match='do not match'):
        load_checkpoint(output / 'model.pt')