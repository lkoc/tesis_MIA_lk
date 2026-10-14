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
    assert all(torch.equal(saved[name], value) for name, value in initial_state.items())


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