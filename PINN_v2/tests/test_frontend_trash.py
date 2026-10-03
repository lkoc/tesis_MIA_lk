import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PINN_V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PINN_V2 / 'front-end'))
import server


def test_start_job_records_case_kind_for_worker(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    monkeypatch.setattr(server, 'RUNS', runs)
    monkeypatch.setattr(server, 'PIDS', {})
    monkeypatch.setattr(server, 'PIDS_PROCS', {})
    monkeypatch.setattr(server, 'running_count', lambda: 0)
    monkeypatch.setattr(server.casestore, 'check', lambda doc: dict(
        ok=True, errors=[], pinn=dict(supported=True), fem=dict(supported=True), summary=dict(sha256='test-sha')))
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *args, **kwargs: SimpleNamespace(pid=12345))
    doc = dict(id='mms_smoke', title='MMS smoke', physical=dict(kind='mms'))

    result = server.start_job(dict(kind='pinn', doc=doc))

    job = json.loads((runs / result['id'] / 'job.json').read_text(encoding='utf-8'))
    assert job['case_kind'] == 'mms'


def make_job(parent, job_id, status='done'):
    folder = parent / job_id
    folder.mkdir(parents=True)
    (folder / 'job.json').write_text(json.dumps(dict(kind='pinn', case_id='case-a', created='2026-10-02T12:00:00')))
    (folder / 'progress.json').write_text(json.dumps(dict(status=status)))
    (folder / 'worker.log').write_text('preserved job output')
    return folder


def isolate_dirs(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    trash = tmp_path / 'trash'
    runs.mkdir()
    trash.mkdir()
    monkeypatch.setattr(server, 'RUNS', runs)
    monkeypatch.setattr(server, 'TRASH', trash)
    monkeypatch.setattr(server, 'alive', lambda pid: False)
    return runs, trash


def test_archive_and_restore_preserve_job_files(tmp_path, monkeypatch):
    runs, trash = isolate_dirs(tmp_path, monkeypatch)
    job = make_job(runs, '20261002_120000_pinn_case_a')

    assert server.archive_job(job.name) == dict(ok=True, id=job.name)
    assert not job.exists()
    assert (trash / job.name / 'worker.log').read_text() == 'preserved job output'
    assert server.list_jobs() == []
    assert server.list_trash()[0]['id'] == job.name

    server.restore_job(job.name)
    assert (runs / job.name / 'worker.log').read_text() == 'preserved job output'
    assert not (trash / job.name).exists()


def test_running_job_cannot_be_archived(tmp_path, monkeypatch):
    runs, trash = isolate_dirs(tmp_path, monkeypatch)
    job = make_job(runs, '20261002_120001_pinn_case_a', status='running')
    monkeypatch.setattr(server, 'alive', lambda pid: True)

    with pytest.raises(ValueError, match='Deten el trabajo'):
        server.archive_job(job.name)
    assert job.exists()
    assert list(trash.iterdir()) == []


def test_permanent_delete_and_empty_trash_only_remove_trashed_jobs(tmp_path, monkeypatch):
    runs, trash = isolate_dirs(tmp_path, monkeypatch)
    retained = make_job(runs, '20261002_120002_pinn_case_a')
    first = make_job(trash, '20261002_120003_pinn_case_a')
    second = make_job(trash, '20261002_120004_pinn_case_a')

    server.delete_trash_job(first.name)
    assert not first.exists()
    assert second.exists()
    assert retained.exists()
    assert server.empty_trash() == dict(ok=True, deleted=1)
    assert not second.exists()
    assert retained.exists()


def test_trash_routes_reject_path_traversal(tmp_path, monkeypatch):
    _, trash = isolate_dirs(tmp_path, monkeypatch)
    outside = tmp_path / 'outside'
    outside.mkdir()
    try:
        (trash / '20261002_120005_pinn_case_a').symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f'El entorno no permite crear symlinks: {exc}')

    with pytest.raises(KeyError):
        server.delete_trash_job('20261002_120005_pinn_case_a')
    assert outside.exists()
