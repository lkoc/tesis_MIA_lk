from pathlib import Path
import pytest
from Benchmarks import full_paths


def test_archived_windows_and_linux_paths_resolve_in_moved_checkout(tmp_path,monkeypatch):
    monkeypatch.setattr(full_paths,'ROOT',tmp_path)
    moved=tmp_path/'Benchmarks/explicit_study/reference.npz';moved.parent.mkdir(parents=True);moved.write_bytes(b'field')
    for original in [r'C:\old\project\Benchmarks\explicit_study\reference.npz','/mnt/c/old/project/Benchmarks/explicit_study/reference.npz','Benchmarks/explicit_study/reference.npz']:
        assert full_paths.artifact_path(original)==moved


def test_missing_or_escaping_reference_is_not_silently_accepted(tmp_path,monkeypatch):
    monkeypatch.setattr(full_paths,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):full_paths.artifact_path('Benchmarks/missing.npz')
    with pytest.raises(ValueError):full_paths.artifact_path('Benchmarks/../../unrelated.npz')
