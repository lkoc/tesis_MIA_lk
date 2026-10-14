"""Run manifests and no-overwrite atomic output. Torch free."""
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pinn_v2.manifest.provenance import PACKAGE, archive_sources, sha256_file


def write_json_atomic(path, payload):
    """Write JSON through a temporary file and a hard link, so an existing file is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        os.link(name, path)
    finally:
        os.unlink(name)
    return path


def replace_json_atomic(path, payload):
    """Atomically replace a mutable JSON file using a temporary in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return path


def git_state(root=None):
    root = Path(root or PACKAGE.parents[1])

    def run(*args):
        done = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None
    return dict(commit=run('rev-parse', 'HEAD'), branch=run('rev-parse', '--abbrev-ref', 'HEAD'),
        dirty=bool(run('status', '--porcelain', '--', 'PINN_v2')))


def host_versions():
    versions = dict(python=sys.version.split()[0], platform=platform.platform())
    for name in ('numpy', 'scipy', 'torch'):
        try:
            module = __import__(name)
            versions[name] = module.__version__
        except ImportError:
            versions[name] = None
    return versions


def build_manifest(kind, command, inputs=(), seeds=None, fem_environment=None, extra=None):
    """Manifest with command, input digests, seeds, environments and source digests."""
    return dict(kind=kind, created_utc=datetime.now(timezone.utc).isoformat(), command=list(command),
        inputs={str(Path(p).name): sha256_file(p) for p in inputs}, seeds=seeds or {},
        host=host_versions(), fem_environment=fem_environment, git=git_state(), extra=extra or {})


def write_manifest(output, manifest, with_sources=True):
    output = Path(output)
    if with_sources:
        manifest = dict(manifest, source_sha256=archive_sources(output))
    return write_json_atomic(output / 'manifest.json', manifest)