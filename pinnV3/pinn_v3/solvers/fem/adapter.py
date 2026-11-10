"""Windows side adapter: runs the FEniCSx reference inside WSL2 and returns its JSON records.

Torch free. FEniCSx lives only in the WSL2 Ubuntu environment, so every FEM result is
produced by subprocess and consumed from JSON files. Every record must carry the fingerprint
of the problem document that the PINN engines load.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from pinn_v3.problem.build import load_problem as load_definition
from pinn_v3.solvers.fem.mesh_resolution import validate_node_multiplier

SEP = chr(47)
# Project root: the directory that contains the pinn_v3 package.
TOP = Path(__file__).resolve().parents[3]
DISTRO = os.environ.get('PINN_V3_WSL_DISTRO', 'Ubuntu')
PYTHON = os.environ.get('PINN_V3_FEM_PYTHON', SEP.join(['', 'home', 'lkoc', 'miniforge3', 'envs', 'fenicsx', 'bin', 'python']))


def wsl_path(path):
    """Translate a Windows path to its /mnt/<drive>/... form inside WSL."""
    path = Path(path).resolve()
    drive = path.drive.rstrip(':').lower()
    return SEP.join(['', 'mnt', drive, *path.parts[1:]])


def fem_available():
    if shutil.which('wsl') is None:
        return False
    done = subprocess.run(['wsl', '-d', DISTRO, '-e', 'test', '-x', PYTHON], capture_output=True)
    return done.returncode == 0


def run_module(module, arguments=(), check=True):
    """Run python -m module inside the FEniCSx environment with the project root as working directory."""
    words = ' '.join([PYTHON, '-m', module, *[wsl_path(a) if isinstance(a, Path) else str(a) for a in arguments]])
    command = f'cd {wsl_path(TOP)} && {words}'
    done = subprocess.run(['wsl', '-d', DISTRO, '-e', 'bash', '-c', command], capture_output=True, text=True)
    if check and done.returncode != 0:
        raise RuntimeError(f'FEM command failed ({done.returncode}): {module}\n{done.stderr[-3000:]}')
    return done


def environment():
    return json.loads(run_module('pinn_v3.solvers.fem.environment').stdout)


def fingerprint(path):
    """SHA-256 of the canonical problem document, the value the PINN engines report as physics_sha256."""
    return hashlib.sha256(json.dumps(load_definition(path).doc, sort_keys=True).encode()).hexdigest()


def run_problem_fem(source, output, levels=(0,), node_multipliers=(1.0,)):
    """FEniCSx reference for one problem document; every record must solve the PINN problem.

    source is a problem.json path or a document dictionary; output must be a new folder.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if isinstance(source, dict):
        path = output / 'input_problem.json'
        path.write_text(json.dumps(source, indent=2), encoding='utf-8')
    else:
        path = Path(source)
    expected = fingerprint(path)
    multipliers = [str(validate_node_multiplier(value)) for value in node_multipliers]
    run_module('pinn_v3.solvers.fem.generic_fem', ['--problem', path, '--levels', *map(str, levels),
        '--node-multipliers', *multipliers, '--output', output])
    records = [json.loads(item.read_text(encoding='utf-8')) for item in sorted(output.glob('fem_*.json'))]
    if not records:
        raise RuntimeError(f'FEM produced no records in {output}')
    for record in records:
        if record['physics_sha256'] != expected:
            raise RuntimeError('FEM solved a different problem than the PINN engines (fingerprint mismatch)')
    return records
