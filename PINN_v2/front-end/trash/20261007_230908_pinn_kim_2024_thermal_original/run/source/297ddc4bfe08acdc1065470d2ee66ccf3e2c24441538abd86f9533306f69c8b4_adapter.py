"""Windows side adapter: runs the FEniCSx reference inside WSL2 and returns its JSON records.

Torch free. FEniCSx lives only in the WSL2 Ubuntu environment documented in REFACTOR_PLAN.md,
so every FEM result is produced by subprocess and consumed from JSON and NPZ files.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

SEP = chr(47)
TOP = Path(__file__).resolve().parents[3]
DISTRO = os.environ.get('PINN_V2_WSL_DISTRO', 'Ubuntu')
PYTHON = os.environ.get('PINN_V2_FEM_PYTHON', SEP.join(['', 'home', 'lkoc', 'miniforge3', 'envs', 'fenicsx', 'bin', 'python']))


def wsl_path(path):
    path = Path(path).resolve()
    drive = path.drive.rstrip(':').lower()
    return SEP.join(['', 'mnt', drive, *path.parts[1:]])


def fem_available():
    if shutil.which('wsl') is None:
        return False
    done = subprocess.run(['wsl', '-d', DISTRO, '-e', 'test', '-x', PYTHON], capture_output=True)
    return done.returncode == 0


def run_module(module, arguments=(), check=True):
    """Run python -m module inside the FEniCSx environment with the package root as working directory."""
    words = ' '.join([PYTHON, '-m', module, *[wsl_path(a) if isinstance(a, Path) else str(a) for a in arguments]])
    command = f'cd {wsl_path(TOP)} && {words}'
    done = subprocess.run(['wsl', '-d', DISTRO, '-e', 'bash', '-c', command], capture_output=True, text=True)
    if check and done.returncode != 0:
        raise RuntimeError(f'FEM command failed ({done.returncode}): {module}\n{done.stderr[-3000:]}')
    return done


def environment():
    return json.loads(run_module('pinn_v2.solvers.fem.environment').stdout)


def run_cable_fem(output, levels, case=None, definition=None, mode='fixed', case_file=None):
    """Run solvers.fem.full_fem for the given levels; returns the list of level records."""
    output = Path(output)
    arguments = ['--output', output, '--levels', *map(str, levels), '--mode', mode]
    if definition is not None:
        arguments += ['--definition', Path(definition)]
    elif case_file is not None:
        arguments += ['--case-file', Path(case_file)]
    else:
        arguments += ['--case', case]
    run_module('pinn_v2.solvers.fem.full_fem', arguments)
    return [json.loads(output.joinpath(f'fem_l{level}.json').read_text(encoding='utf-8')) for level in levels]


def run_mms_fem(output, case=None, levels=(0, 1), definition=None):
    output = Path(output)
    arguments = ['--levels', *map(str, levels), '--output', output]
    if definition is not None:
        definition = Path(definition)
        case = json.loads(definition.read_text(encoding='utf-8'))['id']
        arguments += ['--definition', definition]
    elif case:
        arguments += ['--case', case]
    else:
        raise ValueError('Give case ID or definition path')
    run_module('pinn_v2.solvers.fem.mms_fem', arguments)
    return [json.loads(output.joinpath(f'{case}_l{level}.json').read_text(encoding='utf-8')) for level in levels]