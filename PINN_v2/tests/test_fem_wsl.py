"""Conditional FEM check: runs the FEniCSx reference inside WSL2 and compares with the exact coaxial solution."""
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
import pytest
from pinn_v2.solvers.fem import adapter

TOP=Path(__file__).resolve().parents[1]
DISTRO='Ubuntu'
SEP=chr(47)
PYTHON=SEP.join(['','home','lkoc','miniforge3','envs','fenicsx','bin','python'])
pytestmark=pytest.mark.fem


def wsl_path(path):
    path=Path(path).resolve();drive=path.drive.rstrip(':').lower()
    return SEP.join(['','mnt',drive,*path.parts[1:]])


def wsl_available():
    if shutil.which('wsl') is None:return False
    done=subprocess.run(['wsl','-d',DISTRO,'-e','test','-x',PYTHON],capture_output=True)
    return done.returncode==0


@pytest.mark.skipif(not wsl_available(),reason='FEniCSx environment not available in WSL2 Ubuntu')
def test_coaxial_fem_matches_exact_solution_and_refuses_overwrite():
    with tempfile.TemporaryDirectory() as work:
        output=Path(work).joinpath('out')
        command=f'cd {wsl_path(TOP)} && {PYTHON} -m pinn_v2.solvers.fem.full_fem --case coaxial_full --levels 0 --output {wsl_path(output)}'
        run=lambda:subprocess.run(['wsl','-d',DISTRO,'-e','bash','-c',command],capture_output=True,text=True)
        done=run()
        assert done.returncode==0,done.stderr[-2000:]
        record=json.loads(output.joinpath('fem_l0.json').read_text(encoding='utf-8'))
        assert record['balance_pct']<1.0
        assert record['rmse_exact_K']<5e-3
        assert record['error_Tmax_exact_K']<5e-3
        assert any(name.endswith('full_fem.py') for name in record['source_sha256'])
        repeat=run()
        assert repeat.returncode!=0 and 'FileExistsError' in repeat.stderr


@pytest.mark.skipif(not wsl_available(),reason='FEniCSx environment not available in WSL2 Ubuntu')
def test_kim_multicable_fem_evaluates_curved_material_cells():
    case=TOP.joinpath('pinn_v2','cases','v2','kim_2024_thermal_original.json')
    with tempfile.TemporaryDirectory() as work:
        record=adapter.run_cable_fem(Path(work).joinpath('out'),[0],definition=case)[0]
    assert record['ndofs']>0
    assert len([region for region in record['regions'] if 'Tmax_C' in region])==6
    assert all(math.isfinite(region['Tmax_C']) for region in record['regions'] if 'Tmax_C' in region)
    assert record['balance_consistent_pct']<1e-5