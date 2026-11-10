"""Replay seed 11 with its archived sources, leaving original results untouched."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Benchmarks/multiscale_exploration'
source=BASE/'linear/seed11'
meta=json.loads((source/'training_seed11.json').read_text())
environment=BASE/'reproduction/source_environment'
environment.mkdir(parents=True,exist_ok=False)
for original,sha in meta['source_sha256'].items():
    archived=list((source/'source').glob(sha+'_*'))
    assert len(archived)==1 and hashlib.sha256(archived[0].read_bytes()).hexdigest()==sha
    relative=Path(original)
    assert not relative.is_absolute() and '..' not in relative.parts
    dest=environment/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(archived[0].read_bytes())
for package in ['Benchmarks','pinn_cables','pinn_cables/pinn']:
    (environment/package/'__init__.py').write_text('')
config_path=Path('Benchmarks/explicit_study/C/lr_0.0005/xlpe_single/seed11/configuration.json')
dest=environment/config_path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/config_path).read_bytes())
result=json.loads((source/'pinn_seed11.json').read_text())
assert hashlib.sha256(Path(result['reference']).read_bytes()).hexdigest()==result['reference_sha256']
output=BASE/'reproduction/seed11'
with (BASE/'reproduction/run.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,'-X','utf8',str(environment/'Benchmarks/multiscale_linear.py'),
                    '--seed','11','--output',str(output)],cwd=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
first=np.load(source/'pinn_seed11.npz');last=np.load(output/'pinn_seed11.npz')
a=np.load(source/'linear_system.npz');b=np.load(output/'linear_system.npz')
audit=dict(archived_sources_verified=True,reference_sha256=result['reference_sha256'],
           A_bitwise_identical=np.array_equal(a['A'],b['A']),b_bitwise_identical=np.array_equal(a['b'],b['b']),
           field_bitwise_identical=np.array_equal(first['T'],last['T']),
           field_max_difference_K=float(abs(first['T']-last['T']).max()),
           accepted_before=result['accepted'],accepted_after=json.loads((output/'pinn_seed11.json').read_text())['accepted'])
assert audit['A_bitwise_identical'] and audit['b_bitwise_identical']
assert audit['field_max_difference_K']<1e-6 and audit['accepted_before']==audit['accepted_after']
(BASE/'reproduction/audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
