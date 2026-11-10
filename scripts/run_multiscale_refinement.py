"""Sequential refinements beside the two ongoing full-training pilots."""
from pathlib import Path
import json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Benchmarks/multiscale_exploration'
jobs=[]
for name,options in [('points_half',['--factor','.5']),('points_double',['--factor','2']),('small',['--width','16','--depth','2'])]:
    for seed in [11,23]:
        jobs.append(dict(name=name,seed=seed,options=options))
manifest=BASE/'refinement_manifest.json'
if manifest.exists():
    assert json.loads(manifest.read_text())==jobs
else:manifest.write_text(json.dumps(jobs,indent=2))
for job in jobs:
    out=BASE/job['name']/f"seed{job['seed']}"
    if (out/f"pinn_seed{job['seed']}.json").exists():continue
    cmd=[sys.executable,'-X','utf8','Benchmarks/multiscale_linear.py','--seed',str(job['seed']),'--output',str(out),*job['options']]
    with (BASE/f"{job['name']}_seed{job['seed']}.log").open('w',encoding='utf-8') as log:
        subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(job),flush=True)
