"""Two fixed exploratory transfer cases, with no tuning on their outcomes."""
from pathlib import Path
import json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Benchmarks/multiscale_exploration'
jobs=[dict(case=case,seed=seed) for case in ['coaxial_angular','xlpe_dry_near'] for seed in [11,23]]
manifest=BASE/'transfer_manifest.json'
if manifest.exists():assert json.loads(manifest.read_text())==jobs
else:manifest.write_text(json.dumps(jobs,indent=2))
for job in jobs:
    out=BASE/'transfer'/job['case']/f"seed{job['seed']}"
    if (out/f"pinn_seed{job['seed']}.json").exists():continue
    cmd=[sys.executable,'-X','utf8','Benchmarks/multiscale_linear.py','--seed',str(job['seed']),
         '--case',job['case'],'--output',str(out)]
    with (BASE/f"transfer_{job['case']}_seed{job['seed']}.log").open('w',encoding='utf-8') as log:
        subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(job),flush=True)
