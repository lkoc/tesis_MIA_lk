"""Repeat the eight analytical verification cases with archived current sources."""
from pathlib import Path
import json,subprocess,sys,concurrent.futures
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases
ROOT=Path(__file__).resolve().parents[1]

def run(job):
    name,seed=job
    cfg=dict(cases=[name],seeds=[seed],variant='enriched',width=32,depth=3,
        adam=1200,lbfgs=800,n=768,lr=.001,pde_weight=1.,bc_weight=10.,
        flux_weight=10.,energy_weight=10.,threads=2,output='Benchmarks/verification_final')
    folder=ROOT/'Benchmarks/configurations/verification';folder.mkdir(exist_ok=True)
    p=folder/f'{name}_{seed}.json';p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    with p.with_suffix('.log').open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,'-X','utf8','Benchmarks/pinn.py','--config',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(case=name,seed=seed,exit=r.returncode)),flush=True)
    return r.returncode

if __name__=='__main__':
    jobs=[(name,seed) for name,c in cases().items() if c['kind']!='cable' for seed in [11,23,37]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit('Inspect verification logs')
