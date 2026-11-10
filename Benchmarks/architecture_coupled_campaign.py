"""Matched-budget width ablation for the two difficult six-cable environments."""
from pathlib import Path
import json,subprocess,sys,concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
def run(job):
    name,seed=job
    cfg=dict(cases=[name],seeds=[seed],variant='multipole',width=64,depth=3,
        adam=1200,lbfgs=1600,n=768,lr=.001,pde_weight=25.,bc_weight=10.,
        flux_weight=10.,energy_weight=10.,electrical_weight=100.,limit_weight=100.,
        threads=1,coupled=True,ampacity=False,output='Benchmarks/comparisons/coupled_width64')
    folder=ROOT/'Benchmarks/configurations/coupled';folder.mkdir(exist_ok=True)
    p=folder/f'coupled_width64_{name}_{seed}.json';p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    with p.with_suffix('.log').open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,'-X','utf8','Benchmarks/pinn.py','--config',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(case=name,seed=seed,width=64,exit=r.returncode)),flush=True)
    return r.returncode
if __name__=='__main__':
    jobs=[(name,seed) for name in ['kim_layered','kim_pac'] for seed in [11,23,37]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit('Inspect architecture logs')
