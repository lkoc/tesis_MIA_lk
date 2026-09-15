"""Declared follow-up for heterogeneous six-cable R(T) failures; retain base runs."""
from pathlib import Path
import json,subprocess,sys,concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
def run(job):
    name,seed,weight=job
    tag='coupled_refined' if weight==1000 else 'coupled_budget_control'
    cfg=dict(cases=[name],seeds=[seed],variant='multipole',width=32,depth=3,
        adam=2000,lbfgs=3000,n=768,lr=.001,pde_weight=25.,bc_weight=10.,
        flux_weight=10.,energy_weight=10.,electrical_weight=weight,limit_weight=100.,
        threads=1,coupled=True,ampacity=False,output=f'Benchmarks/comparisons/{tag}')
    folder=ROOT/'Benchmarks/configurations/coupled';folder.mkdir(exist_ok=True)
    p=folder/f'{tag}_{name}_{seed}.json';p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    with p.with_suffix('.log').open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,'-X','utf8','Benchmarks/pinn.py','--config',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(case=name,seed=seed,weight=weight,exit=r.returncode)),flush=True)
    return r.returncode
if __name__=='__main__':
    jobs=[('kim_layered',5,100)]+[(name,seed,1000) for name in ['kim_layered','kim_pac'] for seed in [11,23,37]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit('Inspect failed refinement logs')
