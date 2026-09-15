"""Equal-budget adaptive sampling and random-refresh controls, three seeds."""
from pathlib import Path
import json,subprocess,sys,concurrent.futures
ROOT=Path(__file__).resolve().parents[1]

def run(job):
    name,mode,seed=job;amp=name=='xlpe_backfill';family='ampacity' if amp else 'nominal'
    cfg=dict(cases=[name],seeds=[seed],variant='multipole',width=32,depth=3,
        adam=1200,lbfgs=1600,n=768,lr=.001,pde_weight=25.,bc_weight=10.,flux_weight=10.,energy_weight=10.,
        electrical_weight=1000. if amp else 100.,limit_weight=1000. if amp else 100.,
        coupled=True,ampacity=amp,threads=1,sampling=mode,adapt_steps=[400,800],candidate_multiplier=4,adaptive_fraction=.5,
        output=f'Benchmarks/comparisons/adaptive_{family}_{mode}')
    folder=ROOT/'Benchmarks/configurations/adaptive';folder.mkdir(exist_ok=True)
    p=folder/f'{name}_{mode}_{seed}.json';p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    with p.with_suffix('.log').open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,'-X','utf8','Benchmarks/pinn.py','--config',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(case=name,sampling=mode,seed=seed,exit=r.returncode)),flush=True)
    return r.returncode

if __name__=='__main__':
    jobs=[(name,mode,seed) for name in ['kim_layered','xlpe_backfill'] for mode in ['uniform','residual','gradient'] for seed in [11,23,37]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit('Inspect adaptive study logs')
