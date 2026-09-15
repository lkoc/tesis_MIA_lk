"""Matched-budget enforcement of electrical and temperature-limit constraints."""
from pathlib import Path
import json,subprocess,sys,concurrent.futures,argparse
ROOT=Path(__file__).resolve().parents[1]
def run(job):
    name,seed=job
    cfg=dict(cases=[name],seeds=[seed],variant='multipole',width=32,depth=3,
        adam=1200,lbfgs=1600,n=768,lr=.001,pde_weight=25.,bc_weight=10.,
        flux_weight=10.,energy_weight=10.,electrical_weight=1000.,limit_weight=1000.,
        threads=1,coupled=True,ampacity=True,temperature_limit=90.,output='Benchmarks/comparisons/ampacity_constraints1000')
    folder=ROOT/'Benchmarks/configurations/coupled';folder.mkdir(exist_ok=True)
    p=folder/f'ampacity_constraints1000_{name}_{seed}.json';p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    with p.with_suffix('.log').open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,'-X','utf8','Benchmarks/pinn.py','--config',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(case=name,seed=seed,constraint_weights=1000,exit=r.returncode)),flush=True)
    return r.returncode
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['kim_layered','kim_pac','xlpe_backfill','xlpe_dry_large'])
    args=parser.parse_args()
    jobs=[(name,seed) for name in args.cases for seed in [11,23,37]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit('Inspect ampacity refinement logs')
