"""Run the declared R(T) campaign; each subprocess writes an independent case/seed."""
from pathlib import Path
import sys,json,subprocess,concurrent.futures
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases
ROOT=Path(__file__).resolve().parents[1]

def run(job):
    name,seed,mode=job
    cfg=dict(cases=[name],seeds=[seed],variant='multipole',width=32,depth=3,
        adam=1200,lbfgs=1600,n=768,lr=.001,pde_weight=25.,bc_weight=10.,
        flux_weight=10.,energy_weight=10.,threads=1,coupled=True,
        ampacity=mode=='ampacity',temperature_limit=90.,output=f'Benchmarks/{mode}_final')
    folder=ROOT/'Benchmarks/configurations/coupled';folder.mkdir(exist_ok=True)
    p=folder/f'{mode}_{name}_{seed}.json';p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    log=p.with_suffix('.log')
    with log.open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,'-X','utf8','Benchmarks/pinn.py','--config',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(case=name,seed=seed,mode=mode,exit=r.returncode)),flush=True)
    return r.returncode

if __name__=='__main__':
    jobs=[(name,seed,mode) for mode in ['coupled','ampacity'] for name,c in cases().items() if c['kind']=='cable' for seed in [11,23,37]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit('Some executions failed; inspect configuration logs')
