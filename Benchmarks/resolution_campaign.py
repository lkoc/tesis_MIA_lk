"""Matched studies of PINN width, collocation density and optimization budget."""
from pathlib import Path
import json,subprocess,sys,concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
CONFIGS={'W16':dict(width=16),'W32':dict(width=32),'W64':dict(width=64),
    'N384':dict(n=384),'N1536':dict(n=1536),'B2':dict(budget_factor=2)}

def run(job):
    name,label,seed=job;cable=name=='xlpe_single';option=dict(CONFIGS[label]);factor=option.pop('budget_factor',1)
    cfg=dict(cases=[name],seeds=[seed],variant='multipole' if cable else 'enriched',width=32,depth=3,
        adam=1200*factor,lbfgs=(1600 if cable else 800)*factor,n=768,lr=.001,
        pde_weight=25. if cable else 1.,bc_weight=10.,flux_weight=10.,energy_weight=10.,
        electrical_weight=100.,limit_weight=100.,coupled=cable,ampacity=False,threads=1,
        output='Benchmarks/comparisons/resolution_'+label)
    cfg.update(option)
    folder=ROOT/'Benchmarks/configurations/resolution';folder.mkdir(exist_ok=True)
    p=folder/f'{name}_{label}_{seed}.json';p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    with p.with_suffix('.log').open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,'-X','utf8','Benchmarks/pinn.py','--config',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(case=name,configuration=label,seed=seed,exit=r.returncode)),flush=True)
    return r.returncode

if __name__=='__main__':
    jobs=[(name,label,seed) for name in ['mms_smooth_2d','xlpe_single'] for label in CONFIGS for seed in [11,23,37]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit('Inspect resolution study logs')
