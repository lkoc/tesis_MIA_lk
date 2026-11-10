"""Finite, declared comparison; every candidate and failure is retained."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def configurations():
    base=dict(cases=['xlpe_single'],seeds=[5],variant='conservative',width=32,depth=3,
              adam=1200,lbfgs=800,n=768,lr=.001,pde_weight=25.,bc_weight=10.,
              flux_weight=10.,energy_weight=10.,threads=2)
    changes={
        'C01_enriched':dict(variant='enriched',pde_weight=1.),
        'C02_energy':dict(pde_weight=1.),
        'C03_reference':{},
        'C04_width16':dict(width=16),
        'C05_width64':dict(width=64),
        'C06_depth4':dict(depth=4),
        'C07_lr0005':dict(lr=.0005),
        'C08_sampling1536':dict(n=1536),
    }
    return {name:dict(base,**change) for name,change in changes.items()}

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--write-only',action='store_true');args=ap.parse_args()
    folder=ROOT/'Benchmarks/configurations';folder.mkdir(exist_ok=True)
    for name,config in configurations().items():
        config['output']=f'Benchmarks/comparisons/{name}'
        path=folder/f'{name}.json'
        if not path.exists(): path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
        if args.write_only: continue
        if (ROOT/config['output']/'xlpe_single/pinn_seed5.json').exists(): continue
        print(name,flush=True)
        subprocess.run([sys.executable,'Benchmarks/pinn.py','--config',str(path)],cwd=ROOT,check=True)

if __name__=='__main__': main()
