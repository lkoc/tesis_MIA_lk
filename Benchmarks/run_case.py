"""Active workflow: explicit conductor, every cable layer and soil, in SI units."""
from pathlib import Path
import argparse,json,os,platform,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases
from Benchmarks.full_domain import FullDomain,coaxial_case
from Benchmarks.full_pinn import VARIANTS,train
ROOT=Path(__file__).resolve().parents[1]
def linux_path(p):
    p=Path(p).resolve()
    return '/mnt/'+p.drive[0].lower()+'/'+('/'.join(p.parts[1:])) if platform.system()=='Windows' else str(p)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('case');ap.add_argument('--case-file',type=Path)
    ap.add_argument('--mode',choices=['auto','verification','nominal','ampacity'],default='auto')
    ap.add_argument('--variant',choices=VARIANTS,default='subdomain')
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--levels',nargs='+',type=int,default=[0,1,2])
    ap.add_argument('--n',type=int,default=512);ap.add_argument('--n-layer',type=int,default=256);ap.add_argument('--n-interface',type=int,default=128)
    ap.add_argument('--adam',type=int,default=1500);ap.add_argument('--lbfgs',type=int,default=2000)
    ap.add_argument('--width',type=int,default=32);ap.add_argument('--depth',type=int,default=3);ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--sampling',choices=['mixed','uniform','interface','residual'],default='mixed');ap.add_argument('--limit',type=float,default=90.)
    ap.add_argument('--temperature-weight',type=float,default=100.)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--fenics-python',default='/home/lkoc/miniforge3/envs/fenicsx/bin/python')
    a=ap.parse_args()
    c=json.loads(a.case_file.read_text(encoding='utf-8')) if a.case_file else (coaxial_case() if a.case=='coaxial_full' else cases()[a.case])
    if c['kind']!='cable':ap.error('Manufactured soil tests retain fem.py and pinn.py; this workflow resolves explicit cables')
    if len(set(a.levels))<3:ap.error('At least three distinct FEM refinement levels are required')
    mode='fixed' if a.mode=='verification' or (a.mode=='auto' and c.get('electrical',{}).get('frequency_Hz',0)>0) else 'dc_temperature'
    if a.mode=='ampacity':
        from Benchmarks.full_ampacity import AmpacityDomain,train_ampacity
        domain=AmpacityDomain(c,a.limit);mode='dc_temperature'
    else:domain=FullDomain(c,mode)
    output=a.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a new output directory; evidence is immutable')
    output.mkdir(parents=True,exist_ok=True)
    case_path=output/'case.json';case_path.write_text(json.dumps(c,indent=2),encoding='utf-8')
    (output/'workflow.json').write_text(json.dumps(dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},physics_sha256=domain.fingerprint,source_mode=mode,workers=1,precision='float64'),indent=2),encoding='utf-8')
    os.environ['OMP_NUM_THREADS']=str(a.threads);os.environ['MKL_NUM_THREADS']=str(a.threads)
    reference=output/'reference'
    command=[a.fenics_python,linux_path(ROOT/'Benchmarks/full_fem.py'),'--case-file',linux_path(case_path),'--mode',mode,'--levels',*[str(v) for v in a.levels],'--output',linux_path(reference)]
    if a.mode=='ampacity':command=[a.fenics_python,linux_path(ROOT/'Benchmarks/full_ampacity.py'),'--fem','--case-file',linux_path(case_path),'--limit',str(a.limit),'--levels',*[str(v) for v in a.levels],'--output',linux_path(reference)]
    if platform.system()=='Windows':command=['wsl','-d','Ubuntu','--','env',f'OMP_NUM_THREADS={a.threads}',f'MKL_NUM_THREADS={a.threads}',f'OPENBLAS_NUM_THREADS={a.threads}']+command
    print(f'Case {a.case}; mode {mode}; FEM and PINN; output {output}',flush=True)
    subprocess.run(command,cwd=ROOT,check=True)
    config=dict(variant=a.variant,width=a.width,depth=a.depth,lr=a.lr,sampling=a.sampling,temperature_weight=a.temperature_weight,n=a.n,n_layer=a.n_layer,n_interface=a.n_interface,adam=a.adam,lbfgs=a.lbfgs,threads=a.threads)
    if a.mode=='ampacity':
        reports=[train_ampacity(c,dict(config,seed=seed,sampling_seed=seed),reference/f'fem_ampacity_l{max(a.levels)}.npz',output/a.variant/f'seed{seed}',a.limit) for seed in a.seeds]
        (output/'summary.json').write_text(json.dumps(dict(accepted=all(r['accepted'] for r in reports),runs=reports),indent=2),encoding='utf-8')
        if not all(r['accepted'] for r in reports):raise SystemExit(2)
        return
    records=[json.loads((reference/f'fem_l{level}.json').read_text()) for level in sorted(set(a.levels))]
    with np.load(reference/f"fem_l{records[-1]['level']}.npz") as fine,np.load(reference/f"fem_l{records[-2]['level']}.npz") as previous:
        region_change=max(float(np.sqrt(np.mean((fine['T'][fine['region']==r.id]-previous['T'][previous['region']==r.id])**2))) for r in domain.regions)
    rise=abs(records[-1]['Tmax_C']-c['T0'])
    stable=records[-1]['balance_pct']<=.2 and abs(records[-1]['Tmax_C']-records[-2]['Tmax_C'])<=min(.1,.005*rise) and region_change<=.005*rise
    (output/'mesh_gate.json').write_text(json.dumps(dict(passed=stable,levels=records,max_region_rmse_change_K=region_change,criteria='three levels; last Tmax change <= min(0.1 K, 0.5% rise); every region RMSE change <= 0.5% rise; balance <= 0.2%'),indent=2),encoding='utf-8')
    if not stable:raise RuntimeError('FEM refinement gate failed; refine before comparing PINNs')
    config['reference']=str(reference/f'fem_l{max(a.levels)}.npz')
    reports=[train(domain,dict(config,seed=seed),output/a.variant/f'seed{seed}') for seed in a.seeds]
    (output/'summary.json').write_text(json.dumps(dict(accepted=all(r['accepted'] for r in reports),runs=reports),indent=2),encoding='utf-8')
    if not all(r['accepted'] for r in reports):raise SystemExit(2)
if __name__=='__main__':main()
