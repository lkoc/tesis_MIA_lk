"""Staged, restartable explicit-physics study; no held-out tuning or FEM labels."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,copy,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_paths import artifact_path
BASE=ROOT/'Benchmarks/explicit_study'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def linux(path):return '/mnt/'+path.drive[0].lower()+'/'+('/'.join(path.resolve().parts[1:]))

def catalogue():
    from Benchmarks.cases import cases
    from Benchmarks.full_domain import coaxial_case
    result=cases();c=coaxial_case();c.update(id='coaxial_angular',outer_gradient_K_m=10.)
    result[c['id']]=c;result['coaxial_full']=coaxial_case();return result

def fem_worker(path,mode,output):
    import numpy as np
    from Benchmarks.full_fem import solve
    from Benchmarks.full_domain import FullDomain
    d=FullDomain(read(path),mode);records=[];previous=None;gate=None
    for level in range(5):
        meta=output/f'fem_l{level}.json'
        result=read(meta) if meta.exists() else solve(d,level,output)
        if result['physics_sha256']!=d.fingerprint:raise ValueError('Stale FEM physics')
        with np.load(meta.with_suffix('.npz')) as data:T=data['T'];labels=data['region']
        record=dict(level=level,Tmax_C=result['Tmax_C'],balance_pct=result['balance_pct'],ndofs=result['ndofs'])
        if previous is not None:
            delta=max(abs(result['Tmax_C']-d.case['T0']),1.)
            record['Tmax_change_K']=abs(result['Tmax_C']-records[-1]['Tmax_C'])
            record['max_region_rmse_change_K']=max(float(np.sqrt(np.mean((T[labels==r.id]-previous[labels==r.id])**2))) for r in d.regions)
            passed=level>=2 and record['Tmax_change_K']<=min(.1,.005*delta) and record['max_region_rmse_change_K']<=.005*delta and result['balance_pct']<=.2
            if passed:gate=dict(passed=True,reference=str(meta.with_suffix('.npz').name),physics_sha256=d.fingerprint)
        records.append(record);previous=T.copy()
        if gate:break
    gate=gate or dict(passed=False);gate['levels']=records;write(output/'gate.json',gate)
    if not gate['passed']:raise RuntimeError('FEM refinement did not satisfy the registered gate')

def reference(name,mode='fixed'):
    folder=BASE/'references'/mode/name;casepath=BASE/'cases'/f'{name}.json'
    if not casepath.exists():write(casepath,catalogue()[name])
    if not (folder/'gate.json').exists():
        folder.mkdir(parents=True,exist_ok=True)
        command=['wsl','-d','Ubuntu','--','env','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
            '/home/lkoc/miniforge3/envs/fenicsx/bin/python',linux(Path(__file__)),'fem',linux(casepath),mode,linux(folder)]
        with (folder/'run.log').open('w',encoding='utf-8') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    gate=read(folder/'gate.json')
    if not gate['passed']:raise RuntimeError('Reference rejected')
    return folder/gate['reference']

def worker(jobpath):
    from Benchmarks.full_domain import FullDomain
    from Benchmarks.full_pinn import train
    job=read(jobpath);train(FullDomain(read(Path(job['case_file'])),job['mode']),job['config'],Path(job['output']))

def metrics(folder,seed):
    result=read(folder/f'pinn_seed{seed}.json');training=read(folder/f'training_seed{seed}.json')
    ref=read(artifact_path(training['configuration']['reference']).with_suffix('.json'))
    elevation=abs(ref['Tmax_C']-training['configuration']['physics']['case']['T0'])
    ratio=[result.get('nrmse_pct',1e6)/5,result.get('error_Tmax_pct',1e6)/5,result['balance_pct']/2,result['derived_balance_pct']/2,
        max(result['region_energy_error_pct'].values())/2,max(result['region_derived_energy_error_pct'].values())/2,
        max(i['T_jump_max_K'] for i in result['interfaces'])/.1,max(i['derived_flux_jump_rms_pct'] for i in result['interfaces']),
        max(i['flux_jump_rms_pct'] for i in result['interfaces']),max(result['region_rmse_K'].values())/(.05*elevation)]
    return dict(accepted=result['accepted'],violation=max(ratio),parameters=training['parameters'],seconds=training['training_seconds'],
        rmse_K=result.get('rmse_K'),Tmax_error_K=result.get('error_Tmax_K'),nrmse_pct=result.get('nrmse_pct'),Tmax_C=result['Tmax_C'],
        balance_pct=result['balance_pct'],path=str(folder.relative_to(ROOT)).replace('\\','/'))

def run_one(job):
    output=Path(job['output']);seed=job['config']['seed'];output.mkdir(parents=True,exist_ok=True)
    path=output/'job.json'
    if path.exists() and read(path)!=job:raise ValueError('Job identity changed in an existing directory')
    write(path,job)
    if not (output/f'pinn_seed{seed}.json').exists():
        environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (output/'run.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)),'worker',str(path)],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Worker failed; inspect {output / "run.log"}')
    row=dict(stage=job['stage'],candidate=job['candidate'],case=job['case'],seed=seed,**metrics(output,seed))
    print(json.dumps(row),flush=True);return row

def run_stage(stage,configs,names,seeds=(11,23),mode='fixed'):
    if (BASE/stage/'summary.json').exists():
        rows=read(BASE/stage/'summary.json')
        for row in rows:row.update(metrics(ROOT/row['path'],row['seed']))
        write(BASE/stage/'summary.json',rows);return rows
    references={name:reference(name,mode) for name in names};jobs=[]
    for candidate,config in configs.items():
        for name in names:
            for index,seed in enumerate(seeds):
                parameters=dict(config,seed=seed,sampling_seed=([1001,1003,1007] if stage=='D' else [101,103])[index],threads=1,reference=str(references[name]))
                jobs.append(dict(stage=stage,candidate=candidate,case=name,mode=mode,case_file=str(BASE/'cases'/f'{name}.json'),config=parameters,output=str(BASE/stage/candidate/name/f'seed{seed}')))
    manifest=BASE/stage/'manifest.json'
    if manifest.exists():
        if read(manifest)['jobs']!=jobs:raise ValueError('Registered stage jobs differ from the requested configuration')
    else:write(manifest,dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs))
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_one,j) for j in jobs]
        for future in as_completed(futures):
            rows.append(future.result());write(BASE/stage/'progress.json',rows)
    rows.sort(key=lambda r:(r['candidate'],r['case'],r['seed']));write(BASE/stage/'summary.json',rows);return rows

def select(rows,configs,prefer='parameters'):
    ranking=[]
    for name,config in configs.items():
        group=[r for r in rows if r['candidate']==name];passed=sum(r['accepted'] for r in group)
        budget=config['n']+4*config['n_layer']+4*config['n_interface']
        ranking.append(dict(candidate=name,accepted=passed,total=len(group),robust=passed==len(group),
            violation=max(r['violation'] for r in group),parameters=max(r['parameters'] for r in group),budget=budget,temperature_weight=config.get('temperature_weight',100)))
    def key(r):
        if r['robust']:return (0,r[prefer] if prefer in r else r['violation'],r['violation'])
        return (1,-r['accepted'],r['violation'])
    ranking.sort(key=key);winner=ranking[0]
    return dict(winner=winner['candidate'],qualified=winner['robust'],ranking=ranking,configuration=configs[winner['candidate']])

def main(stop_after):
    BASE.mkdir(exist_ok=True)
    protocol=ROOT/'docs/PROTOCOLO_EXPERIMENTAL_EXPLICITO.md'
    frozen=BASE/'protocol.md'
    if not frozen.exists():frozen.write_bytes(protocol.read_bytes())
    elif frozen.read_bytes()!=protocol.read_bytes():raise ValueError('Registered protocol changed')
    base=dict(n=256,n_layer=128,n_interface=96,adam=1200,lbfgs=2000,lr=.001,sampling='mixed')
    specifications=[('subdomain',8,2),('subdomain',16,2),('subdomain',16,3),('subdomain',32,3),('mixed',16,2),('mixed',32,3),('local',16,2),('fourier',16,2),('multipole',16,2)]
    configs={f'{v}_w{w}_d{d}':dict(base,variant=v,width=w,depth=d) for v,w,d in specifications}
    rows=run_stage('A',configs,['coaxial_angular','xlpe_single']);choice=select(rows,configs);write(BASE/'A/selection.json',choice)
    if stop_after=='A':return
    selected=choice['configuration'];configs={}
    for factor in [1,2,4]:configs[f'mixed_{factor}x']=dict(selected,n=256*factor,n_layer=128*factor,n_interface=96*factor,sampling='mixed')
    for policy in ['uniform','interface','residual']:configs[policy+'_2x']=dict(selected,n=512,n_layer=256,n_interface=192,sampling=policy)
    rows=run_stage('B',configs,['xlpe_single','xlpe_discrete_layers']);choice=select(rows,configs,'budget');write(BASE/'B/selection.json',choice)
    if stop_after=='B':return
    selected=choice['configuration'];configs={f'lr_{lr:g}':dict(selected,lr=lr,adam=1500,lbfgs=3000) for lr in [.0005,.001,.002]}
    rows=run_stage('C',configs,['xlpe_single','xlpe_discrete_layers']);choice=select(rows,configs,'violation');write(BASE/'C/selection.json',choice)
    if stop_after=='C':return
    amendment=ROOT/'docs/PROTOCOLO_CALIBRACION_INTERFACES.md';frozen_amendment=BASE/'protocol_C2.md'
    if frozen_amendment.exists() and frozen_amendment.read_bytes()!=amendment.read_bytes():raise ValueError('Registered C2 amendment changed')
    if not frozen_amendment.exists():frozen_amendment.write_bytes(amendment.read_bytes())
    selected=choice['configuration'];baseline=[dict(r,stage='C2',candidate='weight_100') for r in rows if r['candidate']==choice['winner']]
    configs={f'weight_{weight}':dict(selected,temperature_weight=weight) for weight in [1000,10000]}
    added=run_stage('C2',configs,['xlpe_single','xlpe_discrete_layers'])
    configs['weight_100']=dict(selected,temperature_weight=100)
    choice=select(baseline+added,configs,'temperature_weight');choice.update(baseline_reused_from='C/'+read(BASE/'C/selection.json')['winner'],new_runs=len(added),baseline_reused=len(baseline))
    write(BASE/'C2/selection.json',choice)
    if stop_after=='C2':return
    frozen=dict(choice,sha256=hashlib.sha256(json.dumps(choice['configuration'],sort_keys=True).encode()).hexdigest(),frozen_before_confirmation_utc=datetime.now(timezone.utc).isoformat())
    if not (BASE/'production_configuration.json').exists():write(BASE/'production_configuration.json',frozen)
    elif read(BASE/'production_configuration.json')['configuration']!=choice['configuration']:raise ValueError('A previous frozen production recipe differs from the new selection')
    rows=run_stage('D',{'production':choice['configuration']},['xlpe_single','xlpe_discrete_layers','xlpe_backfill','xlpe_dry_near','xlpe_dry_far','aras_flat','kim_sand'],(71,83,97))
    write(BASE/'status.json',dict(completed=True,production_accepted=sum(r['accepted'] for r in rows),production_total=len(rows)))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(Path(sys.argv[2]))
    elif len(sys.argv)>1 and sys.argv[1]=='fem':fem_worker(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]))
    else:
        parser=argparse.ArgumentParser();parser.add_argument('--stop-after',choices=['A','B','C','C2','D'],default='D');main(parser.parse_args().stop_after)
