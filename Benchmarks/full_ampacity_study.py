"""Execute the preregistered DC current-limit confirmation after stages A--D."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_ampacity import ampacity_cases
from Benchmarks.full_study import read,write
BASE=ROOT/'Benchmarks/explicit_study';OUT=BASE/'ampacity'


def run(job):
    folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'job.json'
    if path.exists() and read(path)!=job:raise ValueError('Existing ampacity job changed')
    write(path,job);seed=job['seed']
    if not (folder/f'training_seed{seed}.json').exists():
        if (folder/f'pinn_seed{seed}.json').exists():raise RuntimeError('Incomplete result needs investigation before restart')
        command=[sys.executable,'-X','utf8','Benchmarks/full_ampacity.py','--case',job['case'],'--output',str(folder),'--config',job['configuration'],'--reference',job['reference'],'--seed',str(seed)]
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    r=read(folder/f'pinn_seed{seed}.json');training=read(folder/f'training_seed{seed}.json')
    row=dict(case=job['case'],seed=seed,accepted=r['accepted'],current_A=r['solution_current_A'],current_error_pct=r['current_error_pct'],Tmax_C=r['Tmax_C'],temperature_limit_error_K=r['temperature_limit_error_K'],rmse_K=r['rmse_K'],balance_pct=r['balance_pct'],parameters=training['parameters'],seconds=training['training_seconds'],path=str(folder.relative_to(ROOT)).replace('\\','/'))
    print(json.dumps(row),flush=True);return row


def main():
    if not read(BASE/'status.json')['completed']:raise RuntimeError('Finish the main registered study first')
    selected=read(BASE/'production_configuration.json');config=dict(selected['configuration'])
    config['adam']*=2;config['lbfgs']*=2
    configuration=dict(configuration=config,parent_configuration_sha256=selected['sha256'],reason='Additional unknown current and nonlinear local DC source: doubled budget registered before confirmation',frozen_utc=datetime.now(timezone.utc).isoformat())
    path=OUT/'configuration.json'
    if path.exists():
        if read(path)['configuration']!=config:raise ValueError('Frozen current-limit recipe changed')
    else:write(path,configuration)
    protocol=ROOT/'docs/PROTOCOLO_AMPACIDAD_EXPLICITA.md';frozen=OUT/'protocol.md'
    if frozen.exists() and frozen.read_bytes()!=protocol.read_bytes():raise ValueError('Current-limit protocol changed')
    if not frozen.exists():frozen.write_bytes(protocol.read_bytes())
    jobs=[]
    for name in ampacity_cases():
        reference=OUT/'references'/name;gate=read(reference/'gate.json')
        if not gate['passed']:raise RuntimeError('FEM current-limit reference rejected')
        for seed in [71,83,97]:
            jobs.append(dict(case=name,seed=seed,output=str(OUT/'confirmation'/name/f'seed{seed}'),configuration=str(path),reference=str(reference/'fem_ampacity_l2.npz')))
    manifest=OUT/'manifest.json'
    if manifest.exists():
        if read(manifest)['jobs']!=jobs:raise ValueError('Current-limit manifest changed')
    else:write(manifest,dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),interface_amendment_sha256=hashlib.sha256((BASE/'protocol_C2.md').read_bytes()).hexdigest(),jobs=jobs))
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            rows.append(future.result());write(OUT/'progress.json',rows)
    rows.sort(key=lambda r:(r['case'],r['seed']));write(OUT/'summary.json',rows)
    write(OUT/'status.json',dict(completed=True,accepted=sum(r['accepted'] for r in rows),total=len(rows)))


if __name__=='__main__':main()
