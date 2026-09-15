"""Create an isolated CPU environment and rerun two representative problems."""
from pathlib import Path
import subprocess,venv,sys,json
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.venv-benchmarks-repro';out=ROOT/'Benchmarks/environment'
log=out/'clean_reproduction.log';steps=[]
with log.open('w',encoding='utf-8') as f:
    venv.EnvBuilder(with_pip=True).create(folder)
    py=folder/'Scripts/python.exe'
    commands=[
      [str(py),'-m','pip','install','torch==2.9.0','--index-url','https://download.pytorch.org/whl/cpu'],
      [str(py),'-m','pip','install','numpy==2.3.4','pytest==8.4.2'],
      [str(py),'-m','pytest','Benchmarks/tests','-q'],
      [str(py),'Benchmarks/pinn.py','--cases','mms_constant','--seeds','11','--threads','1','--output','Benchmarks/reproducibility_runs'],
      [str(py),'Benchmarks/pinn.py','--cases','xlpe_single','--seeds','11','--threads','1','--coupled','--variant','multipole','--lbfgs','1600','--output','Benchmarks/reproducibility_runs'],
    ]
    for cmd in commands:
        f.write('\nCOMMAND '+subprocess.list2cmdline(cmd)+'\n');f.flush()
        r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        steps.append({'command':cmd,'exit_code':r.returncode})
        (out/'clean_reproduction.json').write_text(json.dumps(steps,indent=2),encoding='utf-8')
        print('Completed isolated step',len(steps),'exit',r.returncode,flush=True)
        if r.returncode:raise SystemExit(r.returncode)
