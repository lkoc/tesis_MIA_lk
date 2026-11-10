"""Retrain two selected configurations without overwriting thesis evidence."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'Benchmarks'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    selection=json.loads((BASE/'selection.json').read_text())
    # Fixed before retraining. These reproduce old runs, not confirm new seeds.
    tolerance_K=0.02
    rows=[]
    for name in ['mms_interface','xlpe_single']:
        old=BASE/selection['cases'][name]/'pinn_seed11.json'
        previous=json.loads(old.read_text())
        config=dict(previous['metadata']['configuration'])
        config.update(cases=[name],seeds=[11],output=str(OUT/'retrained'))
        path=OUT/(name+'_repeat_configuration.json')
        path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        command=[sys.executable,'-X','utf8',str(BASE/'pinn.py'),'--config',str(path)]
        print('Retraining '+name,flush=True)
        start=time.perf_counter()
        with (OUT/(name+'_repeat.log')).open('w',encoding='utf-8') as stream:
            completed=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        row=dict(case=name,command=command,configuration_sha256=sha(path),exit_code=completed.returncode,
            wall_seconds=time.perf_counter()-start,previous_record_sha256=sha(old),tolerance_K=tolerance_K)
        if completed.returncode==0:
            new=OUT/'retrained'/name/'pinn_seed11.json'
            current=json.loads(new.read_text())
            with np.load(old.with_suffix('.npz')) as a,np.load(new.with_suffix('.npz')) as b:
                assert np.array_equal(a['xy'],b['xy'])
                row['max_field_difference_K']=float(np.max(abs(a['T']-b['T'])))
            row.update(Tmax_difference_K=abs(previous['Tmax_C']-current['Tmax_C']),
                current_record_sha256=sha(new),thermal_criteria_pass=current['thermal_criteria_pass'],
                previous_Tmax_C=previous['Tmax_C'],new_Tmax_C=current['Tmax_C'],
                rmse_fem_K=current['rmse_fem_K'],balance_pct=current['balance_pct'])
            row['repeat_pass']=bool(row['max_field_difference_K']<=tolerance_K and row['Tmax_difference_K']<=tolerance_K and row['thermal_criteria_pass'])
        else:row['repeat_pass']=False
        rows.append(row)
        (OUT/'repeat_evidence.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),
            scope='Same-seed full retraining with current source and archived configuration. Independent confirmation seeds and explicit-layer validation remain separate.',
            results=rows),indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    if not all(row['repeat_pass'] for row in rows):raise SystemExit(1)

if __name__=='__main__':main()
