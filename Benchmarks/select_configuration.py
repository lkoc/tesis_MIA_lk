"""Select whole configurations from declared three-seed candidates, retaining failures."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases
from Benchmarks.report import records
BASE=Path(__file__).resolve().parent

def candidate(rows,directory,ampacity=False):
    rr=[r for r in rows if r['run']==directory and r['seed'] in [11,23,37]]
    if len(rr)!=3 or {r['seed'] for r in rr}!={11,23,37}:raise ValueError(f'Incomplete declared candidate: {directory}')
    return dict(directory=directory,accepted=sum(bool(r['ampacity_criteria_pass'] if ampacity else r['thermal_criteria_pass']) for r in rr),n=3,
        median_rmse_K=float(np.median([r['rmse_fem_K'] for r in rr])),std_rmse_K=float(np.std([r['rmse_fem_K'] for r in rr],ddof=1)),parameters=rr[0]['parameters'],
        max_current_error_pct=max(r['error_current_pct'] for r in rr) if ampacity else None,
        max_temperature_limit_error_K=max(r['temperature_limit_error_K'] for r in rr) if ampacity else None,
        max_electrical_residual_pct=max((r['electrical_residual_pct'] for r in rr if r['electrical_residual_pct'] is not None),default=None),
        adam=rr[0]['adam'],lbfgs=rr[0]['lbfgs'],width=rr[0]['width'],depth=rr[0]['depth'],electrical_weight=rr[0]['electrical_weight'],limit_weight=rr[0]['limit_weight'])

def choose(options,ampacity=False):
    highest=max(r['accepted'] for r in options);pool=[r for r in options if r['accepted']==highest]
    if ampacity:return min(pool,key=lambda r:(r['max_current_error_pct'],r['median_rmse_K'],r['parameters']))
    error=min(r['median_rmse_K'] for r in pool);near=[r for r in pool if r['median_rmse_K']<=1.05*error]
    return min(near,key=lambda r:(r['parameters'],r['std_rmse_K'],r['median_rmse_K']))

def main():
    rows=records();mapping={};amps={};decisions={}
    for name,c in cases().items():
        if c['kind']!='cable':
            directory=f'verification_final/{name}'
            candidate(rows,directory)
            mapping[name]=directory;continue
        directories=[f'coupled_final/{name}'];adirs=[f'ampacity_final/{name}']
        if name in ['kim_layered','kim_pac']:
            directories += [f'comparisons/coupled_refined/{name}',f'comparisons/coupled_width64/{name}']
        if name in ['kim_layered','kim_pac','xlpe_backfill','xlpe_dry_large']:
            adirs += [f'comparisons/ampacity_constraints1000/{name}']
        if name=='kim_layered':directories += [f'comparisons/adaptive_nominal_{mode}/{name}' for mode in ['uniform','residual','gradient']]
        if name=='xlpe_backfill':adirs += [f'comparisons/adaptive_ampacity_{mode}/{name}' for mode in ['uniform','residual','gradient']]
        options=[candidate(rows,d) for d in directories];ao=[candidate(rows,d,True) for d in adirs]
        selected=choose(options);aselected=choose(ao,True)
        mapping[name]=selected['directory'];amps[name]=aselected['directory']
        decisions[name]=dict(nominal_candidates=options,nominal_selected=selected['directory'],ampacity_candidates=ao,ampacity_selected=aselected['directory'])
    result=dict(criterion='Primero se maximiza la cantidad de semillas aceptadas, conservando los fallos. En operación nominal, entre candidatos dentro del 5 % del mejor RMSE mediano se elige menor cantidad de parámetros y luego menor dispersión del RMSE. En ampacidad se minimiza el peor error de corriente entre las configuraciones con más semillas aceptadas. Es una selección finita sobre casos conocidos, no una prueba independiente de generalización.',cases=mapping,ampacity_cases=amps,decisions=decisions)
    (BASE/'selection.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({n:{'nominal':mapping[n],'ampacity':amps[n]} for n in amps},indent=2))
if __name__=='__main__':main()
