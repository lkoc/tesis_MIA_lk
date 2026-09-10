"""Conditional DC current index, using the linear thermal response.

R is frozen at the stated limit temperature for every conductor. This is not
a full IEC 60287 ampacity and not an individual-cable R(T) coupled solution.
"""
from pathlib import Path
import json
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases
from Benchmarks.report import BASE,selected_directory,write_csv,latex_table,ROOT

def current_limit(c,tmax,limit=90.,tol=.0001):
    rise=tmax-c['T0']
    if rise<=0:raise ValueError('Nonpositive heating')
    factor=1+c['alpha']*(limit-20.)
    def temperature(I):return c['T0']+rise*(I/c['current'])**2*factor
    exact=c['current']*math.sqrt((limit-c['T0'])/(rise*factor))
    lo,hi=0.,c['current'];history=[]
    while temperature(hi)<limit:hi*=2
    for iteration in range(100):
        mid=(lo+hi)/2;T=temperature(mid)
        history.append(dict(iteration=iteration,current_A=mid,T_C=T))
        if abs(T-limit)<tol:break
        if T>limit:hi=mid
        else:lo=mid
    assert abs(mid-exact)<=max(1e-3,exact*1e-6)
    return dict(current_A=mid,closed_form_A=exact,T_limit_C=limit,resistance_factor=factor,thermal_residual_K=T-limit,history=history)

def main():
    rows=[];all_data={}
    for name,c in cases().items():
        if c['kind']!='cable':continue
        fm=json.loads((BASE/'results'/name/'fem_l2.json').read_text())
        f=current_limit(c,fm['Tmax_C']);pins=[]
        for seed in [11,23,37]:
            path=selected_directory(name)/f'pinn_seed{seed}.json'
            if path.exists():
                p=json.loads(path.read_text())
                if p['thermal_criteria_pass']:pins.append(dict(seed=seed,**current_limit(c,p['Tmax_C'])))
        all_data[name]=dict(case=c,assumption='R fixed at 90 C for all conductors; DC losses only; linear thermal scaling',fem=f,pinn=pins)
        rows.append(dict(case=name,pair=c.get('pair'),fem_A=f['current_A'],pinn_mean_A=sum(p['current_A'] for p in pins)/len(pins) if pins else None,accepted_seeds=len(pins),delta_fem_pct=None))
    lookup={r['case']:r for r in rows}
    for r in rows:
        if r['pair']:r['delta_fem_pct']=100*(r['fem_A']/lookup[r['pair']]['fem_A']-1)
    out=BASE/'summary';out.mkdir(exist_ok=True)
    write_csv(out/'ampacity_conditional.csv',rows)
    (out/'ampacity_history.json').write_text(json.dumps(all_data,ensure_ascii=False,indent=2),encoding='utf-8')
    latex_table(ROOT/'Tesis_LaTeX_Borrador_UNI/tablas/benchmark_ampacity.tex',['Caso','$I_{90}^{FEM}$ (A)','$I_{90}^{PINN}$ (A)','$\Delta I_{FEM}$ (\%)'],[[r['case'],f"{r['fem_A']:.1f}",f"{r['pinn_mean_A']:.1f}" if r['pinn_mean_A'] is not None else 'Rechazado',f"{r['delta_fem_pct']:.2f}" if r['delta_fem_pct'] is not None else '---'] for r in rows],'Índice de corriente continua condicionado a resistencia uniforme a 90 °C','tab:benchmark-ampacidad')
    print('Conditional DC indices:',len(rows))

if __name__=='__main__':main()
