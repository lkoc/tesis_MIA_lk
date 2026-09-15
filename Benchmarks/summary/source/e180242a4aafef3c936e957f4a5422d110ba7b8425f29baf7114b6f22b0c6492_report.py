"""Generate tables and figures exclusively from saved execution artifacts."""
from pathlib import Path
import csv
import json
import sys
import re
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases,fingerprint,field_k
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Benchmarks'

def records():
    result=[];catalog=cases()
    for p in sorted(BASE.rglob('pinn_seed*.json')):
        if 'source' in p.parts:continue
        r=json.loads(p.read_text(encoding='utf-8'));m=r['metadata']
        if m['case_sha256']!=fingerprint(catalog[m['case']['id']]):raise ValueError(f'Stale physical data: {p}')
        row=dict(case=m['case']['id'],run=str(p.parent.relative_to(BASE)).replace('\\','/'),mode='ampacity' if m.get('ampacity') else 'coupled' if m.get('coupled') else 'fixed_source',seed=m['seed'],variant=m.get('variant','enriched'),width=m['width'],depth=m['depth'],parameters=m['parameters'],seconds=m['training_seconds'])
        row.update(adam=m['adam'],lbfgs=m['lbfgs_max_iter'],n_interior=m['n_interior'],electrical_weight=m.get('configuration',{}).get('electrical_weight',100.) if m.get('coupled') else None,limit_weight=m.get('configuration',{}).get('limit_weight',100.) if m.get('ampacity') else None)
        for key in ['Tmax_C','Tmax_fem_C','rmse_fem_K','nrmse_fem_pct','max_error_fem_K','error_Tmax_K','error_Tmax_rise_pct','balance_pct','pde_rmse_W_m3','thermal_criteria_pass','rmse_exact_K','current_A','current_fem_A','error_current_pct','electrical_residual_pct','temperature_limit_error_K','ampacity_criteria_pass']:
            row[key]=r.get(key)
        row['path']=str(p.relative_to(ROOT)).replace('\\','/');result.append(row)
    return result

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def selected_directory(name):
    selection=BASE/'selection.json'
    if selection.exists():
        m=json.loads(selection.read_text(encoding='utf-8'))
        if name in m['cases']:return BASE/m['cases'][name]
    return BASE/'results'/name

def selected_ampacity_directory(name):
    selection=BASE/'selection.json'
    if selection.exists():
        m=json.loads(selection.read_text(encoding='utf-8'))
        if name in m.get('ampacity_cases',{}):return BASE/m['ampacity_cases'][name]
    return BASE/'ampacity_final'/name

def plot_case(name,save=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=cases()[name]
    paths=sorted(selected_directory(name).glob('pinn_seed*.npz'))
    if not paths:raise FileNotFoundError(f'No PINN results for {name}')
    meta=json.loads(paths[0].with_suffix('.json').read_text())['metadata']
    f=np.load(BASE/('coupled_results' if meta.get('coupled') else 'results')/name/('fem_ampacity_l2.npz' if meta.get('ampacity') else 'fem_l2.npz'))
    p=np.load(paths[0]);xy=f['xy'];v=f['T'];pred=p['T'];err=pred-v
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    lo=min(v.min(),pred.min());hi=max(v.max(),pred.max())
    for ax,z,title in zip(axes,[v,pred,err],['FEniCSx P2','PINN, semilla '+paths[0].stem.split('seed')[-1],'PINN − FEM']):
        opt=dict(cmap='coolwarm',vmin=-max(abs(err)),vmax=max(abs(err))) if ax is axes[2] else dict(cmap='inferno',vmin=lo,vmax=hi)
        sc=ax.scatter(xy[:,0],xy[:,1],c=z,s=3,rasterized=True,**opt)
        ax.set(title=title,xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
        fig.colorbar(sc,ax=ax,label='K' if ax is axes[2] else '°C',shrink=.8)
    fig.suptitle(name.replace('_',' '))
    if save:fig.savefig(save,dpi=160,bbox_inches='tight')
    return fig

def latex_table(path,headers,rows,caption,label):
    def esc(s):return re.sub(r'(?<=\d)\.(?=\d)',',',str(s)).replace('_',r'\_\allowbreak ').replace('%',r'\%')
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+r'. Fuente: Elaboración propia a partir de los registros de Benchmarks.}',r'\label{'+label+'}',r'\begin{tabularx}{\textwidth}{Y'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    lines.extend(' & '.join(esc(v) for v in row)+r' \\' for row in rows)
    lines.extend([r'\bottomrule',r'\end{tabularx}',r'\end{table}'])
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    out=BASE/'summary';out.mkdir(exist_ok=True)
    rows=records();write_csv(out/'all_runs.csv',rows)
    meshes=[]
    for name,c in cases().items():
        data=[json.loads((BASE/'results'/name/f'fem_l{i}.json').read_text()) for i in range(3)]
        f1=np.load(BASE/'results'/name/'fem_l1.npz');f2=np.load(BASE/'results'/name/'fem_l2.npz')
        delta=data[-1]['Tmax_C']-c['T0']
        meshes.append(dict(case=name,ndofs_l0=data[0]['ndofs'],ndofs_l1=data[1]['ndofs'],ndofs_l2=data[2]['ndofs'],Tmax_C=data[2]['Tmax_C'],change_Tmax_K=abs(data[2]['Tmax_C']-data[1]['Tmax_C']),change_field_K=float(np.sqrt(np.mean((f2['T']-f1['T'])**2))),balance_pct=data[2]['balance_pct'],converged=abs(data[2]['Tmax_C']-data[1]['Tmax_C'])/delta<=.005 and data[2]['balance_pct']<=2))
    write_csv(out/'fem_convergence.csv',meshes)
    figures=ROOT/'Tesis_LaTeX_Borrador_UNI/imagenes/benchmarks';figures.mkdir(parents=True,exist_ok=True)
    tables=ROOT/'Tesis_LaTeX_Borrador_UNI/tablas';tables.mkdir(exist_ok=True)
    latex_table(tables/'benchmark_fem.tex',['Caso','$N_2$',r'$T_{\max}$ (°C)',r'$|\Delta T|$ (K)',r'$E_b$ (\%)'],[[r['case'],r['ndofs_l2'],f"{r['Tmax_C']:.3f}",f"{r['change_Tmax_K']:.4f}",f"{r['balance_pct']:.3f}"] for r in meshes],'Convergencia de las referencias FEM entre las dos mallas más finas','tab:benchmark-fem')
    mainrows=[]
    for name in cases():
        directory=str(selected_directory(name).relative_to(BASE)).replace('\\','/')
        chosen=[r for r in rows if r['case']==name and r['run']==directory and r['seed'] in [11,23,37]]
        if not chosen:continue
        mainrows.append(dict(case=name,run=directory,n=len(chosen),accepted=sum(r['thermal_criteria_pass'] for r in chosen),Tmax_median_C=float(np.median([r['Tmax_C'] for r in chosen])),rmse_median_K=float(np.median([r['rmse_fem_K'] for r in chosen])),rmse_std_K=float(np.std([r['rmse_fem_K'] for r in chosen],ddof=1)) if len(chosen)>1 else 0.,Tmax_error_max_K=max(abs(r['error_Tmax_K']) for r in chosen),balance_max_pct=max(r['balance_pct'] for r in chosen)))
        fig=plot_case(name,figures/f'{name}.png')
        import matplotlib.pyplot as plt
        plt.close(fig)
    write_csv(out/'selected_runs.csv',mainrows)
    latex_table(tables/'benchmark_pinn.tex',['Caso','RMSE (K)',r'$|e_T|_{\max}$ (K)',r'$E_{b,\max}$ (\%)','Aceptadas'],[[r['case'],f"{r['rmse_median_K']:.4f}",f"{r['Tmax_error_max_K']:.3f}",f"{r['balance_max_pct']:.3f}",f"{r['accepted']}/{r['n']}"] for r in mainrows],'Resultados por caso: mediana del RMSE y máximos entre semillas','tab:benchmark-pinn')
    candidates=[r for r in rows if '/C0' in r['run'] and r['seed']==5]
    latex_table(tables/'benchmark_selection.tex',['Candidato','Parámetros','RMSE (K)',r'$E_b$ (\%)','Aceptado'],[[r['run'].split('/')[1],r['parameters'],f"{r['rmse_fem_K']:.4f}",f"{r['balance_pct']:.3f}",'Sí' if r['thermal_criteria_pass'] else 'No'] for r in candidates],'Comparación exploratoria de arquitectura e hiperparámetros en XLPE','tab:benchmark-seleccion')
    (out/'summary.json').write_text(json.dumps(dict(total_runs=len(rows),fem_solves=len(meshes)*3,selected=mainrows,mesh_convergence=meshes),ensure_ascii=False,indent=2),encoding='utf-8')
    coupled_tables(rows,out,tables)
    selection_tables(rows,tables,figures)
    findings=[r for r in rows if r['run'].startswith(('comparisons/global_interface','comparisons/control_multipole_budget','comparisons/pilot_multipole','comparisons/pilot_w64_d4'))]
    latex_table(tables/'benchmark_findings.tex',['Prueba','Semilla','RMSE (K)','$E_b$ (\\%)','Aceptada'],[[r['case']+' / '+r['run'].split('/')[1],r['seed'],f"{r['rmse_fem_K']:.4f}",f"{r['balance_pct']:.3f}",'Sí' if r['thermal_criteria_pass'] else 'No'] for r in findings],'Ablaciones de interfaz y enriquecimiento; cada control identifica su presupuesto','tab:benchmark-hallazgos')
    print(json.dumps(dict(runs=len(rows),fem=len(meshes)*3,selected=len(mainrows))))

def selection_tables(rows,tables,figures):
    selection=json.loads((BASE/'selection.json').read_text(encoding='utf-8'))
    if not selection.get('decisions'):return
    nominal=[];amp=[]
    labels={'coupled_final':'Base','coupled_refined':'Peso y presupuesto','coupled_width64':'Ancho 64','ampacity_final':'Pesos 100','ampacity_constraints1000':'Pesos 1000'}
    for name,d in selection['decisions'].items():
        if len(d['nominal_candidates'])>1:
            for r in d['nominal_candidates']:
                label=labels[r['directory'].split('/')[-2]]
                nominal.append([name+' / '+label,r['parameters'],f"{r['median_rmse_K']:.4f}",f"{r['max_electrical_residual_pct']:.4f}",f"{r['accepted']}/3"])
        if len(d['ampacity_candidates'])>1:
            for r in d['ampacity_candidates']:
                label=labels[r['directory'].split('/')[-2]]
                amp.append([name+' / '+label,f"{r['median_rmse_K']:.3f}",f"{r['max_current_error_pct']:.3f}",f"{r['max_temperature_limit_error_K']:.3f}",f"{r['accepted']}/3"])
    latex_table(tables/'benchmark_coupled_selection.tex',['Caso / alternativa','Parámetros','RMSE (K)',r'$E_{R,\max}$ (\%)','Aceptadas'],nominal,'Selección nominal acoplada por configuración completa, con tres semillas','tab:seleccion-acoplada')
    latex_table(tables/'benchmark_ampacity_selection.tex',['Caso / alternativa','RMSE (K)',r'$E_{I,\max}$ (\%)',r'$|T_{\max}-90|$ (K)','Aceptadas'],amp,'Comparación de pesos para corriente límite con igual presupuesto; errores máximos entre semillas','tab:seleccion-corriente')
    import matplotlib.pyplot as plt
    names=['xlpe_single','xlpe_dry_near','xlpe_dry_far','xlpe_dry_large','xlpe_backfill']
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for index,name in enumerate(names):
        ref=json.loads((BASE/'coupled_results'/name/'fem_ampacity_l2.json').read_text())
        ax.scatter(ref['current_A'],index,marker='|',s=250,c='black',label='FEM' if index==0 else None,zorder=3)
        directory=selected_ampacity_directory(name).relative_to(BASE).as_posix()
        chosen=[r for r in rows if r['run']==directory]
        for j,r in enumerate(chosen):
            accepted=r['ampacity_criteria_pass']
            ax.scatter(r['current_A'],index+(j-1)*.12,marker='o' if accepted else 'x',c='#157a4f' if accepted else '#b53030',s=35)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],color='black',marker='|',linestyle='none',markersize=12,label='FEM'),Line2D([],[],color='#157a4f',marker='o',linestyle='none',label='PINN aceptada'),Line2D([],[],color='#b53030',marker='x',linestyle='none',label='PINN rechazada')])
    ax.set(yticks=range(len(names)),yticklabels=['Homogéneo','Zona seca próxima','Zona seca lejana','Zona seca ampliada','Relleno mejorado'],xlabel='Corriente límite DC (A)')
    ax.grid(axis='x',alpha=.2);fig.savefig(figures/'ampacity_xlpe.png',dpi=160,bbox_inches='tight');plt.close(fig)

def coupled_tables(rows,out,tables):
    convergence=[];amp_rows=[];loss_rows=[];boundary_rows=[]
    for name,c in cases().items():
        if c['kind']!='cable':continue
        for mode,stem in [('nominal','fem'),('ampacity','fem_ampacity')]:
            data=[json.loads((BASE/'coupled_results'/name/f'{stem}_l{i}.json').read_text()) for i in range(3)]
            r=data[-1];change=abs(r['Tmax_C']-data[-2]['Tmax_C'])
            di=abs(r['current_A']/data[-2]['current_A']-1)*100
            convergence.append(dict(case=name,mode=mode,ndofs=r['ndofs'],Tmax_C=r['Tmax_C'],change_Tmax_K=change,current_A=r['current_A'],change_current_pct=di,balance_pct=r['balance_pct'],electrical_residual_pct=r['electrical_residual_pct'],converged=change/(r['Tmax_C']-c['T0'])<=.005 and di<=.5 and r['balance_pct']<=2))
        fixed=json.loads((BASE/'results'/name/'fem_l2.json').read_text())
        nominal=json.loads((BASE/'coupled_results'/name/'fem_l2.json').read_text())
        diagnostics=[json.loads(p.read_text()) for p in selected_directory(name).glob('pinn_seed*.json')]
        boundary_rows.append(dict(case=name,fem_max_power_error_pct=max(b['relative_power_error_pct'] for b in nominal['boundary_diagnostics']),pinn_max_power_error_pct=max(b['relative_power_error_pct'] for r in diagnostics for b in r['boundary_diagnostics'] if 'relative_power_error_pct' in b)))
        loss_rows.append(dict(case=name,T_R20_C=fixed['Tmax_C'],T_coupled_C=nominal['Tmax_C'],P20_W_m=c['power'],P_coupled_min_W_m=min(nominal['powers_W_m']),P_coupled_max_W_m=max(nominal['powers_W_m']),delta_T_K=nominal['Tmax_C']-fixed['Tmax_C']))
        directory=selected_ampacity_directory(name).relative_to(BASE).as_posix()
        rr=[p for p in rows if p['run']==directory]
        good=[p for p in rr if p['ampacity_criteria_pass']]
        amp_rows.append(dict(case=name,run=directory,fem_A=r['current_A'],n=len(rr),accepted=len(good),pinn_median_A=float(np.median([p['current_A'] for p in good])) if good else None,error_current_max_pct=max([p['error_current_pct'] for p in rr],default=None),temperature_limit_error_max_K=max([p['temperature_limit_error_K'] for p in rr],default=None)))
    write_csv(out/'coupled_fem_convergence.csv',convergence);write_csv(out/'temperature_dependent_losses.csv',loss_rows);write_csv(out/'ampacity_coupled.csv',amp_rows)
    write_csv(out/'conductor_boundary_diagnostics.csv',boundary_rows)
    latex_table(tables/'benchmark_boundary.tex',['Caso',r'FEM (\%)',r'PINN (\%)'],[[r['case'],f"{r['fem_max_power_error_pct']:.4f}",f"{r['pinn_max_power_error_pct']:.4f}"] for r in boundary_rows],'Máximo error de potencia integrada por conductor en operación nominal; PINN incluye todas las semillas','tab:balance-conductor')
    latex_table(tables/'benchmark_coupled_fem.tex',['Caso','$\\Delta T$ (K)','$\\Delta I$ (\\%)','$E_b$ (\\%)'],[[r['case'],f"{r['change_Tmax_K']:.4f}",f"{convergence[i+1]['change_current_pct']:.4f}",f"{max(r['balance_pct'],convergence[i+1]['balance_pct']):.3f}"] for i,r in enumerate(convergence) if r['mode']=='nominal'],'Convergencia FEM acoplada: cambio nominal de temperatura, cambio de ampacidad y peor balance fino','tab:benchmark-fem-acoplado')
    latex_table(tables/'benchmark_losses.tex',['Caso','$T(R_{20})$ (°C)','$T(R(T_c))$ (°C)','$\\Delta T$ (K)'],[[r['case'],f"{r['T_R20_C']:.3f}",f"{r['T_coupled_C']:.3f}",f"{r['delta_T_K']:.3f}"] for r in loss_rows],'Efecto de actualizar individualmente la resistencia eléctrica en FEM','tab:benchmark-perdidas')
    latex_table(tables/'benchmark_ampacity.tex',['Caso','FEM (A)','PINN (A)','$E_{I,\\max}$ (\\%)','Aceptadas'],[[r['case'],f"{r['fem_A']:.2f}",f"{r['pinn_median_A']:.2f}" if r['pinn_median_A'] is not None else '---',f"{r['error_current_max_pct']:.3f}" if r['error_current_max_pct'] is not None else '---',f"{r['accepted']}/{r['n']}"] for r in amp_rows],'Ampacidad DC acoplada: mediana PINN de las semillas aceptadas y error de todas las semillas','tab:benchmark-ampacidad')

if __name__=='__main__':main()
