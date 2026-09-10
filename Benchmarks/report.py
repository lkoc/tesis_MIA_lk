"""Generate tables and figures exclusively from saved execution artifacts."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases,fingerprint,field_k
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Benchmarks'

def records():
    result=[]
    for p in sorted(BASE.rglob('pinn_seed*.json')):
        if 'source' in p.parts:continue
        r=json.loads(p.read_text(encoding='utf-8'));m=r['metadata']
        if m['case_sha256']!=fingerprint(cases()[m['case']['id']]):raise ValueError(f'Stale physical data: {p}')
        row=dict(case=m['case']['id'],run=str(p.parent.relative_to(BASE)).replace('\\','/'),seed=m['seed'],variant=m.get('variant','enriched'),width=m['width'],depth=m['depth'],parameters=m['parameters'],seconds=m['training_seconds'])
        for key in ['Tmax_C','Tmax_fem_C','rmse_fem_K','nrmse_fem_pct','max_error_fem_K','error_Tmax_K','error_Tmax_rise_pct','balance_pct','pde_rmse_W_m3','thermal_criteria_pass','rmse_exact_K']:
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

def plot_case(name,save=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=cases()[name];f=np.load(BASE/'results'/name/'fem_l2.npz')
    paths=sorted(selected_directory(name).glob('pinn_seed*.npz'))
    if not paths:raise FileNotFoundError(f'No PINN results for {name}')
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
    def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%')
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
    latex_table(tables/'benchmark_fem.tex',['Caso','$N_2$','$T_{\max}$ (°C)','$|\Delta T|$ (K)','$E_b$ (\%)'],[[r['case'],r['ndofs_l2'],f"{r['Tmax_C']:.3f}",f"{r['change_Tmax_K']:.4f}",f"{r['balance_pct']:.3f}"] for r in meshes],'Convergencia de las referencias FEM entre las dos mallas más finas','tab:benchmark-fem')
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
    latex_table(tables/'benchmark_pinn.tex',['Caso','RMSE (K)','$|e_T|_{\max}$ (K)','$E_{b,\max}$ (\%)','Aceptadas'],[[r['case'],f"{r['rmse_median_K']:.4f}",f"{r['Tmax_error_max_K']:.3f}",f"{r['balance_max_pct']:.3f}",f"{r['accepted']}/{r['n']}"] for r in mainrows],'Resultados por caso: mediana del RMSE y máximos entre semillas','tab:benchmark-pinn')
    candidates=[r for r in rows if '/C0' in r['run'] and r['seed']==5]
    latex_table(tables/'benchmark_selection.tex',['Candidato','Parámetros','RMSE (K)','$E_b$ (\%)','Aceptado'],[[r['run'].split('/')[1],r['parameters'],f"{r['rmse_fem_K']:.4f}",f"{r['balance_pct']:.3f}",'Sí' if r['thermal_criteria_pass'] else 'No'] for r in candidates],'Comparación exploratoria de arquitectura e hiperparámetros en XLPE','tab:benchmark-seleccion')
    (out/'summary.json').write_text(json.dumps(dict(total_runs=len(rows),fem_solves=len(meshes)*3,selected=mainrows,mesh_convergence=meshes),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(runs=len(rows),fem=len(meshes)*3,selected=len(mainrows))))

if __name__=='__main__':main()
