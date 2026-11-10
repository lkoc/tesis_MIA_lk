"""Generate thesis tables, figures and conclusions from all staged evidence."""
from pathlib import Path
import argparse,csv,json,math,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_paths import artifact_path
BASE=ROOT/'Benchmarks/explicit_study';THESIS=ROOT/'Tesis_LaTeX_Borrador_UNI'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def esc(s):return str(s).replace('\\','/').replace('_',r'\_').replace('%',r'\%')
def fmt(v,d=3):return f'{v:.{d}f}' if v is not None else '--'
def failure_components(row):
    result=read(ROOT/row['path']/f"pinn_seed{row['seed']}.json")
    reference=read(artifact_path(result['reference']).with_suffix('.json'))
    rise=abs(reference['Tmax_C']-reference['specification']['case']['T0'])
    return {'Campo':max(result['nrmse_pct']/5,max(result['region_rmse_K'].values())/(.05*rise)),
        'Tmax':result['error_Tmax_pct']/5,
        'Energía':max(result['balance_pct'],result['derived_balance_pct'],*result['region_energy_error_pct'].values(),*result['region_derived_energy_error_pct'].values())/2,
        'Salto T':max(i['T_jump_max_K'] for i in result['interfaces'])/.1,
        'Salto flujo':max(max(i['flux_jump_rms_pct'],i['derived_flux_jump_rms_pct']) for i in result['interfaces'])}
def table(caption,label,headers,rows,columns=None):
    columns=columns or ('L{4.2cm}'+'r'*(len(headers)-1))
    lines=[r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{'+columns+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',
        ' & '.join(headers)+r'\\',r'\midrule\endfirsthead',r'\toprule',' & '.join(headers)+r'\\',r'\midrule\endhead']
    lines+=[' & '.join(str(v) for v in row)+r'\\' for row in rows]
    return '\n'.join(lines+[r'\bottomrule\end{longtable}\endgroup'])

def build(draft=False):
    if not draft and not (BASE/'status.json').exists():raise RuntimeError('Finish confirmation before publishing final results')
    output=BASE/'report';output.mkdir(exist_ok=True);figures=THESIS/'imagenes/explicit';figures.mkdir(exist_ok=True)
    rows=[];stages={};choices={}
    for stage in ['A','B','C','C2','D']:
        path=BASE/stage/'summary.json'
        if path.exists():stages[stage]=read(path);rows+=stages[stage]
        if (BASE/stage/'selection.json').exists():choices[stage]=read(BASE/stage/'selection.json')
    if rows:
        with (output/'runs.csv').open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[r'\subsection{REFERENCIAS Y COBERTURA EFECTIVA}']
    fem=[]
    for path in sorted((BASE/'references').rglob('gate.json')):
        gate=read(path);last=gate['levels'][-1]
        fem.append(dict(case=path.parent.name,**last,passed=gate['passed']))
    lines+=[table('Referencias FEM explícitas refinadas. Fuente: registros de la campaña.','tab:explicit-fem',
        ['Caso','Niveles','Grados de libertad',r'$T_{\max}$, °C',r'$\Delta T_{\max}$, K','Balance, '+r'\pct{}'],
        [[esc(r['case']),len(read(BASE/'references/fixed'/r['case']/'gate.json')['levels']),r['ndofs'],fmt(r['Tmax_C']),fmt(r.get('Tmax_change_K'),5),fmt(r['balance_pct'],5)] for r in fem])]
    lines.append('Todas las comparaciones de las etapas A--D utilizan estas referencias de la misma formulación física. El refinamiento verifica también el cambio de campo por material. La cantidad de semillas no debe confundirse con el número de instalaciones independientes.')
    labels={'A':'ARQUITECTURA Y TAMAÑO','B':'COLOCACIONES Y LOCALIZACIÓN','C':'AJUSTE DE ENTRENAMIENTO','C2':'CALIBRACIÓN ADICIONAL DE CONTINUIDAD TÉRMICA'}
    summaries={}
    for stage,title in labels.items():
        lines.append(r'\subsection{'+title+'}')
        if stage not in stages:
            lines.append('Etapa todavía no finalizada en esta compilación de trabajo.');continue
        stage_rows=stages[stage]
        if stage=='C2':
            stage_rows=stage_rows+[dict(r,stage='C2',candidate='weight_100') for r in stages['C'] if r['candidate']==choices['C']['winner']]
            lines.append('Esta ampliación se registró durante B, antes de C y de toda confirmación, al persistir saltos térmicos excesivos con refinamiento de puntos. Cambia únicamente el peso de continuidad de temperatura; conserva las ecuaciones, normalizaciones y tolerancias. El peso 100 reutiliza cuatro ejecuciones de C; 1000 y 10000 añaden ocho entrenamientos. Las corridas reutilizadas no se cuentan dos veces en el total.')
        grouped=[]
        for name in sorted({r['candidate'] for r in stage_rows}):
            group=[r for r in stage_rows if r['candidate']==name]
            grouped.append(dict(candidate=name,parameters=max(r['parameters'] for r in group),accepted=sum(r['accepted'] for r in group),total=len(group),
                rmse_median=float(np.median([r['rmse_K'] for r in group])),Tmax_error_max=max(r['Tmax_error_K'] for r in group),violation=max(r['violation'] for r in group)))
        summaries[stage]=grouped
        lines+=[table(f'Etapa {stage}: comparación de todas las semillas de desarrollo. Fuente: elaboración propia.',f'tab:explicit-{stage}',
            ['Configuración','Parámetros','Aceptadas','RMSE med., K',r'\makecell{Error máx.\\$T_{\max}$, K}','Violación máxima'],
            [[esc(r['candidate']),r['parameters'],f"{r['accepted']}/{r['total']}",fmt(r['rmse_median']),fmt(r['Tmax_error_max']),fmt(r['violation'])] for r in grouped])]
        choice=choices[stage]
        lines.append('La regla registrada selecciona '+r'\texttt{'+esc(choice['winner'])+'}. '+
            ('La configuración cumple en todos los problemas y semillas de esta etapa.' if choice['qualified'] else 'La selección es provisional: ninguna candidata cumple en todos los problemas y semillas de la etapa. No se declara una red o receta mínima admisible.'))
        lines.append('La violación máxima es el mayor cociente entre una métrica de rechazo y su tolerancia; un valor mayor que uno identifica incumplimiento. Las medianas incluyen también los intentos rechazados. Los parámetros corresponden al mayor tamaño total entre los casos de la etapa.')
        if stage in ('A','B'):
            diagnoses=[]
            for candidate in grouped:
                group=[r for r in stages[stage] if r['candidate']==candidate['candidate']]
                components=[failure_components(r) for r in group]
                worst={key:max(c[key] for c in components) for key in components[0]};cause=max(worst,key=worst.get)
                diagnoses.append([esc(candidate['candidate']),cause,fmt(worst[cause])])
            lines.append(table(f'Mayor restricción relativa de la etapa {stage}, incluidos los intentos rechazados.',f'tab:explicit-{stage}-diagnosis',['Configuración','Restricción dominante','Cociente'],diagnoses,columns='L{5.2cm}L{4cm}r'))
            lines.append('La restricción dominante identifica dónde se incumple más la tolerancia, no demuestra por sí sola la causa de optimización.')
            if stage=='A':lines.append('La formulación mixta añade variables y ecuaciones; su rechazo con este presupuesto no refuta su equivalencia física. La comparación de Fourier conserva frecuencias fijadas y no optimiza su espectro. La variante multipolar ensayada tampoco agota todas las bases posibles.')
            else:lines.append('Un balance integral pequeño puede coexistir con errores del campo o de las interfaces. La nube uniforme deja pocas muestras próximas al cable dentro de un suelo extenso; se evalúa su consecuencia mediante FEM, sin considerar suficiente una pérdida de entrenamiento pequeña.')
    convergence=[]
    case_choices=[]
    if 'A' in stages:
        lines.append(r'\subsection{MENOR RED POR PROBLEMA DENTRO DEL ESPACIO ENSAYADO}')
        for name in sorted({r['case'] for r in stages['A']}):
            eligible=[]
            for candidate in sorted({r['candidate'] for r in stages['A']}):
                group=[r for r in stages['A'] if r['case']==name and r['candidate']==candidate]
                if all(r['accepted'] for r in group):eligible.append(dict(candidate=candidate,parameters=group[0]['parameters'],rmse_median=float(np.median([r['rmse_K'] for r in group]))))
            eligible.sort(key=lambda r:(r['parameters'],r['rmse_median']))
            case_choices.append(dict(case=name,winner=eligible[0] if eligible else None,eligible=eligible))
        lines.append(table('Menor red aceptada por problema entre las candidatas de A. No es una garantía fuera del caso ensayado.','tab:explicit-smallest',
            ['Problema','Menor candidata','Parámetros','RMSE med., K'],
            [[esc(r['case']),esc(r['winner']['candidate']) if r['winner'] else 'Ninguna',r['winner']['parameters'] if r['winner'] else '--',fmt(r['winner']['rmse_median']) if r['winner'] else '--'] for r in case_choices],columns='L{3.5cm}L{4.2cm}rr'))
    if 'B' in stages:
        lines.append(r'\subsection{INSENSIBILIDAD Y PRESUPUESTOS DE COLOCACIÓN}')
        from Benchmarks.full_domain import FullDomain
        point_budgets=[]
        for name in ['xlpe_single','xlpe_discrete_layers']:
            domain=FullDomain(read(BASE/'cases'/f'{name}.json'))
            for factor in [1,2,4]:
                point_budgets.append([esc(name),f'{factor}x',domain.nsoil*256*factor,(len(domain.regions)-domain.nsoil)*128*factor,len(domain.interfaces)*96*factor])
        lines.append(table('Colocaciones efectivas por problema y factor. Las interfaces tienen dos trazas en una misma posición.','tab:explicit-point-counts',
            ['Caso','Factor','Interior suelo','Interior cable','Interfaces'],point_budgets))
        lines.append('Las políticas uniforme, de interfaz y residual se comparan con el presupuesto 2x. Estos conteos excluyen las cuadraturas auxiliares: cada lado exterior emplea 96 veces el factor y cada conductor usa el máximo entre 256 y 128 veces el factor para la integral eléctrica, activa en DC. Por ello, el contraste 1x--4x aumenta el presupuesto espacial conjunto, no aísla solo los puntos interiores. No se interpreta la suma como una malla FEM ni se cuenta cada traza de interfaz como una posición nueva.')
        for name in ['xlpe_single','xlpe_discrete_layers']:
            for seed in [11,23]:
                for low,high in [(1,2),(2,4)]:
                    a=next(r for r in stages['B'] if r['case']==name and r['seed']==seed and r['candidate']==f'mixed_{low}x')
                    b=next(r for r in stages['B'] if r['case']==name and r['seed']==seed and r['candidate']==f'mixed_{high}x')
                    with np.load(ROOT/a['path']/f'pinn_seed{seed}.npz') as data:Ta=data['T'];weights=data['weights'];labels=data['region']
                    with np.load(ROOT/b['path']/f'pinn_seed{seed}.npz') as data:Tb=data['T']
                    ref=read(BASE/'references/fixed'/name/'gate.json');meta=read(BASE/'references/fixed'/name/Path(ref['reference']).with_suffix('.json'))
                    delta=meta['Tmax_C']-20.;D=100*np.sqrt(np.average((Ta-Tb)**2,weights=weights))/delta;H=100*abs(a['Tmax_C']-b['Tmax_C'])/delta
                    region=max(float(np.sqrt(np.mean((Ta[labels==i]-Tb[labels==i])**2))) for i in np.unique(labels))
                    convergence.append(dict(case=name,seed=seed,low=low,high=high,D_pct=float(D),H_pct=float(H),max_region_difference_K=region,
                        insensitive=bool(D<=.5 and H<=.5 and a['accepted'] and b['accepted'])))
        lines.append(table('Insensibilidad observada al aumentar colocaciones. Ambos cálculos deben estar aceptados.','tab:explicit-convergence',
            ['Caso','Semilla','Cambio',r'$D$, \pct{}',r'$H$, \pct{}','Insensible'],
            [[esc(r['case']),r['seed'],f"{r['low']}x--{r['high']}x",fmt(r['D_pct']),fmt(r['H_pct']),'Sí' if r['insensitive'] else 'No'] for r in convergence]))
        lines.append(f"Se cumple la comprobación de insensibilidad en {sum(r['insensitive'] for r in convergence)} de {len(convergence)} comparaciones. Las diferencias restantes impiden afirmar independencia universal del número de colocaciones.")
        lines.append('Este contraste corresponde al presupuesto y pesos de B. El ajuste posterior de C y C2 no convierte retrospectivamente esos pares en convergentes ni demuestra independencia de colocaciones de la receta final. La aceptación de D se refiere a errores y criterios físicos evaluados frente a FEM con el presupuesto congelado.')
    lines.append(r'\subsection{CONFIRMACIÓN CON SEMILLAS INDEPENDIENTES}')
    production=[]
    if 'D' in stages:
        for name in sorted({r['case'] for r in stages['D']}):
            group=[r for r in stages['D'] if r['case']==name]
            production.append(dict(case=name,accepted=sum(r['accepted'] for r in group),total=len(group),rmse_median=float(np.median([r['rmse_K'] for r in group])),
                rmse_min=min(r['rmse_K'] for r in group),rmse_max=max(r['rmse_K'] for r in group),Tmax_error_max=max(r['Tmax_error_K'] for r in group),balance_max=max(r['balance_pct'] for r in group)))
        lines.append(table('Confirmación de la receta congelada: semillas 71, 83 y 97. Fuente: elaboración propia.','tab:explicit-production',
            ['Caso','Aceptadas','RMSE med., K','RMSE mín.--máx., K',r'\makecell{Error máx.\\$T_{\max}$, K}','Balance máx., '+r'\pct{}'],
            [[esc(r['case']),f"{r['accepted']}/{r['total']}",fmt(r['rmse_median']),fmt(r['rmse_min'])+'--'+fmt(r['rmse_max']),fmt(r['Tmax_error_max']),fmt(r['balance_max'])] for r in production],columns='L{3.4cm}rrrrr'))
        lines.append('Los casos aceptados parcialmente no se consideran configuraciones robustas. Ninguna semilla de confirmación se utilizó para elegir la receta de C y C2. Las diferencias de rendimiento entre casos corresponden a transferencia del procedimiento con reentrenamiento, no de una única red ya entrenada.')
    else:lines.append('La confirmación aún no está completa en esta compilación de trabajo; no se publican resultados finales ficticios.')
    effects=[];hotspots=[]
    if production:
        baseline=next(r for r in fem if r['case']=='xlpe_single')
        for name in ['xlpe_backfill','xlpe_dry_near','xlpe_dry_far','xlpe_discrete_layers']:
            ref=next(r for r in fem if r['case']==name);differences=[];passed=0
            for seed in [71,83,97]:
                a=next(r for r in stages['D'] if r['case']=='xlpe_single' and r['seed']==seed);b=next(r for r in stages['D'] if r['case']==name and r['seed']==seed)
                differences.append(b['Tmax_C']-a['Tmax_C']);passed+=a['accepted'] and b['accepted']
            delta=ref['Tmax_C']-baseline['Tmax_C']
            old_case=read(BASE/'references/fixed'/name/'gate.json')['levels'][-2]['Tmax_C']
            old_base=read(BASE/'references/fixed/xlpe_single/gate.json')['levels'][-2]['Tmax_C']
            mesh_change=abs(delta-(old_case-old_base));spread=max(differences)-min(differences)
            resolved=passed==3 and all(value*delta>0 for value in differences) and spread<abs(delta) and mesh_change<abs(delta)
            effects.append(dict(case=name,FEM_delta_K=delta,FEM_effect_mesh_change_K=mesh_change,PINN_deltas_K=differences,PINN_range_K=spread,PINN_delta_median_K=float(np.median(differences)),accepted_pairs=passed,resolved=resolved))
        lines.append(table('Efectos térmicos respecto al control XLPE homogéneo, a igual fuente prescrita.','tab:explicit-effects',
            ['Escenario',r'$\Delta T^F_{\max}$, K',r'Mediana $\Delta T^P_{\max}$, K','Pares aceptados'],
            [[esc(r['case']),fmt(r['FEM_delta_K']),fmt(r['PINN_delta_median_K']),f"{r['accepted_pairs']}/3"] for r in effects]))
        lines.append('Las diferencias FEM caracterizan los pares físicos; una diferencia PINN de un par rechazado es diagnóstica y no confirma el efecto mediante el artefacto. El control homogéneo utiliza k del suelo base, no se identifica automáticamente con un promedio equivalente del mapa heterogéneo.')
        lines.append(table('Resolución empírica del efecto: discretización y variabilidad pareada.','tab:explicit-effect-resolution',
            ['Caso','Cambio de malla, K','Rango PINN, K','Resuelto'],[[esc(r['case']),fmt(r['FEM_effect_mesh_change_K'],5),fmt(r['PINN_range_K']),'Sí' if r['resolved'] else 'No'] for r in effects]))
        lines.append('Se exige aceptación de los tres pares, signos coincidentes con FEM y tanto rango PINN como cambio del efecto por malla menores que el módulo del efecto FEM. Esta regla se explicitó antes de D. Es una comprobación empírica de resolución, sin intervalos poblacionales ni cotas rigurosas de error.')
        from Benchmarks.full_domain import FullDomain
        for r in stages['D']:
            domain=FullDomain(read(BASE/'cases'/f"{r['case']}.json"));seed=r['seed']
            gate=read(BASE/'references/fixed'/r['case']/'gate.json')
            with np.load(ROOT/r['path']/f'pinn_seed{seed}.npz') as data,np.load(BASE/'references/fixed'/r['case']/gate['reference']) as ref:
                mask=np.isin(data['region'],[region.id for region in domain.conductors]);xy=data['xy'][mask]
                ip=int(data['T'][mask].argmax());iff=int(ref['T'][mask].argmax())
                hotspots.append(dict(case=r['case'],seed=seed,accepted=r['accepted'],PINN_xy_m=xy[ip].tolist(),FEM_xy_m=xy[iff].tolist(),separation_mm=float(np.linalg.norm(xy[ip]-xy[iff])*1000)))
        lines.append('La ubicación del máximo se conserva en el expediente sobre la nube común de conductores. El muestreo y la escasa diferencia térmica dentro del metal impiden atribuir precisión milimétrica a esa ubicación. OE3 dispone de temperaturas y efectos contrastados, pero no de una certificación del punto caliente continuo; no se confunde una máxima muestreada con una búsqueda espacial convergente.')
    if rows:
        fig,axes=plt.subplots(3,1,figsize=(7,10),layout='constrained')
        for ax,stage in zip(axes,'ABC'):
            for r in summaries.get(stage,[]):
                ax.scatter(r['parameters'] if stage=='A' else r['candidate'],r['rmse_median'],c='tab:green' if r['accepted']==r['total'] else 'tab:red')
            ax.set_title('Etapa '+stage);ax.set_ylabel('RMSE mediano (K)');ax.grid(alpha=.2)
            if stage=='A':ax.set_xlabel('Parámetros entrenables');ax.set_xscale('log');ax.set_yscale('log')
            else:ax.tick_params(axis='x',labelrotation=20,labelsize=9)
        fig.savefig(figures/'selection.pdf');fig.savefig(figures/'selection.png',dpi=180);plt.close(fig)
        lines += [r'\begin{figure}[p]\centering\includegraphics[width=.88\textwidth,height=.82\textheight,keepaspectratio]{explicit/selection.pdf}',
            r'\caption{Comparación de desarrollo. Verde: aceptación completa; rojo: al menos un rechazo. Los paneles corresponden a etapas distintas y no a un único factorial. Fuente: elaboración propia.}\end{figure}']
    if 'B' in stages:
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
        for ax,name in zip(axes,['xlpe_single','xlpe_discrete_layers']):
            for seed in [11,23]:
                group=[next(r for r in stages['B'] if r['case']==name and r['seed']==seed and r['candidate']==f'mixed_{factor}x') for factor in [1,2,4]]
                ax.plot([1,2,4],[r['rmse_K'] for r in group],'-o',label=f'Semilla {seed}')
            ax.set_title(name);ax.set_xlabel('Factor de colocaciones');ax.set_ylabel('RMSE frente a FEM (K)');ax.set_xticks([1,2,4]);ax.grid(alpha=.2);ax.legend()
        fig.savefig(figures/'convergence.pdf');fig.savefig(figures/'convergence.png',dpi=180);plt.close(fig)
        lines += [r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{explicit/convergence.pdf}',r'\caption{Sensibilidad al número de colocaciones con la misma arquitectura y política geométrica. Fuente: elaboración propia.}\end{figure}']
        for detail in [False,True]:
            from matplotlib.patches import Circle
            fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained')
            for ax,policy in zip(axes.ravel(),['uniform','mixed','interface','residual']):
                folder=BASE/'B'/f'{policy}_2x'/'xlpe_discrete_layers/seed11'
                files=sorted(folder.glob('collocation_seed11_step*.npz'))
                source=files[-1] if files else folder/'collocation_seed11.npz'
                with np.load(source) as cloud:
                    for name in cloud.files:
                        xy=cloud[name];ax.scatter(xy[:,0],xy[:,1],s=2 if detail else 1,alpha=.7,c='tab:blue' if name.startswith('soil') else 'tab:orange')
                ax.axhline(-1,color='black',ls='--',lw=.7)
                for radius in [.0055,.012,.013,.015]:ax.add_patch(Circle((0,-.7),radius,fill=False,color='black',lw=.6))
                ax.set_title(policy+(' (adaptada)' if policy=='residual' else ''))
                ax.set_xlim((-.025,.025) if detail else (-4,4));ax.set_ylim((-.725,-.675) if detail else (-4,0));ax.set_aspect('equal')
                ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.tick_params(labelsize=8)
            stem='collocation_detail' if detail else 'collocation_domain'
            fig.savefig(figures/(stem+'.pdf'));fig.savefig(figures/(stem+'.png'),dpi=180);plt.close(fig)
            lines += [r'\begin{figure}[p]\centering\includegraphics[width=\textwidth]{explicit/'+stem+'.pdf}',
                r'\caption{'+('Detalle del cable' if detail else 'Cobertura del dominio')+r' para las cuatro políticas a igual presupuesto 2x, estratos y semilla 11. Azul: suelo; naranja: materiales del cable; líneas: interfaces geométricas. Se muestran colocaciones interiores, no las cuadraturas de trazas. Fuente: nubes archivadas; la política residual muestra la nube posterior a la adaptación.}\end{figure}']
    if production:
        name='xlpe_dry_near';seed=71;r=next(r for r in stages['D'] if r['case']==name and r['seed']==seed)
        with np.load(ROOT/r['path']/f'pinn_seed{seed}.npz') as data:xy=data['xy'];Tp=data['T'];region=data['region']
        gate=read(BASE/'references/fixed'/name/'gate.json')
        with np.load(BASE/'references/fixed'/name/gate['reference']) as data:Tf=data['T']
        fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained');vmin=min(Tp.min(),Tf.min());vmax=max(Tp.max(),Tf.max());vmaxerr=max(abs(Tp-Tf))
        for row,mask in enumerate([region==0,region>0]):
            for ax,values,title in zip(axes[row],[Tf,Tp,Tp-Tf],['FEM','PINN','PINN − FEM']):
                im=ax.scatter(xy[mask,0],xy[mask,1],c=values[mask],s=5,cmap='coolwarm' if title=='PINN − FEM' else 'inferno',vmin=-vmaxerr if title=='PINN − FEM' else vmin,vmax=vmaxerr if title=='PINN − FEM' else vmax)
                ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');fig.colorbar(im,ax=ax,label='K' if title=='PINN − FEM' else '°C')
        fig.savefig(figures/'field.pdf');fig.savefig(figures/'field.png',dpi=180);plt.close(fig)
        lines += [r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{explicit/field.pdf}',r'\caption{Suelo y detalle interior del cable en XLPE con zona seca cercana, semilla 71 fijada previamente. Las temperaturas comparten escala y el error usa escala simétrica. Fuente: campos de confirmación.}\end{figure}']
    (THESIS/'tablas/explicit_results.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    appendix=table('Todas las ejecuciones de la campaña explícita. Fuente: manifiestos por etapa.','tab:explicit-all',
        ['Etapa / configuración','Caso','Semilla','RMSE, K','Aceptada'],
        [[esc(r['stage']+'/'+r['candidate']),esc(r['case']),r['seed'],fmt(r['rmse_K']),'Sí' if r['accepted'] else 'No'] for r in rows],columns='L{4cm}L{3.5cm}rrr')
    (THESIS/'tablas/explicit_all_runs.tex').write_text(appendix+'\n',encoding='utf-8')
    summary=dict(total=len(rows),stages={k:len(v) for k,v in stages.items()},choices=choices,case_choices=case_choices,production=production,effects=effects,convergence=convergence,hotspots=hotspots,
        production_accepted=sum(r['accepted'] for r in stages.get('D',[])),production_total=len(stages.get('D',[])))
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if 'D' in stages:
        final_choice=choices['C2'];config=final_choice['configuration'];good=summary['production_accepted'];total=summary['production_total'];arch=esc(config['variant'])
        phrase='aceptada en el desarrollo' if final_choice['qualified'] else 'provisional por aceptación incompleta en el desarrollo'
        spanish=(f"La campaña principal comprende {len(rows)} entrenamientos, incluidos ocho de calibración adicional de continuidad térmica. "+
            f"La receta {phrase}, con arquitectura \\texttt{{{arch}}} de {config['width']}×{config['depth']} por material, satisface los criterios conjuntos en {good} de {total} ejecuciones finales. "+
            f"La insensibilidad a colocaciones se observa en {sum(r['insensitive'] for r in convergence)} de {len(convergence)} comparaciones, sin establecer convergencia universal. "+
            'Un contraste FEM 2D de cuatro mallas muestra que redistribuir la misma potencia según el perfil piel modifica la temperatura máxima en aproximadamente 0,004 K para el caso Aras evaluado; el aumento de pérdida AC constituye un efecto distinto.')
        english=(f"The main study comprises {len(rows)} training runs, including eight additional temperature-continuity calibration runs. "+
            f"The frozen {config['variant']} recipe, using {config['width']} neurons in each of {config['depth']} hidden layers per material, satisfies the joint criteria in {good} of {total} confirmation runs. "+
            ('The development selection remains provisional because complete acceptance was not achieved. ' if not final_choice['qualified'] else '')+
            f"Collocation insensitivity is observed in {sum(r['insensitive'] for r in convergence)} of {len(convergence)} comparisons, without establishing universal convergence. "+
            'A four-mesh 2D FEM comparison finds that redistributing the same power through the skin-effect profile changes the maximum temperature by approximately 0.004 K in the evaluated Aras case; the AC power increase is a separate effect.')
    else:spanish='La campaña registrada todavía está en ejecución en esta compilación de trabajo; no se atribuyen resultados finales a etapas incompletas.';english='The registered campaign is still running in this working compilation; unfinished stages are not presented as final results.'
    for lang,paragraph in [('es',spanish),('en',english)]:
        (THESIS/f'tablas/explicit_abstract_{lang}.tex').write_text(paragraph+'\n',encoding='utf-8')
    conclusion=[r'\chapter{CONCLUSIONES Y RECOMENDACIONES}',r'\label{ch:conclusiones}',r'\section{CONCLUSIONES}',
        'OE1 se concreta en una especificación común para conductor, capas y suelo, con fuentes volumétricas, interfaces y unidades explícitas. La temperatura interior procede de la PDE. La procedencia de resistencia DC/AC y construcción sigue siendo una condición necesaria para trasladar los controles a cables reales; una etiqueta R20 no certifica por sí sola esa procedencia.']
    if production:
        robust=sum(r['accepted']==r['total'] for r in production)
        conclusion+=[f"OE2 produce una clasificación verificable: {summary['production_accepted']} de {summary['production_total']} ejecuciones finales cumplen todos los criterios y {robust} de {len(production)} casos cumplen en las tres semillas nuevas. Las aceptaciones parciales delimitan un uso restringido; no se eliminan intentos rechazados para sostener el resultado.",
            'La selección de tamaño es condicional al caso y presupuesto. '+('La receta final satisface los casos de desarrollo, sin demostrar un óptimo global de arquitectura.' if choices['C2']['qualified'] else 'La receta final conserva carácter provisional porque no logró aceptación completa en desarrollo; no se demuestra una red mínima admisible para todo el dominio.')+
            ' El cuadro de menores redes por problema muestra únicamente alternativas aceptadas dentro de las ensayadas.',
            f"El estudio inicial de colocaciones satisface la comprobación de insensibilidad en {sum(r['insensitive'] for r in convergence)} de {len(convergence)} comparaciones. Aumentar puntos y cambiar su distribución tienen efectos que dependen del caso y la semilla. La calibración posterior no establece por sí sola independencia de colocaciones de la receta final. No se demuestra que duplicar, cuadruplicar o concentrar muestras mejore siempre la solución.",
            'OE3 dispone de contrastes FEM de relleno, zona seca cercana, zona seca lejana y estratos respecto al control homogéneo. La confirmación PINN de cada efecto se limita a los pares aceptados; cuando una semilla incumple los criterios, su diferencia térmica no se usa como verificación del efecto mediante el artefacto.']
        for effect in effects:
            conclusion.append(f"En {esc(effect['case'])}, la diferencia FEM de temperatura máxima respecto al control es {effect['FEM_delta_K']:.3f} K a igual fuente, con {effect['accepted_pairs']} de tres pares PINN aceptados. "+('El efecto satisface además los controles empíricos de signo, dispersión y cambio de malla.' if effect['resolved'] else 'El efecto no satisface todos los controles para declararlo resuelto por la campaña PINN.')+' Este resultado corresponde al mapa y contorno declarados, no a una ley universal de cualquier heterogeneidad.')
        conclusion.append('La localización del máximo se informa sobre puntos independientes de evaluación. La campaña no demuestra convergencia de su posición continua ni precisión milimétrica del punto caliente; esa parte de OE3 conserva una limitación explícita.')
    else:conclusion.append('Las etapas numéricas siguen en ejecución; esta compilación no emite conclusiones de aceptación final.')
    conclusion += [r'\input{tablas/explicit_ampacity_conclusion}',
        'La sensibilidad al perfil piel a igual potencia fue de aproximadamente 0,004 K en el ensayo FEM 2D Aras, estable con refinamiento. Ello apoya una influencia pequeña de la redistribución radial en ese régimen estacionario, pero no justifica omitir la corrección de pérdidas DC a AC ni la PDE interior. Las estimaciones para conductor macizo no sustituyen datos de cables segmentados o pérdidas de proximidad.',
        'La contribución DSR consiste en conservar la física, separar desarrollo y confirmación, estudiar tamaño y colocaciones de forma controlada, mantener una referencia FEM común y publicar rechazos. Los resultados no demuestran validación de campo, generalización de pesos, un solucionador transitorio completo ni aceleración frente a FEM.',
        r'\section{RECOMENDACIONES}',
        'Utilizar una receta únicamente en casos y tolerancias verificados. Un escenario nuevo requiere datos físicos trazables y convergencia FEM. Si falla la confirmación, abrir una nueva etapa de desarrollo y reservar nuevas semillas; no adaptar retrospectivamente la receta y conservar la etiqueta de prueba independiente.',
        'Ampliar la búsqueda solo donde el diagnóstico identifique una limitación: aumentar capacidad si la representación es insuficiente, redistribuir colocaciones si faltan regiones difíciles o ampliar optimización si los residuos aún disminuyen. Comparar estas decisiones con igual presupuesto dentro de cada experimento. Los diseños secuenciales no excluyen interacciones entre ellas.',
        'Para la extensión temporal, incorporar capacidades volumétricas documentadas, condiciones iniciales, variación de carga y convergencia temporal. Para aplicación AC, confirmar construcción y resistencias, incorporar proximidad y pérdidas de pantallas cuando correspondan, y verificar el campo electromagnético si se necesita una fuente local acoplada a temperatura.',
        'Mantener las entradas físicas separadas de las numéricas, conservar la configuración congelada, las nubes y los campos FEM, y generar las cifras del resumen y las conclusiones desde los mismos resultados. Una comparación de tiempo requiere ejecuciones aisladas y costes completos de entrenamiento y evaluación.']
    (THESIS/'capitulos/04_conclusiones_y_recomendaciones.tex').write_text('\n\n'.join(conclusion)+'\n',encoding='utf-8')
    for name in ['explicit_ampacity','explicit_ampacity_conclusion']:
        path=THESIS/f'tablas/{name}.tex'
        if not path.exists():path.write_text('La verificación de corriente límite explícita está en ejecución. OE4 no se considera concluido con resultados de potencia fija.\n',encoding='utf-8')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args();print(json.dumps(build(args.draft),ensure_ascii=False))
