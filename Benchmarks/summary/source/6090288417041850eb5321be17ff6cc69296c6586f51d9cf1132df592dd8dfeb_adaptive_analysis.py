"""Matched sampling analysis, replay checks and figures from saved evidence."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys,re
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.report import BASE,ROOT,records,write_csv,latex_table
from Benchmarks.adaptive import resample
from Benchmarks.cases import cases,field_k,in_domain
from Benchmarks.citations import REFERENCES
MODES=['fixed','uniform','residual','gradient']
LABELS=dict(fixed='Fijo geométrico',uniform='Renovación aleatoria',residual='Residuo adaptativo',gradient='Gradiente adaptativo')
STUDIES={'kim_layered':dict(family='nominal',baseline='coupled_final/kim_layered'),
         'xlpe_backfill':dict(family='ampacity',baseline='comparisons/ampacity_constraints1000/xlpe_backfill')}

def directory(name,mode):
    s=STUDIES[name]
    return s['baseline'] if mode=='fixed' else f"comparisons/adaptive_{s['family']}_{mode}/{name}"

def ast_function(data,name):
    return ast.dump(next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)

def verify_cloud(path,c):
    import torch
    from Benchmarks.pinn import Network,predict,tensor,training_points,gradients,laplace_variable_k,source
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    r=json.loads(path.read_text());m=r['metadata'];mode=m['sampling'];cfg=m['configuration'];seed=m['seed']
    cloud=np.load(path.parent/f'collocation_seed{seed}.npz')
    initial=training_points(c,seed,cfg['n'])
    assert np.array_equal(initial,cloud['initial_xy'])
    checks=[]
    for index,h in enumerate(m['adaptive_history']):
        prefix=f'round{index}_';candidate=training_points(c,h['seed'],cfg['n']*cfg['candidate_multiplier'])
        assert np.array_equal(candidate,cloud[prefix+'candidates']) and in_domain(c,candidate).all()
        scores=cloud[prefix+'scores'];points,info=resample(initial,candidate,scores,h['seed'],mode,cfg['adaptive_fraction'])
        assert np.array_equal(points,cloud[prefix+'xy']) and len(points)==m['n_interior']
        for key,value in info.items():assert np.array_equal(value,cloud[prefix+key])
        indices=np.linspace(0,len(candidate)-1,min(96,len(candidate)),dtype=int)
        if mode=='uniform':recomputed=np.ones(len(indices))
        else:
            model=Network(c,m['width'],m['depth'],m['variant'],m['coupled'],m['ampacity'])
            model.load_state_dict({key:torch.tensor(cloud[prefix+'state_'+key]) for key in model.state_dict()})
            probe=tensor(candidate[indices],True);temp=predict(model,probe)
            if mode=='gradient':score=torch.linalg.vector_norm(gradients(temp,probe),dim=1)/c['scale']
            else:
                k=field_k(c,probe[:,:1],probe[:,1:2],torch)
                score=torch.abs((laplace_variable_k(temp,probe,k)+source(c,probe[:,:1],probe[:,1:2],torch))/(k*c['scale'])).ravel()
            recomputed=score.detach().numpy()
        assert np.allclose(recomputed,scores[indices],rtol=1e-7,atol=1e-9),(path,index)
        checks.append(dict(round=index,step=h['step'],candidate_count=len(candidate),interior_count=len(points),
            replayed_score_count=len(indices),max_score_replay_error=float(np.max(abs(recomputed-scores[indices]))),
            score_selection_ratio=h['selected_mean_score']/h['mean_score'] if h['mean_score'] else None))
    assert np.array_equal(points,cloud['final_xy'])
    return checks

def figures(name,summary,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    c=cases()[name];folder=ROOT/'Tesis_LaTeX_Borrador_UNI/imagenes/benchmarks'
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    metric='error_current_pct' if STUDIES[name]['family']=='ampacity' else 'electrical_residual_pct'
    for j,mode in enumerate(MODES):
        rr=[r for r in rows if r['case']==name and r['run']==directory(name,mode)]
        for ax,key in zip(axes,['rmse_fem_K',metric]):
            vals=[r[key] for r in rr]
            ax.plot(np.full(3,j)+np.array([-.08,0,.08]),vals,'o',color='tab:blue')
            ax.plot([j-.2,j+.2],[np.median(vals)]*2,color='black')
            ax.set_xticks(range(4),['Fijo','Aleatorio','Residuo','Gradiente'],rotation=20)
    axes[0].set(ylabel='RMSE frente a FEM (K)')
    axes[1].set(ylabel='Error de corriente (%)' if metric=='error_current_pct' else 'Residuo eléctrico (%)')
    axes[1].axhline(5 if metric=='error_current_pct' else .1,color='tab:red',linestyle='--',label='Umbral del criterio')
    axes[1].legend(fontsize=8)
    fig.suptitle(name+': tres semillas; barra negra = mediana')
    fig.savefig(folder/f'{name}_adaptive.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    first=np.load(BASE/directory(name,'gradient')/'collocation_seed11.npz')
    for ax,mode in zip(axes.ravel(),MODES):
        xy=first['initial_xy'] if mode=='fixed' else np.load(BASE/directory(name,mode)/'collocation_seed11.npz')['final_xy']
        ax.scatter(xy[:,0],xy[:,1],s=1.5,alpha=.55,color='tab:blue',rasterized=True)
        for center in c['cables']:ax.add_patch(Circle(center,c['radius'],fill=False,color='black',lw=.8))
        x0,x1,y0,y1=c['bounds'];gx,gy=np.meshgrid(np.linspace(x0,x1,161),np.linspace(y0,y1,101))
        k=field_k(c,gx,gy);levels=float(k.min())+(float(k.max())-float(k.min()))*np.array([.1,.3,.5,.7,.9])
        ax.contour(gx,gy,k,levels=levels,colors='tab:orange',linewidths=.65,alpha=.85)
        ax.set(xlim=(x0,x1),ylim=(y0,y1),xlabel='x (m)',ylabel='y (m)',title=f'{LABELS[mode]}: {len(xy)} puntos')
        ax.set_aspect('equal')
    fig.suptitle(name+': semilla 11; naranja = isolíneas de conductividad')
    fig.savefig(folder/f'{name}_collocation.png',dpi=170);plt.close(fig)

def main():
    rows=records();catalog=cases();summary=[];paired=[];checks=[];inputs=set()
    current=(BASE/'pinn.py').read_text(encoding='utf-8')
    for name,s in STUDIES.items():
        reference={r['seed']:r for r in rows if r['run']==s['baseline']}
        assert set(reference)=={11,23,37}
        for mode in MODES:
            rr=sorted([r for r in rows if r['run']==directory(name,mode)],key=lambda r:r['seed'])
            assert len(rr)==3 and {r['seed'] for r in rr}=={11,23,37}
            for r in rr:
                base=reference[r['seed']]
                for key in ['width','depth','parameters','adam','lbfgs','n_interior','electrical_weight','limit_weight','mode']:
                    assert r[key]==base[key],(name,mode,key)
                path=ROOT/r['path'];inputs.add(path);m=json.loads(path.read_text())['metadata']
                if mode=='fixed':
                    digest=next(value for key,value in m['source_sha256'].items() if key.replace('\\','/')=='Benchmarks/pinn.py');archived=path.parent/'source'/(digest+'_pinn.py')
                    assert ast_function(archived.read_text(encoding='utf-8'),'training_points')==ast_function(current,'training_points')
                else:
                    inputs.add(path.parent/f"collocation_seed{r['seed']}.npz")
                    checks.append(dict(case=name,mode=mode,seed=r['seed'],rounds=verify_cloud(path,catalog[name])))
                    uniform=next(x for x in rows if x['run']==directory(name,'uniform') and x['seed']==r['seed'])
                    paired.append(dict(case=name,mode=mode,seed=r['seed'],rmse_K=r['rmse_fem_K'],
                        rmse_reduction_vs_fixed_pct=100*(1-r['rmse_fem_K']/base['rmse_fem_K']),
                        rmse_reduction_vs_refresh_pct=100*(1-r['rmse_fem_K']/uniform['rmse_fem_K']),
                        accepted=bool(r['ampacity_criteria_pass'] if s['family']=='ampacity' else r['thermal_criteria_pass'])))
            summary.append(dict(case=name,mode=mode,family=s['family'],directory=directory(name,mode),
                accepted=sum(bool(r['ampacity_criteria_pass'] if s['family']=='ampacity' else r['thermal_criteria_pass']) for r in rr),
                n=3,median_rmse_K=float(np.median([r['rmse_fem_K'] for r in rr])),min_rmse_K=min(r['rmse_fem_K'] for r in rr),max_rmse_K=max(r['rmse_fem_K'] for r in rr),
                max_electrical_residual_pct=max(r['electrical_residual_pct'] for r in rr),
                max_current_error_pct=max(r['error_current_pct'] for r in rr) if s['family']=='ampacity' else None,
                max_limit_error_K=max(r['temperature_limit_error_K'] for r in rr) if s['family']=='ampacity' else None,
                median_seconds=float(np.median([r['seconds'] for r in rr]))))
        figures(name,summary,rows)
    for p in [Path(__file__),BASE/'adaptive.py',BASE/'report.py',BASE/'pinn.py']:
        data=p.read_bytes();target=BASE/'summary/source'/(hashlib.sha256(data).hexdigest()+'_'+p.name)
        target.write_bytes(data)
    result=dict(summary=summary,paired=paired,replay_checks=checks,
        provenance=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}))
    (BASE/'summary/adaptive_sampling.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(BASE/'summary/adaptive_sampling.csv',summary);write_csv(BASE/'summary/adaptive_sampling_pairs.csv',paired)
    narrative(result)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

def narrative(result):
    from Benchmarks.internal_report import table
    summary=result['summary'];lines=['# Muestreo adaptativo: expediente de comparación','',
        'Se incorporan 18 entrenamientos nuevos y seis controles fijos existentes. Los dos escenarios mantienen conductividad espacial continua: seis cables con estratificación regularizada a corriente nominal y un cable XLPE con relleno mejorado a corriente límite. Cada alternativa usa tres semillas, la misma física y la misma referencia FEniCSx acoplada, verificada en tres mallas.','',
        '## Protocolo y fundamento','',
        'Wu et al. (2022: 6–8) comparan renovación aleatoria y muestreo por residuo, y proponen RAD. Aquí se implementa una adaptación propia con anclajes geométricos. El indicador alternativo de gradiente térmico no se atribuye como experimento de esos autores.','',
        '- Arquitectura multipolar 32×3, 1200 pasos Adam, hasta 1600 iteraciones L-BFGS, mismas pérdidas y 768 puntos globales más muestras geométricas. El número interior efectivo se mantiene exactamente por semilla.',
        '- Actualizaciones al comienzo de los pasos Adam 400 y 800. Se conserva una mitad de la nube inicial y se selecciona sin reemplazo la otra mitad de un conjunto candidato generado con n=3072 y la misma distribución geométrica.',
        '- Indicador residual: abs(div(k grad T)+Q)/(k·escala térmica). Indicador de gradiente: norma(grad T)/escala térmica. Ambos utilizan el campo completo, incluidas fuentes analíticas, multipolos y potencias aprendidas.',
        '- Probabilidades discretas proporcionales a score/media(score)+1. Si todos los indicadores son cero, se usa distribución uniforme sobre candidatos. El control aleatorio siempre usa probabilidades uniformes sobre esa misma familia geométrica; no es muestreo espacial uniforme en todo el suelo.',
        '- No se añaden puntos ni se adaptan fronteras o interfaces. L-BFGS mantiene fija la última nube. No se corrige la pérdida por importancia; cambia su ponderación espacial efectiva.',
        '- Se mantienen 6000 puntos externos y los criterios originales de campo, Tmax, balance, electricidad y corriente. El residuo local alto no constituye por sí solo una cota del error térmico.','',
        '## Resultados agregados','',table(['Caso','Muestreo','RMSE mediano K','Rango K','Máx. residuo eléctrico %','Máx. error corriente %','Máx. distancia a 90 °C K','Aceptadas'],
        [[r['case'],LABELS[r['mode']],f"{r['median_rmse_K']:.6f}",f"{r['min_rmse_K']:.6f}–{r['max_rmse_K']:.6f}",f"{r['max_electrical_residual_pct']:.6f}",f"{r['max_current_error_pct']:.6f}" if r['max_current_error_pct'] is not None else 'No aplica',f"{r['max_limit_error_K']:.6f}" if r['max_limit_error_K'] is not None else 'No aplica',f"{r['accepted']}/3"] for r in summary]),
        '## Comparaciones pareadas por semilla','',
        'La reducción positiva indica menor RMSE; una reducción negativa indica empeoramiento. Se distingue el control fijo de la renovación aleatoria.','',
        table(['Caso','Muestreo','Semilla','RMSE K','Reducción frente a fijo %','Reducción frente a renovación %','Aceptada'],[[r['case'],LABELS[r['mode']],r['seed'],f"{r['rmse_K']:.6f}",f"{r['rmse_reduction_vs_fixed_pct']:.3f}",f"{r['rmse_reduction_vs_refresh_pct']:.3f}",r['accepted']] for r in result['paired']])]
    tables=ROOT/'Tesis_LaTeX_Borrador_UNI/tablas';text=r'\subsection{COMPARACIÓN DEL MUESTREO ADAPTATIVO}'+'\n';conclusions=[]
    latex_table(tables/'benchmark_adaptive.tex',['Caso / muestreo','RMSE (K)',r'$E_{R,\max}$ (\%)','Aceptadas'],
        [[r['case']+' / '+LABELS[r['mode']],f"{r['median_rmse_K']:.4f}",f"{r['max_electrical_residual_pct']:.4f}",f"{r['accepted']}/3"] for r in summary],
        'Muestreo con igual arquitectura, número interior y presupuesto; tres semillas por alternativa','tab:muestreo-adaptativo')
    text+=r'\input{tablas/benchmark_adaptive}'+'\n'
    for name in STUDIES:
        rr={r['mode']:r for r in summary if r['case']==name};base=rr['fixed']
        findings=f"En {name}, la nube fija obtiene un RMSE mediano de {base['median_rmse_K']:.4f} K y acepta {base['accepted']}/3 semillas. "
        findings+='; '.join(f"{LABELS[mode].lower()}: {rr[mode]['median_rmse_K']:.4f} K y {rr[mode]['accepted']}/3 aceptadas" for mode in MODES[1:])+'.'
        lines+=['',findings,'',f'![Errores y restricciones: {name}](../Tesis_LaTeX_Borrador_UNI/imagenes/benchmarks/{name}_adaptive.png)','',f'![Nubes de colocación: {name}](../Tesis_LaTeX_Borrador_UNI/imagenes/benchmarks/{name}_collocation.png)']
        for mode in MODES[1:]:
            change=100*(1-rr[mode]['median_rmse_K']/base['median_rmse_K'])
            lines+=['',f"{LABELS[mode]}: la reducción del RMSE mediano frente a la nube fija es {change:.2f} %; la aceptación pasa de {base['accepted']}/3 a {rr[mode]['accepted']}/3. Se conserva el signo negativo cuando el error aumenta."]
        text+='\n'+findings.replace(name,r'\texttt{'+name.replace('_',r'\_')+'}')+'\n'
        conclusions.append(findings.replace(name,r'\texttt{'+name.replace('_',r'\_')+'}'))
        if STUDIES[name]['family']=='ampacity':
            extra='Los errores máximos de corriente son '+', '.join(f"{LABELS[mode].lower()}: {rr[mode]['max_current_error_pct']:.3f} %" for mode in MODES)+'.'
            lines+=['',extra];text+=extra.replace('%',r'\,\pct{}')+'\n'
        selection=json.loads((BASE/'selection.json').read_text(encoding='utf-8'))
        chosen=selection['ampacity_cases' if STUDIES[name]['family']=='ampacity' else 'cases'][name]
        selected_mode=next((mode for mode in MODES if directory(name,mode)==chosen),None)
        if selected_mode:
            decision=f"La configuración seleccionada utiliza {LABELS[selected_mode].lower()}, con {rr[selected_mode]['accepted']}/3 ejecuciones aceptadas. "
            if rr[selected_mode]['accepted']<3:decision+='La aceptación parcial exige contrastar cada ejecución con FEM y no acredita robustez entre semillas.'
            else:decision+='El cumplimiento en estas tres semillas permanece limitado al caso y presupuesto evaluados.'
            lines+=['',decision];text+='\n'+decision+'\n'
            conclusions.append(decision)
        text+=r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{benchmarks/'+name+r'_collocation.png}\caption{Distribución inicial y final en '+name.replace('_',r'\_')+r', semilla 11. Las isolíneas muestran la conductividad. Fuente: Elaboración propia a partir de coordenadas guardadas.}\end{figure}'+'\n'
    discussion='La redistribución se evalúa como una decisión de entrenamiento, no como una mejora garantizada. Concentrar puntos donde el gradiente térmico es alto puede reforzar regiones próximas a los cables cuya variación ya representa el enriquecimiento analítico. El residuo prioriza incumplimientos de la ecuación, pero no mide directamente el error de temperatura ni asegura el cumplimiento eléctrico. La renovación aleatoria permite identificar mejoras que no requieren un indicador físico. La selección utiliza todas las semillas y todos los criterios, sin equiparar una mejora del RMSE con aceptación de corriente.'
    text+='\n'+discussion+'\n';lines+=['','## Interpretación, ventajas y limitaciones','',discussion,'',
        'Ventajas: redistribución sin aumentar el tamaño del sistema de colocaciones, control aleatorio comparable y trazabilidad de candidatos, probabilidades y coordenadas. Limitaciones: dos casos elegidos por dificultades previas, dos actualizaciones, tres semillas, fronteras fijas y ausencia de garantía de error; no constituye generalización independiente ni búsqueda exhaustiva de frecuencia o fracción adaptativa.','',
        '## Auditoría y reproducción','',
        'Se reconstruye exactamente la nube inicial y cada selección a partir de las semillas guardadas. Para residuo y gradiente se archiva además el estado de la red antes de cada actualización; se recalculan 96 puntuaciones por ronda y se contrastan con las guardadas. En el control aleatorio la puntuación es uno y no depende del estado de la red. El algoritmo geométrico de los controles fijos se contrasta con su fuente original archivada antes de reconstruir su nube.','',
        'Comandos: `python Benchmarks/adaptive_campaign.py`, `python Benchmarks/select_configuration.py`, `python Benchmarks/adaptive_analysis.py`. Las configuraciones individuales están en `configurations/adaptive/`; los cuadernos de ambos casos permiten repetirlas en otra carpeta. Las ejecuciones mínimas de comprobación del código están en `docs/auditoria/adaptive_smoke`, fuera de la batería científica.','',
        'El JSON `summary/adaptive_sampling.json` contiene controles de replay, huellas de entradas y comparaciones completas. Los tiempos guardados incluyen evaluación de indicadores y concurrencia; no se usan para afirmar aceleración frente a FEM.','',
        '## Referencias','',REFERENCES['wu2022']]
    (BASE/'MUESTREO_ADAPTATIVO.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    text=re.sub(r'(?<=\d)\.(?=\d)',',',text)
    (tables/'benchmark_adaptive_findings.tex').write_text(text,encoding='utf-8')
    conclusion='El estudio de colocación incorpora dieciocho entrenamientos adicionales, con seis controles fijos.\n\n'+ '\n\n'.join(conclusions)+'\n\nLa evidencia distingue mejorar el campo de satisfacer todos los criterios y no sustenta adoptar un indicador adaptativo de manera universal.'
    (tables/'benchmark_adaptive_conclusion.tex').write_text(re.sub(r'(?<=\d)\.(?=\d)',',',conclusion)+'\n',encoding='utf-8')

if __name__=='__main__':main()
