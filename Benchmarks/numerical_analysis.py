"""Numerical sensitivity, empirical rates and thermal/electrical error budgets."""
from pathlib import Path
import argparse,json,sys,hashlib
from datetime import datetime,timezone
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases,exact
from Benchmarks.report import BASE,ROOT,records,write_csv,latex_table,selected_directory

def scientific_latex(value):
    mantissa,exponent=f'{value:.2e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'

def thermal_state(c,a,current):
    p0=current**2*c['R20'];matrix=np.eye(len(a))-p0*c['alpha']*a
    p=np.linalg.solve(matrix,np.full(len(a),p0*(1+c['alpha']*(c['T0']-20))))
    return c['T0']+a@p,p

def tangent(c,a,current,direction):
    t,p=thermal_state(c,a,current);matrix=np.eye(len(a))-current**2*c['R20']*c['alpha']*a
    dt=np.linalg.solve(matrix,direction@p)
    di=np.linalg.solve(matrix,a@(2*current*c['R20']*(1+c['alpha']*(t-20))))
    return dt,di

def matrix_ampacity(c,a,limit=90.):
    # Bracket below the linear model's singular current; positive response A.
    radius=max(abs(np.linalg.eigvals(a)))
    stable=np.sqrt(1/(c['R20']*c['alpha']*radius)) if c['alpha'] else c['current']*100
    return brentq(lambda i:max(thermal_state(c,a,i)[0])-limit,0,stable*.999,xtol=1e-10)

def propagation():
    summary=[];budgets=[];meshes=[];catalog=cases()
    for name,c in catalog.items():
        if c['kind']!='cable':continue
        fine=json.loads((BASE/'coupled_results'/name/'fem_l2.json').read_text())
        coarse=json.loads((BASE/'coupled_results'/name/'fem_l1.json').read_text())
        a=np.array(fine['response_K_m_W']);da=np.array(coarse['response_K_m_W'])-a
        p0=c['current']**2*c['R20'];matrix=np.eye(len(a))-p0*c['alpha']*a;inverse=np.linalg.inv(matrix)
        gain=np.linalg.norm(inverse,np.inf)
        summary.append(dict(case=name,amplification_inf=float(gain),condition_inf=float(np.linalg.cond(matrix,np.inf)),spectral_radius=float(max(abs(np.linalg.eigvals(p0*c['alpha']*a))))))
        for path in sorted(selected_directory(name).glob('pinn_seed*.json')):
            r=json.loads(path.read_text());t=np.array(r['conductor_C']);p=np.array(r['powers_W_m'])
            e=t-c['T0']-a@p;rp=p-p0*(1+c['alpha']*(t-20))
            thermal=inverse@e;electrical=inverse@a@rp;actual=t-np.array(fine['conductor_C'])
            np.testing.assert_allclose(thermal+electrical,actual,atol=1e-8)
            budgets.append(dict(case=name,seed=r['metadata']['seed'],thermal_contribution_K=float(max(abs(thermal))),electrical_contribution_K=float(max(abs(electrical))),actual_temperature_error_K=float(max(abs(actual))),decomposition_error_K=float(max(abs(thermal+electrical-actual))),norm_upper_bound_K=float(gain*(max(abs(e))+max(abs(a@rp))))))
        dt,_=tangent(c,a,c['current'],da);actual=np.array(coarse['conductor_C'])-np.array(fine['conductor_C'])
        current=matrix_ampacity(c,a);tamp,_=thermal_state(c,a,current);dta,dti=tangent(c,a,current,da)
        active=np.flatnonzero(max(tamp)-tamp<=1e-7)
        predicted=min(-dta[active]/dti[active]);actual_i=matrix_ampacity(c,a+da)-current
        eps=1e-3
        numerical=(thermal_state(c,a+eps*da,c['current'])[0]-thermal_state(c,a-eps*da,c['current'])[0])/(2*eps)
        np.testing.assert_allclose(numerical,dt,atol=1e-7,rtol=1e-5)
        meshes.append(dict(case=name,mesh_temperature_change_K=float(max(abs(actual))),linearized_temperature_change_K=float(max(abs(dt))),linearization_remainder_K=float(max(abs(dt-actual))),mesh_current_change_A=float(actual_i),linearized_current_change_A=float(predicted),current_remainder_A=float(abs(predicted-actual_i)),active_conductors=active.tolist(),tangent_central_difference_error_K=float(max(abs(numerical-dt)))))
    write_csv(BASE/'summary/coupling_amplification.csv',summary);write_csv(BASE/'summary/pinn_error_propagation.csv',budgets);write_csv(BASE/'summary/mesh_error_propagation.csv',meshes)
    latex_table(ROOT/'Tesis_LaTeX_Borrador_UNI/tablas/benchmark_error_propagation.tex',['Caso',r'$\|M^{-1}\|_\infty$',r'$e_{T,\max}$ (K)','Aporte térmico (K)','Aporte eléctrico (K)'],[[r['case'],f"{r['amplification_inf']:.3f}",f"{max(b['actual_temperature_error_K'] for b in budgets if b['case']==r['case']):.4f}",f"{max(b['thermal_contribution_K'] for b in budgets if b['case']==r['case']):.4f}",f"{max(b['electrical_contribution_K'] for b in budgets if b['case']==r['case']):.4f}"] for r in summary],'Descomposición del error nominal PINN: máximos entre conductores y semillas; los aportes pueden compensarse','tab:propagacion-error')
    return summary,budgets,meshes

def fem_orders():
    output=[];budgets=[]
    for name,c in cases().items():
        if c['kind']=='cable':continue
        errors=[]
        for level in range(3):
            d=np.load(BASE/'results'/name/f'fem_l{level}.npz');ex=exact(c,d['xy'][:,0],d['xy'][:,1]);errors.append(float(np.sqrt(np.mean((d['T']-ex)**2))))
        regular=c['kind'].startswith('mms')
        rates=[float(np.log(errors[j]/errors[j+1])/np.log(2)) if regular and min(errors[j:j+2])>1e-9 else None for j in range(2)]
        output.append(dict(case=name,rmse_l0_K=errors[0],rmse_l1_K=errors[1],rmse_l2_K=errors[2],p01=rates[0],p12=rates[1],interpretation='Uniform square mesh h=1/(16*2**level); sampled L2 error' if regular else 'Curved unstructured mesh: no uniform h order assigned'))
        for path in sorted(selected_directory(name).glob('pinn_seed*.json')):
            r=json.loads(path.read_text());epf=r['rmse_fem_K'];epx=r['rmse_exact_K']
            low=abs(epf-errors[2]);high=epf+errors[2]
            assert low-1e-8<=epx<=high+1e-8
            budgets.append(dict(case=name,seed=r['metadata']['seed'],pinn_fem_K=epf,fem_exact_K=errors[2],pinn_exact_K=epx,triangle_lower_K=low,triangle_upper_K=high))
    write_csv(BASE/'summary/fem_observed_orders.csv',output);write_csv(BASE/'summary/exact_error_budget.csv',budgets)
    latex_table(ROOT/'Tesis_LaTeX_Borrador_UNI/tablas/benchmark_observed_orders.tex',['Caso','$E_0$ (K)','$E_1$ (K)','$E_2$ (K)','$p_{01}$','$p_{12}$'],[[r['case'],*[scientific_latex(r[k]) for k in ['rmse_l0_K','rmse_l1_K','rmse_l2_K']],*[f"{r[k]:.3f}" if r[k] is not None else '---' for k in ['p01','p12']]] for r in output],'Orden observado FEM con solución exacta; se omiten órdenes al nivel de redondeo o sin refinamiento uniforme','tab:orden-observado')
    return output,budgets

def resolution():
    rr=records();groups=[];pairs=[];rates=[]
    for name in ['mms_smooth_2d','xlpe_single']:
        grouped={}
        c=cases()[name];ref=json.loads((BASE/('coupled_results' if c['kind']=='cable' else 'results')/name/'fem_l2.json').read_text());rise=ref['Tmax_C']-c['T0']
        for label in ['W16','W32','W64','N384','N1536','B2']:
            r=[x for x in rr if x['run']==f'comparisons/resolution_{label}/{name}']
            if len(r)!=3:raise ValueError(f'Incomplete resolution study {name}/{label}')
            grouped[label]={x['seed']:x for x in r}
            errors=[x['rmse_exact_K'] if c['kind']!='cable' else x['rmse_fem_K'] for x in r]
            groups.append(dict(case=name,configuration=label,width=r[0]['width'],parameters=r[0]['parameters'],n_interior=r[0]['n_interior'],adam=r[0]['adam'],lbfgs=r[0]['lbfgs'],median_error_K=float(np.median(errors)),std_error_K=float(np.std(errors,ddof=1)),min_error_K=min(errors),max_error_K=max(errors),accepted=sum(x['thermal_criteria_pass'] for x in r)))
        for left,right in [('W16','W32'),('W32','W64'),('N384','W32'),('W32','N1536'),('W32','B2')]:
            for seed in [11,23,37]:
                a=grouped[left][seed];b=grouped[right][seed]
                da=np.load(ROOT/Path(a['path']).with_suffix('.npz'));db=np.load(ROOT/Path(b['path']).with_suffix('.npz'))
                difference=float(np.sqrt(np.mean((da['T']-db['T'])**2)));tmax=abs(a['Tmax_C']-b['Tmax_C'])
                stable=a['thermal_criteria_pass'] and b['thermal_criteria_pass'] and 100*difference/rise<=.5 and 100*tmax/rise<=.5
                pairs.append(dict(case=name,comparison=left+'-'+right,seed=seed,field_difference_K=difference,field_difference_rise_pct=100*difference/rise,Tmax_difference_K=tmax,Tmax_difference_rise_pct=100*tmax/rise,within_declared_tolerance=bool(stable)))
        for left,right in [('N384','W32'),('W32','N1536')]:
            a=next(g for g in groups if g['case']==name and g['configuration']==left);b=next(g for g in groups if g['case']==name and g['configuration']==right)
            rates.append(dict(case=name,comparison=left+'-'+right,q_empirical=float(np.log(a['median_error_K']/b['median_error_K'])/np.log(b['n_interior']/a['n_interior'])),interpretation='Empirical slope under fixed optimization budget; not a theoretical order or Richardson/GCI estimate'))
    write_csv(BASE/'summary/pinn_resolution.csv',groups);write_csv(BASE/'summary/pinn_resolution_pairs.csv',pairs);write_csv(BASE/'summary/pinn_empirical_slopes.csv',rates)
    latex_table(ROOT/'Tesis_LaTeX_Borrador_UNI/tablas/benchmark_resolution.tex',['Caso / prueba','Parámetros','$N$','Error (K)','Desv. (K)','Aceptadas'],[[r['case']+' / '+r['configuration'],r['parameters'],r['n_interior'],f"{r['median_error_K']:.5f}",f"{r['std_error_K']:.5f}",f"{r['accepted']}/3"] for r in groups],'Sensibilidad PINN a anchura, colocaciones y presupuesto; error exacto en MMS y error frente a FEM en XLPE','tab:resolucion-pinn')
    pair_summary=[]
    for name in ['mms_smooth_2d','xlpe_single']:
        for comp in ['W16-W32','W32-W64','N384-W32','W32-N1536','W32-B2']:
            rs=[r for r in pairs if r['case']==name and r['comparison']==comp]
            pair_summary.append(dict(case=name,comparison=comp,max_field_difference_pct=max(r['field_difference_rise_pct'] for r in rs),max_Tmax_difference_pct=max(r['Tmax_difference_rise_pct'] for r in rs),stable_seeds=sum(r['within_declared_tolerance'] for r in rs)))
    write_csv(BASE/'summary/pinn_resolution_stability.csv',pair_summary)
    latex_table(ROOT/'Tesis_LaTeX_Borrador_UNI/tablas/benchmark_resolution_stability.tex',['Caso / cambio',r'$D_{\max}$ (\%)',r'$H_{\max}$ (\%)','Estables'],[[r['case']+' / '+r['comparison'],f"{r['max_field_difference_pct']:.4f}",f"{r['max_Tmax_difference_pct']:.4f}",f"{r['stable_seeds']}/3"] for r in pair_summary],'Diferencias entre soluciones PINN: tolerancia declarada de 0,5 % del incremento térmico y aceptación de ambos cálculos','tab:estabilidad-resolucion')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in ['mms_smooth_2d','xlpe_single']:
        fig,axs=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
        for ax,labels,key,xlabel in [(axs[0],['W16','W32','W64'],'width','Neuronas por capa'),(axs[1],['N384','W32','N1536'],'n_interior','Colocaciones interiores efectivas')]:
            data=[next(g for g in groups if g['case']==name and g['configuration']==l) for l in labels]
            x=np.array([d[key] for d in data]);y=np.array([d['median_error_K'] for d in data])
            ax.errorbar(x,y,yerr=[y-[d['min_error_K'] for d in data],[d['max_error_K'] for d in data]-y],fmt='o-',capsize=4)
            ax.set(xscale='log',yscale='log',xlabel=xlabel,ylabel='RMSE (K)');ax.grid(alpha=.2)
            ax.set_xticks(x,[str(int(value)) for value in x])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.suptitle(name+': mediana y rango de tres semillas')
        fig.savefig(ROOT/'Tesis_LaTeX_Borrador_UNI/imagenes/benchmarks'/f'{name}_resolution.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return groups,pairs,rates,pair_summary

def narrative(result):
    from Benchmarks.internal_report import table
    from Benchmarks.citations import REFERENCES
    lines=['# Análisis interno de resolución, convergencia y propagación de error','',
        'Este expediente complementa INFORME_INTERNO.md. Los cálculos se generan desde campos y metadatos, mediante `python Benchmarks/numerical_analysis.py`.','',
        '## Fundamento y alcance','',
        'La separación de capacidad, muestreo y optimización sigue De Ryck y Mishra (2024: 24–27, 51–57). Las cotas de error requieren estabilidad y control del residuo y de su cuadratura (Mishra y Molinaro, 2020: 7–8). Se calculan pendientes empíricas y errores observados; no se asigna un orden teórico universal a PINN ni se aplica GCI al número de neuronas.','',
        '## Protocolo','',
        'Se comparan 16/32/64 neuronas por capa, 384/768/1536 puntos globales y un presupuesto duplicado, en un MMS continuo y un XLPE nominal acoplado. Se conservan tres semillas, un hilo, fronteras y evaluación comunes: 36 entrenamientos. N en las tablas es el número interior efectivo. Para diferencias entre soluciones, la tolerancia es 0,5 % del incremento máximo de referencia, tanto en RMS de campo como en Tmax; además ambos entrenamientos deben cumplir los criterios físicos. La tolerancia se declaró antes de cerrar la campaña.','',
        '## Resultados de resolución','',table(['Caso','Prueba','Parámetros','N interior','Adam/L-BFGS','Error mediano K','Desv. K','Aceptadas'],[[r['case'],r['configuration'],r['parameters'],r['n_interior'],f"{r['adam']}/{r['lbfgs']}",f"{r['median_error_K']:.6f}",f"{r['std_error_K']:.6f}",f"{r['accepted']}/3"] for r in result['resolution']]),
        table(['Caso','Comparación','Máx. diferencia campo %','Máx. diferencia Tmax %','Estables'],[[r['case'],r['comparison'],f"{r['max_field_difference_pct']:.5f}",f"{r['max_Tmax_difference_pct']:.5f}",f"{r['stable_seeds']}/3"] for r in result['resolution_stability']]),
        '## Pendientes empíricas de PINN','',table(['Caso','Intervalo','q observado'],[[r['case'],r['comparison'],f"{r['q_empirical']:.5f}"] for r in result['empirical_slopes']]),
        'Un q negativo indica que el mayor muestreo produjo más error bajo el presupuesto fijado. La pendiente depende de la red entrenada, no solo de una regla de cuadratura. B2 examina si el presupuesto explica parte de esa dependencia.','']
    for name in ['mms_smooth_2d','xlpe_single']:lines += [f'![Sensibilidad de {name}](../Tesis_LaTeX_Borrador_UNI/imagenes/benchmarks/{name}_resolution.png)','']
    lines += ['## Orden observado FEM','',table(['Caso','RMSE nivel 0','RMSE nivel 1','RMSE nivel 2','p01','p12'],[[r['case'],r['rmse_l0_K'],r['rmse_l1_K'],r['rmse_l2_K'],r['p01'],r['p12']] for r in result['fem_orders']]),
        'Las tasas se calculan con h reducido a la mitad en las mallas cuadradas, sobre la misma nube. Tres niveles apoyan una tasa observada; no prueban por sí solos el régimen asintótico. El anillo no recibe un p uniforme.','',
        '## Presupuesto del error frente a una solución exacta','',table(['Caso','Semilla','PINN-FEM K','FEM-exacta K','PINN-exacta K','Cota triangular inferior','Cota triangular superior'],[[r['case'],r['seed'],r['pinn_fem_K'],r['fem_exact_K'],r['pinn_exact_K'],r['triangle_lower_K'],r['triangle_upper_K']] for r in result['exact_error_budget']]),
        '## Propagación del error electrotérmico','',
        'Para corriente nominal fija, M = identidad − I² R20 α A. La identidad ΔT = M⁻¹e + M⁻¹A r separa defecto térmico e incumplimiento eléctrico. Su suma vectorial se comprueba contra el error real frente a FEM; los máximos individuales pueden compensarse. Se calcula una cota de norma del error de temperatura del conductor dentro del modelo reducido. No es una distribución probabilística de incertidumbre de materiales.','',
        table(['Caso','Amplificación infinito','Condición infinito','Radio espectral'],[[r['case'],r['amplification_inf'],r['condition_inf'],r['spectral_radius']] for r in result['amplification']]),
        table(['Caso','Semilla','Aporte térmico K','Aporte eléctrico K','Error real K','Cota K','Defecto identidad K'],[[r['case'],r['seed'],r['thermal_contribution_K'],r['electrical_contribution_K'],r['actual_temperature_error_K'],r['norm_upper_bound_K'],r['decomposition_error_K']] for r in result['error_decomposition']]),
        '## Propagación de la diferencia entre mallas','',
        'La perturbación usa A de la malla intermedia menos A de la malla fina. Se contrasta la linealización con la solución matricial completa y la derivada de temperatura con diferencias centrales. La corriente límite se calcula mediante una raíz de la misma respuesta FEM. En empates se informa la derivada direccional del conductor limitante. La diferencia entre mallas no se convierte en una cota rigurosa del error FEM.','',
        table(['Caso','ΔT real K','ΔT lineal K','Resto K','ΔI real A','ΔI lineal A','Resto A','Activos'],[[r['case'],r['mesh_temperature_change_K'],r['linearized_temperature_change_K'],r['linearization_remainder_K'],r['mesh_current_change_A'],r['linearized_current_change_A'],r['current_remainder_A'],r['active_conductors']] for r in result['mesh_propagation']]),
        '## Ventajas y limitaciones','',
        '- Las comparaciones separan capacidad, colocación y presupuesto; conservan semillas desfavorables.',
        '- Las soluciones exactas permiten distinguir error PINN y error de referencia.',
        '- La identidad electrotérmica explica por qué el acoplamiento amplifica o compensa defectos.',
        '- Dos casos y un intervalo de tamaños no acreditan independencia para todos los cables; las interfaces y geometrías más complejas requieren el mismo protocolo.',
        '- No se varía aquí la densidad de frontera ni se garantiza optimización global. Las tasas empíricas no autorizan extrapolación asintótica.',
        '- La propagación es determinista y local para cambios de malla; no estima probabilidades ni incertidumbre física no medida.','',
        '## Referencias','',REFERENCES['deryck2024'],'',REFERENCES['mishra2020']]
    (BASE/'ANALISIS_NUMERICO.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    stable=result['resolution_stability'];orders=[r['p12'] for r in result['fem_orders'] if r['p12'] is not None]
    g=[r['amplification_inf'] for r in result['amplification']]
    text=r'''\subsection{INSENSIBILIDAD OBSERVADA A LA RESOLUCIÓN PINN}
\input{tablas/benchmark_resolution}
\input{tablas/benchmark_resolution_stability}
Las Tablas~\ref{tab:resolucion-pinn} y~\ref{tab:estabilidad-resolucion} presentan errores externos y diferencias entre configuraciones. El error de MMS se calcula frente a la solución exacta; el de XLPE se calcula frente a FEM acoplado. La estabilidad entre tamaños se interpreta junto con ambos errores.
'''
    for name in ['mms_smooth_2d','xlpe_single']:
        rr=[r for r in stable if r['case']==name]
        text+=f"\nEn \\texttt{{{name.replace('_',r'\_')}}}, {sum(r['stable_seeds'] for r in rr)} de 15 comparaciones pareadas satisfacen la tolerancia declarada y los criterios de ambos cálculos. El máximo cambio relativo de campo es {max(r['max_field_difference_pct'] for r in rr):.4f}\\,\\pct{{}} y el de temperatura máxima es {max(r['max_Tmax_difference_pct'] for r in rr):.4f}\\,\\pct{{}}.\n"
        text+=r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{benchmarks/'+name+r'_resolution.png}\caption{Sensibilidad de '+name.replace('_',r'\_')+r' a anchura y colocaciones. Fuente: Elaboración propia; mediana y rango entre tres semillas.}\end{figure}'+'\n'
    text+='\nLas pendientes empíricas frente al número de colocaciones son '+', '.join(f"{r['case'].replace('_',r'\_')} ({r['comparison']}): {r['q_empirical']:.3f}" for r in result['empirical_slopes'])+'. La variación entre intervalos y el posible signo negativo impiden atribuir a estos valores un orden universal de PINN.\n'
    text+=r'''\subsection{ORDEN OBSERVADO Y PROPAGACIÓN DEL ERROR}
\input{tablas/benchmark_observed_orders}
'''
    text+=f"Las tasas FEM entre las dos mallas finas de los manufacturados se encuentran entre {min(orders):.3f} y {max(orders):.3f}. Corresponden a errores espaciales frente a soluciones exactas y al refinamiento uniforme ejecutado. La desigualdad triangular se verifica para las ocho referencias analíticas y sus tres semillas PINN; permite cuantificar la contribución de FEM al error observado.\n"
    text+=r'\input{tablas/benchmark_error_propagation}'+'\n'
    text+=f"La amplificación nominal calculada mediante la norma de la matriz inversa varía entre {min(g):.3f} y {max(g):.3f}. La descomposición separa los defectos térmico y eléctrico, que pueden compensarse. La mayor diferencia entre la perturbación térmica linealizada de malla y el cambio completo es {scientific_latex(max(r['linearization_remainder_K'] for r in result['mesh_propagation']))} K; para corriente límite es {scientific_latex(max(r['current_remainder_A'] for r in result['mesh_propagation']))} A. Estas pequeñas diferencias corresponden a la perturbación entre las mallas ya refinadas; no constituyen una incertidumbre física de la instalación.\n"
    import re
    text=re.sub(r'(?<=\d)\.(?=\d)',',',text)
    (ROOT/'Tesis_LaTeX_Borrador_UNI/tablas/benchmark_numerical_findings.tex').write_text(text,encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--propagation-only',action='store_true');a=ap.parse_args()
    p=propagation();o=fem_orders();result=dict(amplification=p[0],error_decomposition=p[1],mesh_propagation=p[2],fem_orders=o[0],exact_error_budget=o[1])
    if not a.propagation_only:
        r=resolution();result.update(resolution=r[0],resolution_pairs=r[1],empirical_slopes=r[2],resolution_stability=r[3])
    inputs={BASE/'selection.json'}
    for name in cases():
        for folder in [BASE/'results'/name,BASE/'coupled_results'/name]:
            inputs.update(folder.glob('fem_l*.json'));inputs.update(folder.glob('fem_l*.npz'))
        for path in selected_directory(name).glob('pinn_seed*.json'):
            inputs.add(path);inputs.add(path.with_suffix('.npz'))
    if not a.propagation_only:
        inputs.update(BASE.glob('comparisons/resolution_*/*/pinn_seed*.json'))
        inputs.update(BASE.glob('comparisons/resolution_*/*/pinn_seed*.npz'))
    source_files=[Path(__file__),BASE/'cases.py',BASE/'report.py',BASE/'electrothermal.py']
    result['provenance']=dict(completed_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,numpy=np.__version__,
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)})
    for path in source_files:
        data=path.read_bytes();archive=BASE/'summary/source'/(hashlib.sha256(data).hexdigest()+'_'+path.name)
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():archive.write_bytes(data)
    (BASE/'summary/numerical_analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    if not a.propagation_only:narrative(result)
    print('Numerical analysis completed',flush=True)
if __name__=='__main__':main()
