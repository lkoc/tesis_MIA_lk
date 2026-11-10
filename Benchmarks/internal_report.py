"""Internal Markdown source report: all cases, candidates, failures and plots."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases,field_k,source
from Benchmarks.report import ROOT,BASE,records,selected_directory
from Benchmarks.citations import case_citations,REFERENCES

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows])+'\n'

def material_plot(name,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=cases()[name];data=np.load(BASE/'results'/name/'fem_l2.npz');xy=data['xy']
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    for ax,vals,label in zip(axes,[field_k(c,xy[:,0],xy[:,1]),source(c,xy[:,0],xy[:,1])],['k [W/(m K)]','Q [W/m³]']):
        vals=np.broadcast_to(np.asarray(vals),len(xy))
        sc=ax.scatter(xy[:,0],xy[:,1],c=vals,s=3,cmap='viridis',rasterized=True)
        ax.set(xlabel='x (m)',ylabel='y (m)',title=label);ax.set_aspect('equal');fig.colorbar(sc,ax=ax)
    if c['kind']=='cable':fig.suptitle(name+': Q=0 en suelo; calor inyectado en fronteras circulares')
    else:fig.suptitle(name)
    fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)

def main():
    rows=records();catalog=cases();summary=json.loads((BASE/'summary/summary.json').read_text())
    selected={r['case']:r for r in summary['selected']}
    assets=BASE/'summary/figures';assets.mkdir(exist_ok=True)
    lines=['# Informe interno de todos los casos y configuraciones calculados','',
    'Este documento es el expediente de análisis del que se extraen tablas, figuras y conclusiones para la tesis. Se genera con `python Benchmarks/internal_report.py` a partir de resultados guardados; no contiene resultados simulados para completar la redacción.','',
    f"Batería: **{len(catalog)} casos**, **{summary['fem_solves']} soluciones FEM de verificación**, **66 expedientes FEM acoplados** y **{len(rows)} entrenamientos evaluados**. Los 66 expedientes acoplados incluyen operación y ampacidad en tres mallas. Construir sus matrices y resolver ambos estados requiere 150 sistemas lineales en total, incluidas las excitaciones unitarias. La cantidad de entrenamientos por caso aparece en cada tabla; los intentos interrumpidos se documentan al final.",'',
    '## Reglas de interpretación','',
    '- Referencia física única en JSON; temperaturas FEM excluidas del entrenamiento PINN.',
    '- Tres mallas P2 y 6000 puntos de comparación independientes del entrenamiento.',
    '- Puerta térmica: NRMSE ≤ 5 %, error del incremento máximo ≤ 5 % y balance ≤ 2 %. Se informan además máximos locales y residuos.',
    '- Semilla 5: exploración; semillas 11, 23 y 37: sensibilidad. Ajustar y evaluar sobre un mismo caso no prueba generalización.',
    '- Los cables usan flujo circular uniforme y reconstrucción radial de capas. Se separa esta reducción de un FEM multicapa completo.',
    '- La operación y la ampacidad DC actualizan R(T) individualmente. Los ensayos a R20 fija se identifican como controles térmicos; no incluyen el acoplamiento.',
    '- Acoplamiento: residuo eléctrico ≤ 0,1 %. Ampacidad: además, error de corriente frente a FEM ≤ 5 % y distancia a 90 °C ≤ 0,1 K. No se calcula ampacidad IEC completa.',
    '- En la tabla de entrenamientos, «Pasa» corresponde a la puerta térmica o electrotérmica; «Pasa corriente» incorpora además los criterios de corriente límite.',
    '- Los tiempos incluyen concurrencia y entornos distintos; no sustentan una ventaja de velocidad.','',
    '## Decisión sobre arquitectura e hiperparámetros','',
    'El menor error puntual y la menor complejidad aceptable son criterios distintos. C05 tiene el menor RMSE exploratorio entre los ocho candidatos; C04 obtiene un error próximo con menos parámetros. Se adopta C04 como configuración compacta para transferencia a casos de un cable; el escenario seco próximo conserva la alternativa de 32 neuronas que produjo menor error entre semillas. Esta es una selección observada y condicionada, no un óptimo universal.','',
    'Los seis cables necesitan representar variaciones angulares alrededor de cada superficie. La variante multipolar agrega funciones armónicas y coeficientes entrenables, manteniendo una MLP de 32×3. Se compara con un control de igual presupuesto; el piloto 64×4, con más iteraciones, se conserva incluso cuando no resulta aceptable.','',
    'La campaña acoplada usa esa base de 32×3 y agrega una potencia desconocida por conductor. Mantiene 1200 pasos Adam y hasta 1600 iteraciones L-BFGS en los once casos, con tres semillas. Es una extensión común evaluada, no una nueva búsqueda exhaustiva de arquitectura para cada corriente. La campaña C01–C08 conserva su alcance de comparación térmica a potencia fija.','']
    candidates=[r for r in rows if '/C0' in r['run']]
    lines.append(table(['Candidato','Arquitectura','Parámetros','RMSE K','Balance %','Pasa'],[[r['run'].split('/')[1],f"{r['width']}×{r['depth']}",r['parameters'],f"{r['rmse_fem_K']:.6f}",f"{r['balance_pct']:.4f}",r['thermal_criteria_pass']] for r in candidates]))
    decision=json.loads((BASE/'selection.json').read_text(encoding='utf-8'))
    lines += ['Las ocho verificaciones analíticas se repitieron en `verification_final` con fuentes archivadas. Las ejecuciones precedentes se conservan y figuran en las tablas; no se cuentan como casos físicos distintos. Las configuraciones de corriente límite con pesos 1000 se probaron también en los dos casos XLPE que incumplieron inicialmente. Esta ampliación usa el mismo presupuesto y es una decisión adaptada a la campaña observada.','']
    if decision.get('decisions'):
        lines += ['### Selección completa de configuraciones acopladas','',decision['criterion'],'',
            'Las semillas se agrupan por configuración; no se construye una configuración ficticia eligiendo la mejor semilla de cada alternativa. Los ensayos a mayor presupuesto y los cambios de pesos se identifican por separado. La variante de 64 neuronas mantiene el presupuesto inicial; el control de semilla 5 separa el efecto del presupuesto del peso eléctrico.','']
        for name,d in decision['decisions'].items():
            lines += [f'**{name}** — nominal: `{d["nominal_selected"]}`; ampacidad: `{d["ampacity_selected"]}`.','',
                table(['Modo','Campaña','Ancho','Adam/L-BFGS','Peso R(T)','Peso límite','RMSE mediano K','Máx. residuo eléctrico %','Aceptadas'],[[mode,r['directory'],r['width'],f"{r['adam']}/{r['lbfgs']}",r['electrical_weight'],r.get('limit_weight') or 'No aplica',f"{r['median_rmse_K']:.5f}",f"{r['max_electrical_residual_pct']:.5f}",f"{r['accepted']}/3"] for mode,key in [('Nominal','nominal_candidates'),('Ampacidad','ampacity_candidates')] for r in d[key]])]
    reproduction=BASE/'environment/reproduction_comparison.json'
    if reproduction.exists():
        rr=json.loads(reproduction.read_text())
        lines += ['## Reproducción en un entorno Python nuevo','',
            'Se reinstalaron PyTorch CPU, NumPy y pytest en un entorno virtual sin paquetes compartidos con el original. Las pruebas y los dos recálculos terminaron correctamente. El caso manufacturado también cambió de dos hilos a uno; no se atribuye su diferencia únicamente al paquete PyTorch. FEniCSx se recalculó en el entorno WSL declarado, sin afirmar una reinstalación de Conda.','',
            table(['Caso','PyTorch original','PyTorch aislado','Máx. diferencia de campo K','Aceptado'],[[r['case'],r['original_torch'],r['clean_torch'],f"{r['field_max_difference_K']:.8f}",r['clean_accepted']] for r in rr])]
    lines+=['## Resumen de casos seleccionados','',table(['Caso','RMSE mediano K','Desviación entre semillas K','Máx. error Tmax K','Máx. balance %','Aceptadas'],[[r['case'],f"{r['rmse_median_K']:.5f}",f"{r['rmse_std_K']:.5f}",f"{r['Tmax_error_max_K']:.4f}",f"{r['balance_max_pct']:.4f}",f"{r['accepted']}/{r['n']}"] for r in selected.values()])]
    lines += ['## Resolución, orden observado y propagación del error','',
        'El [análisis numérico completo](ANALISIS_NUMERICO.md) presenta 36 entrenamientos adicionales con controles separados de anchura, colocaciones y presupuesto. Incluye diferencias entre redes, pendientes empíricas, órdenes FEM frente a soluciones exactas, cotas triangulares y propagación de errores térmicos y eléctricos. Sus derivadas se contrastan mediante perturbaciones numéricas independientes. Los mismos entrenamientos aparecen en las tablas por caso de este informe.','']
    lines += ['## Redistribución adaptativa de colocaciones','',
        'El [expediente de muestreo adaptativo](MUESTREO_ADAPTATIVO.md) compara 18 entrenamientos nuevos con seis controles fijos: renovación aleatoria, residuo y gradiente térmico en dos escenarios difíciles. Incluye comparaciones pareadas, nubes iniciales y finales, puntuaciones y verificación de las selecciones. La evaluación FEM y las puertas de aceptación se mantienen. Todos esos resultados aparecen también en las tablas por caso.','']
    for name,c in catalog.items():
        rr=[r for r in rows if r['case']==name];chosen=selected.get(name)
        fm=[json.loads((BASE/'results'/name/f'fem_l{i}.json').read_text()) for i in range(3)]
        lines+=['',f'## Caso `{name}`','',case_citations(name)[0],'',f"Datos completos: [JSON](cases/{name}.json). Revisión ejecutable: [cuaderno](notebooks/{name}.ipynb). Directorio seleccionado: `{selected_directory(name).relative_to(BASE)}`.",'',
        '### Datos y formulación','',f"Familia: `{c['kind']}`. Ambiente: {c['T0']} °C. Conductividad: `{json.dumps(c.get('conductivity',c.get('k','preset manufacturado')),ensure_ascii=False)}`."]
        if c['kind']=='cable':lines += [f"Dominio: {c['bounds']} m; {len(c['cables'])} cable(s); radio exterior {c['radius']} m; corriente base {c['current']} A; R20={c['R20']} Ω/m; potencia de referencia a 20 °C por cable {c['power']:.7g} W/m.",f"Centros: `{c['cables']}`. Capas `[ri,ro,k]`: `{c['layers']}`.",f"Región continua: `{c.get('patch')}`. Estratos heredados: `{c.get('bands')}`. Control pareado: `{c.get('pair')}`."]
        if c.get('exact_expression'):lines += [f"Solución exacta: `{c['exact_expression']}`. Fuente: `{c['source_expression']}`."]
        lines+=['',c.get('adaptation','Caso de verificación con datos controlados.'),'','### Convergencia FEniCSx','',table(['Nivel','Grados de libertad','Tmax °C','Balance %','Tiempo observado s'],[[r['level'],r['ndofs'],f"{r['Tmax_C']:.6f}",f"{r['balance_pct']:.6f}",f"{r['elapsed_s']:.3f}"] for r in fm]),'### Todos los entrenamientos evaluados','',table(['Campaña','Semilla','Ancho×capas','RMSE K','NRMSE %','Máx. error campo K','Error Tmax K','Balance %','Pasa'],[[r['run'].rsplit('/',1)[0],r['seed'],f"{r['width']}×{r['depth']}",f"{r['rmse_fem_K']:.6f}",f"{r['nrmse_fem_pct']:.4f}",f"{r['max_error_fem_K']:.5f}",f"{r['error_Tmax_K']:.5f}",f"{r['balance_pct']:.5f}",'Sí' if r['thermal_criteria_pass'] else 'No'] for r in rr])]
        if chosen:
            lines+=['### Ventajas, limitaciones y decisión','',f"La configuración seleccionada supera la puerta térmica en {chosen['accepted']} de {chosen['n']} semillas. Su RMSE mediano es {chosen['rmse_median_K']:.5f} K; la desviación entre semillas es {chosen['rmse_std_K']:.5f} K."]
        if c['kind']=='cable':
            lines += ['### Operación y ampacidad con R(T) individual','',
                'La tabla FEM anterior corresponde al control a R20 fija. Las siguientes soluciones usan pérdidas actualizadas y constituyen las referencias operativas. Cada potencia se expresa en W/m y cada temperatura corresponde al mismo conductor, en el orden del JSON.','']
            coupled=[]
            for mode,stem in [('Nominal','fem'),('Ampacidad','fem_ampacity')]:
                for level in range(3):
                    p=BASE/'coupled_results'/name/f'{stem}_l{level}.json';r=json.loads(p.read_text())
                    coupled.append([mode,level,f"{r['current_A']:.4f}",f"{r['Tmax_C']:.5f}",f"{r['balance_pct']:.5f}",f"{r['electrical_residual_pct']:.3g}"])
                    if level==2:lines += [f"**{mode}, malla fina:** Tc = {np.round(r['conductor_C'],6).tolist()} °C; P = {np.round(r['powers_W_m'],6).tolist()} W/m. Metadatos: [{p.name}]({p.relative_to(BASE).as_posix()})."]
            lines += ['',table(['Modo FEM','Malla','Corriente A','Tmax °C','Balance %','Residuo eléctrico %'],coupled)]
            operational=[r for r in rr if r['mode']!='fixed_source']
            lines += [table(['Campaña','Semilla','Modo','Corriente A','Error I %','Residuo eléctrico %','Distancia a 90 °C K','Pasa corriente'],[[r['run'].rsplit('/',1)[0],r['seed'],r['mode'],f"{r['current_A']:.4f}",fmt(r['error_current_pct']),fmt(r['electrical_residual_pct']),fmt(r['temperature_limit_error_K']),r['ampacity_criteria_pass'] if r['mode']=='ampacity' else 'No aplica'] for r in operational])]
        if c['kind']=='cable':
            boundary=[]
            ref=BASE/'coupled_results'/name/'fem_l2.json'
            for label,path in [('FEM nominal',ref)]+[(p.stem,p) for p in sorted(selected_directory(name).glob('pinn_seed*.json'))]:
                for b in json.loads(path.read_text()).get('boundary_diagnostics',[]):
                    if 'relative_power_error_pct' in b:
                        boundary.append([label,b['boundary'],fmt(b['prescribed_heat_into_soil_W_m']),fmt(b['computed_heat_into_soil_W_m']),fmt(b['relative_power_error_pct']),fmt(b.get('flux_rmse_W_m2'))])
            lines += ['### Balance por conductor y flujo local','',
                'Diagnóstico adicional para detectar compensaciones entre fronteras. No se añadió retrospectivamente a la puerta de aceptación. La integración FEM usa el gradiente de la solución P2; el flujo local PINN se evalúa sobre 512 puntos por circunferencia.','',
                table(['Método/semilla','Frontera','P impuesta W/m','P integrada W/m','Error potencia %','RMSE flujo W/m²'],boundary)]
        if c['kind'].startswith('mms'):
            lines+=['**Ventaja:** existe una solución exacta independiente y se pueden separar errores de fuente, derivadas y contornos. **Limitación:** la geometría regular y los campos prescritos no reproducen por sí mismos el problema completo del cable.']
            if name in ['mms_variable','mms_smooth_2d','mms_high_contrast']:lines+=['La variación continua de k se evalúa conservando sus derivadas. Los resultados muestran el comportamiento de estos perfiles y contrastes; no demuestran exactitud para cualquier frecuencia espacial, anisotropía o dependencia k(T).']
            if 'interface' in name or 'layered' in name:lines+=['El salto exacto exige continuidad de temperatura y flujo; la red global se conserva como ablación en el caso vertical. La fuente manufacturada se deriva por región y no introduce una fuente superficial.']
        elif c['kind']=='annulus':lines+=['**Ventaja:** referencia logarítmica exacta y comprobación de flujo. **Limitación:** la PINN usa simetría radial y el muestreo de área no concentra puntos cerca del radio interior; se evalúa esa frontera por separado.']
        else:
            lines+=['**Ventaja:** ambos métodos comparten geometría, pérdidas, materiales y reconstrucción del conductor. **Limitación:** la frontera de flujo uniforme y las resistencias radiales reducen la física de las capas; las pérdidas son DC y no se reconstruye el ensayo publicado.']
            if len(c['cables'])>1:lines+=['Las interacciones entre cables exigen capacidad local adicional. El enriquecimiento multipolar conserva la potencia de la fuente y permite ajustar variaciones angulares; los residuos locales y máximos deben revisarse además del promedio espacial.']
            if c.get('pair'):lines+=[f"La interpretación del efecto material debe compararse con `{c['pair']}`, conservando corriente, dominio y contornos."]
        lines+=['','### Gráficos y archivos de auditoría','',f'![FEM, PINN y error de {name}](../Tesis_LaTeX_Borrador_UNI/imagenes/benchmarks/{name}.png)','', 'Fuente: elaboración propia, misma nube de evaluación; la figura usa la primera semilla guardada del directorio seleccionado.','']
        material_plot(name,assets/f'{name}_materials.png')
        lines += [f'![Conductividad y fuente de {name}](summary/figures/{name}_materials.png)','', 'Fuente: elaboración propia a partir del JSON; en los casos de cables la generación se aplica en las fronteras, no en el volumen de suelo.','',f"Campos y mallas: `results/{name}/fem_l*.npz`. Pesos e historiales: `{selected_directory(name).relative_to(BASE)}/pinn_seed*.pt` y `training_seed*.json`. Las tablas completas están en [all_runs.csv](summary/all_runs.csv)."]
    lines+=['','## Ampacidad DC acoplada y efecto de las pérdidas','', 'FEniCSx construye una matriz de respuesta térmica con excitaciones unitarias, actualiza las pérdidas hasta converger y comprueba el punto fijo contra una solución matricial cerrada. La bisección encuentra la corriente que lleva el conductor más caliente a 90 °C. PINN aprende corriente, potencias y campo sin temperaturas FEM de entrenamiento. Se conservan los intentos rechazados; la mediana de corriente solo incluye los aceptados.','']
    with (BASE/'summary/ampacity_coupled.csv').open(encoding='utf-8-sig') as f:amps=list(csv.DictReader(f))
    lines.append(table(['Caso','FEM A','PINN mediana A','Aceptadas','Total','Máx. error I %'],[[r['case'],r['fem_A'],r['pinn_median_A'],r['accepted'],r['n'],r['error_current_max_pct']] for r in amps]))
    with (BASE/'summary/temperature_dependent_losses.csv').open(encoding='utf-8-sig') as f:losses=list(csv.DictReader(f))
    lines += ['### Consecuencia de fijar indebidamente R20','',table(['Caso','T con R20 °C','T con R(Tc) °C','ΔT K','P20 W/m','P mínima W/m','P máxima W/m'],[[r[k] for k in ['case','T_R20_C','T_coupled_C','delta_T_K','P20_W_m','P_coupled_min_W_m','P_coupled_max_W_m']] for r in losses])]
    lines += ['### Índice histórico con resistencia uniforme a 90 °C','',
        'El script `ampacity.py` conserva un índice condicional obtenido por escalado de los campos a fuente fija. Usa R(90 °C) común a todos los conductores; no resuelve su acoplamiento individual. Sus once resultados se guardan en [ampacity_conditional.csv](summary/ampacity_conditional.csv) y su historial en [ampacity_history.json](summary/ampacity_history.json). No se incluyen en la selección ni sustituyen la tabla acoplada de la tesis. El script exige entradas a R20 fija y genera una tabla separada para evitar mezclar los dos cálculos.','']
    lines+=['','## Casos históricos del proyecto','', 'Los ocho directorios de `examples` se mantienen. Sus salidas no se agregan a los resultados verificados porque no comparten necesariamente especificación o referencia convergente. El inventario siguiente permite revisar los antecedentes, incluidos los no seleccionados.','']
    historical=[]
    for folder in sorted((ROOT/'examples').iterdir()):
        if not folder.is_dir() or folder.name.startswith('__'):continue
        files=[p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.md','.ipynb'] and ('result' in str(p).lower() or p.suffix in ['.md','.ipynb'])]
        historical.append(dict(case=folder.name,files=[str(p.relative_to(ROOT)).replace('\\','/') for p in files]))
        lines += [f"### Antecedente `{folder.name}`",'', 'Estado: antecedente conservado; no se considera verificación de la batería común.','']
        lines.extend(f'- [{p.name}](../{p.relative_to(ROOT).as_posix()})' for p in files)
    (BASE/'summary/historical_examples.json').write_text(json.dumps(historical,ensure_ascii=False,indent=2),encoding='utf-8')
    validation=json.loads((BASE/'summary/artifact_validation.json').read_text(encoding='utf-8'))
    lines += ['## Auditoría independiente de archivos','',
        f"Se comprobaron {validation['counts']['pinn']} registros PINN, 57 referencias FEM de verificación y 66 acopladas. Las 90 ejecuciones de la selección principal tienen fuentes de entrenamiento archivadas y verificadas. En {len(validation['historical_training_source_not_archived'])} registros históricos la fuente original no quedó completamente archivada; sus pesos y campos permanecen, y la reevaluación actual sí tiene código identificado.",
        'Las comprobaciones recalculan RMSE y máximos desde NPZ, verifican la referencia FEM y su hash, la superposición de respuestas y la ley R(T) por conductor. El detalle de excepciones históricas está en [artifact_validation.json](summary/artifact_validation.json). Las fuentes recuperadas de otros expedientes se copiaron solo cuando su hash coincidía exactamente; la procedencia está en [source_recovery.json](environment/source_recovery.json).','']
    lines+=['','## Incidencias y límites de procedencia','',
    '- El primer entrenamiento del anillo se detuvo al detectar nubes de evaluación distintas; se guardó el peso y se repitió la evaluación con la nube común.',
    '- El piloto multipolar inicial se interrumpió antes de obtener resultados para precalcular las funciones geométricas y sus derivadas. La versión evaluada conserva esa optimización algebraica.',
    '- La primera ejecución de los perfiles continuos se archivó antes de migrar a expresiones explícitas; se recalcularon FEM y PINN con el nuevo JSON.',
    '- Las primeras ejecuciones exploratorias guardaron datos, semillas y pesos, pero no todas archivaron el código exacto cargado. Las ejecuciones finales posteriores archivan fuentes por SHA-256; no se fabricaron retrospectivamente huellas de versiones anteriores.',
    '- El adjunto histórico identificado como Raissi (2019) corresponde realmente a la prepublicación de 2017. Los cuadernos citan el documento y las páginas efectivamente consultados.',
    '- Los resultados numéricos demuestran verificación del modelo definido; no constituyen validación de campo ni una certificación profesional de ampacidad.','',
    '## Referencias','', '\n\n'.join(REFERENCES[k] for k in sorted(REFERENCES,key=lambda k:REFERENCES[k]))]
    (BASE/'INFORME_INTERNO.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('Internal report written:',len(catalog),'cases,',len(rows),'runs')

def fmt(value):return '—' if value is None else f'{value:.5f}'

if __name__=='__main__':main()
