"""Extract thesis narrative from the internal report and its audited data tables."""
from pathlib import Path
import json,csv,sys,re
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;THESIS=ROOT/'Tesis_LaTeX_Borrador_UNI'

def main():
    internal=(BASE/'INFORME_INTERNO.md').read_text(encoding='utf-8')
    data=json.loads((BASE/'summary/summary.json').read_text());selected={r['case']:r for r in data['selected']}
    if len(selected)!=19 or any(r['n']!=3 for r in selected.values()):raise ValueError('Complete three-seed nominal/verification records required')
    with (BASE/'summary/ampacity_coupled.csv').open(encoding='utf-8-sig') as f:amp={r['case']:r for r in csv.DictReader(f)}
    if len(amp)!=11 or any(int(r['n'])!=3 for r in amp.values()):raise ValueError('Complete three-seed ampacity campaign required')
    if not all(f'## Caso `{name}`' in internal for name in cases()):raise ValueError('Internal report must precede thesis extraction')
    nominal={n:json.loads((BASE/'coupled_results'/n/'fem_l2.json').read_text()) for n,c in cases().items() if c['kind']=='cable'}
    base=nominal['xlpe_single']['Tmax_C'];near=nominal['xlpe_dry_near']['Tmax_C'];far=nominal['xlpe_dry_far']['Tmax_C'];large=nominal['xlpe_dry_large']['Tmax_C'];back=nominal['xlpe_backfill']['Tmax_C']
    i_base=float(amp['xlpe_single']['fem_A']);i_near=float(amp['xlpe_dry_near']['fem_A']);i_large=float(amp['xlpe_dry_large']['fem_A']);i_back=float(amp['xlpe_backfill']['fem_A'])
    accept=sum(r['accepted'] for r in selected.values());total=sum(r['n'] for r in selected.values());ampaccept=sum(int(r['accepted']) for r in amp.values())
    partial=[n for n,r in selected.items() if r['accepted']<r['n']]
    amp_failed=[n for n,r in amp.items() if int(r['accepted'])==0]
    continuous=[selected[n] for n in ['mms_variable','mms_smooth_2d','mms_high_contrast']]
    cmin=min(r['rmse_median_K'] for r in continuous);cmax=max(r['rmse_median_K'] for r in continuous)
    numerical=json.loads((BASE/'summary/numerical_analysis.json').read_text(encoding='utf-8'))
    stable=sum(r['stable_seeds'] for r in numerical['resolution_stability'])
    gains=[r['amplification_inf'] for r in numerical['amplification']]
    esc=lambda n:n.replace('_',r'\_')
    text=fr'''La temperatura FEM del XLPE homogéneo es {base:.3f} °C con pérdidas acopladas. La zona seca próxima eleva el resultado a {near:.3f} °C, mientras la misma zona desplazada alcanza {far:.3f} °C. Ampliar la región seca produce {large:.3f} °C; el relleno mejorado reduce el resultado a {back:.3f} °C.

En seis cables, la temperatura máxima pasa de {nominal['kim_sand']['Tmax_C']:.3f} °C en arena a {nominal['kim_pac']['Tmax_C']:.3f} °C con relleno mejorado. El suelo estratificado regularizado alcanza {nominal['kim_layered']['Tmax_C']:.3f} °C. Cada conductor conserva su propia potencia actualizada en estos resultados.

Las corrientes FEM correspondientes al XLPE homogéneo y a la zona seca próxima son {i_base:.2f} y {i_near:.2f} A. La reducción es {100*(1-i_near/i_base):.2f}\,\pct{{}} bajo los supuestos comunes. La zona seca ampliada alcanza {i_large:.2f} A y el relleno mejorado {i_back:.2f} A.

Los perfiles manufacturados continuos presentan medianas de RMSE frente a FEM entre {cmin:.4f} y {cmax:.4f} K. La comparación incluye el gradiente de conductividad y la fuente correspondiente a cada perfil. Estos resultados delimitan la verificación a las funciones y contrastes efectivamente ejecutados.

La selección principal acepta {accept} de {total} ejecuciones térmicas o electrotérmicas. La selección principal de corriente límite acepta {ampaccept} de 33 intentos al aplicar todos sus criterios. Las proporciones incluyen los fallos y no se interpretan como una probabilidad estadística de éxito fuera de estos casos.
'''
    if partial:text+='\nLos casos con aceptación parcial son '+', '.join(r'\texttt{'+esc(n)+'}' for n in partial)+'. Su uso requiere revisar la semilla y el defecto que impidió cumplir el criterio, según el expediente digital.\n'
    if amp_failed:text+='\nNo se obtiene una corriente PINN aceptada en '+', '.join(r'\texttt{'+esc(n)+'}' for n in amp_failed)+'. Las corrientes FEM de esos escenarios se conservan como referencias; la campaña no acredita una sustitución por PINN.\n'
    spanish=lambda value:re.sub(r'(?<=\d)\.(?=\d)',',',value)
    (THESIS/'tablas/benchmark_operational_findings.tex').write_text(spanish(text),encoding='utf-8')
    conclusions=fr'''\chapter{{CONCLUSIONES Y RECOMENDACIONES}}
\label{{ch:conclusiones}}

\section{{CONCLUSIONES}}

El artefacto permite comparar PINN y FEniCSx sobre una especificación física común de conducción estacionaria. La batería comprende diecinueve casos, tres mallas FEM por caso y ejecuciones PINN con tres semillas. La selección principal satisface los criterios en {accept} de {total} intentos, lo que sustenta una utilidad condicionada a la configuración y al caso.

El OE1 se concreta en archivos JSON para geometría, materiales, fuentes y contornos, junto con configuraciones numéricas y huellas de procedencia. El formato distingue fórmulas continuas de estratos con salto exacto. Esa distinción evita que un suavizado implícito convierta la comparación en dos problemas físicos diferentes.

El OE2 muestra que el campo, la temperatura máxima y la conservación deben evaluarse simultáneamente. En la exploración XLPE, agregar balance y ajustar el peso de la PDE reduce el RMSE de 0,6713 a 0,0367 K. El resultado respalda esas modificaciones en el caso probado, sin demostrar que una arquitectura sea óptima para todos los entornos.

La configuración compacta de 16 neuronas por capa ofrece un compromiso útil en las pruebas a fuente fija, mientras el enriquecimiento multipolar permite representar mejor las interacciones entre cables. Los saltos exactos requieren derivadas laterales y continuidad de flujo. Los perfiles continuos evaluados alcanzan medianas de RMSE entre {cmin:.4f} y {cmax:.4f} K frente a FEM, conservando el término asociado con el gradiente de conductividad.

El OE3 identifica el efecto conjunto de proximidad y extensión de la heterogeneidad. En el XLPE, la referencia acoplada pasa de {base:.3f} °C en suelo homogéneo a {near:.3f} °C con zona seca próxima y a {large:.3f} °C al ampliarla. El relleno mejorado produce {back:.3f} °C bajo la misma corriente y los mismos contornos.

La actualización individual de resistencia modifica la temperatura y resulta necesaria para interpretar la operación. FEniCSx y PINN incorporan la misma ley R(T), pero la segunda debe satisfacerla junto con sus residuos térmicos. Un balance global aceptable no permite omitir una discrepancia eléctrica ni atribuir a todos los conductores la temperatura máxima.

El OE4 obtiene ampacidades DC acopladas y cuantifica sus diferencias en pares controlados. La referencia XLPE disminuye de {i_base:.2f} a {i_near:.2f} A al introducir la zona seca próxima, una reducción de {100*(1-i_near/i_base):.2f}\,\pct{{}}. La selección principal PINN de corriente límite cumple todos sus criterios en {ampaccept} de 33 intentos; los restantes se mantienen como resultados no aceptados.

La contribución profesional es un procedimiento auditable para formular, comparar y clasificar soluciones térmicas. Los cuadernos, registros y fuentes permiten revisar tanto los resultados aceptados como los fallos. La evidencia no demuestra validación de campo, generalización paramétrica, pérdidas AC completas ni una ventaja de tiempo frente a FEM.

El estudio adicional de resolución comprende treinta y seis entrenamientos y satisface el criterio de insensibilidad en {stable} de 30 comparaciones pareadas. La similitud entre redes debe contrastarse con errores externos y sensibilidad al presupuesto; las pendientes de muestreo no constituyen un orden teórico universal. El orden FEM observado en los manufacturados es próximo a tres. La descomposición electrotérmica obtiene factores de amplificación nominal entre {min(gains):.3f} y {max(gains):.3f}, y permite distinguir los aportes térmico y eléctrico al error del conductor.

\input{{tablas/benchmark_adaptive_conclusion}}

\section{{RECOMENDACIONES}}

Se recomienda utilizar cada configuración dentro de los escenarios y criterios documentados. Un entorno nuevo debe incorporar su estudio de malla FEM, comprobación de balance, varias semillas y evaluación del residuo eléctrico. Los casos de aceptación parcial requieren revisar la formulación o el entrenamiento antes de emplear su corriente calculada.

Para heterogeneidades complejas se recomienda ampliar la comparación de frecuencias y fracciones de redistribución, nuevas ponderaciones o formulaciones de temperatura y flujo, conservando controles de presupuesto. Las mejoras deben medirse frente al caso de referencia y a las alternativas existentes. Aumentar la anchura o las iteraciones sin ese control no permite identificar el origen de una mejora.

Se recomienda repetir el protocolo de resolución al incorporar una nueva geometría y ampliar también la densidad de frontera. La propagación numérica debe distinguirse de la incertidumbre de propiedades físicas, que exige datos medidos. Ante un cambio de conductor limitante se debe revisar la sensibilidad direccional de la corriente, sin imponer una derivada única al máximo térmico.

La ampliación a instalaciones reales requiere representar las capas internas con mayor detalle, incorporar pérdidas AC y contrastar propiedades y contornos con mediciones independientes. La dependencia térmica de k, la humedad y el régimen transitorio necesitan verificación específica. Los resultados DC reducidos no justifican por sí solos cambios de capacidad en una instalación.

Se recomienda mantener la especificación física separada de las opciones numéricas y conservar todos los manifiestos y pesos. Cada reevaluación debe identificar su código y su referencia FEM. Para estudiar eficiencia se requieren condiciones de hardware y concurrencia controladas, incluyendo el coste de entrenamiento y el número de consultas posteriores.
'''
    (THESIS/'capitulos/04_conclusiones_y_recomendaciones.tex').write_text(spanish(conclusions),encoding='utf-8')
    abstract=fr'''\chapter*{{RESUMEN}}
\addcontentsline{{toc}}{{chapter}}{{RESUMEN}}

La heterogeneidad térmica del suelo modifica la temperatura y la corriente admisible de cables eléctricos enterrados. Esta investigación diseña y evalúa un artefacto basado en redes neuronales informadas por física, organizado mediante ciencia del diseño. Se resuelve conducción estacionaria bidimensional en el suelo, con reconstrucción radial del cable y pérdidas DC dependientes de la temperatura individual del conductor. Una especificación JSON común permite representar conductividad continua mediante fórmulas y estratos con saltos discretos.

La evaluación comprende diecinueve casos, tres mallas FEniCSx por caso y tres semillas PINN en las configuraciones principales. Se comparan arquitectura, pesos, muestreo y enriquecimiento, con estudios de resolución y propagación del error. La selección acepta {accept} de {total} ejecuciones térmicas o electrotérmicas, conservando los fallos. En el escenario XLPE, una zona seca próxima eleva la temperatura FEM de {base:.2f} a {near:.2f} °C y reduce la ampacidad DC de {i_base:.2f} a {i_near:.2f} A. La selección principal PINN de corriente límite cumple todos sus criterios en {ampaccept} de 33 intentos.

El resultado es un procedimiento reproducible de comparación y aceptación, con cuadernos, configuraciones, campos y fuentes identificadas por huella. La conservación global debe complementarse con error de campo y consistencia eléctrica. La evidencia corresponde al modelo reducido y no demuestra validación de campo, ampacidad AC normativa ni una ventaja temporal frente a FEM.

\noindent\textbf{{Palabras clave:}} cable subterráneo; ampacidad; heterogeneidad térmica; red neuronal informada por física; elementos finitos; ciencia del diseño.

\chapter*{{ABSTRACT}}
\addcontentsline{{toc}}{{chapter}}{{ABSTRACT}}

Soil thermal heterogeneity changes the temperature and allowable current of buried power cables. This study designs and evaluates a physics-informed neural network artifact through design science. The model solves two-dimensional steady-state heat conduction in the soil, with radial cable reconstruction and DC losses that depend on each conductor's temperature. A common JSON specification represents continuous conductivity through formulas and layered materials through explicit discontinuities.

The evaluation comprises nineteen cases, three FEniCSx meshes per case, and three PINN seeds for the main configurations. Architecture, loss weights, sampling, and enrichment are compared, including resolution studies and error propagation. The selection accepts {accept} of {total} thermal or electrothermal runs while retaining failed attempts. In the XLPE scenario, a nearby dry region raises the FEM temperature from {base:.2f} to {near:.2f} °C and reduces DC ampacity from {i_base:.2f} to {i_near:.2f} A. The main PINN current-limit selection satisfies all criteria in {ampaccept} of 33 attempts.

The outcome is a reproducible comparison and acceptance procedure supported by notebooks, configurations, fields, and source hashes. Global conservation must be assessed together with field error and electrical consistency. The evidence applies to the reduced model and does not establish field validation, standard-compliant AC ampacity, or a speed advantage over FEM.

\noindent\textbf{{Keywords:}} underground cable; ampacity; thermal heterogeneity; physics-informed neural network; finite elements; design science research.
'''
    parts=abstract.split(r'\chapter*{ABSTRACT}')
    abstract=spanish(parts[0])+r'\chapter*{ABSTRACT}'+parts[1]
    (THESIS/'capitulos/00_resumen_abstract.tex').write_text(abstract,encoding='utf-8')
    print(dict(accepted=accept,total=total,ampacity_accepted=ampaccept))

if __name__=='__main__':main()
