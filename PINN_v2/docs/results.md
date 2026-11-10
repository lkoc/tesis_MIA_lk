# Resultados de la Fase 6

## Alcance de la comparacion principal

Por decision de alcance, `annulus` queda excluido de la comparacion principal y del agregado de
aceptacion de las variantes PINN. La evaluacion de verificacion principal conserva los siete casos
MMS; los experimentos de annulus que siguen se presentan por separado como diagnostico de un
problema particularmente exigente, no como un caso ordinario para ordenar modelos. Se mantienen
sus corridas y referencias FEM para trazabilidad; la campana historica de ocho casos no se borra,
pero sus conteos agregados no deben interpretarse como los resultados principales.

## FEM annulus: corte horizontal y=0

La referencia FEM P2, nivel1/multiplicador1.0, se evaluo directamente en1800 puntos interiores
del anillo sobre `y=0`; los offsets son0.0006m desde `ri` y0.1m desde `ro` para evitar
consultar fuera de la malla curva. FEM frente a la solucion radial analitica: RMSE=`0.000183 K`,
maximo absoluto=`0.000724 K`.

![Annulus FEM y solución analítica en el corte y=0](../results/phase7/p7_annulus_fem_line_20261021/fem/annulus_line_fem_vs_exact.png)

- [Muestras FEM, analitica y error en CSV](../results/phase7/p7_annulus_fem_line_20261021/fem/annulus_line_fem_vs_exact.csv).
- [NPZ FEM con seccion de linea y evaluacion de area](../results/phase7/p7_annulus_fem_line_20261021/fem/annulus_l1.npz).

## CUSP_EXACT annulus: mixed, 20000 Adam + 200 L-BFGS

Se completo el presupuesto solicitado con los criterios estrictos conjuntos activos: error Tmax
`<0.01 K`, NRMSE `<0.1%` y balance global `<0.5%`. Ninguno se cumplio; se agotaron las 200
llamadas externas L-BFGS. La NRMSE de area fue `10.420552%`, error Tmax `12.489603 K`, balance
`57.309197%` y error maximo en el borde interior `14.692150 K`. El residual PDE RMS fue
`3.20e-5`, pero no representa por si solo la precision fisica global.

El primer proceso guardo estado reanudable tras 20000 Adam y 56 llamadas L-BFGS, pero fallo al
reemplazar metadatos del checkpoint en Windows. Se reanudo desde ese estado y se completaron las
144 llamadas restantes. Tiempo medido combinado: `1164.43 s` (`19.41 min`), suma de ambos procesos.

![CUSP_EXACT mixed, FEM y analitica en y=0](../results/phase7/p7_annulus_cusp_20000_200_20261021/cusp_20000_adam_200_lbfgs_y0.png)

Artefactos: [reporte](../results/phase7/p7_annulus_cusp_20000_200_20261021/report.md),
[CSV del corte y=0](../results/phase7/p7_annulus_cusp_20000_200_20261021/cusp_20000_adam_200_lbfgs_y0.csv),
[metricas y tiempos en JSON](../results/phase7/p7_annulus_cusp_20000_200_20261021/summary.json),
[resultado final reanudado](../results/phase7/p7_annulus_cusp_20000_200_20261021/CUSP_EXACT_mixed_resume2/result.json).

## CUSP_EXACT annulus: presupuesto fijo 8000 Adam + 200 L-BFGS

Se compararon dos inicializaciones frescas, mismo seed0/sampling11, FEM P2 nivel1, 16147DoFs,
32x3/tanh, float64 CPU/un hilo y 512 puntos PDE. Solo cambia el sampler (`mixed` de area frente
a50%area+50%log-radio `annulus_log_v1`). Criterios de parada automaticos desactivados para agotar
el presupuesto. La referencia FEM y analitica se evalua en los mismos 6000 puntos de area; la
linea y=0 es un diagnostico adicional.

| Sampler | NRMSE FEM % | NRMSE analitica % | Error Tmax FEM K | Error interior K | Balance global % | RMSE linea FEM K | Tiempo s | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mixed | 11.119273 | 11.119266 | 13.419518 | 15.642424 | 60.513912 | 9.553651 | 281.87 | Falla |
| annulus_log_v1 | 51.828065 | 51.827971 | 27.172156 | 27.482531 | 372.444579 | 23.568531 | 460.77 | Falla |

En este presupuesto y esta inicializacion, `mixed` supera claramente a `annulus_log_v1` en la
solucion ponderada por area y balance, aunque ninguna es aceptable. Mas puntos cerca de `ri`
no garantizan mejor NRMSE de area bajo una loss PDE con media regional; el muestreo log puede
sobrerrepresentar el collar interior. El lifting log-r reduce el error estructural previo, pero
el sampler 50/50 no es una mejora validada para todos los objetivos. No generalizar desde una seed.

![CUSP mixed/log annulus frente a FEM y analitica](../results/phase7/p7_annulus_cusp_8000_200_20261021/cusp_8000_adam_200_lbfgs_y0.png)

Artefactos: [reporte y tabla](../results/phase7/p7_annulus_cusp_8000_200_20261021/report.md),
[Excel comparativo](../results/phase7/p7_annulus_cusp_8000_200_20261021/cusp_8000_adam_200_lbfgs_comparison.xlsx),
[CSV horizontal](../results/phase7/p7_annulus_cusp_8000_200_20261021/cusp_8000_adam_200_lbfgs_y0.csv),
[JSON con metadatos y tiempos](../results/phase7/p7_annulus_cusp_8000_200_20261021/summary.json).

## Primera campaña annulus (sin log-r): FEM, PINN y analitica

El panel inferior muestra el error FEM-analitica (RMSE `0.000183 K`, maximo `0.000724 K`).
Las curvas PINN corresponden a la primera campana annulus de 400 Adam + L-BFGS, sampler `mixed`.
Sus picos en la frontera interior (`493--508 C`) contrastan con el FEM/analitica (`49.54 C`);
esto documenta el fallo del lifting/radio lineal, no la solucion posterior log-r.

![Primera campaña annulus sin log-r: PINN, FEM y solución analítica](../results/phase7/p7_all_verification_exact_20261020/annulus/temperature_x_axis_fem_exact_pinn_no_log.png)

- [CSV con FEM, analitica y las seis PINN](../results/phase7/p7_all_verification_exact_20261020/annulus/temperature_x_axis_fem_exact_pinn_no_log.csv).
- [Resumen de la comparacion](../results/phase7/p7_all_verification_exact_20261020/annulus/temperature_x_axis_fem_exact_pinn_no_log_summary.json).

## Campana historica de siete MMS y annulus, iniciada 2026-10-20

La ejecucion historica incluyo ocho FEM nuevos, P2/nivel1/multiplicador1.0: annulus y los siete MMS
del catalogo de verificacion. MMS: 32 celdas por lado y4225DoFs; annulus: malla curva y16147DoFs.
No se etiqueta ese nivel como malla convergida sin un estudio adicional.

Con la decision de alcance vigente, solo los siete MMS forman la comparacion principal. Los
resultados annulus de esta campana y sus repeticiones posteriores son diagnosticos independientes;
no se incluyen en el agregado principal ni en el ranking de modelos.

La campana usa las seis variantes exactas, la receta400Adam+5LBFGS/32x3/tanh/LR.003/coseno,
float64/CPUunhilo/init0/sampling11/puntosfijos y PDE-only. El conteo es por region: n_interior512
para soil y n_layer256 para otras regiones; no todos los casos tienen1024puntos totales.
La parada y la comparacion principal usan el FEM del mismo caso; el error analitico se informa
por separado. No hay referencia en loss, pero SI seleccion del momento de parada con referencia.

Si no se logra simultaneamente errorTmax<.01K,NRMSE<.1%,balanceglobal<.5% al agotar el
presupuesto base, se agregan hasta5llamadas externas L-BFGS sobre el MISMO modelo/optimizador.
El resultado final se reporta aunque no alcance los objetivos, sin mas ampliaciones automaticas.
Se corrigio la conversion de coordenadas de la traza Robin para que [0,1] respete las esquinas
Dirichlet. Es una version de codigo nueva respecto al ejercicio anterior, no una repeticion binaria.

Salidas incrementales, actualizadas despues de cada corrida:

- [Reporte Markdown](../results/phase7/p7_all_verification_exact_20261020/report.md).
- [Excel comparativo, una hoja por caso](../results/phase7/p7_all_verification_exact_20261020/comparison.xlsx).
- [Parametros efectivos](../results/phase7/p7_all_verification_exact_20261020/campaign_config.json).
- [Resumen JSON y estado por caso/modelo](../results/phase7/p7_all_verification_exact_20261020/summary.json).
- [FEM de los ocho casos historicos](../results/phase7/p7_all_verification_exact_20261020/fem_summary.json).

Las filas pendientes/incompletas NO representan resultados. El reporte vivo contiene los
conteos de corridas concluidas y objetivos alcanzados; no inferir aceptacion de las48filas
por haber generado el libro Excel. Una semilla por familia/caso no demuestra robustez multisemilla.

Cada fila resuelta del Excel y del reporte Markdown informa interiores PDE, interfaz,
frontera, total de colocacion y coordenadas unicas, obtenidos del `collocation.npz` real.
En PDE-only solo los interiores entran a la loss; las otras superficies son diagnostico.
Validacion, evaluacion y auditoria tienen conteos separados: no se suman al total de colocacion
ni se multiplican por epocas. FEM informa DoFs y puntos de evaluacion, no colocacion PINN.
Las filas pendientes mantienen estos conteos vacios; el conteo planificado se etiqueta como tal.

Ejecucion desde PINN_v2 (entorno con torch,deepxde,openpyxl yFEniCSxWSL):

```powershell
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2.experiments.verification_exact_campaign --output results/phase7/p7_all_verification_exact_20261020
```

`--report-only` regenera Excel/Markdown/JSON sin entrenar. El script omite resultados ya
guardados y rechaza sobrescribir una corrida incompleta; no reinicia en silencio checkpoints.

### Diagnostico del anillo y el hueco central (2026-10-21)

**El orificio NO se modela como parte del dominio.** El dominio fisico es el anillo
`ri=0.06 m <= r <= ro=20 m`. El FEM construye la superficie con Gmsh OCC restando el disco
interior al exterior; los puntos FEM guardados no caen dentro de `r<ri`. La inferencia PINN
rechaza coordenadas fuera del anillo. Por tanto, el centro vacio no explica el fallo.

Lo que falla es la solucion PINN en el propio anillo. Los SEIS modelos terminaron con
`accuracy_targets_met=false`, agotaron 5 llamadas L-BFGS adicionales (10 total) y fallaron
las raw gates. Predicen `Tmax=508.48..531.57 C`, error Tmax FEM de `469.02..492.10 K`,
NRMSE FEM `416.64..498.30%` y balance global `1900.82..2643.71%`. En la circunferencia
interior, la temperatura aprendida media es `501.16..518.18 C`, frente a la solucion analitica
`49.5857 C`. El flujo normal Neumann interior impuesto es exactamente `-106.1033 W/m2` y
la temperatura Dirichlet exterior es exactamente `20 C`; la temperatura interior NO es un dato
Dirichlet, por lo que debe inferirse del problema y no esta fijada por construccion.

La nube PDE guardada contiene 512 puntos uniformes por area; su radio minimo fue `1.56391 m`,
ninguno de los 512 cae en `ri <= r < 0.5 m`. La malla FEM conserva el anillo, pero la nube
comun de 6000 puntos de evaluacion tambien tiene minimo radio `0.43791 m`; no inspecciona
directamente la frontera `r=0.06 m`. Por eso la auditoria explicita de trazas en la circunferencia
interior fue necesaria y se informa aparte de NRMSE/Tmax en la nube de evaluacion.

Prueba de capacidad del levantamiento: se sustituyeron SOLO para el diagnostico las temperaturas
y pendientes radiales analiticas exactas en ambas fronteras del Hermite actual, que interpola
linealmente en `r`. La solucion verdadera es logaritmica en radio. Aun con esos datos exactos,
el Hermite lineal-r tiene error maximo `234.1012 K`, RMSE radial `153.5635 K` y en radio medio
predice `-176.1424 C` frente a `23.5149 C` analiticos. Esto demuestra una incompatibilidad
severa entre la representacion radial actual y este anillo delgado en coordenada lineal; no es
un problema de que el FEM rellene el hueco. Las seis variantes comparten este levantamiento y
ninguna cabeza correctora supero su condicionamiento/cobertura con PDE-only.

### Repeticion annulus con ajuste exclusivo `annulus_log_v1` (2026-10-21)

Se repitieron las seis variantes en una campaña separada, conservando la FEM P2 nivel1,
la fisica/hash, 400 Adam +5 llamadas base L-BFGS +hasta5 adicionales, 32x3/tanh, float64,
init0/sampling11, CPU/un hilo y total512 puntos PDE. SOLO annulus cambia respecto a la
formulacion anterior: lifting Hermite en `s=ln(r/ri)/ln(ro/ri)`, bubble en s y sampler mixto
50% uniforme en area +50% uniforme en log-radio. Se evalua Tmax/error tambien en 256 puntos
de cada circunferencia, con peso cero de area; NRMSE continua usando solo la cuadratura de area.
La nueva nube tiene 130/512 interiores con r<1m y radio minimo 0.060757m; antes eran 0/512
y r_min=1.563911m. Ningun punto cae dentro de r_i=0.06m.

| Modelo | Error max borde interior viejo K | Error max borde interior nuevo K | NRMSE FEM vieja % | NRMSE FEM nueva % | Reduccion NRMSE % | Balance global nuevo % | Raw gates nuevos |
|---|---:|---:|---:|---:|---:|---:|---|
| M0_EXACT | 490.204 | 20.091 | 480.851 | 33.541 | 93.0 | 233.601 | Falla |
| CUSP_EXACT | 514.366 | 8.831 | 498.297 | 14.722 | 97.0 | 106.902 | Falla |
| DCSNN_EXACT | 492.239 | 27.561 | 481.743 | 43.254 | 91.0 | 304.611 | Falla |
| FBPINN_EXACT | 494.750 | 12.100 | 416.635 | 19.188 | 95.4 | 136.879 | Falla |
| DeepXDE_PFNN_EXACT | 495.865 | 15.244 | 479.178 | 26.296 | 94.5 | 189.504 | Falla |
| DeepXDE_FNN_EXACT | 495.865 | 15.244 | 479.178 | 26.296 | 94.5 | 189.504 | Falla |

El ajuste reduce mucho el error frente a la version lineal-r, pero TODAS las seis siguen
rechazadas y agotaron las 10 llamadas externas L-BFGS. En la traza interior la temperatura
queda aun 8.8--27.6 K lejos; en ninguna se logra error Tmax<0.01 K, NRMSE<0.1% y balance<0.5%
simultaneamente. El error de borde interior se recalculo post hoc sobre los mismos 256 angulos
para ambos checkpoints viejos/nuevos; los checkpoints antiguos no se modificaron.
El Tmax nuevo incluye el borde interior; el anterior solo maximizaba sobre la nube de area y no
es comparable. La NRMSE vieja/nueva usa los mismos 6000 puntos y pesos de area. Los errores de
borde de checkpoints viejos se recalcularon directamente; no se guardaron en aquel reporte.

Esto apoya que el lifting lineal-r era un cuello de botella, pero muestra que cambiar solo
coordenada y cobertura no basta. SIGUIENTE: comparar trazas Neumann/Dirichlet y derivadas
en r=ri/ro; separar inicialmente una calibracion radial pura del corrector angular, y comprobar
el gradiente de PDE respecto a la temperatura constante interior. Si esa traza apenas recibe
gradiente, probar un parametro escalar radial trazable o la solucion particular logaritmica
determinada por potencia/k dentro del ansatz, nunca como penalizacion supervisada. No añadir
mas pasos antes de resolver esta identificabilidad/condicionamiento. La nube nueva mantiene 512
puntos (768 contando 256+256 puntos de frontera diagnósticos); el FEM no cambia. Un punto exacto
en Neumann con T desconocida no implica que el aprendizaje PDE haya determinado T.

**Regla de alcance:** `annulus_log_v1` se habilita solo en `AnnulusProblem` (un anillo
concéntrico, material homogeneo, k constante, fuente nula y condiciones radiales compatibles).
Un agujero por si solo no basta: un hueco excentrico o varios huecos no tienen un centro radial
global. Interfaces circulares concentricas de varios materiales requieren levantamientos por
anillo y continuidad de T/flujo; k variable o fuente radial necesita una solucion particular
integrada por material. No se activa automaticamente log-radio para esos problemas. El trainer
rechaza esta politica fuera del annulus concentrico admitido.

Artefactos: [comparacion vieja/nueva](../results/phase7/p7_annulus_log_v1_20261021/annulus/comparison_log_v1_20261021.json),
[reporte y Excel antes/despues](../results/phase7/p7_annulus_log_v1_20261021/report.md),
[nueva corrida M0](../results/phase7/p7_annulus_log_v1_20261021/annulus/M0_EXACT/result.json),
[config annulus](../results/phase7/p7_annulus_log_v1_20261021/configs/annulus.json).
La campaña original y los seis checkpoints previos quedan inmutables.

### Extension 1000 Adam + 50+50 L-BFGS desde pesos previos

Completada para las seis variantes el 2026-10-21. Se cargaron los pesos de inferencia de cada
checkpoint log-v1; Adam comenzo un horizonte nuevo de 1000 pasos y L-BFGS comenzo historial nuevo.
No se reutilizaron momentos del optimizador. Cada modelo hizo 50 llamadas base y 50 adicionales;
ninguno logro simultaneamente error Tmax<0.01 K, NRMSE<0.1% y balance global<0.5%.

| Modelo | NRMSE inicial FEM % | NRMSE final FEM % | Tmax error inicial K | Tmax error final K | Error final borde interior K | Balance global final % | L-BFGS base+extra | Targets AND | Tiempo s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| M0_EXACT | 33.5414 | 36.3830 | 20.0566 | 21.2111 | 21.3445 | 263.4257 | 50+50 | No | 75.91 |
| CUSP_EXACT | 14.7215 | 12.8986 | 8.7595 | 7.7712 | 7.9354 | 94.7435 | 50+50 | No | 67.30 |
| DCSNN_EXACT | 43.2537 | 42.7178 | 27.2222 | 24.5966 | 24.8038 | 307.5191 | 50+50 | No | 72.19 |
| FBPINN_EXACT | 19.1876 | 19.6493 | 11.8967 | 12.4268 | 12.4765 | 140.4403 | 50+50 | No | 570.60 |
| DeepXDE_PFNN_EXACT | 26.2957 | 29.8404 | 15.1631 | 17.2876 | 17.4132 | 218.2496 | 50+50 | No | 114.90 |
| DeepXDE_FNN_EXACT | 26.2957 | 29.8404 | 15.1631 | 17.2876 | 17.4132 | 218.2496 | 50+50 | No | 115.20 |

La extension empeoro NRMSE en M0, FBPINN, PFNN y FNN; CUSP/DCSNN mejoraron solo modestamente.
Las trazas internas siguen lejos aun despues de 160 llamadas externas totales. Esto indica que
mas optimizacion del mismo campo/loss no es la palanca adecuada; medir sensibilidad/identificabilidad
del modo de temperatura interior antes de cambiar presupuesto.

El Excel tiene la pestana `Annulus_extend`; reporte Markdown y resumen estan en
[annulus_extension_1000_50](../results/phase7/p7_annulus_log_v1_20261021/annulus_extension_1000_50/annulus_extension_summary_1000_50.json).
La carpeta `FBPINN_EXACT` parcial se preservo y no se sobrescribio: el resultado final de FBPINN
proviene de `FBPINN_EXACT_retry1`. Los tiempos corresponden a las corridas completas terminadas.

### Corte horizontal de temperatura

Se grafica `y=0`: rama izquierda `-ro<=x<=-ri`, hueco central `-ri<x<ri` y rama derecha
`ri<=x<=ro`. La referencia es la solucion radial analitica; las seis curvas son los pesos
finales de la extension 1000+50+50. Datos del mismo muestreo en el [CSV](../results/phase7/p7_annulus_log_v1_20261021/annulus_extension_1000_50/temperature_x_axis.csv).

![Corte de temperatura annulus en y=0](../results/phase7/p7_annulus_log_v1_20261021/annulus_extension_1000_50/temperature_x_axis.png)

### Curva tras 400 pasos Adam, sin L-BFGS

Esta figura es una corrida diagnostica independiente: seis modelos, 400 Adam, **0 L-BFGS**,
seed0/sampling11 y `annulus_log_v1`. Es una inicializacion reproducible nueva con seed0; NO es
un warm start desde los pesos log-v1. No usa el checkpoint de la extension posterior ni mezcla
sus 100 llamadas L-BFGS; el estado graficado es el checkpoint al terminar los 400 Adam.

![Annulus tras 400 pasos Adam y sin L-BFGS](../results/phase7/p7_annulus_adam400_only_20261021/temperature_x_axis.png)

Datos y metricas de estas corridas: [CSV del corte](../results/phase7/p7_annulus_adam400_only_20261021/temperature_x_axis.csv),
[resumen con los seis resultados](../results/phase7/p7_annulus_adam400_only_20261021/summary.json).

Como comparacion separada, el [corte de los checkpoints finales log-v1 tras 400 Adam +5 llamadas
L-BFGS base +5 extra](../results/phase7/p7_annulus_log_v1_20261021/annulus/temperature_x_axis_adam400_lbfgs10.png)
usa los pesos guardados al concluir la parada/extension; sus [datos CSV](../results/phase7/p7_annulus_log_v1_20261021/annulus/temperature_x_axis_adam400_lbfgs10.csv)
y [metricas por curva](../results/phase7/p7_annulus_log_v1_20261021/annulus/temperature_x_axis_adam400_lbfgs10_summary.json)
estan junto a los checkpoints log-v1. No confundirlo con la grafica Adam-only de arriba ni con
la extension de 1000 Adam mostrada en `Annulus_extend`.

### Evaluacion en el limite interior `ri + epsilon`

Para responder si excluir el contorno puntual cambia la conclusion, se evaluaron los checkpoints
extendidos en256 angulos para epsilons `1e-8`, `1e-6`, `1e-4`, `1e-3` y `1e-2` m. La tabla da
el error maximo de anillo para `1e-8` y `1e-2`; el NRMSE sigue siendo el de los 6000 puntos con
peso de area, sin sumar estas circunferencias diagnosticas.

| Modelo | Error max `ri+1e-8m` K | Error max `ri+1e-2m` K | Cambio del error K | NRMSE de area FEM % | Tmax area+anillo `ri+1e-8m` C | Max error combinado K |
|---|---:|---:|---:|---:|---:|---:|
| M0_EXACT | 21.344498 | 21.345348 | +0.000850 | 36.382942 | 28.374583 | 21.344498 |
| CUSP_EXACT | 7.935350 | 7.935404 | +0.000054 | 12.898555 | 41.814519 | 7.935350 |
| DCSNN_EXACT | 24.803845 | 24.805207 | +0.001363 | 42.717689 | 24.989112 | 24.803845 |
| FBPINN_EXACT | 12.476549 | 12.477933 | +0.001384 | 19.649236 | 37.158953 | 12.476549 |
| DeepXDE_PFNN_EXACT | 17.413176 | 17.413983 | +0.000806 | 29.840393 | 32.298088 | 17.413176 |
| DeepXDE_FNN_EXACT | 17.413176 | 17.413983 | +0.000806 | 29.840393 | 32.298088 | 17.413176 |

El Tmax analitico en `ri+1e-8m` es49.585721C (temperatura de borde 49.585722C). Para
epsilon->0+ el error converge a los errores de traza anteriores; al quitar el punto exacto del
contorno el maximo cambia insignificativamente y ninguna variante pasa los criterios. La razon
matematica es continuidad del campo log-radial: `T(ri+epsilon) -> T(ri)`. La hoja
`Annulus_epsilon` y el JSON detallado estan en
[comparison.xlsx](../results/phase7/p7_annulus_log_v1_20261021/comparison.xlsx) y
[epsilon_evaluation_20261021.json](../results/phase7/p7_annulus_log_v1_20261021/annulus_extension_1000_50/epsilon_evaluation_20261021.json).

## Transferencia implementada a PINN anteriores, 2026-10-20

Las versiones historicas fallidas se conservaron como controles. Se implementaron variantes
nuevas que combinan sus correctores con normalizacion [0,1], trazas/fronteras exactas y loss
PDE-only. No se afirma que cambiar solo el nombre o quitar penalizaciones arregle el modelo.

Se completaron las SEIS corridas en `mms_interface`, con fuente no nula, mismo hash y nube
del control M1_EXACT: 1024 puntos PDE fijos, 32x3/tanh, init=0/sampling=11, float64 CPU/un hilo,
400 Adam LR0.003/coseno y 5 L-BFGS externos max_iter20; sin atencion ni etiquetas de referencia.
La [ficha de reproducibilidad](reproducibility.md) incluye comandos, rutas, conteos y semantica.

### Cuadro consolidado de las seis corridas completas

Fecha de cierre: 2026-10-20. Se incluyen las dos corridas iniciales DCSNN/PFNN, sin repetirlas
ni alterar sus resultados, y las cuatro corridas posteriores M0/CUSP/FBPINN/FNN.

| Caso | Variante nueva | Tmax C | NRMSE analitica % | Error maximo K | Balance global % | Peor balance regional % | Parametros | Tiempo s | Tiempo min | Raw gates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mms_interface | DCSNN_EXACT | 32.483861 | 0.098462 | 0.028676 | 0.432101 | 0.405215 | 11176 | 477.94 | 7.97 | Todas verdaderas |
| mms_interface | DeepXDE_PFNN_EXACT | 32.483934 | 0.064856 | 0.023576 | 0.395652 | 0.392378 | 13482 | 862.83 | 14.38 | Todas verdaderas |
| mms_interface | M0_EXACT | 32.484019 | 0.042050 | 0.016379 | 0.168419 | 0.183392 | 11112 | 1054.98 | 17.58 | Todas verdaderas |
| mms_interface | CUSP_EXACT | 32.483894 | 0.061693 | 0.018993 | 0.384698 | 0.392487 | 11144 | 767.43 | 12.79 | Todas verdaderas |
| mms_interface | FBPINN_EXACT | 32.483914 | 0.082636 | 0.030607 | 0.376787 | 0.377878 | 17835 | 1367.28 | 22.79 | Todas verdaderas |
| mms_interface | DeepXDE_FNN_EXACT | 32.483906 | 0.092434 | 0.032818 | 0.494458 | 0.483721 | 11210 | 1045.17 | 17.42 | Todas verdaderas |

Referencia de esta tabla: solucion analitica, NO FEM. Los saltos de T son 0 K y los saltos de
flujo RMS estan entre 2.7630e-14% y 2.9240e-14%, normalizados por la escala caracteristica de
flujo, no por el FEM. La Tmax analitica sobre estos 6000 puntos es 32.483638668 C; el maximo
continuo es 32.5 C en una frontera impuesta y no sustituye la auditoria interior.

Los tiempos son el `elapsed_s` guardado por cada solver, y los minutos son ese valor/60.
Incluyen el entrenamiento y la evaluacion/auditoria hasta el punto de registro del tiempo;
no son tiempos completos de CLI con imports iniciales y archivado de fuentes. Son observaciones
individuales en momentos distintos, no un benchmark controlado de aceleracion: las arquitecturas,
parametros y reevaluaciones L-BFGS difieren. No atribuir las diferencias solo a normalizacion.

DCSNN conserva una red compartida e indicador regional; PFNN/FNN conservan redes reales de
DeepXDE, no el corrector M1 del wrapper previo. Las seis usan el mismo levantamiento exacto.

Artefactos: [resumen consolidado de seis](../results/phase7/p7_transfer_exact_20261020/comparison_all_six_20261020.json),
[resumen inicial de dos, conservado](../results/phase7/p7_transfer_exact_20261020/comparison_summary_20261020.json),
[DCSNN](../results/phase7/p7_transfer_exact_20261020/DCSNN_EXACT_s0/result.json),
[PFNN](../results/phase7/p7_transfer_exact_20261020/DeepXDE_PFNN_EXACT_s0/result.json),
[M0](../results/phase7/p7_transfer_exact_20261020/M0_EXACT_s0/result.json),
[CUSP](../results/phase7/p7_transfer_exact_20261020/CUSP_EXACT_s0/result.json),
[FBPINN completo](../results/phase7/p7_transfer_exact_20261020/FBPINN_EXACT_s0_complete/result.json),
[FNN](../results/phase7/p7_transfer_exact_20261020/DeepXDE_FNN_EXACT_s0/result.json).
Cada corrida conserva configuracion, fuentes, manifest, puntos, evaluacion y par portable.
Los SEIS pares aislados reprodujeron las 6000 temperaturas exactamente, con flujo finito;
las cuatro variantes nativas guardan estado de continuacion, las dos DeepXDE solo inferencia/warm start.
Se verificaron igualdad de hash fisico, hiperparametros comunes, semillas y cada arreglo de puntos.

El primer intento `FBPINN_EXACT_s0` quedo interrumpido con un checkpoint inicial y se conserva
sin resultado final. No entra en la tabla ni se suma su duracion desconocida: la fila FBPINN
proviene de una corrida nueva completa en `FBPINN_EXACT_s0_complete`, con la misma semilla.
Se observo OMP_NUM_THREADS=1 y MKL_NUM_THREADS no definido en ese proceso; un interprete nuevo
con ese entorno confirma un hilo de computo PyTorch. No se declara que ambas variables fueran 1.

Estas son corridas completas, no los smoke tests de 1 paso. La regresion de la implementacion
ya realizada dio 379 aprobadas, 1 omitida, 10 excluidas (slow/FEM y el hash historico
no relacionado), sin fallos; este cierre no cambia algoritmos ni vuelve a ejecutar esa suite.
Front-end en http://127.0.0.1:8877/: seis opciones nuevas y ocho
pesos no aplicables deshabilitados en modo exacto, comprobado sin errores de pagina.

Limite cientifico: una inicializacion por familia y una nube; no multisemilla, cables ni
demostracion de que todos los modelos antiguos se hayan recuperado. La transferencia SI produjo
seis variantes aceptadas en este MMS, en lugar de quedar solo como propuesta. M2 sigue requiriendo
su ecuacion constitutiva mientras T y q sean independientes.

## Cierre DeepXDE_EXACT PDE-only (2026-10-20)

Las dos corridas de `p7_deepxde_exact_mms_20261019` terminaron; no se reiniciaron al retomar
el trabajo. Caso `mms_interface`, misma fisica, puntos y semillas 0/11, redes 32x3/tanh,
encoder `[0,1]`, 400 Adam con decaimiento coseno + 5 pasos externos L-BFGS de hasta
20 iteraciones internas. Backend PyTorch, CPU y un hilo. Referencia analitica solo para evaluar.

La [ficha experimental fechada el 2026-10-20](reproducibility.md) contiene hiperparametros,
conteos reales de puntos, estrategia espacial, semillas, entorno, comandos y rutas para replicar
estos resultados. La [matriz de transferencia](models.md) separa lo ya comprobado de los
redisenios propuestos para los modelos anteriores; no declara convergencia nueva de cables.

### Comparacion FEM-PINN para reporte de tesis

Caso analizado: `mms_interface`, manufacturado 2D con fuente no nula, conductividades 0.5 y 2,
interfaz vertical y continuidad de temperatura/flujo. Se incorpora el campo FEM guardado de
FEniCSx 0.10.0, elementos P2, nivel 1, malla de 32 celdas por lado y 4225 grados de libertad.
Es el nivel mas fino disponible de esta ejecucion, no una certificacion nueva de convergencia.
Se verificaron el mismo hash fisico y las mismas 6000 coordenadas para los cuatro metodos.

En esta tabla la NRMSE y el error maximo de las PINN se recalcularon frente al FEM nivel 1,
sin reentrenar ni modificar los resultados originales. No se calculo FEM contra si mismo:
sus celdas de NRMSE y error maximo quedan vacias, no en cero. El desbalance es un diagnostico
fisico independiente, calculado de fuente y flujo saliente, y por eso si se informa para FEM.

| Caso | Solucion / motor y dtype | Tmax evaluada C | NRMSE vs FEM % | Error maximo vs FEM K | Desbalance global % | Tiempo registrado s | Resultado |
|---|---|---:|---:|---:|---:|---:|---|
| mms_interface | FEM FEniCSx P2, nivel 1 | 32.483638 | | | 0.079783 | 1.09 | Referencia FEM |
| mms_interface | M1_EXACT float64 | 32.483928 | 0.064126 | 0.023670 | 0.355327 | 211.91 | Aceptado |
| mms_interface | DeepXDE_EXACT float64 | 32.483928 | 0.064126 | 0.023670 | 0.355326 | 683.72 | Aceptado |
| mms_interface | DeepXDE_EXACT float32 | 32.483807 | 0.100492 | 0.042483 | 0.146675 | 428.51 | Aceptado |

La Tmax se evalua en la nube comun, no en los nodos FEM ni mediante una busqueda del maximo
continuo. La NRMSE se normaliza por `max(T_FEM) - 20 C`; el error maximo es puntual sobre esa
nube. Las aceptaciones PINN usan los errores frente al FEM y las auditorias fisicas originales,
incluidos todos los balances regionales. El maximo continuo analitico del MMS es 32.5 C en una
frontera Dirichlet impuesta; Tmax por si sola no certifica la solucion interior.

Los tiempos no constituyen un benchmark controlado de aceleracion: el valor FEM es el tiempo
registrado de ese nivel antes del archivado de fuentes, sin el costo total del worker/WSL;
los valores PINN cubren entrenamiento y evaluacion. FEM usa CPU en WSL2 y las PINN CPU en Windows.

Fuentes del resultado FEM:
- [Metadatos FEM nivel 1](../front-end/runs/20261019_200551_fem_mms_interface/fem/mms_interface_l1.json).
- [Campo FEM sobre la nube comun](../front-end/runs/20261019_200551_fem_mms_interface/fem/mms_interface_l1.npz).
- [Comparacion recalculada y trazabilidad](../results/phase7/p7_deepxde_exact_mms_20261019/comparison_vs_fem_20261020.json).

### Auditoria original frente a la solucion analitica

Los valores siguientes se conservan para no confundir las dos referencias. No son errores
frente al FEM y permanecen iguales a los `result.json` originales.

| Motor y dtype | Tmax evaluada C | NRMSE % | Error maximo K | Desbalance global % | Tiempo s | Resultado |
|---|---:|---:|---:|---:|---:|---|
| M1_EXACT float64, control previo | 32.483928 | 0.064122 | 0.023667 | 0.355327 | 211.91 | Aceptado |
| DeepXDE_EXACT float64 | 32.483928 | 0.064121 | 0.023667 | 0.355326 | 683.72 | Aceptado |
| DeepXDE_EXACT float32 | 32.483807 | 0.100486 | 0.042463 | 0.146675 | 428.51 | Aceptado |

Ambos DeepXDE pasan todas las compuertas crudas: finitud, temperaturas, regiones, balance
global/regional y trazas. `constraint_count=0`: no hay condiciones de frontera/interfaz en la
lista de penalizaciones. La loss contiene solo PDE; no hay etiquetas FEM ni de solucion exacta
interior. Los limites analiticos del caso siguen impuestos por construccion.

DeepXDE_EXACT envuelve el mismo campo regional y levantamiento de M1_EXACT; no se presenta
como una PFNN/FNN independiente. La coincidencia float64 verifica la transferencia del
entrenador, no la superioridad de una arquitectura. Float32 ahorro 37.33% del tiempo frente
a DeepXDE float64 en esta comparacion unica, pero no fue mas preciso en temperatura.
El menor desbalance global no basta para concluir superioridad: el residuo PDE RMS independiente
de `mms_0` fue 0.190773 en float32 frente a 0.017806 en float64; en `mms_1`, 0.068555 frente
a 0.061098. El salto RMS de flujo fue 1.50e-5% frente a 2.73e-14%.

No se extrapolan tiempos a XPU/GPU, a otras semillas ni a cables. M1_EXACT nativo sigue siendo
el control mas rapido observado en CPU; no se modificaron los defaults ni las compuertas.

### Validacion de integracion completada

- 69 pruebas pasaron: DeepXDE historico PFNN/FNN, modo exacto, equivalencia de residuo y
  gradientes con el objetivo nativo, dtypes, L-BFGS, CLI, frontend y checkpoints.
- Suite global no lenta/no FEM: 337 pruebas correctas, 1 omitida y 10 deseleccionadas,
  sin errores ni fallos en las seleccionadas. Se excluyo expresamente la prueba de hash de
  procedencia historica de `Benchmarks/cases.py`, cuya discrepancia ya existia y no se modifico.
  El reporte legible por herramientas es `regression_suite_20261020.xml`.
- Los dos modelos finales se reconstruyeron en carpetas con solo `model.pt` y `model.json`.
  En ambos, la diferencia maxima al reproducir las 6000 temperaturas guardadas fue 0 K;
  la inferencia de flujo resulto finita.
- El par exacto de DeepXDE permite inferencia e inicio desde pesos, no reanudacion completa
  del optimizador; los bundles declaran `resume_available=false`. No se fabrica estado ausente.
- Resultados originales en `results/phase7/p7_deepxde_exact_mms_20261019/{control,float32}/`;
  `screen.json` conserva las auditorias y `verification_20261020.json` las comprobaciones finales.

## MMS heterogeneo con fuente: recomendaciones aplicadas (2026-10-19)

Caso `mms_interface`, conductividades 0.5 y 2, fuente no nula e interfaces con flujo no nulo.
Todos los brazos usan `M1_EXACT` 32x3/tanh, float64 y objetivo solo PDE, sin etiquetas exactas
interiores ni FEM en el entrenamiento. La referencia exacta sobre la nube de evaluacion tiene
Tmax 32.483639 C. El maximo continuo es 32.5 C, en la frontera Dirichlet; la Tmax cercana por si
sola no prueba precision interior ni conservacion.

| Modelo PINN | Entrenamiento, semilla 0 | Tmax evaluada C | NRMSE % | Desbalance global % | Resultado |
|---|---|---:|---:|---:|---|
| M1_EXACT | 400 Adam | 32.481232 | 2.059654 | 7.059050 | Rechazado |
| M1_EXACT + RBA | 400 Adam | 32.483180 | 0.508174 | 2.044604 | Rechazado |
| M1_EXACT + SA | 400 Adam | 32.482634 | 0.944409 | 4.001939 | Rechazado |
| M1_EXACT | 400 Adam, dobles filas | 32.477664 | 4.258184 | 14.963386 | Rechazado |
| M1_EXACT | 400 Adam + 5 L-BFGS | 32.483928 | 0.064122 | 0.355327 | Aceptado |

RBA/SA son pesos adaptativos de la misma arquitectura, no modelos con condiciones distintas.
Los 5 pasos L-BFGS son externos, con hasta 20 iteraciones internas cada uno y busqueda de linea;
no equivalen a 5 actualizaciones Adam. No se afirma superioridad a igual costo computacional.

### Repeticion de la receta recomendada

Se ejecutaron tres corridas nuevas desde cero, manteniendo 400 Adam + 5 L-BFGS, sin atencion,
LR 0.003, 512 interiores por region, puntos fijos, `sampling=mixed` y semilla de colocacion 11.
Solo cambia la inicializacion: 0, 11 o 23. Se comprobo igualdad exacta de las nubes interiores,
misma configuracion y hash fisico. La semilla 0 reprodujo exactamente la loss original.

| Modelo PINN | Semilla inicial | Tmax evaluada C | NRMSE % | Desbalance global % | Resultado |
|---|---:|---:|---:|---:|---|
| M1_EXACT | 0 | 32.483928 | 0.064122 | 0.355327 | Aceptado |
| M1_EXACT | 11 | 32.483989 | 0.080310 | 0.568118 | Aceptado |
| M1_EXACT | 23 | 32.483830 | 0.070552 | 0.271460 | Aceptado |

Las tres pasan todas las compuertas crudas, incluidos los balances de cada region. Media de
NRMSE: 0.071661% (desviacion muestral 0.008151 puntos porcentuales); media de balance global:
0.398302%. Peor balance global: 0.568118%; peor balance regional: 0.564157%; peor error puntual
de temperatura entre las tres: 0.030348 K. No se selecciono solo la mejor semilla.

Con cuadratura de caras/fronteras aumentada de 512 a 2048, los balances globales son
0.355323%, 0.568112% y 0.271457%: todas siguen aceptadas. RBA, evaluado tambien con 2048,
permanece en 2.044598%; su rechazo no se elimino aumentando la cuadratura de auditoria.

Se aplica la recomendacion de refinar la optimizacion y repetir semillas sin aumentar puntos ni
subir a 20000 Adam. La buena precision independiente no justifica ahora duplicar puntos o agregar
atencion. Esto demuestra consistencia frente a tres inicializaciones con una nube fija, no robustez
ante nuevas nubes, todas las geometrías ni los cables. Fronteras/interfaces siguen impuestas por
construccion; no se agrego ningun termino de frontera, flujo o energia a la loss.

Artefactos: `results/phase7/p7_mms_single_validation_20261019/mms_recommendations_3seeds_20261019/`.
Cada `seed_0`, `seed_11` y `seed_23` contiene resultados, checkpoint portable y `dense_audit.json`.
`comparison_summary.json` conserva las dos tablas, estadisticas y verificaciones de configuracion.

## Auditoria M1_EXACT y screening controlado (2026-10-19)

Se implemento persistencia nativa mediante solo `model.pt` + `model.json`, con problema fisico
embebido, encoder/dtype/parametrizacion versionados y estado completo de continuacion.
Las pruebas verifican recarga en proceso aislado, inferencia T/flujo, igualdad CPU entre corrida
continua e interrumpida (sin atencion, RBA, SA y remuestreo), historial L-BFGS y rechazo de pares
mezclados. Esto demuestra contratos de implementacion, no precision termica del cable.

La auditoria del checkpoint anterior encontro gradientes de trazas de orden 4.55e5 frente a
46.1 en las redes, y sensibilidad nula de las trazas desde el suelo en la nube uniforme examinada.
No prueba imposibilidad matematica: indica mal condicionamiento y cobertura insuficiente.
Con semilla 11, el sampler anterior daba conteos de collar `[4,104,3]` (mixed) y `[257,0,0]`
(interface). La seleccion por lote de un solo cable se corrigio a candidatos equilibrados por punto,
manteniendo la fraccion uniforme y registrando `sampling_version=balanced_cables_v2`.

### Comparacion de 400 pasos en CPU

Misma fisica `f08358db...`, referencia FEM nivel 2 solo para evaluar, semillas 0/11, 32x3/tanh,
LR 0.003, puntos 512/256/128/128, encoder `[0,1]`, 400 Adam, sin L-BFGS, un hilo CPU.
La referencia tiene Tmax 50.05758 C. No comparar estos tiempos con las corridas XPU historicas
como si midieran una aceleracion causal.

| Brazo | Tmax C | NRMSE % | Desbalance global % | Tiempo s | Veredicto |
|---|---:|---:|---:|---:|---|
| Control float64, trazas legacy | 20.19270 | 14.30440 | 106.46752 | 442.69 | rechazado |
| Trazas condicionadas | 23.07957 | 14.10958 | 102.33564 | 436.99 | rechazado |
| Flujo medio conservativo | 25.91610 | 35.34181 | 488.92796 | 452.17 | rechazado |
| Control float32 | 20.19274 | 14.30429 | 106.45720 | 423.98 | rechazado |

La mejora parcial de las trazas condicionadas no supera la compuerta para una corrida larga;
la opcion conservativa empeora la solucion global pese a conservar las medias de flujo internas.
Float32 reproduce practicamente el control, pero la diferencia temporal de una corrida no basta
para promoverlo: faltan repeticiones temporales y un modelo fisicamente aceptado. Se conserva
float64 como predeterminado. El RMS de salto de flujo de float32 alcanza aproximadamente 0.42%,
frente a valores cercanos al redondeo en float64: coincidir en decimas de grado no significa
conservar la misma precision en flujos derivados. RBA/SA y ganancia del suelo se probaron en screening de 40 pasos;
tambien rechazados, sin evidencia para promoverlos. No se ejecutaron 20k pasos ni tres semillas
porque ningun candidato supero la compuerta fisica de screening.

Artefactos reproducibles en `results/phase7/p7_exact_recovery_20261019/`:
`audit_normalized_legacy.json`, `cpu_screen40/screen.json`,
`cpu_screen400_core/screen.json`, `cpu_screen400_followup/screen.json`.
Se preservaron los checkpoints anteriores sin reinterpretar su normalizacion.

Limitacion de validacion del repositorio: la prueba de procedencia historica de `Benchmarks/cases.py`
detecta un SHA256 distinto del manifiesto legado; ese archivo no se modifico en esta implementacion
y no se actualizo el hash para ocultar la discrepancia.

Esta página resume lo que ya se midió. Los criterios de aceptación y las compuertas son **propuestos**,
no confirmados. La referencia FEM nunca se usó para entrenar.

## Referencia FEM (estudio p6_fem3)

La malla se considera convergente si el cambio entre los dos últimos niveles es 0.5 % del
calentamiento o menos. El estudio de dominio duplica el semiancho y la profundidad.

| caso | cambio de malla, % del calentamiento | cambio por dominio, % | sensible al dominio |
|---|---|---|---|
| xlpe_single | 0.056 | 0.31 | no |
| xlpe_backfill | 0.035 | 0.49 | no |
| xlpe_discrete_layers | 0.055 | 0.01 | no |
| aras_flat | 0.001 | 3.48 | sí |
| kim_v5_summer | 0.001 | 11.87 | sí |

Para `kim_v5_summer` se usaron los niveles 0 a 2 y la Tmax es 53.24 °C en el dominio ancho,
+4.04 K respecto al dominio base. El nivel 3 agotó la memoria de WSL; los estudios fallidos `p6_fem` y
`p6_fem2` se conservan.

## PINN M1 contra FEM (6000 Adam y 100 L-BFGS)

| caso | NRMSE, % | veredicto |
|---|---|---|
| xlpe_single | 0.558 | aceptado |
| xlpe_backfill | 2.14 | rechazado |
| xlpe_discrete_layers | 0.53 | rechazado |
| aras_flat | 0.981 | rechazado (peor región 4.83 %) |
| kim_v5_summer | 18.7 | rechazado |

## Entrenamientos largos de M1 (20000 Adam y 200 L-BFGS)

| caso | NRMSE, % | veredicto |
|---|---|---|
| xlpe_backfill | 2.43 | rechazado (fallan el balance y el salto de T) |
| xlpe_discrete_layers | 0.407 | rechazado (falla el salto de T en la interfaz) |
| aras_flat | 0.502 | aceptado (balance 0.096 %) |

Alargar el entrenamiento no mejoró `xlpe_backfill`.

### Kim, verano (M1, semilla 0)

El M1 no resuelve kim_v5_summer: NRMSE 13.4 %, error del ascenso de T 54.2 %, error de Tmax 18.4 K, balance 167 %, salto de T en interfaz 0.14 K, pérdida final 0.559 (12000 Adam, sin L-BFGS, 18892 s). Fallan cinco compuertas (Tmax, balance, salto de T, NRMSE, región); solo pasa el salto de flujo. La causa no está diagnosticada: no se atribuye a ninguna decisión de diseño hasta revisar el historial de pérdida y el balance por región.

## Cómo leerlo

- Un rechazo dice qué compuerta falló; el detalle está en el `report.md` de cada estudio.
- Los casos sensibles al dominio no tienen una referencia única: el error del PINN se mide contra
  el FEM del mismo dominio reducido.

### Kim, verano (M1, semilla 0): diagnostico

Hechos verificados sobre `results/phase6/p6_pinn_kim_long/kim_v5_summer/M1_s0`:

- La perdida total sigue bajando al final (6305, 2.57, 1.54, 0.861, 0.624, 0.559 en los pasos 0, 1200, 3000, 6000, 9000, 11999); no hubo L-BFGS.
- El residuo PDE normalizado del suelo es 0.099; el maximo en las capas de cable es 3e-4. El error se concentra en el suelo.
- El campo queda por debajo de la referencia casi en todas partes (99.9 % de los puntos), unos 17 a 22 K en los conductores. El punto caliente queda desplazado 0.40 m.
- Balance global 167 %. RMS en Dirichlet: izquierda 2.17 K, derecha 2.00 K, fondo 0.41 K.
- Los terminos de interfaz son pequenos, asi que el fallo no viene de las interfaces.

Hipotesis no verificada: entrenamiento insuficiente o capacidad limitada (red M1 de ancho 32 y profundidad 3, 256 puntos interiores). Falta una corrida exploratoria (mas pasos, L-BFGS, red mayor) en una carpeta nueva para confirmarlo o descartarlo.

## Comparativa de modelos en `aras_flat` (2026-10-11)

Alcance: benchmark controlado de 8 x 4 m, tres cables, perdidas DC prescritas y frontera a 20 C, evaluado contra FEniCS nivel 2. Las compuertas son criterios propuestos de investigacion, no certificacion de ingenieria. No comparar un smoke corto con una corrida convergida como si fueran resultados equivalentes.

| Modelo | Evidencia mas relevante | Metricas en `aras_flat` | Conclusion |
|---|---|---|---|
| M0 | Screening global-fuerte de 400-800 Adam; dos configuraciones en la busqueda anterior | Mejor de los dos screenings: NRMSE 13.16%, error Tmax 29.43 K, peor region 97.9%, balance 79.35%; rechazado | Mantener como control. El buen resultado de M0 en `mms_high_contrast` no se transfiere al cable. No recomendar como solucion de `aras_flat` con la evidencia actual. [Reporte M0](../results/hparam_optimization_20261002/M0/report.md) |
| M1 | 32x3, `tanh`, `lr=0.003`, 20.000 Adam + 200 L-BFGS, puntos 512/256/128/128, dos seeds | Seed 0: NRMSE 0.511%, error Tmax 0.606 K, balance 0.119%, aceptado. Seed 11: NRMSE 0.679%, error Tmax 0.843 K, balance 0.084%, rechazado solo por peor RMSE regional 5.32% | Mejor baseline para este caso. Conservar esta receta y ampliar robustez de seeds/regiones antes de fijarla como configuracion general. [M1 seed 0](../results/phase7/p7_m1_reproduction_20261008/aras_flat/M1_s0/result.json), [M1 seed 11](../results/phase7/p7_m1_reproduction_20261008/aras_flat/M1_s11/result.json) |
| M2 | M2 32x3 desde cero, 20.000 Adam + 200 L-BFGS; ademas, intento 48x5 desde cero detenido en Adam 14.200 | M2 32x3: NRMSE 5.398%, error Tmax 6.592 K, balance 19.91%, rechazado. M2 48x5: inferencia ligera al detenerlo, Tmax 25.78 C, NRMSE 14.36%, error del aumento 80.77%; entrenamiento incompleto y sin auditoria completa | No promover la corrida 48x5 ni repetirla identica. Investigar primero balance/constitutiva y el conflicto entre residuos; el checkpoint detenido solo permite screening, no una conclusion de convergencia. [M2 32x3](../results/phase7/p7_m1_m2_comparison_20261009/aras_flat/M2_s11/result.json), [M2 48x5 detenido](../results/phase7/p7_m2_final_scratch_20261011/aras_flat/M2_w48_d5_s11/stopped_step14200_light_eval.json) |
| CUSP | Feature de interfaz con escala local, cabeza de salida inicial reducida y `sampling=interface`; 32x3, `lr=0.003`, 20.000 Adam + 200 L-BFGS, puntos 512/256/128/128 | Tmax 20.006 C frente a 50.058 C; NRMSE 14.047%; error del aumento 99.98%; balance global 79.56%; rechazado (solo pasa continuidad T de interfaz) | Completó, pero no aprendió el aumento térmico pese a bajar la pérdida train hasta 6.10; no promover esta variante en `aras_flat`. Revisar conflicto/residuos físicos antes de cambiar más capacidad o puntos. [Resultado final](../results/phase7/p7_cusp_local_interface_20261011/aras_flat/CUSP_local_interface_w32_d3_lr003_20k200_s11/result.json), [smoke XPU](../results/phase7/p7_cusp_local_interface_20261011/aras_flat/CUSP_local_interface_xpu_smoke100_s11/result.json) |
| DCSNN | Smoke XPU 32x3, `tanh`, `lr=0.003`, 300 Adam, puntos 512/256/128/128, `sampling=interface`; larga autorizada con la misma receta, 20.000 Adam + 200 L-BFGS | Smoke: train/validación 30.019/13.994 -> 14.157/14.157; Tmax 26.045 C frente a 50.058 C; NRMSE 14.577%; balance 91.10%; rechazado en todas las compuertas. Corrida larga: en curso, sin resultado final todavía | El smoke no pasó, pero se inició la larga por solicitud del usuario para comprobar si el entrenamiento prolongado mejora la subida térmica. No asignar veredicto al modelo hasta la evaluación completa. [Smoke DCSNN](../results/phase7/p7_dcsnn_next_20261011/aras_flat/DCSNN_w32_d3_lr003_interface_pilot300_s11/result.json), [corrida larga](../results/phase7/p7_dcsnn_next_20261011/aras_flat/DCSNN_w32_d3_lr003_interface_20k200_s11/checkpoint_meta.json) |

### Recomendacion actual

1. Usar M1 32x3 como referencia operativa para `aras_flat`; seed 0 paso todas las compuertas y seed 11 quedo cerca, con el unico fallo en la metrica regional.
2. No seleccionar M0 ni M2 como solucion de cable con los resultados actuales; M0 sigue siendo un control util y M2 requiere diagnostico de balance antes de otra corrida larga.
3. No promover CUSP ni DCSNN con las configuraciones largas evaluadas: ambos completaron y fallaron precisión/balance. DCSNN terminó prácticamente a temperatura ambiente pese a la caída de pérdida; investigar la asignación de fuente y el gradiente térmico antes de probar otra arquitectura compartida.

El ensayo CUSP usa el mismo FEM solo para evaluacion. Todos los snapshots y resultados de entrenamiento se conservan bajo `results/phase7`; las métricas CUSP citadas arriba son del resultado final completo, no del smoke.

**Actualización FBPINN (2026-10-12):** smoke XPU de 100 pasos con 16×2, `tanh`, `lr=0.003`, puntos completos M1 y `sampling=interface`. Terminó en 3.083 s, pero train loss = 11,81 millones, validación = 41,98 millones, Tmax = 20,601 °C frente a 50,058 °C, NRMSE = 59,47% y balance global = 12.211%; rechazado. No iniciar corrida larga con esta parametrización. Revisar ventanas/blending y escalas de residuo antes de otro smoke. [Resultado](../results/phase7/p7_fb%20pinn_next_20261012/aras_flat/FBPINN_w16_d2_lr003_interface_pilot100_s11/result.json).

### DeepXDE: comparación de smoke PFNN/FNN (2026-10-12)

Comparación pareada en CPU: PFNN y FNN 12×3, `tanh`, `lr=0.003`, 400 Adam, Hammersley, semillas 11/11011 y 720 puntos por evaluación del objetivo (48 interiores de suelo, 24 por región de cable, 24 por interfaz y 24 por segmento de frontera). Sin L-BFGS. La auditoría física completa estuvo desactivada; ambos estados son `not_audited`.

| Modelo | Pérdida final | RMSE | NRMSE | Tmax predicha vs FEM | Error abs. Tmax | Error del aumento | Tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|
| PFNN | 0,44267 | **3,937 K** | **13,098%** | 29,789 vs 50,058 °C | 20,268 K | 67,431% | 2.034 s |
| FNN | 0,69761 | 3,993 K | 13,285% | 30,696 vs 50,058 °C | **19,361 K** | **64,414%** | 1.511 s |

PFNN fue ligeramente mejor en error global RMSE/NRMSE; FNN fue más cercano en Tmax y tuvo menor error del aumento. Ambos subestimaron fuertemente el calentamiento, por lo que ninguno pasa como solución según este smoke corto. Se selecciona **PFNN provisionalmente** para una comprobación larga por su menor error global de campo; la elección no equivale a aceptación y se revisará con auditoría física completa. [Resultado PFNN](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w12_d3_hammersley_pilot400_retry3_s11/result.json), [resultado FNN](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_FNN_w12_d3_hammersley_pilot400_s11/result.json).

**Calibración XPU para aumentar capacidad:** la máquina dispone de PyTorch 2.9.0+XPU sobre Intel Graphics. Se habilitó `device='xpu'` en DeepXDE; el test automatizado de dos pasos pasó y observó el modelo en `xpu:0`. `auto` conserva CPU en DeepXDE. Intel AI Boost aparece en Windows, pero no hay `torch.npu`/`torch_npu` en este entorno, por lo que no se usa NPU. Para no saturar los 31,5 GB de RAM, el siguiente perfil intermedio sube a PFNN 16×3 y 2.368 puntos objetivo (192 suelo, 96 por capa, 64 por interfaz y 64 por frontera), en vez de los 5.632 puntos completos. El preflight XPU de 10 pasos usó 4,842 GiB de memoria XPU; es solo una calibración de memoria/rendimiento, no una evaluación convergida. [Preflight](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_pilot10_xpu_s11/result.json).

**Smoke PFNN XPU finalizado:** 400 Adam, mejor paso 400, pérdida 0,47890, tiempo 4.479 s (74,6 min), pico de memoria XPU 4,842 GiB. Evaluación del mejor estado: Tmax 30,179 °C frente a 50,058 °C, error Tmax 19,879 K, RMSE 4,037 K, NRMSE 13,432%, error del aumento 66,135%, peor RMSE regional 21,260 K. `full_audit=false`: no hay balance ni saltos de interfaz, por tanto no es un resultado aceptable/convergido. [Resultado PFNN XPU](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_pilot400_xpu_s11/result.json).

La inferencia periódica previa descrita arriba queda supersedida por la evaluación final del paso 400.

**Comparación pareada PFNN/FNN en XPU:** se fijaron 16×3 `tanh`, 2.368 puntos objetivo (192 suelo, 96 por capa, 64 por interfaz y frontera), Hammersley, seeds 11/11011, `lr=0.003`, 400 Adam y sin L-BFGS. Ambos smokes se evaluaron contra FEM, sin auditoría de balance ni interfaces.

| Variante | Pérdida | RMSE | NRMSE | Tmax predicha vs FEM | Error Tmax | Error del aumento | Pico XPU | Tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PFNN | **0,47890** | **4,037 K** | **13,432%** | **30,179 vs 50,058 °C** | **19,879 K** | **66,135%** | 4,842 GiB | 4.479 s |
| FNN | 0,80865 | 4,239 K | 14,104% | 27,469 vs 50,058 °C | 22,589 K | 75,152% | **0,999 GiB** | 3.301 s |

PFNN fue mejor en las métricas FEM reportadas, aunque usó unos 3,84 GiB más de memoria XPU. Ninguno alcanzó buena precisión térmica ni tuvo auditoría física completa; estos resultados no equivalen a convergencia. Se selecciona PFNN para la corrida larga solicitada, manteniendo la nube intermedia para respetar memoria: 20.000 Adam + 100 L-BFGS en XPU, mismo ancho/profundidad, muestreo y semillas. La tasa del smoke de 400 pasos sugería ~62 h para Adam, pero fue demasiado optimista frente a la corrida real; la ETA se recalcula abajo desde el primer checkpoint. [PFNN XPU](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_pilot400_xpu_s11/result.json), [FNN XPU pareado](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_FNN_w16_d3_hammersley_pilot400_xpu_s11/result.json).

**PFNN largo (2026-10-13):** 16×3 en XPU, Hammersley, 2.368 puntos, 20.000 Adam + 100 L-BFGS, seed 11/11011, auditoría completa habilitada. El worker sigue activo y aún no hay `result.json` ni fallo. Inferencia ligera en CPU sobre el primer checkpoint periódico estable (mtime 13:52:53, primer intervalo de 500 pasos): Tmax 33,470 °C vs 50,058 °C, RMSE 3,252 K, NRMSE 10,819%, error del aumento 55,185% y peor RMSE regional 18,205 K. El checkpoint no contiene metadata del paso; no se calculó balance ni salto de interfaz. [JSON de inferencia](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_xpu_20k100_s11/checkpoint_eval_20261013T205736Z.json). Al corte de las 15:57 había ~2,2 GiB de RAM disponibles y el checkpoint llevaba ~2 h sin cambiar. Los primeros 500 pasos tomaron ~4 h 38 min; extrapolación simple: ~7,7 días para Adam, más L-BFGS/auditoría. ETA incierta y se revisará al próximo checkpoint. [Corrida PFNN XPU](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_xpu_20k100_s11/).

**Inferencia del checkpoint siguiente (2026-10-13 16:15 local):** checkpoint estable, nominalmente intervalo 1.000 (el paso exacto no se guarda en el archivo). Tmax 35,456 °C frente a 50,058 °C, RMSE 2,794 K, NRMSE 9,297%, error del aumento 48,579% y peor RMSE regional 17,014 K (`cable_1_layer_0`). Respecto al primer checkpoint, Tmax subió 1,986 K y NRMSE mejoró 1,522 puntos; indica avance térmico, pero sigue muy lejos de la referencia y no incluye balance/interfaz. [JSON de inferencia](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_xpu_20k100_s11/checkpoint_eval_20261013T213643Z.json).

**Inferencia del checkpoint siguiente (2026-10-13 18:25 local):** snapshot estable, nominalmente intervalo 1.500 (`log_every=500`; el paso exacto no está escrito). Tmax 36,277 °C frente a 50,058 °C, RMSE 2,656 K, NRMSE 8,837%, error del aumento 45,848% y peor RMSE regional 16,359 K (`cable_1_layer_0`). Desde el snapshot anterior, Tmax subió 0,821 K y NRMSE mejoró 0,460 puntos porcentuales. La tendencia es favorable, pero la lectura ligera no comprueba balance ni interfaces ni constituye aceptación. [JSON de inferencia](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_xpu_20k100_s11/checkpoint_eval_20261013T233241Z.json).

**Inferencia del checkpoint siguiente (2026-10-14 01:57 UTC):** `model.pt` se actualizó a las 01:27:41 UTC y se leyó estable, con parámetros y temperaturas finitos. Nominalmente corresponde al intervalo 2.000 (`log_every=500`), aunque el paso no se guarda en el archivo. Tmax 36,764 °C frente a 50,058 °C, error Tmax 13,294 K, RMSE 2,583 K, NRMSE 8,595%, error del aumento 44,227% y peor RMSE regional relativo 52,599% (`cable_1_layer_0`). Frente al snapshot anterior, Tmax mejoró 0,487 K y NRMSE 0,242 puntos porcentuales; el hotspot predicho sigue desplazado 0,33 m. Inferencia ligera en CPU, sin balance ni auditoría de interfaces; no es aceptación. [JSON de inferencia](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_xpu_20k100_s11/checkpoint_eval_20261014T015728Z.json).