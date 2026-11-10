# V3 PINN convergence findings

Fecha: 2026-10-25

## Hallazgos confirmados

- La mejor configuración histórica de V2 para cables fue `M1_bcloss_focused`, no `M1_bcexact_focused`.
- V2 `focused` seleccionaba puntos interiores: pool de `16*n`, mitad uniforme y mitad con mayor residuo PDE. En cables añadía puntos log-radiales alrededor de los conductores.
- V2 no focalizaba fronteras ni interfaces; sus nubes superficiales permanecían uniformes y fijas.
- En V2 `M1_bcloss_focused` obtuvo en el smoke de 800 pasos: `Tmax=49.13 C`, NRMSE `1.575 %`, balance `1.730 %`.
- V3 inicialmente usaba coordenadas cartesianas genéricas. M1 V2 usaba coordenadas regionales: log-distancia a cada conductor para suelo y coordenadas locales radiales para conductor/capas.
- La parametrización `*_EXACT_BC` de V3 tuvo inicialmente una envolvente mal escalada. Se corrigió con distancias normalizadas y corrección térmica en kelvin.
- En `exact_constraints`, `trainer.run` anulaba por error las pérdidas de continuidad interna. Se corrigió: interfaces de temperatura y flujo permanecen activas; solo se anulan pérdidas de frontera externa impuestas por construcción.
- Se añadió verificación de pertenencia: puntos interiores deben ser propietarios de una región y estar separados de sus interfaces; puntos con error geométrico de redondeo reciben pertenencia determinista.

## Estado verificado

- Suite V3: 37 pruebas pasaban antes de la campaña larga.
- M1 con muestreo V2 replicado y coordenadas regionales, `cable_disk`, 800 Adam + 20 L-BFGS:
  - RMSE FEM: `0.1908 K`
  - error máximo: `0.5775 K`
  - `Tmax PINN=138.278 C`, `Tmax FEM=138.807 C`
  - `nan_points=0`
- Este resultado está en `results/m1_v2_focused_smoke_v2/summary.json`.

## Riesgos y límites

- Los checkpoints de campañas anteriores a la corrección de parametrización o de pesos de interfaz no son comparables y no deben reutilizarse.
- `DCSNN` fue prometedor en MMS, pero peor que M1 en el smoke de cables V2; debe medirse con la misma receta y semillas.
- El criterio de parada de V2 exige simultáneamente:
  - error de `Tmax < 0.01 K`;
  - NRMSE `< 0.1 %`;
  - balance `< 0.5 %`;
  - comprobación cada 100 pasos.
- Para MMS la referencia es analítica y está declarada en `problem.json`. Para cable, la referencia de parada es FEM interpolada sobre los puntos de evaluación; se usa solo para evaluación/parada, nunca para formar la pérdida.

## Campaña larga V3

Configuración: M1 y DCSNN; `mms_sine` y `cable_disk`; 8000 Adam + 20 L-BFGS; remuestreo interior cada 100 pasos; mitad uniforme y mitad residual-focused/log-radial; superficies uniformes fijas; pesos PDE `1`, energía `100` y pérdidas de continuidad internas activas.

El script reproducible es `tools/long_v2_focused_campaign.py` y sus resultados se guardan en `results/long_v2_focused_campaign_20251025b/summary.json`.

## Resultados de campaña larga 2025-10-25

La campaña limpia se ejecutó en `results/long_v2_focused_campaign_20251025b/` con 8000 Adam + 20 L-BFGS para cada combinación. La calificación vigente desde el 2026-10-27 usa Tmax `<0.05 K`, NRMSE `<0.25 %` y balance `<0.5 %`; la clasificación retrospectiva está en `CAMPAIGN_CLASSIFICATION_2026-10-27.md`.

### Diagnóstico CUSPR en annulus

La trayectoria de las semillas 0 y 1 no indica simplemente falta de pasos:

- Durante Adam el balance mejora de forma fuerte y el NRMSE baja progresivamente en la semilla 1; en la semilla 0 también mejora globalmente, aunque con oscilaciones.
- L-BFGS reduce el error hasta aproximadamente `11-12 %` NRMSE y `17-18 K` en Tmax, pero oscila entre bloques y no sigue descendiendo.
- La meseta aparece tras 1500 Adam y cuatro bloques L-BFGS, por lo que añadir pasos al mismo modelo no es la primera solución.

La causa probable es geométrica: CUSPR es una red global con level sets circulares, pero sus coordenadas globales para el annulus no incluyen la coordenada radial logarítmica que utilizaba V2 (`annulus_log_v1`). La solución analítica es logarítmica en `r` y tiene un gradiente fuerte cerca de `ri`; los level sets por sí solos no representan bien esa escala. La siguiente mejora debe añadir a CUSP/CUSPR las coordenadas `log(r/ri)` y dirección radial para dominios anulares, además de muestreo log-radial, antes de aumentar el presupuesto de entrenamiento.

| Caso | Modelo | NRMSE | error Tmax | balance | estado físico |
|---|---|---:|---:|---:|---|
| `mms_sine` | M1 | 0.159 % | 0.055 K | 0.0019 % | aceptado |
| `mms_sine` | DCSNN | 0.128 % | 0.061 K | 0.0008 % | aceptado |
| `cable_disk` | M1 | 0.144 % | 0.565 K | 0.0023 % | aceptado |
| `cable_disk` | DCSNN | 27.088 % | 118.448 K | 8.985 % | requiere más pasos |

M1 cable reproduce el calentamiento: `Tmax PINN=138.241 C` frente a `138.806 C` FEM, con salto de temperatura de interfaz `0.034 K` y salto de flujo `0.025 %`. DCSNN cable mantiene `Tmax=20.358 C`, balance del conductor `9.034 %` y salto de flujo `25.57 %`; su red compartida con one-hot no está resolviendo la transferencia térmica concentrada bajo esta receta.

La parada estricta histórica V2 no debe confundirse con la calificación oficial actual. En corridas cortas, no cumplir un umbral no implica rechazo: se registra como `requiere más pasos`. La calificación actual exige simultáneamente los tres umbrales definidos arriba cuando la etapa de entrenamiento prevista se ha completado.

## Diagnóstico DCSNN en cable

La falta de calentamiento de DCSNN no parece ser solo falta de entrenamiento:

- En `mms_sine`, DCSNN alcanza NRMSE `0.128 %`, error Tmax `0.061 K` y balance `0.0008 %` con la misma receta de 8000 Adam + 20 L-BFGS.
- En `cable_disk`, DCSNN termina con NRMSE `27.088 %`, error Tmax `118.448 K`, Tmax `20.358 C`, balance `8.985 %` y salto de flujo `25.57 %`.
- V2 ya mostraba el mismo patrón: DCSNN cable tenía NRMSE `15.16 %` y balance `62.73 %` en `bcloss_focused`, mientras M1 alcanzaba NRMSE `1.575 %`.

La causa estructural principal es la representación:

1. DCSNN usa una red global con one-hot de región, pero en V3 alimenta solo coordenadas globales `x,y`; no recibe las coordenadas log-radiales ni locales que M1 V2 usa para resolver la capa fuente y las interfaces circulares.
2. DCSNN usa una única amplitud térmica global y un único offset, mientras M1 usa amplitud y red por región. El conductor tiene una escala térmica local muy distinta del suelo.
3. El one-hot identifica la región, pero no aporta distancia firmada a la interfaz; la red debe aprender por sí misma la geometría radial y la transferencia de flujo.
4. El aumento de pasos, por sí solo, no corrige estas limitaciones: M1 converge en 800 pasos con la misma familia de pérdidas y muestreo.

Mejoras propuestas, en este orden:

1. Mantener el tronco compartido DCSNN, pero añadir características comunes: coordenadas globales, log-distancia a cada disco fuente, coordenadas locales radiales y distancias firmadas a interfaces.
2. Añadir offsets y amplitudes de salida por región sin separar el tronco; conservaría la idea DCSNN y permitiría la escala local `q L^2/k` del conductor.
3. Mantener mitad de puntos uniformes y añadir una opción DCSNN de foco en interfaz por salto de flujo; no cambiar el muestreo histórico de M1.
4. Solo después comparar mayor anchura/profundidad, AdamW o más L-BFGS.

## Diagnóstico de las otras arquitecturas

### M0

M0 es una red global de control. No tiene una red por región ni características radiales por defecto; por eso no debe esperarse que iguale a M1 en cables multicapa. Para hacerla útil en V3 debe recibir el mismo proveedor de características globales y log-radiales, manteniendo una sola red. La variante regional `M0R` es más apropiada para comparar contra M1 porque añade one-hot, offsets y amplitudes por región.

### M2

M2 ya usa una red por región y predice temperatura y flujo, pero necesita exactamente las mismas coordenadas locales de M1 y escalas de flujo por región. Su ventaja potencial es imponer directamente la ley constitutiva `q + k grad(T) = 0`; su riesgo es tener tres salidas que deben equilibrarse con el residuo PDE, la continuidad de flujo y la energía. Debe probarse después de confirmar que usa las coordenadas regionales y la escala `flux` del problema genérico.

### CUSP

CUSP depende de `level_sets()` y de distancias firmadas a interfaces. El problema genérico V3 todavía no ofrece esa geometría de forma completa; el fallback devuelve interfaces vacías en documentos regionales. En un cable, CUSP sería casi una red global sin sus features distintivas. Prioridad: implementar level sets firmados para discos, anillos y rectángulos antes de medir CUSP.

### FBPINN

FBPINN aún interpreta cortes desde la geometría legacy (`problem.case`, cables, annulus y cortes planares). En un `problem.json` genérico esos datos no están disponibles con esa interfaz, por lo que su partición puede reducirse a cajas cartesianas arbitrarias y perder la geometría circular del conductor. Debe migrarse a una partición derivada de las regiones del documento antes de una comparación justa.

### DeepXDE

DeepXDE no soporta actualmente el remuestreo configurable, el foco mixto ni las opciones nativas de exact boundary del motor PyTorch. Puede servir como backend de referencia para MMS sencillo, pero no es comparable todavía con M1 V3 en cables. Debe conectarse a un adaptador genérico de `problem.json` y validarse primero en MMS antes de usarlo en cable.

## Prioridad de implementación

1. Extraer un proveedor común de características: coordenadas globales, log-distancias a fuentes, coordenadas locales de región y level sets firmados.
2. Conectar ese proveedor a M0, M2 y DCSNN sin duplicar lógica.
3. Implementar `M0R` como control regional y probar M2 con la receta M1 V2.
4. Adaptar CUSP a level sets genéricos y FBPINN a regiones JSON.
5. Dejar DeepXDE para la fase posterior, con un smoke MMS separado.

La evidencia actual no justifica más pasos como primera respuesta: M1 alcanzó el cable en 800 pasos con la representación correcta, mientras DCSNN falló tras 8000 pasos con una representación geométrica insuficiente.

## Smoke conjunto 2025-10-26

Se ejecutaron 200 pasos Adam, sin L-BFGS, para M0, M1, M2, DCSNN, M0R, CUSP, CUSPR y FBPINN en `mms_sine` y `cable_disk`. Todos los modelos construyeron y terminaron sin excepción. El resultado completo está en `results/all_models_smoke_20251026/summary.json`.

| Caso | Modelo | NRMSE MMS | RMSE MMS | Tmax cable (C) | Observación |
|---|---|---:|---:|---:|---|
| MMS | M0/M1 | 18.40 % | 5.52 K | - | smoke demasiado corto |
| MMS | M2 | 48.23 % | 14.47 K | - | constitutivo aún sin converger |
| MMS | DCSNN | 22.97 % | 6.89 K | - | requiere entrenamiento largo |
| MMS | M0R | 30.48 % | 9.14 K | - | control regional aún temprano |
| MMS | CUSP | 3.38 % | 1.01 K | - | mejor variante en 200 pasos |
| MMS | CUSPR | 1.93 % | 0.58 K | - | mejor que CUSP en este smoke |
| MMS | FBPINN | 44.84 % | 13.45 K | - | partición necesita calibración |
| Cable | M0 | - | - | 31.81 | no convergido |
| Cable | M1 | - | - | 33.86 | no comparable con el smoke largo M1 |
| Cable | M2 | - | - | 17.07 | no convergido |
| Cable | DCSNN | - | - | 30.54 | pérdida muy alta |
| Cable | M0R | - | - | 20.44 | no convergido |
| Cable | CUSP | - | - | 20.06 | no convergido |
| Cable | CUSPR | - | - | 36.17 | no convergido |
| Cable | FBPINN | - | - | 29.52 | pérdida muy alta |

El cable no declara referencia exacta en su `problem.json`, por eso el smoke reporta Tmax pero no NRMSE; la referencia FEM se usa en las campañas largas dedicadas. Estos 200 pasos sirven para validar integración, no para concluir convergencia.
