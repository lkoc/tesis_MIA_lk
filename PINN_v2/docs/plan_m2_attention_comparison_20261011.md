# M2: Atención y Comparación de Modelos

**Fecha:** 2026-10-11  
**Caso:** `aras_flat`, referencia fija FEniCSx nivel 2.  
**Estado:** comparación baseline/RBA/SA completada el 2026-10-11. Los tres brazos usaron el mismo checkpoint y completaron 5.000 pasos Adam con auditoría FEM completa.

## Objetivo

Comparar de forma pareada la M2 actual sin atención contra RBA y SA, cambiando únicamente el mecanismo de atención. Determinar si la reponderación punto a punto del residuo PDE mejora la solución y no solo la pérdida de entrenamiento.

La referencia FEM se usa exclusivamente para evaluar. No interviene en el objetivo, la selección del checkpoint ni los pesos.

## Arquitectura y Muestreo por Corrida

El total cuenta los puntos interiores, de interfaz y de frontera que entran en una evaluación del objetivo. No cuenta los puntos de la auditoría FEM. La validación del trainer usa además una nube fija independiente del mismo tamaño.

| Corrida/caso | Arquitectura | Ancho × profundidad | Activación | Interior | Interfaces | Frontera | Total objetivo |
|---|---|---:|---|---:|---:|---:|---:|
| M0 validación previa | M0 global fuerte, 1 red | 24 × 3 | `tanh` | 1.792 | 1.152 | 384 | 3.328 |
| M1 seed 0 | M1 regional fuerte, 13 redes | 32 × 3 | `tanh` | 3.584 | 1.536 | 512 | 5.632 |
| M1 seed 11 | M1 regional fuerte, 13 redes | 32 × 3 | `tanh` | 3.584 | 1.536 | 512 | 5.632 |
| M2 inicial/continuaciones | M2 regional mixto, 13 redes | 32 × 3 | `tanh` | 3.584 | 1.536 | 512 | 5.632 |
| Baseline de atención | M2 regional mixto, 13 redes | 32 × 3 | `tanh` | 3.584 | 1.536 | 512 | 5.632 |
| RBA | M2 regional mixto, 13 redes | 32 × 3 | `tanh` | 3.584 | 1.536 | 512 | 5.632 |
| SA | M2 regional mixto, 13 redes | 32 × 3 | `tanh` | 3.584 | 1.536 | 512 | 5.632 |

Desglose para M1/M2 y la comparación de atención: interiores = 512 en suelo + 12 regiones de cable × 256 = 3.584; interfaces = 12 × 128 = 1.536; frontera = 4 × 128 = 512. Total = **5.632 puntos de entrenamiento por evaluación del objetivo**. Para M0 previo: 256 + 12 × 128 interiores, 12 × 96 interfaces y 4 × 96 frontera = **3.328**.

## Experimento Controlado

Los tres brazos parten de los mismos pesos M2 y usan la misma configuración. Solo cambia `attention`.

| Brazo | Atención | Tasa interna |
|---|---|---:|
| Baseline | ninguna | n/a |
| RBA | residual-based attention | 0.01 (implementación actual) |
| SA | self-adaptive weights | 0.01 (implementación actual) |

**Checkpoint común:** `results/phase7/p7_m1_m2_comparison_20261009/aras_flat/M2_s11_extend10k_lr1e-4/best_train_loss.pt`. Es el checkpoint de inicialización que se usó en los brazos de sampling comparados más abajo; los tres brazos de atención cargarán exactamente este mismo state dict.

**Arquitectura en los tres brazos:**

| Brazo | Modelo | Redes/subredes | Ancho | Profundidad oculta | Activación | Salidas por región |
|---|---|---:|---:|---:|---|---|
| Baseline | M2 | 13 (una por región) | 32 | 3 capas | `tanh` | T, qx, qy |
| RBA | M2 | 13 (una por región) | 32 | 3 capas | `tanh` | T, qx, qy |
| SA | M2 | 13 (una por región) | 32 | 3 capas | `tanh` | T, qx, qy |

**Configuración común ejecutada:**

| Parámetro | Valor |
|---|---:|
| Arquitectura | M2, ancho 32, profundidad 3, `tanh` |
| Dispositivo | XPU |
| Adam | 5.000 pasos, `lr=1e-4` |
| L-BFGS | 0 para los tres brazos; atención y L-BFGS adaptativo no se combinan actualmente |
| Semillas registradas | inicialización 11, muestreo 11011; misma validación fija generada por el trainer |
| Puntos | `n_interior=512`, `n_layer=256`, `n_face=128`, `n_boundary=128` |
| Total de puntos de entrenamiento por evaluación del objetivo | **5.632** = 3.584 interiores + 1.536 interfaces + 512 frontera |
| Validación independiente | nube fija adicional de **5.632** puntos; no se mezcla con los puntos de entrenamiento |
| Sampling | `mixed`, nube fija (`resample_every=0`) |
| Pesos | valores por defecto iguales en los tres brazos |
| Validación | cada 500 pasos, nube física fija independiente |
| Evaluación final | auditoría completa, referencia FEM nivel 2 |

**Directorios de salida propuestos:**

- `results/phase7/p7_m2_attention_20261011/aras_flat/baseline_s11`
- `results/phase7/p7_m2_attention_20261011/aras_flat/rba_s11`
- `results/phase7/p7_m2_attention_20261011/aras_flat/sa_s11`

### Resultados

| Brazo | Tiempo | Pérdida validación (mejor) | NRMSE | Error Tmax | Peor región | Balance global | Balance regional máx. | Salto T máx. | Salto flujo máx. | Veredicto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 1,30 h | 2,1676 (paso 2.000) | **2,628%** | **2,645 K** | **18,774%** | **2,446%** | 0,694% | 0,0338 K | **0,0425%** | rechazado |
| RBA | 1,57 h | **2,1419** (paso 4.500) | 2,647% | 2,679 K | 18,869% | 2,608% | 0,716% | 0,0346 K | 0,0460% | rechazado |
| SA | 1,36 h | 2,1613 (paso 3.500) | 2,640% | 2,699 K | 18,873% | 2,562% | **0,618%** | **0,0330 K** | 0,0448% | rechazado |

Los tres comparten M2, 32×3 `tanh`, 5.000 Adam, `lr=1e-4`, 0 L-BFGS, 5.632 puntos de objetivo y las semillas 11/11011/1011014. Solo cambió `attention`. Artefactos: [baseline](../results/phase7/p7_m2_attention_20261011/aras_flat/baseline_s11/result.json), [RBA](../results/phase7/p7_m2_attention_20261011/aras_flat/rba_s11/result.json), [SA](../results/phase7/p7_m2_attention_20261011/aras_flat/sa_s11/result.json) y [resumen agregado](../results/phase7/p7_m2_attention_20261011/aras_flat/summary.json).

**Conclusión:** en esta comparación de una semilla, RBA minimizó mejor la pérdida física retenida, pero no mejoró la precisión FEM: el baseline tuvo menores NRMSE, error Tmax, error regional, balance global y salto de flujo. SA mejoró levemente el balance regional y el salto máximo de T, sin compensar sus otros errores. Ninguno cumplió todos los gates. No hay evidencia para promover RBA o SA como reemplazo del baseline; el siguiente paso de atención, si se necesita confirmar robustez, sería repetir baseline y RBA con una segunda semilla, no aumentar pasos en los tres brazos.

Registrar por brazo: NRMSE, error de Tmax y del aumento térmico, peor RMSE regional, balance global y regional, saltos de T/flujo en interfaces, pérdida de validación, tiempo y veredicto estricto/con tolerancia. Conservar `latest`, `best_train_loss` y `best_validation`.

### Regla de decisión

1. Comparar primero los tres resultados con idénticos pasos y checkpoint inicial.
2. No declarar ganador usando solo pérdida de entrenamiento o validación física; FEM sigue siendo una auditoría externa.
3. Si un brazo mejora de forma material las métricas FEM sin empeorar balance/interfaz, repetir baseline y ganador con otra semilla (por ejemplo, 23) antes de promoverlo.
4. RBA/SA aquí son mecanismos adaptados que reponderan residuos PDE por punto; no deben describirse como algoritmos idénticos a sus publicaciones originales.

## Por Qué M0 Se Depriorizó

M0 no se considera imposible ni se elimina del catálogo. Se dejó fuera de la búsqueda actual por baja prioridad y evidencia desfavorable en el objetivo de esta fase:

- M0 es una sola red global en forma fuerte. M1/M2 usan una red por región; M2 además predice temperatura y flujo. En `aras_flat` hay múltiples materiales e interfaces, por lo que M0 debe representar en una sola red el campo y sus cambios regionales mientras las condiciones de interfaz siguen siendo pérdidas suaves. Esto es una desventaja estructural plausible, no una causa demostrada por sí sola.
- La validación registrada de M0 con 800 Adam y sin L-BFGS obtuvo NRMSE **14,35%**, error del aumento de Tmax **100,31%**, peor región **100,30%**, balance global **101,34%** y salto RMS de flujo **7,44%**. Falló NRMSE, Tmax, región, balance y flujo; pasó solo el salto de temperatura de interfaz. Resultado: [M0 active validation](../results/pde_active_validation_20261007/M0/result.json).
- La búsqueda staged de M0 eligió su mejor compromiso, pero declaró explícitamente que **ningún candidato cumplió todos los gates**. El candidato seleccionado (ancho 48, profundidad 4, SiLU, SA, dos veces los puntos base y pesos fuertemente modificados) seguía mostrando en `aras_flat` errores del aumento de Tmax de alrededor de **80–90%**, peor región de alrededor de **80–90%** y balance global de alrededor de **29–31%** en los finalistas. Resultado: [M0 staged search](../results/hparam_optimization_20261004_staged/M0/summary.json).
- En cambio, M1 ya tiene un resultado que pasa todos los gates estrictos en la misma referencia/caso. Seguir intentando hacer que M0 pase ahora consume cómputo sin cambiar la comparación principal; se puede reabrir como trabajo de capacidad/representación si cambia la prioridad.

Por tanto, la decisión es **depriorizar M0 para esta ronda**, no afirmar que ningún M0 pueda cumplir con más cambios o cómputo.

## Evidencia de M1

Los gates nominales del proyecto son NRMSE ≤5%, error de Tmax (aumento) ≤5%, peor RMSE regional ≤5%, balance global y regional ≤2%, salto máximo de T ≤0,1 K y salto RMS de flujo ≤1%. Se permite un exceso regional menor de 1 punto porcentual como `accepted_with_tolerance`; ese veredicto no equivale a pasar el umbral nominal estricto.

| Corrida | Pasos | NRMSE | Error Tmax | Peor región | Balance global | T interfaz máx. | Flujo interfaz máx. | Resultado |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Seed 0 | 20.000 Adam + 200 L-BFGS | 0,511% | 0,606 K (2,016% del aumento) | 3,655% | 0,119% | 0,0734 K | 0,0330% | `accepted` |
| Seed 11 | 20.000 Adam + 200 L-BFGS | 0,679% | 0,843 K (2,806% del aumento) | 5,320% | 0,0837% | 0,0711 K | 0,0320% | `accepted_with_tolerance` |

- Seed 0 cumple los umbrales estrictos; tiempo: **5,29 h**. Artefacto: [M1_s0](../results/phase7/p7_m1_reproduction_20261008/aras_flat/M1_s0/result.json).
- Seed 11 cumple los demás umbrales; solo excede el regional estricto por 0,320 puntos y queda dentro de la tolerancia regional declarada. Tiempo: **7,46 h**. Artefacto: [M1_s11](../results/phase7/p7_m1_reproduction_20261008/aras_flat/M1_s11/result.json).
- Es evidencia de que M1 puede pasar en este benchmark; dos semillas no bastan para afirmar robustez general.

## Historial de M2

| Etapa | Configuración | NRMSE | Error Tmax | Peor región | Balance global | Tiempo | Lectura |
|---|---|---:|---:|---:|---:|---:|---|
| M2 inicial | 20.000 Adam + 200 L-BFGS | 5,398% | 6,592 K (21,93%) | 34,95% | 19,91% | 2,90 h | Rechazado en varios gates |
| Warm start | +20.000 Adam, `lr=3e-4`, +200 L-BFGS | 2,650% | 2,750 K (9,15%) | 18,96% | 2,289% | 2,79 h adicionales | Gran mejora, pero aún falla Tmax/región/balance global |
| Warm start | +10.000 Adam, `lr=1e-4`, sin L-BFGS | 2,650% | 2,688 K (8,94%) | 18,89% | 2,584% | 1,08 h | Mejora FEM marginal; balance empeora. El `failed` del trainer fue un `NameError` del callback de snapshots tras completar Adam; los pesos se auditaron por separado |

Artefactos: [M2 inicial](../results/phase7/p7_m1_m2_comparison_20261009/aras_flat/M2_s11/result.json), [warm start +20k](../results/phase7/p7_m1_m2_comparison_20261009/aras_flat/M2_s11_extend20k_lr3e-4/result.json), [warm start +10k](../results/phase7/p7_m1_m2_comparison_20261009/aras_flat/M2_s11_extend10k_lr1e-4/adam_10000_eval.json).

## Mejores Hiperparámetros Ejecutados por Modelo

Esta tabla distingue entre mejor configuración aceptada, mejor candidato exploratorio y warm start; no presenta un candidato fallido como solución válida.

| Modelo | Mejor configuración/resultado disponible en `aras_flat` | Métricas principales | Estado |
|---|---|---|---|
| M0 | Búsqueda staged: global fuerte, ancho 48, profundidad 4, `silu`, `lr=7,974e-4`, SA, 800 Adam, sin L-BFGS, point multiplier 2×; pesos PDE 1, constitutivo 0,1, energía 10, Dirichlet 0,1, interfaz T/flujo 0,1, Neumann 0,1, Robin 10 | Mejor tradeoff seleccionado: error de aumento Tmax 80,33–80,83%; peor región 80,33–80,82%; balance global 29,38–29,51%, según seed/case row. Ningún finalista cumplió todos los gates | Mejor candidato de búsqueda, **no aceptado** |
| M1 | Regional fuerte, ancho 32, profundidad 3, `tanh`, `lr=0,003`, 20.000 Adam + 200 L-BFGS, pesos por defecto, puntos 512/256/128/128 | Seed 0: NRMSE 0,511%, Tmax 0,606 K, peor región 3,655%, balance 0,119% | `accepted`; 5,29 h |
| M2, mejor FEM observado | Regional mixto, ancho 32, profundidad 3, `tanh`, pesos por defecto, puntos 512/256/128/128. Warm-start con extensión 20k a `lr=3e-4`; el baseline de atención añadió 5.000 Adam a `lr=1e-4` sin atención | NRMSE 2,628%, Tmax 2,645 K, peor región 18,774%, balance global 2,446% | Mejor compromiso FEM M2 a la fecha, pero **rechazado**; no es entrenamiento desde cero |
| M2, desde cero | Regional mixto, ancho 32, profundidad 3, `tanh`, `lr=0,003`, 20.000 Adam + 200 L-BFGS, pesos por defecto | NRMSE 5,398%, Tmax 6,592 K, peor región 34,953%, balance global 19,910% | **Rechazado**; 2,90 h |

## Último Ensayo M2 Desde Cero

Para probar si mayor capacidad mejora M2 sin depender de la trayectoria warm-start, se eligió ancho 48 y profundidad 5 (antes 32×3), manteniendo la receta larga que hizo converger M1: `tanh`, `lr=0,003`, 20.000 Adam + 200 L-BFGS, puntos 512/256/128/128, pesos por defecto, XPU, init seed 11 y sampling seed 11011. No se transfiere ningún peso; la comparación mide una inicialización nueva con más capacidad.

Salida: `results/phase7/p7_m2_final_scratch_20261011/aras_flat/M2_w48_d5_s11`. Actualización 2026-10-11: corrida **detenida por no-pass en screening ligero**, en Adam paso 14.200; se preservó el checkpoint. La inferencia ligera dio Tmax 25,781 °C, NRMSE 14,357% y error del aumento 80,767%. Como `full_audit_run=false` y el entrenamiento quedó incompleto, estas cifras no son una evaluación final ni incluyen balance/saltos de interfaz. [Screening al detener](../results/phase7/p7_m2_final_scratch_20261011/aras_flat/M2_w48_d5_s11/stopped_step14200_light_eval.json), [metadata](../results/phase7/p7_m2_final_scratch_20261011/aras_flat/M2_w48_d5_s11/checkpoint_meta.json).

Lecturas intermedias no finales:

| Snapshot | Tmax predicha | Error Tmax | NRMSE | Error del aumento | Peor región | Nota |
|---|---:|---:|---:|---:|---:|---|
| ~800 Adam | 25,857 °C | 24,201 K | 14,534% | 80,514% | 80,514% | checkpoint estable |
| 2.000 Adam | 25,371 °C | 24,686 K | 14,287% | 82,130% | 87,824% | checkpoint estable; pérdida actual 0,463, mejor train loss 0,443 en paso 1.992 |
| 14.200 Adam (al detener) | 25,781 °C | 24,277 K | 14,357% | 80,767% | 80,767% | screening ligero; sin auditoría completa |

Los tres registros de la tabla son inferencias ligeras (`full_audit_run=false`): no informan balance ni saltos de interfaz y no son resultados de convergencia. Entre los snapshots 800 y 2.000, el NRMSE mejoró ligeramente, pero Tmax y el peor error regional empeoraron; al paso 14.200 el campo seguía muy por debajo del FEM. Archivos: [step800](../results/phase7/p7_m2_final_scratch_20261011/aras_flat/M2_w48_d5_s11/step800_light_eval.json), [step2000](../results/phase7/p7_m2_final_scratch_20261011/aras_flat/M2_w48_d5_s11/step2000_light_eval.json), [step14200](../results/phase7/p7_m2_final_scratch_20261011/aras_flat/M2_w48_d5_s11/stopped_step14200_light_eval.json).

### Smoke de pesos y sampling

- Subir energía a ×5 redujo el balance global de aproximadamente **2,58% a 2,46%**, pero no mejoró NRMSE/Tmax/región de manera material.
- Duplicar puntos por capa o usar sampling `interface` con remuestreo cada 500 tampoco produjo una mejora global; el primero empeoró el balance y el segundo no mejoró la precisión térmica.
- Tabla completa: [M2 sampling smoke](../results/phase7/p7_m1_m2_comparison_20261009/aras_flat/M2_s11_sampling_smoke_20261010/summary.json).
- Diagnóstico de gradientes en el checkpoint M2: normas PDE `0,0492`, constitutivo `0,0370`, energía `0,0521`; las magnitudes son comparables, pero coseno PDE–energía **−0,855**. Esto sugiere conflicto de gradientes más que simple desbalance de escala. Archivo: [gradient_norm_probe](../results/phase7/p7_m1_m2_comparison_20261009/aras_flat/M2_s11_sampling_smoke_20261010/gradient_norm_probe.json).

## Interpretación y Alcance

La recomendación inmediata, tras el screening al paso 14.200, es no reiniciar la misma M2 48×5 desde cero. Antes de volver a XPU, aislar el conflicto PDE–energía y comparar una intervención por vez, incluyendo RBA/SA desde el checkpoint común solo si se mantienen las mismas métricas de FEM y la auditoría física. CUSP se documenta junto con la comparación de modelos en [resultados](results.md); su corrida larga terminó rechazada (Tmax 20.006 C, NRMSE 14.047%, balance 79.56%) y no altera la conclusión sobre M2.

El caso `aras_flat` de estas corridas usa pérdidas DC uniformes de aproximadamente 18,60471 W/m por cable y frontera a 20 °C. No es una reproducción térmica completa del artículo Aras con pérdidas AC/pérdidas distribuidas; estos resultados solo justifican conclusiones para el benchmark controlado descrito.

## Comparativa consolidada de modelos en `aras_flat`

Actualización: 2026-10-12. Esta sección reúne el resultado disponible por arquitectura para el mismo benchmark controlado. Los entrenamientos completos se comparan con sus auditorías completas; los screenings ligeros y smokes se identifican explícitamente y no se interpretan como convergencia.

| Modelo/corrida | Configuración | Evidencia y métricas | Estado/lectura |
|---|---|---|---|
| M0, screening | Una red global fuerte; selección staged 48×4 SiLU+SA, 800 Adam, 2× puntos | En los finalistas `aras_flat`: error del aumento Tmax aprox. 80–90%, peor región 80–90%, balance global 29–31%; ningún finalista pasó todas las compuertas. Un screening anterior obtuvo NRMSE 13.16%, error Tmax 29.43 K y balance 79.35%. | Rechazado para este cable; conservar como control. Su buen resultado en MMS no se transfiere automáticamente a `aras_flat`. Ver [M0 staged](../results/hparam_optimization_20261004_staged/M0/report.md) y [M0 screening](../results/hparam_optimization_20261002/M0/report.md). |
| M1 seed 0, largo | Regional fuerte 32×3, tanh, `lr=0.003`, 20.000 Adam + 200 L-BFGS, puntos 512/256/128/128, mixed | NRMSE 0.511%; error Tmax 0.606 K (2.016% del aumento); peor región 3.655%; balance 0.119%; veredicto `accepted`. | Mejor referencia operativa observada para `aras_flat`; es una seed, no una garantía de robustez. [Resultado](../results/phase7/p7_m1_reproduction_20261008/aras_flat/M1_s0/result.json). |
| M1 seed 11, largo | Misma receta de M1 seed 0 | NRMSE 0.679%; error Tmax 0.843 K (2.806%); balance 0.084%; rechazado solo por peor RMSE regional 5.32%. | Confirma precisión global alta, pero muestra sensibilidad regional a la seed; ampliar semillas antes de declarar robustez. [Resultado](../results/phase7/p7_m1_reproduction_20261008/aras_flat/M1_s11/result.json). |
| M2 seed 11, largo desde cero | Regional mixto 32×3, tanh, `lr=0.003`, 20.000 Adam + 200 L-BFGS, puntos completos | NRMSE 5.398%; error Tmax 6.592 K (21.93%); peor región 34.95%; balance 19.91%; rechazado. | Inferior a M1 en esta receta y caso; no promover sin aislar antes el balance y el conflicto entre residuos. [Resultado](../results/phase7/p7_m1_m2_comparison_20261009/aras_flat/M2_s11/result.json). |
| M2 48×5, intento desde cero | 20.000 Adam + 200 L-BFGS solicitados; detenido en Adam 14.200 | Screening ligero: Tmax 25.781 °C, NRMSE 14.357%, error del aumento 80.767%; sin auditoría completa. | Corrida incompleta, marcada no-pass de screening; no es evidencia de un M2 convergido. [Screening](../results/phase7/p7_m2_final_scratch_20261011/aras_flat/M2_w48_d5_s11/stopped_step14200_light_eval.json). |
| CUSP, largo completo | Red global fuerte con features de interfaz de escala local y cabeza inicial reducida; 32×3, tanh, `lr=0.003`, `sampling=interface`, 20.000 Adam + 200 L-BFGS, puntos completos | Tmax 20.006 °C frente a 50.058 °C; NRMSE 14.047%; error del aumento 99.98%; balance 79.56%; solo pasó continuidad de temperatura de interfaz. | Rechazado pese a completar el entrenamiento; no promover esta variante en `aras_flat` con la formulación actual. [Resultado](../results/phase7/p7_cusp_local_interface_20261011/aras_flat/CUSP_local_interface_w32_d3_lr003_20k200_s11/result.json). |
| DCSNN, smoke | Red compartida con etiqueta one-hot por región; M1 32×3/tanh/`lr=0.003`, 300 Adam, `sampling=interface`, puntos completos | Train/validación 30.019/13.994 → 14.157/14.157; Tmax 26.045 °C; NRMSE 14.577%; balance 91.10%; rechazado en todas las compuertas. | El smoke no pasa precisión ni física, pero la pérdida descendió mucho; por solicitud del usuario se promovió a corrida larga para comprobar si el entrenamiento prolongado corrige el campo. [Smoke](../results/phase7/p7_dcsnn_next_20261011/aras_flat/DCSNN_w32_d3_lr003_interface_pilot300_s11/result.json). |
| DCSNN, largo en curso | DCSNN 32×3, tanh, `lr=0.003`, 20.000 Adam + 200 L-BFGS, XPU, seed 11/11011, `sampling=interface`, puntos 512/256/128/128 | Metadata al checkpoint (2026-10-12 07:58 UTC): Adam paso 6.400/20.000, train loss 11.864; mejor validation 11.749 (paso 6.000). Inferencia ligera CPU estable al paso 6.400: Tmax 20.172 °C, NRMSE 13.945%, error del aumento 99.426%, peor región 99.420%; parámetros/campo finitos. La inferencia es `full_audit_run=false`, sin balance ni saltos de interfaz. | En ejecución. El campo sigue prácticamente en ambiente a 6.400 pasos; no decidir aceptación final antes de terminar, pero esta lectura intermedia no muestra todavía recuperación térmica. [Inferencia ligera paso 6.400](../results/phase7/p7_dcsnn_next_20261011/aras_flat/DCSNN_w32_d3_lr003_interface_20k200_s11/step6400_light_eval.json), [metadata viva](../results/phase7/p7_dcsnn_next_20261011/aras_flat/DCSNN_w32_d3_lr003_interface_20k200_s11/checkpoint_meta.json). |

**Corrección de estado DCSNN (2026-10-12):** la fila “DCSNN, largo en curso” de la tabla conserva el snapshot histórico del paso 6.400 y queda supersedida por el resultado completo. Terminó 20.000 Adam + 200 L-BFGS y fue rechazado: Tmax 20,132 °C, NRMSE 13,913%, error del aumento 99,562%, peor región 99,558% y balance global 70,617%. Solo pasó continuidad T de interfaz. [Resultado final](../results/phase7/p7_dcsnn_next_20261011/aras_flat/DCSNN_w32_d3_lr003_interface_20k200_s11/result.json).

### Síntesis y decisión

- **M1** es el mejor método disponible para producir resultados de `aras_flat`; seed 0 pasó todas las compuertas y seed 11 quedó cerca, con un único fallo regional. Usarlo como baseline, no afirmar todavía robustez multi-seed.
- **M0** sigue siendo control útil, pero su representación global no rindió en el cable con las búsquedas realizadas; no invertir más cómputo ahí sin una hipótesis nueva.
- **M2** no alcanzó M1 en el largo estándar; la variante 48×5 se detuvo tras screening desfavorable y no debe confundirse con una corrida completa.
- **CUSP** terminó largo y mantuvo T cerca del ambiente pese a bajar la pérdida; su veredicto es no-pass para esta formulación, no “faltaron unos pocos pasos”.
- **DCSNN** completó el largo y fue rechazado: Tmax 20,132 °C, NRMSE 13,913% y balance 70,617%. La caída de pérdida no se tradujo en calentamiento; no repetir la misma receta. El siguiente modelo de la secuencia es **FBPINN**, sujeto a smoke propio.

En todos los casos, la referencia FEM fue de evaluación únicamente. Las conclusiones son específicas al caso controlado `aras_flat` (fuente DC prescrita y frontera a 20 °C); no equivalen a reproducir el caso completo publicado de Aras.

## Actualización final: DCSNN (2026-10-12)

Esta actualización **reemplaza** el estado previo que describía la corrida DCSNN larga como “en curso”. Completó 20.000 pasos Adam + 200 L-BFGS en XPU, con DCSNN 32×3, `tanh`, `lr=0.003`, seed 11/11011, `sampling=interface` y puntos 512/256/128/128. Runtime: 6.378 s (1,77 h); train loss final 8,737 y validación final 8,739.

La auditoría FEM completa rechazó el modelo: Tmax 20,132 °C frente a 50,058 °C; error Tmax 29,926 K; NRMSE 13,913%; error del aumento 99,562%; peor RMSE regional 99,558%; balance global 70,617%. Solo pasó continuidad de temperatura en interfaces; fallaron Tmax, balance, flujo, NRMSE y RMSE regional. El smoke de 300 pasos ya anticipaba el campo casi ambiente; el largo no lo corrigió.

**Conclusión DCSNN:** no repetir la misma receta ni promover esta configuración para `aras_flat`. Investigar la representación compartida y el transporte térmico desde las fuentes: el residuo PDE cayó en las capas exteriores, pero se mantuvo alto en los núcleos de cobre y el balance por conductor quedó cerca de 23,6%.

**Siguiente arquitectura de la secuencia:** FBPINN. El smoke XPU de 100 pasos ya se ejecutó con la receta M1, `sampling=interface` y la configuración conservadora FBPINN 16×2; terminó con pérdidas explosivas y falló la evaluación, por lo que **no se autorizó corrida larga**. Antes de repetir, revisar la normalización de las ventanas/blending y las escalas del residuo. [Resultado smoke FBPINN](../results/phase7/p7_fb%20pinn_next_20261012/aras_flat/FBPINN_w16_d2_lr003_interface_pilot100_s11/result.json).

## Actualización FBPINN (2026-10-12)

Smoke XPU de 100 pasos, FBPINN 16×2, `tanh`, `lr=0.003`, seed 11/11011, puntos 512/256/128/128 y `sampling=interface`. Completó en 3.083 s (51,4 min), pero train loss final **11,81 millones**, validación **41,98 millones**, Tmax 20,601 °C frente a 50,058 °C, NRMSE **59,47%** y balance global **12.211%**. La pérdida es finita, pero la magnitud/divergencia, el campo casi ambiente y el rechazo de las compuertas hacen que no sea un smoke sano.

**Decisión:** no iniciar la corrida larga FBPINN con estos parámetros. Prioridad de diagnóstico: estabilidad del blending/ventanas locales, normalización de coordenadas y residuo, luego un smoke corto de CPU/XPU antes de asignar otra corrida de ~20.000 pasos. Este resultado no descarta toda la familia FBPINN, pero sí esta configuración/parametrización inicial. [Resultado](../results/phase7/p7_fb%20pinn_next_20261012/aras_flat/FBPINN_w16_d2_lr003_interface_pilot100_s11/result.json).

### Auditoría de colocación y ventanas FBPINN

Se revisó `collocation.npz` del smoke anterior, no solo la configuración declarada:

- 13 regiones con los conteos pedidos: 512 puntos en suelo y 256 en cada una de 12 capas. Los 3.584 puntos interiores pasan `domain.contains` en su región asignada.
- 12 interfaces con 128 puntos cada una; error máximo de ubicación respecto a la geometría de interfaz: `1.1e-16 m`. Las 4 fronteras tienen 128 puntos cada una.
- `sampling=interface` dejó entre 57% y 64% de los puntos de cada capa en la banda del 10% del espesor próxima a una de sus interfaces. Esto es concentración deliberada, no evidencia de puntos fuera de región.
- FBPINN construyó 40 cajas; todas reciben puntos: mínimo 164 y mediana 424 de los 3.584 interiores tenían peso de ventana >0.01. No hay cajas vacías ni soporte de ventana nulo.
- Las cajas más estrechas miden `0.05335 m` en x/y. La implementación añade cortes en centro ± radio **y en el centro del cable**; el centro no es una interfaz material y genera cajas pequeñas. Además `_window` usa `tau=0.02*ancho_de_caja`, mientras el input local también se normaliza por ese ancho; esto puede producir gradientes físicos muy grandes para una cabeza inicial de amplitud O(1). Es la hipótesis principal para la pérdida PDE inicial extrema; no es un problema de validez geométrica de los puntos.

### Contraste bibliográfico y siguiente prueba

La fuente primaria, Moseley, Markham y Nissen-Meyer, *Finite Basis Physics-Informed Neural Networks: a scalable domain decomposition approach for solving differential equations* (arXiv:2107.07871; DOI 10.1007/s10444-023-10065-9), define FBPINN con subdominios **solapados**, una ventana suave que se vuelve despreciable fuera de la zona de solape y normalización independiente por subdominio. El artículo estudia/ajusta anchuras de solape y tamaños de subdominio por problema; no prescribe cortar por el centro de un objeto ni usar universalmente `tau=2%` del ancho completo de caja. En algunos ejemplos también usa calendarios de entrenamiento activos por subdominio.

Por tanto, la colocación actual está bien contenida y suficientemente poblada; no conviene aumentar puntos a ciegas. Antes de otro smoke FBPINN, comparar en un preflight con la misma inicialización: (a) quitar cortes por centros de cable, conservando cortes físicos; (b) usar una banda de solape explícita ligada al radio exterior/espesor de material, en lugar de `tau` fijo relativo a la caja; (c) reducir amplitud inicial de las redes locales o usar salida unnormalizada por subdominio. Cambiar una sola decisión por brazo y exigir pérdida inicial finita razonable antes de XPU. Esta heurística aún no está implementada ni validada, y el artículo no respalda esos valores concretos; es una adaptación motivada por el diagnóstico local.

Referencia: [arXiv:2107.07871](https://arxiv.org/abs/2107.07871), secciones 4.1–4.3 y 5.2; [DOI](https://doi.org/10.1007/s10444-023-10065-9). Artefacto de puntos: [collocation.npz](../results/phase7/p7_fb%20pinn_next_20261012/aras_flat/FBPINN_w16_d2_lr003_interface_pilot100_s11/collocation.npz).

## DeepXDE: PFNN vs FNN y selección para corrida larga (2026-10-12)

Se compararon PFNN y FNN en condiciones pareadas: `aras_flat`, CPU, 12×3 `tanh`, `lr=0.003`, 400 pasos Adam, 0 L-BFGS, Hammersley, init seed 11 y sampling seed 11011. Ambos usaron 48 puntos interiores del suelo, 24 por región de cable, 24 por interfaz y 24 por segmento de frontera. Son 720 puntos por evaluación del objetivo. La referencia FEniCS nivel 2 se usó solo para evaluación. Se fijó `full_audit=false` para estos smokes: no hay métricas de balance ni saltos de interfaz y el estado de aceptación es `not_audited`.

| Modelo | Pérdida final | RMSE | NRMSE | Tmax predicha (ref.) | Error abs. Tmax | Error del aumento | Peor RMSE regional | Tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PFNN | 0,44267 | **3,937 K** | **13,098%** | 29,789 °C (50,058 °C) | 20,268 K | 67,431% | 22,771 K | 2.034 s (33,9 min) |
| FNN | 0,69761 | 3,993 K | 13,285% | 30,696 °C (50,058 °C) | **19,361 K** | **64,414%** | **21,254 K** | 1.511 s (25,2 min) |

Ambos smokes terminaron en el paso 400 con métricas finitas, pero ninguno aprendió suficientemente el calentamiento: Tmax quedó unos 19–20 K por debajo del FEM y NRMSE superó 13%. FNN quedó más cerca en Tmax y tuvo menor error del aumento y peor RMSE regional; PFNN obtuvo menor RMSE/NRMSE globales. Las diferencias globales son pequeñas y no constituyen un pase de las compuertas físicas.

**Selección provisional:** PFNN para el ensayo largo, dando prioridad al error global de campo FEM (RMSE/NRMSE); la pérdida menor es evidencia secundaria, no el criterio de selección. Esta decisión es exploratoria porque los smokes no tuvieron auditoría física completa. Se mantendrá la nube reducida del smoke para controlar el costo CPU y se ejecutará PFNN 12×3, Hammersley, seed 11/11011, 20.000 Adam + 100 L-BFGS, con auditoría completa. DeepXDE está limitado a CPU en el wrapper actual. Proyección lineal basada en el smoke PFNN: aproximadamente 29,7 h para los pasos Adam, antes del costo adicional de L-BFGS; es solo una estimación.

Artefactos: [PFNN smoke](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w12_d3_hammersley_pilot400_retry3_s11/result.json), [FNN smoke](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_FNN_w12_d3_hammersley_pilot400_s11/result.json).

### Dispositivo XPU, presupuesto de memoria e inferencia ligera en curso

La PC tiene PyTorch 2.9.0+XPU y una Intel Graphics (`torch.xpu.is_available() == true`); DeepXDE usa backend PyTorch. Se habilitó `device='xpu'` para DeepXDE y se comprobó con un test XPU automatizado y un PFNN real de 2 pasos: los parámetros quedaron en `xpu:0`, la evaluación terminó y se restauró el dispositivo global. `device='auto'` mantiene CPU para DeepXDE, sin alterar su comportamiento previo. Windows detecta Intel AI Boost, pero este entorno no tiene `torch.npu` ni `torch_npu`; NPU no está disponible para este entrenamiento DeepXDE.

La configuración completa de 5.632 puntos objetivo no deja suficiente margen de RAM con el footprint observado. Se eligió como escala intermedia PFNN 16×3 con 2.368 puntos: 192 de suelo, 96 por región de cable, 64 por interfaz y 64 por segmento de frontera. El preflight de 10 pasos en XPU completó y registró un pico de 4,842 GiB de memoria XPU; al paso 10, la pérdida era 106,48 y NRMSE 36,23%, cifras de preflight no convergidas. En el smoke de 400 pasos en curso el proceso ha usado aproximadamente 7,0 GiB de RAM y el host mantuvo aproximadamente 3,5–3,9 GiB disponibles. [Preflight XPU 10 pasos](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_pilot10_xpu_s11/result.json).

**Smoke PFNN XPU finalizado:** 400/400 Adam, mejor paso 400, pérdida 0,47890, 4.479 s (74,6 min), pico XPU 4,842 GiB y aproximadamente 7,0 GiB de RAM residente del worker. La evaluación final usa el mejor estado en memoria (guardado como `model.pt` al final): Tmax 30,179 °C vs 50,058 °C, error Tmax 19,879 K, RMSE 4,037 K, NRMSE 13,432%, error del aumento 66,135% y peor RMSE regional 21,260 K (`cable_0_layer_0`). No se calcularon balances ni saltos de interfaz (`not_audited`); no es criterio de aceptación.

La inferencia ligera previa del `model.pt` periódico (mtime 22:25:57) midió Tmax 24,520 °C, RMSE 4,102 K y NRMSE 13,648%; queda supersedida por la evaluación final del paso 400.

**Comparación pareada PFNN/FNN en XPU:** ambos usaron 16×3 `tanh`, 2.368 puntos (192/96/64/64), Hammersley, seed 11/11011, `lr=0.003`, 400 Adam y 0 L-BFGS. La auditoría física estuvo desactivada en ambos, así que no hay balance ni saltos de interfaz.

| Variante | Pérdida | RMSE | NRMSE | Tmax | Error Tmax | Error del aumento | Peor RMSE regional | Pico XPU | Tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PFNN | **0,47890** | **4,037 K** | **13,432%** | **30,179 °C** | **19,879 K** | **66,135%** | **21,260 K** | 4,842 GiB | 4.479 s (74,6 min) |
| FNN | 0,80865 | 4,239 K | 14,104% | 27,469 °C | 22,589 K | 75,152% | 23,253 K | **0,999 GiB** | 3.301 s (55,0 min) |

PFNN fue mejor en todas las métricas FEM de campo registradas, a costa de unos 3,84 GiB más de pico XPU. Las dos variantes subestiman mucho la temperatura; el smoke no demuestra convergencia ni aceptación. Por solicitud del usuario, se selecciona PFNN para la larga, manteniendo la nube intermedia para no agotar la memoria: XPU, 16×3, Hammersley, seed 11/11011, 20.000 Adam + 100 L-BFGS. La tasa del smoke de 400 pasos sugería ~62 h para Adam, pero fue demasiado optimista frente a la corrida real. [PFNN XPU](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_pilot400_xpu_s11/result.json), [FNN XPU pareado](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_FNN_w16_d3_hammersley_pilot400_xpu_s11/result.json).

**Corrida larga (2026-10-13):** PFNN 16×3 en `xpu:0`, 2.368 puntos, Hammersley, `lr=0.003`, seed 11/11011, 20.000 Adam + 100 L-BFGS; auditoría completa habilitada al finalizar. El proceso sigue activo, sin `result.json` ni `failure.json`. El `model.pt` se actualizó por primera vez a las 13:52:53, correspondiente al primer intervalo de log (paso 500); el paso no está escrito dentro del checkpoint. Inferencia ligera en CPU sobre ese snapshot estable: Tmax 33,470 °C vs 50,058 °C, RMSE 3,252 K, NRMSE 10,819%, error del aumento 55,185% y peor RMSE regional 18,205 K (`cable_1_layer_0`). No incluye balance ni saltos de interfaz. Reporte: [inferencia ligera del checkpoint](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_xpu_20k100_s11/checkpoint_eval_20261013T205736Z.json). Al corte de las 15:57 el checkpoint llevaba ~2 h sin cambiar y había ~2,2 GiB de RAM libre. Los primeros 500 pasos tomaron ~4 h 38 min; extrapolación simple: ~7,7 días para Adam, más L-BFGS y auditoría. ETA incierta, revisar al siguiente checkpoint. Directorio: [PFNN XPU 20k+100](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_xpu_20k100_s11/).

**Checkpoint siguiente (2026-10-13 16:15 local):** `model.pt` cambió y se leyó de forma estable; el paso no está codificado, pero por ser el siguiente intervalo de `log_every=500` corresponde nominalmente al checkpoint 1.000. Inferencia ligera: Tmax 35,456 °C vs 50,058 °C, error Tmax 14,602 K, RMSE 2,794 K, NRMSE 9,297%, error del aumento 48,579% y peor RMSE regional 17,014 K (`cable_1_layer_0`). Frente a la primera lectura, Tmax aumentó 1,986 K y NRMSE bajó 1,522 puntos porcentuales: el campo va acercándose al FEM. No hay balance ni saltos de interfaz; no es veredicto de aceptación. [JSON de inferencia](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_xpu_20k100_s11/checkpoint_eval_20261013T213643Z.json).

**Checkpoint siguiente (2026-10-13 18:25 local):** snapshot estable; nominalmente el intervalo 1.500, aunque el paso no está guardado en el checkpoint. Inferencia ligera: Tmax 36,277 °C vs 50,058 °C, error Tmax 13,781 K, RMSE 2,656 K, NRMSE 8,837%, error del aumento 45,848% y peor RMSE regional 16,359 K (`cable_1_layer_0`). Contra la lectura de las 16:15, Tmax aumentó 0,821 K y NRMSE bajó 0,460 puntos; continúa la mejora hacia el FEM. No incluye balance ni interfaz y no es aceptación. [JSON de inferencia](../results/phase7/p7_deepxde_next_20261012/aras_flat/DeepXDE_PFNN_w16_d3_hammersley_xpu_20k100_s11/checkpoint_eval_20261013T233241Z.json).
