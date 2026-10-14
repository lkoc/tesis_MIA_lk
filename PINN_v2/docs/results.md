# Resultados de la Fase 6

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