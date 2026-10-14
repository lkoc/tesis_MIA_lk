# Diagnóstico de la discrepancia de Tmax en PINN

Fecha: 2026-10-08. Caso: `aras_flat`. Este informe registra la campaña con PDE activa y separa resultados medidos de hipótesis. Hay siete corridas cerradas y una corrida DeepXDE-FNN interrumpida a solicitud antes de la evaluación final; su checkpoint parcial se conserva.

## 1. Protocolo y referencia

Se usa una referencia FEniCSx de nivel 2, con 264 969 grados de libertad, los mismos puntos de evaluación y la misma huella física que el caso PINN. La potencia prescrita es 55,81413 W/m; la integración FEM da 55,81410 W/m y el flujo exterior 55,80827 W/m, con error de balance de 0,01045 %. La referencia es numéricamente consistente para esta especificación.

La campaña fija semilla de pesos 11, semilla de muestreo 20272010, 800 pasos Adam, tasa 0,003, ancho 24, profundidad 3, `tanh`, los mismos puntos para todos los modelos y `pde=1`. No usa etiquetas FEM en la pérdida. DeepXDE se ejecuta en CPU; las demás arquitecturas seleccionan XPU cuando está disponible. Las compuertas (NRMSE 5 %, error de incremento de Tmax 5 %, balance 2 %, salto T 0,1 K y flujo de interfaz 1 %) son propuestas del proyecto, todavía no confirmadas formalmente.

Aunque el FEM es una referencia fiable para este problema matemático, varios parámetros físicos del caso están marcados como asumidos. El JSON declara una adaptación con 18,60471 W/m por cable, fuente uniforme y pérdidas DC a 20 °C; no incluye pérdidas AC por efecto piel/proximidad. El ejemplo heredado tiene sus propios parámetros y formulación (véanse [capas del ejemplo](../examples/aras_2005_154kv_flat/data/cable_layers.csv) y [parámetros de solver](../examples/aras_2005_154kv_flat/data/solver_params.csv)). Por tanto, el nombre `aras_flat` no implica reproducción completa del paper. Esa diferencia limita la comparación externa con literatura, pero no explica la brecha de esta campaña: PINN y FEM usan la misma especificación/hash.

## 2. Resultados de campaña

Las métricas provienen de auditorías independientes del campo predicho. `PDE RMS máx.` es el mayor residuo normalizado por región; `balance` es el error global derivado. Resultados cerrados al corte:

| Modelo | Error Tmax (K) | NRMSE (%) | Error de incremento (%) | PDE RMS máx. | Balance (%) | Salto T interfaz máx. (K) | Aceptado |
|---|---:|---:|---:|---:|---:|---:|---|
| M0 | 30,150 | 14,352 | 100,306 | 1,991 | 101,342 | 0,000 | No |
| M1 | 21,070 | 13,915 | 70,099 | 1,445 | 93,629 | 0,532 | No |
| M2 | 23,409 | 13,889 | 77,880 | 0,322 | 85,899 | 0,188 | No |
| CUSP | 31,152 | 15,972 | 103,640 | 1,852 | 90,143 | ~0 | No |
| DCSNN | 27,671 | 12,836 | 92,061 | 2,017 | 95,904 | 1,053 | No |
| FBPINN | 7,971 | 89,758 | 26,520 | 3 465,391 | 952,313 | 0,000 | No |
| DeepXDE-PFNN | NaN | NaN | NaN | 6,240 | 93,839 | 2,694 | No |
| DeepXDE-FNN | Sin evaluación | Sin evaluación | Sin evaluación | Sin evaluación | Sin evaluación | Sin evaluación | Interrumpido |

Las siete corridas evaluadas fallan las compuertas; FNN no tiene veredicto porque fue detenida antes de evaluar. Entre resultados finitos, `M2` tiene el menor residuo PDE máximo (0,322), mientras `DCSNN` tiene el menor NRMSE (12,836 %). Ninguno conserva bien la energía: el balance global de M2 aún yerra 85,9 %. `FBPINN` predice Tmax a 7,97 K del FEM, pero su NRMSE es 89,76 %, el residuo PDE máximo 3 465 y el balance 952 %; la proximidad de una sola métrica de hotspot no representa una solución física aceptable. DeepXDE-PFNN termina con pérdida de entrenamiento finita (0,589), pero la evaluación devuelve NaN en Tmax y en las métricas de error, así que esas métricas no se pueden interpretar como precisión.

La inspección de checkpoints muestra que los offsets de nivel aprendidos quedaron pequeños respecto del incremento FEM de ~30 K: M0 aporta `0,00593 × 50 K ≈ 0,30 K`; el mayor offset regional de M1 aporta ~8,95 K y el de M2 ~6,62 K. Esto es evidencia directa de que los niveles térmicos principales no se aprendieron en esos ajustes, pero no identifica por sí solo qué término o gradiente bloqueó la optimización. FNN conserva `model.pt`, pero sin métrica final.

### Smoke test y selección de receta

La corrección de escala DeepXDE tiene prueba unitaria del factor `T_scale` y compatibilidad de checkpoint. Se amplió el smoke para entrenar una iteración PFNN sobre `aras_flat`, recargarlo y exigir temperaturas finitas en las 13 regiones y en los centros exactos de los tres cables. La causa de los NaN quedó localizada: la transformación de características calculaba `log(0)` y `delta/radius` en el centro para regiones radiales inactivas; la máscara posterior no puede neutralizar `0 × NaN`. Se estabilizaron esos cocientes en [problem.py](../PINN_v2/pinn_v2/solvers/pinn/problem.py) y se añadió regresión a [test_deepxde_solver.py](../PINN_v2/tests/test_deepxde_solver.py). Pasaron 5 tests DeepXDE (112,54 s) y 154 pruebas de coordenadas/capas/variantes (99,49 s).

Se hizo además una corrida larga PFNN de 800 pasos Adam, mismo seed/configuración/puntos/referencia que la corrida anterior, en CPU. Duró 17 477 s (4,86 h), terminó con pérdida 0,2941 y checkpoint completamente finito. Antes de corregir el centro exacto, la evaluación seguía devolviendo NaN en conductores porque la malla incluye explícitamente los centros de cable. Para medir el efecto del offset de salida sin confundirlo con ese NaN, ambos checkpoints (PFNN anterior sin offsets regionales y nuevo PFNN con offsets) se reevaluaron después con la misma transformación estable, el mismo mallado FEM y auditoría completa. El baseline sin offset da `Tmax=20,07 °C`, error 29,99 K, NRMSE 14,16 % y balance 93,84 %; el entrenado con offset da `34,04 °C`, error 16,02 K, NRMSE 10,29 % y balance 56,01 %. Es una mejora pareada sustancial en las cuatro métricas, pero el nuevo checkpoint aún es rechazado: error de incremento 53,30 %, hotspot desplazado 0,33 m, salto máximo de T en interfaces 0,377 K y RMS PDE máximo 0,1794. La corrección sí mejora el resultado y elimina el NaN bajo evaluación estable; no lo hace físicamente aceptable.

Los registros originales de ambas corridas permanecen intactos en sus respectivos `result.json`. La comparación post hoc queda en [posthoc_evaluation_after_coordinate_fix.json](../PINN_v2/results/deepxde_offset_fix_20261008/DeepXDE_PFNN_seed11/posthoc_evaluation_after_coordinate_fix.json) y [posthoc_baseline_after_coordinate_fix.json](../PINN_v2/results/pde_active_validation_20261007/DeepXDE_PFNN/posthoc_baseline_after_coordinate_fix.json); el campo corregido está en [evaluation_after_coordinate_fix.npz](../PINN_v2/results/deepxde_offset_fix_20261008/DeepXDE_PFNN_seed11/evaluation_after_coordinate_fix.npz). Se reproduce con [evaluate_deepxde_offset_fix.py](../PINN_v2/evaluate_deepxde_offset_fix.py). El sampler de cable genera puntos en anillos con radio positivo y excluye el centro; el cambio de coordenadas afecta aquí al punto de evaluación central, no al conjunto de entrenamiento.

También existe una búsqueda histórica de 12 recetas M1: 400 pasos de smoke, promoción a 800 y finalistas de cinco semillas en los 11 casos. Para `aras_flat`, la comparación directamente pareada M1 base/trial 6 compartió `init=11`, `sampling=20272010`, hash físico, malla de puntos y 800 pasos Adam; no usó etiquetas FEM en la pérdida. Trial 6 cambió conjuntamente la receta a RBA, ancho 24/profundidad 4, `tanh`, `lr=0,00799`, peso constitutivo 1000, energía 10, temperatura de interfaz 10 y PDE 1. Con seed 11 redujo el error de Tmax de 21,07 a 12,74 K, NRMSE de 13,92 % a 7,64 % y desbalance global de 93,63 % a 28,51 %. Es una mejora reproducible frente al baseline en esa comparación, no una atribución causal a RBA o a un peso individual.

El resumen histórico eligió trial 10 por su score agregado, pero su `weights.pde=0`; por tanto, no es elegible bajo el requisito de mantener PDE activa y no se recomienda como PINN física. Reordenando solo finalistas con `weights.pde>0`, trial 6 es el mejor candidato evaluado. Aun así, falla claramente aceptación: sus cinco semillas en `aras_flat` alcanzaron NRMSE medio 6,30 % (máximo 8,37 %) y error máximo del incremento de Tmax 48,31 %; seed 11 conserva un desbalance de 28,51 %. Se mantiene como receta provisional para el siguiente ciclo, no como solución ni nuevo default validado.

La corrida FBPINN tardó 23 668 s (6,57 h) y DeepXDE-PFNN 18 849 s (5,24 h) en esta configuración de una semilla. Son tiempos de esta implementación/caso, no una comparación general de velocidad entre familias.

### Reproducción M1 en el código actual

La receta Phase 6 fue reproducida en un directorio nuevo con el source tree actual, misma definición y referencia FEM L2, M1 width 32/depth 3/tanh, `lr=0,003`, 20 000 Adam + 200 L-BFGS, puntos 512/256/128/128, seed init 0/sampling 11 y XPU. Terminó en 19 057 s (5,29 h), con estado `completed`, `evaluation_finite=true`, hash físico coincidente y todas las compuertas propuestas aceptadas. Métricas finales: Tmax 50,664 °C vs 50,058 °C (error 0,606 K; error de incremento 2,02 %), NRMSE 0,511 %, worst-region RMSE-rise 3,61 %, balance global 0,119 %, salto máximo de T 0,0744 K y RMS máximo de flujo 0,0330 %. El hotspot coincide con el FEM.

Esto confirma que la receta M1 larga funciona en el código actual para seed 0; corrige el estado anterior de “solo existe un checkpoint histórico”. Todavía no es una declaración robusta: es una semilla. El siguiente paso es repetir la misma receta con seeds 11 y 23 y resumir peor caso/media antes de tocar hiperparámetros. Esta ejecución empezó antes de que se agregara `checkpoint_meta.json`, por lo que no contiene ese sidecar; sí tiene manifest/result/checkpoint/evaluación/collocation y finitud verificada.

El error no es solo un desplazamiento del hotspot. Para M2, el hotspot predicho está a 1,37 mm del FEM, mientras el error absoluto de campo tiene mediana de 21,38 K y percentil 95 de 23,41 K: es principalmente un sesgo térmico amplio y frío. Predice Tmax 26,65 °C frente a 50,06 °C. En M1, el RMSE es 4,18 K, pero el error de Tmax todavía es 21,07 K: las métricas deben leerse junto con la región crítica y el balance.

Como evidencia auxiliar, un ensayo M1 previo de optimización de hiperparámetros también usó `pde=1`, pero empleó RBA, ancho 24/profundidad 4, tasa aproximada 0,00799, peso constitutivo 1000 y energía 10. En la comparación pareada descrita arriba obtuvo error Tmax 12,74 K, mejor que el M1 uniforme actual, aunque siguió fuera de aceptación. La comparación controla semilla, muestreo y pasos, pero cambia varios hiperparámetros juntos; por eso no aísla el efecto de cada mejora ni demuestra que PDE positivo por sí solo baste.

## 3. Diagnóstico causal

**Conclusión con confianza moderada:** la principal causa común es que esta receta de entrenamiento no llegó a los niveles de temperatura requeridos y no satisfizo conservación/transmisión: 800 pasos Adam, pesos fijos y una representación global/local que debe resolver capas finas y saltos grandes de conductividad. Hay evidencia de underfit en los offsets guardados y fallos de balance; todavía no hay medición de normas/cosenos de gradientes para afirmar que un peso concreto domina. No es evidencia de incapacidad intrínseca de las PINN. En DeepXDE, el offset regional sí mejora de forma pareada Tmax, NRMSE y balance; el segundo fix elimina la singularidad en centros. Aun combinados, el error Tmax 16,02 K y balance 56,01 % dejan el modelo muy lejos de los umbrales propuestos.

- **Referencia/evaluación:** el hash físico y los puntos de evaluación coinciden; la solución FEM cierra su balance mucho mejor que cualquiera de estas PINN. Un error de mapeo FEM/PINN o de selección del hotspot es una explicación poco probable.
- **Normalización y singularidades:** M0/M1/M2 y las variantes propias incluyen niveles entrenables escalados por `T_scale`; no aparece una conversión común Celsius/Kelvin errónea. En DeepXDE se corrigió la fórmula previa `T = T_ref + amplitude_region * raw`, que omitía `T_scale * offset_region`; para el conductor, `qscale * length / k` da ~0,0074 K. La corrida posterior confirma que el offset alcanza ~14 K, todavía por debajo del incremento FEM de ~30 K. También se halló que `CableProblem.coordinates` calculaba características de materiales inactivos antes de enmascararlos: en los centros, `log(0)` y `delta/0` introducían NaN. Se estabilizaron los radios con un mínimo dependiente del dtype en [problem.py](../PINN_v2/pinn_v2/solvers/pinn/problem.py). El test de centros y el smoke PFNN en centros reales pasan; la evaluación post hoc del checkpoint queda finita, pero su campo sigue sesgado en frío.
- **Pesos y entrenamiento:** la receta base usa multiplicadores de 100 para Dirichlet/interfaz-T, 10 para interfaz-flujo y 1 para PDE/energía. La sonda sobre un checkpoint M1 encontró normas ponderadas `PDE=5,01`, `energy=8,26`, `interface_flux=14,58`, `Dirichlet=6,49` e `interface_T=5,02`; los cosenos de gradiente PDE–flujo y energía–flujo fueron −0,886 y −0,982. Esto demuestra competencia local de objetivos en ese checkpoint, no que sea la única causa del error ni que se mantenga toda la trayectoria. La sonda trial 6 no reproduce RBA; ampliar a seeds/M0/M2 antes de diseñar balance adaptativo. M2 tiene PDE RMS máx. 0,322, pero balance 85,9 %; un residuo PDE bajo en puntos auditados no implica conservación global.
- **Representación:** todas las familias comparten el fallo, lo que hace menos probable que una sola arquitectura explique la discrepancia. Las capas con alto contraste y el acoplamiento de muchas regiones pueden dificultar el problema, pero los ensayos previos con trazas de interfaz y solución lineal por SVD alcanzaron errores de Tmax de 0,035–0,103 K en otro caso XLPE. Eso refuta una imposibilidad general, no garantiza transferencia a `aras_flat`.
- **FBPINN:** la versión del proyecto es una adaptación con ventanas sigmoidales y coordenadas locales, no una reproducción exacta del FBPINN original. Para este caso crea unas 40 cajas cartesianas a partir de los cortes geométricos y evalúa todas las redes de caja dentro de cada llamada a `field`. La corrida duró 6,57 h y terminó con PDE RMS máximo 3 465 y balance 952 %; su máximo térmico queda relativamente cerca, pero el error espacial y los balances muestran que no aprendió una solución física adecuada. El alto coste es medido; atribuirlo en concreto al bucle serial y a la auditoría común (2 048 puntos interiores por región de suelo, 1 024 por material y 512 por cara/interfaz y contorno) sigue siendo una hipótesis basada en el volumen de llamadas, no en un perfilado. Un resultado con 800 pasos y sin la estrategia completa de solapamiento/agenda no debe interpretarse como veredicto sobre la familia.

## 4. Acciones recomendadas

1. **Consolidar M1 antes de volver a buscar.** La receta larga M1 del Phase 6 acaba de reproducirse con éxito en el source tree actual para seed 0. Ejecutar seeds 11 y 23 con idéntico protocolo y resumir peor seed/media; no abrir otra búsqueda amplia mientras esa verificación esté en curso. Trial 6 queda como antecedente histórico, no reemplaza esta receta reproducida.
2. **DeepXDE: conservar el offset y cambiar la receta de entrenamiento.** La comparación pareada muestra que el offset sí mejora el resultado, pero no cierra el error físico. Probar una sola mejora cada vez con PDE activa (p. ej. inicializar/optimizar el nivel regional, balancear pérdidas, muestreo y más etapas Adam/L-BFGS), guardar métricas FEM y auditoría completas, y reentrenar FNN por separado. El trial M1-6 no se transfiere automáticamente a DeepXDE.
3. **Completar el diagnóstico de gradientes.** La sonda M1 ya detecta gradientes en conflicto en un checkpoint; extenderla a M0/M2, varias semillas y checkpoints tempranos/tardíos, guardando también los términos por región. Mantener la solución FEM como control del operador/signo/unidades. Recién entonces elegir balance adaptativo o curricula, sin retocar pesos por intuición.
4. **Rehacer el entrenamiento con presupuesto y criterio adecuados.** Después de medir gradientes, extender Adam con decaimiento, varias semillas y un criterio de parada por validación física; probar L-BFGS solo con una pérdida fija. Comparar por tiempo y métricas físicas, no por pérdida final o pasos.
5. **Endurecer las restricciones donde ayuden.** Probar Dirichlet duro y trazas compartidas exactas de T/flujo para sacar continuidad del objetivo aprendido; validar siempre balance y PDE, porque continuidad exacta no garantiza la solución.
6. **Aprovechar linealidad y cambiar la representación.** Con fuentes/conductividades prescritas, probar ajuste escalado de salidas/trazas por QR/SVD, y refinamiento radial/local de suelo. El SVD local fue preciso en otro caso, pero aún debe validarse en `aras_flat`.
7. **No descartar FBPINN por esta adaptación.** Reproducir la agenda/normalización local del repositorio original o etiquetar el resultado como adaptación propia; comparar con mismo tiempo y métricas.

Las tolerancias de aceptación siguen siendo propuestas; cualquier informe final debe mostrar los valores y no rebautizar como “aceptado” un modelo por destacar una sola métrica.

## 5. Literatura y proyectos consultados

Fuentes principales usadas para el diagnóstico y sus decisiones metodológicas:

- Raissi, Perdikaris y Karniadakis, *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*: https://arxiv.org/abs/1711.10561
- Wang, Teng y Perdikaris, *Understanding and mitigating gradient pathologies in physics-informed neural networks*: https://arxiv.org/abs/2001.04536
- Krishnapriyan et al., *Characterizing possible failure modes in physics-informed neural networks*: https://arxiv.org/abs/2109.01050
- Moseley, Markham y Nissen-Meyer, *Finite Basis Physics-Informed Neural Networks (FBPINNs): a scalable domain decomposition approach for solving differential equations*: https://arxiv.org/abs/2107.07871 ; publicación: https://doi.org/10.1007/s10444-023-10065-9
- Lu et al., *DeepXDE: A deep learning library for solving differential equations*: https://arxiv.org/abs/1907.04502 ; publicación: https://doi.org/10.1137/19M1274067
- Hao et al., *PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs*: https://arxiv.org/abs/2306.08827

Repositorios contrastados con las variantes y alcance del proyecto:

- FBPINNs original: https://github.com/benmoseley/FBPINNs
- DeepXDE: https://github.com/lululxvi/DeepXDE
- Benchmark PINNacle: https://github.com/i207M/PINNacle
- PINA: https://github.com/PINA-org/PINA
- Gradiente patológico en PINN: https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs

Revisión técnica al 2026-10-08:

| Proyecto | Qué aporta según su README/código | Recomendación para este caso |
|---|---|---|
| [DeepXDE](https://github.com/lululxvi/DeepXDE) | BC duras aproximadas, varios muestreadores, Fourier features, Adam/L-BFGS y varias autodiferenciaciones; framework general de SciML. | Mantenerlo como backend/comparador, ahora con el offset corregido; probar hard Dirichlet y resampling adaptativo por separado. No suponer que el framework arregla un adaptador mal condicionado. |
| [FBPINNs](https://github.com/benmoseley/FBPINNs) | Redes en subdominios solapados, normalización por subdominio y agenda; implementación actual JAX/vmap; rama PyTorch para reproducir el paper. | Usar como referencia para reimplementar fielmente la variante. La versión actual de este proyecto usa ~40 cajas cartesianas, ventanas sigmoidales y llamadas seriales, así que no es una comparación directa. |
| [PINNacle](https://github.com/i207M/PINNacle) | Benchmark de 20 problemas y controles de reponderación, Adam+L-BFGS, muestreo residual, MultiAdam, gPINN, hp-VPINN y FBPINN. | Reutilizar su diseño de ablaciones/controles; sus PDE no son este cable y sus cifras no se transfieren como desempeño esperado. |
| [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) | Framework PyTorch amplio con modelos/recetas de surrogate, operadores, meshes y workflows científicos; el repo actual no es un solver PINN clásico de cable listo para usar. | No migrar como arreglo inmediato. Evaluarlo solo si la tesis evoluciona a surrogate FEM/datos o entrenamiento distribuido y hay caso de uso que lo justifique. |

### Estructura de red y tamaño

El [paper Grossmann et al. (2023)](https://arxiv.org/abs/2302.04107) y su [código oficial 2D Poisson](https://github.com/TamaraGrossmann/FEM-vs-PINNs/blob/main/PINN_codes/2D_Poisson.py) usan MLPs fully connected con `tanh`, optimizador Adam y refinamiento L-BFGS. Para 2D Poisson exploran de 1 a 5 capas ocultas, anchos 20/60 y una red más grande de ancho 120; usan 20 000 épocas Adam, LHS remuestreado por época y L-BFGS. Para Allen–Cahn 1D con interfaz difusa comparan anchos 20/100/500 y 3–7 capas; los de ancho 20 no recuperan la solución, mientras 100/500 mejoran. Ese caso también emplea 7 000 épocas de preentrenamiento de la condición inicial, luego 50 000 sobre la pérdida completa y L-BFGS. En cambio, para Poisson 1D no encuentran que más capas mejore sistemáticamente los resultados. El paper, por tanto, respalda que capacidad importa **según el PDE**, no una regla universal de “más capas siempre mejor”. Es un estudio vanilla sobre Poisson, Allen–Cahn y Schrödinger, no una receta directa para una sección de cable con 13 materiales.

| Red evaluada en `aras_flat` | Estructura/capacidad | Error Tmax | NRMSE | Balance global |
|---|---|---:|---:|---:|
| M1 base | 13 redes locales, ancho 24 × 3 capas ocultas; 17 162 parámetros | 21,07 K | 13,92 % | 93,63 % |
| M1 trial 6 | 13 redes locales, ancho 24 × 4; 24 962 parámetros; RBA y varios pesos también cambiaron, con PDE=1 | 12,74 K | 7,64 % | 28,51 % |
| DeepXDE PFNN post-offset | 13 ramas, ancho 24 × 3; entrada codificada de 65 características y 36 530 parámetros | 16,02 K | 10,29 % | 56,01 % |

El PFNN DeepXDE fallido tiene más parámetros que ambos M1s; por eso la evidencia actual **no apoya que el problema sea simplemente “faltan nodos”**. En la comparación DeepXDE pareada, agregar solo 13 offsets regionales mejoró Tmax `20,07 → 34,04 °C` sin cambiar ancho/profundidad: esa mejora grande con capacidad prácticamente igual apunta a la parametrización y al nivel de salida como factores importantes. La entrada DeepXDE tiene 65 valores (bloques de características locales, uno activo según región) y ramas por región; M1 entrega a cada subred directamente sus coordenadas locales de 2–5 componentes. Para interfaces radiales finas, M1 es una representación más explícitamente local; eso no la hace automáticamente correcta, como demuestra que aún incumple los gates.

Se ejecutó la ablación M1 que separa ancho y profundidad, con `PDE=1`, todos los demás pesos/configuración iguales al baseline, misma semilla de inicialización y muestreo y 800 pasos:

| Ablación | Parámetros | Error Tmax | NRMSE | Balance | Decisión |
|---|---:|---:|---:|---:|---|
| 24×3 baseline | 17 162 | 21,07 K | 13,92 % | 93,63 % | Rechazada |
| 48×3, solo más ancho | 64 250 | 20,54 K | 13,39 % | 73,68 % | Rechazada |
| 24×5, solo más profunda | 32 762 | 18,52 K | 12,16 % | 78,45 % | Rechazada |

Más ancho casi no cambia Tmax/NRMSE; más profundidad aporta una mejora modesta, pero ninguna variante se acerca a los gates. Trial 6 consigue un salto mayor con menos parámetros que 48×3, aunque combina RBA, LR, profundidad y pesos; eso hace plausible que entrenamiento/reponderación importen más que tamaño bruto, pero no permite atribuir cuánto a cada uno.

La sonda de gradientes sobre el checkpoint M1 base, con los mismos puntos y pesos, mide normas ponderadas PDE 5,01, energía 8,26, flujo de interfaz 14,58, Dirichlet 6,49 e interfaz-T 5,02. Los cosenos no ponderados de flujo con PDE y energía son −0,886 y −0,982; interfaz-T con Dirichlet es −0,922. Es evidencia concreta de conflicto de optimización en ese checkpoint. El probe de trial 6 muestra valores ponderados PDE 1,38, energía 3,80 y flujo 8,80, con cosenos PDE–flujo −0,890 y energía–flujo −0,869. **Límite:** son gradientes de una sola muestra/checkpoint con puntos fijos; la sonda no reproduce el reweighting RBA dinámico de trial 6 ni la trayectoria durante Adam, así que diagnostica conflicto local, no prueba por sí sola la causa histórica.

El PFNN DeepXDE se entrenó 800 pasos Adam, sin L-BFGS ni remuestreo por época; el paper compara 20 000–50 000 pasos/épocas y L-BFGS, con pérdidas más simples y otras escalas. Con una semilla por configuración no podemos concluir causalidad de ninguna diferencia. El veredicto más defendible es **capacidad mayor ayuda algo, pero no basta; parametrización de salida y optimización tienen evidencia más directa como cuellos de botella**. No hay evidencia para afirmar que la arquitectura actual sea universalmente suficiente o fundamentalmente incapaz.

Los resultados y gradientes de esta sección se generaron con [diagnose_m1_architecture.py](../PINN_v2/diagnose_m1_architecture.py); datos por ensayo en [capacity_sweep.json](../PINN_v2/results/m1_architecture_diagnostics_20261008/capacity_sweep.json), [baseline_gradient_probe.json](../PINN_v2/results/m1_architecture_diagnostics_20261008/baseline_gradient_probe.json) y [trial6_gradient_probe.json](../PINN_v2/results/m1_architecture_diagnostics_20261008/trial6_gradient_probe.json). Antes de elegir un tamaño por defecto, repetir al menos baseline y el mejor candidato con varias semillas; la evaluación actual es una semilla y las compuertas siguen rechazando todos los modelos.

La página [“Top 10 PINN Frameworks” de DevOpsSchool](https://www.devopsschool.com/blog/top-10-physics-informed-neural-network-pinn-frameworks-features-pros-cons-comparison/) sirve como mapa inicial de nombres, no como evidencia para seleccionar arquitectura: no ofrece protocolo reproducible ni trazabilidad clara de sus puntuaciones, mezcla frameworks con bibliotecas y workflows (por ejemplo, OpenFOAM+PINN), y cuenta Modulus/PhysicsNeMo como entradas separadas. La elección anterior se apoya en los repositorios primarios y papers enlazados, no en su ranking.

Copias locales útiles ya presentes: [Raissi et al. (2019)](../Estado%20del%20arte/zotero_Physics_informed_neural_networks_A_deep_learning_framework_for_solving_forward_a_10.1016_j.jcp.2018.10.045.pdf), [Wang et al. (2020), gradientes](../benchmark_papers/Wang_2020_gradientes.pdf), [muestreo adaptativo (Wu et al.)](../benchmark_papers/Wu_2022_sampling.pdf) y [Mishra & Molinaro (2020)](../benchmark_papers/Mishra_Molinaro_2020_error.pdf). Se registran URLs para Krishnapriyan et al., el Expert’s Guide y PINNacle arriba. El duplicado `Raissi_2019_JCP_verificado.pdf` en el manifiesto de fuentes no es un PDF válido; usar la copia Zotero indicada.

Hallazgos aplicados al proyecto:

- **FBPINNs:** subdominios solapados, normalización local y agendas; el repositorio principal usa JAX/vmap y mantiene una rama PyTorch para reproducir el artículo. El ejemplo base usa 20 000 pasos, frente a 800 aquí. El modelo local es una adaptación PyTorch con ventanas sigmoidales y sin esa agenda: su error/coste no juzga la familia original.
- **DeepXDE:** proporciona autodiferenciación, muestreadores, arquitecturas y optimizadores combinables. El adaptador local comparte un campo por región mediante selector/transformación de características y usa el mismo objetivo físico, pero CPU y 800 pasos. Compararlo con M0–M2 evalúa esa parametrización/backend, no un “solver DeepXDE” independiente ni una corrección automática del balance de gradientes.
- **PINNacle:** implementa comparaciones de reponderación, muestreo residual, MultiAdam, gPINN, hp-VPINN y FBPINN. El proyecto actual cubre solo una parte de ese espacio; sirve para seleccionar controles posteriores, no para transferir resultados de benchmark al cable.
- **PINA:** ofrece organización modular de problemas, condiciones, pérdidas y solvers. Es una referencia de arquitectura de software si crecen las variantes, no una solución numérica probada para este caso.
- **GradientPathologiesPINNs:** acompaña el trabajo de Wang et al. sobre rigidez y gradientes desbalanceados. Justifica medir gradientes por término antes de cambiar pesos; no es evidencia de que el mecanismo concreto ya se haya verificado aquí.

La adaptación FBPINN de este caso crea unas 40 cajas cartesianas a partir de cortes geométricos y evalúa todas las redes de caja en cada llamada a `field`; ese bucle serial es un posible contribuyente al coste observado, todavía sin perfilado formal.

El código local y los antecedentes experimentales están en [investigación multiescala](INVESTIGACION_PINN_MULTIESCALA_2026-09-15.md), [formulaciones y selección](teoria_mejoras_pinn.md) y las auditorías y resultados archivados bajo `PINN_v2/results/pde_active_validation_20261007/`. Los enlaces se registran aquí; no se duplican descargas de PDF ya archivadas en la carpeta de literatura.

## 6. Estado de cierre

La campaña original queda cerrada administrativamente con siete evaluaciones completas y una FNN interrumpida antes de evaluación; ocho entradas están registradas en `validation_summary.json`, y su checkpoint parcial se conserva. La M1 larga del Phase 6 fue reproducida con éxito en source actual para seed 0 (`result.json`: evaluation finita y todas las gates verdaderas). Queda pendiente replicarla en seeds 11/23 antes de afirmar robustez. La comparación de capacidad M1 (24×3, 48×3, 24×5) y las sondas de gradientes quedan bajo `results/m1_architecture_diagnostics_20261008/`; ambos tamaños mayores mejoran algunas métricas, pero siguen rechazados, con efecto de ancho pequeño y de profundidad moderado. Junto con los gradientes conflictivos, la evidencia favorece optimización/condicionamiento como cuello principal sobre una mera falta de parámetros, sin descartar interacciones ni efectos de otras semillas. DeepXDE post-offset sigue rechazado (Tmax error 16,02 K, NRMSE 10,29 %, balance 56,01 %); M1 es la línea principal de recuperación.