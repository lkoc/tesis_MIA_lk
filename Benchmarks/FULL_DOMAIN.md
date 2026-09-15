# Formulación activa: calor 2D en todo el cable

Revisión metodológica iniciada el 14 de septiembre de 2026 y ampliada a la campaña A–D y corriente límite. El [dictamen integral](../docs/REVISION_INTEGRAL_TESIS.md) y los JSON de `explicit_study/report` identifican el alcance de aceptación obtenido. Este documento sustituye las instrucciones operativas de cables de los informes anteriores. La evidencia histórica se conserva para reproducción, pero no valida la nueva física.

## Decisión física

Cada conductor, aislamiento, pantalla, cubierta y región de suelo tiene su propio dominio. Todas las variantes activas imponen

\[
\rho c_p\partial_t T+\nabla\cdot\mathbf q=Q,\qquad \mathbf q=-k\nabla T.
\]

En esta campaña estacionaria el término temporal es cero. La temperatura del conductor sale del campo calculado dentro del conductor; no se reconstruye con una resistencia térmica. Las interfaces exigen continuidad de temperatura y flujo normal. No se impone un flujo uniforme en la superficie del cable.

En anillos se usan coordenadas locales logradiales y seno/coseno angular, conservando la difusión angular completa mediante diferenciación respecto de las coordenadas físicas. El centro del conductor usa coordenadas cartesianas regulares. No se presupone simetría radial en los cables enterrados.

`full_physics.py` también contiene los operadores temporales cartesiano y polar verificados con soluciones manufacturadas. Esto prepara la extensión futura; todavía no constituye un simulador transitorio. Se necesitan capacidades volumétricas documentadas, condiciones iniciales, cargas temporales y una campaña de convergencia temporal.

## Qué se compara

| Arquitectura efectiva | Nombres admitidos | Interpretación |
|---|---|---|
| Red global suave de T | `global`, `direct` | Control de limitación arquitectónica ante saltos de k |
| Redes de T por material | `subdomain`, `enriched`, `conservative` | Coordenadas locales, interfaces y balance en todas ellas |
| Redes de T con coordenadas cartesianas locales | `local` | Ablación de la transformación logradial |
| Redes de T con armónicos | `multipole` | Armónicos regulares dentro del cable; no solución térmica impuesta |
| Redes de T y flujo por material | `mixed` | Conservación y ley constitutiva equivalentes a la ecuación de calor |
| Redes de T con Fourier | `fourier` | Ablación de representación de las entradas |

Los alias de una misma fila **no son métodos independientes** ni deben contarse como nuevos experimentos. Todos comparten el mismo residuo interior y las mismas interfaces. La candidata principal inicial es `subdomain`; `mixed` es la alternativa para reducir el orden de derivación. La variante mixta también verifica el flujo derivado de T, evitando aceptar flujos predichos que no satisfagan Fourier. Véase [FO-PINNs](https://arxiv.org/abs/2210.14320) para la formulación equivalente de primer orden; la implementación no se presenta como una reproducción exacta de ese artículo.

Una única red global C1 tiene la misma derivada normal a ambos lados de una interfaz. Si k cambia y el flujo normal es distinto de cero, la solución física necesita derivadas normales distintas. Por tanto, `global/direct` no pueden satisfacer exactamente esa transmisión con esta representación: sirven como control negativo, no como candidata a aprobar por simple aumento de puntos. Una alternativa de pesos compartidos necesitaría entradas de material y trazas diferenciadas, o enriquecimiento no suave controlado; eso constituye otra arquitectura que debe declararse. Esta incompatibilidad refuerza la elección de redes por material.

## Pérdidas eléctricas y efecto piel

El caso JSON puede incluir:

```json
"electrical": {
  "resistance_basis": "dc",
  "resistance_ohm_m": 0.000193,
  "reference_temperature_C": 20.0,
  "source_temperature_C": 20.0,
  "frequency_Hz": 60.0,
  "skin_model": "solid_round",
  "profile": "skin",
  "mu_r": 1.0,
  "provenance": "Identificar ficha, página, construcción y temperatura del dato"
}
```

`resistance_basis=dc` con frecuencia positiva exige corrección documentada. `resistance_basis=ac` exige frecuencia y prohíbe corregir otra vez por piel. Se usa la resistencia AC a su temperatura de referencia: no se multiplica automáticamente por el coeficiente DC de temperatura. Una resistencia medida AC no identifica por sí sola el perfil de corriente. Su perfil uniforme constituye una hipótesis de fuente que requiere sensibilidad; no simplifica la ecuación térmica interior.

`solid_round` calcula la solución electromagnética cilíndrica con funciones de Bessel y fasores RMS, y suministra el calor volumétrico `Q(r)=rho_e*|J(r)|²`. La integral es `I² Rac`. Es una estimación para conductor circular macizo homogéneo aislado; trenzado, segmentación, proximidad, pantallas y armaduras necesitan los datos/modelos correspondientes. [Patel, Gustavsen y Triverio](https://arxiv.org/abs/1303.5452) muestran el tratamiento conjunto de piel y proximidad; [COMSOL](https://www.comsol.com/support/learning-center/article/81171) explica la relación entre profundidad de penetración, tamaño y resolución de pérdidas.

Sin bloque `electrical`, el catálogo histórico se interpreta explícitamente como DC a 20 °C y frecuencia cero. Esto reproduce su hipótesis; **no certifica que los datos originales sean DC**. No se ha recalificado automáticamente un caso bibliográfico como caso AC real.

Dos modos de fuente:

- `fixed`: fuente volumétrica prescrita, DC o AC, a temperatura eléctrica declarada. El perfil piel puede usarse en todas las variantes.
- `dc_temperature`: conductividad eléctrica local dependiente de T, `Ez=I/integral(sigma(T)dA)` y `Q=sigma(T)Ez²`, resuelta dentro de cada conductor. No usa temperatura media para reconstruir la conducción térmica. El acoplamiento AC con temperatura no uniforme necesita resolver el problema electromagnético adicional y aún no está implementado.

## Estimación cuantitativa a 60 Hz

Resultados condicionales reproducibles mediante `python Benchmarks/skin_study.py`, conservados en `docs/auditoria/metodologia_2026-09-14/skin_effect.json`. Se supone que R20 es DC y se mantienen las corrientes del catálogo.

| Caso | Radio, mm | Aumento de pérdida a 20 °C | Aumento a 90 °C | Cambio de Tmax por redistribuir la misma pérdida, 20 °C |
|---|---:|---:|---:|---:|
| XLPE | 5,50 | 0,317 % | 0,195 % | −0,0000118 K |
| Aras | 18,85 | 37,136 % | 25,578 % | −0,004045 K |
| Kim | 21,20 | 37,136 % | 25,578 % | −0,001551 K |

La última columna se obtiene integrando la ecuación de Poisson **solo como diagnóstico externo**, con temperatura superficial fija, simetría radial, k del conductor y el mismo calor total. No es un error demostrado para el arreglo 2D completo ni para transitorios. La escala máxima de elevación centro-superficie uniforme `P/(4 pi k)` es 0,0028–0,0132 K en estos cálculos. Por ello la redistribución radial parece secundaria frente al umbral térmico de ingeniería de la tesis; el aumento de potencia de los conductores grandes no puede descartarse. No confundir 37 % de pérdida con 37 % de temperatura absoluta ni con 37 % de reducción de ampacidad.

Decisión: conservar el perfil piel disponible, comparar fuente uniforme y fuente piel **con igual potencia** para aislar su efecto térmico, y comparar después DC frente a AC para aislar el aumento de pérdida. Para declarar despreciable el perfil en un caso 2D, exigir cambio de Tmax inferior a 0,1 K y al 10 % del presupuesto térmico, con referencia refinada. Estos son criterios adoptados por la tesis, no umbrales universales de la bibliografía.

Comprobación posterior: `skin_fem_study.py`, ejecutado en FEniCSx, resolvió ambas fuentes en el dominio 2D Aras con cuatro mallas. La pérdida total se igualó mediante integración. Los cambios de Tmax fueron −0,004132 / −0,004051 / −0,004045 / −0,004045 K. La última malla tiene 380113 grados de libertad y balance de 0,002402 %. La variación de la diferencia entre las dos últimas mallas es aproximadamente 0,00000041 K. Esta comparación respalda la pequeña sensibilidad térmica del perfil para ese caso estacionario y ese modelo eléctrico. La evidencia está en `explicit_results/skin_aras_60Hz/equal_power/comparison.json`.

## Ejecución y datos

En Windows, Python para PINN y FEniCSx en WSL:

```powershell
python Benchmarks/run_case.py coaxial_full --mode verification --seeds 11 23 37 --output Benchmarks/explicit_results/reproduction/coaxial
python Benchmarks/run_case.py xlpe_single --mode nominal --variant subdomain --output Benchmarks/explicit_results/reproduction/xlpe
python Benchmarks/full_pinn.py --case-file caso_ac.json --mode fixed --variant mixed --reference referencia/fem_l2.npz --output Benchmarks/explicit_results/reproduction/ac
```

Usar un directorio nuevo por corrida. El flujo guarda caso, configuración, identidad SHA256 de física y fuentes, mallas/tags FEM, puntos, pesos de evaluación, T en cada material, semilla, tiempos, historial y estados de optimizadores. Los checkpoints completos permiten inspección y reproducción; no se ofrece todavía reanudación automática a mitad de una optimización.

`full_fem.py` usa elementos P2 y geometría curva con cada capa etiquetada. Las evaluaciones se asignan al material correcto antes de interpolar; usar la primera celda candidata en una interfaz curva produce errores espurios. El refinamiento exige al menos tres niveles: cambio de Tmax ≤ min(0,1 K; 0,5 % de elevación), cambio RMSE por región ≤0,5 % de elevación y balance ≤0,2 %. Una referencia reducida o con distinto caso/fuente es rechazada por identidad física.

El directorio nuevo `explicit_results` tiene su propio contrato: los NPZ contienen `xy`, `region`, `weights`, `T`; FEM añade grados de libertad y XDMF/H5. Los informes anteriores no deben consumir estos archivos como si fueran resultados de suelo perforado.

## Muestreo, verificación y cómputo

La entrada `n` cuenta puntos **por región de suelo**, `n_layer` por material de cable, `n_interface` por interfaz. En suelo se usa una mezcla mitad global y mitad cercana a cables; todas las capas tienen presupuesto propio, aunque su área sea pequeña. La evaluación independiente incluye puntos en cada capa y el centro. Reportar error por región junto con el promedio ponderado por área para evitar que los 32 m² de suelo oculten errores del aislamiento.

El estudio debe comparar 1×, 2× y 4× por estrato; añadir 10× solo si la curva y el presupuesto lo justifican. Duplicar todo sin estudiar dónde está el error no tiene respaldo universal. [Wu et al.](https://arxiv.org/abs/2207.10289) motivan comparar también distribuciones. `full_sampling.py` implementa políticas uniforme, mezcla global/cercana, bandas de interfaces y adaptación por residuo. La adaptación conserva mitad uniforme, usa cuatro candidatos por punto y fija la nube antes de L-BFGS; no utiliza temperaturas FEM.

Puerta PINN: NRMSE y error relativo de elevación máxima ≤5 %, RMSE por región ≤5 % de elevación de referencia, balance global y por región ≤2 %, salto máximo de T ≤0,1 K, salto RMS de flujo ≤1 % de su escala declarada. En `mixed`, ambos flujos deben pasar. Estos controles adicionales son más exigentes que aceptar únicamente una temperatura máxima próxima.

Se usa CPU float64 y un hilo por entrenamiento como punto de partida medido. El flujo de un caso es secuencial; la campaña A–D ejecuta hasta tres entrenamientos independientes simultáneos. Las redes por material no implican paralelismo automático. El equipo tiene Intel XPU disponible, pero el ensayo previo no mostró ventaja en float64 frente a CPU con un hilo. No se atribuye a la nueva campaña una aceleración sin medirla. FEM rechaza MPI de más de un proceso hasta implementar ensamblaje/evaluación global correctamente.

## Estado y evidencia

- Pruebas de identidad física: todas las variantes atraviesan todos los materiales y permiten diferenciación; se verifica difusión angular, almacenamiento temporal, regularidad en el centro y conservación de la fuente piel.
- Coaxial fijo: referencia FEM refinada `coaxial_full/fixed/reference_verified`; error de Tmax en nivel 2 de 0,00000368 K frente a solución analítica. Pilotos `subdomain` y `mixed_verified` aceptados; no son confirmación estadística de toda la tesis.
- XLPE 8 × 4 m: referencia FEM fija 36,48194 / 36,81872 / 36,85249 °C en niveles 0 / 1 / 2; balance final 0,00718 %.
- PINN `subdomain` en XLPE 8 × 4 m: 36,54465 °C, error máximo de conductor 0,30784 K, NRMSE 1,6922 %, balance 0,3929 %, aceptada con los criterios completos.
- Flujo integrado nominal con fuente DC dependiente de T local, coaxial: FEM 32,57158 °C, PINN 32,57557 °C, error 0,00399 K; aceptado.
- `coaxial_full/fixed/mixed` conserva un piloto rechazado por flujo derivado; evita seleccionar resultados ocultando fallos.
- La campaña A–D y la extensión de ampacidad tienen protocolos y registros propios en `explicit_study`. Las cifras de aceptación y las conclusiones se generan desde sus resúmenes; los pilotos anteriores no se suman como réplicas finales. El constructor `scripts/build_final_thesis.py` utiliza únicamente la evidencia explícita y verifica huellas y compilación.

El informe `docs/AUDITORIA_POST_METODOLOGIA_2026-09-14.md` describe la evidencia reducida anterior. Las entradas históricas `pinn.py`, `fem.py` y `fem_coupled.py` exigen `--legacy-reduced` para ejecutar cables; los ensayos manufacturados de suelo siguen disponibles.

Los tiempos de estos pilotos se registraron durante trabajo concurrente y no constituyen un benchmark aislado de aceleración. El resumen trazable de resultados, pruebas y depuración es `docs/REVISION_FISICA_EXPLICITA.md`.

## Campaña y corriente límite

```powershell
python -X utf8 Benchmarks/full_study.py
python -X utf8 Benchmarks/full_ampacity_study.py
python -X utf8 scripts/build_final_thesis.py
```

A compara 9 configuraciones × 2 problemas × 2 semillas; B, 6 presupuestos/políticas × 2 problemas × 2 semillas; C, 3 tasas × 2 problemas × 2 semillas; C2 añade 2 pesos de continuidad × 2 problemas × 2 semillas y reutiliza el control de C; D, 7 problemas × 3 semillas nuevas. La [enmienda C2](../docs/PROTOCOLO_CALIBRACION_INTERFACES.md) se registró durante B, antes de C/D, para investigar saltos térmicos persistentes sin relajar tolerancias. La selección no descarta fallos ni busca la mejor semilla. `production_configuration.json` conserva la receta congelada antes de D. El detalle y la comparación FEM están en `report/runs.csv` y `report/summary.json`.

La extensión de corriente límite usa la receta de C y el peso de continuidad de C2, con el doble del presupuesto, fijado previamente, y tres semillas. En una copia nueva sin referencias de ampacidad, generarlas primero en WSL (ruta relativa desde la raíz montada del proyecto):

```bash
for case in xlpe_single xlpe_dry_near xlpe_dry_near_hom_arithmetic; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/lkoc/miniforge3/envs/fenicsx/bin/python Benchmarks/full_ampacity.py --fem --case "$case" --output "Benchmarks/explicit_study/ampacity/references/$case"
done
```

La ruta de Python se ajusta al entorno instalado. El problema mantiene la PDE interior y la ley local `Q=sigma(T)Ez²`. FEM busca la raíz de corriente; PINN aprende `I=I0 exp(eta)` con límite 90 °C. La identidad de corriente desconocida conserva por separado el estado solucionado. `run_case.py --mode ampacity` permite resolver otro caso JSON con las mismas ecuaciones; usar un directorio nuevo y revisar sus puertas de aceptación.

Los resultados tienen rutas originales como procedencia. Los informes resuelven los archivos dentro del checkout actual y verifican su SHA-256. Una réplica nueva necesita un directorio de campaña vacío en una copia aislada, sin borrar la evidencia original. Reiniciar un directorio completo reutiliza resultados y no constituye un nuevo entrenamiento. El estado de aceptación no equivale a ampacidad AC normativa.

Para repetir exactamente una semilla guardada con **sus propias fuentes archivadas**, en otra carpeta:

```powershell
python -X utf8 Benchmarks/repeat_explicit.py --run Benchmarks/explicit_study/A/subdomain_w32_d3/xlpe_single/seed11 --output Benchmarks/explicit_reproduction/seed11
```

El repetidor verifica los hashes, reconstruye un paquete de fuentes aislado, conserva el campo FEM y comprueba la diferencia final. También admite resultados de `full_ampacity.py`. La opción `--prepare-only` verifica y prepara sin entrenar. Las bibliotecas instaladas siguen formando parte del entorno; cambiar PyTorch o hardware exige revisar la diferencia obtenida.

