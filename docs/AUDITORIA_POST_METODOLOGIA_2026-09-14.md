<!-- explicit-heat-2d-v1 -->
Esta auditoría conserva el diagnóstico de la formulación reducida anterior. La implementación posterior con ecuación de calor interior y su evidencia se describen en [REVISION_FISICA_EXPLICITA.md](REVISION_FISICA_EXPLICITA.md).
<!-- /explicit-heat-2d-v1 -->

# Auditoría posterior a la definición metodológica

Fecha: 14 de septiembre de 2026. La revisión de implementación comenzó después de guardar y aprobar técnicamente `METODOLOGIA_PROPUESTA_2026-09-14.md`, a las 13:47:19 hora de Lima. Su SHA-256 previo a la auditoría es `0713FCC8DCCE51B0AA4031BBEB242263B5A5878FFA0B3426C5B9FFA3D9F90C06`. Esa aprobación se refiere al protocolo, no a resultados ni a una aprobación institucional.

**Dictamen del artefacto:** existe evidencia útil y reproducible para un modelo DC reducido de suelo 2D y reconstrucción radial del cable. No corresponde aprobarlo todavía como solución verificada del problema general 2D multicapa explicitado en la metodología. Los principales pendientes son cuantificar el error de esa reducción, hacer obligatoria la verificación de interfaces, separar selección y confirmación, y completar convergencia de dominio y cobertura de casos difíciles. La buena concordancia PINN–FEM no elimina errores de modelado compartidos.

Se inspeccionó la ruta activa `Benchmarks`, su conexión con `pinn_cables/pinn/pde.py`, los contratos, campañas, selección y generación de tablas, además de los capítulos de desarrollo y conclusiones. No se modificó el solucionador ni se sobrescribieron las campañas de tesis. Los archivos nuevos de diagnóstico y repetición están en `docs/auditoria/metodologia_2026-09-14/`.

## 1. Hallazgos que condicionan la aceptación

### A1. Las capas del cable no forman parte del dominio 2D resuelto

Evidencia: `Benchmarks/fem.py:create_mesh` recorta discos y resuelve el exterior; `Benchmarks/pinn.py:boundary_points` prescribe flujo en circunferencias; `Benchmarks/cases.py:radial_resistance` y `pinn.py:evaluate_model` reconstruyen la temperatura del conductor a partir de la media superficial y resistencias radiales. `README.md` y el capítulo de desarrollo reconocen expresamente esta reducción; no se trata de una limitación ocultada.

Para `xlpe_single`, el dominio es 8×4 m y el radio exterior es 15 mm. El catálogo contiene conductor de radio 5,5 mm, aislamiento de 6,5 mm de espesor, pantalla de 1 mm y cubierta de 2 mm. Esos datos intervienen en una resistencia, pero esas capas no reciben puntos interiores PINN ni elementos FEM propios. Aumentar N en el código actual incrementa puntos en suelo; no mejora la resolución de esas capas ausentes.

La referencia y la PINN comparten flujo circular uniforme y reconstrucción a partir de la media superficial. Su concordancia verifica la solución del modelo reducido, no el error de reducción. En los campos seleccionados a corriente límite, `kim_sand` muestra hasta 16,8237 K de variación angular de temperatura en una superficie de cable. Esta diferencia no es una cota del error interno: demuestra que el entorno no es axisimétrico y hace prioritario contrastar la aproximación con un modelo multicapa explícito. La conducción circunferencial de pantallas y el flujo angular se pierden con una resistencia radial única.

**Decisión:** conservar el modelo reducido como línea base. Antes de usarlo para sostener conclusiones sobre el máximo real del conductor en entornos próximos heterogéneos, compararlo con FEM de capas explícitas en geometría simétrica y asimétrica. Si excede el presupuesto de error, implementar M1/M2 por materiales o restringir formalmente el alcance. No es necesario desechar las verificaciones manufacturadas ni las fuentes existentes.

### A2. La aceptación omite las interfaces que ya se diagnostican

Evidencia: `pinn.py:evaluate_model`, líneas 341–376, calcula `interface_T_max_K` y `interface_flux_max_W_m2`; el booleano `thermal_criteria_pass` usa NRMSE, error de elevación máxima y balance global, más residuo eléctrico cuando corresponde. No incorpora los saltos ni una regla local por conductor.

`xlpe_discrete_layers` está aceptado en las tres semillas nominales y de ampacidad bajo ese filtro. En nominal, sus saltos máximos de temperatura son 0,3508; 0,2444 y 0,3160 K. En ampacidad son 1,6139; 1,3591 y 1,3888 K; los saltos máximos de flujo normal son 2,2384; 2,1715 y 2,7026 W/m². El contacto de suelo declarado es perfecto, por lo que el salto térmico debería tender a cero. Los valores en W/m² requieren una escala física local para juzgar relevancia; no se interpretan como porcentajes.

Asimismo hay semillas marcadas como aceptadas con errores integrados de potencia de un conductor superiores a 2 %, aunque el balance global pase. Por ejemplo, `kim_layered` nominal semilla 11 tiene 2,1325 % y `kim_pac` nominal semilla 11 tiene 2,4736 %. La compensación global no reemplaza el balance por fuente.

**Decisión:** el estado «aceptado» histórico significa exclusivamente que pasó el filtro histórico. La aceptación de la nueva metodología exige RMS/máximo de interfaz, balance por región y error constitutivo en M2, con escalas y tolerancias fijadas antes del nuevo piloto. No se deben cambiar retrospectivamente los registros originales; la nueva clasificación debe quedar versionada. No basta recalcular los mismos tres indicadores.

### A3. La selección de configuración no es confirmación independiente

Evidencia: `select_configuration.py:choose` selecciona primero el mayor número de semillas aceptadas, aunque sea uno de tres. Luego usa error, dispersión y tamaño; las semillas 11,23,37 se utilizan para comparar y elegir. El propio `selection.json` declara correctamente que no demuestra generalización. La tesis muestra intentos rechazados.

La palabra «selected» no autoriza pasar automáticamente a OE3/OE4. Se necesita separar configuración escogida para investigar de artefacto aceptado dentro de un dominio. Por ejemplo, las familias Kim nominales solo tienen 1/3 semillas aceptadas con las reglas actuales.

**Decisión:** conservar las tres semillas como desarrollo, congelar la configuración por familia o una regla de elección previa y ejecutar cinco semillas nuevas de confirmación. El fracaso de una familia debe producir restricción/inconclusión explícita. La mediana calculada solo con semillas aprobadas es un resumen condicionado al éxito: acompañarla por resultados de todos los intentos y tasa de fallo. El código de tablas ya informa el número aceptado y el error máximo de todas las semillas; se debe mantener esa transparencia.

### A4. Regularización de materiales y equivalencia requieren estudios físicos separados

Los casos `patch` usan transiciones tanh de ancho `epsilon`; por ejemplo, `xlpe_dry_near` declara 0,08 m. No es una interfaz exacta ni una capa de cable de ese espesor. Un campo suave puede ser válido como escenario prescrito, pero no se equipara automáticamente a un bloque de material con borde abrupto. Hay saltos exactos implementados para estratos planos; eso no constituye una geometría de interfaz general para inclusiones y cables.

Los pares actuales usan principalmente el suelo base `xlpe_single` como control. Esto permite estudiar introducir una heterogeneidad frente al terreno base. No cubre automáticamente todo el catálogo C4 de homogéneo equivalente, conservador y optimista del plan. El ancho de transición debe tratarse como parámetro físico/modelador y estudiarse frente a interfaces exactas o mediante una sensibilidad controlada, sin afinarlo para mejorar concordancia.

No se identificó en las campañas y tablas revisadas un barrido de expansión del dominio artificial. Los tres niveles FEM refinan la malla dentro del dominio existente; son distintos de alejar las fronteras. No se atribuye a la dimensión 8×4 m una independencia de contorno que no se ha medido.

### A5. Cantidad de puntos: hay un estudio útil, pero no cubre la dificultad multicapa

`training_points` mezcla muestras globales con muestras logradiales alrededor de cada cable y, en algunos casos, un rectángulo de atención prefijado. Es mejor que depender exclusivamente de puntos uniformes. Las coordenadas de cables se escalan con constantes globales; no existe normalización independiente para cada capa interna, ya que no se resuelven.

El parámetro `n` no coincide con el número real de colocaciones. En la campaña XLPE simple, n=384,768,1536 produce aproximadamente 760,1515,3013 puntos para la semilla mostrada; varía ligeramente con los rechazos geométricos. Contornos e interfaces se fijan a 80 puntos por segmento y 128 por interfaz durante entrenamiento. Duplicar n no duplica esos controles. La evaluación usa 6000 puntos interiores independientes y 512 puntos por borde; la reconstrucción superficial utiliza 128 posiciones.

Las tres medianas de error de campo archivadas y comprobadas son:

| Caso | N384 | N768 (W32) | N1536 |
|---|---:|---:|---:|
| MMS suave 2D | 0,003680 K | 0,002575 K | 0,002251 K |
| XLPE simple, R(T) | 0,012309 K | 0,008370 K | 0,009142 K |

En el segundo caso, duplicar de 768 a 1536 no mejoró el error bajo el presupuesto usado. Esto respalda medir suficiencia, no ordenar ×10 indiscriminadamente. No demuestra que el mismo presupuesto baste para inclusiones de alto contraste o capas explícitas.

La campaña adaptativa ensaya una adaptación propia de tamaño fijo, con mitad de anclajes y dos redistribuciones en Adam. Conserva nubes, candidatos y puntuaciones, y congela la nube durante L-BFGS. Su documentación distingue la variante propia del algoritmo de Wu. En `kim_layered`, la mediana RMSE pasó de aproximadamente 0,5036 K con puntos fijos a 0,7053 K por residuo, y la aceptación quedó en 1/3. En `xlpe_backfill` a ampacidad, el residuo tiene 2/3 semillas aceptadas. La adaptación no fue una mejora universal.

**Decisión:** extender la curva de resolución a casos difíciles con estratos materiales, variando interiores e interfaces por separado. Antes, resolver A1/A2: no se corrige un modelo reducido ni una condición ausente mediante más puntos en suelo. El protocolo N₀,2N₀,4N₀,8N₀ de la propuesta es un experimento a realizar, no un resultado producido en esta auditoría.

## 2. Lo que ya funciona y conviene conservar

El contrato JSON valida radios, espesores, unidades implícitas en campos documentados, potencia I²R20, ausencia de solapamiento, conductividades y expresiones; rechaza combinaciones no soportadas. La PDE incluye ∇k·∇T para coeficientes suaves. Las interfaces planas usan subredes y trazas laterales. Los casos manufacturados y el anillo separan capacidades elementales del modelo.

Los solvers comparten la ficha física y comparan la huella del caso y puntos. El entrenamiento declara cero etiquetas FEM y el código lee la referencia para evaluación después de entrenar. Las métricas de temperatura ya usan elevación respecto de ambiente: este aspecto de la implementación es más preciso que la fórmula todavía escrita en el plan. Debe trasladarse documentalmente esa precisión, preservando diferencias entre la NRMSE actual —normalizada por elevación máxima de cada caso— y una escala fija común a pares propuesta para análisis comparativo.

FEM usa P2, refinamiento local y tres mallas. En las 41 series FEM inspeccionadas —19 a fuente fija y 22 nominal/ampacidad— el mayor cambio de Tmáx entre niveles 1 y 2 fue 0,0070161 K. Esto es evidencia de estabilidad de temperatura para el problema discretizado; no es una cota certificada del error total ni del dominio o de las capas omitidas. La variación de Tmáx en ampacidad está forzada por Tlim, por lo que su convergencia debe juzgarse además con corriente y campo; las tablas existentes incluyen cambio de corriente.

La actualización R(T) es individual por conductor. FEM construye matriz de respuesta y contrasta el punto fijo con una solución algebraica independiente. La PINN aprende potencias y corriente límite, penalizando ley eléctrica y temperatura. Esta última formulación simultánea es admisible como alternativa a la bisección propuesta, siempre que se comprueben solución estable, potencias positivas, restricciones y una verificación externa de corriente. Reducir el residuo del límite sin resolver bien el campo no basta.

`Benchmarks/ampacity.py` es un índice histórico con R fijada a la temperatura límite; la ruta operativa acoplada utiliza `electrothermal.py`, `fem_coupled.py` y la variante correspondiente de PINN. La documentación distingue ambas rutas. Es importante mantenerlas separadas para que una reproducción no use accidentalmente la fórmula histórica como resultado principal.

## 3. Reproducibilidad: suficiente para releer y repetir, incompleta para reanudar

Se verificaron hashes de casos, referencias y fuentes archivadas de las 90 ejecuciones seleccionadas. Sus RMSE fueron recalculados directamente desde los campos NPZ y coincidieron con los JSON. Hay buenas bases: configuración separada, fuentes archivadas, versiones, nubes de colocación, historial, pesos y salidas anteriores conservadas.

Hay cuatro mejoras concretas:

1. `pinn.py` guarda `state_dict` y metadatos, pero no estado del optimizador ni todos los estados RNG/paso para reanudación exacta. Actualmente se puede reiniciar una corrida desde su configuración; cargar pesos no equivale a continuar Adam/L-BFGS donde se interrumpió.
2. `campaign.py` omite una corrida si encuentra un JSON existente, sin validar ahí su identidad y estado. La auditoría separada detecta parte de la obsolescencia, pero el planificador debe consultar un manifiesto compatible antes de reutilizar.
3. `run_case.py` usa una configuración genérica; no reproduce automáticamente la seleccionada para cada caso. Además, el JSON de configuración sobrescribe argumentos CLI en `pinn.py`. Para cambiar la salida de una reproducción con `--config`, hay que modificar la copia del JSON, no confiar en que `--output` prevalezca. La repetición de esta auditoría usa copias explícitas.
4. Faltan campos generales de procedencia por dato y contornos arbitrarios. `source` puede apuntar a una carpeta interna y la descripción genérica de adaptación menciona Kim incluso en XLPE simple. Se requiere enlazar radios, k, pérdidas y contornos a documento y tabla/página, separando supuestos de datos publicados. Los archivos `environment` son una base útil; se conserva la advertencia existente de que inventariar paquetes no prueba por sí solo reconstrucción de todo el entorno.

La portabilidad de referencias está contemplada por `validate_artifacts.reference_path`. El entorno Windows tenía PyTorch 2.9.0+xpu, pero las campañas ejecutan CPU. El entorno FEM está en Ubuntu/WSL. La repetición debe conservar modos eléctricos compatibles; no basta coincidir en nombre de caso.

## 4. Recursos y paralelismo medidos

Inventario consultado en esta auditoría: Intel Core Ultra 7 165U, 12 núcleos físicos/14 procesadores lógicos, aproximadamente 31,5 GiB de RAM, una GPU integrada Intel Graphics, driver 32.0.101.8826. `torch.cuda.is_available()` es falso y `torch.xpu.is_available()` verdadero. El backend reconoce y ejecuta una prueba de primeras/segundas derivadas y retropropagación tanto en float64 como en float32; no se presume soporte nativo de FP64 a partir de esa prueba.

La ruta PINN actual construye tensores y redes en CPU, fija float64 y no ofrece selector de dispositivo. Las campañas sí explotan paralelismo entre procesos: acoplamiento con cuatro trabajos de un hilo; resolución y adaptación con tres de un hilo; verificación con tres trabajos de dos hilos. `ThreadPoolExecutor` coordina subprocesos Python separados, por lo que no corresponde concluir que el GIL impide ese paralelismo. Dentro de una corrida no hay distribución de subredes entre dispositivos.

Se ejecutó un microbenchmark aislado: MLP 32×3, 1536 puntos, residual con segundas derivadas y backward; diez calentamientos, tres repeticiones de 25 pasos, sincronización XPU. Medianas:

| Dispositivo/precisión | Hilos CPU | ms por paso |
|---|---:|---:|
| CPU float64 | 1 | 8,49 |
| CPU float64 | 2 | 14,40 |
| CPU float64 | 4 | 15,87 |
| CPU float64 | 8 | 18,14 |
| XPU float64 | — | 13,11 |
| XPU float32 | — | 6,21 |

Estos datos miden exclusivamente el trabajo sintético indicado: no incluyen optimizador, interfaces, exactitud final ni rendimiento de una campaña completa. La fila float32 no se compara como aceleración equivalente contra CPU float64, porque cambia la precisión. La prueba no demuestra que una migración de toda la PINN a XPU vaya a mejorar el tiempo hasta solución aceptada.

**Respuesta sobre uso apropiado:** existe paralelismo útil y un hilo por red pequeña tiene justificación local. La máquina no está siendo utilizada como entrenamiento XPU ni como solucionador distribuido, pero eso no prueba desaprovechamiento: a igual float64, la microprueba favoreció CPU de un hilo. Falta medir campañas con 1,2,3,4 trabajadores y tiempo hasta exactitud comparable para elegir su concurrencia; no se debería elevar automáticamente a 14 hilos.

FEM usa un rango MPI. No basta añadir `mpirun -n`: `fem.py:evaluate` supone que todos los puntos consultados pertenecen a la malla local, y los balances ensamblados no hacen reducción MPI global. Distribuirlo exige localizar y reunir evaluaciones y reducir integrales correctamente. Los recursos NPU no se evaluaron; no hay evidencia para recomendar su uso en entrenamiento con derivadas espaciales de orden superior.

La limitación general de reproducibilidad entre dispositivos está descrita por [PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness). La disponibilidad del acelerador se comprobó localmente; las [notas oficiales Intel](https://www.intel.com/content/www/us/en/developer/articles/release-notes/gpu-dependencies-for-pytorch-release-notes.html) no sustituyen esas pruebas ni implican ventaja de rendimiento de este modelo.

## 5. Balance de resultados existentes

Conteos recalculados desde las configuraciones seleccionadas, bajo sus criterios históricos:

| Grupo | Aceptadas/intentos | Interpretación |
|---|---:|---|
| Ocho referencias analíticas/manufacturadas | 24/24 | Verificación elemental útil |
| Once escenarios de cable nominal | 29/33 | Fallos concentrados en Kim estratificado/PAC |
| Once escenarios de ampacidad | 30/33 | Fallos en Kim estratificado/PAC y XLPE con relleno |

Esto suma 83/90 intentos aceptados. No es una tasa de generalización: se escogieron configuraciones usando estos mismos casos y semillas. La aceptación agregada tampoco resuelve A1/A2.

| Caso | Nominal | Ampacidad | Decisión metodológica |
|---|---:|---:|---|
| XLPE simple | 3/3 | 3/3 | Base reducida con buen acuerdo; falta comparación multicapa explícita |
| Kim estratificado | 1/3 | 2/3 | No aprobar robustez de familia |
| Kim PAC | 1/3 | 2/3 | No aprobar robustez de familia |
| XLPE relleno | 3/3 | 2/3 | Confirmación operativa pendiente |
| XLPE estratos exactos | 3/3 | 3/3 | Revisar clasificación por saltos de interfaz |

El caso XLPE con zona seca cercana reduce la ampacidad FEM del modelo de 487,2270 a 418,2753 A, aproximadamente 14,15 %. Puede conservarse como resultado del escenario reducido y de la representación específica declarada. No se interpreta como reducción observada en una instalación ni como prueba de todos los patrones/niveles del plan. El catálogo de once cables es una cobertura inicial, no las hasta 108 realizaciones potenciales del plan.

## 6. Comprobaciones nuevas y evidencia

- `python -m pytest Benchmarks/tests -q`: **29 pruebas aprobadas** en 4,32 s durante esta revisión.
- `collect_evidence.py`: comprobó las 90 ejecuciones seleccionadas; regeneró RMSE desde campos y verificó huellas físicas/de referencias/fuentes. Resumen y detalle: `evidence.json`.
- `probe_compute.py`: consultó hardware y produjo `compute_probe.json`; la prueba de XPU se amplió con `--xpu-only` conservando las mediciones CPU.
- `repeat_selected.py`: reinició desde la configuración archivada dos entrenamientos completos con semilla 11, `mms_interface` y `xlpe_single` acoplado, en un directorio nuevo. Antes de ejecutarlos fijó tolerancia de reproducción de 0,02 K para diferencia máxima de campo y Tmáx, además de exigir aceptación térmica histórica. **Ambas repeticiones pasaron y sus diferencias de campo y Tmáx fueron exactamente 0 K en los arreglos guardados.** `mms_interface`: Tmáx=32,49533958285904 °C, RMSE FEM=0,0022003243594650303 K; `xlpe_single`: Tmáx=38,061060260240396 °C, RMSE FEM=0,0063160924632416566 K. El resultado completo está en `repeat_evidence.json`. La identidad de estos arreglos no significa identidad binaria de todos los archivos, que incluyen tiempos y procedencia nuevos.

Estas repeticiones prueban reinicio desde configuración en el entorno presente. No son cinco semillas nuevas, no recalculan FEM y no constituyen evaluación de M1/M2 multicapa. No se relanzó toda la campaña porque no resolvería las brechas A1/A2: repetir un modelo reducido no cuantifica el error de reducir las capas.

## 7. Orden de ejecución recomendado tras esta auditoría

1. Incorporar al plan la especificación metodológica adjunta y registrar claramente alcance completo frente a línea base reducida. Fijar denominadores, efectos mínimos y criterios de aceptación por interfaz, región y semilla.
2. Construir FEM de capas explícitas, comprobarlo con conductor/anillos analíticos y convergencia geométrica/local. Compararlo con el reducido en XLPE simétrico y Kim/inclusión próxima asimétrica. Medir error en máximo del conductor, distribución angular y ampacidad.
3. Implementar la candidata M1 por materiales y la ablación M2 mixta, con coordenadas locales. Mantener enriquecimiento como variante declarada; verificar balance derivado de temperatura y ley de Fourier.
4. Ejecutar convergencia de colocación por región/interfaz y expansión de dominio en casos difíciles. Elegir presupuesto por exactitud, no por un multiplicador arbitrario.
5. Congelar configuración, ejecutar confirmación con semillas nuevas y ampliar escenarios/pares hasta la cobertura justificada del plan. Solo entonces renovar tablas y conclusiones principales de OE3/OE4.

El cambio prioritario no es «usar diez veces más puntos» ni «activar GPU». Es asegurar que el problema implementado y sus criterios de aceptación corresponden a la pregunta de tesis. Las mejoras de cómputo vienen después de esa correspondencia y se eligen por medición.
