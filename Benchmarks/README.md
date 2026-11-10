<!-- explicit-heat-2d-v1 -->
**Flujo vigente:** [calor explícito en todas las capas](FULL_DOMAIN.md). Las instrucciones que siguen documentan la campaña reducida histórica. Para cables, las entradas antiguas requieren `--legacy-reduced`; el flujo activo de un caso es `run_case.py`; la campaña y su selección se ejecutan con `full_study.py` y `full_ampacity_study.py`. La [revisión integral](../docs/REVISION_INTEGRAL_TESIS.md) identifica los resultados vigentes.
<!-- /explicit-heat-2d-v1 -->

# Casos reproducibles de conducción térmica

Los archivos `cases/*.json` constituyen la especificación física común. PINN y
FEniCSx leen exactamente el mismo archivo, evalúan la misma conductividad y usan
los mismos puntos de comparación. Las opciones numéricas están separadas en
`configurations/*.json`. Los resultados conservan el caso completo y su SHA-256.

## Alcance físico

Se estudian diecinueve problemas estacionarios: siete soluciones manufacturadas,
un anillo y once escenarios de cables. En los escenarios de cables se resuelve
el suelo con agujeros circulares y flujo uniforme. Las capas internas se
recuperan con resistencias radiales en serie. Esta reducción permite comparar
dos solucionadores del **mismo problema**; no representa un cálculo 2D completo
de cada capa ni reproduce exactamente los artículos de Aras o Kim.

**Los resultados de operación usan la temperatura actual de cada conductor:**
`P_j = current**2 * R20 * (1 + alpha * (T_conductor_j - 20))`, en W/m.
La ley y las tolerancias comunes están en `electrical_model.json`. FEniCSx
calcula una matriz de respuesta térmica, converge la actualización individual
y verifica el resultado mediante una solución matricial independiente. La
PINN aprende las potencias junto con el campo, imponiendo la misma ley R(T).

El campo `power = current**2 * R20` del JSON es la **potencia de referencia**
y la escala del entrenamiento. Los registros con modo `fixed_source` la mantienen
fija para aislar la verificación térmica y comparar arquitecturas. Las salidas
`coupled_results/` (FEM), `coupled_final/` (PINN nominal) y `ampacity_final/`
(PINN a temperatura límite) contienen la física electrotérmica acoplada.
No se incluyen pérdidas dieléctricas, pantallas, proximidad ni efecto pelicular.
La ley DC y α = 0,00393 K⁻¹ se contrastan con CIGRÉ WG B1.56 (2022: 132,
*Power cable rating examples for calculation tool verification*, TB 880).
El suelo ocupa un rectángulo de 8 × 4 m con 20 °C en sus cuatro lados.

## Formato único, versión 1

El contrato de la versión 1 está definido por `cases.py:validate_case` y esta
documentación. JSON usa punto decimal, codificación UTF-8 y unidades SI. No se
admiten comentarios JSON; las explicaciones se guardan en `source` y `adaptation`.

| Campo | Contenido y unidad |
|---|---|
| `id`, `kind` | Identificador estable y familia de ecuaciones |
| `bounds` | `[xmin, xmax, ymin, ymax]`, m; superficie en y = 0 |
| `T0`, `scale` | Temperatura ambiente, °C; escala numérica, K |
| `cables` | Lista de centros `[x,y]`, m |
| `radius` | Radio exterior común, m |
| `layers` | Capas `[radio_interior, radio_exterior, k]`, m, m, W/(m K) |
| `current`, `R20`, `power` | A, Ω/m y W/m por cable |
| `k` | Conductividad del suelo de base, W/(m K) |
| `patch` | `[cx,cy,ancho,alto,k_relleno,epsilon]`; metros y W/(m K) |
| `bands` | Interfaces horizontales `[y,k_arriba,k_abajo]`, de arriba abajo |
| `band_smoothing` | Ancho de transición, m; predeterminado 0,1 m |
| `pair` | Identificador del control homogéneo, o `null` |
| `source`, `adaptation` | Procedencia y supuestos explícitos |

El campo `conductivity` admite números, fórmulas y estratos con salto exacto;
su contrato completo está en [FORMAT.md](FORMAT.md). Los materiales localizados
regularizados usan tangentes hiperbólicas y un ancho explícito. Los estratos
exactos usan subredes con continuidad de temperatura y flujo. Ambos modelos
están presentes tanto en soluciones manufacturadas como en escenarios de cables.

La versión actual admite un tipo de cable y una corriente común por escenario;
las potencias acopladas son distintas si las temperaturas lo son.
Para otro entorno basta copiar un JSON, cambiar `id`, centros, capas y
propiedades, y ejecutar ambos métodos. Los cables no pueden superponerse ni
tocar el borde exterior. Extender a varios tipos exige versionar el contrato
y actualizar ambos lectores; no debe ignorarse silenciosamente un campo.

## Ejecución

La entrada común ejecuta ambos solucionadores y usa R(T) por defecto para
los cables. `--mode verification` solicita explícitamente una fuente fija:

```powershell
python Benchmarks/run_case.py xlpe_single
python Benchmarks/run_case.py xlpe_single --mode ampacity
python Benchmarks/run_case.py mms_high_contrast
```

Las nuevas salidas se guardan en `user_runs/`, conservando las campañas de la
tesis. La opción `--fenics-python` permite indicar el intérprete Linux propio.

Desde la raíz del repositorio, en el entorno Windows usado en esta auditoría:

```powershell
python Benchmarks/pinn.py --cases mms_constant --seeds 11 23 37
python Benchmarks/campaign.py
python Benchmarks/pinn.py --config Benchmarks/configurations/C03_reference.json
```

FEniCSx 0.10.0 se ejecutó en Ubuntu/WSL, con Gmsh 4.15.2:

```powershell
wsl -d Ubuntu -- /home/lkoc/miniforge3/envs/fenicsx/bin/python /mnt/c/usr/ths_mia_fiis/tesis_MIA_lk/Benchmarks/fem.py --cases all --levels 0 1 2
```

En Linux con FEniCSx instalado: `python Benchmarks/fem.py --cases all`.
No se necesita PyTorch en el entorno FEM. Python/Windows contiene PyTorch
2.9.0+xpu, NumPy y Matplotlib; los entrenamientos usaron CPU, precisión doble y
uno o dos hilos, según el manifiesto. Los cuadernos usan además nbformat, nbclient e ipykernel.

La reproducción puede cambiar ligeramente entre plataformas y bibliotecas.
Los manifiestos registran las versiones efectivamente utilizadas. Los tiempos
históricos incluyen procesos concurrentes y diferencias Windows/WSL: no sirven
para afirmar aceleración de PINN frente a FEM.

## Selección y auditoría

`campaign.py` define ocho candidatos, variando una decisión respecto a C03.
Se compara adición del balance, peso de PDE, anchura, profundidad, tasa de
aprendizaje y cantidad de puntos. La semilla 5 es exploratoria. Las semillas
11, 23 y 37 caracterizan sensibilidad; los casos usados en el ajuste se
identifican como tales. Una repetición con otra semilla no constituye por sí
sola validación fuera de distribución.

FEM nunca proporciona etiquetas al entrenamiento. La evaluación usa 6000
puntos independientes y contornos más densos. La aceptación térmica exige
NRMSE ≤ 5 %, error de incremento máximo de temperatura ≤ 5 % y desequilibrio
global ≤ 2 %. Los errores se normalizan con `Tmax_FEM - T0`, evitando cocientes
de temperaturas Celsius dependientes del origen de la escala. Se conservan
errores máximos, residuos y resultados por semilla incluso cuando se rechaza.

La temperatura del conductor es una reconstrucción radial a partir de la
temperatura media de su superficie externa. Los campos de ambas técnicas se
comparan fuera de los cables. Los puntos superficiales se desplazan 0,2 % del
radio para evitar ambigüedades de localización sobre la malla curva. En el
anillo se excluye del muestreo de área el borde exterior de 0,5 % y se evalúa
por separado un perfil cercano al radio interior. Estas decisiones son comunes
a ambos métodos y constan en `cases.py`.

`results/` contiene referencias FEM y la línea base PINN; `comparisons/`
conserva alternativas y pilotos. Los ficheros `.pt` guardan pesos y metadatos,
`.json` métricas e historial, y `.npz` campos numéricos. Los cuadernos muestran
estas salidas y permiten ejecutar de nuevo cada caso. Para una nueva campaña,
usar otro directorio `--output` conserva los registros anteriores.

## Reproducir el acoplamiento y la corriente límite

```powershell
wsl -d Ubuntu -- /home/lkoc/miniforge3/envs/fenicsx/bin/python /mnt/c/usr/ths_mia_fiis/tesis_MIA_lk/Benchmarks/fem_coupled.py --cases all --levels 0 1 2
python Benchmarks/coupled_campaign.py
python Benchmarks/verification_campaign.py
python Benchmarks/resolution_campaign.py
python Benchmarks/refinement_campaign.py
python Benchmarks/architecture_coupled_campaign.py
python Benchmarks/ampacity_refinement_campaign.py
python Benchmarks/adaptive_campaign.py
python Benchmarks/reevaluate.py
python Benchmarks/select_configuration.py
python Benchmarks/report.py
python Benchmarks/numerical_analysis.py
python Benchmarks/adaptive_analysis.py
python Benchmarks/internal_report.py
python Benchmarks/thesis_results.py
python Benchmarks/notebooks.py
```

La campaña conserva tres semillas por caso y modo. El modo `--ampacity`
añade una corriente desconocida y exige Tmax = 90 °C; el modo `--coupled`
conserva la corriente nominal del JSON. Se exige residuo eléctrico ≤ 0,1 %,
además de la puerta térmica. Para corriente límite se verifica error frente
a FEM ≤ 5 % y distancia a 90 °C ≤ 0,1 K. `ampacity.py` conserva únicamente
el índice histórico con resistencia uniforme a 90 °C; no genera la tabla
principal de ampacidad acoplada.

El [informe interno](INFORME_INTERNO.md) presenta todos los intentos, ventajas,
limitaciones y gráficos. La [especificación](FORMAT.md) explica fórmulas y
estratos. Los [cuadernos](notebooks/) incluyen citas y referencias; sus salidas
guardadas revisan ejecuciones realizadas previamente por los solucionadores.

El [análisis numérico](ANALISIS_NUMERICO.md) separa sensibilidad a neuronas,
colocaciones y presupuesto, orden observado FEM, pendientes empíricas PINN y
propagación electrotérmica. La similitud entre redes debe cumplir tolerancias
de campo y Tmax, además de la comparación externa. No se identifica el número
de neuronas con un tamaño de malla ni se aplica GCI sin una expansión justificada.

Después de completar las campañas, `python scripts/build_final_thesis.py`
genera tablas, informe interno, análisis, cuadernos y PDF en el orden de sus
dependencias. `python scripts/review_final_artifacts.py` comprueba cuadernos y
compilación y genera páginas para revisión visual. El comando
`python Benchmarks/manifest.py --verify` comprueba la integridad de la entrega
registrada; una campaña nueva requiere conservar su propio manifiesto.

El [expediente de muestreo adaptativo](MUESTREO_ADAPTATIVO.md) compara nube
fija, renovación aleatoria, residuo y gradiente térmico con igual número
interior y presupuesto. Las 18 ejecuciones adicionales usan dos escenarios
y tres semillas. `collocation_seed*.npz` conserva candidatos, coordenadas,
puntuaciones y probabilidades; para indicadores físicos se archivan también
los estados de red de cada actualización. No se adapta la frontera ni se
afirma que un gradiente alto sea un error. Los cuadernos de `kim_layered` y
`xlpe_backfill` incluyen la comparación y sus gráficos.

