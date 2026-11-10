# Modelos

Cambiar de modelo es cambiar un argumento: `--model NOMBRE` (o `"model"` en el JSON).

| Nombre | Partición | Formulación | Idea |
|---|---|---|---|
| `M0` | global | fuerte | una red para todo el dominio; control |
| `M1` | por región | fuerte | una red por material o subdominio; interfaces como pérdidas |
| `M1_EXACT` | por región + trazas compartidas | fuerte | fronteras e interfaces exactas; objetivo solo PDE; experimental |
| `DeepXDE_EXACT` | mismo campo exacto compartido | fuerte, PyTorch | entrenado por DeepXDE, sin penalizaciones de frontera/interfaz |
| `M2` | por región | mixta | salidas $(T, q_x, q_y)$; la ley constitutiva $q=-k\nabla T$ es una pérdida |
| `CUSP` | adaptada | fuerte | variante de la fase 5 |
| `DCSNN` | adaptada | fuerte | variante de la fase 5 |
| `FBPINN` | adaptada | fuerte | descomposición del dominio adaptada |
| `<modelo>+rba` | igual que el base | igual | atención por residuo (Adam solamente) |
| `<modelo>+sa` | igual que el base | igual | pesos autoadaptativos (Adam solamente) |

Las variantes de la fase 5 son **adaptaciones** de las ideas originales, no reproducciones. I-PINN,
AdaI-PINN y VPINN no están implementados.

## Referencias

Citas comprobadas el 23/10/2026 contra Crossref y arXiv (DOI y metadatos). Las marcas "por verificar"
de `REFACTOR_PLAN.md` quedan superadas por esta sección.

| Modelo del front-end | Origen | Tipo |
|---|---|---|
| `M0`, `M1`, `M2` y variantes con penalización | Raissi et al. (2019) | base PINN |
| `M1_EXACT`, `M0_EXACT`, `DeepXDE_EXACT` (traza exacta) | método propio del proyecto | sin referencia externa |
| `CUSP`, `CUSP_EXACT` | Tseng et al. (2023) | adaptación |
| `DCSNN`, `DCSNN_EXACT` | Hu et al. (2022) | adaptación |
| `FBPINN`, `FBPINN_EXACT` | Moseley et al. (2023) | adaptación |
| `DeepXDE`, `DeepXDE_PFNN_EXACT`, `DeepXDE_FNN_EXACT` | Lu et al. (2021); PFNN de Sheng y Yang (2021) | librería y arquitectura |
| `+rba` | Anagnostopoulos et al. (2024) | atención residual |
| `+sa` | sin cita verificada en la base documental | pendiente |
| Caso `kim_v5_summer` | Kim et al. (2025) | datos del caso |

- Anagnostopoulos, S. J., Toscano, J. D., Stergiopulos, N., y Karniadakis, G. E. (2024). Residual-based attention in physics-informed neural networks. *Computer Methods in Applied Mechanics and Engineering, 421*, 116805. https://doi.org/10.1016/j.cma.2024.116805 (preprint: arXiv:2307.00379)
- Hu, W.-F., Lin, T.-S., y Lai, M.-C. (2022). A discontinuity capturing shallow neural network for elliptic interface problems. *Journal of Computational Physics, 469*, 111576. https://doi.org/10.1016/j.jcp.2022.111576
- Kim, Y.-S., Nguyen Cong, H., Dinh, B. H., y Kim, H.-K. (2025). Effect of ambient air and ground temperatures on heat transfer in underground power cable system buried in newly developed cable bedding material. *Geothermics, 125*, 103151. https://doi.org/10.1016/j.geothermics.2024.103151
- Lu, L., Meng, X., Mao, Z., y Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review, 63*(1), 208–228. https://doi.org/10.1137/19M1274067 (preprint: arXiv:1907.04502)
- Moseley, B., Markham, A., y Nissen-Meyer, T. (2023). Finite basis physics-informed neural networks (FBPINNs): A scalable domain decomposition approach for solving differential equations. *Advances in Computational Mathematics, 49*(4), 62. https://doi.org/10.1007/s10444-023-10065-9 (preprint: arXiv:2107.07871)
- Raissi, M., Perdikaris, P., y Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics, 378*, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Sheng, H., y Yang, C. (2021). PFNN: A penalty-free neural network method for solving a class of second-order boundary-value problems on complex geometries. *Journal of Computational Physics, 428*, 110085. https://doi.org/10.1016/j.jcp.2020.110085
- Tseng, Y.-H., Lin, T.-S., Hu, W.-F., y Lai, M.-C. (2023). A cusp-capturing PINN for elliptic interface problems. *Journal of Computational Physics, 491*, 112359. https://doi.org/10.1016/j.jcp.2023.112359

## Modo por penalización

- `M0`, `M1`, `M2` y las variantes conservan las condiciones de frontera e interfaces como pérdidas blandas.
- La salida es `T_ref + amplitud x crudo`, con escalas por región para que las pérdidas sean adimensionales.
- La ecuación `-div(k grad T) = Q` se evalúa en cada región con su propia conductividad y fuente.

## Términos de la pérdida

| Término | Qué penaliza | Peso por defecto |
|---|---|---|
| `pde` | residuo de la ecuación en cada región | 1 |
| `constitutive` | `q + k grad T` (solo M2) | 1 |
| `interface_temperature` | salto de T en interfaces | 100 |
| `interface_flux` | salto del flujo normal | 10 |
| `dirichlet` | error de temperatura en lados Dirichlet | 100 |
| `neumann` | error de flujo en lados Neumann | 10 |
| `robin` | error de la ley de Robin | 10 |
| `energy` | balance de energía por región | 1 |

Cambiar un peso: `--weights dirichlet=50 interface_flux=20` o `"weights": {...}` en el JSON.

## Modo exacto `M1_EXACT`

Es optativo; `M1` sigue siendo el modo por defecto. En cada interfaz circular, los dos materiales
comparten una traza Fourier para $T$ y otra para el flujo normal $q_n$. El núcleo extiende esas
trazas con un polinomio armónico y cada capa con una interpolación radial de Hermite. Así, los dos
lados tienen exactamente el mismo $T$ y $q_n$, aunque sus gradientes normales sean distintos:
$\partial_n T_i=-q_n/k_i$. No se impone gradiente cero.

En la frontera externa rectangular, una envolvente anula la corrección de red y fija el Dirichlet
constante. El objetivo que ve el optimizador contiene **solo `pde`**; Dirichlet, interfaz,
constitutiva, Neumann/Robin y energía no se suman a la pérdida. La evaluación final puede medir esas
condiciones como auditoría independiente.

El motor selecciona un levantamiento según el tipo de problema:

- **Cables flat/trefoil:** cantidad y posiciones variables dentro del rectángulo, si las capas no se
    solapan y mantienen collars separados. Requiere fuente fija, suelo homogéneo, contacto perfecto y
    Dirichlet constante igual en los cuatro lados. Las interfaces circulares comparten trazas de T y flujo.
- **MMS:** rectángulo unitario, interfaces materiales planas alineadas con un eje y esquinas
    compatibles. Impone Dirichlet, Neumann y Robin según el caso; las interfaces comparten trazas planas.
- **Annulus CIGRE:** anillo concéntrico de una región con condiciones Dirichlet, Neumann o Robin en
    los radios interior/exterior, incluyendo el benchmark de flujo interior y temperatura exterior.

En cables todavía rechaza suelo estratificado, perfiles o condiciones externas mixtas, dominio
exterior circular, contacto imperfecto, geometrías solapadas y capas/collares sin separación. En MMS,
un lado Neumann/Robin requiere valores Dirichlet compatibles en ambas esquinas. Los casos incompatibles
se rechazan; el modo no hace fallback a M1.

```powershell
python -m pinn_v2 train --case aras_flat --model M1_EXACT --adam-steps 20000 --lbfgs-steps 200 --reference results/phase6/p6_fem3/aras_flat/base/fem_l2.npz
```

El `result.json` mantiene el formato habitual: `config.architecture` identifica `M1_EXACT`,
`config.loss_policy` registra `pde_only_exact_transmission` y `config.weights` guarda el peso PDE
efectivo con los demás términos en cero. `config.input_normalization` versiona el encoder. Los
checkpoints anteriores sin ese campo se reconstruyen con `legacy_region_coordinates`; los nuevos
usan `unit_box_v1`, con entradas en `[0,1]`; los encoders previos se reconstruyen sin cambios.
Checkpoints históricos sin `loss_policy`
siguen usando su arquitectura original y la pérdida por penalización. `M1_EXACT` es experimental; requiere
auditoría física completa y no se considera válido solo por ajustar Tmax.

## DeepXDE exacto

`--model DeepXDE_EXACT` reutiliza el levantamiento y las redes regionales de `M1_EXACT` dentro
de una red personalizada `dde.nn.NN`. DeepXDE recibe las temperaturas fisicas y calcula sus
residuos PDE por region con autograd PyTorch. La lista de restricciones de `dde.data.PDE`
queda vacia: no hay terminos Dirichlet, Neumann, Robin, interfaces ni energia en la loss.
Esto compara entrenadores con el mismo ansatz, no una PFNN/FNN distinta. El `DeepXDE` historico
con PFNN/FNN y penalizaciones sigue disponible sin cambiar de formulacion.

Admite `float32`/`float64`, encoder `unit_box_v1`, fuentes heterogeneas o nulas y las mismas
capacidades geometricas del levantamiento exacto. Adam usa decaimiento coseno y L-BFGS se
ejecuta como optimizador PyTorch dentro de DeepXDE: cada paso externo permite 20 iteraciones
internas, con busqueda de linea. Los puntos permanecen fijos durante el refinamiento.

Los tests comparan residuo y gradientes con `Objective` nativo, no solo la temperatura.
No se implementa atencion RBA/SA ni remuestreo dinamico en esta ruta. El par PT/JSON permite
inferencia y warm start; no se anuncia reanudacion exacta del optimizador DeepXDE. Las
condiciones generales de compatibilidad y la auditoria fisica siguen siendo obligatorias.

## Atencion nativa (RBA y SA)

Las dos modifican los pesos por punto del residuo interior de forma cerrada, a partir del residuo
sin gradiente. No leen ninguna solución de referencia. Como el peso cambia durante el entrenamiento,
L-BFGS no se puede usar: CLI/API/trainer rechazan esa combinacion en vez de cambiarla silenciosamente.
`total` conserva el valor PDE no ponderado; su gradiente incorpora los pesos por punto.
`optimized_pde` permite observar el valor ponderado y el bundle guarda el estado de la atencion.

## Parametrizaciones exactas experimentales

`conditioned_v1` representa la temperatura media mediante un modo compartido por cable y
caidas entre capas escaladas con las escalas termicas de los materiales. Los modos angulares
tienen escala regional. `conservative_v1` agrega la media de flujo `P_encerrada/(2*pi*r)`
derivada de la fuente y geometria declaradas, manteniendo libres los otros modos Fourier.
Ambas conservan temperatura/flujo compartidos, no necesitan datos FEM y no agregan perdidas.

Son opciones de investigacion, no mejoras aceptadas: el screening de 400 pasos del 19 de octubre
rechazo ambas en `aras_flat`. La parametrizacion conservativa empeoro el balance global, aunque
las medias de flujo de las interfaces internas satisfacen su integral. Ver [resultados](results.md).

El motor se selecciona por capacidades de geometria/BC/fuente, no por nombre flat/trefoil ni ID.
Admite fuente nula sin introducir calor artificial: las escalas se obtienen de k, delta-T y longitud,
y la auditoria publica balances absolutos y la base de normalizacion. Problemas puramente Neumann
sin fijacion de nivel siguen siendo rechazados. No se afirma soporte universal de geometria CAD,
contacto imperfecto, transitorios o combinaciones BC que el levantamiento no haya verificado.

## Transferencia de aprendizajes: revision del 2026-10-20

Evidencia de partida: MMS heterogeneo con fuente, 400 Adam + 5 L-BFGS, puntos fijos y float64,
con fronteras/interfaces exactas y solo PDE. M1_EXACT paso tres inicializaciones con una nube;
DeepXDE_EXACT paso una por dtype. La [ficha reproducible](reproducibility.md) detalla los parametros.
El cable de alto contraste todavia no esta resuelto: no extrapolar aceptacion del MMS a Aras/Kim.

Son transferibles como practicas: JSON/physics hash verificados, escalas fisicas, nube separada
de validacion, Adam seguido de refinamiento de objetivo fijo, semillas y auditoria global/regional.
No es transferible sin redisenar la representacion la receta de poner todos los pesos salvo PDE
en cero. La continuidad de temperatura y flujo debe estar controlada ANTES de eliminar esas losses.

| Modelo actual | Reutilizacion con sentido | Obstaculo para PDE-only | Prioridad razonada |
|---|---|---|---|
| M1 / M1_EXACT | Mantener M1 por penalizacion como control; M1_EXACT ya es su ruta de transmision exacta | No quitar interfaces al M1 original sin el levantamiento | Alta: referencia verificada en MMS, diagnostico local para cables |
| DeepXDE PFNN | `DeepXDE_PFNN_EXACT`: PFNN real como corrector dentro del levantamiento compartido | La PFNN historica sigue por penalizaciones; la nueva variante conserva ramas propias y agrega trazas exactas | Implementada y aceptada en MMS con init=0, sampling=11 |
| DeepXDE FNN | `DeepXDE_FNN_EXACT`: tronco/cabezas regionales como corrector exacto | Mismos 32x3 no implican mismos parametros que PFNN/M1 | Corrida completa aceptada en MMS con init=0, sampling=11 |
| DCSNN | `DCSNN_EXACT`: red compartida e indicador regional dentro de trazas/burbujas exactas | El one-hot historico no impone continuidad por si solo | Implementada y aceptada en MMS con init=0, sampling=11 |
| CUSP | `CUSP_EXACT`: features one-sided normalizadas, escalas locales y cabeza inicial reducida como corrector | El quiebre de gradiente historico no garantiza flujo continuo; ahora lo impone el levantamiento final | Corrida completa aceptada en MMS con init=0, sampling=11 |
| FBPINN | `FBPINN_EXACT`: mezcla de ventanas como corrector, ANTES del levantamiento final regional | El blend suave solo no representa el salto; el campo final regional ya puede hacerlo sin separar cada ventana por material | Corrida completa aceptada en MMS con init=0, sampling=11 |
| M2 | Mantener el control mixto con ecuacion constitutiva y balance de sus escalas; estudiar una parametrizacion exacta del flujo aparte | Con q independiente, div(q)=Q no determina T al retirar q+k*grad(T)=0; desactivar la loss constitutiva deja temperatura sin determinar | No trasladar PDE-only directamente; su redisenio cambia la formulacion |
| M0 global suave | `M0_EXACT`: corrector global suave dentro del campo final regional exacto | El M0 historico sigue C1; el nuevo campo final no depende solo de esa red global | Corrida completa aceptada en MMS con init=0, sampling=11 |

`DeepXDE_EXACT` conserva el corrector M1 original. `DeepXDE_PFNN_EXACT` y `DeepXDE_FNN_EXACT`
usan las clases PFNN/FNN reales de DeepXDE; no son alias del corrector M1. Todas reutilizan el
levantamiento exacto, porque la transmision es una propiedad del campo final, no del motor.
Los seis nombres nuevos estan disponibles por CLI y front-end. Los modelos historicos siguen
por penalizacion; sus corridas fallidas NO se relabelan ni se reutilizan como aceptadas.

### Implementacion y primera validacion del 2026-10-20

Se conserva la forma T=levantamiento+burbuja*escala*corrector y se sustituye SOLO el corrector.
Se eliminan del modelo los MLP/offsets regionales M1 reemplazados, para no registrar parametros
inactivos. Se mantienen las trazas compartidas y las redes auxiliares de frontera. El optimizador
solo recibe PDE regional normalizada; no se usa preentrenamiento supervisado ni penalizaciones.

Normalizacion nueva: coordenadas base en [0,1]; indicador DCSNN en 0/1; features CUSP
`(tanh(sign*d/escala)+1)/2`. FBPINN conserva ventanas y subredes, pero usa coordenadas globales
afines [0,1] en cada subred: no recorta las coordenadas locales fuera de una ventana, lo que
romperia derivadas. PFNN/FNN usan bloques regionales normalizados y enmascarados. Los encoders
historicos mantienen su significado y no se modifica un checkpoint antiguo para reinterpretarlo.

Resultados medidos de la receta 400 Adam + 5 L-BFGS, misma nube de 1024 puntos PDE:
las seis variantes pasan TODAS las raw_gates del MMS. NRMSE: DCSNN=0.098462%,
PFNN=0.064856%, M0=0.042050%, CUSP=0.061693%, FBPINN=0.082636%, FNN=0.092434%.
Sus tiempos registrados son, respectivamente, 477.94, 862.83, 1054.98, 767.43, 1367.28 y
1045.17 s; no son un benchmark controlado. Ver el [cuadro completo, auditoria y rutas](results.md).
No se atribuye el exito a un factor aislado: se transfirio
el conjunto normalizacion+levantamiento+PDE-only+refinamiento; no es una ablacion factorial.

Pruebas: transmision con trazas no nulas, gradientes PDE hacia los correctores, entradas [0,1]
en MMS/cable, entrenamiento corto, recarga de pares y preflight MMS/annulus/cables. La regresion
general no lenta/no FEM aprobo 379 pruebas, omitio una y excluyo diez, incluido el test historico
de hash ya conocido. Los SEIS pares finales aislados reproducen 6000 temperaturas
con diferencia maxima 0 K y flujo finito. Nativo conserva continuacion completa;
DeepXDE conserva solo inferencia/warm start, no resume exacto del optimizador.

Una corrida aceptada por familia no demuestra robustez multisemilla ni convergencia de cables.
M0/CUSP/FBPINN/FNN ya tienen corrida completa; esta evidencia reemplaza su anterior estado
de solo validacion funcional, sin ampliar la aceptacion a casos o semillas no ejecutados.

### Respaldo local de los limites de transferencia

El 2026-10-20 se agregaron y aprobaron tres comprobaciones en
[test_phase5_variants.py](../tests/test_phase5_variants.py), sin entrenar ni alterar las formulas
de los modelos historicos. Usan MMS heterogeneo, seed=0, float64, ancho 12 y profundidad 2:

- M0 y FBPINN: derivadas normales iguales a ambos lados de la interfaz, pero flujos normales
    distintos por el salto de conductividad. El campo suave actual no impone la transmision exigida.
- M2: al retropropagar SOLO `parts['pde']`, offsets y filas de salida de temperatura tienen
    gradiente nulo/ausente, mientras las filas de flujo reciben gradiente. La ecuacion constitutiva
    no es una penalizacion prescindible si T y q siguen siendo salidas independientes.

Son diagnosticos estructurales en pesos iniciales, no resultados de convergencia ni pruebas de
que cualquier red entrenada fracasara. Junto con las pruebas existentes de continuidad y quiebre
de CUSP, justificaron incorporar las trazas exactas en vez de repetir los modelos sin cambios.

### Comprobaciones que condicionan una nueva implementacion

1. Usar el MISMO JSON MMS con Q no nula y salto de k; comprobar signos de fuente y normales,
    T y q.n compartidos con flujo no nulo y gradientes normales distintos. Repetir con fuente nula.
2. Aplicar las restricciones a la salida FINAL del campo, incluidos offsets, escalas y ventanas.
    En FBPINN no basta con restringir cada subred y despues mezclarla con pesos variables.
3. Probar derivadas fisicas hasta orden 2 y autograd antes de entrenar; rechazar features que
    ocultan discontinuidades o recortan gradientes. Conservar/versionar los encoders de checkpoints.
4. Mantener nube y validacion independientes. En MMS, uniformes por material; en cables,
    auditar cobertura de cada capa/collar y solo entonces considerar puntos residuales adicionales.
5. Comparar una modificacion cada vez: receta Adam->L-BFGS, encoder, trazas o sampler.
    Registrar puntos/iteraciones/tiempo/memoria; no igualar 5 L-BFGS a 5 Adam. Cambiar formulacion,
    numero de parametros y muestreo juntos impide atribuir una mejora.
6. Exigir todas las compuertas y al menos tres inicializaciones; despues variar nube/caso y
    pasar a aras_single con su FEM propio antes de aras_flat. No reutilizar potencias/referencias.

Atencion RBA/SA y dobles filas siguen como ablations, no como ingrediente obligatorio: en el MMS
probado el control Adam->L-BFGS sin atencion fue mejor. Las filas/anillos son PDE interiores a
ambos lados de una superficie, no condiciones de flujo. En cables no se justifican 20k pasos
por repetir una receta que siga a temperatura ambiente o con balances grandes.

Float64 sigue como control de derivadas/alto contraste. Float32 se estudia por modelo/caso y
con auditoria en el dtype declarado; pasar en este MMS no prueba precision de flujo en cables.
Las versiones historicas CUSP/DCSNN/FBPINN conservan sus encoders. Las nuevas `*_EXACT`
usan el encoder implementado y verificado arriba; no basta con renombrar un JSON historico.

## Implementar un modelo nuevo

1. Escribe una clase en `pinn_v2/solvers/pinn/variants.py` que implemente `field(region, xy)`,
   `temperature(xy, labels)` y `parameter_count()` (mira `PINNModel` en `networks.py`).
2. Regístrala en el diccionario `VARIANTS` con el nombre que usarás en `--model`.
3. Corre `pytest tests -q`; los tests de las fases 4 y 5 sirven de plantilla.

Las variantes ordinarias pueden registrarse en `VARIANTS`. `M1_EXACT` también reutiliza el ciclo
común, pero declara su política de pérdida y las condiciones geométricas compatibles explícitamente.