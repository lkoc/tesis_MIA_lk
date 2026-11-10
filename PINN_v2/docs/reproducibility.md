# Reproducibilidad

Cada corrida deja suficiente información para repetirla y para saber qué código la produjo.

## Qué se guarda

En la carpeta de salida de un entrenamiento hay:

| archivo | contenido |
|---|---|
| `result.json` | configuración usada, historial de pérdida, evaluación y auditoría |
| `model.pt` | bundle de pesos y, en los modos nativos compatibles, estado de entrenamiento |
| `model.json` | especificacion fisica y numerica inmutable para el par portable exacto |
| `collocation.npz` | puntos de colocación usados |
| `evaluation.npz` | campo predicho y, si hubo referencia, el campo de referencia |
| `manifest.json` | comando, semillas, versiones, estado de git y huellas sha256 |
| `source/` | copia de los fuentes tal como estaban al correr |

El manifest guarda el comando exacto, las semillas, las versiones de Python, NumPy, SciPy y PyTorch,
el commit de git y si la carpeta `PINN_v2` tenía cambios sin confirmar (`dirty`).

## Semillas

- `init_seed`: inicialización de la red.
- `sampling_seed`: puntos de colocación del entrenamiento.
- Auditoría: semilla fija 909091, que el entrenamiento nunca usa.
- Búsqueda de hiperparámetros: semilla fija 424243 para la pérdida held-out.

Las mismas semillas son necesarias, no una garantia universal de igualdad bit a bit. Hay que
conservar tambien fuentes, versiones, dtype, dispositivo, configuracion y estrategia de muestreo.
La receta CPU/float64 de abajo reprodujo exactamente la loss del control nativo con semilla 0.

## Huellas del caso

Un caso tiene dos huellas: `sha256` del archivo completo y `physics_sha256` solo de la parte física.
Si cambias el título o las notas, cambia la primera pero no la segunda. Úsala para comprobar que dos
corridas resolvieron la misma física.

## Nada se sobrescribe

- Una carpeta de salida que ya existe se rechaza.
- Los JSON se escriben de forma atómica y no reemplazan un archivo existente.
- Los estudios fallidos se conservan; no se borran.

## Repetir una corrida

1. Abre `manifest.json` y copia el campo `command`.
2. Usa una carpeta de salida nueva.
3. Compara `physics_sha256` y las semillas con las de la corrida original.

Si el experimento se lanzo mediante API, `command` puede ser un identificador del modulo y no
un comando CLI completo. No ejecutar ciegamente esa linea: contrastar `config`, codigo archivado
y la receta explicitada abajo. Un commit con `dirty=true` tampoco identifica todos los fuentes;
`manifest.source_sha256` y `source/` son la evidencia del codigo realmente usado.

## Receta aceptada MMS exacto: ficha del 2026-10-20

Alcance: `mms_interface`, dominio unitario 2D, interfaz vertical x=0.5, conductividades 0.5 y 2,
fuente manufacturada no nula y contacto perfecto. El caso es
`pinn_v2/cases/v2/references/verification/mms_interface.json`. Su hash fisico es
`3924c8005dbe614fcce5b4299054aef1de4652ba40d3060e48e764ef193f34d5`.
La solucion analitica prescribe los datos de frontera y sirve para evaluar, no como etiquetas
interiores de entrenamiento. El maximo continuo exacto es 32.5 C en una frontera; el maximo
de la nube de evaluacion es 32.483638668 C. No confundir esos dos valores.

### Parametros e hiperparametros efectivamente usados

| Concepto | M1_EXACT | DeepXDE_EXACT |
|---|---|---|
| Motor | PyTorch nativo | DeepXDE, backend PyTorch |
| Campo aproximante | Redes regionales y trazas exactas compartidas | El mismo campo envuelto en `dde.nn.NN`, no PFNN/FNN |
| MLP regional y auxiliar | 3 capas ocultas, 32 neuronas/capa, tanh, salida escalar | Igual |
| Inicializacion | Xavier normal, biases/offsets/trazas inicialmente cero; construccion float32 y conversion al dtype | Misma construccion del campo |
| Traza de interfaz MMS | Legendre grados 0 a 16, 17 coeficientes por T y por flujo; `fourier_order=8` es el nombre interno | Igual |
| Normalizacion de entradas | `unit_box_v1`, [0,1], conservando derivadas respecto a metros | Igual |
| Temperatura de referencia / escala declarada | 20 C / 30 K; sin recorte de temperatura | Igual |
| Parametrizacion / ganancia del suelo | `legacy` / 1.0; la ganancia radial no interviene en MMS | Igual |
| Precision aceptada | float64 | float64 y float32, brazos separados |
| Dispositivo / hilos de calculo | CPU Windows / 1 hilo PyTorch | Igual |
| Optimizador inicial | Adam, 400 actualizaciones, LR inicial 0.003 | Igual |
| Defaults de Adam no sobrescritos | betas=(0.9,0.999), eps=1e-8, weight_decay=0 | Igual, optimizador PyTorch |
| Scheduler | Coseno, horizonte 400, LR final 0.00003 | Coseno equivalente mediante DeepXDE |
| Refinamiento | 5 llamadas externas L-BFGS, max_iter=20, LR=1, history_size=50, strong_wolfe | Mismo optimizador PyTorch usado por DeepXDE |
| Defaults de L-BFGS no sobrescritos | tolerance_grad=1e-7, tolerance_change=1e-9; line search puede reevaluar la closure | Igual |
| Seleccion de estado | Menor loss PDE registrada post-actualizacion | Menor loss PDE registrada por DeepXDE |
| Registro | Cada 100 pasos Adam y cierre; tambien refinamiento | `display_every=100`, callbacks y cierre/refinamiento |
| Atencion / remuestreo | Ninguna / 0, nube fija | Ninguna / no dinamico en esta ruta |
| Loss efectiva | `pde=1`; todos los otros pesos=0 | Igual; lista de restricciones vacia |
| Referencia en el optimizador | Ninguna; `fem_labels_used=0`, `reference_used_in_loss=false` | Igual |

La loss es la suma de las medias regionales del residuo `-div(k grad T)-Q`, al cuadrado y
normalizado por `problem.scales(region)['divergence']`. Para este MMS las escalas de divergencia
son k*30: 15 y 60. No se agregan perdidas Dirichlet, Neumann, Robin, interfaz, flujo o energia.
Las condiciones compatibles se satisfacen en cada forward mediante el levantamiento, no
mediante proyecciones del optimizador ni porque se hayan muestreado puntos sobre la superficie.
El balance se mide despues del entrenamiento. Finitud y trazas exactas no certifican por si
solas precision de la solucion.

### Colocacion: conteos reales y separacion de usos

| Uso | Conteo y estrategia en el MMS aceptado |
|---|---|
| PDE de entrenamiento | 512 interiores por cada una de 2 regiones: 1024 puntos totales |
| `n_layer=256` | Registrado, pero no usado en MMS: ambas regiones son de tipo soil, no capas de cable |
| `sampling=mixed` | En `MMSProblem` es uniforme pseudorrandom por region; NO mezcla radial de cable |
| Semillas regionales | sampling_seed + 7919*region.id: 11 y 7930 |
| Distribucion espacial | Uniforme en [0,0.5]x[0,1] y [0.5,1]x[0,1], nube fija |
| Interfaz registrada | `n_face=128`: 128 puntos medios equiespaciados, no loss de interfaz |
| Frontera registrada | `n_boundary=128` por lado: 512 puntos externos totales, repartidos entre 6 piezas material/lado |
| Total de la nube guardada | 1024 interiores + 128 interfaz + 512 frontera = 1664; SOLO 1024 alimentan la loss exacta |
| DeepXDE | Mismos anchors regionales; `deepxde_sampling=mixed`, num_domain=0, num_boundary=0; sin QMC adicional |
| Validacion independiente | 512 interiores por region, semilla 1000014, fija y sin atencion |
| Cadencia de validacion | Nativo: validation_every=100 y final; DeepXDE: score independiente final, no afirmar validacion independiente cada callback |
| Auditoria fisica | Semilla 909091; 2048 interiores/region = 4096; 512/interfaz y 512/lado = 2560 de cuadratura |
| Evaluacion de temperatura | Los mismos 6000 puntos y pesos del problema, separados del entrenamiento |
| Auditoria densa adicional nativa | 2048/interfaz y 2048/lado, sin reentrenar; las tres semillas siguen aceptadas |

No se usaron en la receta ganadora dobles filas/anillos, refinamiento residual, RBA/SA, LHS,
Halton, Hammersley ni Sobol. El nombre `balanced_cables_v2` queda en el registro compartido,
pero la correccion del muestreo cercano a cables no transforma el muestreo uniforme MMS.
En cables, `mixed` tiene otro significado; copiar solo su nombre no replica esta distribucion.

Comprobacion de artefactos del 2026-10-20: los tres `collocation.npz` originales tienen
interiores (512,2)+(512,2), interfaz (128,2) y seis piezas externas de 64,64,64,64,128,128
puntos. Todas las coordenadas archivadas son float64 e identicas entre M1_EXACT y ambos brazos
DeepXDE_EXACT, con diferencia maxima 0. Esto no implica aritmetica float64 en el brazo
float32: el dtype efectivo de tensores/parametros se aplica al calcular el campo y sus derivadas.

### Semillas, versiones y artefactos originales

- Control nativo y DeepXDE: init=0, sampling=11, validacion=1000014, auditoria=909091.
- Repeticion nativa: init=0/11/23, MISMA nube sampling=11; no prueba de nuevas semillas de colocacion.
- DeepXDE: una combinacion init/sampling por dtype; no declarar robustez multisemilla.
- Manifests historicos: Python 3.12.9, NumPy 2.3.4, SciPy 1.18.1, PyTorch 2.9.0+xpu,
	Windows 11. Se ejecuto en CPU aunque la distribucion PyTorch incluya XPU.
- DeepXDE instalado: 1.15.0, verificado el 2026-10-20; ese paquete no estaba enumerado en el
	campo host de los manifests originales. Conservar esta distincion de procedencia.
- Git registrado: commit `66d55d61f0870598186e1f15540b2be9247aad9f`, dirty=true.
	Para repetir codigo historico usar los hashes y copias de `source/`, no solo ese commit.

Rutas relativas al directorio `PINN_v2`:

| Resultado | Ruta original, no sobrescribir |
|---|---|
| M1_EXACT float64, control | `results/phase7/p7_mms_single_validation_20261019/mms_interface_adam400_lbfgs5/` |
| M1_EXACT float64, tres inicializaciones | `results/phase7/p7_mms_single_validation_20261019/mms_recommendations_3seeds_20261019/seed_0/`, `seed_11/`, `seed_23/` |
| DeepXDE_EXACT float64 | `results/phase7/p7_deepxde_exact_mms_20261019/control/` |
| DeepXDE_EXACT float32 | `results/phase7/p7_deepxde_exact_mms_20261019/float32/` |
| Comparacion FEM nivel 1 | `results/phase7/p7_deepxde_exact_mms_20261019/comparison_vs_fem_20261020.json` |

Cada corrida conserva result.json, manifest.json, model.pt/model.json, collocation.npz,
evaluation.npz y fuentes. Para inferencia basta el par y codigo compatible. La continuacion
exacta del optimizador existe en el trainer nativo; DeepXDE_EXACT solo declara inferencia/warm start.

### Comandos para replicar, sin sobrescribir resultados

La configuracion [train_mms_exact_repro_20261020.json](../configs/train_mms_exact_repro_20261020.json)
registra los parametros comunes; `--model` y `--dtype` seleccionan el brazo. Estas instrucciones
replican el experimento mediante la CLI publica; no afirman que fuera el comando literal de la API
historica. El JSON no se debe reutilizar en modelos por penalizacion poniendo sus condiciones a cero.

Preparacion en PowerShell, desde el mismo equipo; en otro equipo adaptar solo las rutas y
crear un entorno compatible con las versiones registradas:

```powershell
Set-Location C:\usr\ths_mia_fiis\tesis_MIA_lk\PINN_v2
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:DDE_BACKEND = 'pytorch'
```

Se verifico en un proceso nuevo el 2026-10-20 que OMP_NUM_THREADS=1 y MKL_NUM_THREADS=1
producen `torch.get_num_threads() == 1` en el interprete registrado.

```powershell
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2 train --config configs/train_mms_exact_repro_20261020.json --model M1_EXACT --dtype float64 --output results/runs/repro_mms_20261020_m1_exact64_s0
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2 train --config configs/train_mms_exact_repro_20261020.json --model DeepXDE_EXACT --dtype float64 --output results/runs/repro_mms_20261020_deepxde_exact64_s0
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2 train --config configs/train_mms_exact_repro_20261020.json --model DeepXDE_EXACT --dtype float32 --output results/runs/repro_mms_20261020_deepxde_exact32_s0
```

Para las otras inicializaciones nativas, repetir el primer comando con `--seed 11` o `--seed 23`
y carpetas nuevas terminadas en s11/s23; no cambiar `--sampling-seed`. Si una salida existe,
elegir otro nombre: no borrar la evidencia. La evaluacion original usa la solucion analitica,
sin `--reference`. Usar el campo FEM en otra evaluacion cambia la referencia y debe etiquetarse.

Verificar despues: status completed, mismo hash/config/semillas, pesos no PDE cero, campos
finitos y TODAS las raw_gates verdaderas. No basta con repetir Tmax. Los valores objetivo estan
en [Resultados](results.md), separados entre comparacion analitica y FEM; no prometer igualdad
bit a bit entre plataformas, dtypes o versiones distintas. Un tiempo unico no acredita aceleracion.

Para retomar la linea de investigacion con otros modelos, ver la
[matriz de transferencia](models.md): tras la aclaracion del usuario se implementaron seis
correctores exactos y se ejecutaron nuevas validaciones, separadas de las corridas historicas.
M2 no se convierte a PDE-only sin cambiar su formulacion mixta.

### Transferencia ejecutada a otros PINN, 2026-10-20

Se implementaron M0_EXACT, CUSP_EXACT, DCSNN_EXACT, FBPINN_EXACT,
DeepXDE_PFNN_EXACT y DeepXDE_FNN_EXACT. Todos usan normalizacion [0,1], levantamiento
exacto y PDE-only; sus correctores y cantidades de parametros son distintos, no renombres de M1.

Se ejecutaron SEIS corridas completas con la misma receta, fuente/hash, puntos, init=0 y
sampling=11. Se conservaron las dos iniciales DCSNN_EXACT/PFNN y se agregaron M0_EXACT,
CUSP_EXACT, FBPINN_EXACT y DeepXDE_FNN_EXACT. El [cuadro completo](results.md) incluye
parametros, errores, balances y el tiempo de cada corrida en segundos y minutos.
Se conservaron TODOS los hiperparametros de la ficha: 32x3/tanh, Adam400 LR0.003/coseno,
5 L-BFGS externos max_iter20, float64/CPU/un hilo, n_interior512 por region,
n_layer256 sin uso MMS, n_face128, n_boundary128 por lado, sin atencion/remuestreo/referencia
en loss. Las SEIS nubes archivadas se verificaron identicas al control M1_EXACT. Para las
cuatro variantes nativas la validacion independiente es periodica/final; PFNN/FNN conservan
la semantica final de DeepXDE. Los seis pares finales aislados reprodujeron las 6000 temperaturas
guardadas con diferencia maxima 0 K y flujo finito.

Rutas originales, relativas a PINN_v2:

- `results/phase7/p7_transfer_exact_20261020/DCSNN_EXACT_s0/`.
- `results/phase7/p7_transfer_exact_20261020/DeepXDE_PFNN_EXACT_s0/`.
- `results/phase7/p7_transfer_exact_20261020/M0_EXACT_s0/`.
- `results/phase7/p7_transfer_exact_20261020/CUSP_EXACT_s0/`.
- `results/phase7/p7_transfer_exact_20261020/FBPINN_EXACT_s0_complete/`.
- `results/phase7/p7_transfer_exact_20261020/DeepXDE_FNN_EXACT_s0/`.
- [Resumen verificado de seis](../results/phase7/p7_transfer_exact_20261020/comparison_all_six_20261020.json).
- [Resumen inicial de dos, conservado](../results/phase7/p7_transfer_exact_20261020/comparison_summary_20261020.json).
- [Regresion de integracion](../results/phase7/p7_transfer_exact_20261020/regression_suite_20261020.xml).

Repetir desde PINN_v2, con las variables de entorno anteriores y carpetas NUEVAS:

```powershell
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2 train --config configs/train_mms_exact_repro_20261020.json --model DCSNN_EXACT --dtype float64 --output results/runs/repro_mms_20261020_dcsnn_exact64_s0
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2 train --config configs/train_mms_exact_repro_20261020.json --model DeepXDE_PFNN_EXACT --dtype float64 --output results/runs/repro_mms_20261020_pfnn_exact64_s0
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2 train --config configs/train_mms_exact_repro_20261020.json --model M0_EXACT --dtype float64 --output results/runs/repro_mms_20261020_m0_exact64_s0
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2 train --config configs/train_mms_exact_repro_20261020.json --model CUSP_EXACT --dtype float64 --output results/runs/repro_mms_20261020_cusp_exact64_s0
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2 train --config configs/train_mms_exact_repro_20261020.json --model FBPINN_EXACT --dtype float64 --output results/runs/repro_mms_20261020_fbpinn_exact64_s0
..\.venv-benchmarks-repro\Scripts\python.exe -m pinn_v2 train --config configs/train_mms_exact_repro_20261020.json --model DeepXDE_FNN_EXACT --dtype float64 --output results/runs/repro_mms_20261020_fnn_exact64_s0
```

Las seis corridas completas pasaron todas las raw_gates; no confundirlas con los smoke previos
de 1 paso. No se reentrenaron los modelos historicos sin cambios; la aceptacion corresponde
a las variantes nuevas y a una sola semilla de inicializacion/colocacion por familia.

El intento `FBPINN_EXACT_s0` quedo interrumpido tras un checkpoint inicial y se conserva;
su duracion incompleta no se incorpora a la tabla. La corrida reportada se ejecuto de principio
a fin con la misma semilla en `FBPINN_EXACT_s0_complete`. En ese proceso se observo
OMP_NUM_THREADS=1 y MKL_NUM_THREADS no definido: se verifico en un proceso nuevo que ese
entorno da `torch.get_num_threads()==1`. Los comandos anteriores fijan ambas variables como
preparacion explicita para futuras replicas, sin afirmar que ambas se registraron en ese brazo.

Los tiempos son los `elapsed_s` originales, no la edad de las carpetas ni la duracion de un
smoke. Cubren entrenamiento/evaluacion hasta el punto de registro de cada solver, no todo el
proceso CLI con imports y archivado. Se comparan una vez por familia, con diferentes arquitecturas
y reevaluaciones L-BFGS; no acreditan una aceleracion controlada entre modelos.
La validacion final de la transferencia aprobo 379 pruebas no lentas/no FEM, con una omitida y
diez excluidas, incluido el conocido hash historico no relacionado. El servidor actualizado para
seleccionar las variantes esta en http://127.0.0.1:8877/; las instancias previas se conservaron.

Validacion inicial de esta ficha y configuracion, antes de implementar la transferencia,
2026-10-20: 114 pruebas aprobadas en los archivos
[test_cli_framework.py](../tests/test_cli_framework.py) y
[test_phase5_variants.py](../tests/test_phase5_variants.py), incluidas tres pruebas nuevas de
forwarding de la receta y tres comprobaciones de limites de transferencia. La documentacion
compila 14 paginas; persiste un aviso previo por un enlace historico al reporte M0 ausente.
Estas comprobaciones no repiten los entrenamientos historicos ni amplian su evidencia cientifica.

## Qué no entra al entrenamiento

### Conteos por corrida de la campana completa

El Excel y el Markdown incrementales de `p7_all_verification_exact_20261020` leen los puntos
de cada `collocation.npz`: interiores PDE + interfaces + frontera = total de colocacion.
Tambien informan coordenadas unicas. Son conteos de la nube fija, no puntos por numero de epocas.
Los conteos de validacion fija configurada, evaluacion y cuadratura de auditoria se muestran
aparte; para FEM se registran DoFs y evaluacion. Las filas sin archivo de colocacion no
se rellenan con el conteo nominal como si el muestreo ya hubiera ocurrido.

La referencia FEM y los datos de artículos se pasan solo a la evaluación. El resultado dice
`reference_used_in_loss: false` y `fem_labels_used: 0`.