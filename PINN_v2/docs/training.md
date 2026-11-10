# Entrenamiento

## Comando completo

```bash
python -m pinn_v2 train --case xlpe_single --model M1 --adam-steps 6000 --lbfgs-steps 100 --lr 3e-3 --width 32 --depth 3 --activation tanh --seed 0 --sampling-seed 11 --sampling interface --resample-every 500 --weights dirichlet=100 interface_flux=10 --points n_interior=512 n_layer=256 n_face=128 n_boundary=128 --output results/runs/xlpe_m1
```

Modo optativo con Dirichlet e interfaces impuestos por construcción, y objetivo PDE-only:

```powershell
python -m pinn_v2 train --case aras_flat --model M1_EXACT --adam-steps 20000 --lbfgs-steps 200 --reference results/phase6/p6_fem3/aras_flat/base/fem_l2.npz --output results/runs/aras_flat_exact
```

`M1_EXACT` admite cables flat/trefoil no solapados con suelo homogéneo y Dirichlet constante,
MMS rectangulares con interfaces planas y el benchmark annulus CIGRE. Annulus admite condiciones
Dirichlet/Neumann/Robin en ambos radios; MMS admite esos tipos con esquinas compatibles. Cables
requieren fuente fija, contacto perfecto y Dirichlet constante. La CLI rechaza combinaciones no
compatibles; no cambia silenciosamente al modo `M1`. El argumento `--weights` puede fijar el peso
`pde`; cualquier otro peso solicitado queda en cero en el objetivo efectivo. El FEM de referencia
sigue siendo solo para evaluación.

## Parámetros

### Receta exacta aceptada y fechada

La [ficha de reproducibilidad del 2026-10-20](reproducibility.md) registra la configuracion
aceptada de `mms_interface` para M1_EXACT y DeepXDE_EXACT: 400 Adam + 5 L-BFGS, 32x3/tanh,
LR 0.003 con coseno, puntos interiores fijos 512 por region, sin atencion y solo PDE.
Incluye los conteos efectivos, entorno/hilos, comandos completos, semillas y rutas historicas.
La configuracion ejecutable es
[train_mms_exact_repro_20261020.json](../configs/train_mms_exact_repro_20261020.json).
No usarla con modelos por penalizacion suprimiendo sus condiciones de frontera/interfaz.

Tras la transferencia implementada el 2026-10-20, tambien admite los modelos nuevos
M0_EXACT, CUSP_EXACT, DCSNN_EXACT, FBPINN_EXACT, DeepXDE_PFNN_EXACT y DeepXDE_FNN_EXACT.
Cambiar `--model` selecciona su corrector propio dentro del levantamiento exacto; no transforma
una corrida historica por penalizaciones. DCSNN_EXACT y DeepXDE_PFNN_EXACT ya tienen una corrida
MMS aceptada con esta receta; al cierre del 2026-10-20 tambien M0_EXACT, CUSP_EXACT,
FBPINN_EXACT y DeepXDE_FNN_EXACT completaron 400 Adam + 5 L-BFGS y pasaron todas las raw_gates.
El cuadro de resultados incluye las SEIS corridas, sus tiempos y rutas; no son solo smoke tests.
Ver [resultados y limites](results.md) antes de extrapolar a cables o a otras semillas.

### Opciones generales

### Parada conjunta y extension L-BFGS (2026-10-20)

En el JSON de entrenamiento, `accuracy_stop` habilita evaluacion independiente para detenerse:

```json
{
  "accuracy_stop": {
    "error_Tmax_abs_K": 0.01,
    "nrmse_pct": 0.1,
    "balance_pct": 0.5,
    "every": 100,
    "after_phase": "any"
  },
  "extra_lbfgs_steps": 5
}
```

Los TRES umbrales deben cumplirse simultaneamente con comparacion estricta `<`, sobre el mismo
estado. `after_phase` puede ser `adam`, `lbfgs` o `any`; `any` comprueba los registros Adam con
la cadencia indicada y despues de CADA llamada externa L-BFGS. No equivale a comprobar cada
iteracion interna ni cada evaluacion de line search.
Si tras las 5 llamadas base no se cumple el criterio, se permiten hasta5 llamadas adicionales,
con el MISMO optimizador, historial, pesos y puntos. No hay reinicio ni ampliacion indefinida.
Se conserva el estado que detiene el entrenamiento o el ultimo al agotar el presupuesto,
no se lo sustituye por el de menor loss. `accuracy_stop` en result.json registra las metricas,
el motivo `accuracy_targets_met`/`budget_exhausted` y las llamadas base/adicionales ejecutadas.

Sin `accuracy_stop`, el comportamiento historico no cambia. Con referencia FEM, esta se usa
para evaluar/detener y NO en la loss PDE: debe declararse `reference_used_for_stopping=true`.
La auditoria final de interfaces y de TODOS los balances regionales sigue siendo independiente
de estos tres objetivos de parada. Un campo con objetivos logrados no se promueve si falla la
auditoria final. DeepXDE conserva su limite de inferencia/warm start, no resume del optimizador.

### Opciones generales de la CLI

| Opción | Valor por defecto | Significado |
|---|---|---|
| `--case` o `--definition` | uno obligatorio | id del catálogo o ruta de un JSON |
| `--model` | `M1` | ver [Modelos](models.md) |
| `--adam-steps` | 3000 | pasos de Adam con decaimiento coseno |
| `--lbfgs-steps` | 0 | pasos de L-BFGS al final (cada uno con 20 iteraciones internas) |
| `--lr` | 0.002 | tasa de aprendizaje inicial de Adam |
| `--width`, `--depth` | 32, 3 | neuronas por capa y número de capas ocultas |
| `--activation` | `tanh` | `tanh` o `silu` |
| `--seed` | 0 | semilla de los pesos |
| `--sampling-seed` | 11 | semilla de los puntos de colocación |
| `--sampling` | `mixed` | `uniform`, `mixed` o `interface`; la última reserva la mitad de los puntos interiores a bandas próximas a interfaces |
| `--resample-every` | 0 | renueva los puntos interiores cada N pasos Adam; 0 deja fija la nube. Las cuadraturas de caras/fronteras permanecen fijas |
| `--deepxde-sampling` | `mixed` | Para `--model DeepXDE`: `pseudo`, `LHS`, `Halton`, `Hammersley` o `Sobol` genera anchors por región con los samplers de DeepXDE; `mixed` conserva el sampler previo del proyecto |
| `--weights` | ver tabla de pérdidas | pares `termino=valor` |
| `--points` | 512, 256, 128, 128 | `n_interior` (suelo), `n_layer` (capas), `n_face`, `n_boundary` |
| `--reference` | ninguno | NPZ con `xy` y `T`; solo para evaluar |
| `--output` | `results/runs/<caso>_<modelo>` | carpeta nueva; si existe, se rechaza |

## Archivo de configuración

El mismo contenido se puede guardar en un JSON (ver `configs/train_m1.json`):

```json
{
  "case": "xlpe_single",
  "model": "M1",
  "adam_steps": 6000,
  "lbfgs_steps": 100,
  "lr": 0.003,
  "width": 32,
  "depth": 3,
  "weights": {"dirichlet": 100, "interface_flux": 10},
  "points": {"n_interior": 512, "n_layer": 256}
}
```

Orden de prioridad: valores por defecto, luego el archivo, luego las banderas.
Para modo exacto, usa `"model": "M1_EXACT"`; el registro normal guardará además el `loss_policy`
efectivo, el mapa de pesos PDE-only y la versión del encoder en `input_normalization`. En cables,
las coordenadas cartesianas y las distancias logarítmicas suelo-cable nuevas se escalan a `[0,1]`
con `unit_box_v1`. Los modelos anteriores conservan sus encoders `[-1,1]` o legacy.
Los configs/checkpoints históricos que no contienen `input_normalization` usan el encoder legacy al
reconstruirse; los nuevos checkpoints conservan el encoder normalizado registrado.

## Qué se guarda

| Archivo | Contenido |
|---|---|
| `result.json` | configuración, semillas, historia, pérdidas finales, evaluación y auditoría |
| `evaluation.npz` | puntos de evaluación, temperatura predicha y, si hubo referencia, la referencia |
| `collocation.npz` | puntos de colocación interiores usados |
| `model.pt` | bundle versionado: pesos para inferencia y estado completo del ultimo checkpoint de entrenamiento |
| `model.json` | especificacion inmutable: problema fisico completo, arquitectura, dtype, normalizacion, escalas y configuracion |
| `manifest.json` | versiones, huellas, semillas y comando |
| `failure.json` | solo si la corrida falló; la historia queda registrada |

## Inferencia y continuacion portatiles

### DeepXDE exacto con el mismo campo

```powershell
python -m pinn_v2 train --case mms_interface --model DeepXDE_EXACT --adam-steps 400 --lbfgs-steps 5 --lr 0.003 --width 32 --depth 3 --dtype float64 --device cpu --output results/runs/mms_deepxde_exact64
```

Para comparar precision numerica, repetir en otra carpeta con `--dtype float32`. Se conservan
fuente, materiales, semillas y puntos; no asumir aceptacion ni aceleracion hasta revisar las
auditorias. La inicializacion del campo exacto se realiza en float32 y se convierte al dtype
pedido, igual que el trainer nativo, para no cambiar tambien los pesos iniciales del control CPU.

`DeepXDE_EXACT` mantiene loss PDE-only, sin restricciones `PointSetOperatorBC`. Sus pasos
L-BFGS son externos con hasta 20 iteraciones internas. No confundirlos con el presupuesto total
de iteraciones internas usado por el DeepXDE por penalizacion historico.

Esta ruta exporta un par autosuficiente para inferencia y comienzo desde pesos, pero no tiene
estado de continuacion exacta del optimizador; `resume_available=false` lo declara. El dtype
y dispositivo global se restauran despues de la corrida. El backend debe ser PyTorch.

### Contrato del par nativo

Solo `model.pt` y `model.json`, con el mismo nombre base, son datos necesarios para reconstruir
un modelo nativo. Se requiere el paquete compatible instalado, no el catalogo, FEM, NPZ ni otros
archivos de resultados. El JSON queda vinculado al bundle mediante SHA256; pares mezclados o
versiones desconocidas se rechazan. DeepXDE por penalizacion conserva su formato previo;
DeepXDE_EXACT comparte la inferencia portable pero no la reanudacion del optimizador nativo.

```powershell
python -m pinn_v2 infer --checkpoint results/runs/mi_modelo/model.pt --xy 0 -1.2 --flux
python -m pinn_v2 train --resume-from results/runs/mi_modelo/model.pt --output results/runs/continuado --device cpu
```

El bundle conserva pesos, momentos Adam/historial L-BFGS, scheduler, RNG, puntos, fuentes,
atencion, remuestreo y proximo paso. Distingue el estado seleccionado para inferencia del ultimo
estado de entrenamiento. Se guarda despues de actualizaciones completas en los limites de registro;
un corte brusco pierde pasos aun no guardados. La reanudacion mantiene horizonte, dtype, encoder,
fisica, dispositivo y version PyTorch. Cambiar esos contratos exige un inicio desde pesos explicito.
Los checkpoints historicos de solo pesos no contienen los momentos que permitirian reanudar.

La seleccion de mejor entrenamiento ahora compara estados post-actualizacion registrados, no
copias en cada mejora de la loss. `checkpoint_policy=post_update_logged_v2` identifica esta politica.
`optimized_pde` registra el residuo ponderado por steering; `total` sigue siendo el score no ponderado.

Opciones nuevas: `--dtype float32|float64`, `--input-normalization unit_box_v1`,
`--trace-parameterization legacy|conditioned_v1|conservative_v1` y `--soil-envelope-gain`.
Las dos ultimas opciones requieren geometria circular de cables y son experimentales. Float64
permanece por defecto; mostrar decimas de grado no controla el error de segundas derivadas o flujo.
No se recorta la temperatura ni se fuerza su salida a `[0,1]`.

`sampling_version=balanced_cables_v2` reparte candidatos proximos entre los cables del JSON;
las versiones antiguas podian concentrar un lote entero en un solo cable. La mitad uniforme
se conserva. Las nubes efectivas se incluyen en el bundle para una continuacion reproducible.

## Semillas de entrenamiento

Hay tres semillas independientes y se guardan siempre:

- `init_seed`: pesos de la red.
- `sampling_seed`: puntos de colocación del entrenamiento.
- semilla de auditoría (909091): puntos de la evaluación; el entrenamiento no la usa.

Con las mismas tres y el mismo código, la corrida se repite en el mismo equipo.

## Consejos prácticos

- Para depurar, usa pocos pasos (`--adam-steps 200`) y un caso pequeño como `mms_constant`.
- Casos con muchas capas (Kim, 55 regiones) son pesados: reduce `n_layer`, `n_face` y `n_boundary`.
- Si se remuestrea, el mejor `train_loss` se compara solo dentro de la última nube fija; `best_validation` usa una nube independiente fija para comparar toda la corrida.
- Una pérdida baja no garantiza una buena temperatura. Mira la auditoría: saltos en interfaces y balance de energía.