# Entrenamiento

## Comando completo

```bash
python -m pinn_v2 train --case xlpe_single --model M1 --adam-steps 6000 --lbfgs-steps 100 --lr 3e-3 --width 32 --depth 3 --activation tanh --seed 0 --sampling-seed 11 --weights dirichlet=100 interface_flux=10 --points n_interior=512 n_layer=256 n_face=128 n_boundary=128 --output results/runs/xlpe_m1
```

## Parámetros

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

## Qué se guarda

| Archivo | Contenido |
|---|---|
| `result.json` | configuración, semillas, historia, pérdidas finales, evaluación y auditoría |
| `evaluation.npz` | puntos de evaluación, temperatura predicha y, si hubo referencia, la referencia |
| `collocation.npz` | puntos de colocación interiores usados |
| `model.pt` | pesos de la red |
| `manifest.json` | versiones, huellas, semillas y comando |
| `failure.json` | solo si la corrida falló; la historia queda registrada |

## Semillas

Hay tres semillas independientes y se guardan siempre:

- `init_seed`: pesos de la red.
- `sampling_seed`: puntos de colocación del entrenamiento.
- semilla de auditoría (909091): puntos de la evaluación; el entrenamiento no la usa.

Con las mismas tres y el mismo código, la corrida se repite en el mismo equipo.

## Consejos prácticos

- Para depurar, usa pocos pasos (`--adam-steps 200`) y un caso pequeño como `mms_constant`.
- Casos con muchas capas (Kim, 55 regiones) son pesados: reduce `n_layer`, `n_face` y `n_boundary`.
- Una pérdida baja no garantiza una buena temperatura. Mira la auditoría: saltos en interfaces y balance de energía.