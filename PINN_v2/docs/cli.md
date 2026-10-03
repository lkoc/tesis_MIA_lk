# Referencia de la línea de comandos

Un solo punto de entrada: `python -m pinn_v2 <comando>`. Los textos de ayuda integrados se leen con
`python -m pinn_v2 help` y `python -m pinn_v2 help <tema>`; los temas son overview, cases, models, train,
search, fem y files.

En PowerShell, antes de empezar:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

## Comandos

| comando | para qué sirve |
|---|---|
| `help [tema]` | ayuda integrada |
| `cases` | lista los casos disponibles |
| `show` | resume un caso |
| `validate` | revisa un archivo de caso; devuelve 1 si no es válido |
| `train` | entrena un PINN |
| `fem` | corre la referencia FEniCSx en WSL2 |
| `search` | búsqueda de hiperparámetros |

## Cómo elegir el caso

Todos los comandos aceptan lo mismo:

- `--case NOMBRE`: un id del catálogo (por ejemplo `xlpe_single`, `kim_v5_summer` o `annulus`).
- `--definition RUTA`: un JSON propio, en formato v2 o en el formato anterior.

## train

```powershell
python -m pinn_v2 train --case xlpe_single --model M1 --adam-steps 6000 --lbfgs-steps 100
```

Opciones: `--model`, `--adam-steps`, `--lbfgs-steps`, `--lr`, `--width`, `--depth`, `--activation`,
`--seed`, `--sampling-seed`, `--output`, `--config`, `--reference`, `--weights nombre=valor ...`
y `--points nombre=valor ...`.

La prioridad es: valores por defecto, luego el archivo `--config`, luego las banderas.
Si la carpeta de salida ya existe, el comando se rechaza.

## fem

```powershell
python -m pinn_v2 fem --case xlpe_single --levels 0 1 2
python -m pinn_v2 fem --case annulus --levels 0
```

Opciones: `--levels`, `--mode {fixed,dc_temperature}`, `--output`. Con `fixed` la fuente es la misma
que usa el PINN para casos de cable. Casos annulus/MMS usan el solver FEM de verificación. Necesita
WSL2; ver [Referencia FEM](fem.md).

## search

```powershell
python -m pinn_v2 search --case xlpe_single --config configs/search_example.json --trials 8
```

Opciones: `--trials`, `--strategy {random,grid}`, `--seed`, `--output`. Detalles en
[Búsqueda de hiperparámetros](search.md).

## Salidas por defecto

- `results/runs/<caso>_<modelo>` para `train`.
- `results/runs/fem_<nombre>` para `fem`.
- `results/runs/search_<nombre>` para `search`.