# Kim et al. (2024) benchmark

Este directorio contiene tres variantes del benchmark de 6 cables XLPE 154 kV (two-flat):

- `run_example.py`: modelo base con suelo efectivo homogéneo.
- `run_research_pac.py`: zona PAC explícita con `k(x,y)` variable.
- `run_multilayer.py`: suelo multicapa + modelo de cable 9 capas (CLSM/PE), casos A/B.

## Perfiles de ejecución

- `quick`: configuración reducida para validación rápida.
- `research`: configuración extendida para mayor precisión.

## Comandos

```bash
python examples/kim_2024_154kv_bedding/run_example.py --profile quick
python examples/kim_2024_154kv_bedding/run_example.py --profile research

python examples/kim_2024_154kv_bedding/run_research_pac.py --profile quick
python examples/kim_2024_154kv_bedding/run_research_pac.py --profile research

python examples/kim_2024_154kv_bedding/run_multilayer.py --profile quick
python examples/kim_2024_154kv_bedding/run_multilayer.py --profile research
```

## Condición lateral dependiente de profundidad

El perfil lateral se define en `data/boundary_profiles.csv` y se aplica a las fronteras `left/right`.

Nudos usados (piecewise linear):

- z = [0.0, 0.5, 1.0, 1.4, 2.0, 3.0, 3.6] m
- T = [26.1, 23.7, 21.5, 19.9, 18.0, 15.8, 15.2] °C

## Cuadro comparativo (todos los modelos evaluados con `eval_all.py`)

| Script | Perfil | T_PINN (°C) | T_FEM ref (°C) | Error | Loss final |
|--------|--------|-------------|----------------|-------|------------|
| `run_example.py` | quick | 77.6 | 70.6 (PAC) | +7.0 K | 5.2039e+00 |
| `run_example.py` | research | 66.0 | 70.6 (PAC) | −4.6 K | 5.4564e+00 |
| `run_research_pac.py` | quick | 72.9 | 70.6 (PAC) | +2.3 K | — |
| `run_research_pac.py` | research | 69.0 | 70.6 (PAC) | −1.6 K | 7.8321e-02 |
| `run_multilayer.py` Case A | quick | 77.4 | 77.6 (sand) | −0.2 K | — |
| `run_multilayer.py` Case B | quick | 69.2 | 70.6 (PAC) | −1.4 K | — |
| `run_multilayer.py` Case A | research | 72.1 | 77.6 (sand) | −5.5 K | — |
| `run_multilayer.py` Case B | research | 63.2 | 70.6 (PAC) | −7.4 K | — |

Referencias FEM Kim 2024 verano: sand 77.6 °C, PAC 70.6 °C.

> **Mejores resultados:** `run_multilayer.py Case A quick` (−0.2 K vs sand) y
> `run_research_pac.py research` (−1.6 K vs PAC).
> Los modelos `research` multicapa presentan sobrecompensación; se recomienda
> revisar la tasa de aprendizaje o el número de pasos L-BFGS.
