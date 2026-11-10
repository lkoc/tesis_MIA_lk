# Benchmark Kim, Cho & Choi (2024) — 6 cables XLPE 154 kV two-flat

Reproduce el benchmark térmico de Kim, Cho & Choi (2024) para un sistema de
6 cables XLPE 1000 mm² Al, 154 kV, I = 865 A, configuración two-flat,
enterrados a 1.4 m. Referencia: análisis FEM en COMSOL Multiphysics para
condiciones de verano (T_amb = 27.2 °C en superficie).

## Scripts disponibles

| Script | Modelo de suelo |
|--------|-----------------|
| `run_example.py` | Suelo efectivo homogéneo (k global) |
| `run_research_pac.py` | Zona PAC explícita con $k(x,y)$ sigmoide variable |
| `run_multilayer.py` | Suelo multicapa (3 capas) + zona PAC — Casos A y B |
| `run_multilayer_dense.py` | Igual que multilayer pero con red 256×6 — referencia de convergencia |
| `eval_all.py` | Evaluación comparativa de todos los modelos guardados |

## Perfiles de entrenamiento

| Perfil | Arquitectura | Parámetros aprox. | Adam | L-BFGS | Adam2 |
|--------|-------------|-------------------|------|--------|-------|
| `quick` | 64×4 MLP tanh | ~17 k | 5 000 | 500 | — |
| `research` | 128×5 MLP tanh | ~83 k | 10 000 | 500 | 500 |
| `dense` (ref.) | 256×6 MLP tanh | ~400 k | 20 000 | 1 000 | 1 000 |

Todos los perfiles usan formulación residual $T = T_{bg} + u_\theta$ con
Kennelly como fondo analítico y curriculum warmup del 30 % de los pasos Adam
(k homogéneo primero, luego k variable).

## Comandos

```bash
python examples/kim_2024_154kv_bedding/run_example.py --profile quick
python examples/kim_2024_154kv_bedding/run_example.py --profile research

python examples/kim_2024_154kv_bedding/run_research_pac.py --profile quick
python examples/kim_2024_154kv_bedding/run_research_pac.py --profile research

python examples/kim_2024_154kv_bedding/run_multilayer.py --profile quick
python examples/kim_2024_154kv_bedding/run_multilayer.py --profile research

# Red densa (larga ejecución, ~14-16 h en CPU)
python examples/kim_2024_154kv_bedding/run_multilayer_dense.py

# Evaluar todos los modelos guardados en una tabla comparativa
python examples/kim_2024_154kv_bedding/eval_all.py
```

## Condición lateral dependiente de profundidad

Las fronteras `left` y `right` usan un perfil T(z) piecewise linear
cargado desde `data/boundary_profiles.csv`:

| z (m) | T (°C) |
|-------|--------|
| 0.0 | 26.1 |
| 0.5 | 23.7 |
| 1.0 | 21.5 |
| 1.4 | 19.9 |
| 2.0 | 18.0 |
| 3.0 | 15.8 |
| 3.6 | 15.2 |

## Resultados (evaluados con `eval_all.py`, modelos entrenados)

| Script | Perfil | T_PINN (°C) | T_FEM ref (°C) | Error |
|--------|--------|-------------|----------------|-------|
| `run_example.py` | quick | 76.9 | 77.6 (sand) | −0.7 K |
| `run_example.py` | research | 75.8 | 77.6 (sand) | −1.8 K |
| `run_research_pac.py` | quick | 72.9 | 70.6 (PAC) | +2.3 K |
| `run_research_pac.py` | research | 77.3 | 70.6 (PAC) | +6.7 K ⚠️ |
| `run_multilayer.py` Caso A | quick | **77.4** | 77.6 (sand) | **−0.2 K** ✅ |
| `run_multilayer.py` Caso A | research | 79.3 | 77.6 (sand) | +1.7 K |
| `run_multilayer.py` Caso B | quick | 69.2 | 70.6 (PAC) | −1.4 K |
| `run_multilayer.py` Caso B | research | **69.8** | 70.6 (PAC) | **−0.8 K** ✅ |
| `run_multilayer_dense.py` Caso A | dense | (en entrenamiento) | 77.6 (sand) | — |
| `run_multilayer_dense.py` Caso B | dense | (en entrenamiento) | 70.6 (PAC) | — |

Referencia FEM COMSOL — Kim, Cho & Choi (2024), escenario verano:

| Configuración | T_max FEM |
|----------------|-----------|
| Suelo arena (sin PAC) | 77.6 °C |
| Suelo + zona PAC | 70.6 °C |

### Notas sobre discrepancias

- **`run_research_pac.py` research (+6.7 K)**: la zona PAC con $k(x,y)$ sigmoide
  genera gradientes desbalanceados entre los términos de pérdida PDE y BC cuando
  la red es grande (128×5). Ver análisis en `docs/teoria_mejoras_pinn.md`
  (sección 3.1).
- **`run_multilayer` research Caso A (+1.7 K)**: leve sobreestimación en el perfil
  de mayor capacidad; aceptable para propósitos de tesis.
- **`run_multilayer_dense`**: sirve como referencia de campo completo para
  calcular el RMSE espacial en la zona de cables (independiente del punto único
  del paper).

## Métricas de evaluación

`eval_all.py` reporta por modelo:

| Métrica | Significado |
|---------|-------------|
| **Error [K]** | T_max_PINN − T_FEM_paper (punto único del paper) |
| **Loss final** | Último valor de la función de pérdida al terminar L-BFGS |
| **PDE_rms [W/m²]** | RMSE del residuo $\nabla\cdot(k\nabla T)$ en zona alrededor de cables |
| **RMSE_zona [K]** | Error RMS del campo T vs modelo denso (multilayer) o vs quick/research |

## Referencia

Kim, J., Cho, S., & Choi, S. (2024). Thermal analysis of 154 kV underground
cable system with PAC bedding using COMSOL Multiphysics. *(Datos: 6 cables
XLPE 1000 mm² Al, two-flat, I = 865 A, f = 60 Hz, Q_d = 3.57 W/m.)*

