# pinn-cables — PINN para conducción de calor en cables subterráneos

Solver basado en **Physics-Informed Neural Networks (PINNs)** en PyTorch para resolver
la ecuación de conducción de calor 2D en cables subterráneos multicapa con coeficientes
térmicos espacialmente variables. Soporta modo **estacionario** y **transitorio**.

> Proyecto de tesis de Maestría en Ingeniería de Sistemas — FIIS.

---

## Descripción

El solver aproxima el campo de temperatura T(x,y) (o T(x,y,t) en transitorio) con una
red neuronal entrenada para satisfacer la PDE de conducción:

```
∂/∂x(k ∂T/∂x) + ∂/∂y(k ∂T/∂y) + Q = ρc ∂T/∂t
```

junto con las condiciones de contorno (Dirichlet, Neumann, Robin) y la continuidad de
temperatura y flujo en las interfaces entre capas del cable.

### Formulación residual (T = T_bg + u)

Los ejemplos usan una **formulación residual** donde el modelo aprende la corrección
`u` respecto a un perfil analítico de fondo `T_bg`:

- **T_bg**: superposición de Kennelly (imagen para semi-espacio) + perfil cilíndrico
  multicapa (logarítmico/parabólico según la región).
- **u**: corrección de dominio finito aprendida por la red neuronal.

Esta estrategia reduce el problema de aprendizaje a una perturbación pequeña, eliminando
mínimos locales espurios y acelerando la convergencia.

**Todo en CSV:** los datos físicos (geometría, materiales, escenarios) *y* los
hiperparámetros del solver (arquitectura de la red, pasos de entrenamiento, pesos de
la función de pérdida, muestreo) se definen en archivos CSV editables sin tocar código.

---

## Estado actual del proyecto

### Resultados de validación

| Ejemplo | Método ref. | T_cond PINN | T_cond ref. | Error |
|---------|-------------|-------------|-------------|-------|
| Aras 2005 — cable único 154 kV | FEM ANSYS [1] | 89.9 °C | 90.0 °C | −0.1 K |
| Aras 2005 — 3 cables flat | FEM ANSYS [1] | 90.7 °C (central) | 90.0 °C | +0.7 K |
| Kim 2024 — suelo efectivo (`run_example`) | FEM COMSOL [2] | 76.9 °C | 77.6 °C (sand) | −0.7 K |
| Kim 2024 — zona PAC explícita (`run_research_pac`) | FEM COMSOL [2] | 72.9 °C | 70.6 °C (PAC) | +2.3 K |
| Kim 2024 — multicapa Caso A (`run_multilayer`) | FEM COMSOL [2] | 77.4 °C | 77.6 °C (sand) | **−0.2 K** |
| Kim 2024 — multicapa Caso B (`run_multilayer`) | FEM COMSOL [2] | 69.2 °C | 70.6 °C (PAC) | −1.4 K |
| XLPE 95 mm² Cu / 270 A | IEC 60287 [3] | ≈ 38 °C | ≈ 38 °C | < 0.1 K |
| Trefoil 3×95 mm² / 300 A | IEC 60287 [3] | ~75–80 °C | ~75–80 °C | < 1 K |
| 3 trefoils 9×95 mm² / 150 A | IEC 60287 [3] | ~35–45 °C | ~35–45 °C | < 1 K |

### Tests

- **74 unit tests** pasando (losses, materials, model, pde, readers, sampler).
- **2 integration tests** (MMS con k constante, Laplace en rectángulo).
- Ejecutar con `pytest pinn_cables/tests/ -v`.

---

## Requisitos

- Python 3.10 o superior
- PyTorch ≥ 2.1
- numpy ≥ 1.24, matplotlib ≥ 3.7, pyyaml ≥ 6.0

---

## Instalación

```bash
gh repo clone lkoc/tesis_MIA_lk
cd tesis_MIA_lk

# Instalar en modo editable (incluye dependencias de tests)
pip install -e ".[dev]"
```

---

## Ejemplos disponibles

### 1. Cable XLPE único — 12/20 kV (cable estándar)

```bash
python examples/xlpe_single_cable/run_example.py
python examples/xlpe_single_cable/run_example.py --profile research
```

Cable XLPE 95 mm² Cu a 270 A, enterrado a 70 cm en suelo húmedo (k=1.0 W/(m·K)).
Perfiles: quick (~5–8 min) o research (~25–35 min).

### 2. Trefoil — 3 cables XLPE tocantes

```bash
python examples/xlpe_trefoil/run_example.py
python examples/xlpe_trefoil/run_example.py --profile research
```

3 cables 95 mm² Cu a 300 A en formación trefoil tocante, centroide a 70 cm.
Incluye R(T) iterativo y k(x,y) sigmoide espacialmente variable.

### 3. Tres trefoils — 9 cables (3 circuitos)

```bash
python examples/xlpe_three_trefoils/run_example.py
python examples/xlpe_three_trefoils/run_example.py --profile research
```

9 cables en 3 circuitos trefoil separados 0.30 m, con interferencia térmica mutua.

### 4. Benchmark Aras (2005) — cable único 154 kV

```bash
python examples/aras_2005_154kv/run_example.py
python examples/aras_2005_154kv/run_example.py --profile research
```

### 5. Benchmark Aras (2005) — 3 cables en formación flat

```bash
python examples/aras_2005_154kv_flat/run_example.py
python examples/aras_2005_154kv_flat/run_example.py --profile research
```

3 cables XLPE 1200 mm² Cu a 1110 A, separación 0.33 m, profundidad 1.2 m:
- **Resultado**: T_cond(central) = 90.7 °C vs 90.0 °C FEM (error +0.7 K)
- Simetría preservada: cables laterales a 86.4 °C
- Cable central +4.3 K más caliente (calentamiento mutuo)

### 6. Benchmark Kim et al. (2024) — 6 cables 154 kV + PAC bedding

```bash
# Suelo efectivo homogéneo
python examples/kim_2024_154kv_bedding/run_example.py --profile quick
python examples/kim_2024_154kv_bedding/run_example.py --profile research

# Zona PAC con k(x,y) variable
python examples/kim_2024_154kv_bedding/run_research_pac.py --profile quick
python examples/kim_2024_154kv_bedding/run_research_pac.py --profile research

# Suelo multicapa (3 capas) + PAC — Casos A y B
python examples/kim_2024_154kv_bedding/run_multilayer.py --profile quick
python examples/kim_2024_154kv_bedding/run_multilayer.py --profile research

# Modelo denso 256×6 — referencia de convergencia en tamaño de red
python examples/kim_2024_154kv_bedding/run_multilayer_dense.py

# Evaluación comparativa de todos los modelos
python examples/kim_2024_154kv_bedding/eval_all.py
```

Reproduce Kim, Cho & Choi (2024) [2]: 6 cables XLPE 1000 mm² Al, I = 865 A,
enterrados 1.4 m (two-flat), 154 kV. Perfil lateral de frontera T(z) piecewise
linear desde 26.1 °C (sup.) hasta 15.2 °C (3.6 m).

**Perfiles de red:**

| Perfil | Arquitectura | Parámetros | Adam + L-BFGS |
|--------|-------------|-----------|---------------|
| quick | 64×4 MLP tanh | ~17 k | 5 000 + 500 |
| research | 128×5 MLP tanh | ~83 k | 10 000 + 500 |
| dense (ref.) | 256×6 MLP tanh | ~400 k | 20 000 + 1 000 |

**Resultados (evaluados con `eval_all.py`):**

| Script | Perfil | T_PINN (°C) | T_FEM ref (°C) | Error |
|--------|--------|-------------|----------------|-------|
| `run_example.py` | quick | 76.9 | 77.6 (sand) | −0.7 K |
| `run_example.py` | research | 75.8 | 77.6 (sand) | −1.8 K |
| `run_research_pac.py` | quick | 72.9 | 70.6 (PAC) | +2.3 K |
| `run_research_pac.py` | research | 77.3 | 70.6 (PAC) | +6.7 K ⚠️ |
| `run_multilayer.py` Caso A | quick | **77.4** | 77.6 (sand) | **−0.2 K** |
| `run_multilayer.py` Caso A | research | 79.3 | 77.6 (sand) | +1.7 K |
| `run_multilayer.py` Caso B | quick | 69.2 | 70.6 (PAC) | −1.4 K |
| `run_multilayer.py` Caso B | research | **69.8** | 70.6 (PAC) | **−0.8 K** |

> ⚠️ `run_research_pac.py` research presenta sobreestimación (+6.7 K). La zona PAC con
> $k(x,y)$ sigmoide variable es un caso duro para PINNs con pesos fijos; consultar
> [docs/teoria\_mejoras\_pinn.md](docs/teoria_mejoras_pinn.md) para análisis detallado.

---

## Uso completo (solver de alta precisión)

Cuando el directorio de datos contiene `solver_params.csv`, no se necesita `--config`:

```bash
# Usando solver_params.csv del directorio de datos (recomendado)
python -m pinn_cables.scripts.run \
    --data    pinn_cables/data \
    --scenario base_steady \
    --output  results/
```

También se puede pasar un YAML explícito (compatibilidad hacia atrás):

```bash
python -m pinn_cables.scripts.run \
    --data    pinn_cables/data \
    --config  pinn_cables/configs/solver.yaml \
    --scenario base_steady \
    --output  results/
```

Escenarios disponibles en `pinn_cables/data/scenarios.csv`:

| scenario_id    | mode      | Descripción                        |
|----------------|-----------|-------------------------------------|
| `base_steady`  | steady    | Carga nominal, suelo húmedo        |
| `high_load`    | steady    | Doble carga (Q_scale=2)            |
| `dry_soil`     | steady    | Suelo seco (k=0.5 W/m·K)          |
| `transient_1h` | transient | Transitorio 1 hora                 |

---

## Estructura del proyecto

```
tesis_MIA_lk/
│
├── pinn_cables/              # Paquete principal
│   ├── configs/
│   │   └── solver.yaml       # Plantilla de referencia de hiperparámetros ML
│   │
│   ├── data/                 # Datos del problema (editar aquí)
│   │   ├── cable_layers.csv      # Capas: radio, k, ρc, Q
│   │   ├── domain.csv            # Dominio computacional
│   │   ├── cables_placement.csv  # Posición(es) de los cables
│   │   ├── boundary_conditions.csv # BCs por borde
│   │   ├── soil_properties.csv   # Propiedades del suelo
│   │   ├── scenarios.csv         # Escenarios paramétricos
│   │   └── solver_params.csv     # Hiperparámetros ML en CSV (recomendado)
│   │
│   ├── io/
│   │   └── readers.py        # Lectura CSV → dataclasses
│   ├── geom/
│   │   └── sampler.py        # Muestreo de puntos de colocación
│   ├── materials/
│   │   └── props.py          # Conductividad k(x,y), ρc, Q por región
│   ├── pinn/
│   │   ├── model.py          # MLP y Fourier Feature Network
│   │   ├── pde.py            # Operadores PDE con autograd
│   │   ├── losses.py         # Pérdidas: PDE, BC, interfaces, IC
│   │   ├── train.py          # Entrenadores: steady y transient
│   │   └── utils.py          # Seeds, dispositivo, normalización, logging
│   ├── post/
│   │   ├── eval.py           # Métricas, evaluación en grilla, benchmarks MMS
│   │   └── plots.py          # Visualización: campos T, errores, geometría
│   ├── scripts/
│   │   └── run.py            # Entrypoint CLI
│   └── tests/                # Suite de tests (pytest)
│
├── examples/
│   ├── xlpe_single_cable/        # Cable XLPE 95 mm² Cu, 270 A
│   ├── xlpe_trefoil/             # 3 cables tocantes en trefoil, 300 A
│   ├── xlpe_three_trefoils/      # 9 cables (3 circuitos trefoil), 150 A
│   ├── aras_2005_154kv/          # Benchmark: cable único 154 kV, 1657 A [1]
│   ├── aras_2005_154kv_flat/     # Benchmark: 3 cables flat, 1110 A [1]
│   └── kim_2024_154kv_bedding/   # Benchmark: 6 cables two-flat PAC/multicapa [2]
│       ├── data/                 # CSVs del ejemplo
│       ├── run_example.py        # Suelo efectivo homogéneo
│       ├── run_research_pac.py   # Zona PAC con k(x,y) variable
│       ├── run_multilayer.py     # Suelo multicapa 3 capas, Casos A y B
│       ├── run_multilayer_dense.py  # Modelo denso 256×6 — referencia convergencia
│       ├── eval_all.py           # Evaluación comparativa de todos los modelos
│       └── results*/             # Modelos guardados por perfil
│
├── pinns_cables_tesis_con_citas.md  # Documento de tesis base
└── pyproject.toml
```

---

## Personalizar el problema

Para simular un cable diferente, solo edita los CSV en `pinn_cables/data/` (o crea
una carpeta nueva siguiendo el modelo `examples/xlpe_single_cable/data/`):

### `cable_layers.csv`
```csv
layer_name,r_inner,r_outer,k,rho_c,Q
conductor,0.0,0.0087,400.0,3.45e6,50000.0   # Cu 240mm², 400 A
xlpe,0.0087,0.0197,0.286,1.9e6,0.0
screen,0.0197,0.021,380.0,3.4e6,0.0
sheath,0.021,0.023,0.45,1.9e6,0.0
```

Unidades: radios en **m**, k en **W/(m·K)**, ρc en **J/(m³·K)**, Q en **W/m³**.

### `scenarios.csv`
```csv
scenario_id,mode,Q_scale,k_soil,T_amb,t_end
nominal,steady,1.0,1.0,293.15,0
sobrecarga,steady,2.5,1.0,293.15,0
suelo_seco,steady,1.0,0.5,303.15,0
transitorio,transient,1.0,1.0,293.15,7200
```

### `solver_params.csv` — hiperparámetros ML
```csv
param,value
model_width,128
model_depth,6
model_activation,tanh
lr,1.0e-3
adam_steps,20000
lbfgs_steps,5000
print_every,200
n_interior,8000
n_interface,500
n_boundary,400
normalize_coords,true
w_pde,1.0
w_bc_dirichlet,10.0
device,auto
seed,42
```

Los parámetros no incluidos usan los valores por defecto definidos en `SolverParams`.
El archivo YAML `configs/solver.yaml` sigue funcionando con `--config` (compatibilidad
hacia atrás), pero `solver_params.csv` es el método recomendado para mantener toda la
configuración en un solo lugar.

---

## Tests

```bash
# Tests unitarios rápidos (~30 s, 74 tests)
pytest pinn_cables/tests/ -v

# Incluir tests lentos de integración (MMS + Laplace, ~2 min)
pytest pinn_cables/tests/ -v --runslow -m slow
```

---

## Referencias

1. **Aras, Oysu & Yilmaz (2005)** — Aras, F., Oysu, C., & Yilmaz, G. (2005). An assessment of the methods for calculating ampacity of underground power cables. *Electric Power Components and Systems, 33*(12), 1385–1402. https://doi.org/10.1080/15325000590964969

2. **Kim, Cho & Choi (2024)** — Kim, J., Cho, S., & Choi, S. (2024). Thermal analysis of 154 kV underground cable system with PAC bedding using COMSOL Multiphysics. *IEEE Transactions on Power Delivery* (in review / preprint). *(Datos FEM: sand 77.6 °C, PAC 70.6 °C, verano, I = 865 A.)*

3. **IEC 60287** — IEC. (2023). *IEC 60287-1-1: Electric cables — Calculation of the current rating — Part 1-1: Current rating equations and calculation of losses*. International Electrotechnical Commission.

4. **Raissi, Perdikaris & Karniadakis (2019)** — Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics, 378*, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

5. **Carslaw & Jaeger (1959)** — Carslaw, H. S., & Jaeger, J. C. (1959). *Conduction of Heat in Solids* (2nd ed.). Oxford University Press.

6. **Wang & Perdikaris (2021)** — Wang, S., & Perdikaris, P. (2021). Understanding and mitigating gradient pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing, 43*(5), A3055–A3081. https://arxiv.org/abs/2001.04536
