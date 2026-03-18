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

**Todo en CSV:** los datos físicos (geometría, materiales, escenarios) *y* los
hiperparámetros del solver (arquitectura de la red, pasos de entrenamiento, pesos de
la función de pérdida, muestreo) se definen en archivos CSV editables sin tocar código.
El archivo `solver.yaml` existe como plantilla de referencia, pero no es necesario
cuando el directorio de datos incluye `solver_params.csv`.

---

## Requisitos

- Python 3.10 o superior
- PyTorch ≥ 2.1
- numpy ≥ 1.24, matplotlib ≥ 3.7, pyyaml ≥ 6.0

---

## Instalación

```bash
git clone <url-del-repo>
cd tesis_MIA_lk

# Instalar en modo editable (incluye dependencias de tests)
pip install -e ".[dev]"
```

---

## Inicio rápido — ejemplo cable XLPE 95 mm²

Verifica que todo funciona con el ejemplo incluido (~2-3 min en CPU):

```bash
python examples/xlpe_single_cable/run_example.py
```

Salida esperada:

```
============================================================
  PINN — Cable XLPE 95 mm² en suelo húmedo (estacionario)
============================================================
  Dispositivo  : cpu
  Parámetros   : 21,057
  Adam steps   : 500
  ...
  [Adam  50/500   9.1%] loss=2.4312e+00  pde=1.234e+00 ...
  ...
------------------------------------------------------------
  RESULTADOS
  T_amb (frontera)   = 293.15 K  (20.0 °C)
  T_max predicha     = ~330-360 K  (~60-90 °C)
  ΔT (conductor)     ≈ 40-70 K
------------------------------------------------------------
  Plots guardados en: examples/xlpe_single_cable/results/
    - loss_history.png
    - temperature_field.png
    - geometry.png
```

Los plots se guardan en `examples/xlpe_single_cable/results/`.

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
│   └── xlpe_single_cable/    # Ejemplo completo
│       ├── data/                 # CSVs del ejemplo (cable 95 mm², suelo húmedo)
│       │   └── solver_params.csv # Config rápida (500 Adam, red 64×4)
│       ├── run_example.py        # Script standalone
│       └── results/              # Plots generados (gitignored)
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

```bash
# Tests unitarios rápidos (~30 s)
pytest pinn_cables/tests/ -v

# Incluir tests lentos de integración (entrena ~2 min)
pytest pinn_cables/tests/ -v --runslow -m slow
```

---

## Referencias

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics, 378*, 686–707.
- Aras, F., Oysu, C., & Yilmaz, G. (2005). An assessment of the methods for calculating ampacity of underground power cables. *Electric Power Components and Systems, 33*(12), 1385–1402.
- IEC. (2023). *IEC 60287-1-1: Electric cables — Calculation of the current rating*.
- Carslaw, H. S., & Jaeger, J. C. (1959). *Conduction of Heat in Solids* (2nd ed.). Oxford University Press.
