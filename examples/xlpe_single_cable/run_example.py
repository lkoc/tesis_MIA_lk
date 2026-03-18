"""Ejemplo de verificacion: cable XLPE 95 mm2 a 12/20 kV enterrado a 70 cm.

Este script demuestra el uso completo de la libreria pinn_cables en un caso
fisicamente representativo. Ejecuta en ~2-3 min en CPU con hiperparametros
reducidos. Para mayor precision usar pinn_cables/configs/solver.yaml.

Uso::

    python examples/xlpe_single_cable/run_example.py

Referencia IEC 60287: T_max admisible XLPE = 90 degC (363 K).
Para este cable (95 mm2, 150 kW/m3 Joule) la ampacidad aproximada es ~310 A.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegurar que pinn_cables sea importable sin pip install -e .
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CONFIG_PATH = HERE / "solver_quick.yaml"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402

from pinn_cables.io.readers import load_problem
from pinn_cables.pinn.model import build_model
from pinn_cables.pinn.train import SteadyStatePINNTrainer
from pinn_cables.pinn.utils import get_device, load_config, set_seed, setup_logging
from pinn_cables.post.eval import evaluate_on_grid
from pinn_cables.post.plots import (
    plot_cable_geometry,
    plot_loss_history,
    plot_temperature_field,
)


def _init_output_bias(model: nn.Module, value: float) -> None:
    """Set the last Linear layer's bias to *value* for warm-start near T_amb."""
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        last_linear.bias.data.fill_(value)


def main() -> None:
    """Cargar datos, entrenar PINN y mostrar resumen de resultados."""
    print("=" * 60)
    print("PINN -- Cable XLPE 95 mm2 / 12-20 kV / zanja estandar")
    print("=" * 60)

    # ------ 1. Cargar problema y configuracion ---------------------
    problem = load_problem(DATA_DIR)
    if problem.solver_params is not None:
        solver_cfg = problem.solver_params.to_solver_cfg()
    else:
        solver_cfg = load_config(CONFIG_PATH)
    scenario = problem.scenarios[0]

    print(f"\nEscenario   : {scenario.scenario_id}  ({scenario.mode})")
    q_kw = problem.layers[0].Q * scenario.Q_scale / 1000
    print(f"Q_conductor : {q_kw:.0f} kW/m3  (perdidas Joule ~300 A)")
    print(f"k_suelo     : {scenario.k_soil} W/(m*K)  (suelo humedo tipico)")
    print(f"T_ambiente  : {scenario.T_amb - 273.15:.1f} degC")

    # ------ 2. Configuracion del entorno ---------------------------
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="example")
    logger.info("Device: %s", device)

    # ------ 3. Construir modelo -----------------------------------
    model = build_model(solver_cfg["model"], in_dim=2, device=device)
    # Warm-start: initialize output near T_amb so Adam converges faster
    _init_output_bias(model, scenario.T_amb)
    n_params = sum(p.numel() for p in model.parameters())
    w = solver_cfg["model"]["width"]
    d = solver_cfg["model"]["depth"]
    print(f"\nRed neuronal : {model.__class__.__name__}  ({n_params:,} params)")
    print(f"Arquitectura : width={w}  depth={d}  act=tanh")

    # ------ 4. Graficar geometria del cable -----------------------
    geo_path = RESULTS_DIR / "geometry.png"
    plot_cable_geometry(
        problem.layers, problem.placements[0], problem.domain,
        title="Cable XLPE 95 mm2 -- seccion transversal",
        save_path=geo_path,
    )
    print(f"\nGeometria   : {geo_path}")

    # ------ 5. Entrenar PINN --------------------------------------
    print("\n--- Entrenamiento (Adam -> L-BFGS) ---")
    trainer = SteadyStatePINNTrainer(
        model=model,
        layers=problem.layers,
        placement=problem.placements[0],
        domain=problem.domain,
        soil=problem.soil,
        bcs=problem.bcs,
        scenario=scenario,
        solver_cfg=solver_cfg,
        device=device,
        logger=logger,
    )
    history = trainer.train()

    # ------ 6. Graficar perdida ----------------------------------
    loss_path = RESULTS_DIR / "loss_history.png"
    plot_loss_history(history, title="Historia de perdida (Adam + L-BFGS)",
                      save_path=loss_path)
    print(f"Perdida final : {history['total'][-1]:.4e}")

    # ------ 7. Evaluar y graficar campo de temperatura -----------
    normalize = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    X, Y, T = evaluate_on_grid(
        model, problem.domain, nx=120, ny=120,
        device=device, normalize=normalize,
    )
    T_field_path = RESULTS_DIR / "temperature_field.png"
    plot_temperature_field(
        X, Y, T,
        title="Campo de temperatura T(x,y) [K] -- XLPE 95 mm2",
        save_path=T_field_path,
    )

    # ------ 8. Resumen de resultados -----------------------------
    T_max = float(T.max())
    T_min = float(T.min())
    T_amb_K = scenario.T_amb

    place = problem.placements[0]
    dist = np.sqrt((X - place.cx) ** 2 + (Y - place.cy) ** 2)
    idx = np.unravel_index(dist.argmin(), dist.shape)
    T_cond = float(T[idx])

    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS (config rapida)")
    print("=" * 60)
    print(f"T_max dominio           : {T_max:.1f} K  ({T_max - 273.15:.1f} degC)")
    print(f"T en conductor (~ctr.)  : {T_cond:.1f} K  ({T_cond - 273.15:.1f} degC)")
    print(f"T_amb  (BC superficie)  : {T_amb_K:.2f} K  ({T_amb_K - 273.15:.1f} degC)")
    print(f"DT conductor-suelo      : {T_cond - T_amb_K:.1f} K")
    print(f"T_min dominio           : {T_min:.1f} K  ({T_min - 273.15:.1f} degC)")
    print()
    print("Referencia IEC 60287:")
    print("  T_max admisible XLPE : 363 K (90 degC)")
    print("  DT_max permitido     : ~70 K  ->  ampacidad ~ 310 A")
    print()
    adam_n = solver_cfg["training"]["adam_steps"]
    lbfgs_n = solver_cfg["training"]["lbfgs_steps"]
    print(f"NOTA: Config rapida ({adam_n} Adam + {lbfgs_n} L-BFGS).  ")
    print("      Para precision de investigacion (~30-40 min):")
    print("      python -m pinn_cables.scripts.run \\")
    print("        --data examples/xlpe_single_cable/data \\")
    print("        --config pinn_cables/configs/solver.yaml \\")
    print("        --scenario base_steady")
    print(f"\nResultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
