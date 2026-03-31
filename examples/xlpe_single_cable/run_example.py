"""Ejemplo cable XLPE a 12/20 kV enterrado a 70 cm.

Soporta cables XLPE de 95, 150, 240, 400 y 600 mm2 con conductor de
cobre (cu) o aluminio (al).  La seccion, material y corriente se
especifican en ``cables_placement.csv`` (columnas ``section_mm2``,
``conductor_material``, ``current_A``).

Soporta dos perfiles de ejecucion:

- **quick**    (~5-8 min CPU) : 5 000 Adam + 500 L-BFGS, red 64x4
- **research** (~25-35 min CPU): 15 000 Adam + 3 000 L-BFGS, red 128x5

Uso::

    python examples/xlpe_single_cable/run_example.py
    python examples/xlpe_single_cable/run_example.py --profile research

Referencia IEC 60287: T_max admisible XLPE = 90 degC (363 K).
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

import torch  # noqa: E402

from pinn_cables.io.readers import load_problem, load_solver_params  # noqa: E402
from pinn_cables.physics.kennelly import iec60287_estimate  # noqa: E402
from pinn_cables.pinn.model import build_model, ResidualPINNModel  # noqa: E402
from pinn_cables.pinn.train import SteadyStatePINNTrainer  # noqa: E402
from pinn_cables.pinn.train_custom import (  # noqa: E402
    init_output_bias,
    pretrain_multicable,
)
from pinn_cables.pinn.utils import get_device, set_seed, setup_logging  # noqa: E402
from pinn_cables.post.eval import eval_conductor_temps, evaluate_on_grid  # noqa: E402
from pinn_cables.post.plots import (  # noqa: E402
    plot_cable_geometry,
    plot_loss_history,
    plot_temperature_field,
)


def main() -> None:
    """Cargar datos, entrenar PINN y mostrar resumen comparativo con IEC 60287."""
    parser = argparse.ArgumentParser(
        description="Ejemplo PINN: cable XLPE en zanja estandar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Perfiles disponibles:\n"
            "  quick    : 5 000 Adam + 500 L-BFGS, red 64x4  (~5-8 min CPU)\n"
            "  research : 15 000 Adam + 3 000 L-BFGS, red 128x5 (~25-35 min CPU)\n"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "research"],
        default="quick",
        help="Perfil de ejecucion (default: quick)",
    )
    args = parser.parse_args()
    profile = args.profile

    RESULTS_DIR = HERE / ("results" if profile == "quick" else "results_research")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar problema fisico (CSV)
    problem = load_problem(DATA_DIR)
    scenario = problem.scenarios[0]
    placement = problem.placements[0]

    # Obtener capas del cable (per-cable del catalogo o fallback a cable_layers.csv)
    layers = problem.get_layers(placement.cable_id)

    # Informacion del tipo de cable
    section_mm2 = placement.section_mm2
    material = placement.conductor_material.upper()
    current_A = placement.current_A

    SEP = "=" * 62
    print(SEP)
    if section_mm2 > 0:
        print("  PINN -- Cable XLPE %d mm2 %s / 12-20 kV / %.0f A" % (
            section_mm2, material, current_A))
    else:
        print("  PINN -- Cable XLPE / 12-20 kV / zanja estandar")
    print("  Perfil de ejecucion : %s" % profile.upper())
    print(SEP)

    # Cargar config del perfil seleccionado
    params_csv = DATA_DIR / (
        "solver_params.csv" if profile == "quick" else "solver_params_research.csv"
    )
    solver_params = load_solver_params(params_csv)
    solver_cfg = solver_params.to_solver_cfg()

    adam_n   = solver_cfg["training"]["adam_steps"]
    lbfgs_n  = solver_cfg["training"]["lbfgs_steps"]
    print_ev = solver_cfg["training"]["print_every"]
    width    = solver_cfg["model"]["width"]
    depth    = solver_cfg["model"]["depth"]
    n_int    = solver_cfg["sampling"]["n_interior"]
    n_ifc    = solver_cfg["sampling"]["n_interface"]
    n_bnd    = solver_cfg["sampling"]["n_boundary"]

    conductor = layers[0]
    q_kw  = conductor.Q * scenario.Q_scale / 1000
    q_lin = conductor.Q * scenario.Q_scale * math.pi * conductor.r_outer ** 2
    print("\n  Problema fisico:")
    print("  Escenario   : %s  (%s)" % (scenario.scenario_id, scenario.mode))
    if section_mm2 > 0:
        print("  Cable       : XLPE %d mm2  %s  I=%.0f A" % (
            section_mm2, material, current_A))
    print("  Q_conductor : %.0f kW/m3  (%.2f W/m lineal)" % (q_kw, q_lin))
    print("  k_suelo     : %.1f W/(m*K)  (suelo humedo tipico)" % scenario.k_soil)
    print("  T_ambiente  : %.1f degC  (%.2f K)" % (scenario.T_amb - 273.15, scenario.T_amb))

    # Estimacion analitica IEC 60287 (unified)
    iec = iec60287_estimate(
        [layers], [placement], scenario.k_soil, scenario.T_amb,
        Q_lins=[q_lin],
    )
    T_ref_K = iec["T_cond_ref"]
    dT_total = T_ref_K - scenario.T_amb
    dT_soil = iec["cables"][0]["dT_soil"]

    print("\n  Referencia analitica IEC 60287 (resistencias en serie):")
    print("  Calor total    : %.2f W/m lineal" % q_lin)
    for name, dT in iec["dT_by_layer"].items():
        print("  dT %-10s   : %+.2f K" % (name, dT))
    print("  dT suelo       : %+.2f K" % dT_soil)
    print("  dT TOTAL       : %+.2f K" % dT_total)
    print("  T_cond ref.    : %.1f K  (%.1f degC)" % (T_ref_K, T_ref_K - 273.15))

    # Configuracion del modelo
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="example_" + profile)

    # Modelo residual: T = T_bg(Kennelly) + u(MLP)
    # T_bg aporta la solucion analitica de fondo y u aprende la correccion.
    normalize = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    base_model = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModel(
        base_model,
        [layers],
        [placement],
        scenario.k_soil,
        scenario.T_amb,
        [q_lin],
        problem.domain,
        normalize=normalize,
    )
    init_output_bias(model.base, 0.0)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n  Pre-entrenando en perfil cilindrico multicapa (500 pasos)...", flush=True)
    rmse_pre = pretrain_multicable(
        model, [placement], problem.domain,
        [layers], [q_lin], scenario.k_soil, scenario.T_amb,
        device=device, normalize=normalize,
        n_per_cable=2000, n_bc=200, n_steps=500, lr=1e-3,
    )
    print("  Pre-entrenamiento OK: RMSE = %.3f K" % rmse_pre, flush=True)

    print("\n  Configuracion del solver:")
    print("  Red neuronal : MLP %dx%d  (%d params)" % (width, depth, n_params))
    print("  Puntos       : int=%d  ifc=%d  bnd=%d" % (n_int, n_ifc, n_bnd))
    print("  Entrenamiento: %d Adam + %d L-BFGS" % (adam_n, lbfgs_n))
    print("  Avance cada  : %d pasos Adam" % print_ev)
    logger.info(
        "Device=%s | Perfil=%s | red MLP%dx%d (%d params)",
        device, profile, width, depth, n_params,
    )

    # Geometria del cable
    geo_path = RESULTS_DIR / "geometry.png"
    cable_title = "Cable XLPE %d mm2 %s" % (section_mm2, material) if section_mm2 > 0 else "Cable XLPE"
    plot_cable_geometry(
        layers, placement, problem.domain,
        title="%s -- seccion transversal" % cable_title,
        save_path=geo_path,
    )

    # Entrenamiento PINN: Adam (exploracion global) + L-BFGS (refinamiento local)
    print("\n" + "-" * 62)
    print("  ENTRENAMIENTO  (Adam --> L-BFGS)")
    print("  Columnas del log:  [fase paso/total pct%%] loss  pde  bc  ifc")
    print("-" * 62)
    trainer = SteadyStatePINNTrainer(
        model=model,
        layers=layers,
        placement=placement,
        domain=problem.domain,
        soil=problem.soil,
        bcs=problem.bcs,
        scenario=scenario,
        solver_cfg=solver_cfg,
        device=device,
        logger=logger,
    )
    history = trainer.train()
    print("-" * 62)

    # Guardar modelo entrenado
    model_path = RESULTS_DIR / "model_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo guardado: %s", model_path)

    # Graficas de perdida y campo de temperatura
    loss_path  = RESULTS_DIR / "loss_history.png"
    T_map_path = RESULTS_DIR / "temperature_field.png"
    plot_loss_history(
        history,
        title="Historia de perdida (%s) -- Adam + L-BFGS" % profile,
        save_path=loss_path,
    )
    X, Y, T = evaluate_on_grid(
        model, problem.domain, nx=300, ny=300,
        device=device, normalize=normalize,
    )
    cable_label = "XLPE %d mm2 %s %.0f A" % (section_mm2, material, current_A) if section_mm2 > 0 else "XLPE"
    plot_temperature_field(
        X, Y, T,
        title="T(x,y) [K] -- %s  [%s]" % (cable_label, profile),
        save_path=T_map_path,
    )

    # Resumen comparativo PINN vs IEC 60287
    T_max_pinn  = float(T.max())
    T_min_pinn  = float(T.min())
    T_amb_K     = scenario.T_amb

    # Evaluacion directa en el centro del conductor
    T_cond_pinn = eval_conductor_temps(
        model, [placement], problem.domain, device, normalize,
    )[0]
    loss_final  = history["total"][-1]
    error_K     = T_cond_pinn - T_ref_K

    C = 32  # column width
    print("\n" + "=" * 62)
    print("  RESULTADOS FINALES  [%s]" % profile.upper())
    if section_mm2 > 0:
        print("  Cable: XLPE %d mm2  %s  I=%.0f A" % (section_mm2, material, current_A))
    print("=" * 62)
    print("  %-*s  %10s  %10s  %8s" % (C, "Magnitud", "PINN", "IEC ref.", "Error"))
    print("  " + "-" * 58)
    print("  %-*s  %10.2f  %10.2f  %+7.2f K" % (
        C, "T conductor (K)", T_cond_pinn, T_ref_K, error_K))
    print("  %-*s  %10.1f  %10.1f" % (
        C, "T conductor (degC)", T_cond_pinn - 273.15, T_ref_K - 273.15))
    print("  %-*s  %10.2f  %10.2f" % (
        C, "dT conductor-amb (K)", T_cond_pinn - T_amb_K, dT_total))
    print("  %-*s  %10.2f" % (C, "T max dominio (K)", T_max_pinn))
    print("  %-*s  %10.2f" % (C, "T min dominio (K)", T_min_pinn))
    print("  %-*s  %10.4e" % (C, "Perdida final", loss_final))
    print("  " + "-" * 58)
    print("  Limite IEC 60287 XLPE : 363 K (90 degC)")
    print("  Margen termico PINN   : %.1f K  (ref: %.1f K)" % (
        363.0 - T_cond_pinn, 363.0 - T_ref_K))
    if profile == "quick":
        print()
        print("  Para resultados de investigacion (~25-35 min):")
        print("    python examples/xlpe_single_cable/run_example.py --profile research")
    print()
    print("  Graficas guardadas en: %s" % RESULTS_DIR)
    print("    - geometry.png  |  loss_history.png  |  temperature_field.png")
    print("    - model_final.pt  (pesos de la red entrenada)")


if __name__ == "__main__":
    main()
