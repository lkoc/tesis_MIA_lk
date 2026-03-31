"""Benchmark: Aras, Oysu & Yilmaz (2005) — 154 kV Single Underground Cable.

Replica el caso "154 kV Single Underground Cable Analysis with FEM and IEC"
del articulo:

    F. Aras, C. Oysu & G. Yilmaz, "An Assessment of the Methods for
    Calculating Ampacity of Underground Power Cables",
    Electric Power Components and Systems, 33:1385–1402, 2005.

Parametros del paper
--------------------
- Cable XLPE 154 kV, conductor Cu 1200 mm2 (D = 37.7 mm)
  D_xlpe = 81.7 mm, D_screen = 98.7 mm, D_cover = 106.7 mm
- k_xlpe = 0.2857 W/(mK), k_screen = 384.6 W/(mK)
- k_suelo = 1.2 W/(mK), T_amb = 20 degC (293.15 K)
- Profundidad de enterramiento: 1.2 m
- Perdidas dielectricas: W_d = 3.57 W/m (sobre aislacion XLPE)
- Dominio FEM: 10 m profundidad x 18 m ancho

Resultados de referencia (Tabla 4 del paper)
---------------------------------------------
- FEM (ANSYS):  ampacity = 1657 A  (a T_cond = 90 degC)
- IEC 60287:    ampacity = 1635 A  (diferencia 1.3 %%)

Estrategia del benchmark
-------------------------
Se reproducen las condiciones de borde exactas del paper (Figuras 1-2):
  - Calor en conductor: Q_cond = 70.0 W/m (retro-calculado a partir del
    resultado FEM T_cond = 90 degC; incluye I²R_ac + efecto piel + lambda_1
    de la pantalla metalica)
  - Perdidas dielectricas: W_d = 3.57 W/m aplicadas como fuente
    volumetrica distribuida en la capa XLPE (NO concentradas en el
    conductor)
  - Q_total por cable = 70.0 + 3.57 = 73.57 W/m

El PINN se entrena y se verifica que prediga T_conductor ~ 90 degC,
consistente con el FEM del paper.

Uso::

    python examples/aras_2005_154kv/run_example.py
    python examples/aras_2005_154kv/run_example.py --profile research
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

import torch  # noqa: E402

from pinn_cables.io.readers import (  # noqa: E402
    load_problem, load_solver_params, override_conductor_Q,
)
from pinn_cables.physics.iec60287 import compute_iec60287_Q  # noqa: E402
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

# ---------------------------------------------------------------------------
# Datos del paper Aras et al. (2005)
# ---------------------------------------------------------------------------
PAPER_FEM_AMPACITY = 1657   # A  — Tabla 4, single cable
PAPER_IEC_AMPACITY = 1635   # A  — Tabla 4, single cable
PAPER_T_MAX = 363.15        # K  (90 degC)  — limite XLPE
PAPER_W_D = 3.57            # W/m  — perdidas dielectricas en XLPE
PAPER_K_XLPE = 0.2857       # W/(mK) — conductividad termica del XLPE
PAPER_K_SCREEN = 384.6      # W/(mK) — conductividad termica de la pantalla
PAPER_FREQ = 50.0           # Hz  — frecuencia de la red
PAPER_Q_COND = 70.0         # W/m  — calor en conductor (Fig. 2)


def main() -> None:
    """Cargar datos, entrenar PINN y comparar con Aras et al. (2005)."""
    parser = argparse.ArgumentParser(
        description="Benchmark PINN: Aras et al. (2005) — cable XLPE 154 kV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Perfiles disponibles:\n"
            "  quick    : 5 000 Adam + 500 L-BFGS, red 64x4  (~5-8 min CPU)\n"
            "  research : 10 000 Adam + 1 000 L-BFGS, red 128x5 (~25-35 min CPU)\n"
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

    # Obtener capas del cable (per-cable del catalogo)
    layers = problem.get_layers(placement.cable_id)

    section_mm2 = placement.section_mm2
    material = placement.conductor_material.upper()
    material_lc = placement.conductor_material.strip().lower()
    current_A = placement.current_A

    # -----------------------------------------------------------------
    # Fuentes de calor: valores del paper (Fig. 2)
    # -----------------------------------------------------------------
    Q_d_paper = PAPER_W_D                            # 3.57 W/m
    Q_total_lin = PAPER_Q_COND + Q_d_paper           # 73.57 W/m

    # Referencia IEC solo para comparar (no se usa en el PINN)
    iec_q = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=PAPER_T_MAX, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )

    # Sobreescribir Q_vol del conductor con Q_total (Q_cond + Q_d)
    layers = override_conductor_Q(layers, Q_total_lin)

    SEP = "=" * 72
    print(SEP)
    print("  BENCHMARK: Aras, Oysu & Yilmaz (2005)")
    print("  154 kV Single Underground Cable — PINN vs FEM vs IEC 60287")
    print("  Cable XLPE 1200 mm2 %s / I = %.0f A (FEM ampacity)" % (
        material, current_A))
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
    q_lin = Q_total_lin
    print("\n  Problema fisico:")
    print("  Escenario       : %s  (%s)" % (scenario.scenario_id, scenario.mode))
    print("  Cable           : XLPE 1200 mm2  %s  I=%.0f A" % (material, current_A))
    print("  T_operacion ref.: 90 degC  (para calculo R(T) y skin effect)")
    print()
    print("  Fuentes de calor del paper (Fig. 2):")
    print("  q conductor     : %.1f W/m  (retro-calc.: I2Rac + lambda1 pantalla)" % PAPER_Q_COND)
    print("  q_d XLPE (vol.) : %.2f W/m  (perdidas dielectricas distribuidas)" % Q_d_paper)
    print("  Q_TOTAL         : %.2f W/m" % Q_total_lin)
    print()
    print("  Referencia IEC 60287 (solo I^2Rac + diel, sin lambda1):")
    print("  Q_cond (I^2Rac) : %.2f W/m" % iec_q["Q_cond_W_per_m"])
    print("  Q_total IEC     : %.2f W/m  (paper usa %.1f + %.2f = %.2f)" % (
        iec_q["Q_total_W_per_m"], PAPER_Q_COND, Q_d_paper, Q_total_lin))
    print()
    print("  k_suelo         : %.1f W/(m*K)" % scenario.k_soil)
    print("  T_ambiente      : %.1f degC  (%.2f K)" % (
        scenario.T_amb - 273.15, scenario.T_amb))
    print("  Profundidad     : %.1f m" % abs(placement.cy))
    print("  Dominio         : [%.0f, %.0f] x [%.0f, %.0f] m" % (
        problem.domain.xmin, problem.domain.xmax,
        problem.domain.ymin, problem.domain.ymax))

    # Estimacion analitica (resistencias termicas + fuente volumetrica XLPE)
    iec = iec60287_estimate(
        [layers], [placement], scenario.k_soil, scenario.T_amb,
        Q_lins=[q_lin], Q_d=Q_d_paper,
    )
    T_ref_K = iec["T_cond_ref"]

    print("\n  Referencia analitica (resistencias en serie + Q_d vol.):")
    print("  Q total (lin.)   : %.2f W/m (cond %.1f + diel %.2f)" % (
        q_lin, PAPER_Q_COND, Q_d_paper))
    for name, dT in iec["dT_by_layer"].items():
        print("  dT %-10s   : %+.2f K" % (name, dT))
    dT_soil = iec["cables"][0]["dT_soil"]
    dT_total = T_ref_K - scenario.T_amb
    print("  dT suelo         : %+.2f K" % dT_soil)
    print("  dT TOTAL         : %+.2f K  -->  T_cond = %.1f K (%.1f degC)" % (
        dT_total, T_ref_K, T_ref_K - 273.15))

    # Referencia del paper (FEM ANSYS)
    print("\n  Referencia paper Aras et al. (2005):")
    print("  Ampacity FEM     : %d A  (T_cond = 90 degC)" % PAPER_FEM_AMPACITY)
    print("  Ampacity IEC     : %d A  (T_cond = 90 degC)" % PAPER_IEC_AMPACITY)
    print("  Diferencia       : %.1f %%" % (
        100.0 * abs(PAPER_FEM_AMPACITY - PAPER_IEC_AMPACITY) / PAPER_IEC_AMPACITY))

    # Configuracion y modelo residual: T = T_bg(Kennelly) + u(MLP)
    # T_bg contiene la superposicion analitica; u corrige en dominio finito.
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="example_" + profile)

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
        Q_d=Q_d_paper,
    )
    init_output_bias(model.base, 0.0)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n  Pre-entrenando en perfil cilindrico multicapa (500 pasos)...", flush=True)
    rmse_pre = pretrain_multicable(
        model, [placement], problem.domain,
        [layers], [q_lin], scenario.k_soil, scenario.T_amb,
        device=device, normalize=normalize, Q_d=Q_d_paper,
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
    plot_cable_geometry(
        layers, placement, problem.domain,
        title="Cable XLPE 154 kV 1200 mm2 — Aras (2005)",
        save_path=geo_path,
    )

    # Entrenamiento PINN: Adam (explorac. global) + L-BFGS (refinamiento)
    print("\n" + "-" * 72)
    print("  ENTRENAMIENTO  (Adam --> L-BFGS)")
    print("  Columnas del log:  [fase paso/total pct%%] loss  pde  bc  ifc")
    print("-" * 72)
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
    print("-" * 72)

    # Guardar modelo entrenado
    model_path = RESULTS_DIR / "model_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo guardado: %s", model_path)

    # Graficas de perdida y campo de temperatura
    loss_path  = RESULTS_DIR / "loss_history.png"
    T_map_path = RESULTS_DIR / "temperature_field.png"
    plot_loss_history(
        history,
        title="Perdida — Aras (2005) 154 kV (%s)" % profile,
        save_path=loss_path,
    )
    X, Y, T = evaluate_on_grid(
        model, problem.domain, nx=300, ny=300,
        device=device, normalize=normalize,
    )
    plot_temperature_field(
        X, Y, T,
        title="T(x,y) [K] — Aras (2005), 154 kV (%s)" % profile,
        save_path=T_map_path,
    )

    # Evaluacion final — temperatura en el centro del conductor
    T_cond_pinn = eval_conductor_temps(
        model, [placement], problem.domain, device, normalize,
    )[0]

    T_max_pinn = float(T.max())
    T_min_pinn = float(T.min())
    loss_final = history["total"][-1]

    print("\n" + "=" * 72)
    print("  RESULTADOS FINALES  [%s]" % profile.upper())
    print("=" * 72)
    err_ref = T_cond_pinn - T_ref_K
    err_fem = T_cond_pinn - PAPER_T_MAX
    print("  %-34s  %10s  %10s  %8s" % ("Magnitud", "PINN", "Ref.", "Error"))
    print("  " + "-" * 62)
    print("  %-34s  %10.2f  %10.2f  %+7.2f K" % (
        "T_cond (K)", T_cond_pinn, T_ref_K, err_ref))
    print("  %-34s  %10.1f  %10.1f  %+7.1f K" % (
        "T_cond (degC)", T_cond_pinn - 273.15, T_ref_K - 273.15, err_ref))
    print("  %-34s  %10.2f  %10.2f  %+7.2f K" % (
        "T_cond vs FEM paper (K)", T_cond_pinn, PAPER_T_MAX, err_fem))
    print("  %-34s  %10.2f" % ("T max dominio (K)", T_max_pinn))
    print("  %-34s  %10.2f" % ("T min dominio (K)", T_min_pinn))
    print("  %-34s  %10.4e" % ("Perdida final", loss_final))
    print("  " + "-" * 62)
    print("  Limite IEC 60287 XLPE : 363 K (90 degC)")
    if T_cond_pinn < 363.0:
        print("  Margen termico PINN   : %.1f K" % (363.0 - T_cond_pinn))
    else:
        print("  EXCEDE limite IEC por  : %.1f K" % (T_cond_pinn - 363.0))
    if profile == "quick":
        print()
        print("  Para resultados de investigacion (~25-35 min):")
        print("    python examples/aras_2005_154kv/run_example.py --profile research")
    print()
    print("  Graficas guardadas en: %s" % RESULTS_DIR)
    print("    - geometry.png  |  loss_history.png  |  temperature_field.png")
    print("    - model_final.pt  (pesos de la red entrenada)")


if __name__ == "__main__":
    main()
