"""Benchmark: Aras, Oysu & Yilmaz (2005) — 154 kV Three Cables Flat Formation.

Replica el caso "Analysis of 154 kV Three Cables in Flat Installation"
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
- Separacion entre cables: s = 0.33 m (centro a centro, Fig. 9)
- Formacion plana ("flat"): 3 cables en linea horizontal
- Perdidas dielectricas: W_d = 3.57 W/m (sobre aislacion XLPE)
- Dominio FEM: 10 m profundidad x 18 m ancho

Resultados de referencia (Tabla 4 del paper)
---------------------------------------------
- FEM (ANSYS):  ampacity = 1110 A  (a T_cond = 90 degC)
- IEC 60287:    ampacity = 1070 A  (diferencia 3.7 %%)

Estrategia del benchmark
-------------------------
Para reproducir las condiciones exactas del paper (Fig. 9), se aplican:
- q = 44.4 W/m como fuente de calor en cada conductor (valor FEM del paper)
- q_d = 3.57 W/m como fuente termica volumetrica sobre la capa XLPE
El total evacuado por cable es Q_total = 44.4 + 3.57 = 47.97 W/m.

El efecto de interferencia termica (mutual heating) entre los 3 cables se
captura tanto en el T_bg analitico (superposicion de imagenes de Kennelly)
como en la correccion *u* de la red neuronal.

Uso::

    python examples/aras_2005_154kv_flat/run_example.py
    python examples/aras_2005_154kv_flat/run_example.py --profile research
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
# Datos del paper Aras et al. (2005) — tres cables en formacion plana
# ---------------------------------------------------------------------------
PAPER_FEM_AMPACITY = 1110   # A  — Tabla 4, three cables flat
PAPER_IEC_AMPACITY = 1070   # A  — Tabla 4, three cables flat
PAPER_T_MAX = 363.15        # K  (90 degC)  — limite XLPE
PAPER_W_D = 3.57            # W/m  — perdidas dielectricas en XLPE
PAPER_K_XLPE = 0.2857       # W/(mK) — conductividad termica del XLPE
PAPER_K_SCREEN = 384.6      # W/(mK) — conductividad termica de la pantalla
PAPER_FREQ = 50.0           # Hz  — frecuencia de la red
PAPER_Q_COND = 44.4         # W/m — calor en conductor (paper Fig. 9, incluye lambda1+yp)
PAPER_CABLE_SEP = 0.33      # m  — separacion centro a centro (paper Fig. 9)


def main() -> None:
    """Cargar datos, entrenar PINN y comparar con Aras et al. (2005) — flat."""
    parser = argparse.ArgumentParser(
        description="Benchmark PINN: Aras (2005) — 3 cables XLPE 154 kV flat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Perfiles disponibles:\n"
            "  quick    : 5 000 Adam + 500 L-BFGS, red 64x4  (~8-12 min CPU)\n"
            "  research : 10 000 Adam + 1 000 L-BFGS, red 128x5 (~30-45 min CPU)\n"
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
    all_placements = problem.placements

    # Obtener capas del cable (compartidas — mismo tipo de cable)
    layers_template = problem.get_layers(all_placements[0].cable_id)

    section_mm2 = all_placements[0].section_mm2
    material = all_placements[0].conductor_material.upper()
    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A = all_placements[0].current_A
    n_cables = len(all_placements)

    # -----------------------------------------------------------------
    # Calculo de calor: valores del paper (Fig. 9)
    # q = 44.4 W/m en cada conductor (incluye lambda1, yp, etc.)
    # q_d = 3.57 W/m distribuida volumetricamente en XLPE
    # Q_total = 44.4 + 3.57 = 47.97 W/m evacuados por cable
    # -----------------------------------------------------------------
    # IEC 60287 solo como referencia comparativa
    iec_q = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=PAPER_T_MAX, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_cond_paper = PAPER_Q_COND      # 44.4 W/m at conductor
    Q_d_paper = PAPER_W_D            # 3.57 W/m distributed in XLPE
    Q_total_lin = Q_cond_paper + Q_d_paper  # 47.97 W/m total per cable

    # Sobreescribir Q_vol del conductor con el calor total
    layers = override_conductor_Q(layers_template, Q_total_lin)

    Q_lins = [Q_total_lin] * n_cables  # misma corriente en los 3 cables

    SEP = "=" * 72
    print(SEP)
    print("  BENCHMARK: Aras, Oysu & Yilmaz (2005)")
    print("  154 kV Three Cables in Flat Installation — PINN vs FEM vs IEC")
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

    q_lin = Q_total_lin

    print("\n  Problema fisico:")
    print("  Escenario       : %s  (%s)" % (scenario.scenario_id, scenario.mode))
    print("  Cables          : %d x XLPE 1200 mm2 %s  I=%.0f A" % (
        n_cables, material, current_A))
    print("  Formacion       : plana (flat), s = %.2f m entre centros" % PAPER_CABLE_SEP)
    print("  T_operacion ref.: 90 degC  (para calculo R(T) y skin effect)")
    print()
    print("  Fuentes de calor del paper (Fig. 9):")
    print("  q conductor     : %.1f W/m  (incluye lambda1, yp, skin)" % Q_cond_paper)
    print("  q_d XLPE (vol.) : %.2f W/m  (perdidas dielectricas distribuidas)" % Q_d_paper)
    print("  Q_TOTAL (x1)    : %.2f W/m  (x%d = %.2f W/m)" % (
        q_lin, n_cables, q_lin * n_cables))
    print()
    print("  Referencia IEC 60287 (solo I^2Rac + diel, sin lambda1/yp):")
    print("  Q_cond (I^2Rac) : %.2f W/m" % iec_q["Q_cond_W_per_m"])
    print("  Q_total IEC     : %.2f W/m  (paper usa %.1f + %.2f = %.2f)" % (
        iec_q["Q_total_W_per_m"], Q_cond_paper, Q_d_paper, Q_total_lin))
    print()
    print("  k_suelo         : %.1f W/(m*K)" % scenario.k_soil)
    print("  T_ambiente      : %.1f degC  (%.2f K)" % (
        scenario.T_amb - 273.15, scenario.T_amb))
    print("  Profundidad     : %.1f m" % abs(all_placements[0].cy))
    print("  Posiciones (cx) :", ", ".join(
        "%.2f" % p.cx for p in all_placements))
    print("  Dominio         : [%.0f, %.0f] x [%.0f, %.0f] m" % (
        problem.domain.xmin, problem.domain.xmax,
        problem.domain.ymin, problem.domain.ymax))

    # Estimacion analitica IEC 60287 (unified)
    layers_list = [layers] * n_cables
    iec = iec60287_estimate(
        layers_list, all_placements, scenario.k_soil, scenario.T_amb,
        Q_lins=Q_lins, Q_d=Q_d_paper,
    )
    T_ref_K = iec["T_cond_ref"]
    hottest = iec["hottest_idx"]

    # Compute self+mutual split for display (central cable = hottest)
    r_sheath = layers[-1].r_outer
    d_central = abs(all_placements[hottest].cy)
    dT_soil_self = Q_lins[hottest] / (2.0 * math.pi * scenario.k_soil) * math.log(
        2.0 * d_central / r_sheath)
    dT_mutual = iec["cables"][hottest]["dT_soil"] - dT_soil_self

    print("\n  Referencia analitica (cable central, peor caso):")
    print("  Q total (lin.)   : %.2f W/m (cond %.1f + diel %.2f)" % (
        q_lin, Q_cond_paper, Q_d_paper))
    for name, dT in iec["dT_by_layer"].items():
        print("  dT %-10s   : %+.2f K" % (name, dT))
    print("  dT suelo propio  : %+.2f K" % dT_soil_self)
    print("  dT mutual (otros): %+.2f K" % dT_mutual)
    dT_total = T_ref_K - scenario.T_amb
    print("  dT TOTAL         : %+.2f K  -->  T_cond = %.1f K (%.1f degC)" % (
        dT_total, T_ref_K, T_ref_K - 273.15))

    print("\n  Referencia paper Aras et al. (2005):")
    print("  Ampacity FEM     : %d A  (T_cond = 90 degC)" % PAPER_FEM_AMPACITY)
    print("  Ampacity IEC     : %d A  (T_cond = 90 degC)" % PAPER_IEC_AMPACITY)
    print("  Diferencia       : %.1f %%" % (
        100.0 * abs(PAPER_FEM_AMPACITY - PAPER_IEC_AMPACITY) / PAPER_IEC_AMPACITY))

    # Configuracion y modelo residual: T = T_bg(Kennelly N cables) + u(MLP)
    # T_bg superpone Kennelly para los 3 cables; u corrige efecto bordes.
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="example_flat_" + profile)

    normalize = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    base_model = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModel(
        base_model,
        layers_list,
        all_placements,
        scenario.k_soil,
        scenario.T_amb,
        Q_lins,
        problem.domain,
        normalize=normalize,
        Q_d=Q_d_paper,
    )
    init_output_bias(model.base, 0.0)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n  Pre-entrenando en perfil multicable (500 pasos)...", flush=True)
    rmse_pre = pretrain_multicable(
        model, all_placements, problem.domain,
        layers_list, Q_lins, scenario.k_soil, scenario.T_amb,
        device=device, normalize=normalize, Q_d=Q_d_paper,
        n_per_cable=1500, n_bc=200, n_steps=500, lr=1e-3,
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

    # Geometria
    geo_path = RESULTS_DIR / "geometry.png"
    plot_cable_geometry(
        layers, all_placements[0], problem.domain,
        title="3 Cables XLPE 154 kV Flat — Aras (2005)",
        save_path=geo_path,
    )

    # Entrenamiento: Adam (explorac. global) luego L-BFGS (refinamiento)
    print("\n" + "-" * 72)
    print("  ENTRENAMIENTO  (Adam --> L-BFGS)")
    print("  Columnas del log:  [fase paso/total pct%%] loss  pde  bc  ifc")
    print("-" * 72)
    trainer = SteadyStatePINNTrainer(
        model=model,
        layers=layers,
        placement=all_placements[0],
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

    # Graficas
    loss_path  = RESULTS_DIR / "loss_history.png"
    T_map_path = RESULTS_DIR / "temperature_field.png"
    plot_loss_history(
        history,
        title="Perdida — Aras (2005) 154 kV Flat (%s)" % profile,
        save_path=loss_path,
    )
    X, Y, T = evaluate_on_grid(
        model, problem.domain, nx=300, ny=300,
        device=device, normalize=normalize,
    )
    plot_temperature_field(
        X, Y, T,
        title="T(x,y) [K] — 3 x XLPE 1200 mm2 %s %.0f A flat [%s]" % (
            material, current_A, profile),
        save_path=T_map_path,
    )

    # -----------------------------------------------------------------
    # Resumen comparativo PINN vs Paper
    # -----------------------------------------------------------------
    T_max_pinn = float(T.max())
    T_min_pinn = float(T.min())
    T_amb_K    = scenario.T_amb

    # Evaluar T en el centro de CADA conductor
    T_cond_pinns = eval_conductor_temps(
        model, all_placements, problem.domain, device, normalize,
    )

    # Cable central (hottest) es el peor caso
    T_cond_central = T_cond_pinns[hottest]
    T_cond_max = max(T_cond_pinns)
    loss_final = history["total"][-1]
    error_K = T_cond_central - T_ref_K
    error_vs_paper = T_cond_central - PAPER_T_MAX

    C = 40
    print("\n" + SEP)
    print("  RESULTADOS — BENCHMARK Aras et al. (2005)")
    print("  3 x Cable XLPE 154 kV 1200 mm2 %s / I = %d A / Flat" % (
        material, PAPER_FEM_AMPACITY))
    print(SEP)

    print("\n  --- Temperaturas por cable ---")
    for i, (p, Tc) in enumerate(zip(all_placements, T_cond_pinns)):
        tag = " (central)" if i == hottest else " (lateral)"
        print("  Cable %d (cx=%+.2f m)%s : T = %.2f K (%.1f degC)" % (
            i + 1, p.cx, tag, Tc, Tc - 273.15))

    print("\n  %-*s  %10s  %10s  %9s" % (C, "Magnitud", "PINN", "Referencia", "Error"))
    print("  " + "-" * 68)

    print("\n  --- PINN vs IEC 60287 (cable central) ---")
    print("  %-*s  %10.2f  %10.2f  %+8.2f K" % (
        C, "T conductor (K)", T_cond_central, T_ref_K, error_K))
    print("  %-*s  %10.1f  %10.1f  %+8.1f K" % (
        C, "T conductor (degC)",
        T_cond_central - 273.15, T_ref_K - 273.15, error_K))
    print("  %-*s  %10.2f  %10.2f" % (
        C, "dT conductor-amb (K)", T_cond_central - T_amb_K, dT_total))
    print("  %-*s  %10.2f  %10s" % (
        C, "dT mutual (K)", dT_mutual, "-"))
    print("  %-*s  %10.2f" % (C, "Q_total por cable (W/m)", q_lin))

    print("\n  --- PINN vs Paper FEM (cable central) ---")
    print("  %-*s  %10.2f  %10.2f  %+8.2f K" % (
        C, "T conductor (K)", T_cond_central, PAPER_T_MAX, error_vs_paper))
    print("  %-*s  %10.1f  %10.1f  %+8.1f K" % (
        C, "T conductor (degC)",
        T_cond_central - 273.15, 90.0, error_vs_paper))

    print("\n  --- Comparacion entre metodos ---")
    print("  %-*s  %10s  %10s" % (C, "Metodo", "Ampacity", "T_cond"))
    print("  " + "-" * 68)
    print("  %-*s  %10d A  %9.1f degC" % (
        C, "FEM ANSYS (paper)", PAPER_FEM_AMPACITY, 90.0))
    print("  %-*s  %10d A  %9.1f degC" % (
        C, "IEC 60287 (paper)", PAPER_IEC_AMPACITY, 90.0))
    print("  %-*s  %10d A  %9.1f degC" % (
        C, "IEC analitico (este script)", PAPER_FEM_AMPACITY,
        T_ref_K - 273.15))
    print("  %-*s  %10d A  %9.1f degC" % (
        C, "PINN (cable central)", PAPER_FEM_AMPACITY,
        T_cond_central - 273.15))

    print("\n  " + "-" * 68)
    print("  %-*s  %10.2f" % (C, "T max dominio (K)", T_max_pinn))
    print("  %-*s  %10.2f" % (C, "T min dominio (K)", T_min_pinn))
    print("  %-*s  %10.4e" % (C, "Perdida final", loss_final))
    print("  " + "-" * 68)
    print("  Limite IEC XLPE: 363 K (90 degC)")
    print("  Margen termico PINN (central): %.1f K" % (363.0 - T_cond_central))

    if profile == "quick":
        print()
        print("  Para resultados de investigacion:")
        print("    python examples/aras_2005_154kv_flat/run_example.py --profile research")
    print()
    print("  Graficas guardadas en: %s" % RESULTS_DIR)
    print("    - geometry.png  |  loss_history.png  |  temperature_field.png")
    print("    - model_final.pt  (pesos de la red entrenada)")


if __name__ == "__main__":
    main()
