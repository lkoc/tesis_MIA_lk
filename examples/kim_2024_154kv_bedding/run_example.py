"""Benchmark Kim, Nguyen Cong, Dinh & Kim (2024) — 6 cables XLPE 154 kV
con PAC cable bedding material.

Referencia:
    Kim, Y.-S., Nguyen Cong, H., Dinh, B.H. & Kim, H.-K. (2024).
    "Effect of ambient air and ground temperatures on heat transfer in
    underground power cable system buried in newly developed cable
    bedding material."  Geothermics 125, 103151.
    DOI: 10.1016/j.geothermics.2024.103151

Sistema modelado (caso critico verano — caso 5 del paper):
- 6 cables XLPE 154 kV 1200 mm² Cu en formacion dos-plano (two-flat)
  3 columnas × 2 filas dentro de un duct-bank
- I = 1026 A (ampacidad critica determinada por temperaturas reales)
- Cable bedding: PAC (prepacked aggregate concrete) λ = 2.094 W/(mK)
- Relleno entre cable y casing pipe: CLSM λ = 2.150 W/(mK)
- Suelo natural: 3 capas con λ promedio ~1.55 W/(mK)
- T_amb_air = 27.2 °C, T_sur = 17.0 °C, T_g = 15.2 °C (fondo)
- Conveccion forzada en superficie: h = 7.371 W/(m²K) (v = 1.17 m/s)

Simplificaciones vs. el modelo FEM original (COMSOL):
- Los cables tienen las mismas 4 capas internas (conductor, XLPE, screen,
  sheath) sin modelar casing pipe/CLSM explicitamente; su efecto se
  captura en el calor total back-calculated.
- Suelo efectivo uniforme k = 1.55 W/(mK) (promedio ponderado de las 3 capas)
- Dominio 91×45.5 m (Db = 44 m segun convergencia del paper, Fig. 8)
- Condiciones de frontera simplificadas:
  - Top: Robin (conveccion forzada, h=7.371, T_amb_air=27.2°C=300.35K)
  - Bottom: Dirichlet T_g = 15.2°C = 288.35 K
  - Left/Right: Dirichlet T ~ T_g = 288.35 K (aproximacion; paper usa T(z,t))

Resultados de referencia FEM (COMSOL, PAC, verano):
- T_max conductor (cable B1 central fondo) = 70.6 °C
- T_max conductor (cable T1 central arriba) ≈ 70.6 - 1.4 = 69.2 °C

Formulacion residual: T = T_bg + u
  T_bg : superposicion de Kennelly para 6 cables + perfil cilindr. por cable
  u    : correccion aprendida por la red neuronal

Soporta dos perfiles de ejecucion:

- **quick**    (~15-20 min CPU): 5 000 Adam + 500 L-BFGS, red 64x4
- **research** (~60-90 min CPU): 10 000 Adam + 1 000 L-BFGS, red 128x5

Uso::

    python examples/kim_2024_154kv_bedding/run_example.py
    python examples/kim_2024_154kv_bedding/run_example.py --profile research
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import torch  # noqa: E402

from pinn_cables.io.readers import (  # noqa: E402
    load_problem,
    load_solver_params,
    override_conductor_Q,
)
from pinn_cables.physics.iec60287 import compute_iec60287_Q  # noqa: E402
from pinn_cables.physics.kennelly import iec60287_estimate  # noqa: E402
from pinn_cables.pinn.model import build_model, ResidualPINNModel  # noqa: E402
from pinn_cables.pinn.train import SteadyStatePINNTrainer  # noqa: E402
from pinn_cables.pinn.train_custom import (  # noqa: E402
    init_output_bias,
    pretrain_multicable,
)
from pinn_cables.pinn.utils import (  # noqa: E402
    get_device,
    set_seed,
    setup_logging,
)
from pinn_cables.post.eval import eval_conductor_temps, evaluate_on_grid  # noqa: E402
from pinn_cables.post.plots import (  # noqa: E402
    plot_cable_geometry,
    plot_loss_history,
    plot_temperature_field,
)


# ---------------------------------------------------------------------------
# Paper constants — Kim et al. (2024) Table 10 case 5 (summer critical)
# ---------------------------------------------------------------------------
PAPER_T_MAX_PAC = 343.75     # K  (70.6 degC)  — max cable B1, PAC summer
PAPER_T_MAX_SAND = 350.75    # K  (77.6 degC)  — max cable B1, sand summer
PAPER_T_MAX_PAC_W = 323.05   # K  (49.9 degC)  — max cable B1, PAC winter
PAPER_T_MAX_SAND_W = 330.25  # K  (57.1 degC)  — max cable B1, sand winter
PAPER_T_MAX = 363.15         # K  (90 degC)  — XLPE limit
PAPER_W_D = 3.57             # W/m  — dielectric losses in XLPE
PAPER_K_XLPE = 0.2857        # W/(mK)
PAPER_K_SCREEN = 384.6       # W/(mK)
PAPER_FREQ = 50.0            # Hz
PAPER_CURRENT = 1026.0       # A — critical ampacity (case 5)
PAPER_T_AMB_AIR = 300.35     # K  (27.2 degC)
PAPER_T_SUR = 290.15         # K  (17.0 degC) — cable-surrounding ground temp
PAPER_T_G = 288.35           # K  (15.2 degC) — constant ground temp at depth
PAPER_K_PAC = 2.094           # W/(mK) — PAC thermal conductivity
PAPER_K_NC = 2.093            # W/(mK) — normal concrete
PAPER_K_SAND = 1.365          # W/(mK) — natural sand
PAPER_K_CLSM = 2.150          # W/(mK) — CLSM filling casing pipe
PAPER_K_SOIL_L1 = 1.804       # W/(mK) — soil layer 1 (SC)
PAPER_K_SOIL_L2 = 1.351       # W/(mK) — soil layer 2 (CL)
PAPER_K_SOIL_L3 = 1.517       # W/(mK) — soil layer 3 (CL)
PAPER_LOAD_LOSS_FACTOR = 0.8  # KEPCO DS-6210
PAPER_WIND_SPEED_SUMMER = 1.17  # m/s

# Posiciones de los 6 cables (Fig. 7b del paper)
CABLE_SEP_H = 0.40   # m — separacion horizontal centro a centro
CABLE_SEP_V = 0.40   # m — separacion vertical entre filas
DEPTH_BOTTOM = 1.6    # m — profundidad fila inferior
DEPTH_TOP = 1.2       # m — profundidad fila superior


def main() -> None:
    """Cargar datos, entrenar PINN y comparar con Kim et al. (2024)."""
    parser = argparse.ArgumentParser(
        description="Benchmark PINN: Kim et al. (2024) — 6 cables XLPE 154 kV two-flat, PAC bedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Perfiles disponibles:\n"
            "  quick    : 5 000 Adam + 500 L-BFGS, red 64x4\n"
            "  research : 10 000 Adam + 1 000 L-BFGS, red 128x5\n"
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

    # Capas del cable (compartidas — mismo tipo de cable 1200mm² Cu)
    layers_template = problem.get_layers(all_placements[0].cable_id)

    section_mm2 = all_placements[0].section_mm2
    material = all_placements[0].conductor_material.upper()
    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A = all_placements[0].current_A
    n_cables = len(all_placements)

    # -----------------------------------------------------------------
    # Calculo de calor del paper
    # -----------------------------------------------------------------
    T_op_K = PAPER_T_MAX_PAC
    iec_q = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=T_op_K, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )

    Q_cond_iec = iec_q["Q_cond_W_per_m"]
    Q_d = PAPER_W_D
    Q_total_lin = Q_cond_iec + Q_d

    # Sobreescribir Q_vol del conductor
    layers = override_conductor_Q(layers_template, Q_total_lin)

    Q_lins = [Q_total_lin] * n_cables

    SEP = "=" * 72
    print(SEP)
    print("  BENCHMARK: Kim, Nguyen Cong, Dinh & Kim (2024)")
    print("  154 kV XLPE 6 cables two-flat + PAC bedding — PINN vs FEM")
    print("  Cable XLPE 1200 mm2 %s / I = %.0f A (critical ampacity)" % (
        material, current_A))
    print("  Perfil de ejecucion : %s" % profile.upper())
    print(SEP)

    # Cargar config
    params_csv = DATA_DIR / (
        "solver_params.csv" if profile == "quick" else "solver_params_research.csv"
    )
    solver_params = load_solver_params(params_csv)
    solver_cfg = solver_params.to_solver_cfg()

    adam_n = solver_cfg["training"]["adam_steps"]
    lbfgs_n = solver_cfg["training"]["lbfgs_steps"]
    print_ev = solver_cfg["training"]["print_every"]
    width = solver_cfg["model"]["width"]
    depth = solver_cfg["model"]["depth"]
    n_int = solver_cfg["sampling"]["n_interior"]
    n_ifc = solver_cfg["sampling"]["n_interface"]
    n_bnd = solver_cfg["sampling"]["n_boundary"]

    print("\n  Problema fisico:")
    print("  Escenario       : %s  (%s)" % (scenario.scenario_id, scenario.mode))
    print("  Cables          : %d x XLPE 1200 mm2 %s  I=%.0f A" % (
        n_cables, material, current_A))
    print("  Formacion       : two-flat (3 col x 2 filas), s_h=%.2f m, s_v=%.2f m" % (
        CABLE_SEP_H, CABLE_SEP_V))
    print("  Cable bedding   : PAC (lambda=%.3f W/(mK))" % PAPER_K_PAC)
    print()
    print("  Fuentes de calor (IEC 60287 @ T_op=%.1f degC):" % (T_op_K - 273.15))
    print("  Q_cond (I^2Rac) : %.2f W/m" % Q_cond_iec)
    print("  Q_d XLPE (vol.) : %.2f W/m" % Q_d)
    print("  Q_TOTAL (x1)    : %.2f W/m  (x%d = %.2f W/m)" % (
        Q_total_lin, n_cables, Q_total_lin * n_cables))
    print()
    print("  k_suelo efectivo: %.2f W/(m*K)" % scenario.k_soil)
    print("  T_amb (suelo)   : %.1f degC  (%.2f K)" % (
        scenario.T_amb - 273.15, scenario.T_amb))
    print("  T_amb_air       : %.1f degC  (conveccion top)" % (
        PAPER_T_AMB_AIR - 273.15))
    print("  T_g (fondo)     : %.1f degC = %.2f K" % (
        PAPER_T_G - 273.15, PAPER_T_G))
    print("  Posiciones:")
    for i, p in enumerate(all_placements):
        row = "B" if abs(p.cy) > 1.3 else "T"
        col = {-0.40: "izq", 0.00: "cen", 0.40: "der"}.get(round(p.cx, 2), "?")
        print("    Cable %d (%s%s): cx=%+.2f, cy=%.2f m" % (
            i + 1, row, col, p.cx, p.cy))
    print("  Dominio         : [%.1f, %.1f] x [%.1f, %.0f] m" % (
        problem.domain.xmin, problem.domain.xmax,
        problem.domain.ymin, problem.domain.ymax))

    # Estimacion analitica (unified)
    layers_list = [layers] * n_cables
    iec = iec60287_estimate(
        layers_list, all_placements, scenario.k_soil, scenario.T_amb,
        Q_lins=Q_lins, Q_d=Q_d,
    )
    T_ref_K = iec["T_cond_ref"]
    worst_idx = iec["hottest_idx"]

    # Compute self+mutual split for display
    r_sheath = layers[-1].r_outer
    d_worst = abs(all_placements[worst_idx].cy)
    dT_soil_self = Q_lins[worst_idx] / (2.0 * math.pi * scenario.k_soil) * math.log(
        2.0 * d_worst / r_sheath)
    dT_mutual = iec["cables"][worst_idx]["dT_soil"] - dT_soil_self

    print("\n  Referencia analitica (cable B1 central bottom, peor caso):")
    print("  Q total (lin.)   : %.2f W/m (cond %.2f + diel %.2f)" % (
        Q_total_lin, Q_cond_iec, Q_d))
    for name, dT in iec["dT_by_layer"].items():
        print("  dT %-10s   : %+.2f K" % (name, dT))
    print("  dT suelo propio  : %+.2f K" % dT_soil_self)
    print("  dT mutual (otros): %+.2f K" % dT_mutual)
    dT_total = T_ref_K - scenario.T_amb
    print("  dT TOTAL         : %+.2f K  -->  T_cond = %.1f K (%.1f degC)" % (
        dT_total, T_ref_K, T_ref_K - 273.15))

    print("\n  Referencia FEM Kim et al. (2024):")
    print("  T_max cable B1 (PAC, verano)  : %.1f degC" % (
        PAPER_T_MAX_PAC - 273.15))
    print("  T_max cable B1 (sand, verano) : %.1f degC" % (
        PAPER_T_MAX_SAND - 273.15))
    print("  T_max cable B1 (PAC, invierno): %.1f degC" % (
        PAPER_T_MAX_PAC_W - 273.15))

    # Modelo residual: T = T_bg(Kennelly 6 cables) + u(MLP)
    # T_bg captura interferencia termica entre cables; u corrige CC.
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="example_kim2024_" + profile)

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
        Q_d=Q_d,
    )
    init_output_bias(model.base, 0.0)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n  Pre-entrenando en perfil multicable (500 pasos)...", flush=True)
    rmse_pre = pretrain_multicable(
        model, all_placements, problem.domain,
        layers_list, Q_lins, scenario.k_soil, scenario.T_amb,
        device=device, normalize=normalize, Q_d=Q_d,
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
        title="6 Cables XLPE 154 kV Two-Flat + PAC — Kim et al. (2024)",
        save_path=geo_path,
    )

    # Entrenamiento: Adam (explorac. global) + L-BFGS (refinamiento)
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

    # Guardar modelo
    model_path = RESULTS_DIR / "model_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo guardado: %s", model_path)

    # Graficas
    loss_path = RESULTS_DIR / "loss_history.png"
    T_map_path = RESULTS_DIR / "temperature_field.png"
    plot_loss_history(
        history,
        title="Perdida — Kim et al. (2024) 154 kV Two-Flat PAC (%s)" % profile,
        save_path=loss_path,
    )
    X, Y, T = evaluate_on_grid(
        model, problem.domain, nx=300, ny=300,
        device=device, normalize=normalize,
    )
    plot_temperature_field(
        X, Y, T,
        title="T(x,y) [K] — 6 x XLPE 1200 mm2 %s %.0f A PAC [%s]" % (
            material, current_A, profile),
        save_path=T_map_path,
    )

    # -----------------------------------------------------------------
    # Resumen comparativo
    # -----------------------------------------------------------------
    T_max_pinn = float(T.max())
    T_min_pinn = float(T.min())

    # Evaluar T en el centro de CADA conductor
    T_cond_pinns = eval_conductor_temps(
        model, all_placements, problem.domain, device, normalize,
    )

    # Cable B1 (central bottom) es el peor caso
    T_cond_worst = T_cond_pinns[worst_idx]
    T_cond_max = max(T_cond_pinns)
    loss_final = history["total"][-1]
    error_K = T_cond_worst - T_ref_K
    error_vs_fem = T_cond_worst - PAPER_T_MAX_PAC

    C = 42
    print("\n" + SEP)
    print("  RESULTADOS — BENCHMARK Kim et al. (2024)")
    print("  6 x Cable XLPE 154 kV 1200 mm2 %s / I = %d A / Two-Flat + PAC" % (
        material, int(current_A)))
    print(SEP)

    print("\n  --- Temperaturas por cable ---")
    for i, (p, Tc) in enumerate(zip(all_placements, T_cond_pinns)):
        row = "B" if abs(p.cy) > 1.3 else "T"
        col = {-0.40: "izq", 0.00: "cen", 0.40: "der"}.get(round(p.cx, 2), "?")
        tag = " (*)" if i == worst_idx else ""
        print("  Cable %d (%s%s, cx=%+.2f cy=%.2f)%s : T = %.2f K (%.1f degC)" % (
            i + 1, row, col, p.cx, p.cy, tag, Tc, Tc - 273.15))

    print("\n  %-*s  %10s  %10s  %9s" % (C, "Magnitud", "PINN", "Referencia", "Error"))
    print("  " + "-" * 70)

    print("\n  --- PINN vs IEC 60287 analitico (cable B1) ---")
    print("  %-*s  %10.2f  %10.2f  %+8.2f K" % (
        C, "T conductor (K)", T_cond_worst, T_ref_K, error_K))
    print("  %-*s  %10.1f  %10.1f  %+8.1f K" % (
        C, "T conductor (degC)",
        T_cond_worst - 273.15, T_ref_K - 273.15, error_K))

    print("\n  --- PINN vs FEM COMSOL (Kim et al., PAC verano) ---")
    print("  %-*s  %10.2f  %10.2f  %+8.2f K" % (
        C, "T conductor (K)", T_cond_worst, PAPER_T_MAX_PAC, error_vs_fem))
    print("  %-*s  %10.1f  %10.1f  %+8.1f K" % (
        C, "T conductor (degC)",
        T_cond_worst - 273.15, PAPER_T_MAX_PAC - 273.15, error_vs_fem))

    print("\n  --- Comparacion entre metodos ---")
    print("  %-*s  %10s  %10s  %10s" % (C, "Metodo", "I (A)", "T_max", "Bedding"))
    print("  " + "-" * 70)
    print("  %-*s  %10d  %9.1f degC  %7s" % (
        C, "FEM COMSOL (PAC, verano)", int(current_A),
        PAPER_T_MAX_PAC - 273.15, "PAC"))
    print("  %-*s  %10d  %9.1f degC  %7s" % (
        C, "FEM COMSOL (sand, verano)", int(current_A),
        PAPER_T_MAX_SAND - 273.15, "sand"))
    print("  %-*s  %10d  %9.1f degC  %7s" % (
        C, "IEC analitico (este script)", int(current_A),
        T_ref_K - 273.15, "homog."))
    print("  %-*s  %10d  %9.1f degC  %7s" % (
        C, "PINN (cable B1)", int(current_A),
        T_cond_worst - 273.15, "homog."))

    print("\n  " + "-" * 70)
    print("  %-*s  %10.2f" % (C, "T max dominio (K)", T_max_pinn))
    print("  %-*s  %10.2f" % (C, "T min dominio (K)", T_min_pinn))
    print("  %-*s  %10.4e" % (C, "Perdida final", loss_final))
    print("  " + "-" * 70)
    print("  Limite IEC XLPE: 363 K (90 degC)")
    print("  Margen termico PINN (B1): %.1f K" % (363.0 - T_cond_worst))
    print("  Reduccion PAC vs sand (paper): %.1f K" % (
        PAPER_T_MAX_SAND - PAPER_T_MAX_PAC))

    if profile == "quick":
        print()
        print("  Para resultados de investigacion:")
        print("    python examples/kim_2024_154kv_bedding/run_example.py --profile research")
    print()
    print("  Graficas guardadas en: %s" % RESULTS_DIR)
    print("    - geometry.png  |  loss_history.png  |  temperature_field.png")
    print("    - model_final.pt  (pesos de la red entrenada)")


if __name__ == "__main__":
    main()
