"""Benchmark Kim et al. (2024) — RESEARCH profile con zona PAC explicita.

Mejoras respecto a ``run_example.py --profile quick``:

1. **Red mas grande**: MLP 128x5 con Fourier features (10 000 Adam + 1 000 L-BFGS)
2. **Zona PAC heterogenea**: conductividad k_PAC = 2.094 W/(mK) dentro
   del duct-bank y k_soil = 1.55 W/(mK) fuera, con transicion suave
   (sigmoid) para evitar discontinuidades en el gradiente.

La zona PAC se modela como un rectangulo suavizado centrado en la
region donde se encuentran los cables (duct-bank):

    k(x,y) = k_bad + (k_good - k_bad) * sigmoid(-d / k_transition)

donde d = max(|x - k_cx| - k_width/2, |y - k_cy| - k_height/2).

Uso::

    python examples/kim_2024_154kv_bedding/run_research_pac.py
    python examples/kim_2024_154kv_bedding/run_research_pac.py --profile research

Comparacion automatica con resultados previos (quick, homogeneo).
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

# Perfiles disponibles: nombre -> (solver_csv, results_subdir)
PROFILES = {
    "quick":    ("solver_params.csv",          "results_pac_quick"),
    "research": ("solver_params_research.csv",  "results_pac_research"),
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark Kim et al. (2024) — PINN + zona PAC")
    ap.add_argument(
        "--profile", "-p",
        choices=list(PROFILES.keys()),
        default="quick",
        help="Perfil de solver: quick (64x4, ~5k Adam) o research (128x5, 10k Adam). Default: quick",
    )
    return ap.parse_args()


import matplotlib  # noqa: E402
matplotlib.use("Agg")
import torch  # noqa: E402

from pinn_cables.io.readers import (  # noqa: E402
    load_problem,
    load_solver_params,
    override_conductor_Q,
)
from pinn_cables.io.report import write_bc_report  # noqa: E402
from pinn_cables.physics.iec60287 import compute_iec60287_Q  # noqa: E402
from pinn_cables.physics.k_field import (  # noqa: E402
    load_physics_params,
    make_k_functions,
)
from pinn_cables.physics.kennelly import iec60287_estimate  # noqa: E402
from pinn_cables.pinn.model import (  # noqa: E402
    ResidualPINNModel,
    build_model,
)
from pinn_cables.pinn.train_custom import (  # noqa: E402
    init_output_bias,
    pretrain_multicable,
    train_adam_lbfgs,
)
from pinn_cables.pinn.utils import (  # noqa: E402
    get_device,
    set_seed,
    setup_logging,
)
from pinn_cables.post.eval import (  # noqa: E402
    eval_conductor_temps,
    evaluate_on_grid,
)
from pinn_cables.post.plots import (  # noqa: E402
    plot_comparison_bar,
    plot_k_field,
    plot_loss_history,
    plot_temperature_field,
    plot_zoom_temperature,
)

# ---------------------------------------------------------------------------
# Constantes del paper
# ---------------------------------------------------------------------------
PAPER_T_MAX_PAC = 273.15 + 70.6    # K — FEM T_max con PAC bedding
PAPER_T_MAX_SAND = 273.15 + 77.6   # K — FEM T_max con sand bedding
PAPER_W_D = 3.57                    # W/m — perdidas dielectricas
PAPER_FREQ = 60.0                   # Hz
PAPER_K_PAC = 2.094                 # W/(mK)


# ---------------------------------------------------------------------------
# Helper: cargar modelo quick para comparacion
# ---------------------------------------------------------------------------

def _load_quick_results(problem, layers_list, Q_lins, device):
    """Intentar cargar modelo quick (k homogeneo) para comparar."""
    quick_model_path = HERE / "results" / "model_final.pt"
    if not quick_model_path.exists():
        return None

    try:
        sp_q = load_solver_params(DATA_DIR / "solver_params.csv")
        cfg_q = sp_q.to_solver_cfg()
        norm_q = cfg_q.get("normalization", {}).get("normalize_coords", True)

        base_q = build_model(cfg_q["model"], in_dim=2, device=device)

        scenario = problem.scenarios[0]
        model_q = ResidualPINNModel(
            base_q, layers_list, problem.placements,
            scenario.k_soil, scenario.T_amb, Q_lins,
            problem.domain, normalize=norm_q,
        )
        model_q.load_state_dict(
            torch.load(quick_model_path, map_location=device, weights_only=True))
        model_q.eval()

        T_quick = eval_conductor_temps(
            model_q, problem.placements, problem.domain, device, norm_q)
        return T_quick
    except Exception as e:
        print("  (No se pudieron cargar resultados quick: %s)" % e)
        return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    profile = args.profile
    solver_csv, results_subdir = PROFILES[profile]
    RESULTS_DIR = HERE / results_subdir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    profile_tag = profile.upper()

    SEP = "=" * 72

    # --- Cargar problema y parametros ---
    problem = load_problem(DATA_DIR)
    scenario = problem.scenarios[0]
    all_placements = problem.placements
    n_cables = len(all_placements)

    pp = load_physics_params(DATA_DIR / "physics_params.csv")

    solver_params = load_solver_params(DATA_DIR / solver_csv)
    solver_cfg = solver_params.to_solver_cfg()

    adam_n = solver_cfg["training"]["adam_steps"]
    lbfgs_n = solver_cfg["training"]["lbfgs_steps"]
    lbfgs_hist = solver_cfg["training"].get("lbfgs_history", 50)
    adam2_n = solver_cfg["training"].get("adam2_steps", 0)
    adam2_lr_val = solver_cfg["training"].get("adam2_lr", 1e-5)
    print_ev = solver_cfg["training"]["print_every"]
    width = solver_cfg["model"]["width"]
    depth = solver_cfg["model"]["depth"]
    n_int = solver_cfg["sampling"]["n_interior"]
    n_bnd = solver_cfg["sampling"]["n_boundary"]
    oversamp = solver_cfg["sampling"]["oversample"]
    normalize = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    w_pde = solver_cfg["loss_weights"].get("pde", 1.0)
    w_bc = solver_cfg["loss_weights"].get("bc_dirichlet", 10.0)
    k_soil = scenario.k_soil
    T_amb = scenario.T_amb

    # --- Capas y Q por cable ---
    section_mm2 = all_placements[0].section_mm2
    material = all_placements[0].conductor_material.upper()
    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A = all_placements[0].current_A

    layers_template = problem.get_layers(all_placements[0].cable_id)

    T_op_K = PAPER_T_MAX_PAC
    iec_q = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=T_op_K, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_cond_iec = iec_q["Q_cond_W_per_m"]
    Q_d = PAPER_W_D
    Q_total_lin = Q_cond_iec + Q_d

    layers = override_conductor_Q(layers_template, Q_total_lin)
    Q_lins = [Q_total_lin] * n_cables

    layers_list = [layers] * n_cables
    r_sheaths = [layers[-1].r_outer] * n_cables

    # --- k(x,y) variable (zona PAC / suelo mejorado) ---
    k_fn_pde, k_eff_fn_iec, k_eff_bg = make_k_functions(
        pp, k_soil, placements=all_placements,
    )

    # --- Imprimir informacion ---
    print(SEP)
    print("  BENCHMARK Kim et al. (2024) — %s + ZONA PAC" % profile_tag)
    print("  154 kV XLPE 6 cables two-flat — k heterogeneo")
    print("  Cable XLPE %d mm² %s / I = %.0f A" % (section_mm2, material, current_A))
    print("  Perfil: %s  |  Solver: %s" % (profile, solver_csv))
    print(SEP)
    write_bc_report(
        problem, RESULTS_DIR,
        label="Kim (2024) — PAC zona k variable — %s" % profile_tag,
    )
    if pp.k_variable:
        print("\n  ZONA PAC (duct-bank) — patron sigmoide:")
        print("    k_good (PAC)  = %.3f W/(mK)" % pp.k_good)
        print("    k_bad (suelo) = %.2f W/(mK)" % pp.k_bad)
        print("    Centro        : (%.2f, %.2f) m" % (pp.k_cx, pp.k_cy))
        print("    Dimension     : %.2f x %.2f m" % (pp.k_width, pp.k_height))
        print("    Transicion    : %.3f m" % pp.k_transition)
        print("    k_eff_bg      : %.3f W/(mK) (en centroide)" % k_eff_bg)

    print("\n  Fuentes de calor:")
    print("    Q_cond = %.2f W/m  +  Q_d = %.2f W/m  =  Q_total = %.2f W/m" % (
        Q_cond_iec, Q_d, Q_total_lin))

    # IEC analitico (con k variable)
    iec_est = iec60287_estimate(
        layers_list, all_placements, k_soil, T_amb, Q_lins,
        Q_d=Q_d, k_eff_fn=k_eff_fn_iec,
    )
    T_iec = iec_est["T_cond_ref"]
    print("  Estimacion IEC (k variable) : %.1f K (%.1f °C)" % (T_iec, T_iec - 273.15))

    # IEC sin k variable (k homogeneo)
    iec_hom = iec60287_estimate(
        layers_list, all_placements, k_soil, T_amb, Q_lins, Q_d=Q_d,
    )
    T_iec_hom = iec_hom["T_cond_ref"]
    print("  Estimacion IEC (k homog.)   : %.1f K (%.1f °C)" % (T_iec_hom, T_iec_hom - 273.15))

    # Grafica campo k
    plot_k_field(
        problem.domain, pp, all_placements, layers_list,
        RESULTS_DIR / "k_field_pac_zone.png",
    )

    # --- Modelo ---
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="kim2024_research_pac")

    base_model = build_model(solver_cfg["model"], in_dim=2, device=device)
    # NOTA: T_bg (Kennelly) usa k_soil (homogeneo far-field = 1.55 W/(mK)),
    # NO k_eff_bg (PAC centroide = 2.094); la diferencia entre k_soil y k(x,y)
    # la resuelve la correccion u via la PDE div(k(x,y)*grad(T_bg + u))=0.
    # enable_grad_Tbg=True permite que autograd propague grad a traves de T_bg,
    # necesario cuando k(x,y) varia y el Laplaciano de T_bg no es cero.
    model = ResidualPINNModel(
        base_model, layers_list, all_placements,
        k_soil, T_amb, Q_lins,
        problem.domain, normalize=normalize, Q_d=Q_d,
        enable_grad_Tbg=True,
    )
    init_output_bias(model.base, 0.0)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n  Red: MLP %dx%d (%d params)%s" % (
        width, depth, n_params,
        ", Fourier features" if solver_cfg["model"].get("fourier_features", False) else ""))
    print("  Entrenamiento: %d Adam + %d L-BFGS%s" % (
        adam_n, lbfgs_n,
        " + %d Adam2" % adam2_n if adam2_n > 0 else ""))

    # Pre-entrenamiento
    print("\n  Pre-entrenando (1000 pasos, 2000 pts/cable)...", flush=True)
    rmse_pre = pretrain_multicable(
        model, all_placements, problem.domain,
        layers_list, Q_lins, k_soil, T_amb,
        device=device, normalize=normalize,
        n_per_cable=2000, n_bc=300, n_steps=1000, lr=1e-3,
    )
    print("  Pre-entrenamiento OK: RMSE = %.3f K" % rmse_pre, flush=True)

    # JIT trace del MLP base: compila el forward pass para ~15% speedup
    # en Adam/L-BFGS. Seguro dado que el MLP es puramente feed-forward.
    try:
        dummy = torch.randn(64, 2, device=device)
        model.base = torch.jit.trace(model.base, dummy)
        print("  JIT trace aplicado al MLP base (speedup ~15%%)")
    except Exception as e:
        print("  JIT trace no disponible: %s" % e)

    # Entrenamiento
    print("\n" + "-" * 72)
    print("  ENTRENAMIENTO %s + PAC ZONE (curriculum)" % profile_tag)
    print("-" * 72)

    # Curriculum: los primeros pasos (30%) usan k homogeneo para estabilizar
    # la red, luego se activa k(x,y) variable (con transicion sigmoide).
    # Esto mejora la convergencia cuando el salto k_good/k_bad es grande.
    WARMUP_FRAC = 0.30 if pp.k_variable else 0.0

    def k_fn_homog(xy_phys: torch.Tensor) -> torch.Tensor:
        return torch.full((xy_phys.shape[0], 1), k_soil,
                          device=xy_phys.device, dtype=xy_phys.dtype)

    history = train_adam_lbfgs(
        model=model,
        domain=problem.domain,
        placements=all_placements,
        bcs=problem.bcs,
        T_amb=T_amb,
        r_sheaths=r_sheaths,
        k_fn=k_fn_pde,
        adam_steps=adam_n,
        lbfgs_steps=lbfgs_n,
        lbfgs_history=lbfgs_hist,
        n_int=n_int,
        n_bnd=n_bnd,
        oversample=oversamp,
        w_pde=w_pde,
        w_bc=w_bc,
        lr=solver_cfg["training"]["lr"],
        print_every=print_ev,
        normalize=normalize,
        device=device,
        logger=logger,
        pp=pp,
        k_fn_warmup=k_fn_homog if pp.k_variable else None,
        warmup_frac=WARMUP_FRAC,
        adam2_steps=adam2_n,
        adam2_lr=adam2_lr_val,
    )
    print("-" * 72)

    # Guardar
    model_path = RESULTS_DIR / "model_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo guardado: %s", model_path)

    # Graficas
    plot_loss_history(
        history,
        title="Pérdida — Kim et al. (2024) Research + PAC zone",
        save_path=RESULTS_DIR / "loss_history.png",
    )
    X, Y, T = evaluate_on_grid(
        model, problem.domain, nx=300, ny=300,
        device=device, normalize=normalize,
    )
    plot_temperature_field(
        X, Y, T,
        title="T(x,y) [K] — Research + PAC zone — 6 × XLPE 1200 mm² 1026 A",
        save_path=RESULTS_DIR / "temperature_field.png",
    )

    # Zoom con etiquetas de cable
    cable_labels = []
    for p in all_placements:
        row = "B" if abs(p.cy) > 1.3 else "T"
        col = {-0.40: "1", 0.00: "2", 0.40: "3"}.get(round(p.cx, 2), "?")
        cable_labels.append("%s%s" % (row, col))
    plot_zoom_temperature(
        model, problem.domain, all_placements, layers_list,
        device, normalize, RESULTS_DIR / "temperature_zoom_pac.png",
        zoom=(-1.5, 1.5, -2.5, -0.5), pp=pp,
        celsius=True, annotate_max=True,
        cable_labels=cable_labels,
        title=(
            "T [°C] — Research + PAC zone\n"
            "6 × XLPE 154 kV 1200 mm² Cu @ 1026 A"
        ),
    )

    # --- Evaluar conductores ---
    T_cond_research = eval_conductor_temps(
        model, all_placements, problem.domain, device, normalize,
    )
    T_worst_idx = T_cond_research.index(max(T_cond_research))
    T_worst_research = T_cond_research[T_worst_idx]

    # --- Cargar resultados quick ---
    print("\n  Cargando resultados QUICK (k homogeneo) para comparar...")
    T_quick_list = _load_quick_results(problem, layers_list, Q_lins, device)
    T_worst_quick = max(T_quick_list) if T_quick_list else None

    # =================================================================
    # TABLA COMPARATIVA
    # =================================================================
    loss_final = history["total"][-1]
    C = 44

    print("\n" + SEP)
    print("  RESULTADOS — %s + ZONA PAC" % profile_tag)
    print(SEP)

    print("\n  --- Temperaturas por cable (%s + PAC) ---" % profile_tag)
    for i, (p, Tc) in enumerate(zip(all_placements, T_cond_research)):
        row = "B" if abs(p.cy) > 1.3 else "T"
        col = {-0.40: "izq", 0.00: "cen", 0.40: "der"}.get(round(p.cx, 2), "?")
        tag = " <-- peor" if i == T_worst_idx else ""
        print("  Cable %d (%s%s): T_cond = %.2f K (%.1f degC)%s" % (
            i + 1, row, col, Tc, Tc - 273.15, tag))

    print("\n  " + "=" * 74)
    print("  %-*s  %10s  %10s  %9s" % (C, "Metodo", "T_max", "Error/FEM", "k modelo"))
    print("  " + "-" * 74)

    print("  %-*s  %9.1f °C  %9s  %9s" % (
        C, "FEM COMSOL — PAC (Kim et al.)",
        PAPER_T_MAX_PAC - 273.15, "ref", "heterog."))
    print("  %-*s  %9.1f °C  %9s  %9s" % (
        C, "FEM COMSOL — sand (Kim et al.)",
        PAPER_T_MAX_SAND - 273.15,
        "%+.1f K" % (PAPER_T_MAX_SAND - PAPER_T_MAX_PAC), "heterog."))
    print("  %-*s  %9.1f °C  %9s  %9s" % (
        C, "IEC 60287 (k homogeneo)",
        T_iec_hom - 273.15,
        "%+.1f K" % (T_iec_hom - PAPER_T_MAX_PAC), "homog."))
    print("  %-*s  %9.1f °C  %9s  %9s" % (
        C, "IEC 60287 (k PAC variable)",
        T_iec - 273.15,
        "%+.1f K" % (T_iec - PAPER_T_MAX_PAC), "PAC zone"))
    if T_worst_quick is not None:
        print("  %-*s  %9.1f °C  %9s  %9s" % (
            C, "PINN quick (64x4, 5.5k steps, k homog.)",
            T_worst_quick - 273.15,
            "%+.1f K" % (T_worst_quick - PAPER_T_MAX_PAC), "homog."))
    pinn_label = "PINN %s+PAC (%dx%d, %dk steps)" % (
        profile, width, depth, (adam_n + lbfgs_n) // 1000)
    print("  %-*s  %9.1f °C  %9s  %9s" % (
        C, pinn_label,
        T_worst_research - 273.15,
        "%+.1f K" % (T_worst_research - PAPER_T_MAX_PAC), "PAC zone"))
    print("  " + "=" * 74)

    if T_worst_quick is not None:
        mejora = abs(T_worst_quick - PAPER_T_MAX_PAC) - abs(T_worst_research - PAPER_T_MAX_PAC)
        print("\n  Mejora research+PAC vs quick: %.1f K (|error| de %.1f -> %.1f K)" % (
            mejora,
            abs(T_worst_quick - PAPER_T_MAX_PAC),
            abs(T_worst_research - PAPER_T_MAX_PAC),
        ))

    print("\n  Perdida final: %.4e" % loss_final)
    print("  Margen termico (90°C): %.1f K" % (363.15 - T_worst_research))

    # Grafico de barras comparativo
    bar_data = {
        "FEM PAC\n(Kim et al.)": PAPER_T_MAX_PAC,
        "FEM sand\n(Kim et al.)": PAPER_T_MAX_SAND,
        "IEC 60287\n(k homogéneo)": T_iec_hom,
        "IEC 60287\n(k PAC var.)": T_iec,
    }
    if T_worst_quick is not None:
        bar_data["PINN quick\n(k homogéneo)"] = T_worst_quick
    bar_data["PINN %s\n+ PAC zone" % profile] = T_worst_research
    plot_comparison_bar(
        bar_data, RESULTS_DIR / "comparison_bar.png",
        ref_line=(70.6, "FEM PAC = 70.6 °C"),
        limit_line=(90.0, "Límite XLPE = 90 °C"),
        title="Comparación T_max — Kim et al. (2024) Benchmark",
    )

    print("\n  Graficas guardadas en: %s" % RESULTS_DIR)
    print("    k_field_pac_zone.png     — campo k(x,y)")
    print("    loss_history.png         — curva de perdida")
    print("    temperature_field.png    — mapa T dominio completo")
    print("    temperature_zoom_pac.png — zoom cables + duct-bank")
    print("    comparison_bar.png       — comparacion metodos")
    print()


if __name__ == "__main__":
    main()
