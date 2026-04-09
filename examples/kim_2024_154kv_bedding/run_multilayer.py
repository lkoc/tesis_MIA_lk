"""Kim et al. (2024) — multilayer soil + CLSM/PE-casing cable model.

Runs two cases using the full paper geometry:

  Case A (multilayer soil, no PAC)
    Three horizontal soil layers (Sandy Clay → Clay Loam → Clay Loam) with
    k values from Table 6.  No special backfill.  Reference: FEM sand 77.6 °C.

  Case B (multilayer soil + PAC zone)
    Same three soil layers, plus the PAC backfill zone (k_PAC = 2.094 W/(mK))
    around the cables.  Reference: FEM PAC 70.6 °C.

Both cases use the 9-layer Kim 2024 cable cross-section (Table 2), which
includes CLSM (k = 2.150 W/(mK)) and PE casing pipe layers absent from the
Aras 2005 model used in ``run_example.py``.

Usage::

    python examples/kim_2024_154kv_bedding/run_multilayer.py
    python examples/kim_2024_154kv_bedding/run_multilayer.py --case A
    python examples/kim_2024_154kv_bedding/run_multilayer.py --case B
    python examples/kim_2024_154kv_bedding/run_multilayer.py --profile research
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

# Solver profiles
PROFILES = {
    "quick":    ("solver_params.csv",          "results_multilayer_quick"),
    "research": ("solver_params_research.csv",  "results_multilayer_research"),
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Kim et al. (2024) — multilayer soil + CLSM cable PINN")
    ap.add_argument(
        "--case", "-c",
        choices=["A", "B", "both"],
        default="both",
        help="A = multilayer only | B = multilayer + PAC | both (default: both)",
    )
    ap.add_argument(
        "--profile", "-p",
        choices=list(PROFILES.keys()),
        default="quick",
        help="Solver profile (default: quick)",
    )
    return ap.parse_args()


import matplotlib  # noqa: E402
matplotlib.use("Agg")
import torch  # noqa: E402

from pinn_cables.io.readers import (  # noqa: E402
    load_problem,
    load_solver_params,
)
from pinn_cables.materials.props import get_kim2024_cable_layers  # noqa: E402
from pinn_cables.physics.iec60287 import compute_iec60287_Q  # noqa: E402
from pinn_cables.physics.k_field import (  # noqa: E402
    KFieldModel,
    load_physics_params,
    load_soil_layers,
)
from pinn_cables.physics.kennelly import iec60287_estimate  # noqa: E402
from pinn_cables.pinn.model import ResidualPINNModel, build_model  # noqa: E402
from pinn_cables.pinn.train_custom import (  # noqa: E402
    init_output_bias,
    pretrain_multicable,
    train_adam_lbfgs,
)
from pinn_cables.pinn.utils import get_device, set_seed, setup_logging  # noqa: E402
from pinn_cables.post.eval import eval_conductor_temps, evaluate_on_grid  # noqa: E402
from pinn_cables.post.plots import (  # noqa: E402
    plot_k_field,
    plot_loss_history,
    plot_temperature_field,
    plot_zoom_temperature,
)

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------
PAPER_T_MAX_PAC  = 273.15 + 70.6   # K — FEM T_max with PAC bedding   (Table 10, case 5 summer)
PAPER_T_MAX_SAND = 273.15 + 77.6   # K — FEM T_max with sand bedding
PAPER_W_D  = 3.57                  # W/m — dielectric losses
PAPER_FREQ = 60.0                  # Hz
PAPER_K_PAC = 2.094                # W/(mK) — PAC thermal conductivity


# ---------------------------------------------------------------------------
# Helper: build KFieldModel for each case
# ---------------------------------------------------------------------------

def _build_k_model_A(soil_bands) -> KFieldModel:
    """Case A: multilayer soil only, no PAC zone.

    k(x,y) = k_L(y) — the native soil layer at depth y.
    """
    return KFieldModel(
        k_soil=1.351,          # Layer 2 k (where cables are located, 0.56–1.76 m)
        soil_bands=soil_bands,
    )


def _build_k_model_B(soil_bands, pac_params) -> KFieldModel:
    """Case B: multilayer soil + PAC backfill zone.

    k(x,y) = k_L(y) + [k_PAC - k_L(y)] * sigma_PAC(x,y)
    """
    return KFieldModel(
        k_soil=1.351,
        soil_bands=soil_bands,
        pac_params=pac_params,
    )


# ---------------------------------------------------------------------------
# Run one case
# ---------------------------------------------------------------------------

def run_case(
    case_tag: str,           # "A" or "B"
    k_model: KFieldModel,
    problem,
    layers_list: list,
    Q_lins: list[float],
    Q_d: float,
    solver_cfg: dict,
    results_dir: Path,
    device: torch.device,
    logger,
) -> tuple[float, dict]:
    """Train the PINN for one case and return (T_worst_K, history)."""
    T_amb = problem.scenarios[0].T_amb
    all_placements = problem.placements
    r_sheaths = [layers_list[i][-1].r_outer for i in range(len(all_placements))]

    adam_n    = solver_cfg["training"]["adam_steps"]
    lbfgs_n   = solver_cfg["training"]["lbfgs_steps"]
    adam2_n   = solver_cfg["training"].get("adam2_steps", 0)
    adam2_lr_v = solver_cfg["training"].get("adam2_lr", 1e-5)
    print_ev  = solver_cfg["training"]["print_every"]
    n_int    = solver_cfg["sampling"]["n_interior"]
    n_bnd    = solver_cfg["sampling"]["n_boundary"]
    oversamp = solver_cfg["sampling"]["oversample"]
    normalize = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    w_pde = solver_cfg["loss_weights"].get("pde", 1.0)
    w_bc  = solver_cfg["loss_weights"].get("bc_dirichlet", 10.0)
    lr    = solver_cfg["training"]["lr"]

    # k_soil for Kennelly background = k at cable centroid (Layer 2 soil)
    k_soil_bg = k_model.k_eff_bg(all_placements)

    # Build model
    set_seed(solver_cfg.get("seed", 42))
    base = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModel(
        base, layers_list, all_placements,
        k_soil_bg, T_amb, Q_lins,
        problem.domain, normalize=normalize, Q_d=Q_d,
        enable_grad_Tbg=True,
    )
    init_output_bias(model.base, 0.0)

    # Pre-train
    pretrain_multicable(
        model, all_placements, problem.domain,
        layers_list, Q_lins, k_soil_bg, T_amb,
        device=device, normalize=normalize,
        n_per_cable=2000, n_bc=300, n_steps=1000, lr=1e-3,
        Q_d=Q_d,
    )

    # JIT trace
    try:
        dummy = torch.randn(64, 2, device=device)
        model.base = torch.jit.trace(model.base, dummy)
    except Exception:
        pass

    # Warmup k (homogeneous)
    def k_fn_homog(xy: torch.Tensor) -> torch.Tensor:
        return torch.full((xy.shape[0], 1), k_soil_bg,
                          device=xy.device, dtype=xy.dtype)

    # Train
    history = train_adam_lbfgs(
        model=model,
        domain=problem.domain,
        placements=all_placements,
        bcs=problem.bcs,
        T_amb=T_amb,
        r_sheaths=r_sheaths,
        k_fn=k_model,
        adam_steps=adam_n,
        lbfgs_steps=lbfgs_n,
        n_int=n_int,
        n_bnd=n_bnd,
        oversample=oversamp,
        w_pde=w_pde,
        w_bc=w_bc,
        lr=lr,
        print_every=print_ev,
        normalize=normalize,
        device=device,
        logger=logger,
        k_fn_warmup=k_fn_homog,
        warmup_frac=0.30,
        k_model=k_model,
        adam2_steps=adam2_n,
        adam2_lr=adam2_lr_v,
    )

    # Save
    torch.save(model.state_dict(), results_dir / ("model_case_%s.pt" % case_tag))

    # Plots
    plot_loss_history(
        history,
        title="Loss — Kim 2024 Case %s (multilayer soil + CLSM)" % case_tag,
        save_path=results_dir / ("loss_case_%s.png" % case_tag),
    )
    X_g, Y_g, T_g = evaluate_on_grid(
        model, problem.domain, nx=300, ny=300,
        device=device, normalize=normalize,
    )
    plot_temperature_field(
        X_g, Y_g, T_g,
        title="T [K] — Case %s" % case_tag,
        save_path=results_dir / ("temperature_field_case_%s.png" % case_tag),
    )

    cable_labels = []
    for p in all_placements:
        row = "B" if abs(p.cy) > 1.3 else "T"
        col = {-0.40: "1", 0.00: "2", 0.40: "3"}.get(round(p.cx, 2), "?")
        cable_labels.append("%s%s" % (row, col))

    plot_zoom_temperature(
        model, problem.domain, all_placements, layers_list,
        device, normalize,
        results_dir / ("zoom_case_%s.png" % case_tag),
        zoom=(-1.5, 1.5, -2.5, -0.5),
        pp=k_model.pac_params,
        celsius=True, annotate_max=True,
        cable_labels=cable_labels,
        title="T [°C] — Case %s (multilayer + CLSM)" % case_tag,
    )

    T_conds = eval_conductor_temps(model, all_placements, problem.domain, device, normalize)
    T_worst = max(T_conds)
    return T_worst, history


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    solver_csv, results_subdir = PROFILES[args.profile]
    RESULTS_DIR = HERE / results_subdir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 72

    # ----------------------------------------------------------------
    # Load problem data
    # ----------------------------------------------------------------
    problem = load_problem(DATA_DIR)
    scenario = problem.scenarios[0]
    all_placements = problem.placements
    n_cables = len(all_placements)

    # Soil layer bands (Kim 2024, Table 6)
    soil_bands = load_soil_layers(DATA_DIR / "soil_layers.csv")

    # PAC zone params (physics_params.csv — same as run_research_pac.py)
    pac_params = load_physics_params(DATA_DIR / "physics_params.csv")

    solver_params = load_solver_params(DATA_DIR / solver_csv)
    solver_cfg = solver_params.to_solver_cfg()

    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="kim2024_multilayer")

    # ----------------------------------------------------------------
    # Cable layers — Kim 2024 9-layer structure (Table 2)
    # ----------------------------------------------------------------
    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A   = all_placements[0].current_A
    section_mm2 = all_placements[0].section_mm2

    # IEC Q (use PAC reference T_op for consistency with run_research_pac.py)
    iec_q = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=PAPER_T_MAX_PAC, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_cond = iec_q["Q_cond_W_per_m"]
    Q_d    = PAPER_W_D
    Q_lin  = Q_cond + Q_d

    layers = get_kim2024_cable_layers(Q_lin)
    layers_list = [layers] * n_cables
    Q_lins = [Q_lin] * n_cables

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    print(SEP)
    print("  BENCHMARK Kim et al. (2024) — MULTILAYER SOIL + CLSM CABLE")
    print("  154 kV XLPE 6 cables two-flat  |  Profile: %s" % args.profile)
    print(SEP)
    print("\n  Soil layers (Table 6, Kim 2024):")
    for b in soil_bands:
        print("    Layer y=[%.2f, %.2f] m:  k = %.3f W/(mK)" % (
            b.y_top, b.y_bottom, b.k))
    print("  Note: cables at y=-1.2 m and y=-1.6 m are in Layer 2 (k=%.3f)" % soil_bands[1].k)
    print("\n  Cable: 9-layer Kim 2024 structure (Table 2)")
    print("    Outermost layer (PE casing): r = %.0f mm" % (layers[-1].r_outer * 1e3))
    print("    CLSM layer:                  k = %.3f W/(mK)" % layers[-2].k)
    print("  Q_cond = %.2f W/m  +  Q_d = %.2f W/m  =  %.2f W/m" % (Q_cond, Q_d, Q_lin))

    # ----------------------------------------------------------------
    # Run cases
    # ----------------------------------------------------------------
    run_A = args.case in ("A", "both")
    run_B = args.case in ("B", "both")
    results: dict[str, float] = {}

    # --- Case A ---
    if run_A:
        k_model_A = _build_k_model_A(soil_bands)
        print("\n" + "-" * 72)
        print("  CASE A — Multilayer soil only (no PAC)")
        print("  k_model: %s" % k_model_A)
        # Plot k field for Case A
        plot_k_field(
            problem.domain, None, all_placements, layers_list,
            RESULTS_DIR / "k_field_case_A.png",
            title="k(x,y) — Case A: 3 soil layers, no PAC",
            k_model=k_model_A,
        )
        T_A, _ = run_case(
            "A", k_model_A, problem, layers_list, Q_lins, Q_d,
            solver_cfg, RESULTS_DIR, device, logger,
        )
        results["A"] = T_A

    # --- Case B ---
    if run_B:
        k_model_B = _build_k_model_B(soil_bands, pac_params)
        print("\n" + "-" * 72)
        print("  CASE B — Multilayer soil + PAC zone")
        print("  k_model: %s" % k_model_B)
        # Plot k field for Case B
        plot_k_field(
            problem.domain, None, all_placements, layers_list,
            RESULTS_DIR / "k_field_case_B.png",
            title="k(x,y) — Case B: 3 soil layers + PAC zone",
            k_model=k_model_B,
        )
        T_B, _ = run_case(
            "B", k_model_B, problem, layers_list, Q_lins, Q_d,
            solver_cfg, RESULTS_DIR, device, logger,
        )
        results["B"] = T_B

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    C = 46
    print("\n" + SEP)
    print("  SUMMARY — Kim et al. (2024) Multilayer + CLSM")
    print(SEP)
    print("  %-*s  %10s  %10s" % (C, "Method", "T_max [°C]", "Error / FEM"))
    print("  " + "-" * 68)
    print("  %-*s  %10.1f  %10s" % (
        C, "FEM COMSOL — sand bedding (Kim 2024)", PAPER_T_MAX_SAND - 273.15, "ref"))
    print("  %-*s  %10.1f  %10.1f K" % (
        C, "FEM COMSOL — PAC bedding  (Kim 2024)", PAPER_T_MAX_PAC - 273.15,
        PAPER_T_MAX_PAC - PAPER_T_MAX_SAND))
    if "A" in results:
        err = results["A"] - PAPER_T_MAX_SAND
        print("  %-*s  %10.1f  %+10.1f K" % (
            C, "PINN Case A (multilayer soil, no PAC)",
            results["A"] - 273.15, err))
    if "B" in results:
        err = results["B"] - PAPER_T_MAX_PAC
        print("  %-*s  %10.1f  %+10.1f K" % (
            C, "PINN Case B (multilayer soil + PAC)",
            results["B"] - 273.15, err))
    print(SEP)


if __name__ == "__main__":
    main()
