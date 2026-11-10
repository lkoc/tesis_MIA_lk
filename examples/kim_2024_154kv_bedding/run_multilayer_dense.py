"""Kim et al. (2024) — Multilayer Cases A & B with dense network.

Purpose
-------
Obtain a solution that is *independent of network size* (analogous to a
mesh-independent solution in FEM).  The dense profile uses a significantly
larger network (256×6) with more training steps and collocation points than
the quick/research profiles.  By comparing quick → research → dense we can
verify that T_max has converged as a function of network capacity.

Cases
-----
  Case A  multilayer soil only (no PAC).   Reference: FEM sand  77.6 °C
  Case B  multilayer soil + PAC zone.      Reference: FEM PAC   70.6 °C

Network comparison (capacity)
------------------------------
  quick    :  64×4   ~  17 k params  |  5 000 Adam +   500 L-BFGS
  research :  128×5  ~  83 k params  | 10 000 Adam +   500 L-BFGS
  dense    :  256×6  ~ 400 k params  | 20 000 Adam + 1 000 L-BFGS + 1 000 Adam2

Usage::

    python examples/kim_2024_154kv_bedding/run_multilayer_dense.py
    python examples/kim_2024_154kv_bedding/run_multilayer_dense.py --case A
    python examples/kim_2024_154kv_bedding/run_multilayer_dense.py --case B
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
RESULTS_DIR = HERE / "results_multilayer_dense"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import torch  # noqa: E402

from pinn_cables.io.readers import (  # noqa: E402
    load_problem,
    load_solver_params,
)
from pinn_cables.io.report import write_bc_report  # noqa: E402
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
PAPER_T_MAX_PAC  = 273.15 + 70.6   # K — FEM PAC summer
PAPER_T_MAX_SAND = 273.15 + 77.6   # K — FEM sand summer
PAPER_W_D  = 3.57                  # W/m
PAPER_FREQ = 60.0                  # Hz


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Kim et al. (2024) — multilayer dense (network-independent)")
    ap.add_argument(
        "--case", "-c",
        choices=["A", "B", "both"],
        default="both",
        help="A = multilayer only | B = multilayer + PAC | both (default: both)",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# KFieldModel builders (same logic as run_multilayer.py)
# ---------------------------------------------------------------------------

def _build_k_model_A(soil_bands) -> KFieldModel:
    return KFieldModel(k_soil=1.351, soil_bands=soil_bands)


def _build_k_model_B(soil_bands, pac_params) -> KFieldModel:
    return KFieldModel(k_soil=1.351, soil_bands=soil_bands, pac_params=pac_params)


# ---------------------------------------------------------------------------
# Load previously trained T_max for comparison
# ---------------------------------------------------------------------------

def _load_T_max(results_subdir: str, case_tag: str, problem, layers_list,
                Q_lins, Q_d, k_soil_bg, T_amb, solver_csv: str,
                device) -> float | None:
    pt = HERE / results_subdir / ("model_case_%s.pt" % case_tag)
    if not pt.exists():
        return None
    try:
        sp = load_solver_params(DATA_DIR / solver_csv)
        cfg = sp.to_solver_cfg()
        norm = cfg.get("normalization", {}).get("normalize_coords", True)
        base = build_model(cfg["model"], in_dim=2, device=device)
        model = ResidualPINNModel(
            base, layers_list, problem.placements,
            k_soil_bg, T_amb, Q_lins,
            problem.domain, normalize=norm, Q_d=Q_d,
            enable_grad_Tbg=True,
        )
        state = torch.load(pt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        T_conds = eval_conductor_temps(
            model, problem.placements, problem.domain, device, norm)
        return max(T_conds)
    except Exception as e:
        print("    (no se pudo cargar %s: %s)" % (pt.name, e))
        return None


# ---------------------------------------------------------------------------
# Run one case
# ---------------------------------------------------------------------------

def run_case(
    case_tag: str,
    k_model: KFieldModel,
    problem,
    layers_list: list,
    Q_lins: list,
    Q_d: float,
    solver_cfg: dict,
    device: torch.device,
    logger,
) -> tuple[float, dict]:

    T_amb        = problem.scenarios[0].T_amb
    all_placements = problem.placements
    r_sheaths    = [l[-1].r_outer for l in layers_list]

    adam_n    = solver_cfg["training"]["adam_steps"]
    lbfgs_n   = solver_cfg["training"]["lbfgs_steps"]
    lbfgs_hist = solver_cfg["training"].get("lbfgs_history", 100)
    adam2_n   = solver_cfg["training"].get("adam2_steps", 0)
    adam2_lr_v = solver_cfg["training"].get("adam2_lr", 1e-5)
    print_ev  = solver_cfg["training"]["print_every"]
    n_int     = solver_cfg["sampling"]["n_interior"]
    n_bnd     = solver_cfg["sampling"]["n_boundary"]
    oversamp  = solver_cfg["sampling"]["oversample"]
    normalize  = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    w_pde = solver_cfg["loss_weights"].get("pde", 1.0)
    w_bc  = solver_cfg["loss_weights"].get("bc_dirichlet", 10.0)
    lr    = solver_cfg["training"]["lr"]
    width = solver_cfg["model"]["width"]
    depth = solver_cfg["model"]["depth"]

    k_soil_bg = k_model.k_eff_bg(all_placements)

    set_seed(solver_cfg.get("seed", 42))
    base = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModel(
        base, layers_list, all_placements,
        k_soil_bg, T_amb, Q_lins,
        problem.domain, normalize=normalize, Q_d=Q_d,
        enable_grad_Tbg=True,
    )
    init_output_bias(model.base, 0.0)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n  Red: MLP %dx%d  (%d params)" % (width, depth, n_params))
    print("  Entrenamiento: %d Adam + %d L-BFGS%s" % (
        adam_n, lbfgs_n,
        " + %d Adam2" % adam2_n if adam2_n > 0 else ""))
    print("  Puntos: %d interior | %d bnd | oversample=%d" % (
        n_int, n_bnd, oversamp))

    # Pre-train on Kennelly background
    rmse_pre = pretrain_multicable(
        model, all_placements, problem.domain,
        layers_list, Q_lins, k_soil_bg, T_amb,
        device=device, normalize=normalize,
        n_per_cable=2000, n_bc=300, n_steps=1000, lr=1e-3,
        Q_d=Q_d,
    )
    print("  Pre-entrenamiento: RMSE = %.3f K" % rmse_pre)

    # JIT trace
    try:
        dummy = torch.randn(64, 2, device=device)
        model.base = torch.jit.trace(model.base, dummy)
    except Exception:
        pass

    def k_fn_homog(xy: torch.Tensor) -> torch.Tensor:
        return torch.full((xy.shape[0], 1), k_soil_bg,
                          device=xy.device, dtype=xy.dtype)

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
        lbfgs_history=lbfgs_hist,
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
    model_path = RESULTS_DIR / ("model_case_%s.pt" % case_tag)
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo guardado: %s", model_path)

    # Plots
    plot_loss_history(
        history,
        title="Loss — Kim 2024 Case %s (dense)" % case_tag,
        save_path=RESULTS_DIR / ("loss_case_%s.png" % case_tag),
    )
    X_g, Y_g, T_g = evaluate_on_grid(
        model, problem.domain, nx=400, ny=400,
        device=device, normalize=normalize,
    )
    plot_temperature_field(
        X_g, Y_g, T_g,
        title="T [K] — Case %s (dense)" % case_tag,
        save_path=RESULTS_DIR / ("temperature_field_case_%s.png" % case_tag),
    )

    cable_labels = []
    for p in all_placements:
        row = "B" if abs(p.cy) > 1.3 else "T"
        col = {-0.40: "1", 0.00: "2", 0.40: "3"}.get(round(p.cx, 2), "?")
        cable_labels.append("%s%s" % (row, col))

    plot_zoom_temperature(
        model, problem.domain, all_placements, layers_list,
        device, normalize,
        RESULTS_DIR / ("zoom_case_%s.png" % case_tag),
        zoom=(-1.5, 1.5, -2.5, -0.5),
        pp=k_model.pac_params,
        celsius=True, annotate_max=True,
        cable_labels=cable_labels,
        title="T [degC] - Case %s dense (multilayer + CLSM)" % case_tag,
    )

    T_conds = eval_conductor_temps(model, all_placements, problem.domain, device, normalize)
    T_worst = max(T_conds)
    return T_worst, history


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 72

    # Load problem
    problem = load_problem(DATA_DIR)
    scenario = problem.scenarios[0]
    all_placements = problem.placements
    n_cables = len(all_placements)

    soil_bands = load_soil_layers(DATA_DIR / "soil_layers.csv")
    pac_params = load_physics_params(DATA_DIR / "physics_params.csv")

    solver_params = load_solver_params(DATA_DIR / "solver_params_dense.csv")
    solver_cfg = solver_params.to_solver_cfg()

    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="kim2024_multilayer_dense")

    # Cable + Q
    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A   = all_placements[0].current_A
    section_mm2 = all_placements[0].section_mm2

    iec_q = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=PAPER_T_MAX_PAC, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_cond = iec_q["Q_cond_W_per_m"]
    Q_d    = PAPER_W_D
    Q_lin  = Q_cond + Q_d

    layers = get_kim2024_cable_layers(Q_lin)
    layers_list = [layers] * n_cables
    Q_lins      = [Q_lin] * n_cables

    # IEC reference (for comparison table)
    k_soil_L2 = soil_bands[1].k   # layer where cables sit
    iec_hom = iec60287_estimate(
        layers_list, all_placements, k_soil_L2,
        scenario.T_amb, Q_lins, Q_d=Q_d,
    )

    # Header
    print(SEP)
    print("  BENCHMARK Kim et al. (2024) — MULTILAYER DENSE (network-independent)")
    print("  154 kV XLPE 6 cables two-flat")
    print(SEP)
    write_bc_report(
        problem, RESULTS_DIR,
        label="Kim (2024) — multilayer dense",
    )
    print("\n  Soil layers:")
    for b in soil_bands:
        print("    y=[%.2f, %.2f] m:  k = %.3f W/(mK)" % (b.y_top, b.y_bottom, b.k))
    print("\n  Q_cond = %.2f W/m  +  Q_d = %.2f W/m  =  %.2f W/m" % (
        Q_cond, Q_d, Q_lin))
    print("  IEC 60287 (k=%.3f, homog.): %.1f degC" % (
        k_soil_L2, iec_hom["T_cond_ref"] - 273.15))

    # ----------------------------------------------------------------
    # Cases
    # ----------------------------------------------------------------
    run_A = args.case in ("A", "both")
    run_B = args.case in ("B", "both")
    results: dict[str, float] = {}

    if run_A:
        k_model_A = _build_k_model_A(soil_bands)
        print("\n" + "-" * 72)
        print("  CASE A — Multilayer soil only (no PAC bedding)")
        plot_k_field(
            problem.domain, None, all_placements, layers_list,
            RESULTS_DIR / "k_field_case_A.png",
            title="k(x,y) — Case A: 3 soil layers, no PAC",
            k_model=k_model_A,
        )
        T_A, _ = run_case(
            "A", k_model_A, problem, layers_list, Q_lins, Q_d,
            solver_cfg, device, logger,
        )
        results["A"] = T_A

    if run_B:
        k_model_B = _build_k_model_B(soil_bands, pac_params)
        print("\n" + "-" * 72)
        print("  CASE B — Multilayer soil + PAC bedding zone")
        plot_k_field(
            problem.domain, None, all_placements, layers_list,
            RESULTS_DIR / "k_field_case_B.png",
            title="k(x,y) — Case B: 3 soil layers + PAC zone",
            k_model=k_model_B,
        )
        T_B, _ = run_case(
            "B", k_model_B, problem, layers_list, Q_lins, Q_d,
            solver_cfg, device, logger,
        )
        results["B"] = T_B

    # ----------------------------------------------------------------
    # Convergence table (quick -> research -> dense)
    # ----------------------------------------------------------------
    # Load T_max from quick and research for comparison
    k_bg_A = _build_k_model_A(soil_bands).k_eff_bg(all_placements)
    k_bg_B = _build_k_model_B(soil_bands, pac_params).k_eff_bg(all_placements)

    T_quick_A = _load_T_max(
        "results_multilayer_quick", "A", problem, layers_list, Q_lins, Q_d,
        k_bg_A, scenario.T_amb, "solver_params.csv", device)
    T_research_A = _load_T_max(
        "results_multilayer_research", "A", problem, layers_list, Q_lins, Q_d,
        k_bg_A, scenario.T_amb, "solver_params_research.csv", device)

    T_quick_B = _load_T_max(
        "results_multilayer_quick", "B", problem, layers_list, Q_lins, Q_d,
        k_bg_B, scenario.T_amb, "solver_params.csv", device)
    T_research_B = _load_T_max(
        "results_multilayer_research", "B", problem, layers_list, Q_lins, Q_d,
        k_bg_B, scenario.T_amb, "solver_params_research.csv", device)

    C = 46
    print("\n" + SEP)
    print("  CONVERGENCE TABLE — T_max vs network size")
    print("  (analogous to mesh-independence study in FEM)")
    print(SEP)
    print("  %-*s  %10s  %10s  %10s" % (C, "Method", "T_max [degC]", "vs FEM [K]", "capacity"))
    print("  " + "-" * 78)
    print("  %-*s  %10.1f  %10s  %10s" % (
        C, "FEM COMSOL — sand (Kim 2024)", PAPER_T_MAX_SAND - 273.15, "ref", "—"))
    print("  %-*s  %10.1f  %10.1f K  %10s" % (
        C, "FEM COMSOL — PAC  (Kim 2024)", PAPER_T_MAX_PAC - 273.15,
        PAPER_T_MAX_PAC - PAPER_T_MAX_SAND, "—"))

    def _row(label, T_K, T_ref, cap):
        if T_K is None:
            print("  %-*s  %10s  %10s  %10s" % (C, label, "(no model)", "—", cap))
        else:
            print("  %-*s  %10.1f  %+10.1f K  %10s" % (
                C, label, T_K - 273.15, T_K - T_ref, cap))

    print("  " + "-" * 78)
    print("  Case A — ref: sand 77.6 degC")
    _row("  quick    (64x4,  ~17 k params,  5.5k steps)", T_quick_A,    PAPER_T_MAX_SAND, "17 k")
    _row("  research (128x5, ~83 k params, 11.5k steps)", T_research_A, PAPER_T_MAX_SAND, "83 k")
    _row("  dense    (256x6, ~400k params, 22  k steps)", results.get("A"), PAPER_T_MAX_SAND, "~400 k")

    print("  " + "-" * 78)
    print("  Case B — ref: PAC 70.6 degC")
    _row("  quick    (64x4,  ~17 k params,  5.5k steps)", T_quick_B,    PAPER_T_MAX_PAC, "17 k")
    _row("  research (128x5, ~83 k params, 11.5k steps)", T_research_B, PAPER_T_MAX_PAC, "83 k")
    _row("  dense    (256x6, ~400k params, 22  k steps)", results.get("B"), PAPER_T_MAX_PAC, "~400 k")
    print(SEP)


if __name__ == "__main__":
    main()
