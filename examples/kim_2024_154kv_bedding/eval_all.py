"""Evaluar todos los modelos guardados — Kim et al. (2024) benchmark.

Carga cada modelo .pt sin reentrenar, evalua T_max por conductor, e imprime
una tabla comparativa completa contra los resultados FEM del paper.

Uso::

    python examples/kim_2024_154kv_bedding/eval_all.py

No requiere GPU ni reentrenamiento.
"""

from __future__ import annotations

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
from pinn_cables.materials.props import get_kim2024_cable_layers  # noqa: E402
from pinn_cables.physics.iec60287 import compute_iec60287_Q  # noqa: E402
from pinn_cables.physics.k_field import (  # noqa: E402
    KFieldModel,
    load_physics_params,
    load_soil_layers,
)
from pinn_cables.pinn.model import ResidualPINNModel, build_model  # noqa: E402
from pinn_cables.pinn.pde import pde_residual_steady  # noqa: E402
from pinn_cables.pinn.utils import get_device  # noqa: E402
from pinn_cables.post.eval import eval_conductor_temps, evaluate_on_grid  # noqa: E402

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------
PAPER_T_MAX_PAC  = 273.15 + 70.6   # K — FEM PAC summer
PAPER_T_MAX_SAND = 273.15 + 77.6   # K — FEM sand summer
PAPER_W_D   = 3.57                  # W/m
PAPER_FREQ  = 60.0                  # Hz

# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------

def _load_common(device):
    """Load problem, Q values and layers for run_example.py-style models."""
    problem = load_problem(DATA_DIR)
    scenario = problem.scenarios[0]
    all_placements = problem.placements
    n_cables = len(all_placements)

    section_mm2 = all_placements[0].section_mm2
    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A   = all_placements[0].current_A

    T_op_K = PAPER_T_MAX_PAC
    iec_q  = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=T_op_K, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_cond  = iec_q["Q_cond_W_per_m"]
    Q_d     = PAPER_W_D
    Q_total = Q_cond + Q_d

    layers_template = problem.get_layers(all_placements[0].cable_id)
    layers  = override_conductor_Q(layers_template, Q_total)
    Q_lins  = [Q_total] * n_cables
    layers_list = [layers] * n_cables

    return problem, scenario, all_placements, layers_list, Q_lins, Q_d


def _reconstruct_model(model_path: Path, solver_csv: Path, problem, scenario,
                       all_placements, layers_list, Q_lins, Q_d,
                       k_soil_override=None, enable_grad_Tbg=False, device=None):
    """Reconstruct ResidualPINNModel and load saved state dict."""
    if device is None:
        device = torch.device("cpu")

    solver_params = load_solver_params(solver_csv)
    solver_cfg    = solver_params.to_solver_cfg()
    normalize     = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    k_soil        = k_soil_override if k_soil_override is not None else scenario.k_soil

    base_model = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModel(
        base_model, layers_list, all_placements,
        k_soil, scenario.T_amb, Q_lins,
        problem.domain, normalize=normalize, Q_d=Q_d,
        enable_grad_Tbg=enable_grad_Tbg,
    )
    state  = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    normalize_flag = normalize  # capture for caller
    return model, normalize_flag


@torch.no_grad()
def _eval_model(model, problem, all_placements, normalize, device):
    """Evaluate T_max on grid and T per conductor."""
    _, _, T_grid = evaluate_on_grid(
        model, problem.domain, nx=200, ny=200,
        device=device, normalize=normalize,
    )
    T_max_grid = float(T_grid.max())
    T_conds = eval_conductor_temps(
        model, all_placements, problem.domain, device, normalize,
    )
    T_worst = max(T_conds)
    return T_max_grid, T_conds, T_worst


# ---------------------------------------------------------------------------
# Multilayer-specific: build kim2024 cable layers
# ---------------------------------------------------------------------------

def _load_multilayer_common(device):
    """Load problem + Q for run_multilayer.py-style models."""
    problem = load_problem(DATA_DIR)
    scenario = problem.scenarios[0]
    all_placements = problem.placements
    n_cables = len(all_placements)

    section_mm2 = all_placements[0].section_mm2
    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A   = all_placements[0].current_A

    iec_q = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=PAPER_T_MAX_PAC, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_cond = iec_q["Q_cond_W_per_m"]
    Q_d    = PAPER_W_D
    Q_lin  = Q_cond + Q_d

    layers      = get_kim2024_cable_layers(Q_lin)
    layers_list = [layers] * n_cables
    Q_lins      = [Q_lin]  * n_cables

    return problem, scenario, all_placements, layers_list, Q_lins, Q_d


# ---------------------------------------------------------------------------
# Zoomed-zone metrics
# ---------------------------------------------------------------------------

# Region enclosing the 6-cable array with ~1 m buffer on each side.
# Cables span x ∈ [-0.4, 0.4] m, y ∈ [-1.6, -1.2] m.
ZOOM_XMIN, ZOOM_XMAX = -1.5, 1.5   # m
ZOOM_YMIN, ZOOM_YMAX = -2.5, -0.3  # m
ZOOM_NX,   ZOOM_NY   = 120, 80      # grid resolution inside zoomed zone


def _eval_zoomed_T(model, domain, device, normalize):
    """Evaluate model on the zoomed grid; return flat temperature array (K)."""
    xs = torch.linspace(ZOOM_XMIN, ZOOM_XMAX, ZOOM_NX, device=device)
    ys = torch.linspace(ZOOM_YMIN, ZOOM_YMAX, ZOOM_NY, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    xy_phys = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    if normalize:
        lo = torch.tensor([domain.xmin, domain.ymin], device=device, dtype=xy_phys.dtype)
        hi = torch.tensor([domain.xmax, domain.ymax], device=device, dtype=xy_phys.dtype)
        xy_in = (xy_phys - lo) / (hi - lo) * 2.0 - 1.0
    else:
        xy_in = xy_phys

    with torch.no_grad():
        T = model(xy_in).reshape(-1)
    return T.cpu()


def _zoomed_pde_rms(model, domain, device, normalize, n_pts: int = 3000):
    """PDE residual RMS [W/m²] in the zoomed zone (soil; Q=0).

    Differentiates T w.r.t. physical coordinates using the chain rule through
    the normalisation transform.  Returns root-mean-square of ∇·(k ∇T) over
    n_pts random soil points in the zoomed zone.
    """
    torch.manual_seed(42)
    x = torch.rand(n_pts, 1) * (ZOOM_XMAX - ZOOM_XMIN) + ZOOM_XMIN
    y = torch.rand(n_pts, 1) * (ZOOM_YMAX - ZOOM_YMIN) + ZOOM_YMIN
    xy_phys = torch.cat([x, y], dim=1).to(device).requires_grad_(True)

    if normalize:
        lo = torch.tensor([domain.xmin, domain.ymin], device=device, dtype=xy_phys.dtype)
        hi = torch.tensor([domain.xmax, domain.ymax], device=device, dtype=xy_phys.dtype)
        xy_in = (xy_phys - lo) / (hi - lo) * 2.0 - 1.0
    else:
        xy_in = xy_phys

    T = model(xy_in)  # (N, 1) — depends on xy_phys through xy_in

    # ∇·(k ∇T) w.r.t. physical coords; Q=0 in soil
    k_soil = float(model._k_soil)
    res = pde_residual_steady(T, xy_phys, k_soil, 0.0)  # (N, 1)
    rms = float(res.detach().pow(2).mean().sqrt())
    return rms


def _zoomed_rmse_pair(T_a: torch.Tensor, T_b: torch.Tensor) -> float:
    """RMSE [K] between two flat temperature tensors on the same zoomed grid."""
    return float((T_a - T_b).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Evaluate a single model and return formatted row
# ---------------------------------------------------------------------------

def _fmt_row(label, T_worst, T_ref, profile, final_loss=None, pde_rms=None, rmse_pair=None):
    err = T_worst - T_ref
    loss_str = ("%.4e" % final_loss) if final_loss is not None else "—"
    pde_str  = ("%.3e" % pde_rms)   if pde_rms  is not None else "—"
    pair_str = ("%.2f K" % rmse_pair) if isinstance(rmse_pair, (int, float)) else (rmse_pair if rmse_pair is not None else "—")
    return (label, profile,
            "%.1f" % (T_worst - 273.15),
            "%.1f" % (T_ref - 273.15),
            "%+.1f K" % err,
            loss_str,
            pde_str,
            pair_str)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    device = get_device("cpu")
    print("=" * 72)
    print("  EVALUACION — Kim et al. (2024) — todos los modelos guardados")
    print("  FEM ref: PAC 70.6 °C | sand 77.6 °C")
    print("  (8 quick/research + 2 dense = hasta 10 modelos)")
    print("=" * 72)

    rows = []

    # ---------------------------------------------------------------
    # Helper: read final loss from train.log
    # ---------------------------------------------------------------
    def _read_loss(results_dir):
        tl = HERE / results_dir / "train.log"
        loss = None
        if tl.exists():
            for line in tl.read_text(encoding="utf-8", errors="replace").splitlines():
                if "Final loss=" in line or "L-BFGS done" in line or "Adam2 done" in line:
                    try:
                        if "loss=" in line.lower():
                            loss = float(line.split("loss=")[-1].strip())
                    except Exception:
                        pass
        return loss

    # Helper: compute both zoomed metrics for a loaded model
    def _zoomed_metrics(model, prob, norm):
        pde_rms = _zoomed_pde_rms(model, prob.domain, device, norm)
        T_zone  = _eval_zoomed_T(model, prob.domain, device, norm)
        return pde_rms, T_zone

    # ---------------------------------------------------------------
    # Pair 1 — run_example.py quick / research
    # ---------------------------------------------------------------
    mp_q = HERE / "results" / "model_final.pt"
    mp_r = HERE / "results_research" / "model_final.pt"
    T_zone_eq = T_zone_er = None

    if mp_q.exists():
        print("\n  [1/8] run_example.py --profile quick ...")
        prob, scen, plac, llist, qlins, qd = _load_common(device)
        model_q, norm_q = _reconstruct_model(
            mp_q, DATA_DIR / "solver_params.csv",
            prob, scen, plac, llist, qlins, qd, device=device)
        _, _, T_w = _eval_model(model_q, prob, plac, norm_q, device)
        pde_q, T_zone_eq = _zoomed_metrics(model_q, prob, norm_q)
        rows.append(_fmt_row("run_example.py", T_w, PAPER_T_MAX_SAND, "quick",
                             _read_loss("results"), pde_q))
        print("    T_max=%.1f °C | PDE_rms=%.3e W/m²" % (T_w - 273.15, pde_q))
    else:
        print("  [1/8] SKIP")

    if mp_r.exists():
        print("\n  [2/8] run_example.py --profile research ...")
        prob, scen, plac, llist, qlins, qd = _load_common(device)
        model_r, norm_r = _reconstruct_model(
            mp_r, DATA_DIR / "solver_params_research.csv",
            prob, scen, plac, llist, qlins, qd, device=device)
        _, _, T_w = _eval_model(model_r, prob, plac, norm_r, device)
        pde_r, T_zone_er = _zoomed_metrics(model_r, prob, norm_r)
        rows.append(_fmt_row("run_example.py", T_w, PAPER_T_MAX_SAND, "research",
                             _read_loss("results_research"), pde_r))
        print("    T_max=%.1f °C | PDE_rms=%.3e W/m²" % (T_w - 273.15, pde_r))
    else:
        print("  [2/8] SKIP")

    # Back-fill RMSE pair for run_example rows
    if T_zone_eq is not None and T_zone_er is not None:
        rmse_e = _zoomed_rmse_pair(T_zone_eq, T_zone_er)
        for i, r in enumerate(rows):
            if r[0] == "run_example.py":
                rows[i] = r[:-1] + ("%.2f K" % rmse_e,)
        print("    RMSE(quick<->research) en zona zoomed = %.2f K" % rmse_e)

    # ---------------------------------------------------------------
    # Pair 2 — run_research_pac.py quick / research
    # ---------------------------------------------------------------
    mp_q = HERE / "results_pac_quick" / "model_final.pt"
    mp_r = HERE / "results_pac_research" / "model_final.pt"
    T_zone_pq = T_zone_pr = None

    if mp_q.exists():
        print("\n  [3/8] run_research_pac.py --profile quick ...")
        prob, scen, plac, llist, qlins, qd = _load_common(device)
        model_q, norm_q = _reconstruct_model(
            mp_q, DATA_DIR / "solver_params.csv",
            prob, scen, plac, llist, qlins, qd,
            enable_grad_Tbg=True, device=device)
        _, _, T_w = _eval_model(model_q, prob, plac, norm_q, device)
        pde_q, T_zone_pq = _zoomed_metrics(model_q, prob, norm_q)
        rows.append(_fmt_row("run_research_pac.py", T_w, PAPER_T_MAX_PAC, "quick",
                             _read_loss("results_pac_quick"), pde_q))
        print("    T_max=%.1f °C | PDE_rms=%.3e W/m²" % (T_w - 273.15, pde_q))
    else:
        print("  [3/8] SKIP")

    if mp_r.exists():
        print("\n  [4/8] run_research_pac.py --profile research ...")
        prob, scen, plac, llist, qlins, qd = _load_common(device)
        model_r, norm_r = _reconstruct_model(
            mp_r, DATA_DIR / "solver_params_research.csv",
            prob, scen, plac, llist, qlins, qd,
            enable_grad_Tbg=True, device=device)
        _, _, T_w = _eval_model(model_r, prob, plac, norm_r, device)
        pde_r, T_zone_pr = _zoomed_metrics(model_r, prob, norm_r)
        rows.append(_fmt_row("run_research_pac.py", T_w, PAPER_T_MAX_PAC, "research",
                             _read_loss("results_pac_research"), pde_r))
        print("    T_max=%.1f °C | PDE_rms=%.3e W/m²" % (T_w - 273.15, pde_r))
    else:
        print("  [4/8] SKIP")

    if T_zone_pq is not None and T_zone_pr is not None:
        rmse_p = _zoomed_rmse_pair(T_zone_pq, T_zone_pr)
        for i, r in enumerate(rows):
            if r[0] == "run_research_pac.py":
                rows[i] = r[:-1] + ("%.2f K" % rmse_p,)
        print("    RMSE(quick<->research) en zona zoomed = %.2f K" % rmse_p)

    # ---------------------------------------------------------------
    # Pairs 3 & 4 — run_multilayer.py Case A and B
    # ---------------------------------------------------------------
    prob_ml, scen_ml, plac_ml, llist_ml, qlins_ml, qd_ml = _load_multilayer_common(device)
    soil_bands = load_soil_layers(DATA_DIR / "soil_layers.csv")
    pac_params = load_physics_params(DATA_DIR / "physics_params.csv")

    run_idx = 5
    for case_tag, ref_K, ref_label, k_bg_val in [
        ("A", PAPER_T_MAX_SAND, "sand 77.6 °C",
         KFieldModel(k_soil=1.351, soil_bands=soil_bands).k_eff_bg(
             _load_multilayer_common(device)[2])),
        ("B", PAPER_T_MAX_PAC,  "PAC  70.6 °C",
         KFieldModel(k_soil=1.351, soil_bands=soil_bands,
                     pac_params=load_physics_params(DATA_DIR / "physics_params.csv")
                     ).k_eff_bg(_load_multilayer_common(device)[2])),
    ]:
        label = "run_multilayer.py Case %s" % case_tag
        T_zones_m = {}  # profile -> flat T tensor on zoomed grid

        for profile, solver_csv_name, results_dir_name in [
            ("quick",    "solver_params.csv",          "results_multilayer_quick"),
            ("research", "solver_params_research.csv", "results_multilayer_research"),
            ("dense",    "solver_params_dense.csv",    "results_multilayer_dense"),
        ]:
            mp = HERE / results_dir_name / ("model_case_%s.pt" % case_tag)
            if not mp.exists():
                print("\n  [%d/8] SKIP — %s/model_case_%s.pt no encontrado" % (
                    run_idx, results_dir_name, case_tag))
                run_idx += 1
                continue

            print("\n  [%d/8] %s --profile %s Case %s ..." % (
                run_idx, label, profile, case_tag))

            solver_params = load_solver_params(DATA_DIR / solver_csv_name)
            solver_cfg    = solver_params.to_solver_cfg()
            norm_ml       = solver_cfg.get("normalization", {}).get("normalize_coords", True)

            base_model = build_model(solver_cfg["model"], in_dim=2, device=device)
            model = ResidualPINNModel(
                base_model, llist_ml, plac_ml,
                k_bg_val, scen_ml.T_amb, qlins_ml,
                prob_ml.domain, normalize=norm_ml, Q_d=qd_ml,
                enable_grad_Tbg=True,
            )
            state = torch.load(mp, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()

            with torch.no_grad():
                _, _, T_w = _eval_model(model, prob_ml, plac_ml, norm_ml, device)

            pde_rms, T_zone = _zoomed_metrics(model, prob_ml, norm_ml)
            T_zones_m[profile] = T_zone

            rmse_tag = "ref (dense)" if profile == "dense" else "—"
            rows.append(_fmt_row(label, T_w, ref_K, profile,
                                 _read_loss(results_dir_name), pde_rms, rmse_tag))
            print("    T_max=%.1f °C (ref %s) | PDE_rms=%.3e W/m²" % (
                T_w - 273.15, ref_label, pde_rms))
            run_idx += 1

        # Back-fill RMSE vs dense for quick and research
        T_dense_zone = T_zones_m.get("dense")
        if T_dense_zone is not None:
            for prof in ("quick", "research"):
                T_p = T_zones_m.get(prof)
                if T_p is None:
                    continue
                rmse_vs_dense = _zoomed_rmse_pair(T_p, T_dense_zone)
                for i, r in enumerate(rows):
                    if r[0] == label and r[1] == prof:
                        rows[i] = r[:-1] + ("%.2f K" % rmse_vs_dense,)
                print("    RMSE(%s vs dense) Case %s zona zoomed = %.2f K" % (
                    prof, case_tag, rmse_vs_dense))
        else:
            # Dense not yet available — fall back to quick vs research
            if "quick" in T_zones_m and "research" in T_zones_m:
                rmse_qr = _zoomed_rmse_pair(T_zones_m["quick"], T_zones_m["research"])
                for i, r in enumerate(rows):
                    if r[0] == label and r[1] in ("quick", "research"):
                        rows[i] = r[:-1] + ("%.2f K" % rmse_qr,)
                print("    RMSE(quick<->research) Case %s zona zoomed = %.2f K (dense aun no disponible)" % (
                    case_tag, rmse_qr))

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    SEP = "=" * 110
    print("\n\n" + SEP)
    print("  TABLA COMPARATIVA — Kim et al. (2024) — PINN vs FEM COMSOL")
    print("  Zona zoomed: x in [%.1f,%.1f] m, y in [%.1f,%.1f] m (alrededor del arreglo de cables)" % (
        ZOOM_XMIN, ZOOM_XMAX, ZOOM_YMIN, ZOOM_YMAX))
    print(SEP)
    print("  %-30s  %-10s  %10s  %10s  %10s  %12s  %12s  %12s" % (
        "Script", "Perfil", "T_PINN(°C)", "T_FEM(°C)", "Error", "Loss final",
        "PDE_rms(W/m²)", "RMSE_zona(K)"))
    print("  " + "-" * 106)
    for r in rows:
        print("  %-30s  %-10s  %10s  %10s  %10s  %12s  %12s  %12s" % r)
    print("  " + "-" * 106)
    print("  Error    : T_PINN(max conductor) - T_FEM_COMSOL(paper) [K]  -- comparacion vs punto unico del paper")
    print("  RMSE_zona: error RMS del CAMPO T en zona alrededor de cables (supera la limitacion del punto unico):")
    print("             multilayer quick/research -> RMSE vs solucion DENSA (256x6, ~400k params, ref interna)")
    print("             multilayer dense          -> 'ref (dense)' (es la referencia, sin error relativo)")
    print("             otros casos               -> RMSE entre quick y research")
    print("  PDE_rms  : RMS residual div(k*grad(T)) en zona zoomed (cumplimiento de la PDE, ideal -> 0)")
    print("  Referencia FEM COMSOL (paper): PAC=70.6°C | sand=77.6°C")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
