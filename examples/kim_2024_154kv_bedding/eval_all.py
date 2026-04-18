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
# Evaluate a single model and return formatted row
# ---------------------------------------------------------------------------

def _fmt_row(label, T_worst, T_ref, profile, final_loss=None):
    err = T_worst - T_ref
    loss_str = ("%.4e" % final_loss) if final_loss is not None else "—"
    return (label, profile,
            "%.1f" % (T_worst - 273.15),
            "%.1f" % (T_ref - 273.15),
            "%+.1f K" % err,
            loss_str)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    device = get_device("cpu")
    print("=" * 72)
    print("  EVALUACION — Kim et al. (2024) — todos los modelos guardados")
    print("  FEM ref: PAC 70.6 °C | sand 77.6 °C")
    print("=" * 72)

    rows = []

    # ---------------------------------------------------------------
    # 1. run_example.py — quick  (results/)
    # ---------------------------------------------------------------
    mp = HERE / "results" / "model_final.pt"
    if mp.exists():
        print("\n  [1/6] run_example.py --profile quick ...")
        prob, scen, plac, llist, qlins, qd = _load_common(device)
        model, norm = _reconstruct_model(
            mp, DATA_DIR / "solver_params.csv",
            prob, scen, plac, llist, qlins, qd, device=device)
        _, _, T_w = _eval_model(model, prob, plac, norm, device)
        # read final loss from train.log if available
        tl = HERE / "results" / "train.log"
        loss = None
        if tl.exists():
            for line in tl.read_text(encoding="utf-8", errors="replace").splitlines():
                if "Final loss=" in line:
                    try:
                        loss = float(line.split("Final loss=")[1].strip())
                    except Exception:
                        pass
        rows.append(_fmt_row("run_example.py", T_w, PAPER_T_MAX_PAC, "quick", loss))
        print("    T_max conductor = %.1f °C  (FEM ref 70.6 °C)" % (T_w - 273.15))
    else:
        print("  [1/6] SKIP — results/model_final.pt no encontrado")

    # ---------------------------------------------------------------
    # 2. run_example.py — research  (results_research/)
    # ---------------------------------------------------------------
    mp = HERE / "results_research" / "model_final.pt"
    if mp.exists():
        print("\n  [2/6] run_example.py --profile research ...")
        prob, scen, plac, llist, qlins, qd = _load_common(device)
        model, norm = _reconstruct_model(
            mp, DATA_DIR / "solver_params_research.csv",
            prob, scen, plac, llist, qlins, qd, device=device)
        _, _, T_w = _eval_model(model, prob, plac, norm, device)
        tl = HERE / "results_research" / "train.log"
        loss = None
        if tl.exists():
            for line in tl.read_text(encoding="utf-8", errors="replace").splitlines():
                if "Final loss=" in line:
                    try:
                        loss = float(line.split("Final loss=")[1].strip())
                    except Exception:
                        pass
        rows.append(_fmt_row("run_example.py", T_w, PAPER_T_MAX_PAC, "research", loss))
        print("    T_max conductor = %.1f °C  (FEM ref 70.6 °C)" % (T_w - 273.15))
    else:
        print("  [2/6] SKIP — results_research/model_final.pt no encontrado")

    # ---------------------------------------------------------------
    # 3. run_research_pac.py — quick  (results_pac_quick/)
    # ---------------------------------------------------------------
    mp = HERE / "results_pac_quick" / "model_final.pt"
    if mp.exists():
        print("\n  [3/6] run_research_pac.py --profile quick ...")
        prob, scen, plac, llist, qlins, qd = _load_common(device)
        model, norm = _reconstruct_model(
            mp, DATA_DIR / "solver_params.csv",
            prob, scen, plac, llist, qlins, qd,
            enable_grad_Tbg=True, device=device)
        _, _, T_w = _eval_model(model, prob, plac, norm, device)
        rows.append(_fmt_row("run_research_pac.py", T_w, PAPER_T_MAX_PAC, "quick"))
        print("    T_max conductor = %.1f °C  (FEM ref 70.6 °C)" % (T_w - 273.15))
    else:
        print("  [3/6] SKIP — results_pac_quick/model_final.pt no encontrado")

    # ---------------------------------------------------------------
    # 4. run_research_pac.py — research  (results_pac_research/)
    # ---------------------------------------------------------------
    mp = HERE / "results_pac_research" / "model_final.pt"
    if mp.exists():
        print("\n  [4/6] run_research_pac.py --profile research ...")
        prob, scen, plac, llist, qlins, qd = _load_common(device)
        model, norm = _reconstruct_model(
            mp, DATA_DIR / "solver_params_research.csv",
            prob, scen, plac, llist, qlins, qd,
            enable_grad_Tbg=True, device=device)
        _, _, T_w = _eval_model(model, prob, plac, norm, device)
        tl = HERE / "results_pac_research" / "train.log"
        loss = None
        if tl.exists():
            for line in tl.read_text(encoding="utf-8", errors="replace").splitlines():
                if "Final loss=" in line or "L-BFGS done" in line:
                    try:
                        if "loss=" in line:
                            loss = float(line.split("loss=")[1].strip())
                    except Exception:
                        pass
        rows.append(_fmt_row("run_research_pac.py", T_w, PAPER_T_MAX_PAC, "research", loss))
        print("    T_max conductor = %.1f °C  (FEM ref 70.6 °C)" % (T_w - 273.15))
    else:
        print("  [4/6] SKIP — results_pac_research/model_final.pt no encontrado")

    # ---------------------------------------------------------------
    # 5 & 6. run_multilayer.py — quick + research  (Case A and B each)
    # ---------------------------------------------------------------
    for profile, results_dir_name, solver_csv_name in [
        ("quick",    "results_multilayer_quick",    "solver_params.csv"),
        ("research", "results_multilayer_research", "solver_params_research.csv"),
    ]:
        prob, scen, plac, llist, qlins, qd = _load_multilayer_common(device)
        soil_bands = load_soil_layers(DATA_DIR / "soil_layers.csv")
        pac_params = load_physics_params(DATA_DIR / "physics_params.csv")

        for case_tag, ref_K, ref_label in [
            ("A", PAPER_T_MAX_SAND, "sand 77.6 °C"),
            ("B", PAPER_T_MAX_PAC,  "PAC  70.6 °C"),
        ]:
            mp = HERE / results_dir_name / ("model_case_%s.pt" % case_tag)
            idx_str = "5" if (profile == "quick" and case_tag == "A") else \
                      "6" if (profile == "quick" and case_tag == "B") else \
                      "7" if (profile == "research" and case_tag == "A") else "8"
            label = "run_multilayer.py Case %s" % case_tag
            if not mp.exists():
                print("\n  [%s/%d] SKIP — %s/model_case_%s.pt no encontrado" % (
                    idx_str, 8, results_dir_name, case_tag))
                continue

            print("\n  [%s/8] %s --profile %s Case %s ..." % (
                idx_str, label, profile, case_tag))

            # Case A: multilayer soil, no PAC
            # Case B: multilayer soil + PAC
            if case_tag == "A":
                k_model = KFieldModel(k_soil=1.351, soil_bands=soil_bands)
            else:
                k_model = KFieldModel(k_soil=1.351, soil_bands=soil_bands,
                                      pac_params=pac_params)
            k_bg = k_model.k_eff_bg(plac)

            solver_params = load_solver_params(DATA_DIR / solver_csv_name)
            solver_cfg    = solver_params.to_solver_cfg()
            normalize     = solver_cfg.get("normalization", {}).get("normalize_coords", True)

            base_model = build_model(solver_cfg["model"], in_dim=2, device=device)
            model = ResidualPINNModel(
                base_model, llist, plac,
                k_bg, scen.T_amb, qlins,
                prob.domain, normalize=normalize, Q_d=qd,
                enable_grad_Tbg=True,
            )
            state = torch.load(mp, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()

            with torch.no_grad():
                _, _, T_w = _eval_model(model, prob, plac, normalize, device)

            rows.append(_fmt_row(label, T_w, ref_K, profile))
            print("    T_max conductor = %.1f °C  (FEM ref %s)" % (
                T_w - 273.15, ref_label))

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    SEP = "=" * 90
    print("\n\n" + SEP)
    print("  TABLA COMPARATIVA — Kim et al. (2024) — PINN vs FEM COMSOL")
    print(SEP)
    print("  %-30s  %-10s  %10s  %10s  %10s  %12s" % (
        "Script", "Perfil", "T_PINN(°C)", "T_FEM(°C)", "Error", "Loss final"))
    print("  " + "-" * 86)
    for r in rows:
        print("  %-30s  %-10s  %10s  %10s  %10s  %12s" % r)
    print("  " + "-" * 86)
    print("  Referencia FEM COMSOL (paper): PAC=70.6°C | sand=77.6°C")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
