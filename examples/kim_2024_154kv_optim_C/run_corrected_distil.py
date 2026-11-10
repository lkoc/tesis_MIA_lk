"""Distillation 64×4→128×5 with corrected T_amb = 300.30 K.

Must run AFTER both:
  - run_corrected_64x4.py   → results_corrected/model_best_64x4.pt  (teacher)
  - run_corrected_128x5.py  → results_corrected/model_best_128x5.pt (baseline)

See run_corrected_64x4.py for the T_amb correction rationale.

Curriculum:
  Phase 0: Kennelly pretrain   (analytical background, same as always)
  Phase 1: Distillation        (student learns teacher field — soft warm-start 200 steps)
  Phase 2: PDE fine-tune       (Adam + L-BFGS with full physics)

Usage::
    python examples/kim_2024_154kv_optim_C/run_corrected_distil.py
    python examples/kim_2024_154kv_optim_C/run_corrected_distil.py --n-seeds 8
    python examples/kim_2024_154kv_optim_C/run_corrected_distil.py --distil-steps 200 --distil-lr 2e-4
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE        = Path(__file__).resolve().parent
DATA_DIR    = HERE.parent / "kim_2024_154kv_optim_B" / "data"
MODEL_64x4  = HERE.parent / "kim_2024_154kv_optim_B" / "results_corrected" / "model_best_64x4.pt"
MODEL_128x5_BASELINE = HERE / "results_corrected" / "model_best_128x5.pt"
RESULTS_DIR = HERE / "results_corrected"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F

from pinn_cables.io.readers import load_problem
from pinn_cables.materials.props import get_kim2024_cable_layers
from pinn_cables.physics.iec60287 import compute_iec60287_Q
from pinn_cables.physics.k_field import KFieldModel, load_physics_params, load_soil_layers
from pinn_cables.physics.kennelly import iec60287_estimate
from pinn_cables.pinn.model import ResidualPINNModel, build_model
from pinn_cables.pinn.train_custom import init_output_bias, pretrain_multicable, train_adam_lbfgs
from pinn_cables.pinn.utils import get_device, set_seed
from pinn_cables.post.eval import eval_conductor_temps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPER_T_MAX_PAC = 273.15 + 70.6
PAPER_W_D       = 3.57
PAPER_FREQ      = 60.0
FEM_REF_C       = 70.6
T_AMB_CORRECTED = 300.30

# Best 128×5 params (from run_optim_C.py search, same as baseline)
BEST_128x5 = dict(
    lr               = 5e-4,
    adam_steps       = 2000,
    lbfgs_steps      = 1000,
    lbfgs_history    = 100,
    warmup_frac      = 0.50,
    w_pde            = 0.5,
    w_bc             = 200.0,
    n_interior       = 2000,
    n_boundary       = 500,
    oversample       = 1,
    frac_pac_bnd     = 0.50,
    layer_transition = 0.10,
    pac_transition   = 0.20,
    width            = 128,
    depth            = 5,
    fourier_features = False,
)


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def load_problem_corrected() -> dict:
    problem = load_problem(DATA_DIR)
    T_amb   = problem.scenarios[0].T_amb
    if abs(T_amb - T_AMB_CORRECTED) > 0.05:
        raise RuntimeError(
            f"scenarios.csv T_amb={T_amb:.2f} K, expected {T_AMB_CORRECTED:.2f} K.\n"
            "  Please set T_amb=300.30 in examples/kim_2024_154kv_optim_B/data/scenarios.csv"
        )
    all_placements = problem.placements
    n_cables = len(all_placements)
    soil_bands      = load_soil_layers(DATA_DIR / "soil_layers.csv")
    pac_params_base = load_physics_params(DATA_DIR / "physics_params.csv")
    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A   = all_placements[0].current_A
    section_mm2 = all_placements[0].section_mm2
    iec_q = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=PAPER_T_MAX_PAC, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_cond = iec_q["Q_cond_W_per_m"]
    Q_lin  = Q_cond + PAPER_W_D
    layers      = get_kim2024_cable_layers(Q_lin)
    layers_list = [layers] * n_cables
    Q_lins      = [Q_lin]  * n_cables
    return dict(
        problem=problem, all_placements=all_placements, n_cables=n_cables,
        soil_bands=soil_bands, pac_params_base=pac_params_base,
        layers_list=layers_list, Q_lins=Q_lins,
        Q_d=PAPER_W_D, Q_cond=Q_cond, Q_lin=Q_lin,
    )


# ---------------------------------------------------------------------------
# Teacher loader (reconstructs 64×4 from saved checkpoint)
# ---------------------------------------------------------------------------

def load_teacher(shared: dict, device: torch.device) -> ResidualPINNModel:
    ckpt   = torch.load(MODEL_64x4, map_location=device, weights_only=False)
    params = ckpt["params"]

    p = params if isinstance(params, dict) else dataclasses.asdict(params)
    pac_params = dataclasses.replace(
        shared["pac_params_base"], k_transition=p["pac_transition"])
    k_model = KFieldModel(
        k_soil=1.351,
        soil_bands=shared["soil_bands"],
        pac_params=pac_params,
        layer_transition=p["layer_transition"],
    )
    T_amb     = shared["problem"].scenarios[0].T_amb
    k_soil_bg = k_model.k_eff_bg(shared["all_placements"])

    cfg_model = {
        "architecture":        "mlp",
        "width":               p.get("width", 64),
        "depth":               p.get("depth", 4),
        "activation":          "tanh",
        "fourier_features":    p.get("fourier_features", False),
        "fourier_mapping_size": p.get("fourier_mapping_size", 64),
        "fourier_scale":       p.get("fourier_scale", 1.0),
    }
    base  = build_model(cfg_model, in_dim=2, device=device)
    model = ResidualPINNModel(
        base, shared["layers_list"], shared["all_placements"],
        k_soil_bg, T_amb, shared["Q_lins"],
        shared["problem"].domain, normalize=True, Q_d=shared["Q_d"],
        enable_grad_Tbg=True,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Distillation phase
# ---------------------------------------------------------------------------

def distil_from_teacher(
    student: ResidualPINNModel,
    teacher: ResidualPINNModel,
    domain,
    device: torch.device,
    n_points: int = 4000,
    n_steps: int = 200,
    lr: float = 2e-4,
    print_every: int = 0,
) -> list[float]:
    """Phase 1: supervised distillation — student learns T(x,y) from teacher."""
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []
    for step in range(1, n_steps + 1):
        x  = torch.FloatTensor(n_points).uniform_(domain.xmin, domain.xmax)
        y  = torch.FloatTensor(n_points).uniform_(domain.ymin, domain.ymax)
        xy = torch.stack([x, y], dim=1).to(device)
        with torch.no_grad():
            T_teacher = teacher(xy)
        T_student = student(xy)
        loss = F.mse_loss(T_student, T_teacher)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if print_every > 0 and step % print_every == 0:
            print("      distil step %4d/%d  loss=%.4e" % (step, n_steps, loss.item()))
    return losses


# ---------------------------------------------------------------------------
# Single distilled trial
# ---------------------------------------------------------------------------

def run_distil_trial(
    teacher:      ResidualPINNModel,
    shared:       dict,
    device:       torch.device,
    seed:         int = 42,
    distil_steps: int = 200,
    distil_lr:    float = 2e-4,
    distil_pts:   int = 2000,
    verbose:      bool = False,
) -> dict:
    p    = BEST_128x5
    t0   = time.time()
    problem        = shared["problem"]
    all_placements = shared["all_placements"]
    layers_list    = shared["layers_list"]
    Q_lins         = shared["Q_lins"]
    Q_d            = shared["Q_d"]

    pac_params = dataclasses.replace(
        shared["pac_params_base"], k_transition=p["pac_transition"])
    k_model = KFieldModel(
        k_soil=1.351,
        soil_bands=shared["soil_bands"],
        pac_params=pac_params,
        layer_transition=p["layer_transition"],
    )
    T_amb     = problem.scenarios[0].T_amb   # 300.30 K
    r_sheaths = [layers_list[i][-1].r_outer for i in range(len(all_placements))]
    k_soil_bg = k_model.k_eff_bg(all_placements)

    set_seed(seed)

    # Build 128×5 student
    cfg_model = {
        "architecture":        "mlp",
        "width":               p["width"],
        "depth":               p["depth"],
        "activation":          "tanh",
        "fourier_features":    p["fourier_features"],
        "fourier_mapping_size": 64,
        "fourier_scale":       1.0,
    }
    base    = build_model(cfg_model, in_dim=2, device=device)
    student = ResidualPINNModel(
        base, layers_list, all_placements,
        k_soil_bg, T_amb, Q_lins,
        problem.domain, normalize=True, Q_d=Q_d,
        enable_grad_Tbg=True,
    )
    init_output_bias(student.base, 0.0)

    # Phase 0: Kennelly pretrain (use actual Dirichlet BCs at bottom/sides)
    _bc_pretrain = {
        "top":    T_amb,
        "bottom": problem.bcs["bottom"].value,
        "left":   problem.bcs["left"].value,
        "right":  problem.bcs["right"].value,
    }
    pretrain_multicable(
        student, all_placements, problem.domain,
        layers_list, Q_lins, k_soil_bg, T_amb,
        device=device, normalize=True,
        n_per_cable=1000, n_bc=200, n_steps=500, lr=1e-3, Q_d=Q_d,
        bc_temps=_bc_pretrain,
    )
    t_after_pretrain = time.time() - t0

    # Phase 1: Distillation warm-start
    print_ev = 200 if verbose else 0
    distil_from_teacher(
        student, teacher, problem.domain, device,
        n_points=distil_pts,
        n_steps=distil_steps,
        lr=distil_lr,
        print_every=print_ev,
    )
    t_after_distil = time.time() - t0

    # Phase 2: PDE fine-tune
    def k_fn_homog(xy: torch.Tensor) -> torch.Tensor:
        return torch.full((xy.shape[0], 1), k_soil_bg,
                          device=xy.device, dtype=xy.dtype)

    try:
        history = train_adam_lbfgs(
            model=student, domain=problem.domain, placements=all_placements,
            bcs=problem.bcs, T_amb=T_amb, r_sheaths=r_sheaths,
            k_fn=k_model,
            adam_steps=p["adam_steps"], lbfgs_steps=p["lbfgs_steps"],
            n_int=p["n_interior"], n_bnd=p["n_boundary"],
            oversample=p["oversample"], w_pde=p["w_pde"], w_bc=p["w_bc"],
            lr=p["lr"], print_every=10**9, normalize=True, device=device, logger=None,
            k_fn_warmup=k_fn_homog, warmup_frac=p["warmup_frac"],
            k_model=k_model, adam2_steps=0, adam2_lr=1e-5,
            lbfgs_history=p["lbfgs_history"], frac_pac_bnd=p["frac_pac_bnd"],
        )
        final_loss = history["total"][-1] if history.get("total") else float("nan")
    except Exception as exc:
        return {
            "T_max_C": float("nan"), "error_K": float("nan"),
            "final_loss": float("nan"), "elapsed_s": time.time() - t0,
            "t_pretrain_s": t_after_pretrain,
            "t_distil_s": t_after_distil - t_after_pretrain,
            "exception": str(exc), "model": None,
        }

    with torch.no_grad():
        T_conds = eval_conductor_temps(
            student, all_placements, problem.domain, device, normalize=True)
    T_worst = max(T_conds)
    return {
        "T_max_C": T_worst - 273.15,
        "error_K": T_worst - PAPER_T_MAX_PAC,
        "final_loss": final_loss,
        "elapsed_s": time.time() - t0,
        "t_pretrain_s": t_after_pretrain,
        "t_distil_s": t_after_distil - t_after_pretrain,
        "exception": None,
        "model": student,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(
        description="Distillation 64×4→128×5 with corrected T_amb=300.30 K")
    ap.add_argument("--n-seeds",      type=int,   default=5)
    ap.add_argument("--trial-seed",   type=int,   default=0)
    ap.add_argument("--distil-steps", type=int,   default=200,
                    help="Distillation Adam steps (default: 200 — soft warm-start)")
    ap.add_argument("--distil-lr",    type=float, default=2e-4)
    ap.add_argument("--distil-pts",   type=int,   default=2000)
    ap.add_argument("--verbose",      action="store_true")
    return ap.parse_args()


def main():
    args   = _parse_args()
    device = get_device("cpu")

    print("=" * 72)
    print("  CORRECTED T_amb — DISTILLATION 64×4→128×5")
    print("  T_amb = 300.30 K  (FEM surface temp, corrected from 290.15 K)")
    print("  FEM ref: %.1f °C" % FEM_REF_C)
    print("=" * 72)
    print()
    print("  Curriculum:")
    print("    Phase 0: Kennelly pretrain")
    print("    Phase 1: Distil from 64×4 teacher (%d steps @ lr=%.0e)" % (
        args.distil_steps, args.distil_lr))
    print("    Phase 2: PDE fine-tune (Adam %d + L-BFGS %d)" % (
        BEST_128x5["adam_steps"], BEST_128x5["lbfgs_steps"]))
    print()

    if not MODEL_64x4.exists():
        print("  ERROR: Teacher checkpoint not found:")
        print("    %s" % MODEL_64x4)
        print("  Run first: python examples/kim_2024_154kv_optim_B/run_corrected_64x4.py")
        return

    if not MODEL_128x5_BASELINE.exists():
        print("  WARNING: Baseline 128×5 not found (needed only for comparison):")
        print("    %s" % MODEL_128x5_BASELINE)
        print("  Proceeding without baseline comparison...")
    print()

    shared = load_problem_corrected()
    T_amb_csv = shared["problem"].scenarios[0].T_amb
    print("  T_amb from CSV: %.2f K (%.2f °C)" % (T_amb_csv, T_amb_csv - 273.15))
    print("  Q_total: %.2f W/m\n" % shared["Q_lin"])

    print("  Loading corrected 64×4 teacher …")
    teacher = load_teacher(shared, device)
    with torch.no_grad():
        T_teach = eval_conductor_temps(
            teacher, shared["all_placements"], shared["problem"].domain,
            device, normalize=True)
    print("  Teacher T_max = %.2f °C  (Error vs FEM: %+.2f K)\n" % (
        max(T_teach) - 273.15, max(T_teach) - (273.15 + FEM_REF_C)))

    seeds = list(range(args.trial_seed, args.trial_seed + args.n_seeds))
    print("  Seeds: %s\n" % seeds)
    print("  %-6s  %-8s  %-9s  %-10s  %-8s  %-10s  %-8s" % (
        "seed", "T_max°C", "Err[K]", "Loss", "t_tot[s]", "t_distil[s]", "t_pde[s]"))
    print("  " + "-" * 70)

    best_result      = None
    best_model_state = None
    best_seed        = seeds[0]

    for seed in seeds:
        r = run_distil_trial(
            teacher=teacher, shared=shared, device=device, seed=seed,
            distil_steps=args.distil_steps, distil_lr=args.distil_lr,
            distil_pts=args.distil_pts, verbose=args.verbose,
        )
        marker = " ✓" if r["exception"] is None and abs(r["error_K"]) <= 2.0 else ""
        if r["exception"]:
            print("  %-6d  ERROR: %s" % (seed, r["exception"]))
        else:
            t_pde = r["elapsed_s"] - r["t_pretrain_s"] - r["t_distil_s"]
            print("  %-6d  %7.2f   %+7.2f K  %.4e   %6.0fs   %6.0fs   %6.0fs%s" % (
                seed, r["T_max_C"], r["error_K"], r["final_loss"],
                r["elapsed_s"], r["t_distil_s"], t_pde, marker))
        if r["model"] is not None and (
            best_result is None or abs(r["error_K"]) < abs(best_result["error_K"])
        ):
            best_result      = r
            best_model_state = r["model"].state_dict()
            best_seed        = seed

    if best_model_state is None:
        print("\n  ERROR: all trials failed.")
        return

    model_path = RESULTS_DIR / "model_best_128x5_distil.pt"
    torch.save({
        "state_dict":   best_model_state,
        "seed":         best_seed,
        "T_max_C":      best_result["T_max_C"],
        "error_K":      best_result["error_K"],
        "distil_steps": args.distil_steps,
        "distil_lr":    args.distil_lr,
        "params":       BEST_128x5,
        "T_amb":        T_amb_csv,
        "note":         "corrected T_amb=300.30 K, distilled from corrected 64×4",
    }, model_path)

    print("\n  ── Best distilled (across %d seeds) ──" % len(seeds))
    print("  seed=%d  T_max=%.2f °C  Error=%+.2f K" % (
        best_seed, best_result["T_max_C"], best_result["error_K"]))
    print("  Model saved → %s" % model_path)

    # Print comparison summary
    print()
    print("  ── Summary: corrected T_amb=300.30 K ──")
    if MODEL_64x4.exists():
        ckpt64 = torch.load(MODEL_64x4, map_location="cpu", weights_only=False)
        print("  64×4  small  : %.2f °C  (%+.2f K vs FEM)" % (
            ckpt64["T_max_C"], ckpt64["T_max_C"] - FEM_REF_C))
    if MODEL_128x5_BASELINE.exists():
        ckpt128 = torch.load(MODEL_128x5_BASELINE, map_location="cpu", weights_only=False)
        print("  128×5 dense  : %.2f °C  (%+.2f K vs FEM)" % (
            ckpt128["T_max_C"], ckpt128["T_max_C"] - FEM_REF_C))
    print("  128×5 distil : %.2f °C  (%+.2f K vs FEM)" % (
        best_result["T_max_C"], best_result["T_max_C"] - FEM_REF_C))
    print("  FEM reference: %.1f °C" % FEM_REF_C)
    print()
    print("  Next: run  python examples/kim_2024_154kv_optim_C/run_all_corrected.py  (comparison)")


if __name__ == "__main__":
    main()
