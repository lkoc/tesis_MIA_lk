"""Distillation experiment — Kim 2024, Case B: 64×4 teacher → 128×5 student.

Multigrid-inspired 3-level curriculum:
  Level 0: Kennelly analytical solution  (pretrain_multicable, as always)
  Level 1: 64×4 PINN  ← trained model, loaded from results_optim_B
  Level 2: 128×5 PINN ← student trained via solution distillation then PDE fine-tune

Phase 0 — Pretrain (Kennelly):   same as baseline, already included in build
Phase 1 — Distillation warm-start:
    loss_distil = MSE(student(xy), teacher(xy).detach())
    → student learns the full T(x,y) field from teacher before seeing the PDE
Phase 2 — PDE fine-tune:
    loss_pde + loss_bc  (same train_adam_lbfgs as baseline 128×5)

Results are saved to  results_distil/  (separate from results_optim/)
to allow direct comparison with the baseline 128×5.

Comparison produced at the end:
  - Baseline 128×5  (best seed from run_optim_C --best)
  - Distilled 128×5 (this script, best seed)
  - FEM reference   70.6 °C

Usage::

    python examples/kim_2024_154kv_optim_C/run_distil_C.py
    python examples/kim_2024_154kv_optim_C/run_distil_C.py --n-seeds 5
    python examples/kim_2024_154kv_optim_C/run_distil_C.py --distil-steps 200 --distil-lr 2e-4

Key insight on distil_steps:
    Distillation must be a *soft warm-start*, NOT full convergence.
    Too many steps (>=500) specialise the student to the teacher field and
    destroy plasticity needed for PDE fine-tune (L-BFGS gets stuck).
    200 steps @ lr=2e-4 nudges the student toward T_teacher without locking in.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE        = Path(__file__).resolve().parent
DATA_DIR    = HERE.parent / "kim_2024_154kv_optim_B" / "data"
MODEL_64x4  = HERE.parent / "kim_2024_154kv_optim_B" / "results_optim" / "model_best_64x4.pt"
MODEL_128x5_BASELINE = HERE / "results_optim" / "model_best_128x5.pt"
RESULTS_DIR = HERE / "results_distil"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F  # noqa: N812

from pinn_cables.io.readers import load_problem                   # noqa: E402
from pinn_cables.materials.props import get_kim2024_cable_layers  # noqa: E402
from pinn_cables.physics.iec60287 import compute_iec60287_Q       # noqa: E402
from pinn_cables.physics.k_field import (                         # noqa: E402
    KFieldModel,
    load_physics_params,
    load_soil_layers,
)
from pinn_cables.physics.kennelly import iec60287_estimate        # noqa: E402
from pinn_cables.pinn.model import ResidualPINNModel, build_model # noqa: E402
from pinn_cables.pinn.train_custom import (                       # noqa: E402
    init_output_bias,
    pretrain_multicable,
    train_adam_lbfgs,
)
from pinn_cables.pinn.utils import get_device, set_seed           # noqa: E402
from pinn_cables.post.eval import eval_conductor_temps            # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPER_T_MAX_PAC = 273.15 + 70.6
PAPER_W_D       = 3.57
PAPER_FREQ      = 60.0
FEM_REF_C       = 70.6

# Best known params for 128×5 (from run_optim_C search)
BEST_128x5 = dict(
    lr           = 5e-4,
    adam_steps   = 2000,
    lbfgs_steps  = 1000,
    lbfgs_history= 100,
    warmup_frac  = 0.50,
    w_pde        = 0.5,
    w_bc         = 200.0,
    n_interior   = 2000,
    n_boundary   = 500,
    oversample   = 1,
    frac_pac_bnd = 0.50,
    layer_transition = 0.10,
    pac_transition   = 0.20,
)

# ---------------------------------------------------------------------------
# Problem helpers (identical to run_optim_C)
# ---------------------------------------------------------------------------

def load_case_b_problem() -> dict:
    problem = load_problem(DATA_DIR)
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
    layers     = get_kim2024_cable_layers(Q_lin)
    layers_list = [layers] * n_cables
    Q_lins      = [Q_lin]  * n_cables

    return dict(
        problem         = problem,
        all_placements  = all_placements,
        n_cables        = n_cables,
        soil_bands      = soil_bands,
        pac_params_base = pac_params_base,
        layers_list     = layers_list,
        Q_lins          = Q_lins,
        Q_d             = PAPER_W_D,
        Q_cond          = Q_cond,
        Q_lin           = Q_lin,
    )


def build_k_model(shared: dict, pac_transition: float, layer_transition: float) -> KFieldModel:
    pac_params = dataclasses.replace(
        shared["pac_params_base"], k_transition=pac_transition)
    return KFieldModel(
        k_soil=1.351,
        soil_bands=shared["soil_bands"],
        pac_params=pac_params,
        layer_transition=layer_transition,
    )


# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------

def load_teacher(shared: dict, device: torch.device) -> ResidualPINNModel:
    """Reconstruct the 64×4 ResidualPINNModel and load saved weights."""
    ckpt = torch.load(MODEL_64x4, map_location=device, weights_only=False)
    params = ckpt["params"]  # dataclasses.asdict(BEST_KNOWN) from run_optim_B

    k_model = build_k_model(
        shared,
        pac_transition   = params["pac_transition"],
        layer_transition = params["layer_transition"],
    )
    all_placements = shared["all_placements"]
    layers_list    = shared["layers_list"]
    Q_lins         = shared["Q_lins"]
    T_amb          = shared["problem"].scenarios[0].T_amb
    k_soil_bg      = k_model.k_eff_bg(all_placements)

    cfg_model = {
        "architecture":       "mlp",
        "width":              params["width"],
        "depth":              params["depth"],
        "activation":         "tanh",
        "fourier_features":   params["fourier_mapping_size"] > 0,
        "fourier_mapping_size": params.get("fourier_mapping_size", 64),
        "fourier_scale":      params.get("fourier_scale", 1.0),
    }
    base  = build_model(cfg_model, in_dim=2, device=device)
    model = ResidualPINNModel(
        base, layers_list, all_placements,
        k_soil_bg, T_amb, Q_lins,
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
    domain,          # Domain2D
    device: torch.device,
    n_points: int = 4000,
    n_steps: int = 1000,
    lr: float = 1e-3,
    print_every: int = 200,
) -> list[float]:
    """Phase 1: supervised distillation — student learns T(x,y) from teacher.

    Teacher output is detached (no gradients flow back to teacher).
    Returns list of distillation loss values (one per step).
    """
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []

    for step in range(1, n_steps + 1):
        # Sample uniform interior points each step
        x = torch.FloatTensor(n_points).uniform_(domain.xmin, domain.xmax)
        y = torch.FloatTensor(n_points).uniform_(domain.ymin, domain.ymax)
        xy = torch.stack([x, y], dim=1).to(device)
        xy.requires_grad_(False)

        with torch.no_grad():
            T_teacher = teacher(xy)   # (N,1)

        T_student = student(xy)       # (N,1)
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
    distil_steps: int = 1000,
    distil_lr:    float = 1e-3,
    distil_pts:   int = 4000,
    verbose:      bool = False,
) -> dict:
    """Build 128×5 student, distil from teacher, then PDE fine-tune."""

    all_placements = shared["all_placements"]
    layers_list    = shared["layers_list"]
    Q_lins         = shared["Q_lins"]
    Q_d            = shared["Q_d"]
    problem        = shared["problem"]

    p = BEST_128x5
    t0 = time.time()

    k_model   = build_k_model(shared, p["pac_transition"], p["layer_transition"])
    T_amb     = problem.scenarios[0].T_amb
    r_sheaths = [layers_list[i][-1].r_outer for i in range(len(all_placements))]
    k_soil_bg = k_model.k_eff_bg(all_placements)

    set_seed(seed)

    # ── Build 128×5 student ─────────────────────────────────────────
    cfg_model = {
        "architecture":        "mlp",
        "width":               128,
        "depth":               5,
        "activation":          "tanh",
        "fourier_features":    False,
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

    # ── Phase 0: Kennelly pretrain ───────────────────────────────────
    pretrain_multicable(
        student, all_placements, problem.domain,
        layers_list, Q_lins, k_soil_bg, T_amb,
        device=device, normalize=True,
        n_per_cable=1000, n_bc=200, n_steps=500, lr=1e-3,
        Q_d=Q_d,
    )
    t_after_pretrain = time.time() - t0

    # ── Phase 1: Distillation warm-start ────────────────────────────
    print_ev = 200 if verbose else 0
    _ = distil_from_teacher(
        student, teacher, problem.domain, device,
        n_points=distil_pts,
        n_steps=distil_steps,
        lr=distil_lr,
        print_every=print_ev,
    )
    t_after_distil = time.time() - t0

    # ── Phase 2: PDE fine-tune ───────────────────────────────────────
    def k_fn_homog(xy: torch.Tensor) -> torch.Tensor:
        return torch.full((xy.shape[0], 1), k_soil_bg,
                          device=xy.device, dtype=xy.dtype)

    try:
        history = train_adam_lbfgs(
            model        = student,
            domain       = problem.domain,
            placements   = all_placements,
            bcs          = problem.bcs,
            T_amb        = T_amb,
            r_sheaths    = r_sheaths,
            k_fn         = k_model,
            adam_steps   = p["adam_steps"],
            lbfgs_steps  = p["lbfgs_steps"],
            n_int        = p["n_interior"],
            n_bnd        = p["n_boundary"],
            oversample   = p["oversample"],
            w_pde        = p["w_pde"],
            w_bc         = p["w_bc"],
            lr           = p["lr"],
            print_every  = 10**9,
            normalize    = True,
            device       = device,
            logger       = None,
            k_fn_warmup  = k_fn_homog,
            warmup_frac  = p["warmup_frac"],
            k_model      = k_model,
            adam2_steps  = 0,
            adam2_lr     = 1e-5,
            lbfgs_history= p["lbfgs_history"],
            frac_pac_bnd = p["frac_pac_bnd"],
        )
        final_loss = history["total"][-1] if history.get("total") else float("nan")
    except Exception as exc:
        return {
            "T_max_C":        float("nan"),
            "error_K":        float("nan"),
            "final_loss":     float("nan"),
            "elapsed_s":      time.time() - t0,
            "t_pretrain_s":   t_after_pretrain,
            "t_distil_s":     t_after_distil - t_after_pretrain,
            "exception":      str(exc),
            "model":          None,
        }

    with torch.no_grad():
        T_conds = eval_conductor_temps(
            student, all_placements, problem.domain, device, normalize=True)
        T_worst = max(T_conds)

    T_max_C = T_worst - 273.15
    error_K = T_worst - PAPER_T_MAX_PAC

    return {
        "T_max_C":      T_max_C,
        "error_K":      error_K,
        "final_loss":   final_loss,
        "elapsed_s":    time.time() - t0,
        "t_pretrain_s": t_after_pretrain,
        "t_distil_s":   t_after_distil - t_after_pretrain,
        "exception":    None,
        "model":        student,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Distillation 64×4→128×5 experiment, Kim 2024 Case B")
    ap.add_argument("--n-seeds",      type=int,   default=5,
                    help="Number of seeds (default: 5)")
    ap.add_argument("--trial-seed",   type=int,   default=0,
                    help="Base seed for trials (default: 0)")
    ap.add_argument("--distil-steps", type=int,   default=200,
                    help="Distillation Adam steps (default: 200 — soft warm-start, not convergence)")
    ap.add_argument("--distil-lr",    type=float, default=2e-4,
                    help="Distillation learning rate (default: 2e-4 — gentle nudge)")
    ap.add_argument("--distil-pts",   type=int,   default=2000,
                    help="Domain points sampled per distil step (default: 2000)")
    ap.add_argument("--verbose",      action="store_true",
                    help="Print distillation loss every 200 steps")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args  = _parse_args()
    device = get_device("cpu")

    print("=" * 72)
    print("  DISTILLATION EXPERIMENT — Kim 2024 Case B")
    print("  Teacher: 64×4  →  Student: 128×5 (3-level curriculum)")
    print("  FEM ref: %.1f °C" % FEM_REF_C)
    print("=" * 72)
    print()
    print("  Curriculum levels:")
    print("    0: Kennelly pretrain    (analytical homogeneous)")
    print("    1: 64×4 teacher         (soft warm-start %d steps @ lr=%.0e)" % (
        args.distil_steps, args.distil_lr))
    print("    2: PDE fine-tune        (Adam %d + L-BFGS %d, same as baseline)" % (
        BEST_128x5["adam_steps"], BEST_128x5["lbfgs_steps"]))
    print()

    # ── Check teacher checkpoint exists ────────────────────────────
    if not MODEL_64x4.exists():
        print("  ERROR: Teacher checkpoint not found:")
        print("    %s" % MODEL_64x4)
        print("  Run first:  python examples/kim_2024_154kv_optim_B/run_optim_B.py --best")
        return

    print("  Loading problem data …")
    shared = load_case_b_problem()
    print("  Q_cond = %.2f W/m  Q_d = %.2f W/m  Q_total = %.2f W/m" % (
        shared["Q_cond"], shared["Q_d"], shared["Q_lin"]))
    iec_res = iec60287_estimate(
        shared["layers_list"], shared["problem"].placements,
        shared["soil_bands"][1].k, shared["problem"].scenarios[0].T_amb,
        shared["Q_lins"], Q_d=shared["Q_d"],
    )
    print("  IEC 60287 (k=%.3f, homog.): %.1f °C" % (
        shared["soil_bands"][1].k, iec_res["T_cond_ref"] - 273.15))
    print()

    print("  Loading teacher (64×4) from:")
    print("    %s" % MODEL_64x4)
    teacher = load_teacher(shared, device)
    # Verify teacher quality
    with torch.no_grad():
        T_teach = eval_conductor_temps(
            teacher, shared["all_placements"], shared["problem"].domain,
            device, normalize=True)
    print("  Teacher T_max = %.2f °C  (Error vs FEM: %+.2f K)" % (
        max(T_teach) - 273.15, max(T_teach) - (273.15 + FEM_REF_C)))
    print()

    # ── Multi-seed distilled run ─────────────────────────────────────
    seeds = list(range(args.trial_seed, args.trial_seed + args.n_seeds))
    print("  ── Distilled 128×5 — %d seeds ──" % len(seeds))
    print("  %-6s  %-8s  %-9s  %-10s  %-8s  %-8s  %-8s" % (
        "seed", "T_max°C", "Err[K]", "Loss", "t_tot[s]", "t_distil[s]", "t_pde[s]"))
    print("  " + "-" * 70)

    best_result      = None
    best_model_state = None
    best_seed        = args.trial_seed
    all_results      = []

    for seed in seeds:
        r = run_distil_trial(
            teacher      = teacher,
            shared       = shared,
            device       = device,
            seed         = seed,
            distil_steps = args.distil_steps,
            distil_lr    = args.distil_lr,
            distil_pts   = args.distil_pts,
            verbose      = args.verbose,
        )
        all_results.append(r)
        marker = " ✓" if abs(r["error_K"]) <= 2.0 else ""
        t_pde  = r["elapsed_s"] - r["t_pretrain_s"] - r["t_distil_s"]
        print("  %-6d  %7.2f   %+7.2f K  %.4e   %6.0fs   %6.0fs   %6.0fs%s" % (
            seed, r["T_max_C"], r["error_K"], r["final_loss"],
            r["elapsed_s"], r["t_distil_s"], t_pde, marker))
        if best_result is None or abs(r["error_K"]) < abs(best_result["error_K"]):
            best_result      = r
            best_model_state = r["model"].state_dict() if r["model"] else None
            best_seed        = seed

    # ── Save best distilled model ────────────────────────────────────
    model_path = RESULTS_DIR / "model_best_128x5_distil.pt"
    if best_model_state is not None:
        torch.save({
            "state_dict":   best_model_state,
            "seed":         best_seed,
            "T_max_C":      best_result["T_max_C"],
            "error_K":      best_result["error_K"],
            "distil_steps": args.distil_steps,
            "distil_lr":    args.distil_lr,
            "params":       BEST_128x5,
        }, model_path)

    # ── Summary ──────────────────────────────────────────────────────
    valid = [r for r in all_results if r["exception"] is None]
    n_ok  = sum(1 for r in valid if abs(r["error_K"]) <= 2.0)
    errors = [abs(r["error_K"]) for r in valid]
    mean_err = sum(errors) / len(errors) if errors else float("nan")

    print()
    print("=" * 72)
    print("  DISTILLATION RESULTS (128×5 student)")
    print("=" * 72)
    print("  Best:    seed=%d  T_max=%.2f °C  Error=%+.2f K" % (
        best_seed, best_result["T_max_C"], best_result["error_K"]))
    print("  Seeds ✓: %d/%d within ±2K  |  Mean |err| = %.2f K" % (
        n_ok, len(seeds), mean_err))
    if best_model_state:
        print("  Saved → %s" % model_path)
    print()

    # ── Comparison table ─────────────────────────────────────────────
    print("  ── Comparison: IEC 60287 vs PINN 64×4 vs PINN 128×5 vs Distilled 128×5 vs FEM ──")
    print()
    print("  %-25s  %-10s  %-12s  %-10s" % ("Method", "T_max °C", "Error vs FEM", "Seeds ✓/5"))
    print("  " + "-" * 62)
    print("  %-25s  %-10s  %-12s  %-10s" % (
        "IEC 60287 (homog.)", "73.9", "+3.3 K", "—"))
    print("  %-25s  %-10s  %-12s  %-10s" % (
        "PINN 64×4 (baseline)", "70.19", "−0.41 K", "2/5"))
    print("  %-25s  %-10s  %-12s  %-10s" % (
        "PINN 128×5 (baseline)", "70.18", "−0.42 K", "3/5"))
    print("  %-25s  %-10s  %-12s  %-10s" % (
        "PINN 128×5 (distilled)",
        "%.2f" % best_result["T_max_C"],
        "%+.2f K" % best_result["error_K"],
        "%d/%d" % (n_ok, len(seeds))))
    print("  %-25s  %-10s  %-12s  %-10s" % (
        "FEM (reference)", "70.6", "—", "—"))
    print("=" * 72)

    # ── Timing analysis ───────────────────────────────────────────────
    if valid:
        avg_distil = sum(r["t_distil_s"] for r in valid) / len(valid)
        avg_total  = sum(r["elapsed_s"]  for r in valid) / len(valid)
        baseline_128x5_avg = 434.0  # avg from run_optim_C --best (from logs)
        overhead = avg_distil
        print()
        print("  ── Timing (avg over %d seeds) ──" % len(valid))
        print("  Distillation phase:  %.0f s" % avg_distil)
        print("  Total (distilled):   %.0f s" % avg_total)
        print("  Baseline 128×5:      %.0f s" % baseline_128x5_avg)
        print("  Overhead:           +%.0f s (+%.0f%%)" % (
            overhead,
            100.0 * overhead / baseline_128x5_avg if baseline_128x5_avg > 0 else 0))
    print()


if __name__ == "__main__":
    main()
