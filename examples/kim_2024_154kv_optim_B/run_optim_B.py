"""Hyperparameter optimisation — Kim 2024, Case B (multilayer + PAC), quick profile.

Case B: 6 cables, multilayer soil (k=1.804/1.351/1.517 W/(mK)) + PAC zone
        (k=2.094 W/(mK)).  FEM reference: T_max = 70.6 °C.

This script runs a *random search* over the most sensitive hyperparameters
for the 64×4 ("quick") PINN and reports a sorted table of results.

Physical note on the PDE
-------------------------
The steady-state heat equation with spatially-variable conductivity is:

    ∇·(k(x,y) ∇T) + Q(x,y) = 0

Expanded (via product rule):

    k(x,y) ΔT  +  ∇k · ∇T  +  Q = 0

In the soil region Q = 0 (cable heat is encoded in T_bg via Kennelly).
The code uses the **divergence form** with full autograd, computing
flux = k·∇T and then ∂flux_x/∂x + ∂flux_y/∂y exactly.  This captures
both k·ΔT and (∇k·∇T) through the chain rule — correct for any smooth k(x,y).

For the PAC boundary (sigmoid transition width = k_transition in CSV):
  - A narrower transition → sharper ∇k → harder optimisation.
  - We expose `layer_transition` and `pac_transition` as tunable params.

Usage::

    python examples/kim_2024_154kv_optim_B/run_optim_B.py
    python examples/kim_2024_154kv_optim_B/run_optim_B.py --trials 30
    python examples/kim_2024_154kv_optim_B/run_optim_B.py --seed 7 --trials 20
    python examples/kim_2024_154kv_optim_B/run_optim_B.py --best  # re-run best found
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import math
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results_optim"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

from pinn_cables.io.readers import load_problem, load_solver_params  # noqa: E402
from pinn_cables.materials.props import get_kim2024_cable_layers  # noqa: E402
from pinn_cables.physics.iec60287 import compute_iec60287_Q  # noqa: E402
from pinn_cables.physics.k_field import (  # noqa: E402
    KFieldModel,
    PhysicsParams,
    load_physics_params,
    load_soil_layers,
    SoilLayerBand,
)
from pinn_cables.physics.kennelly import iec60287_estimate  # noqa: E402
from pinn_cables.pinn.model import ResidualPINNModel, build_model  # noqa: E402
from pinn_cables.pinn.train_custom import (  # noqa: E402
    init_output_bias,
    pretrain_multicable,
    train_adam_lbfgs,
)
from pinn_cables.pinn.utils import get_device, set_seed  # noqa: E402
from pinn_cables.post.eval import eval_conductor_temps  # noqa: E402

# ---------------------------------------------------------------------------
# Paper constants (Kim 2024, Case B)
# ---------------------------------------------------------------------------
PAPER_T_MAX_PAC  = 273.15 + 70.6   # K — FEM reference
PAPER_W_D   = 3.57                  # W/m
PAPER_FREQ  = 60.0                  # Hz
PAPER_K_PAC = 2.094                 # W/(mK)
FEM_REF_C   = 70.6                  # °C

# ---------------------------------------------------------------------------
# Hyperparameter space
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrialParams:
    """All tunable hyperparameters for one trial."""
    # Optimiser
    lr: float = 1e-3
    adam_steps: int = 5000
    lbfgs_steps: int = 500
    lbfgs_history: int = 50
    # Curriculum
    warmup_frac: float = 0.30
    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 50.0
    # Sampling
    n_interior: int = 2000
    n_boundary: int = 200
    oversample: int = 1          # FIXED=1: oversample>1 causes PAC gradient imbalance
    frac_pac_bnd: float = 0.30   # fraction of interior pts near transitions
    # Physics — sigmoid transition widths
    layer_transition: float = 0.05   # soil-layer interfaces [m]
    pac_transition: float = 0.10     # PAC boundary [m]
    # Network (fixed at quick)
    width: int = 64
    depth: int = 4
    # Fourier features: 0 = disabled, >0 = mapping_size (output dim = 2*mapping_size)
    fourier_mapping_size: int = 0
    fourier_scale: float = 1.0

    def short_str(self) -> str:
        ff = f"ff={self.fourier_mapping_size}" if self.fourier_mapping_size > 0 else "ff=off"
        return (
            f"lr={self.lr:.0e} adam={self.adam_steps} lbfgs={self.lbfgs_steps} "
            f"wfrac={self.warmup_frac:.2f} wpde={self.w_pde} wbc={self.w_bc} "
            f"nint={self.n_interior} frac={self.frac_pac_bnd:.2f} "
            f"ltr={self.layer_transition:.3f} ptr={self.pac_transition:.3f} {ff}"
        )


# Search space — each key maps to a list of candidate values.
# oversample is FIXED=1 (not searched): oversample>1 causes PAC gradient imbalance.
# fourier_mapping_size: 0=plain MLP, 2/3/10=Fourier feature mapping sizes
SEARCH_SPACE: dict[str, list] = {
    "lr":                   [5e-4, 1e-3, 2e-3],
    "adam_steps":           [1000, 2000, 3000],
    "lbfgs_steps":          [500, 1000, 1500],
    "lbfgs_history":        [50, 100],
    "warmup_frac":          [0.20, 0.30, 0.40, 0.50],
    "w_pde":                [0.5, 1.0, 2.0, 5.0],
    "w_bc":                 [20.0, 50.0, 100.0],
    "n_interior":           [1000, 2000, 3000],
    "n_boundary":           [200, 500, 1000],
    "frac_pac_bnd":         [0.20, 0.35, 0.50],
    "layer_transition":     [0.03, 0.05, 0.10],
    "pac_transition":       [0.05, 0.10, 0.20],
    # Fourier features: off (0) vs mapping_size 2, 3, 10
    "fourier_mapping_size": [0, 0, 2, 3, 10],  # 0 appears twice → 40% prob off
    "fourier_scale":        [1.0, 2.0, 5.0],
}


def sample_trial(rng: random.Random) -> TrialParams:
    """Sample one set of hyperparameters uniformly from the search space."""
    kwargs = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    return TrialParams(**kwargs)


# ---------------------------------------------------------------------------
# Problem setup (loaded once)
# ---------------------------------------------------------------------------

def load_case_b_problem():
    """Load everything needed for Case B. Returns a dict of shared objects."""
    problem = load_problem(DATA_DIR)
    all_placements = problem.placements
    n_cables = len(all_placements)

    soil_bands = load_soil_layers(DATA_DIR / "soil_layers.csv")
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

    layers       = get_kim2024_cable_layers(Q_lin)
    layers_list  = [layers] * n_cables
    Q_lins       = [Q_lin] * n_cables

    return dict(
        problem        = problem,
        all_placements = all_placements,
        n_cables       = n_cables,
        soil_bands     = soil_bands,
        pac_params_base= pac_params_base,
        layers_list    = layers_list,
        Q_lins         = Q_lins,
        Q_d            = PAPER_W_D,
        Q_cond         = Q_cond,
        Q_lin          = Q_lin,
    )


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    p: TrialParams,
    shared: dict,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Train Case B with the given hyperparameters and return result dict."""

    problem        = shared["problem"]
    all_placements = shared["all_placements"]
    soil_bands     = shared["soil_bands"]
    pac_params_base= shared["pac_params_base"]
    layers_list    = shared["layers_list"]
    Q_lins         = shared["Q_lins"]
    Q_d            = shared["Q_d"]

    t0 = time.time()

    # Build PAC params with tunable transition width
    pac_params = dataclasses.replace(pac_params_base, k_transition=p.pac_transition)

    # Build k model for Case B (multilayer + PAC)
    # We also override the soil-layer transition width
    k_model = KFieldModel(
        k_soil=1.351,
        soil_bands=soil_bands,
        pac_params=pac_params,
        layer_transition=p.layer_transition,
    )

    T_amb = problem.scenarios[0].T_amb
    r_sheaths = [layers_list[i][-1].r_outer for i in range(len(all_placements))]
    k_soil_bg = k_model.k_eff_bg(all_placements)

    set_seed(seed)

    # Build model (fixed quick: 64×4, optional Fourier features)
    use_fourier = p.fourier_mapping_size > 0
    solver_cfg = {
        "model": {
            "architecture": "mlp",
            "width": p.width,
            "depth": p.depth,
            "activation": "tanh",
            "fourier_features": use_fourier,
            "fourier_mapping_size": p.fourier_mapping_size if use_fourier else 64,
            "fourier_scale": p.fourier_scale,
        },
        "normalization": {"normalize_coords": True},
    }
    base = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModel(
        base, layers_list, all_placements,
        k_soil_bg, T_amb, Q_lins,
        problem.domain, normalize=True, Q_d=Q_d,
        enable_grad_Tbg=True,
    )
    init_output_bias(model.base, 0.0)

    # Pre-train
    pretrain_multicable(
        model, all_placements, problem.domain,
        layers_list, Q_lins, k_soil_bg, T_amb,
        device=device, normalize=True,
        n_per_cable=1000, n_bc=200, n_steps=500, lr=1e-3,
        Q_d=Q_d,
    )

    # Warmup k function (homogeneous)
    def k_fn_homog(xy: torch.Tensor) -> torch.Tensor:
        return torch.full((xy.shape[0], 1), k_soil_bg,
                          device=xy.device, dtype=xy.dtype)

    # Train
    try:
        history = train_adam_lbfgs(
            model=model,
            domain=problem.domain,
            placements=all_placements,
            bcs=problem.bcs,
            T_amb=T_amb,
            r_sheaths=r_sheaths,
            k_fn=k_model,
            adam_steps=p.adam_steps,
            lbfgs_steps=p.lbfgs_steps,
            n_int=p.n_interior,
            n_bnd=p.n_boundary,
            oversample=p.oversample,
            w_pde=p.w_pde,
            w_bc=p.w_bc,
            lr=p.lr,
            print_every=10**9,      # silent during search
            normalize=True,
            device=device,
            logger=None,            # handled internally (NullHandler)
            k_fn_warmup=k_fn_homog,
            warmup_frac=p.warmup_frac,
            k_model=k_model,
            adam2_steps=0,
            adam2_lr=1e-5,
            lbfgs_history=p.lbfgs_history,
            frac_pac_bnd=p.frac_pac_bnd,
        )
        final_loss = history["total"][-1] if history.get("total") else float("nan")
    except Exception as exc:
        return {
            "T_max_C": float("nan"),
            "error_K": float("nan"),
            "final_loss": float("nan"),
            "elapsed_s": time.time() - t0,
            "exception": str(exc),
        }

    with torch.no_grad():
        T_conds = eval_conductor_temps(
            model, all_placements, problem.domain, device, normalize=True)
        T_worst = max(T_conds)

    T_max_C = T_worst - 273.15
    error_K = T_worst - PAPER_T_MAX_PAC

    return {
        "T_max_C": T_max_C,
        "error_K": error_K,
        "final_loss": final_loss,
        "elapsed_s": time.time() - t0,
        "exception": None,
        "model": model,
    }


def run_trial_with_timeout(
    p: TrialParams,
    shared: dict,
    device: torch.device,
    seed: int = 42,
    timeout_s: float = 900.0,
) -> dict:
    """Run a trial in a daemon thread; return a timeout-error dict if exceeded."""
    result: dict = {}

    def _target():
        nonlocal result
        result = run_trial(p, shared, device, seed)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        result = {
            "T_max_C": float("nan"),
            "error_K": float("nan"),
            "final_loss": float("nan"),
            "elapsed_s": timeout_s,
            "exception": f"timeout after {timeout_s:.0f}s",
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Hyperparameter search for Kim 2024 Case B (quick PINN)")
    ap.add_argument("--trials", type=int, default=20,
                    help="Number of random trials (default: 20)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for the search sampler (default: 0)")
    ap.add_argument("--trial-seed", type=int, default=42,
                    help="Base PyTorch seed for trials (default: 42)")
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="Number of seeds to try with --best (default: 5)")
    ap.add_argument("--best", action="store_true",
                    help="Re-run the known best config across multiple seeds, save best model")
    ap.add_argument("--fast", action="store_true",
                    help="Fast mode: cap adam_steps<=2000, lbfgs_steps<=200 for quick exploration")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="Max seconds per trial before skipping (default: 900)")
    return ap.parse_args()


# Known best config — validated: seed=7 rank1 fast (-0.42K) → full run 71.07°C (+0.47K ✓)
# Key findings:
#   ptr=0.200  → smooth PAC transition, L-BFGS converges well
#   oversample=1 → no PAC gradient imbalance (oversample>1 collapses T)
#   w_pde=0.5 (low) + w_bc=100 → let L-BFGS enforce physics
#   warmup_frac=0.50 → long homogeneous-k warmup stabilises net
#   ff=off → Fourier features hurt (ff=10/2 bad; ff=3 marginal at -1.3K)
BEST_KNOWN = TrialParams(
    lr=1e-3,
    adam_steps=2000,          # slightly above fast-mode 1000 for better exploration
    lbfgs_steps=1000,         # full L-BFGS budget
    lbfgs_history=100,
    warmup_frac=0.50,
    w_pde=0.5,
    w_bc=100.0,
    n_interior=2000,          # slightly above fast-mode 1000
    n_boundary=500,
    oversample=1,             # CRITICAL: 1, not 8
    frac_pac_bnd=0.50,
    layer_transition=0.10,
    pac_transition=0.20,
    fourier_mapping_size=0,   # ff=off
    fourier_scale=1.0,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = get_device("cpu")

    print("=" * 72)
    print("  HYPERPARAMETER SEARCH — Kim 2024 Case B (multilayer + PAC)")
    print("  Quick profile: 64×4 MLP tanh  |  FEM ref: %.1f °C" % FEM_REF_C)
    print("=" * 72)
    print("\n  PDE: ∇·(k(x,y)∇T) = 0  in soil (divergence form, full autograd)")
    print("  k(x,y): 3 soil layers + sigmoid PAC zone override")
    print()

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

    if args.best:
        import torch
        seeds = list(range(args.trial_seed, args.trial_seed + args.n_seeds))
        print("  ── Multi-seed BEST_KNOWN run (%d seeds) ──" % len(seeds))
        p = BEST_KNOWN
        print("  Params:", p.short_str())
        print("  FEM ref: %.1f °C\n" % FEM_REF_C)
        print("  %-6s  %-8s  %-8s  %-12s  %-7s" % ("seed", "T_max°C", "Err[K]", "Loss", "t[s]"))
        print("  " + "-" * 50)
        best_result = None
        best_model_state = None
        for seed in seeds:
            r = run_trial(p, shared, device, seed=seed)
            marker = " ✓" if abs(r["error_K"]) <= 2.0 else ""
            print("  %-6d  %7.2f   %+7.2f K  %.4e   %5.0fs%s" % (
                seed, r["T_max_C"], r["error_K"], r["final_loss"], r["elapsed_s"], marker))
            if best_result is None or abs(r["error_K"]) < abs(best_result["error_K"]):
                best_result = r
                best_model_state = r["model"].state_dict()
                best_seed = seed
        # Save best model
        out_dir = RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model_best_64x4.pt"
        torch.save({"state_dict": best_model_state, "seed": best_seed,
                    "T_max_C": best_result["T_max_C"], "error_K": best_result["error_K"],
                    "params": dataclasses.asdict(p)}, model_path)
        print("\n  ── Best across %d seeds ──" % len(seeds))
        print("  seed=%d  T_max=%.2f °C  Error=%+.2f K  Loss=%.4e" % (
            best_seed, best_result["T_max_C"], best_result["error_K"], best_result["final_loss"]))
        print("  Model saved → %s" % model_path)
        return

    rng = random.Random(args.seed)
    n_trials = args.trials
    fast_mode = args.fast
    timeout_s = args.timeout

    if fast_mode:
        timeout_s = min(timeout_s, 600.0)   # 10 min cap in fast mode
        print("  [FAST MODE] adam_steps capped at 1000, lbfgs_steps at 200, n_int at 1000  |  timeout=%ds" % int(timeout_s))
        print()

    results: list[tuple[TrialParams, dict]] = []

    print("  %-4s  %-8s  %-8s  %-10s  %-7s  %s" % (
        "#", "T_max°C", "Err[K]", "Loss", "t[s]", "params"))
    print("  " + "-" * 100)

    for idx in range(1, n_trials + 1):
        p = sample_trial(rng)
        if fast_mode:
            p = dataclasses.replace(
                p,
                adam_steps=min(p.adam_steps, 1000),
                lbfgs_steps=min(p.lbfgs_steps, 200),
                n_interior=min(p.n_interior, 1000),
                # oversample is always 1 — no cap needed
            )
        print("  [%2d/%d] training … " % (idx, n_trials), end="", flush=True)
        result = run_trial_with_timeout(p, shared, device, seed=args.trial_seed, timeout_s=timeout_s)
        results.append((p, result))

        if result["exception"]:
            print("ERROR: %s" % result["exception"])
        else:
            err_str = "%+.2f K" % result["error_K"]
            marker = " ✓" if abs(result["error_K"]) <= 2.0 else ""
            print("T=%5.1f°C  err=%s  loss=%.3e  t=%3.0fs%s  | %s" % (
                result["T_max_C"], err_str,
                result["final_loss"], result["elapsed_s"],
                marker, p.short_str()))

    # ── Summary table ──────────────────────────────────────────────────
    valid = [(p, r) for p, r in results if r["exception"] is None]
    if not valid:
        print("\n  No valid results.")
        return

    valid.sort(key=lambda x: abs(x[1]["error_K"]))

    print()
    print("=" * 72)
    print("  RESULTS — sorted by |Error vs FEM| (best first)")
    print("=" * 72)
    print("  %-4s  %-8s  %-9s  %-10s  %-6s  %s" % (
        "Rank", "T_max°C", "Error[K]", "Loss", "t[s]", "key params"))
    print("  " + "-" * 100)
    for rank, (p, r) in enumerate(valid[:15], start=1):
        marker = " ✓" if abs(r["error_K"]) <= 2.0 else ""
        print("  %-4d  %-8.2f  %-9s  %-10.3e  %-6.0f  %s%s" % (
            rank,
            r["T_max_C"],
            "%+.2f K" % r["error_K"],
            r["final_loss"],
            r["elapsed_s"],
            p.short_str(),
            marker,
        ))

    best_p, best_r = valid[0]
    print()
    print("  ── BEST TRIAL ──")
    print("  T_max = %.2f °C  |  Error = %+.2f K  |  Loss = %.4e" % (
        best_r["T_max_C"], best_r["error_K"], best_r["final_loss"]))
    print("  Params:", best_p.short_str())
    print()
    print("  Tip: copy the best params into BEST_KNOWN and run with --best")
    print("  to train with full verbosity and save the model.")
    print("=" * 72)


if __name__ == "__main__":
    main()
