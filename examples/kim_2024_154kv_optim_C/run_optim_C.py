"""Hyperparameter search — Kim 2024, Case B, deep profile (128×5 MLP).

Builds on the findings from the 64×4 quick search (run_optim_B.py):
  - oversample = 1        CRITICAL (oversample>1 → PAC gradient imbalance → T collapse)
  - pac_transition = 0.20  dominant physics parameter
  - warmup_frac = 0.50     long homogeneous-k warmup stabilises net
  - ff = off               Fourier features hurt (amplify noise at PAC sigmoid)
  - layer_transition = 0.10

Only four hyperparameters are searched for the deeper architecture:
  lr, adam_steps, lbfgs_steps, w_pde, w_bc

FEM reference: T_max = 70.6 °C (Kim 2024, Case B).

Usage::

    python examples/kim_2024_154kv_optim_C/run_optim_C.py
    python examples/kim_2024_154kv_optim_C/run_optim_C.py --trials 20 --fast
    python examples/kim_2024_154kv_optim_C/run_optim_C.py --best --n-seeds 5
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE      = Path(__file__).resolve().parent
# Reuse Case B data (same problem, different search)
DATA_DIR  = HERE.parent / "kim_2024_154kv_optim_B" / "data"
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
PAPER_T_MAX_PAC = 273.15 + 70.6   # K — FEM reference
PAPER_W_D       = 3.57             # W/m
PAPER_FREQ      = 60.0             # Hz
PAPER_K_PAC     = 2.094            # W/(mK)
FEM_REF_C       = 70.6             # °C

# ---------------------------------------------------------------------------
# Hyperparameter dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrialParams:
    """Tunable hyperparameters for the 128×5 deep profile.

    Fixed by prior 64×4 search (not searched here):
        pac_transition  = 0.20   dominant physics param
        warmup_frac     = 0.50   long warmup stabilises net
        oversample      = 1      CRITICAL — never increase
        fourier_mapping = 0      ff=off, Fourier features hurt
        layer_transition = 0.10  from BEST_KNOWN
        frac_pac_bnd    = 0.50
        lbfgs_history   = 100
        n_boundary      = 500
        n_interior      = 2000
    """
    # ── Searched ──────────────────────────────────────────────────────
    lr:           float = 1e-3
    adam_steps:   int   = 3000
    lbfgs_steps:  int   = 1000
    w_pde:        float = 0.5
    w_bc:         float = 100.0

    # ── Fixed from 64×4 findings ──────────────────────────────────────
    lbfgs_history:    int   = 100
    warmup_frac:      float = 0.50
    n_interior:       int   = 2000
    n_boundary:       int   = 500
    oversample:       int   = 1        # CRITICAL
    frac_pac_bnd:     float = 0.50
    layer_transition: float = 0.10
    pac_transition:   float = 0.20     # dominant physics param
    # ── Network (deep profile) ────────────────────────────────────────
    width:             int   = 128
    depth:             int   = 5
    fourier_mapping_size: int   = 0    # ff=off
    fourier_scale:        float = 1.0

    def short_str(self) -> str:
        return (
            f"lr={self.lr:.0e} adam={self.adam_steps} lbfgs={self.lbfgs_steps} "
            f"wpde={self.w_pde} wbc={self.w_bc} "
            f"[{self.width}×{self.depth} | ptr=0.20 wfrac=0.50 ff=off]"
        )


# ---------------------------------------------------------------------------
# Reduced search space (only the 5 params that matter for the deep net)
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, list] = {
    "lr":          [5e-4, 1e-3, 2e-3],
    "adam_steps":  [2000, 3000, 5000],
    "lbfgs_steps": [500, 1000, 1500],
    "w_pde":       [0.5, 1.0, 2.0],
    "w_bc":        [50.0, 100.0, 200.0],
}


def sample_trial(rng: random.Random) -> TrialParams:
    kwargs = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    return TrialParams(**kwargs)


# ---------------------------------------------------------------------------
# Problem setup (loaded once)
# ---------------------------------------------------------------------------

def load_case_b_problem() -> dict:
    problem = load_problem(DATA_DIR)
    all_placements = problem.placements
    n_cables = len(all_placements)

    soil_bands     = load_soil_layers(DATA_DIR / "soil_layers.csv")
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


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    p: TrialParams,
    shared: dict,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Train Case B with 128×5 MLP and given hyperparameters."""

    problem         = shared["problem"]
    all_placements  = shared["all_placements"]
    soil_bands      = shared["soil_bands"]
    pac_params_base = shared["pac_params_base"]
    layers_list     = shared["layers_list"]
    Q_lins          = shared["Q_lins"]
    Q_d             = shared["Q_d"]

    t0 = time.time()

    pac_params = dataclasses.replace(pac_params_base, k_transition=p.pac_transition)
    k_model = KFieldModel(
        k_soil=1.351,
        soil_bands=soil_bands,
        pac_params=pac_params,
        layer_transition=p.layer_transition,
    )

    T_amb       = problem.scenarios[0].T_amb
    r_sheaths   = [layers_list[i][-1].r_outer for i in range(len(all_placements))]
    k_soil_bg   = k_model.k_eff_bg(all_placements)

    set_seed(seed)

    # Build 128×5 model (ff=off always)
    solver_cfg = {
        "model": {
            "architecture": "mlp",
            "width":         p.width,
            "depth":         p.depth,
            "activation":    "tanh",
            "fourier_features": False,
            "fourier_mapping_size": 64,
            "fourier_scale": 1.0,
        },
        "normalization": {"normalize_coords": True},
    }
    base  = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModel(
        base, layers_list, all_placements,
        k_soil_bg, T_amb, Q_lins,
        problem.domain, normalize=True, Q_d=Q_d,
        enable_grad_Tbg=True,
    )
    init_output_bias(model.base, 0.0)

    pretrain_multicable(
        model, all_placements, problem.domain,
        layers_list, Q_lins, k_soil_bg, T_amb,
        device=device, normalize=True,
        n_per_cable=1000, n_bc=200, n_steps=500, lr=1e-3,
        Q_d=Q_d,
    )

    def k_fn_homog(xy: torch.Tensor) -> torch.Tensor:
        return torch.full((xy.shape[0], 1), k_soil_bg,
                          device=xy.device, dtype=xy.dtype)

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
            print_every=10**9,
            normalize=True,
            device=device,
            logger=None,
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
            "T_max_C":   float("nan"),
            "error_K":   float("nan"),
            "final_loss": float("nan"),
            "elapsed_s": time.time() - t0,
            "exception": str(exc),
            "model":     None,
        }

    with torch.no_grad():
        T_conds = eval_conductor_temps(
            model, all_placements, problem.domain, device, normalize=True)
        T_worst = max(T_conds)

    T_max_C = T_worst - 273.15
    error_K = T_worst - PAPER_T_MAX_PAC

    return {
        "T_max_C":    T_max_C,
        "error_K":    error_K,
        "final_loss": final_loss,
        "elapsed_s":  time.time() - t0,
        "exception":  None,
        "model":      model,
    }


def run_trial_with_timeout(
    p: TrialParams,
    shared: dict,
    device: torch.device,
    seed: int = 42,
    timeout_s: float = 1200.0,
) -> dict:
    result: dict = {}

    def _target():
        nonlocal result
        result = run_trial(p, shared, device, seed)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        result = {
            "T_max_C":    float("nan"),
            "error_K":    float("nan"),
            "final_loss": float("nan"),
            "elapsed_s":  timeout_s,
            "exception":  f"timeout after {timeout_s:.0f}s",
            "model":      None,
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Hyperparameter search for Kim 2024 Case B (deep 128×5 profile)")
    ap.add_argument("--trials",     type=int,   default=20,
                    help="Number of random trials (default: 20)")
    ap.add_argument("--seed",       type=int,   default=0,
                    help="Random seed for the search sampler (default: 0)")
    ap.add_argument("--trial-seed", type=int,   default=42,
                    help="Base PyTorch seed for trials (default: 42)")
    ap.add_argument("--n-seeds",    type=int,   default=5,
                    help="Seeds to try with --best (default: 5)")
    ap.add_argument("--best",       action="store_true",
                    help="Re-run BEST_KNOWN_C across multiple seeds, save model")
    ap.add_argument("--fast",       action="store_true",
                    help="Fast mode: cap adam≤2000, lbfgs≤300 for quick exploration")
    ap.add_argument("--timeout",    type=float, default=1200.0,
                    help="Max seconds per trial (default: 1200)")
    return ap.parse_args()


# Known best config for 128×5 — from fast search (seed=0, 20 trials):
#   rank1: T=70.18°C, err=-0.42K  lr=5e-4 wpde=0.5 wbc=200
#   key finding: wbc=200 (higher than 64×4's 100) helps deeper net
#   lbfgs_steps=1000 (full budget, was capped at 300 in fast mode)
BEST_KNOWN_C = TrialParams(
    lr=5e-4,
    adam_steps=2000,
    lbfgs_steps=1000,
    lbfgs_history=100,
    warmup_frac=0.50,
    w_pde=0.5,
    w_bc=200.0,
    n_interior=2000,
    n_boundary=500,
    oversample=1,
    frac_pac_bnd=0.50,
    layer_transition=0.10,
    pac_transition=0.20,
    width=128,
    depth=5,
    fourier_mapping_size=0,
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
    print("  Deep profile: 128×5 MLP tanh  |  FEM ref: %.1f °C" % FEM_REF_C)
    print("=" * 72)
    print("\n  PDE: ∇·(k(x,y)∇T) = 0  in soil (divergence form, full autograd)")
    print("  k(x,y): 3 soil layers + sigmoid PAC zone override")
    print("  Fixed: pac_transition=0.20  warmup_frac=0.50  oversample=1  ff=off")
    print("  Search: lr, adam_steps, lbfgs_steps, w_pde, w_bc")
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

    # ── Multi-seed --best ─────────────────────────────────────────────
    if args.best:
        seeds = list(range(args.trial_seed, args.trial_seed + args.n_seeds))
        print("  ── Multi-seed BEST_KNOWN_C run (%d seeds) ──" % len(seeds))
        p = BEST_KNOWN_C
        print("  Params:", p.short_str())
        print("  FEM ref: %.1f °C\n" % FEM_REF_C)
        print("  %-6s  %-8s  %-9s  %-12s  %-7s" % (
            "seed", "T_max°C", "Err[K]", "Loss", "t[s]"))
        print("  " + "-" * 50)
        best_result     = None
        best_model_state = None
        best_seed       = args.trial_seed
        for seed in seeds:
            r = run_trial(p, shared, device, seed=seed)
            marker = " ✓" if abs(r["error_K"]) <= 2.0 else ""
            print("  %-6d  %7.2f   %+7.2f K  %.4e   %5.0fs%s" % (
                seed, r["T_max_C"], r["error_K"], r["final_loss"],
                r["elapsed_s"], marker))
            if best_result is None or abs(r["error_K"]) < abs(best_result["error_K"]):
                best_result      = r
                best_model_state = r["model"].state_dict()
                best_seed        = seed
        # Save
        model_path = RESULTS_DIR / "model_best_128x5.pt"
        torch.save({
            "state_dict": best_model_state,
            "seed":       best_seed,
            "T_max_C":    best_result["T_max_C"],
            "error_K":    best_result["error_K"],
            "params":     dataclasses.asdict(p),
        }, model_path)
        print("\n  ── Best across %d seeds ──" % len(seeds))
        print("  seed=%d  T_max=%.2f °C  Error=%+.2f K  Loss=%.4e" % (
            best_seed, best_result["T_max_C"], best_result["error_K"],
            best_result["final_loss"]))
        print("  Model saved → %s" % model_path)
        return

    # ── Random search ────────────────────────────────────────────────
    rng      = random.Random(args.seed)
    n_trials = args.trials
    fast_mode = args.fast
    timeout_s = args.timeout

    if fast_mode:
        timeout_s = min(timeout_s, 900.0)
        print("  [FAST MODE] adam≤2000, lbfgs≤300  |  timeout=%ds\n" % int(timeout_s))

    results: list[tuple[TrialParams, dict]] = []

    print("  %-4s  %-8s  %-9s  %-10s  %-7s  %s" % (
        "#", "T_max°C", "Err[K]", "Loss", "t[s]", "params"))
    print("  " + "-" * 100)

    for idx in range(1, n_trials + 1):
        p = sample_trial(rng)
        if fast_mode:
            p = dataclasses.replace(
                p,
                adam_steps  = min(p.adam_steps,  2000),
                lbfgs_steps = min(p.lbfgs_steps, 300),
            )
        print("  [%2d/%d] training … " % (idx, n_trials), end="", flush=True)
        result = run_trial_with_timeout(p, shared, device,
                                        seed=args.trial_seed, timeout_s=timeout_s)
        results.append((p, result))

        if result["exception"]:
            print("ERROR: %s" % result["exception"])
        else:
            err_str = "%+.2f K" % result["error_K"]
            marker  = " ✓" if abs(result["error_K"]) <= 2.0 else ""
            print("T=%5.1f°C  err=%s  loss=%.3e  t=%3.0fs%s  | %s" % (
                result["T_max_C"], err_str,
                result["final_loss"], result["elapsed_s"],
                marker, p.short_str()))

    # ── Summary table ────────────────────────────────────────────────
    valid = [(p, r) for p, r in results if r["exception"] is None]
    if not valid:
        print("\n  No valid results.")
        return

    valid.sort(key=lambda x: abs(x[1]["error_K"]))

    print()
    print("=" * 72)
    print("  RESULTS — sorted by |Error vs FEM| (best first)")
    print("  (Deep: 128×5 | Fixed: ptr=0.20, wfrac=0.50, oversample=1, ff=off)")
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
    print("  ── BEST TRIAL (128×5) ──")
    print("  T_max = %.2f °C  |  Error = %+.2f K  |  Loss = %.4e" % (
        best_r["T_max_C"], best_r["error_K"], best_r["final_loss"]))
    print("  Params:", best_p.short_str())
    print()
    print("  Tip: copy best params into BEST_KNOWN_C and run --best --n-seeds 5")
    print("  to save model_best_128x5.pt and compare with 64×4.")
    print()
    print("  ── Comparison vs 64×4 ──")
    print("  FEM reference:  %.1f °C" % FEM_REF_C)
    print("  64×4 best:      70.19 °C  (Error = −0.41 K) [seed=0]")
    print("  128×5 best:     %.2f °C  (Error = %+.2f K)" % (
        best_r["T_max_C"], best_r["error_K"]))
    print("=" * 72)


if __name__ == "__main__":
    main()
