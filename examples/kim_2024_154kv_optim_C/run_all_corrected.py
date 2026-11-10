"""Master pipeline: run all three corrected PINN models sequentially.

Runs the complete corrected T_amb pipeline:
  Step 1: 64×4  small network  → results_corrected/model_best_64x4.pt
  Step 2: 128×5 dense network  → results_corrected/model_best_128x5.pt
  Step 3: 128×5 distilled      → results_corrected/model_best_128x5_distil.pt
  Step 4: Spatial comparison   → results_corrected/comparison_corrected.csv

T_amb correction: 290.15 K → 300.30 K (FEM surface temperature).
This eliminates the systematic ~2 K underestimate in the cable zone.

Usage::
    python examples/kim_2024_154kv_optim_C/run_all_corrected.py
    python examples/kim_2024_154kv_optim_C/run_all_corrected.py --n-seeds 8
    python examples/kim_2024_154kv_optim_C/run_all_corrected.py --skip-64x4
    python examples/kim_2024_154kv_optim_C/run_all_corrected.py --skip-128x5
    python examples/kim_2024_154kv_optim_C/run_all_corrected.py --only-compare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent

SCRIPT_64x4   = ROOT / "examples" / "kim_2024_154kv_optim_B" / "run_corrected_64x4.py"
SCRIPT_128x5  = HERE / "run_corrected_128x5.py"
SCRIPT_DISTIL = HERE / "run_corrected_distil.py"

MODEL_64x4      = ROOT / "examples" / "kim_2024_154kv_optim_B" / "results_corrected" / "model_best_64x4.pt"
MODEL_128x5     = HERE / "results_corrected" / "model_best_128x5.pt"
MODEL_DISTIL    = HERE / "results_corrected" / "model_best_128x5_distil.pt"

FEM_CSV_PATH    = HERE / "results_fem" / "fem_field_kim2024B.npz"


def run_step(script: Path, extra_args: list[str], step_name: str) -> bool:
    """Run a subprocess step and return True on success."""
    cmd = [sys.executable, str(script)] + extra_args
    print("\n" + "=" * 72)
    print("  STEP: %s" % step_name)
    print("  Cmd: %s" % " ".join(cmd))
    print("=" * 72 + "\n")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print("\n  !! Step FAILED (exit code %d, %.0fs)" % (proc.returncode, elapsed))
        return False
    print("\n  ✓ Step completed in %.0fs" % elapsed)
    return True


def print_summary():
    """Print a summary of corrected model results."""
    import torch
    print("\n" + "=" * 72)
    print("  CORRECTED T_amb PIPELINE — SUMMARY")
    print("  T_amb = 300.30 K (was 290.15 K, +10.15 K correction)")
    print("=" * 72)
    print("  %-25s  %-10s  %-10s" % ("Model", "T_max [°C]", "Err vs FEM"))
    print("  " + "-" * 50)
    FEM_REF_C = 70.6
    for label, path in [
        ("64×4  small",         MODEL_64x4),
        ("128×5 dense",         MODEL_128x5),
        ("128×5 distilled",     MODEL_DISTIL),
    ]:
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            print("  %-25s  %-10.2f  %+.2f K" % (
                label, ckpt["T_max_C"], ckpt["T_max_C"] - FEM_REF_C))
        else:
            print("  %-25s  %-10s  %s" % (label, "not found", "—"))
    print("  %-25s  %-10.1f  —" % ("FEM reference", FEM_REF_C))
    print()


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Run complete corrected T_amb pipeline for Kim 2024 Case B")
    ap.add_argument("--n-seeds",     type=int, default=5,
                    help="Seeds per model (default: 5)")
    ap.add_argument("--trial-seed",  type=int, default=42,
                    help="Base seed (default: 42)")
    ap.add_argument("--skip-64x4",   action="store_true",
                    help="Skip 64×4 training (use existing model)")
    ap.add_argument("--skip-128x5",  action="store_true",
                    help="Skip 128×5 training (use existing model)")
    ap.add_argument("--skip-distil", action="store_true",
                    help="Skip distillation (use existing model)")
    ap.add_argument("--only-compare", action="store_true",
                    help="Only run the comparison analysis (no training)")
    return ap.parse_args()


def main():
    args = _parse_args()

    print("=" * 72)
    print("  KIM 2024 CASE B — CORRECTED T_amb FULL PIPELINE")
    print("  T_amb correction: 290.15 K → 300.30 K (FEM surface temperature)")
    print("  Models: 64×4 small | 128×5 dense | 128×5 distilled")
    print("  Seeds per model: %d" % args.n_seeds)
    print("=" * 72)

    seed_args = ["--n-seeds", str(args.n_seeds), "--trial-seed", str(args.trial_seed)]

    t_start = time.time()

    # Step 1: 64×4 small network
    if not args.only_compare and not args.skip_64x4:
        ok = run_step(SCRIPT_64x4, seed_args, "64×4 small network (corrected T_amb)")
        if not ok:
            print("  Pipeline aborted at step 1 (64×4)")
            return 1
    elif args.skip_64x4 or args.only_compare:
        if MODEL_64x4.exists():
            print("  Skipping 64×4 (model exists: %s)" % MODEL_64x4)
        else:
            print("  WARNING: 64×4 model not found, distillation will fail!")

    # Step 2: 128×5 dense network
    if not args.only_compare and not args.skip_128x5:
        ok = run_step(SCRIPT_128x5, seed_args, "128×5 dense network (corrected T_amb)")
        if not ok:
            print("  Pipeline aborted at step 2 (128×5)")
            return 1
    elif args.skip_128x5 or args.only_compare:
        if MODEL_128x5.exists():
            print("  Skipping 128×5 (model exists: %s)" % MODEL_128x5)

    # Step 3: Distillation
    if not args.only_compare and not args.skip_distil:
        distil_args = ["--n-seeds", str(args.n_seeds),
                       "--trial-seed", str(args.trial_seed),
                       "--distil-steps", "200", "--distil-lr", "2e-4"]
        ok = run_step(SCRIPT_DISTIL, distil_args, "128×5 distillation (corrected T_amb)")
        if not ok:
            print("  Pipeline aborted at step 3 (distillation)")
            return 1

    # Step 4: Summary
    print_summary()

    # Step 5: Run comparison analysis if available
    analyze_script = HERE / "analyze_fem_vs_pinn.py"
    plot_script    = HERE / "results_fem" / "plot_zoom_comparison.py"
    if analyze_script.exists() and FEM_CSV_PATH.exists():
        ok = run_step(analyze_script, ["--corrected"], "Spatial comparison vs FEM (corrected)")
        if not ok:
            print("  Comparison analysis failed (non-fatal).")
    else:
        if not FEM_CSV_PATH.exists():
            print("  Note: FEM solution .npz not found at %s" % FEM_CSV_PATH)
    if plot_script.exists() and (HERE / "results_fem" / "fem_field_kim2024B.npz").exists():
        ok = run_step(plot_script, ["--corrected"], "Publication zoom figures (corrected)")
        if not ok:
            print("  Plot generation failed (non-fatal).")

    total_time = time.time() - t_start
    print("\n  Pipeline complete in %.0f min %.0f s" % (total_time // 60, total_time % 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
