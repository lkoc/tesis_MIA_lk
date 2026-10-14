"""Run a light, FEM-grid evaluation of a saved PINN checkpoint without training or full audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.solvers.pinn.networks import build_model
from pinn_v2.solvers.pinn.problem import load_problem
from pinn_v2.evaluation.audit import evaluate_model


ROOT = Path(__file__).resolve().parent


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("M0", "M1", "M2"), default="M1")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", choices=("tanh", "silu"), default="tanh")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    before = checkpoint.stat()
    problem = load_problem(args.case)
    reference = load_reference_npz(args.reference, problem)
    model = build_model(problem, args.model, args.width, args.depth, args.activation).double().cpu()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    after = checkpoint.stat()
    checkpoint_stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    if not checkpoint_stable:
        raise RuntimeError("Checkpoint changed while loading; rerun the lightweight evaluation")
    model.load_state_dict(state)
    model.eval()

    report, arrays = evaluate_model(model, reference, "FEniCS level-2 FEM", include_audit=False)
    temperatures_finite = bool(np.isfinite(arrays["T"]).all())
    parameters_finite = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    result = dict(
        evaluated_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint),
        checkpoint_size_bytes=after.st_size,
        checkpoint_mtime_ns=after.st_mtime_ns,
        checkpoint_stable=True,
        model=dict(architecture=args.model, width=args.width, depth=args.depth, activation=args.activation),
        case=problem.name,
        physics_sha256=problem.physics_sha256,
        reference=str(args.reference.resolve()),
        full_audit_run=False,
        parameters_finite=parameters_finite,
        temperatures_finite=temperatures_finite,
        evaluation=report,
    )
    result["evaluation_metrics_finite"] = _finite_tree(report)

    output = args.output
    if output is None:
        output = checkpoint.parent / f"checkpoint_eval_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "checkpoint_stable": checkpoint_stable,
        "parameters_finite": parameters_finite,
        "temperatures_finite": temperatures_finite,
        "Tmax_pred_C": report.get("Tmax_pred_C"),
        "Tmax_ref_C": report.get("Tmax_ref_C"),
        "error_Tmax_abs_K": report.get("error_Tmax_abs_K"),
        "nrmse_pct": report.get("nrmse_pct"),
        "error_rise_pct": report.get("error_rise_pct"),
        "worst_region_rmse_rise_pct": max(report.get("region_rmse_rise_pct", {}).values(), default=None),
    }, allow_nan=True), flush=True)
    return 0 if checkpoint_stable and parameters_finite and temperatures_finite and result["evaluation_metrics_finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())