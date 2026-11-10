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
    parser.add_argument("--case", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("M0", "M1", "M1_EXACT", "M2", "DeepXDE"), default="M1")
    parser.add_argument("--deepxde-model", choices=("PFNN", "FNN"), default="PFNN")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", choices=("tanh", "silu"), default="tanh")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-audit", action="store_true",
        help="Include PDE, boundary, interface, and energy-balance audits")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    before = checkpoint.stat()
    portable_pair = checkpoint.with_suffix('.json').is_file()
    if portable_pair:
        from pinn_v2.solvers.pinn.checkpoint import load_checkpoint
        model, specification, _ = load_checkpoint(checkpoint)
        problem = model.problem
        config = specification['config']
        args.model, args.width, args.depth, args.activation = (
            config['architecture'], config['width'], config['depth'], config['activation'])
    else:
        if args.case is None:
            parser.error('Legacy checkpoints require --case')
        problem = load_problem(args.case)
    reference = load_reference_npz(args.reference, problem) if args.reference is not None else None
    input_normalization = None
    if args.model == 'M1_EXACT':
        for metadata_path in (checkpoint.with_name('checkpoint_meta.json'), checkpoint.with_name('result.json')):
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                input_normalization = metadata.get('config', {}).get('input_normalization')
                if input_normalization is not None:
                    break
        if input_normalization is None:
            input_normalization = 'legacy_region_coordinates'
    if portable_pair:
        input_normalization = config.get('input_normalization')
    elif args.model == "DeepXDE":
        import deepxde as dde
        from pinn_v2.solvers.pinn.deepxde_solver import DeepXDEModel, _load_network_state, _network

        dde.config.set_default_float("float64")
        network = _network(dde, problem, args.deepxde_model, args.width, args.depth, args.activation)
        model = DeepXDEModel(problem, network, args.deepxde_model)
    else:
        model = build_model(problem, args.model, args.width, args.depth, args.activation,
            input_normalization=input_normalization).double().cpu()
    state = None if portable_pair else torch.load(checkpoint, map_location="cpu", weights_only=True)
    after = checkpoint.stat()
    checkpoint_stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    if not checkpoint_stable:
        raise RuntimeError("Checkpoint changed while loading; rerun the lightweight evaluation")
    if portable_pair:
        model.eval()
    elif args.model == "DeepXDE":
        _load_network_state(model.network, state)
        model.network.eval()
    else:
        model.load_state_dict(state)
        model.eval()

    report, arrays = evaluate_model(model, reference, "FEniCS level-2 FEM", include_audit=args.full_audit)
    temperatures_finite = bool(np.isfinite(arrays["T"]).all())
    parameters_finite = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    result = dict(
        evaluated_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint),
        checkpoint_size_bytes=after.st_size,
        checkpoint_mtime_ns=after.st_mtime_ns,
        checkpoint_stable=True,
        model=dict(architecture=args.model, width=args.width, depth=args.depth, activation=args.activation,
            **({'input_normalization': input_normalization} if args.model == 'M1_EXACT' else {}),
            **({"deepxde_model": args.deepxde_model} if args.model == "DeepXDE" else {})),
        case=problem.name,
        physics_sha256=problem.physics_sha256,
        reference=str(args.reference.resolve()) if args.reference is not None else None,
        full_audit_run=args.full_audit,
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