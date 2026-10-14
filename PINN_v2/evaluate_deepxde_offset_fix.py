import json
from pathlib import Path

import numpy as np

from pinn_v2.evaluation.audit import evaluate_model
from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.solvers.pinn import deepxde_solver
from pinn_v2.solvers.pinn.problem import load_problem


root = Path(__file__).resolve().parent
baseline_dir = root / "results/pde_active_validation_20261007/DeepXDE_PFNN"
offset_fix_dir = root / "results/deepxde_offset_fix_20261008/DeepXDE_PFNN_seed11"
problem = load_problem(root / "pinn_v2/cases/v2/references/cables/aras_flat.json")
reference_path = root / "results/hparam_optimization_20261003_adaptive/shared_references/aras_flat/fem_l2.npz"
reference = load_reference_npz(reference_path, problem)

comparison = {}
for label, run_dir in (("baseline_without_offset", baseline_dir), ("trained_with_offset", offset_fix_dir)):
    model = deepxde_solver.load_model(run_dir, problem)
    report, arrays = evaluate_model(model, reference, "FEniCSx level 2", include_audit=True)
    training = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    posthoc = {
        "case": problem.name,
        "physics_sha256": problem.physics_sha256,
        "checkpoint": "model.pt",
        "training_elapsed_s": training["elapsed_s"],
        "retrained_after_coordinate_fix": False,
        "evaluation_only_coordinate_fix": "Clamp inactive-region radial/log features to finite values at cable centers.",
        "evaluation": report,
        "all_model_parameters_finite": all(
            bool(np.isfinite(value.detach().cpu().numpy()).all()) for value in model.state_dict().values()
        ),
        "temperature_offsets_K": (model.network.temperature_offsets.detach().cpu() * problem.T_scale).tolist(),
    }
    name = "posthoc_baseline_after_coordinate_fix.json" if label.startswith("baseline") else "posthoc_evaluation_after_coordinate_fix.json"
    (run_dir / name).write_text(json.dumps(posthoc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if label.startswith("trained"):
        np.savez_compressed(run_dir / "evaluation_after_coordinate_fix.npz", **arrays)
    comparison[label] = {
        "Tmax_pred_C": report["Tmax_pred_C"],
        "Tmax_ref_C": report["Tmax_ref_C"],
        "error_Tmax_abs_K": report["error_Tmax_abs_K"],
        "nrmse_pct": report["nrmse_pct"],
        "error_rise_pct": report["error_rise_pct"],
        "balance_global_pct": report["balance"]["global_derived_pct"],
        "accepted": report["acceptance"]["accepted"],
    }
print(json.dumps(comparison, indent=2, allow_nan=False), flush=True)