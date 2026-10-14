import json
from pathlib import Path

import torch

from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.solvers.pinn import trainer
from pinn_v2.solvers.pinn.problem import load_problem


root = Path(__file__).resolve().parent
case = root / "pinn_v2/cases/v2/references/cables/aras_flat.json"
reference_path = root / "results/hparam_optimization_20261003_adaptive/shared_references/aras_flat/fem_l2.npz"
output = root / "results/deepxde_offset_fix_20261008/DeepXDE_PFNN_seed11"
problem = load_problem(case)
reference = load_reference_npz(reference_path, problem)


def log_progress(row, _model):
    print(f"step={row['step']} loss={row['total']:.8g}", flush=True)


result = trainer.run(
    case,
    "DeepXDE",
    output,
    deepxde_model="PFNN",
    adam_steps=800,
    lbfgs_steps=0,
    lr=0.003,
    width=24,
    depth=3,
    activation="tanh",
    weights={
        "pde": 1.0,
        "constitutive": 0.0,
        "energy": 1.0,
        "interface_temperature": 100.0,
        "interface_flux": 10.0,
        "dirichlet": 100.0,
        "neumann": 10.0,
        "robin": 10.0,
    },
    points={"n_interior": 256, "n_layer": 128, "n_face": 96, "n_boundary": 96},
    init_seed=11,
    sampling_seed=20272010,
    log_every=100,
    reference=reference,
    reference_label=f"FEniCSx level 2 ({problem.physics_sha256})",
    study="deepxde_offset_fix_20261008",
    full_audit=True,
    on_log=log_progress,
)

state = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
evaluation = result.get("evaluation", {})
print(
    json.dumps(
        {
            "status": result["status"],
            "elapsed_s": result.get("elapsed_s"),
            "best_step": result.get("best_step"),
            "final_loss": result.get("final_loss"),
            "Tmax_pred_C": evaluation.get("Tmax_pred_C"),
            "Tmax_ref_C": evaluation.get("Tmax_ref_C"),
            "error_Tmax_abs_K": evaluation.get("error_Tmax_abs_K"),
            "nrmse_pct": evaluation.get("nrmse_pct"),
            "error_rise_pct": evaluation.get("error_rise_pct"),
            "balance": evaluation.get("balance"),
            "acceptance": evaluation.get("acceptance"),
            "temperature_offsets": state["temperature_offsets"].tolist(),
            "error": result.get("error"),
            "output": str(output),
        },
        allow_nan=True,
    ),
    flush=True,
)