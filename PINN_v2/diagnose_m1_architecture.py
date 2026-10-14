"""Probe M1 loss gradients, then isolate width/depth effects on aras_flat."""
import argparse
import json
from pathlib import Path

import torch

from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.solvers.pinn.networks import build_model
from pinn_v2.solvers.pinn.objective import DEFAULT_WEIGHTS, Objective
from pinn_v2.solvers.pinn.problem import load_problem
from pinn_v2.solvers.pinn import trainer


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "pinn_v2/cases/v2/references/cables/aras_flat.json"
REFERENCE_PATH = ROOT / "results/hparam_optimization_20261003_adaptive/shared_references/aras_flat/fem_l2.npz"
OUTPUT = ROOT / "results/m1_architecture_diagnostics_20261008"
POINTS = {"n_interior": 256, "n_layer": 128, "n_face": 96, "n_boundary": 96}
SEED = 11
SAMPLING_SEED = 20272010
BASE_WEIGHTS = dict(DEFAULT_WEIGHTS)


def _device_name():
    return "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"


def _gradient_probe(name, checkpoint, width, depth, weights):
    device = torch.device(_device_name())
    problem = load_problem(CASE)
    model = build_model(problem, "M1", width, depth, "tanh").to(device=device, dtype=torch.float64)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    objective = Objective(model, seed=SAMPLING_SEED, weights=weights, **POINTS)
    _, parts, _ = objective()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    vectors = {}
    norms = {}
    weighted_norms = {}
    loss_values = {}
    for term, loss in parts.items():
        loss_values[term] = float(loss.detach().cpu())
        if loss.requires_grad:
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            vector_parts = [
                gradient.detach().reshape(-1) if gradient is not None else parameter.new_zeros(parameter.numel())
                for parameter, gradient in zip(parameters, gradients)
            ]
            vector = torch.cat(vector_parts)
        else:
            vector = torch.zeros(sum(parameter.numel() for parameter in parameters), device=device, dtype=torch.float64)
        vectors[term] = vector
        norm = torch.linalg.vector_norm(vector)
        norms[term] = float(norm.cpu())
        weighted_norms[term] = float((norm * weights[term]).cpu())

    terms = list(vectors)
    cosine = {}
    for first in terms:
        cosine[first] = {}
        for second in terms:
            denominator = torch.linalg.vector_norm(vectors[first]) * torch.linalg.vector_norm(vectors[second])
            value = torch.dot(vectors[first], vectors[second]) / denominator.clamp_min(1e-30)
            cosine[first][second] = float(value.cpu())

    del objective, model, state, vectors
    if device.type == "xpu":
        torch.xpu.empty_cache()
    return dict(
        checkpoint=str(checkpoint.relative_to(ROOT)),
        architecture="M1",
        width=width,
        depth=depth,
        parameters=sum(parameter.numel() for parameter in parameters),
        device=str(device),
        points=POINTS,
        weights=weights,
        loss_values=loss_values,
        gradient_norms_unweighted=norms,
        gradient_norms_weighted=weighted_norms,
        gradient_cosines_unweighted=cosine,
    )


def run_gradients():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    targets = (
        ("baseline", OUTPUT / "baseline_gradient_probe.json",
         ROOT / "results/pde_active_validation_20261007/M1/model.pt", 24, 3, BASE_WEIGHTS),
        ("trial6", OUTPUT / "trial6_gradient_probe.json",
         ROOT / "results/hparam_optimization_20261004_staged/M1/finalist/trial_006/aras_flat/seed_11/model.pt",
         24, 4, dict(pde=1., constitutive=1000., energy=10., interface_temperature=10.,
                     interface_flux=10., dirichlet=100., neumann=10., robin=10.)),
    )
    for name, path, checkpoint, width, depth, weights in targets:
        if path.exists():
            raise FileExistsError(path)
        result = _gradient_probe(name, checkpoint, width, depth, weights)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(name, result["gradient_norms_weighted"], flush=True)


def run_capacity_sweep():
    problem = load_problem(CASE)
    reference = load_reference_npz(REFERENCE_PATH, problem)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for width, depth in ((48, 3), (24, 5)):
        run_name = f"M1_w{width}_d{depth}"
        output = OUTPUT / run_name
        result = trainer.run(
            CASE,
            "M1",
            output,
            adam_steps=800,
            lbfgs_steps=0,
            lr=0.003,
            width=width,
            depth=depth,
            activation="tanh",
            weights=BASE_WEIGHTS,
            points=POINTS,
            init_seed=SEED,
            sampling_seed=SAMPLING_SEED,
            log_every=100,
            reference=reference,
            reference_label=f"FEniCSx level 2 ({problem.physics_sha256})",
            study="m1_architecture_diagnostics_20261008",
            full_audit=True,
        )
        evaluation = result.get("evaluation", {})
        row = dict(
            architecture="M1",
            width=width,
            depth=depth,
            parameters=result.get("parameters"),
            elapsed_s=result.get("elapsed_s"),
            status=result["status"],
            pde_weight=BASE_WEIGHTS["pde"],
            Tmax_error_K=evaluation.get("error_Tmax_abs_K"),
            nrmse_pct=evaluation.get("nrmse_pct"),
            error_rise_pct=evaluation.get("error_rise_pct"),
            balance_global_pct=evaluation.get("balance", {}).get("global_derived_pct"),
            interface_T_jump_max_K=max((row["T_jump_max_K"] for row in evaluation.get("interfaces", [])), default=None),
            interface_flux_jump_rms_pct=max((row["flux_jump_rms_pct"] for row in evaluation.get("interfaces", [])), default=None),
            accepted=evaluation.get("acceptance", {}).get("accepted"),
            output=str(output.relative_to(ROOT)),
        )
        results.append(row)
        print(row, flush=True)
    summary_path = OUTPUT / "capacity_sweep.json"
    summary_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gradients", "capacity", "all"), default="all")
    mode = parser.parse_args().mode
    if mode in ("gradients", "all"):
        run_gradients()
    if mode in ("capacity", "all"):
        run_capacity_sweep()


if __name__ == "__main__":
    main()