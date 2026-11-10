"""Probe M1 loss gradients, then isolate width/depth effects on aras_flat."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.solvers.pinn.networks import build_model
from pinn_v2.solvers.pinn.objective import DEFAULT_WEIGHTS, Objective
from pinn_v2.solvers.pinn.problem import load_problem
from pinn_v2.solvers.pinn import trainer


def inspect_checkpoint(checkpoint, device='cpu', repeats=3):
    """Audit native model conditioning and synchronized objective cost without changing its weights."""
    from pinn_v2.solvers.pinn.checkpoint import load_checkpoint
    model, specification, _ = load_checkpoint(checkpoint, device=device)
    config = specification['config']
    objective = Objective(model, seed=909091, sampling='uniform', weights=config.get('weights'), **config['points'])

    def synchronize():
        if str(device).startswith('xpu'):
            torch.xpu.synchronize()

    timings = []
    for iteration in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        synchronize()
        started = time.perf_counter()
        total, _, _ = objective()
        forward = time.perf_counter()
        total.backward()
        synchronize()
        finished = time.perf_counter()
        if iteration:
            timings.append(dict(forward_s=forward - started, backward_s=finished - forward,
                total_s=finished - started))
    total, parts, regional = objective()
    trace = getattr(model, 'trace_coefficients', None)
    regions = {}
    for name, value in regional.items():
        item = dict(pde=float(value.detach()))
        if trace is not None and trace.numel():
            gradient = torch.autograd.grad(value, trace, retain_graph=True, allow_unused=True)[0]
            if gradient is not None:
                item.update(trace_temperature_gradient=float(gradient[:, 0].norm()),
                    trace_flux_gradient=float(gradient[:, 1].norm()),
                    constant_temperature_gradient=float(gradient[:, 0, 0].norm()),
                    constant_flux_gradient=float(gradient[:, 1, 0].norm()))
        regions[name] = item
    groups = {}
    for name, parameter in model.named_parameters():
        group = 'traces' if 'trace' in name else 'offsets' if 'offsets' in name else 'networks'
        groups.setdefault(group, []).append(float(parameter.grad.norm()) if parameter.grad is not None else 0.)
    parameter = next(model.parameters())
    return dict(checkpoint=str(Path(checkpoint).resolve()), architecture=config['architecture'],
        input_normalization=config.get('input_normalization'), dtype=str(parameter.dtype), device=str(device),
        physics_sha256=model.problem.physics_sha256, points=config['points'], audit_seed=909091,
        pde_loss=float(total.detach()), loss_parts={name: float(value.detach()) for name, value in parts.items()},
        regional=regions, scales={region.name: model.problem.scales(region) for region in model.problem.regions},
        gradient_norms={group: float(np.linalg.norm(values)) for group, values in groups.items()},
        trace_max_abs=None if trace is None or not trace.numel() else float(trace.detach().abs().max()),
        trace_constant_T_K=[] if trace is None else (trace[:, 0, 0].detach().cpu().numpy() * model.problem.T_scale).tolist(),
        timing=dict(repeats=repeats, median_s=float(np.median([row['total_s'] for row in timings])),
            p90_s=float(np.percentile([row['total_s'] for row in timings], 90)), samples=timings),
        source_free=not any(model.problem.region_power.values()),
        hypotheses_not_proven=['trace conditioning', 'soil bubble attenuation', 'training budget'])


def run_exact_screen(source, reference_path, output, device='cpu', steps=400, arms=None,
    architecture='M1_EXACT', lbfgs_steps=0):
    """Single-factor screening with independent full audits; a completed run is not acceptance."""
    from pinn_v2.manifest.run_manifest import write_json_atomic
    problem = load_problem(source)
    reference = load_reference_npz(reference_path, problem) if reference_path else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    choices = dict(control={}, float32=dict(dtype='float32'), rba=dict(attention='rba'), sa=dict(attention='sa'),
        paired=dict(sampling='paired'))
    if architecture == 'DeepXDE_EXACT':
        choices.pop('rba')
        choices.pop('sa')
    elif architecture != 'M1_EXACT':
        raise ValueError('Exact screening requires M1_EXACT or DeepXDE_EXACT')
    selected = list(arms) if arms is not None else list(choices)
    if not selected or set(selected) - set(choices):
        raise ValueError('Requested screen arm is incompatible with the problem geometry')
    rows = []
    for name in selected:
        changes = choices[name]
        settings = dict(adam_steps=steps, lbfgs_steps=lbfgs_steps, width=32, depth=3, activation='tanh',
            lr=.003, points=dict(trainer.POINTS), init_seed=0, sampling_seed=11,
            input_normalization='unit_box_v1', log_every=max(steps // 4, 1),
            validation_every=max(steps // 4, 1), device=device, full_audit=True)
        settings.update(changes)
        if settings.get('attention') and lbfgs_steps:
            raise ValueError('Dynamic attention screening requires zero L-BFGS steps')
        result = trainer.run(source, architecture, output / name, reference=reference,
            reference_label='same-physics FEM, evaluation only' if reference_path else 'manufactured',
            study='exact_single_factor_screen', **settings)
        evaluation = result.get('evaluation', {})
        row = dict(arm=name, settings_delta=changes, status=result['status'],
            elapsed_s=result['elapsed_s'], final_loss=result.get('final_loss'),
            validation_loss=result.get('checkpoint_selection', {}).get('final_validation_loss'),
            evaluation=evaluation, error=result.get('error'))
        rows.append(row)
        write_json_atomic(output / f'{name}_summary.json', row)
        print(name, result['status'], evaluation.get('Tmax_pred_C'), evaluation.get('nrmse_pct'),
            evaluation.get('balance', {}).get('global_derived_pct'), result['elapsed_s'], flush=True)
    write_json_atomic(output / 'screen.json', dict(case=problem.name, physics_sha256=problem.physics_sha256,
        device=device, steps=steps, architecture=architecture, lbfgs_outer_steps=lbfgs_steps,
        screening_only=True, fem_labels_used=0, rows=rows))
    return rows


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
    parser.add_argument("--mode", choices=("gradients", "capacity", "all", "audit", "screen"), default="all")
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'xpu'), default='cpu')
    parser.add_argument('--case', default='aras_flat')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--arms', nargs='+')
    parser.add_argument('--architecture', choices=('M1_EXACT', 'DeepXDE_EXACT'), default='M1_EXACT')
    parser.add_argument('--lbfgs-steps', type=int, default=0)
    args = parser.parse_args()
    mode = args.mode
    if mode == 'screen':
        if args.output is None:
            parser.error('--output must be a new directory for screen')
        torch.set_num_threads(args.threads)
        run_exact_screen(args.case, args.reference, args.output, args.device, args.steps, args.arms,
            args.architecture, args.lbfgs_steps)
    if mode == 'audit':
        if args.checkpoint is None:
            parser.error('--checkpoint is required for audit')
        report = inspect_checkpoint(args.checkpoint, args.device)
        if args.output:
            from pinn_v2.manifest.run_manifest import write_json_atomic
            write_json_atomic(args.output, report)
        print(json.dumps(report, indent=2, allow_nan=False))
    if mode in ("gradients", "all"):
        run_gradients()
    if mode in ("capacity", "all"):
        run_capacity_sweep()


if __name__ == "__main__":
    main()