"""Train one PINN run and write a complete, never-overwritten record.

Seeds are separate: init_seed fixes the network weights, sampling_seed fixes the collocation points.
Neither the exact solution nor any FEM field is read by the loss (fem_labels_used = 0); a reference
is used only after training, by evaluation. A diverged or crashed run is kept as failure.json.
"""
import copy
import numbers
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pinn_v2.evaluation.audit import evaluate_model
from pinn_v2.manifest.run_manifest import build_manifest, write_manifest, write_json_atomic
from pinn_v2.solvers.pinn.networks import build_model
from pinn_v2.solvers.pinn.objective import Objective
from pinn_v2.solvers.pinn.problem import load_problem

POINTS = dict(n_interior=512, n_layer=256, n_face=128, n_boundary=128)


def _resolve_device(requested):
    if requested not in ('auto', 'cpu', 'xpu'):
        raise ValueError("device must be 'auto', 'cpu', or 'xpu'")
    xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
    if requested == 'xpu' and not xpu_available:
        raise RuntimeError('XPU was requested, but PyTorch reports no available XPU device')
    if requested == 'auto':
        return torch.device('xpu' if xpu_available else 'cpu')
    return torch.device(requested)


def _row(step, phase, total, parts, lr):
    return dict(step=step, phase=phase, total=float(total.detach()), lr=lr, **{k: float(v.detach()) for k, v in parts.items()})


def _save_checkpoint(model, path):
    temporary = path.with_suffix('.pt.tmp')
    torch.save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, temporary)
    os.replace(temporary, path)


def _write_checkpoint_metadata(path, run_metadata, progress, status='running', failure=None):
    metadata = dict(run_metadata, schema_version=1, checkpoint_file=path.name,
        checkpoint_size_bytes=path.stat().st_size, checkpoint_mtime_ns=path.stat().st_mtime_ns,
        updated_utc=datetime.now(timezone.utc).isoformat(), status=status, progress=progress)
    if failure is not None:
        metadata['failure'] = failure
    write_json_atomic(path.with_name('checkpoint_meta.json'), metadata)


def fit(model, objective, adam_steps, lbfgs_steps, lr, log_every, lbfgs_iterations=20, on_log=None,
    stop_callback=None):
    """Adam with cosine decay, then L-BFGS; the best finite state is restored at the end.

    on_log(row, model) is an optional observer called after each logged row (the front-end uses it
    for live progress). It must not change the model; with on_log=None the behaviour is unchanged.
    """
    history = []
    best = dict(total=float('inf'), state=None, step=-1)

    def keep(total, step):
        value = float(total.detach())
        if np.isfinite(value) and value < best['total']:
            best.update(total=value, state=copy.deepcopy(model.state_dict()), step=step)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(adam_steps, 1), eta_min=lr * 1e-2)
    for step in range(adam_steps):
        optimizer.zero_grad()
        total, parts, _ = objective()
        if not torch.isfinite(total):
            raise FloatingPointError(f'Adam loss is not finite at step {step}')
        keep(total, step)
        total.backward()
        optimizer.step()
        schedule.step()
        objective.update_attention()
        if step % log_every == 0 or step == adam_steps - 1:
            history.append(_row(step, 'adam', total, parts, optimizer.param_groups[0]['lr']))
            if on_log is not None:
                on_log(history[-1], model)
            if stop_callback is not None and stop_callback(history[-1], model):
                break
    if lbfgs_steps:
        if best['state'] is not None:
            model.load_state_dict(best['state'])
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=lbfgs_iterations, history_size=50, line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            total, _, _ = objective()
            total.backward()
            return total
        for step in range(lbfgs_steps):
            try:
                lbfgs.step(closure)
            except (FloatingPointError, RuntimeError):
                break
            total, parts, _ = objective()
            if not torch.isfinite(total):
                break
            keep(total, adam_steps + step)
            if step % max(log_every // 20, 1) == 0 or step == lbfgs_steps - 1:
                history.append(_row(adam_steps + step, 'lbfgs', total, parts, 1.))
                if on_log is not None:
                    on_log(history[-1], model)
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    return history, best


def _arrays(output, name, **arrays):
    path = Path(output).joinpath(name)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)


def _tree_is_finite(value):
    if isinstance(value, dict):
        return all(_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_tree_is_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return not np.issubdtype(value.dtype, np.number) or bool(np.isfinite(value).all())
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def _evaluation_is_finite(report, arrays):
    return _tree_is_finite(report) and _tree_is_finite(arrays)


def run(source, architecture, output, adam_steps=3000, lbfgs_steps=0, lr=2e-3, width=32, depth=3, activation='tanh', weights=None,
        points=None, init_seed=0, sampling_seed=11, log_every=100, reference=None, command=None, study=None, attention=None, reference_label='supplied', on_log=None,
    device='auto', deepxde_model='PFNN', stop_callback=None, full_audit=True, initial_state=None):
    """One run. Returns the result dictionary; raises FileExistsError if output already exists."""
    if architecture == 'DeepXDE':
        if attention:
            raise ValueError('DeepXDE does not support the custom adaptive-attention losses')
        if device == 'xpu':
            raise ValueError('DeepXDE uses the CPU backend in this benchmark')
        from pinn_v2.solvers.pinn import deepxde_solver
        return deepxde_solver.run(source, output, internal_model=deepxde_model, adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps, lr=lr, width=width, depth=depth, activation=activation,
            weights=weights, points=points, init_seed=init_seed, sampling_seed=sampling_seed,
            log_every=log_every, reference=reference, reference_label=reference_label,
            stop_callback=stop_callback, full_audit=full_audit,
            command=command, study=study, on_log=on_log, initial_state=initial_state)
    selected_device = _resolve_device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    problem = load_problem(source)
    points = dict(POINTS, **(points or {}))
    config = dict(architecture=architecture, width=width, depth=depth, activation=activation, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
        lr=lr, weights=weights or {}, points=points, attention=attention)
    seeds = dict(init=init_seed, sampling=sampling_seed)
    history, status, error = [], 'completed', None
    result = dict(case=problem.name, config=config, seeds=seeds, status=status, physics_sha256=problem.physics_sha256, fem_labels_used=0,
        reference_used_in_loss=False, evaluation_finite=None, study=study)
    result['device'] = str(selected_device)
    try:
        torch.manual_seed(init_seed)
        if attention and lbfgs_steps:
            raise ValueError('L-BFGS needs a fixed loss; use lbfgs_steps=0 with adaptive attention weights')
        model = build_model(problem, architecture, width, depth, activation).to(device=selected_device, dtype=torch.float64)
        if initial_state is not None:
            model.load_state_dict(initial_state)
        checkpoint = output.joinpath('model.pt')
        checkpoint_metadata = dict(case=problem.name, architecture=architecture, model_kind=model.kind,
            physics_sha256=problem.physics_sha256, config=config, seeds=seeds, device=str(selected_device))
        last_checkpoint_progress = dict(phase='initialized', step=None, step_zero_based=True,
            optimizer_steps_completed=0, learning_rate=None, loss_total=None, loss_terms={})
        _save_checkpoint(model, checkpoint)
        _write_checkpoint_metadata(checkpoint, checkpoint_metadata, last_checkpoint_progress)

        def observe(row, current_model):
            nonlocal last_checkpoint_progress
            _save_checkpoint(current_model, checkpoint)
            last_checkpoint_progress = dict(phase=row['phase'], step=int(row['step']), step_zero_based=True,
                optimizer_steps_completed=int(row['step']) + 1, learning_rate=float(row['lr']),
                loss_total=float(row['total']),
                loss_terms={key: float(value) for key, value in row.items() if key not in ('phase', 'step', 'lr', 'total')})
            _write_checkpoint_metadata(checkpoint, checkpoint_metadata, last_checkpoint_progress)
            if on_log is not None:
                on_log(row, current_model)

        objective = Objective(model, seed=sampling_seed, weights=weights, attention=attention, **points)
        result['parameters'] = model.parameter_count()
        result['variant_kind'] = model.kind
        history, best = fit(model, objective, adam_steps, lbfgs_steps, lr, log_every,
            on_log=observe, stop_callback=stop_callback)
        if best['state'] is None:
            raise FloatingPointError('No finite state was reached')
        total, parts, region_pde = objective()
        result.update(best_step=best['step'], final_loss=float(total.detach()), final_parts={k: float(v.detach()) for k, v in parts.items()},
            final_region_pde={k: float(v.detach()) for k, v in region_pde.items()})
        if objective.attention is not None:
            result['attention_weights'] = objective.attention.summary()
        report, arrays = evaluate_model(model, reference, reference_label, include_audit=full_audit)
        result['evaluation'] = report
        result['evaluation_finite'] = _evaluation_is_finite(report, arrays)
        _arrays(output, 'evaluation.npz', **arrays)
        if not result['evaluation_finite']:
            raise FloatingPointError('Evaluation contains non-finite metrics or field values')
        _arrays(output, 'collocation.npz', **{f'interior_{i}': t.detach().cpu().numpy() for i, t in objective.interior.items()})
        _save_checkpoint(model, checkpoint)
        final_progress = dict(phase='finished', step=int(best['step']), step_zero_based=True,
            optimizer_steps_completed=int(best['step']) + 1, best_step=int(best['step']),
            best_loss=float(best['total']), final_loss=float(result['final_loss']),
            learning_rate=history[-1]['lr'] if history else None,
            loss_terms=result['final_parts'], evaluation_finite=True)
        _write_checkpoint_metadata(checkpoint, checkpoint_metadata, final_progress, status='completed')
    except Exception as exc:
        status = result['status'] = 'failed'
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        write_json_atomic(output.joinpath('failure.json'), dict(error=error, history=history, config=config, seeds=seeds,
            evaluation=result.get('evaluation'), evaluation_finite=result.get('evaluation_finite')))
        if 'checkpoint' in locals() and checkpoint.exists():
            _write_checkpoint_metadata(checkpoint, checkpoint_metadata, last_checkpoint_progress,
                status='failed', failure=dict(type=error['type'], message=error['message']))
    result.update(status=status, history=history, elapsed_s=time.perf_counter() - started, error=error)
    write_json_atomic(output.joinpath('result.json'), result)
    manifest = build_manifest('phase4_pinn_run', command or [sys.executable, '-m', 'pinn_v2.solvers.pinn.trainer'], seeds=seeds,
        extra=dict(case=problem.name, config=config, status=status, physics_sha256=problem.physics_sha256, fem_labels_used=0, study=study,
            device=str(selected_device)))
    write_manifest(output, manifest)
    return result
