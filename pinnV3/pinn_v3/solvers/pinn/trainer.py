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

from pinn_v3.evaluation.audit import evaluate_model
from pinn_v3.manifest.run_manifest import atomic_replace, build_manifest, write_manifest, write_json_atomic, replace_json_atomic
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.objective import DEFAULT_WEIGHTS, FOCUS_CRITERIA, Objective
from pinn_v3.solvers.pinn.problem import load_problem
from pinn_v3.solvers.pinn import checkpoint as portable

POINTS = dict(n_interior=512, n_layer=256, n_face=128, n_boundary=128)


class AccuracyStop:
    DEFAULTS = dict(error_Tmax_abs_K=.01, nrmse_pct=.1, balance_pct=.5, every=100, after_phase='adam')

    def __init__(self, settings, reference=None, reference_label='supplied'):
        if set(settings) - set(self.DEFAULTS):
            raise ValueError('Unknown accuracy stopping setting')
        self.settings = dict(self.DEFAULTS, **settings)
        numeric = [self.settings[name] for name in ('error_Tmax_abs_K', 'nrmse_pct', 'balance_pct', 'every')]
        if any(not np.isfinite(value) or value <= 0 for value in numeric):
            raise ValueError('Accuracy stopping thresholds and cadence must be positive and finite')
        if int(self.settings['every']) != self.settings['every']:
            raise ValueError('Accuracy stopping cadence must be an integer')
        if self.settings['after_phase'] not in ('adam', 'lbfgs', 'any'):
            raise ValueError("accuracy_stop after_phase must be 'adam', 'lbfgs' or 'any'")
        self.reference, self.reference_label = reference, reference_label
        self.checks = []
        self.acceptance_checks = []
        self.thresholds = {name: self.settings[name] for name in ('error_Tmax_abs_K', 'nrmse_pct', 'balance_pct')}
        self.relaxations = []
        self.triggered = False
        self.snapshot = None

    def check(self, row, model, force=False):
        if self.triggered:
            return True
        if self.settings['after_phase'] == 'lbfgs' and row['phase'] != 'lbfgs':
            return False
        step = int(row['step'])
        if self.checks and self.checks[-1]['step'] == step:
            return False
        if self.checks and not force and step - self.checks[-1]['step'] < self.settings['every']:
            return False
        report, _ = evaluate_model(model, self.reference, self.reference_label, include_audit=True)
        values = dict(error_Tmax_abs_K=report.get('error_Tmax_abs_K'), nrmse_pct=report.get('nrmse_pct'),
            balance_pct=report.get('balance', {}).get('global_derived_pct'))
        gates = {name: value is not None and np.isfinite(value) and value < self.thresholds[name]
            for name, value in values.items()}
        self.triggered = all(gates.values())
        self.checks.append(dict(step=step, phase=row['phase'], values=values,
            gates={name: bool(value) for name, value in gates.items()}, met=bool(self.triggered)))
        if self.triggered:
            self.snapshot = portable.cpu_state(model.state_dict())
        return self.triggered

    def relax(self, factor, step):
        self.thresholds = {name: value * factor for name, value in self.thresholds.items()}
        self.relaxations.append(dict(step=int(step), factor=float(factor), thresholds=dict(self.thresholds)))

    def accepted(self, model, step):
        report, _ = evaluate_model(model, self.reference, self.reference_label, include_audit=True)
        accepted = bool((report.get('acceptance') or {}).get('accepted'))
        self.acceptance_checks.append(dict(step=int(step), accepted=accepted,
            nrmse_pct=report.get('nrmse_pct'), error_Tmax_abs_K=report.get('error_Tmax_abs_K'),
            balance_pct=(report.get('balance') or {}).get('global_derived_pct')))
        return accepted

    def summary(self):
        return dict(enabled=True, settings=self.settings, triggered=bool(self.triggered), checks=self.checks,
            acceptance_checks=self.acceptance_checks, thresholds=self.thresholds, relaxations=self.relaxations,
            reference_label=self.reference_label, comparison='strict_less_than_all_three',
            reference_used_for_stopping=True, reference_used_in_loss=False)


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
    _save_state_dict(model.state_dict(), path)


def _save_state_dict(state, path):
    temporary = path.with_suffix('.pt.tmp')
    torch.save({name: tensor.detach().cpu() for name, tensor in state.items()}, temporary)
    atomic_replace(temporary, path)


def _write_checkpoint_metadata(path, run_metadata, progress, status='running', failure=None):
    metadata = dict(run_metadata, schema_version=1, checkpoint_file=path.name,
        checkpoint_size_bytes=path.stat().st_size, checkpoint_mtime_ns=path.stat().st_mtime_ns,
        updated_utc=datetime.now(timezone.utc).isoformat(), status=status, progress=progress)
    if failure is not None:
        metadata['failure'] = failure
    replace_json_atomic(path.with_name('checkpoint_meta.json'), metadata)


def fit(model, objective, adam_steps, lbfgs_steps, lr, log_every, lbfgs_iterations=20, on_log=None,
    stop_callback=None, on_checkpoint=None, resample_every=0, resume_state=None,
    accuracy_stop=None, extra_lbfgs_steps=0, point_schedule_stages=0, point_schedule_start=0.5,
    lazy_point_growth=False, cycle_lbfgs_chunk=0, cycle_extension=None):
    """Select finite post-update states at log boundaries and preserve the latest resumable state."""
    history = copy.deepcopy(resume_state['history']) if resume_state else []
    best = copy.deepcopy(resume_state['best']) if resume_state else dict(total=float('inf'), state=None, step=-1)
    stopped = False
    if int(lbfgs_iterations) != lbfgs_iterations or lbfgs_iterations < 1:
        raise ValueError('lbfgs_iterations must be a positive integer')
    if extra_lbfgs_steps < 0 or int(extra_lbfgs_steps) != extra_lbfgs_steps:
        raise ValueError('extra_lbfgs_steps must be a nonnegative integer')
    if (int(point_schedule_stages) != point_schedule_stages or point_schedule_stages < 0 or
            point_schedule_stages == 1 or point_schedule_stages > adam_steps):
        raise ValueError('point_schedule_stages must be zero or an integer from 2 through adam_steps')
    if not np.isfinite(point_schedule_start) or not 0 < point_schedule_start <= 1:
        raise ValueError('point_schedule_start must be finite and in (0, 1]')
    if point_schedule_stages and not hasattr(objective, 'set_point_fraction'):
        raise ValueError('Progressive point schedules require an objective with set_point_fraction')

    def keep(total, step):
        value = float(total.detach())
        if np.isfinite(value) and value < best['total']:
            best.update(total=value, state=portable.cpu_state(model.state_dict()), step=step)

    def notify(row, optimizer, schedule, phase, next_step):
        if accuracy_stop is not None and not (cycle_lbfgs_chunk and phase == 'lbfgs'):
            accuracy_stop.check(row, model, force=phase == 'lbfgs')
        if hasattr(objective, 'optimized_pde'):
            row['optimized_pde'] = float(objective.optimized_pde)
        best['training'] = dict(model=portable.cpu_state(model.state_dict()),
            optimizer=portable.cpu_state(optimizer.state_dict()),
            scheduler=portable.cpu_state(schedule.state_dict()) if schedule is not None else None,
            phase=phase, next_step=next_step, completed_updates=int(row['step']) + 1,
            device_type=next(model.parameters()).device.type,
            adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
            best=portable.cpu_state({key: best[key] for key in ('total', 'state', 'step')}),
            history=copy.deepcopy(history), rng=portable.cpu_state(portable.capture_rng()),
            objective=portable.objective_state(objective) if isinstance(objective, Objective) else None)
        if accuracy_stop is not None:
            best['training'].update(accuracy_stop=accuracy_stop.summary(), extra_lbfgs_steps=extra_lbfgs_steps,
                accuracy_reference=torch.as_tensor(accuracy_stop.reference).cpu()
                    if getattr(accuracy_stop, 'reference', None) is not None else None)
        if on_checkpoint is not None:
            on_checkpoint(row, model, best)
        if on_log is not None:
            on_log(row, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(adam_steps, 1), eta_min=lr * 1e-2)
    adam_start = 0
    if resume_state is not None:
        if resume_state['device_type'] != next(model.parameters()).device.type:
            raise ValueError('Exact resume requires the original device type; use an explicit warm start to migrate')
        if (resume_state['adam_steps'], resume_state['lbfgs_steps']) != (adam_steps, lbfgs_steps):
            raise ValueError('Exact resume requires the original optimizer horizons')
        model.load_state_dict(resume_state['model'])
        if isinstance(objective, Objective):
            portable.restore_objective(objective, resume_state['objective'])
        portable.restore_rng(resume_state['rng'])
        if resume_state['phase'] == 'adam':
            optimizer.load_state_dict(resume_state['optimizer'])
            schedule.load_state_dict(resume_state['scheduler'])
            adam_start = resume_state['next_step']
        else:
            adam_start = adam_steps
        best['training'] = portable.cpu_state(resume_state)
        if accuracy_stop is not None and resume_state.get('accuracy_stop'):
            accuracy_stop.checks = copy.deepcopy(resume_state['accuracy_stop']['checks'])
            accuracy_stop.triggered = resume_state['accuracy_stop']['triggered']
            accuracy_stop.reference_label = resume_state['accuracy_stop']['reference_label']
            if resume_state.get('extra_lbfgs_steps', 0) != extra_lbfgs_steps:
                raise ValueError('Exact resume requires the original extra L-BFGS budget')
            if accuracy_stop.triggered:
                best['stopped_state'] = portable.cpu_state(model.state_dict())
                best['stopped_step'] = int(history[-1]['step'])
                return history, best
    def synchronize_device():
        if next(model.parameters()).device.type == 'xpu':
            torch.xpu.synchronize()

    if cycle_lbfgs_chunk:
        if resume_state is not None or accuracy_stop is None:
            raise ValueError('Cycled training needs accuracy_stop and does not support exact resume')
        if int(cycle_lbfgs_chunk) != cycle_lbfgs_chunk or cycle_lbfgs_chunk < 1:
            raise ValueError('cycle_lbfgs_chunk must be a positive integer')
        extension = dict(steps=0, times=0, factor=1.) if cycle_extension is None else dict(cycle_extension)
        if int(extension['steps']) != extension['steps'] or extension['steps'] < 0 or int(extension['times']) != extension['times'] or extension['times'] < 0:
            raise ValueError('cycle_extension steps and times must be nonnegative integers')
        if extension['times'] and (extension['steps'] < 1 or not extension['factor'] > 1):
            raise ValueError('cycle_extension needs positive steps and a factor above 1')
        adam_left, lbfgs_left, step_index, accepted = adam_steps, lbfgs_steps, 0, False
        adam_seconds = lbfgs_seconds = 0.0
        extensions_used = 0
        lbfgs_failure = None

        def lbfgs_segment(count):
            nonlocal lbfgs_left, step_index, lbfgs_seconds, lbfgs_failure
            segment_started = time.perf_counter()
            lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=lbfgs_iterations, history_size=50, line_search_fn='strong_wolfe')

            def closure():
                lbfgs.zero_grad()
                total, _, _ = objective()
                if not torch.isfinite(total):
                    raise FloatingPointError('L-BFGS objective became non-finite')
                total.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    raise FloatingPointError('L-BFGS gradient became non-finite')
                return total
            for _ in range(count):
                step_state = portable.cpu_state(model.state_dict())
                try:
                    lbfgs.step(closure)
                except (FloatingPointError, RuntimeError) as exc:
                    model.load_state_dict(step_state)
                    lbfgs_left = 0
                    lbfgs_failure = f'{type(exc).__name__}: {exc}'
                    best['cycle_lbfgs_failure'] = lbfgs_failure
                    break
                total, parts, _ = objective()
                if not torch.isfinite(total):
                    model.load_state_dict(step_state)
                    lbfgs_left = 0
                    lbfgs_failure = 'FloatingPointError: L-BFGS post-step objective became non-finite'
                    best['cycle_lbfgs_failure'] = lbfgs_failure
                    break
                lbfgs_left -= 1
                keep(total, step_index)
                history.append(_row(step_index, 'lbfgs', total, parts, 1.))
                notify(history[-1], lbfgs, None, 'lbfgs', step_index + 1)
                step_index += 1
            synchronize_device()
            lbfgs_seconds += time.perf_counter() - segment_started
            if lbfgs_failure is not None:
                return False
            return accuracy_stop.accepted(model, step_index - 1)

        while not accepted and lbfgs_failure is None and (adam_left > 0 or extensions_used < extension['times']):
            if adam_left == 0:
                extensions_used += 1
                adam_left = extension['steps']
                accuracy_stop.relax(extension['factor'], step_index)
            accuracy_stop.triggered = False
            segment_started = time.perf_counter()
            while adam_left > 0 and not accuracy_stop.triggered:
                optimizer.zero_grad()
                total, parts, _ = objective()
                if not torch.isfinite(total):
                    raise FloatingPointError(f'Adam loss is not finite at step {step_index}')
                total.backward()
                optimizer.step()
                schedule.step()
                objective.update_attention()
                adam_left -= 1
                if step_index % log_every == 0 or adam_left == 0:
                    total, parts, _ = objective()
                    keep(total, step_index)
                    history.append(_row(step_index, 'adam', total, parts, optimizer.param_groups[0]['lr']))
                    notify(history[-1], optimizer, schedule, 'adam', step_index + 1)
                step_index += 1
            synchronize_device()
            adam_seconds += time.perf_counter() - segment_started
            if not accuracy_stop.triggered or lbfgs_left == 0:
                continue
            accepted = lbfgs_segment(min(cycle_lbfgs_chunk, lbfgs_left))
        if extensions_used and extensions_used == extension['times'] and lbfgs_failure is None:
            while not accepted and lbfgs_left > 0 and lbfgs_failure is None:
                accepted = lbfgs_segment(min(cycle_lbfgs_chunk, lbfgs_left))
        if accepted:
            best['stopped_state'] = portable.cpu_state(model.state_dict())
            best['stopped_step'] = step_index - 1
        best['phase_elapsed_s'] = dict(adam=adam_seconds, lbfgs=lbfgs_seconds)
        return history, best

    phase_elapsed = dict(adam=0.0, lbfgs=0.0)
    synchronize_device()
    adam_started = time.perf_counter()
    active_point_stage = None
    stage_interval = max((adam_steps + point_schedule_stages - 1) // point_schedule_stages, 1) if point_schedule_stages else 1
    for step in range(adam_start, adam_steps):
        if resample_every and step > 0 and step % resample_every == 0:
            objective.resample()
            best.update(total=float('inf'), state=None, step=-1)
        if point_schedule_stages:
            point_stage = min(step // stage_interval, point_schedule_stages - 1)
            if point_stage != active_point_stage:
                fraction = point_schedule_start + (1 - point_schedule_start) * point_stage / (point_schedule_stages - 1)
                if lazy_point_growth:
                    objective.grow_interior(fraction, point_stage)
                    if active_point_stage is not None:
                        best.update(total=float('inf'), state=None, step=-1)
                else:
                    objective.set_point_fraction(fraction)
                active_point_stage = point_stage
        optimizer.zero_grad()
        total, parts, _ = objective()
        if not torch.isfinite(total):
            raise FloatingPointError(f'Adam loss is not finite at step {step}')
        total.backward()
        optimizer.step()
        schedule.step()
        objective.update_attention()
        if step % log_every == 0 or step == adam_steps - 1:
            total, parts, _ = objective()
            keep(total, step)
            history.append(_row(step, 'adam', total, parts, optimizer.param_groups[0]['lr']))
            notify(history[-1], optimizer, schedule, 'adam', step + 1)
            if (accuracy_stop is not None and accuracy_stop.triggered) or (
                    stop_callback is not None and stop_callback(history[-1], model)):
                best['stopped_state'] = portable.cpu_state(model.state_dict())
                best['stopped_step'] = step
                stopped = True
                break
    synchronize_device()
    phase_elapsed['adam'] = time.perf_counter() - adam_started
    if lbfgs_steps and not stopped:
        synchronize_device()
        lbfgs_started = time.perf_counter()
        if hasattr(objective, 'set_point_fraction'):
            objective.set_point_fraction(1.)
        continuing_lbfgs = resume_state is not None and resume_state['phase'] == 'lbfgs'
        if best['state'] is not None and not continuing_lbfgs:
            model.load_state_dict(best['state'])
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=lbfgs_iterations, history_size=50, line_search_fn='strong_wolfe')
        lbfgs_start = 0
        if continuing_lbfgs:
            lbfgs.load_state_dict(resume_state['optimizer'])
            lbfgs_start = resume_state['next_step']

        def closure():
            lbfgs.zero_grad()
            total, _, _ = objective()
            total.backward()
            return total
        budget = lbfgs_steps + (extra_lbfgs_steps if accuracy_stop is not None else 0)
        for step in range(lbfgs_start, budget):
            try:
                lbfgs.step(closure)
            except (FloatingPointError, RuntimeError):
                break
            total, parts, _ = objective()
            if not torch.isfinite(total):
                break
            if accuracy_stop is not None or step % max(log_every // 20, 1) == 0 or step == lbfgs_steps - 1:
                keep(total, adam_steps + step)
                history.append(_row(adam_steps + step, 'lbfgs', total, parts, 1.))
                notify(history[-1], lbfgs, None, 'lbfgs', step + 1)
                if (accuracy_stop is not None and accuracy_stop.triggered) or (
                    stop_callback is not None and stop_callback(history[-1], model)):
                    best['stopped_state'] = portable.cpu_state(model.state_dict())
                    best['stopped_step'] = adam_steps + step
                    break
                synchronize_device()
                phase_elapsed['lbfgs'] = time.perf_counter() - lbfgs_started
    if 'stopped_state' in best:
        model.load_state_dict(best['stopped_state'])
    elif accuracy_stop is not None and best.get('training'):
        model.load_state_dict(best['training']['model'])
    elif best['state'] is not None:
        model.load_state_dict(best['state'])
    best['phase_elapsed_s'] = phase_elapsed
    return history, best


def _arrays(output, name, **arrays):
    path = Path(output).joinpath(name)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)


def _save_collocation(path, objective):
    path = Path(path)
    temporary = path.with_name('collocation.tmp.npz')
    arrays = {f'interior_{region_id}': points.detach().cpu().numpy()
        for region_id, points in objective.interior.items()}
    arrays.update({f'interface_{index}': entry[1].detach().cpu().numpy()
        for index, entry in enumerate(objective.faces)})
    arrays.update({f'interface_right_{index}': entry[3].detach().cpu().numpy()
        for index, entry in enumerate(objective.faces)})
    arrays.update({f'boundary_{index}': xy.detach().cpu().numpy()
        for index, (_, xy, _) in enumerate(objective.pieces)})
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    atomic_replace(temporary, path)


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
    device='auto', deepxde_model='PFNN', stop_callback=None, full_audit=True, initial_state=None, validation_every=500,
    sampling='mixed', resample_every=0, deepxde_sampling='mixed', resume_from=None, dtype=portable.DEFAULT_DTYPE, input_normalization=None,
    trace_parameterization='standard', soil_envelope_gain=1., accuracy_stop=None, extra_lbfgs_steps=5,
    lbfgs_iterations=20, point_schedule_stages=0, point_schedule_start=0.5,
    warm_start_source=None, exact_energy=False, focused=False, focus_criterion='residual', focus_surfaces=True,
    rbf_initialization=None, cycle_lbfgs_chunk=0, cycle_extension=None):
    """One run. Returns the result dictionary; raises FileExistsError if output already exists."""
    from pinn_v3.solvers.pinn.networks import EXACT_CORRECTORS, GENERIC_EXACT_ARCHITECTURES
    exact_backends = dict(DeepXDE_EXACT='EXACT', DeepXDE_PFNN_EXACT='EXACT_PFNN', DeepXDE_FNN_EXACT='EXACT_FNN')
    if architecture == 'DeepXDE' or architecture in exact_backends:
        if rbf_initialization is not None:
            raise NotImplementedError('RBF-FD initialization is not implemented for the DeepXDE dispatcher')
        exact_backend = architecture in exact_backends
        if resume_from is not None:
            raise ValueError('DeepXDE checkpoints support inference/warm start, not exact optimizer resume')
        if not exact_backend and (input_normalization is not None or trace_parameterization != 'standard' or soil_envelope_gain != 1.):
            raise ValueError('Portable resume, configurable dtype/encoding and exact trace policies require a native PINN backend')
        if (not exact_backend and sampling != 'mixed') or resample_every:
            raise ValueError('DeepXDE does not support configurable PINN v2 sampling or resampling')
        if attention:
            raise ValueError('DeepXDE does not support the custom adaptive-attention losses')
        if exact_energy or focused:
            raise ValueError('exact_energy and focused resampling require a native PINN backend')
        if device not in ('auto', 'cpu', 'xpu'):
            raise ValueError("device must be 'auto', 'cpu', or 'xpu'")
        deepxde_device = 'cpu' if device == 'auto' else str(_resolve_device(device))
        from pinn_v3.solvers.pinn import deepxde_solver
        return deepxde_solver.run(source, output, internal_model=exact_backends.get(architecture, deepxde_model), adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps, lr=lr, width=width, depth=depth, activation=activation,
            weights=weights, points=points, init_seed=init_seed, sampling_seed=sampling_seed,
            log_every=log_every, reference=reference, reference_label=reference_label,
            stop_callback=stop_callback, full_audit=full_audit,
            command=command, study=study, on_log=on_log, initial_state=initial_state,
            deepxde_sampling=deepxde_sampling, device=deepxde_device, dtype=dtype,
            input_normalization=input_normalization, sampling=sampling,
            trace_parameterization=trace_parameterization, soil_envelope_gain=soil_envelope_gain,
            accuracy_stop=accuracy_stop, extra_lbfgs_steps=extra_lbfgs_steps,
            warm_start_source=warm_start_source)
    loss_policy = None
    if architecture == 'M1_EXACT' or architecture in EXACT_CORRECTORS or architecture in GENERIC_EXACT_ARCHITECTURES:
        supplied_weights = weights or {}
        unknown_weights = set(supplied_weights) - set(DEFAULT_WEIGHTS)
        if unknown_weights:
            raise ValueError(f'Invalid objective weights: {sorted(unknown_weights)}')
        requested_weights = dict(DEFAULT_WEIGHTS, **supplied_weights)
        if any(not np.isfinite(value) or value < 0 for value in requested_weights.values()):
            raise ValueError('Objective weights must be finite and non-negative')
        if requested_weights['pde'] <= 0:
            raise ValueError('Exact PINNs require a positive PDE weight')
        kept = ('pde', 'energy') if exact_energy else ('pde',)
        weights = {name: requested_weights[name] if name in kept else 0. for name in DEFAULT_WEIGHTS}
        loss_policy = 'pde_energy_exact_transmission' if exact_energy else 'pde_only_exact_transmission'
    elif exact_energy:
        raise ValueError('exact_energy applies only to exact PINN architectures')
    selected_device = _resolve_device(device)
    validation_every = int(validation_every)
    if validation_every < 1:
        raise ValueError('validation_every must be a positive integer')
    resample_every = int(resample_every)
    if resample_every < 0:
        raise ValueError('resample_every must be zero or a positive integer')
    if focused and not resample_every:
        raise ValueError('focused resampling requires resample_every > 0')
    if focus_criterion not in FOCUS_CRITERIA:
        raise ValueError(f'focus_criterion must be one of {FOCUS_CRITERIA}')
    if int(lbfgs_iterations) != lbfgs_iterations or lbfgs_iterations < 1:
        raise ValueError('lbfgs_iterations must be a positive integer')
    if (int(point_schedule_stages) != point_schedule_stages or point_schedule_stages < 0 or
            point_schedule_stages == 1 or point_schedule_stages > adam_steps):
        raise ValueError('point_schedule_stages must be zero or an integer from 2 through adam_steps')
    if not np.isfinite(point_schedule_start) or not 0 < point_schedule_start <= 1:
        raise ValueError('point_schedule_start must be finite and in (0, 1]')
    if dtype not in portable.DTYPES:
        raise ValueError('dtype must be float32 or float64')
    rbf_settings = None
    if rbf_initialization is not None:
        defaults = dict(fraction=0.05, mode='extrapolate', pretrain_steps=100,
            pretrain_lr=1e-3, surface_count=24, stencil_size=21, growth_stages=10,
            source='rbf', fem_node_multiplier=1.0, fem_transfer='pretrain', fem_gradient_weight=1.0, fem_ridge=0.0,
            fem_surfaces=architecture in ('M0', 'M0R', 'M1', 'M1_EXACT', 'M2', 'CUSP', 'CUSPR', 'DCSNN', 'FBPINN'),
            rbf_transfer='readout_gls', rbf_interior_per_region=100, rbf_anchor_fraction=1.0)
        if set(rbf_initialization) - set(defaults):
            raise ValueError('Unknown RBF-FD initialization settings')
        rbf_settings = dict(defaults, **rbf_initialization)
        if rbf_settings['mode'] not in ('extrapolate', 'incremental'):
            raise ValueError('RBF-FD mode must be extrapolate or incremental')
        if rbf_settings['source'] not in ('rbf', 'fem', 'rbf_readout'):
            raise ValueError('RBF-FD source must be rbf, fem or rbf_readout')
        if rbf_settings['rbf_transfer'] not in ('readout', 'readout_gls'):
            raise ValueError('RBF-FD readout transfer must be readout or readout_gls')
        if not 0 < rbf_settings['rbf_anchor_fraction'] <= 1.0:
            raise ValueError('RBF-FD anchor fraction must be in (0, 1]')
        if rbf_settings['fem_transfer'] not in ('pretrain', 'readout', 'readout_gls'):
            raise ValueError('FEM transfer must be pretrain, readout or readout_gls')
        if not isinstance(rbf_settings['fem_surfaces'], bool):
            raise ValueError('fem_surfaces must be a boolean')
        if not np.isfinite(rbf_settings['fem_gradient_weight']) or rbf_settings['fem_gradient_weight'] <= 0:
            raise ValueError('FEM gradient weight must be positive and finite')
        if not np.isfinite(rbf_settings['fem_ridge']) or rbf_settings['fem_ridge'] < 0:
            raise ValueError('FEM ridge must be non-negative and finite')
        if not 0 < rbf_settings['fraction'] <= 1.0:
            raise ValueError('RBF-FD fraction must be in (0, 1]')
        if rbf_settings['mode'] == 'incremental':
            if attention or resample_every or point_schedule_stages:
                raise ValueError('Incremental RBF-FD mode requires no attention, resampling, or other point schedule')
            stages = rbf_settings['growth_stages']
            if int(stages) != stages or stages < 2 or stages > adam_steps:
                raise ValueError('RBF-FD growth_stages must be an integer from 2 through adam_steps')
    validation_seed = (int(sampling_seed) + 1_000_003) % (2 ** 32)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    problem = load_problem(source)
    if sampling == 'annulus_log_v1' and problem.name != 'annulus':
        raise ValueError('annulus_log_v1 sampling is supported only by the concentric annulus problem')
    points = dict(POINTS, **(points or {}))
    config = dict(architecture=architecture, width=width, depth=depth, activation=activation, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
        lbfgs_iterations=int(lbfgs_iterations), lr=lr, weights=weights or {}, points=points, attention=attention, sampling=sampling,
        resample_every=resample_every, validation_every=validation_every,
        validation_seed=validation_seed, dtype=dtype, checkpoint_policy='post_update_logged_v2',
        sampling_version='paired_surfaces_v1' if sampling == 'paired' else
            'annulus_log_v1' if sampling == 'annulus_log_v1' else 'balanced_cables_v2', log_every=log_every,
        trace_parameterization=trace_parameterization, soil_envelope_gain=soil_envelope_gain,
        focus_criterion=focus_criterion, focus_surfaces=focus_surfaces)
    if loss_policy is not None:
        config['loss_policy'] = loss_policy
        config['input_normalization'] = 'unit_box_v1'
    if focused:
        config['resampling_policy'] = f'{focus_criterion}_focused_v1'
    if point_schedule_stages:
        config.update(point_schedule_stages=int(point_schedule_stages),
            point_schedule_start=float(point_schedule_start))
    if rbf_settings is not None:
        config['rbf_initialization'] = rbf_settings
    if input_normalization is not None:
        config['input_normalization'] = input_normalization
    if initial_state is not None:
        config['initialization_policy'] = 'weights_only_warm_start'
        config['warm_start_source'] = str(warm_start_source) if warm_start_source is not None else None
    monitor = AccuracyStop(accuracy_stop, reference, reference_label) if accuracy_stop is not None else None
    if monitor is not None:
        config.update(accuracy_stop=monitor.settings, extra_lbfgs_steps=extra_lbfgs_steps)
        if cycle_lbfgs_chunk:
            config['cycle_lbfgs_chunk'] = int(cycle_lbfgs_chunk)
        if cycle_extension:
            config['cycle_extension'] = dict(cycle_extension)
        if reference is None and problem.reference(problem.evaluation()[0]) is None:
            raise ValueError('Accuracy stopping requires an analytic or supplied reference')
    resume_state = None
    resumed_specification = None
    if resume_from is not None:
        if initial_state is not None:
            raise ValueError('Choose exact resume or a weights-only warm start, not both')
        _, resumed_specification, bundle = portable.load_checkpoint(resume_from, selection='latest')
        if not bundle.get('resume_available'):
            raise ValueError('The checkpoint has no training state; only a weights-only warm start is possible')
        if config != resumed_specification['config'] or problem.physics_sha256 != resumed_specification['physics_sha256']:
            raise ValueError('Exact resume requires identical physics, model, dtype and training configuration')
        resume_state = bundle['training']
        if resumed_specification['runtime']['torch'] != str(torch.__version__):
            raise ValueError('Exact resume requires the original PyTorch version')
    output.mkdir(parents=True, exist_ok=False)
    seeds = dict(init=init_seed, sampling=sampling_seed, validation=validation_seed)
    history, status, error = [], 'completed', None
    result = dict(case=problem.name, config=config, seeds=seeds, status=status, physics_sha256=problem.physics_sha256, fem_labels_used=0,
        reference_used_in_loss=False, evaluation_finite=None, study=study)
    result['device'] = str(selected_device)
    try:
        torch.manual_seed(init_seed)
        if attention and lbfgs_steps:
            raise ValueError('L-BFGS needs a fixed loss; use lbfgs_steps=0 with adaptive attention weights')
        model = build_model(problem, architecture, width, depth, activation,
            input_normalization=config.get('input_normalization'), trace_parameterization=trace_parameterization,
            soil_envelope_gain=soil_envelope_gain).to(device=selected_device, dtype=portable.DTYPES[dtype])
        if initial_state is not None:
            model.load_state_dict(initial_state)
        checkpoint = output.joinpath('model.pt')
        specification = resumed_specification or portable.specification_for(problem, config, seeds)
        portable.write_specification(checkpoint, specification)
        latest_checkpoint = output.joinpath('latest.pt')
        best_train_checkpoint = output.joinpath('best_train_loss.pt')
        best_validation_checkpoint = output.joinpath('best_validation.pt')
        checkpoint_metadata = dict(case=problem.name, architecture=architecture, model_kind=model.kind,
            physics_sha256=problem.physics_sha256, config=config, seeds=seeds, device=str(selected_device))
        best_train = dict(loss=float('inf'), step=None)
        last_train_resampling_count = 0
        best_validation = dict(loss=float('inf'), step=None)
        best_validation_model = None
        last_validation_step = None
        last_checkpoint_progress = dict(phase='initialized', step=None, step_zero_based=True,
            optimizer_steps_completed=0, learning_rate=None, loss_total=None, loss_terms={})
        portable.save_checkpoint(checkpoint, model, specification)
        _save_checkpoint(model, latest_checkpoint)
        _write_checkpoint_metadata(checkpoint, checkpoint_metadata, last_checkpoint_progress)

        incremental_rbf = rbf_settings is not None and rbf_settings['mode'] == 'incremental'
        objective = Objective(model, seed=sampling_seed, sampling=sampling, weights=weights, attention=attention,
            exact_energy=exact_energy, focused=focused, focus_criterion=focus_criterion,
            focus_surfaces=focus_surfaces,
            initial_fraction=rbf_settings['fraction'] if incremental_rbf else None, **points)
        validation_objective = Objective(model, seed=validation_seed, weights=weights, exact_energy=exact_energy, **points)
        if resume_state is not None and 'validation' in resume_state:
            portable.restore_objective(validation_objective, resume_state['validation'])
            best_validation.update(resume_state.get('best_validation', best_validation))
            best_validation_model = resume_state.get('best_validation_model')
            if best_validation_model is not None:
                _save_state_dict(best_validation_model, best_validation_checkpoint)
            last_validation_step = resume_state.get('last_validation_step')
        if rbf_settings is not None and resume_state is None:
            if initial_state is not None:
                raise ValueError('Choose RBF-FD pretraining or a weights-only warm start, not both')
            if not hasattr(problem, 'definition'):
                raise ValueError('RBF-FD initialization requires a canonical V3 problem')
            if rbf_settings['source'] == 'fem':
                from pinn_v3.solvers.pinn.fem_initializer import fem_initialization
                result['rbf_initialization'] = fem_initialization(model, problem, objective, rbf_settings, output)
            if rbf_settings['source'] == 'rbf_readout':
                from pinn_v3.solvers.pinn.rbf_readout import rbf_readout_initialization
                result['rbf_initialization'] = rbf_readout_initialization(model, problem, objective, rbf_settings, sampling_seed)
        if rbf_settings is not None and resume_state is None and rbf_settings['source'] == 'rbf':
            from pinn_v3.solvers.pinn.rbf_fd import pretrain_pinn, solve_regions
            rng = np.random.default_rng(sampling_seed)
            anchors = {}
            full_points = {}
            for region in problem.regions:
                xy = objective.interior[region.id].detach().cpu().numpy()
                full_points[region.id] = xy
                count = len(xy) if incremental_rbf else max(1, int(np.ceil(len(xy) * rbf_settings['fraction'])))
                indices = rng.choice(len(xy), count, replace=False)
                anchors[region.id] = xy[indices]
            rbf_started = time.perf_counter()
            rbf_result = solve_regions(problem.definition, surface_count=rbf_settings['surface_count'],
                stencil_size=rbf_settings['stencil_size'], seed=sampling_seed, interior_points=anchors)
            solve_elapsed = time.perf_counter() - rbf_started
            _arrays(output, 'rbf_initialization.npz', xy=rbf_result['xy'], labels=rbf_result['labels'],
                temperature=rbf_result['temperature'],
                anchor_indices=np.concatenate([rbf_result['anchors'][region.id] for region in problem.regions]))
            report = pretrain_pinn(model, rbf_result, steps=rbf_settings['pretrain_steps'],
                learning_rate=rbf_settings['pretrain_lr'], target_points=None if incremental_rbf else full_points)
            report.update(anchor_count=sum(len(xy) for xy in anchors.values()),
                solver_nodes=len(rbf_result['xy']), solve_elapsed_s=solve_elapsed,
                scaled_residual_inf=rbf_result['scaled_residual_inf'], mode=rbf_settings['mode'],
                temperature_min=float(np.min(rbf_result['temperature'])),
                temperature_max=float(np.max(rbf_result['temperature'])))
            result['rbf_initialization'] = report
        _save_collocation(output.joinpath('collocation.npz'), objective)

        def observe(row, current_model, best):
            nonlocal last_checkpoint_progress, last_validation_step, last_train_resampling_count, best_validation_model
            _save_checkpoint(current_model, latest_checkpoint)
            if objective.resampling_count != last_train_resampling_count:
                best_train.update(loss=float('inf'), step=None)
                last_train_resampling_count = objective.resampling_count
                _save_collocation(output.joinpath('collocation.npz'), objective)
            if best['state'] is not None and best['total'] < best_train['loss']:
                _save_state_dict(best['state'], best_train_checkpoint)
                best_train.update(loss=float(best['total']), step=int(best['step']))

            validation_loss = None
            validation_parts = None
            if last_validation_step is None or int(row['step']) - last_validation_step >= validation_every:
                validation_total, validation_terms, _ = validation_objective()
                validation_loss = float(validation_total.detach())
                validation_parts = {key: float(value.detach()) for key, value in validation_terms.items()}
                row['validation_loss'] = validation_loss
                last_validation_step = int(row['step'])
                if np.isfinite(validation_loss) and validation_loss < best_validation['loss']:
                    _save_checkpoint(current_model, best_validation_checkpoint)
                    best_validation.update(loss=validation_loss, step=int(row['step']))
                    best_validation_model = portable.cpu_state(current_model.state_dict())

            training = best['training']
            training.update(validation=portable.objective_state(validation_objective),
                best_validation=dict(best_validation), best_validation_model=best_validation_model,
                last_validation_step=last_validation_step)
            training['history'][-1] = dict(row)
            portable.save_checkpoint(checkpoint, current_model, specification, training)

            last_checkpoint_progress = dict(phase=row['phase'], step=int(row['step']), step_zero_based=True,
                optimizer_steps_completed=int(row['step']) + 1, learning_rate=float(row['lr']),
                loss_total=float(row['total']),
                loss_terms={key: float(value) for key, value in row.items() if key not in ('phase', 'step', 'lr', 'total', 'validation_loss')},
                best_train_loss=dict(best_train), best_validation=dict(best_validation),
                train_loss_scope='latest_fixed_collocation_cloud' if resample_every else 'full_run_fixed_collocation',
                validation_loss=validation_loss, validation_parts=validation_parts,
                checkpoint_files=dict(latest=latest_checkpoint.name,
                    best_train_loss=best_train_checkpoint.name if best_train_checkpoint.exists() else None,
                    best_validation=best_validation_checkpoint.name if best_validation_checkpoint.exists() else None))
            _write_checkpoint_metadata(checkpoint, checkpoint_metadata, last_checkpoint_progress)

        result['parameters'] = model.parameter_count()
        result['variant_kind'] = model.kind
        history, best = fit(model, objective, adam_steps, lbfgs_steps, lr, log_every,
            lbfgs_iterations=lbfgs_iterations,
            on_log=on_log, stop_callback=stop_callback, on_checkpoint=observe,
            resample_every=resample_every, resume_state=resume_state,
            accuracy_stop=monitor, extra_lbfgs_steps=extra_lbfgs_steps,
            point_schedule_stages=rbf_settings['growth_stages'] if incremental_rbf else point_schedule_stages,
            point_schedule_start=rbf_settings['fraction'] if incremental_rbf else point_schedule_start,
            lazy_point_growth=incremental_rbf, cycle_lbfgs_chunk=cycle_lbfgs_chunk, cycle_extension=cycle_extension)
        result['training_elapsed_s'] = best.get('phase_elapsed_s', {})
        if best['state'] is None:
            raise FloatingPointError('No finite state was reached')
        if monitor is not None:
            monitor.check(history[-1], model, force=True)
            completed_lbfgs = sum(row['phase'] == 'lbfgs' for row in history)
            result['accuracy_stop'] = dict(monitor.summary(),
                termination_reason=('lbfgs_non_finite' if best.get('cycle_lbfgs_failure') else
                    'accuracy_targets_met' if monitor.triggered else 'budget_exhausted'),
                lbfgs_failure=best.get('cycle_lbfgs_failure'),
                lbfgs_outer_calls=completed_lbfgs,
                extra_lbfgs_outer_calls=max(completed_lbfgs - lbfgs_steps, 0),
                selected_step=int(history[-1]['step']), selection_policy='stopping_or_final_state')
        _save_state_dict(best['state'], best_train_checkpoint)
        best_train.update(loss=float(best['total']), step=int(best['step']))
        validation_total, validation_terms, _ = validation_objective()
        final_validation_loss = float(validation_total.detach())
        if not np.isfinite(final_validation_loss):
            final_validation_loss = None
        elif final_validation_loss < best_validation['loss']:
            _save_checkpoint(model, best_validation_checkpoint)
            best_validation.update(loss=final_validation_loss, step=int(best['step']))
            best_validation_model = portable.cpu_state(model.state_dict())
        total, parts, region_pde = objective()
        result.update(best_step=best['step'], final_loss=float(total.detach()), final_parts={k: float(v.detach()) for k, v in parts.items()},
            final_region_pde={k: float(v.detach()) for k, v in region_pde.items()},
            checkpoint_selection=dict(best_train_loss=best_train, best_validation=best_validation,
                final_validation_loss=final_validation_loss, validation_seed=validation_seed,
                validation_every=validation_every,
                best_train_loss_scope='latest_fixed_collocation_cloud' if resample_every else 'full_run_fixed_collocation'),
            resampling_count=objective.resampling_count)
        if objective.attention is not None:
            result['attention_weights'] = objective.attention.summary()
        report, arrays = evaluate_model(model, reference, reference_label, include_audit=full_audit)
        result['evaluation'] = report
        result['evaluation_finite'] = _evaluation_is_finite(report, arrays)
        _arrays(output, 'evaluation.npz', **arrays)
        if not result['evaluation_finite']:
            raise FloatingPointError('Evaluation contains non-finite metrics or field values')
        final_step = int(history[-1]['step']) if history else int(best['step'])
        best['training'].update(best_validation=dict(best_validation), best_validation_model=best_validation_model)
        portable.save_checkpoint(checkpoint, model, specification, training=best.get('training'))
        final_progress = dict(phase='finished', step=final_step, step_zero_based=True,
            optimizer_steps_completed=final_step + 1, best_step=int(best['step']),
            best_loss=float(best['total']), final_loss=float(result['final_loss']),
            learning_rate=history[-1]['lr'] if history else None,
            loss_terms=result['final_parts'], evaluation_finite=True,
            best_train_loss=dict(best_train), best_validation=dict(best_validation),
            train_loss_scope='latest_fixed_collocation_cloud' if resample_every else 'full_run_fixed_collocation',
            final_validation_loss=final_validation_loss,
            checkpoint_files=dict(latest=latest_checkpoint.name, best_train_loss=best_train_checkpoint.name,
                best_validation=best_validation_checkpoint.name if best_validation_checkpoint.exists() else None))
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
    manifest = build_manifest('phase4_pinn_run', command or [sys.executable, '-m', 'pinn_v3.solvers.pinn.trainer'], seeds=seeds,
        extra=dict(case=problem.name, config=config, status=status, physics_sha256=problem.physics_sha256, fem_labels_used=0, study=study,
            device=str(selected_device)))
    write_manifest(output, manifest)
    return result


def resume_training(checkpoint, output, device='cpu', on_log=None, stop_callback=None,
    reference=None, full_audit=True):
    model, specification, bundle = portable.load_checkpoint(checkpoint, device=device, selection='latest')
    if not bundle.get('resume_available'):
        raise ValueError('This checkpoint supports warm start only, not exact resume')
    reference_label = 'supplied'
    if bundle['training'].get('accuracy_reference') is not None:
        saved_reference = bundle['training']['accuracy_reference'].cpu().numpy()
        if reference is not None and not np.array_equal(reference, saved_reference):
            raise ValueError('Exact resume requires the original accuracy stopping reference')
        reference = saved_reference
        reference_label = bundle['training']['accuracy_stop']['reference_label']
    config = specification['config']
    keys = ('adam_steps', 'lbfgs_steps', 'lbfgs_iterations', 'lr', 'width', 'depth', 'activation', 'weights',
        'points', 'attention', 'sampling', 'resample_every', 'validation_every', 'dtype',
        'input_normalization', 'log_every', 'trace_parameterization', 'soil_envelope_gain',
        'accuracy_stop', 'extra_lbfgs_steps', 'warm_start_source', 'rbf_initialization',
        'focus_criterion', 'focus_surfaces', 'point_schedule_stages', 'point_schedule_start')
    settings = {key: config[key] for key in keys if key in config}
    settings['focused'] = 'resampling_policy' in config
    return run(model.problem, config['architecture'], output, **settings,
        init_seed=specification['seeds']['init'], sampling_seed=specification['seeds']['sampling'],
        resume_from=checkpoint, device=device, on_log=on_log, stop_callback=stop_callback,
        reference=reference, reference_label=reference_label, full_audit=full_audit)


def continue_training(checkpoint, output, adam_steps, lbfgs_steps=0, device='cpu',
    reference=None, full_audit=True):
    """Continue from saved weights with fresh optimizers and a new step budget."""
    model, specification, _ = portable.load_checkpoint(checkpoint, device=device, selection='inference')
    config = specification['config']
    keys = ('lbfgs_iterations', 'lr', 'width', 'depth', 'activation', 'weights', 'points',
        'attention', 'sampling', 'resample_every', 'validation_every', 'dtype', 'input_normalization',
        'log_every', 'trace_parameterization', 'soil_envelope_gain', 'focus_criterion', 'focus_surfaces')
    settings = {key: config[key] for key in keys if key in config}
    settings['focused'] = 'resampling_policy' in config
    return run(model.problem, config['architecture'], output, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
        initial_state=model.state_dict(), warm_start_source=str(checkpoint),
        init_seed=specification['seeds']['init'], sampling_seed=specification['seeds']['sampling'],
        device=device, reference=reference, full_audit=full_audit, **settings)
