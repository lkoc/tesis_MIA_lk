"""Portable, versioned problem/model pairs and strict native PINN reconstruction."""
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from pinn_v3.manifest.run_manifest import atomic_replace, write_json_atomic
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.problem import load_problem

FORMAT = 'pinn-v2-checkpoint-2'
DEFAULT_DTYPE = 'float32'
DTYPES = dict(float32=torch.float32, float64=torch.float64)


def fingerprint(specification):
    encoded = json.dumps(specification, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(item) for item in value)
    return copy.deepcopy(value)


def specification_for(problem, config, seeds):
    resolved = dict(config)
    resolved.setdefault('dtype', DEFAULT_DTYPE)
    return dict(format=FORMAT, schema_version=2, problem=copy.deepcopy(problem.doc), config=resolved,
        seeds=dict(seeds), physics_sha256=problem.physics_sha256,
        model_version=1, flux_convention='q=-k*grad(T); outward normal',
        runtime=dict(torch=str(torch.__version__)),
        fourier_order=getattr(problem, 'fourier_order', 8))


def write_specification(path, specification):
    path = Path(path).with_suffix('.json')
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != specification:
            raise ValueError('The immutable model specification does not match this checkpoint')
    else:
        write_json_atomic(path, specification)
    return path


def save_checkpoint(path, model, specification, training=None):
    path = Path(path)
    write_specification(path, specification)
    bundle = dict(format=FORMAT, schema_version=2, specification_sha256=fingerprint(specification),
        model=cpu_state(model.state_dict()), training=cpu_state(training),
        resume_available=training is not None and 'optimizer' in training)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(bundle, temporary)
    atomic_replace(temporary, path)


def capture_rng():
    numpy_state = np.random.get_state()
    state = dict(python=random.getstate(), torch=torch.get_rng_state(),
        numpy=(numpy_state[0], torch.as_tensor(numpy_state[1].astype(np.int64)),
            numpy_state[2], numpy_state[3], numpy_state[4]))
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        state['xpu'] = torch.xpu.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], numpy_state[1].cpu().numpy().astype(np.uint32),
        numpy_state[2], numpy_state[3], numpy_state[4]))
    if 'xpu' in state and hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.set_rng_state_all([item.cpu() for item in state['xpu']])


def objective_state(objective):
    attention = objective.attention
    return cpu_state(dict(interior=getattr(objective, '_full_interior', objective.interior),
        sources=getattr(objective, '_full_sources', objective.sources),
        point_fraction=getattr(objective, 'point_fraction', 1.),
        resampling_count=objective.resampling_count,
        attention=None if attention is None else dict(kind=attention.kind, rate=attention.rate,
            decay=attention.decay, state=attention.state)))


def restore_objective(objective, state):
    reference = next(objective.model.parameters())
    objective._full_interior = {key: value.to(reference).detach().clone().requires_grad_(True)
        for key, value in state['interior'].items()}
    objective._full_sources = {key: value.to(reference).detach().clone() for key, value in state['sources'].items()}
    objective.resampling_count = state['resampling_count']
    if (objective.attention is None) != (state['attention'] is None):
        raise ValueError('Incompatible adaptive attention in the saved training state')
    if objective.attention is not None:
        attention = state['attention']
        if objective.attention.kind != attention['kind']:
            raise ValueError('Incompatible attention kind')
        objective.attention.rate, objective.attention.decay = attention['rate'], attention['decay']
        objective.attention.state = {key: value.to(reference).clone() for key, value in attention['state'].items()}
    objective.set_point_fraction(state.get('point_fraction', 1.))
    objective.residuals = {}


def model_weights(payload, selection='inference'):
    if payload.get('format') != FORMAT:
        return payload
    if selection == 'latest':
        training = payload.get('training')
        if training is None:
            raise ValueError('This pair has no training continuation state')
        return training['model']
    if selection != 'inference':
        raise ValueError('selection must be inference or latest')
    return payload['model']


def load_checkpoint(path, device='cpu', selection='inference'):
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Unsupported checkpoint format')
    specification = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if specification.get('format') != FORMAT or specification.get('schema_version') != 2:
        raise ValueError('Unsupported portable model schema')
    if specification.get('model_version') != 1:
        raise ValueError('Unsupported model implementation version')
    if payload.get('schema_version') != 2 or fingerprint(specification) != payload['specification_sha256']:
        raise ValueError('Checkpoint and JSON specification do not match')
    problem = load_problem(specification['problem'])
    if problem.physics_sha256 != specification['physics_sha256']:
        raise ValueError('Reconstructed problem physics does not match the checkpoint')
    config = specification['config']
    if config['architecture'] == 'DeepXDE':
        raise ValueError('Portable native checkpoints do not support the DeepXDE backend')
    dtype = DTYPES.get(config.get('dtype', 'float64'))
    if dtype is None:
        raise ValueError('Unsupported model dtype')
    with torch.random.fork_rng(devices=[]):
        model = build_model(problem, config['architecture'], config['width'], config['depth'], config['activation'],
            input_normalization=config.get('input_normalization'),
            trace_parameterization=config.get('trace_parameterization', 'standard'),
            soil_envelope_gain=config.get('soil_envelope_gain', 1.)).to(device=device, dtype=dtype)
    weights = model_weights(payload, selection)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, specification, payload


def infer(model, xy, flux=False):
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Inference coordinates must be a finite (N, 2) array in physical metres')
    problem = model.problem
    labels = np.asarray(problem.owner_index(points), dtype=np.int64)
    if (labels < 0).any():
        raise ValueError('Inference coordinates lie outside the physical domain')
    reference = next(model.parameters())
    temperatures, heat_flux = np.empty(len(points)), np.empty((len(points), 2))
    with torch.set_grad_enabled(flux):
        for region in problem.regions:
            mask = labels == region.id
            if not mask.any():
                continue
            coordinates = torch.as_tensor(points[mask], dtype=reference.dtype, device=reference.device).requires_grad_(flux)
            temperature, _ = model.field(region, coordinates)
            temperatures[mask] = temperature.detach().cpu().numpy().ravel()
            if flux:
                from pinn_v3.physics.operators import gradients
                derived = -problem.conductivity(region, coordinates) * gradients(temperature, coordinates)[:, :2]
                heat_flux[mask] = derived.detach().cpu().numpy()
    result = dict(T=temperatures, region_ids=labels)
    if flux:
        result['q'] = heat_flux
    return result