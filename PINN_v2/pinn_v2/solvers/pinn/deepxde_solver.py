"""DeepXDE implementation of the shared steady heat-conduction cases."""
import copy
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from pinn_v2.evaluation.audit import evaluate_model
from pinn_v2.manifest.run_manifest import build_manifest, write_json_atomic, write_manifest
from pinn_v2.solvers.pinn.objective import DEFAULT_WEIGHTS
from pinn_v2.solvers.pinn.problem import load_problem

POINTS = dict(n_interior=512, n_layer=256, n_face=128, n_boundary=128)
INTERNAL_MODELS = ('PFNN', 'FNN')
DEEPXDE_SAMPLERS = ('mixed', 'pseudo', 'LHS', 'Halton', 'Hammersley', 'Sobol')


class DeepXDEModel:
    """Adapt a DeepXDE multi-output network to the project's PINN audit API."""

    mixed = False
    kind = 'DeepXDE strong-form PINN'

    def __init__(self, problem, network, internal_model):
        self.problem = problem
        self.network = network
        self.internal_model = internal_model

    def parameters(self):
        return self.network.parameters()

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.network.parameters() if parameter.requires_grad)

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state):
        _load_network_state(self.network, state)

    def field(self, region, xy):
        selector = xy.new_full((len(xy), 1), float(region.id))
        raw = self.network(torch.cat((xy, selector), dim=1))[:, region.id:region.id + 1]
        return _regional_temperature(self.problem, self.network, region, raw), None

    def temperature(self, xy, labels):
        selectors = labels.to(dtype=xy.dtype).reshape(-1, 1)
        raw = self.network(torch.cat((xy, selectors), dim=1))
        indices = labels.to(dtype=torch.long).reshape(-1, 1)
        amplitudes = xy.new_tensor([self.problem.amplitude(region, 'region') for region in self.problem.regions])
        scales = self.problem.T_scale * self.network.temperature_offsets[indices]
        return self.problem.T_ref + scales + raw.gather(1, indices) * amplitudes[indices]


def _regional_temperature(problem, network, region, raw):
    return (problem.T_ref + problem.T_scale * network.temperature_offsets[region.id]
            + problem.amplitude(region, 'region') * raw)


def _load_network_state(network, state):
    state = dict(state)
    offset_name = 'temperature_offsets'
    if offset_name not in state:
        with torch.no_grad():
            network.temperature_offsets.zero_()
    incompatible = network.load_state_dict(state, strict=False)
    unexpected = incompatible.unexpected_keys
    missing = set(incompatible.missing_keys) - {offset_name}
    if unexpected or missing:
        raise RuntimeError(f'Incompatible DeepXDE checkpoint: missing={sorted(missing)}, unexpected={unexpected}')


def _augmented(points, region_id):
    points = np.asarray(points, dtype=np.float64)
    return np.column_stack((points, np.full(len(points), region_id, dtype=np.float64)))


def _save_collocation(path, anchors, problem, points):
    arrays = {f'interior_{region.id}': anchors[np.round(anchors[:, 2]).astype(int) == region.id, :2]
        for region in problem.regions}
    arrays.update({f'interface_{index}': face.xy
        for index, face in enumerate(problem.faces(points['n_face']))})
    arrays.update({f'boundary_{index}': piece.xy
        for index, piece in enumerate(problem.pieces(points['n_boundary']))})
    temporary = Path(path).with_name('collocation.tmp.npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def sample_deepxde_interior(dde, problem, region, count, seed, sampler='mixed'):
    """Sample physical per-region points, optionally using DeepXDE's QMC samplers."""
    if sampler not in DEEPXDE_SAMPLERS:
        raise ValueError(f'Unknown DeepXDE sampler {sampler!r}; choose from {DEEPXDE_SAMPLERS}')
    if sampler == 'mixed':
        return problem.sample_interior(region, count, seed, 'mixed')

    from deepxde.geometry.sampler import sample as unit_sample

    def unit_points(size, batch):
        batch_seed = (int(seed) + 104729 * batch) % (2 ** 32)
        requested_size = size
        if sampler == 'Sobol':
            size = (1 << size.bit_length()) - 1
        state = np.random.get_state()
        np.random.seed(batch_seed)
        try:
            values = unit_sample(size, 2, sampler=sampler)
        finally:
            np.random.set_state(state)
        if sampler != 'pseudo':
            shift = np.random.default_rng(batch_seed).random(2)
            values = (values + shift) % 1.0
        return values[:requested_size]

    case = getattr(problem, 'case', {})
    if case.get('kind') == 'annulus':
        uv = unit_points(count, 0)
        angle = 2 * np.pi * uv[:, 0]
        radius = np.sqrt(case['ri'] ** 2 + uv[:, 1] * (case['ro'] ** 2 - case['ri'] ** 2))
        return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))

    if region.kind in ('conductor', 'layer'):
        uv = unit_points(count, 0)
        angle = 2 * np.pi * uv[:, 0]
        radius = np.sqrt(region.ri ** 2 + uv[:, 1] * (region.ro ** 2 - region.ri ** 2))
        center = np.asarray(case['cables'][region.cable])
        return center + np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))

    if region.kind != 'soil':
        raise ValueError(f'Unsupported region kind for DeepXDE sampling: {region.kind}')

    if hasattr(problem, '_box'):
        x0, x1, y0, y1 = problem._box(region)
    else:
        x0, x1, y0, y1 = problem.bounds
        if region.axis == 0:
            x0, x1 = region.lower, region.upper
        else:
            y0, y1 = region.lower, region.upper

    batches = []
    accepted = 0
    for batch in range(1000):
        remaining = count - accepted
        candidates_uv = unit_points(max(256, 2 * remaining), batch)
        candidates = np.column_stack((x0 + (x1 - x0) * candidates_uv[:, 0],
                                      y0 + (y1 - y0) * candidates_uv[:, 1]))
        if hasattr(problem, 'domain'):
            candidates = candidates[problem.domain.contains(region, candidates)]
        if len(candidates):
            batches.append(candidates)
            accepted += len(candidates)
        if accepted >= count:
            return np.vstack(batches)[:count]
    raise RuntimeError(f'Unable to draw {count} DeepXDE points in {region.name} using {sampler}')


def _region_inputs(inputs, region_id):
    selector = inputs.new_full((len(inputs), 1), float(region_id))
    return torch.cat((inputs[:, :2], selector), dim=1)


def _aligned_normals(inputs, X, points, normals, cache):
    if 'values' not in cache or len(cache['values']) != len(X):
        from scipy.spatial import cKDTree

        coordinates = np.asarray(X)[:, :2]
        distances, indices = cKDTree(np.asarray(points)).query(coordinates)
        values = np.zeros((len(coordinates), 2), dtype=float)
        matched = distances < 1e-10
        values[matched] = np.asarray(normals)[indices[matched]]
        cache['values'] = values
    return inputs.new_tensor(cache['values'])


def _network(dde, problem, internal_model, width, depth, activation):
    count = len(problem.regions)
    feature_sizes = [problem.input_size(region, 'region') for region in problem.regions]
    feature_width = max(feature_sizes)
    sizes = [feature_width * count] + [width] * depth + [count]
    if internal_model == 'PFNN' and count > 1:
        sizes = [feature_width * count] + [[width] * count for _ in range(depth)] + [count]
        network = dde.nn.PFNN(sizes, activation, 'Glorot normal')
    else:
        network = dde.nn.FNN(sizes, activation, 'Glorot normal')

    def feature_transform(inputs):
        xy = inputs[:, :2]
        region_ids = torch.round(inputs[:, 2:3]).to(dtype=torch.long)
        features = []
        for region, size in zip(problem.regions, feature_sizes):
            local = problem.coordinates(region, xy, 'region')
            if size < feature_width:
                local = torch.nn.functional.pad(local, (0, feature_width - size))
            active = (region_ids == region.id).to(dtype=inputs.dtype)
            features.append(local * active)
        return torch.cat(features, dim=1)

    network.apply_feature_transform(feature_transform)
    network.register_parameter('temperature_offsets', torch.nn.Parameter(next(network.parameters()).new_zeros(count)))
    return network


def _build_constraints(dde, problem, network, anchors, points, weights):
    from deepxde import grad

    region_ids = np.asarray(anchors[:, 2], dtype=int)
    residuals = []
    for region in problem.regions:
        region_id = region.id
        count = int(np.count_nonzero(region_ids == region_id))
        if not count:
            raise ValueError(f'No interior points sampled for {region.name}')
        amplitude = problem.amplitude(region, 'region')
        scale = problem.scales(region)['divergence']
        correction = float(np.sqrt(len(anchors) / count))

        def residual(inputs, outputs, region=region, region_id=region_id, amplitude=amplitude,
                     scale=scale, correction=correction):
            gradient_x = amplitude * grad.jacobian(outputs, inputs, i=region_id, j=0)
            gradient_y = amplitude * grad.jacobian(outputs, inputs, i=region_id, j=1)
            coordinates = inputs[:, :2]
            conductivity = problem.conductivity(region, coordinates)
            flux_x, flux_y = -conductivity * gradient_x, -conductivity * gradient_y
            divergence = grad.jacobian(flux_x, inputs, i=0, j=0) + grad.jacobian(flux_y, inputs, i=0, j=1)
            source = problem.source(region, coordinates)
            selected = (torch.round(inputs[:, 2:3]) == region_id).to(dtype=inputs.dtype)
            return ((divergence - source) / scale) * selected * correction

        residuals.append(residual)

    def pde_residual(inputs, outputs):
        return [residual(inputs, outputs) for residual in residuals]

    conditions = []
    condition_weights = []

    def operator_bc(xy, residual_fn):
        augmented = _augmented(xy, 0)
        return dde.icbc.PointSetOperatorBC(augmented, np.zeros((len(augmented), 1)), residual_fn)

    for face in problem.faces(points['n_face']):
        left, right = problem.regions[face.left], problem.regions[face.right]
        amplitude_left = problem.amplitude(left, 'region')
        amplitude_right = problem.amplitude(right, 'region')
        flux_scale = max(problem.scales(left)['flux'], problem.scales(right)['flux'])
        normal_cache = {}

        def temperature_jump(inputs, outputs, X, left=left, right=right):
            left_outputs = network(_region_inputs(inputs, left.id))
            right_outputs = network(_region_inputs(inputs, right.id))
            left_t = _regional_temperature(problem, network, left, left_outputs[:, left.id:left.id + 1])
            right_t = _regional_temperature(problem, network, right, right_outputs[:, right.id:right.id + 1])
            return (left_t - right_t) / problem.T_scale

        def flux_jump(inputs, outputs, X, left=left, right=right, amplitude_left=amplitude_left,
                      amplitude_right=amplitude_right, flux_scale=flux_scale, face=face,
                      normal_cache=normal_cache):
            left_outputs = network(_region_inputs(inputs, left.id))
            right_outputs = network(_region_inputs(inputs, right.id))
            gradient_left = torch.cat([grad.jacobian(left_outputs, inputs, i=left.id, j=j) for j in (0, 1)], dim=1) * amplitude_left
            gradient_right = torch.cat([grad.jacobian(right_outputs, inputs, i=right.id, j=j) for j in (0, 1)], dim=1) * amplitude_right
            coordinates = inputs[:, :2]
            q_left = -problem.conductivity(left, coordinates) * gradient_left
            q_right = -problem.conductivity(right, coordinates) * gradient_right
            n = _aligned_normals(inputs, X, face.xy, face.normal, normal_cache)
            return ((q_left - q_right) * n).sum(1, keepdim=True) / flux_scale

        conditions.extend((operator_bc(face.xy, temperature_jump), operator_bc(face.xy, flux_jump)))
        condition_weights.extend((weights['interface_temperature'], weights['interface_flux']))

    for piece in problem.pieces(points['n_boundary']):
        region = problem.regions[piece.region]
        amplitude = problem.amplitude(region, 'region')
        flux_scale = problem.scales(region)['flux']
        normal_cache = {}
        bc = piece.bc

        def boundary_residual(inputs, outputs, X, region=region, amplitude=amplitude,
                      flux_scale=flux_scale, piece=piece, normal_cache=normal_cache, bc=bc):
            region_outputs = network(_region_inputs(inputs, region.id))
            temperature = _regional_temperature(problem, network, region, region_outputs[:, region.id:region.id + 1])
            gradient = torch.cat([grad.jacobian(region_outputs, inputs, i=region.id, j=j) for j in (0, 1)], dim=1) * amplitude
            flux = -problem.conductivity(region, inputs[:, :2]) * gradient
            n = _aligned_normals(inputs, X, piece.xy, piece.normal, normal_cache)
            normal_flux = (flux * n).sum(1, keepdim=True)
            if bc['type'] == 'dirichlet':
                return (temperature - bc['value'](inputs[:, :2])) / problem.T_scale
            if bc['type'] == 'neumann':
                return (normal_flux - bc['value'](inputs[:, :2])) / flux_scale
            return (normal_flux - bc['h'] * (temperature - bc['T_inf'](inputs[:, :2]))) / flux_scale

        conditions.append(operator_bc(piece.xy, boundary_residual))
        condition_weights.append(weights[bc['type']])

    energy_points = []
    energy_weights = []
    if weights['energy'] > 0:
        for region in problem.regions:
            def energy_residual(inputs, outputs, X, region=region):
                balance = inputs.new_tensor(-problem.region_power[region.id])
                for face in problem.faces(points['n_face']):
                    if region.id not in (face.left, face.right):
                        continue
                    coordinates = inputs.new_tensor(face.xy).detach().requires_grad_(True)
                    selector = coordinates.new_full((len(coordinates), 1), float(region.id))
                    values = network(torch.cat((coordinates, selector), dim=1))[:, region.id:region.id + 1]
                    temperature = _regional_temperature(problem, network, region, values)
                    gradient = torch.autograd.grad(temperature, coordinates, torch.ones_like(temperature), create_graph=True, retain_graph=True)[0]
                    flux = -problem.conductivity(region, coordinates) * gradient
                    normal = inputs.new_tensor(face.normal)
                    sign = 1. if region.id == face.left else -1.
                    balance = balance + sign * face.weight * (flux * normal).sum(1).sum()
                for piece in problem.pieces(points['n_boundary']):
                    if piece.region != region.id:
                        continue
                    coordinates = inputs.new_tensor(piece.xy).detach().requires_grad_(True)
                    selector = coordinates.new_full((len(coordinates), 1), float(region.id))
                    values = network(torch.cat((coordinates, selector), dim=1))[:, region.id:region.id + 1]
                    temperature = _regional_temperature(problem, network, region, values)
                    gradient = torch.autograd.grad(temperature, coordinates, torch.ones_like(temperature), create_graph=True, retain_graph=True)[0]
                    flux = -problem.conductivity(region, coordinates) * gradient
                    normal = inputs.new_tensor(piece.normal)
                    balance = balance + piece.weight * (flux * normal).sum(1).sum()
                return (balance / max(problem.power_scale, 1.0)).reshape(1, 1).expand(len(inputs), 1)

            energy_points.append(_augmented(problem.evaluation()[0][:1], region.id))
            energy_weights.append(weights['energy'])
            conditions.append(dde.icbc.PointSetOperatorBC(energy_points[-1], np.zeros((1, 1)), energy_residual))
    else:
        energy_weights = []

    condition_weights.extend(condition_weights[:0])
    loss_weights = [weights['pde']] * len(problem.regions) + condition_weights + energy_weights
    return pde_residual, conditions, loss_weights


def run(source, output, internal_model='PFNN', adam_steps=3000, lbfgs_steps=0, lr=2e-3,
        width=32, depth=3, activation='tanh', weights=None, points=None, init_seed=0,
        sampling_seed=11, log_every=100, reference=None, reference_label='supplied',
    study=None, command=None, on_log=None, stop_callback=None, full_audit=True, initial_state=None,
    deepxde_sampling='mixed', device='cpu'):
    """Train one DeepXDE PINN from the same case JSON and return the common run record."""
    try:
        import deepxde as dde
    except ImportError as exc:
        raise RuntimeError('Install DeepXDE with: pip install -e ".[deepxde]"') from exc
    if internal_model not in INTERNAL_MODELS:
        raise ValueError(f'Unknown DeepXDE network {internal_model!r}; choose from {INTERNAL_MODELS}')
    if deepxde_sampling not in DEEPXDE_SAMPLERS:
        raise ValueError(f'Unknown DeepXDE sampler {deepxde_sampling!r}; choose from {DEEPXDE_SAMPLERS}')
    if device not in ('cpu', 'xpu'):
        raise ValueError("DeepXDE device must be 'cpu' or 'xpu'")
    if device == 'xpu' and (not hasattr(torch, 'xpu') or not torch.xpu.is_available()):
        raise RuntimeError('XPU was requested, but PyTorch reports no available XPU device')
    if min(adam_steps, lbfgs_steps) < 0 or width < 1 or depth < 1 or lr <= 0:
        raise ValueError('Invalid DeepXDE training parameters')

    selected_device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    problem = load_problem(source)
    points = dict(POINTS, **(points or {}))
    loss_weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
    loss_weights['constitutive'] = 0.0
    unknown = set(loss_weights) - set(DEFAULT_WEIGHTS)
    if unknown or any(not np.isfinite(value) or value < 0 for value in loss_weights.values()):
        raise ValueError(f'Invalid objective weights: {sorted(unknown)}')
    if activation == 'silu':
        activation = 'swish'
    if activation not in ('tanh', 'sin', 'gelu', 'swish'):
        raise ValueError(f'Unsupported DeepXDE activation {activation!r}')

    config = dict(architecture='DeepXDE', deepxde_model=internal_model, width=width, depth=depth,
                  activation=activation, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps, lr=lr,
                  weights=loss_weights, points=points, attention=None, deepxde_sampling=deepxde_sampling)
    result = dict(case=problem.name, config=config, seeds=dict(init=init_seed, sampling=sampling_seed),
                  status='failed', physics_sha256=problem.physics_sha256, fem_labels_used=0,
                  reference_used_in_loss=False, study=study, device=str(selected_device), solver='DeepXDE')
    history = []
    previous_default_device = torch.get_default_device()
    try:
        torch.set_default_device(selected_device)
        random.seed(init_seed)
        np.random.seed(init_seed)
        torch.manual_seed(init_seed)
        dde.config.set_default_float('float64')
        dde.config.set_random_seed(init_seed)

        interior = []
        for region in problem.regions:
            count = points['n_interior'] if region.kind == 'soil' else points['n_layer']
            xy = sample_deepxde_interior(dde, problem, region, count,
                sampling_seed + 7919 * region.id, deepxde_sampling)
            interior.append(_augmented(xy, region.id))
        anchors = np.vstack(interior)
        _save_collocation(output.joinpath('collocation.npz'), anchors, problem, points)
        x0, x1, y0, y1 = problem.bounds
        selector_max = max(len(problem.regions) - 1, 1)
        geometry = dde.geometry.Cuboid([x0, y0, -0.5], [x1, y1, selector_max + 0.5])
        network = _network(dde, problem, internal_model, width, depth, activation)
        actual_device = str(next(network.parameters()).device)
        if torch.device(actual_device).type != selected_device.type:
            raise RuntimeError(f'DeepXDE network is on {actual_device}, expected {selected_device}')
        result['device'] = actual_device
        if initial_state is not None:
            _load_network_state(network, initial_state)
        checkpoint = output.joinpath('model.pt')

        def save_checkpoint():
            temporary = checkpoint.with_suffix('.pt.tmp')
            torch.save({name: value.detach().cpu() for name, value in network.state_dict().items()}, temporary)
            os.replace(temporary, checkpoint)

        save_checkpoint()
        pde_residual, conditions, dde_loss_weights = _build_constraints(dde, problem, network, anchors, points, loss_weights)
        # Region-tagged anchors define the domain set; DeepXDE's generic domain sampler stays disabled.
        data = dde.data.PDE(geometry, pde_residual, conditions, num_domain=0, num_boundary=0,
                    anchors=anchors, train_distribution='pseudo')
        dde_model = dde.Model(data, network)
        model = DeepXDEModel(problem, network, internal_model)
        labels = [f'pde_{region.name}' for region in problem.regions]
        labels.extend(f'constraint_{index}' for index in range(len(conditions)))
        best = dict(total=float('inf'), state=None, step=-1)
        phase = {'name': 'adam'}

        class Observer(dde.callbacks.Callback):
            def on_epoch_end(self):
                step = int(self.model.train_state.iteration)
                if step % max(log_every, 1) and step != adam_steps + lbfgs_steps:
                    return
                values = np.asarray(self.model.train_state.loss_train, dtype=float).ravel()
                if not len(values):
                    return
                total = float(np.sum(values))
                if np.isfinite(total) and total < best['total']:
                    best.update(total=total, state=copy.deepcopy(network.state_dict()), step=step)
                row = dict(step=step, phase=phase['name'], total=total, lr=lr)
                row.update({name: float(value) for name, value in zip(labels, values)})
                history.append(row)
                save_checkpoint()
                if on_log is not None:
                    on_log(row, model)
                if stop_callback is not None and stop_callback(row, model):
                    self.model.stop_training = True

        observer = Observer()
        dde_model.compile('adam', lr=lr, loss_weights=dde_loss_weights, verbose=0)
        dde_model.train(iterations=adam_steps, display_every=max(log_every, 1), callbacks=[observer], verbose=0)
        if lbfgs_steps:
            phase['name'] = 'lbfgs'
            dde.optimizers.set_LBFGS_options(maxiter=lbfgs_steps)
            dde_model.compile('L-BFGS', loss_weights=dde_loss_weights, verbose=0)
            dde_model.train(display_every=max(log_every, 1), callbacks=[observer], verbose=0)
        if best['state'] is None:
            values = np.asarray(dde_model.train_state.loss_train, dtype=float).ravel()
            total = float(np.sum(values))
            if not np.isfinite(total):
                raise FloatingPointError('DeepXDE did not reach a finite loss')
            best.update(total=total, state=copy.deepcopy(network.state_dict()), step=int(dde_model.train_state.iteration))
        network.load_state_dict(best['state'])
        report, arrays = evaluate_model(model, reference, reference_label, include_audit=full_audit)
        result.update(status='completed', parameters=model.parameter_count(), variant_kind=model.kind,
                      best_step=best['step'], final_loss=best['total'], final_parts={},
                      evaluation=report, elapsed_s=time.perf_counter() - started, history=history)
        np.savez_compressed(output.joinpath('evaluation.npz'), **arrays)
        save_checkpoint()
    except Exception as exc:
        result.update(status='failed', elapsed_s=time.perf_counter() - started,
                      error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        write_json_atomic(output.joinpath('failure.json'), dict(error=result['error'], history=history, config=config))
    finally:
        torch.set_default_device(previous_default_device)
    write_json_atomic(output.joinpath('result.json'), result)
    write_manifest(output, build_manifest('deepxde_pinn_run', command or [sys.executable, '-m', 'pinn_v2.solvers.pinn.deepxde_solver'],
        seeds=result['seeds'], extra=dict(case=problem.name, config=config, status=result['status'],
            physics_sha256=problem.physics_sha256, fem_labels_used=0, study=study, device=result['device'])))
    return result


def load_model(run_dir, problem):
    import json
    import deepxde as dde

    config = json.loads(Path(run_dir).joinpath('result.json').read_text(encoding='utf-8'))['config']
    dde.config.set_default_float('float64')
    network = _network(dde, problem, config['deepxde_model'], config['width'], config['depth'], config['activation'])
    _load_network_state(network, torch.load(Path(run_dir).joinpath('model.pt'), map_location='cpu', weights_only=True))
    network.eval()
    return DeepXDEModel(problem, network, config['deepxde_model'])