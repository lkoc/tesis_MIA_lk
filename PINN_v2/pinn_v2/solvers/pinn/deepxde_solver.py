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
from pinn_v2.manifest.run_manifest import atomic_replace, build_manifest, write_json_atomic, write_manifest
from pinn_v2.solvers.pinn.objective import DEFAULT_WEIGHTS
from pinn_v2.solvers.pinn.problem import load_problem

POINTS = dict(n_interior=512, n_layer=256, n_face=128, n_boundary=128)
EXACT_MODELS = dict(EXACT='DeepXDE_EXACT', EXACT_PFNN='DeepXDE_PFNN_EXACT', EXACT_FNN='DeepXDE_FNN_EXACT')
INTERNAL_MODELS = ('PFNN', 'FNN', *EXACT_MODELS)
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
    atomic_replace(temporary, path)


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


def _network(dde, problem, internal_model, width, depth, activation, input_normalization=None):
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
            local = (problem.unit_coordinates(region, xy, 'region') if input_normalization == 'unit_box_v1'
                else problem.coordinates(region, xy, 'region'))
            if size < feature_width:
                local = torch.nn.functional.pad(local, (0, feature_width - size))
            active = (region_ids == region.id).to(dtype=inputs.dtype)
            features.append(local * active)
        return torch.cat(features, dim=1)

    network.apply_feature_transform(feature_transform)
    network.register_parameter('temperature_offsets', torch.nn.Parameter(next(network.parameters()).new_zeros(count)))
    return network


def _exact_network(dde, problem, width, depth, activation, dtype, input_normalization,
    trace_parameterization, soil_envelope_gain, architecture='M1_EXACT'):
    from pinn_v2.solvers.pinn.networks import build_model

    class ExactNetwork(dde.nn.NN):
        def __init__(self):
            super().__init__()
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                self.field_model = build_model(problem, architecture, width, depth, activation,
                    input_normalization=input_normalization, trace_parameterization=trace_parameterization,
                    soil_envelope_gain=soil_envelope_gain).to(dtype=dtype)
            finally:
                torch.set_default_dtype(previous_dtype)

        def forward(self, inputs):
            temperature = inputs[:, :1] * 0
            labels = torch.round(inputs[:, 2]).long()
            for region in problem.regions:
                selected = labels == region.id
                if selected.any():
                    temperature[selected] = self.field_model.field(region, inputs[selected, :2])[0]
            return temperature

    return ExactNetwork()


def _build_exact_constraints(dde, problem, anchors, pde_weight):
    labels = np.asarray(anchors[:, 2], dtype=int)
    residuals = []
    for region in problem.regions:
        count = int(np.count_nonzero(labels == region.id))
        if not count:
            raise ValueError(f'No PDE anchors in {region.name}')
        correction = float(np.sqrt(len(anchors) / count))

        def residual(inputs, outputs, region=region, correction=correction):
            coordinates = inputs[:, :2]
            gradient_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
            gradient_y = dde.grad.jacobian(outputs, inputs, i=0, j=1)
            conductivity = problem.conductivity(region, coordinates)
            flux_x, flux_y = -conductivity * gradient_x, -conductivity * gradient_y
            divergence = (dde.grad.jacobian(flux_x, inputs, i=0, j=0) +
                dde.grad.jacobian(flux_y, inputs, i=0, j=1))
            selected = (torch.round(inputs[:, 2:3]) == region.id).to(inputs.dtype)
            source = problem.source(region, coordinates).detach()
            return selected * correction * (divergence - source) / problem.scales(region)['divergence']

        residuals.append(residual)

    def pde(inputs, outputs):
        return [residual(inputs, outputs) for residual in residuals]

    return pde, [], [pde_weight] * len(problem.regions)


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
    deepxde_sampling='mixed', device='cpu', dtype='float64', input_normalization=None,
    sampling='mixed', trace_parameterization='legacy', soil_envelope_gain=1.,
    accuracy_stop=None, extra_lbfgs_steps=5, warm_start_source=None):
    """Train one DeepXDE PINN from the same case JSON and return the common run record."""
    try:
        import deepxde as dde
    except ImportError as exc:
        raise RuntimeError('Install DeepXDE with: pip install -e ".[deepxde]"') from exc
    if internal_model not in INTERNAL_MODELS:
        raise ValueError(f'Unknown DeepXDE network {internal_model!r}; choose from {INTERNAL_MODELS}')
    if deepxde_sampling not in DEEPXDE_SAMPLERS:
        raise ValueError(f'Unknown DeepXDE sampler {deepxde_sampling!r}; choose from {DEEPXDE_SAMPLERS}')
    exact_mode = internal_model in EXACT_MODELS
    if dde.backend.backend_name != 'pytorch':
        raise ValueError('This DeepXDE adapter requires the PyTorch backend')
    if dtype not in ('float32', 'float64'):
        raise ValueError('dtype must be float32 or float64')
    if exact_mode and activation not in ('tanh', 'silu'):
        raise ValueError('DeepXDE_EXACT supports the native tanh and silu liftings')
    if sampling not in ('mixed', 'uniform', 'interface', 'paired', 'annulus_log_v1'):
        raise ValueError('Unsupported native sampling policy')
    if deepxde_sampling != 'mixed' and sampling != 'mixed':
        raise ValueError('Choose a native sampler or a DeepXDE sampler, not both')
    if not exact_mode and (input_normalization is not None or sampling != 'mixed' or
            trace_parameterization != 'legacy' or soil_envelope_gain != 1.):
        raise ValueError('Exact encoding and trace controls require DeepXDE_EXACT')
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
    if sampling == 'annulus_log_v1' and problem.name != 'annulus':
        raise ValueError('annulus_log_v1 sampling is supported only by the concentric annulus problem')
    points = dict(POINTS, **(points or {}))
    loss_weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
    loss_weights['constitutive'] = 0.0
    unknown = set(loss_weights) - set(DEFAULT_WEIGHTS)
    if unknown or any(not np.isfinite(value) or value < 0 for value in loss_weights.values()):
        raise ValueError(f'Invalid objective weights: {sorted(unknown)}')
    if exact_mode:
        if loss_weights['pde'] <= 0:
            raise ValueError('DeepXDE_EXACT requires a positive PDE weight')
        loss_weights = {name: value if name == 'pde' else 0. for name, value in loss_weights.items()}
        input_normalization = input_normalization or 'unit_box_v1'
    if activation == 'silu' and not exact_mode:
        activation = 'swish'
    if activation not in ('tanh', 'sin', 'gelu', 'swish'):
        raise ValueError(f'Unsupported DeepXDE activation {activation!r}')

    config = dict(architecture=EXACT_MODELS[internal_model] if exact_mode else 'DeepXDE', deepxde_model=internal_model, width=width, depth=depth,
                  activation=activation, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps, lr=lr,
                  weights=loss_weights, points=points, attention=None, deepxde_sampling=deepxde_sampling,
                  dtype=dtype)
    if exact_mode:
        config.update(loss_policy='pde_only_exact_transmission', input_normalization=input_normalization,
            sampling=sampling, sampling_version='paired_surfaces_v1' if sampling == 'paired' else 'balanced_cables_v2',
            trace_parameterization=trace_parameterization, soil_envelope_gain=soil_envelope_gain,
            optimizer_policy='adam_cosine_then_fixed_lbfgs_outer20', continuation_policy='warm_start_only',
            log_every=log_every)
    if initial_state is not None:
        config.update(initialization_policy='weights_only_warm_start',
            warm_start_source=str(warm_start_source) if warm_start_source is not None else None)
    from pinn_v2.solvers.pinn.trainer import AccuracyStop
    monitor = AccuracyStop(accuracy_stop, reference, reference_label) if accuracy_stop is not None else None
    if monitor is not None:
        if not exact_mode:
            raise ValueError('Accuracy-driven L-BFGS extension requires an exact DeepXDE model')
        if extra_lbfgs_steps < 0 or int(extra_lbfgs_steps) != extra_lbfgs_steps:
            raise ValueError('extra_lbfgs_steps must be a nonnegative integer')
        if reference is None and problem.reference(problem.evaluation()[0]) is None:
            raise ValueError('Accuracy stopping requires an analytic or supplied reference')
        config.update(accuracy_stop=monitor.settings, extra_lbfgs_steps=extra_lbfgs_steps)
    result = dict(case=problem.name, config=config, seeds=dict(init=init_seed, sampling=sampling_seed),
                  status='failed', physics_sha256=problem.physics_sha256, fem_labels_used=0,
                  reference_used_in_loss=False, study=study, device=str(selected_device), solver='DeepXDE')
    history = []
    previous_default_device = torch.get_default_device()
    previous_default_dtype = torch.get_default_dtype()
    previous_dde_float = dde.config.default_float()
    try:
        torch.set_default_device(selected_device)
        random.seed(init_seed)
        np.random.seed(init_seed)
        torch.manual_seed(init_seed)
        dde.config.set_default_float(dtype)
        dde.config.set_random_seed(init_seed)

        interior = []
        for region in problem.regions:
            count = points['n_interior'] if region.kind == 'soil' else points['n_layer']
            if exact_mode and deepxde_sampling == 'mixed':
                xy = problem.sample_interior(region, count, sampling_seed + 7919 * region.id, sampling)
            else:
                xy = sample_deepxde_interior(dde, problem, region, count,
                    sampling_seed + 7919 * region.id, deepxde_sampling)
            interior.append(_augmented(xy, region.id))
        anchors = np.vstack(interior)
        _save_collocation(output.joinpath('collocation.npz'), anchors, problem, points)
        x0, x1, y0, y1 = problem.bounds
        selector_max = max(len(problem.regions) - 1, 1)
        geometry = dde.geometry.Cuboid([x0, y0, -0.5], [x1, y1, selector_max + 0.5])
        network = (_exact_network(dde, problem, width, depth, activation, getattr(torch, dtype),
            input_normalization, trace_parameterization, soil_envelope_gain,
            architecture=EXACT_MODELS[internal_model]) if exact_mode else
            _network(dde, problem, internal_model, width, depth, activation))
        actual_device = str(next(network.parameters()).device)
        if torch.device(actual_device).type != selected_device.type:
            raise RuntimeError(f'DeepXDE network is on {actual_device}, expected {selected_device}')
        result['device'] = actual_device
        if initial_state is not None:
            if exact_mode:
                network.field_model.load_state_dict(initial_state, strict=True)
            else:
                _load_network_state(network, initial_state)
        checkpoint = output.joinpath('model.pt')
        model = network.field_model if exact_mode else DeepXDEModel(problem, network, internal_model)
        if exact_mode:
            from pinn_v2.solvers.pinn import checkpoint as portable
            specification = portable.specification_for(problem, config, result['seeds'], source)

        def save_checkpoint():
            if exact_mode:
                portable.save_checkpoint(checkpoint, model, specification)
                return
            temporary = checkpoint.with_suffix('.pt.tmp')
            torch.save({name: value.detach().cpu() for name, value in network.state_dict().items()}, temporary)
            atomic_replace(temporary, checkpoint)

        save_checkpoint()
        if exact_mode:
            pde_residual, conditions, dde_loss_weights = _build_exact_constraints(dde, problem, anchors, loss_weights['pde'])
        else:
            pde_residual, conditions, dde_loss_weights = _build_constraints(dde, problem, network, anchors, points, loss_weights)
        result['constraint_count'] = len(conditions)
        # Region-tagged anchors define the domain set; DeepXDE's generic domain sampler stays disabled.
        data = dde.data.PDE(geometry, pde_residual, conditions, num_domain=0, num_boundary=0,
                    anchors=anchors, train_distribution='pseudo')
        dde_model = dde.Model(data, network)
        labels = [f'pde_{region.name}' for region in problem.regions]
        labels.extend(f'constraint_{index}' for index in range(len(conditions)))
        best = dict(total=float('inf'), state=None, step=-1)
        phase = {'name': 'adam'}

        class Observer(dde.callbacks.Callback):
            def on_epoch_end(self):
                self.record()

            def record(self, force=False):
                step = int(self.model.train_state.iteration)
                if not force and phase['name'] != 'lbfgs' and step % max(log_every, 1) and step != adam_steps + lbfgs_steps:
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
                if monitor is not None and monitor.check(row, model, force=force or phase['name'] == 'lbfgs'):
                    self.model.stop_training = True
                save_checkpoint()
                if on_log is not None:
                    on_log(row, model)
                if stop_callback is not None and stop_callback(row, model):
                    self.model.stop_training = True

        observer = Observer()
        decay = ('cosine', max(adam_steps, 1), lr * 1e-2) if exact_mode else None
        dde_model.compile('adam', lr=lr, loss_weights=dde_loss_weights, decay=decay, verbose=0)
        dde_model.train(iterations=adam_steps, display_every=max(log_every, 1), callbacks=[observer], verbose=0)
        observer.record(force=True)
        if lbfgs_steps and not dde_model.stop_training:
            phase['name'] = 'lbfgs'
            if exact_mode:
                optimizer = torch.optim.LBFGS(network.parameters(), lr=1., max_iter=20,
                    history_size=50, line_search_fn='strong_wolfe')
                dde_model.compile(optimizer, loss_weights=dde_loss_weights, verbose=0)
                dde_model.train(iterations=lbfgs_steps + (extra_lbfgs_steps if monitor is not None else 0),
                    display_every=1, callbacks=[observer], verbose=0)
                result['lbfgs_function_evaluations'] = int(next(iter(optimizer.state.values()), {}).get('func_evals', 0))
            else:
                dde.optimizers.set_LBFGS_options(maxiter=lbfgs_steps)
                dde_model.compile('L-BFGS', loss_weights=dde_loss_weights, verbose=0)
                dde_model.train(display_every=max(log_every, 1), callbacks=[observer], verbose=0)
            observer.record(force=True)
        if best['state'] is None:
            values = np.asarray(dde_model.train_state.loss_train, dtype=float).ravel()
            total = float(np.sum(values))
            if not np.isfinite(total):
                raise FloatingPointError('DeepXDE did not reach a finite loss')
            best.update(total=total, state=copy.deepcopy(network.state_dict()), step=int(dde_model.train_state.iteration))
        if monitor is None:
            network.load_state_dict(best['state'])
        else:
            monitor.check(history[-1], model, force=True)
            lbfgs_calls = len({row['step'] for row in history if row['phase'] == 'lbfgs'})
            result['accuracy_stop'] = dict(monitor.summary(),
                termination_reason='accuracy_targets_met' if monitor.triggered else 'budget_exhausted',
                lbfgs_outer_calls=lbfgs_calls, extra_lbfgs_outer_calls=max(lbfgs_calls - lbfgs_steps, 0),
                selected_step=int(history[-1]['step']), selection_policy='stopping_or_final_state')
        report, arrays = evaluate_model(model, reference, reference_label, include_audit=full_audit)
        result.update(status='completed', parameters=model.parameter_count(), variant_kind=model.kind,
                      best_step=best['step'], final_loss=best['total'], final_parts={},
                      evaluation=report, elapsed_s=time.perf_counter() - started, history=history)
        from pinn_v2.solvers.pinn.trainer import _evaluation_is_finite
        result['evaluation_finite'] = _evaluation_is_finite(report, arrays)
        if not result['evaluation_finite']:
            raise FloatingPointError('DeepXDE evaluation contains non-finite values')
        if exact_mode:
            from pinn_v2.solvers.pinn.objective import Objective
            objective = Objective(model, seed=sampling_seed, sampling=sampling, weights=loss_weights, **points)
            for region in problem.regions:
                objective.interior[region.id] = objective.tensor(anchors[anchors[:, 2] == region.id, :2])
                objective.sources[region.id] = problem.source(region, objective.interior[region.id]).detach()
            total, parts, regional = objective()
            result.update(final_loss=float(total.detach()), final_parts={name: float(value.detach()) for name, value in parts.items()},
                final_region_pde={name: float(value.detach()) for name, value in regional.items()})
            validation_seed = (int(sampling_seed) + 1_000_003) % (2 ** 32)
            validation = Objective(model, seed=validation_seed, weights=loss_weights, **points)
            validation_total, _, _ = validation()
            result['checkpoint_selection'] = dict(final_validation_loss=float(validation_total.detach()),
                validation_seed=validation_seed, selection_policy='minimum_logged_train_pde')
        np.savez_compressed(output.joinpath('evaluation.npz'), **arrays)
        save_checkpoint()
    except Exception as exc:
        result.update(status='failed', elapsed_s=time.perf_counter() - started,
                      error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        write_json_atomic(output.joinpath('failure.json'), dict(error=result['error'], history=history, config=config))
    finally:
        dde.config.set_default_float(previous_dde_float)
        torch.set_default_dtype(previous_default_dtype)
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
    if config['architecture'] in EXACT_MODELS.values():
        from pinn_v2.solvers.pinn.checkpoint import load_checkpoint
        model, _, _ = load_checkpoint(Path(run_dir).joinpath('model.pt'))
        if model.problem.physics_sha256 != problem.physics_sha256:
            raise ValueError('DeepXDE exact checkpoint belongs to different physics')
        return model
    dde.config.set_default_float(config.get('dtype', 'float64'))
    network = _network(dde, problem, config['deepxde_model'], config['width'], config['depth'], config['activation'])
    _load_network_state(network, torch.load(Path(run_dir).joinpath('model.pt'), map_location='cpu', weights_only=True))
    network.eval()
    return DeepXDEModel(problem, network, config['deepxde_model'])