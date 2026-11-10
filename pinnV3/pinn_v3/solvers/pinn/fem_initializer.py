"""Warm start for PINN collocation values from the FEniCSx reference run through the WSL adapter."""
import json
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from pinn_v3.problem.build import MEMBERSHIP_TOLERANCE
from pinn_v3.problem.shapes import boundary_distance, contains

_FEM_REFERENCE_CACHE = {}


def fem_token(node_multiplier):
    return '' if node_multiplier == 1.0 else f'_m{node_multiplier:g}'.replace('.', 'p')


def _region_node_masks(definition, dof_xy, include_shared=False):
    owners = definition.owner(dof_xy[:, 0], dof_xy[:, 1])
    masks = {region_id: owners == region_id for region_id in range(len(definition.regions))}
    if include_shared:
        x, y = dof_xy[:, 0], dof_xy[:, 1]
        x0, x1, y0, y1 = definition.bounds
        tolerance = MEMBERSHIP_TOLERANCE * max(x1 - x0, y1 - y0, 1.0)
        for owner_id, owner_region in enumerate(definition.regions):
            shared = (owners == owner_id) & (boundary_distance(owner_region['shape'], x, y) <= tolerance)
            for region_id, region in enumerate(definition.regions):
                if region_id != owner_id:
                    masks[region_id] |= shared & contains(region['shape'], x, y, tolerance)
    return masks


def points_result(definition, dof_xy, dof_temperature, collocation, node_masks=None):
    """Interpolate FEM nodal values inside each region at its collocation points."""
    masks = _region_node_masks(definition, dof_xy) if node_masks is None else node_masks
    xy_blocks, value_blocks, label_blocks, anchors = [], [], [], {}
    cursor = 0
    for region_id in range(len(definition.regions)):
        points = np.asarray(collocation[region_id], dtype=float).reshape(-1, 2)
        mine = masks[region_id]
        if not np.any(mine):
            raise ValueError(f'FEM solution has no nodes in region {region_id}')
        source_xy, source_temperature = dof_xy[mine], dof_temperature[mine]
        values = np.asarray(LinearNDInterpolator(source_xy, source_temperature)(points), dtype=float).reshape(-1)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = source_temperature[cKDTree(source_xy).query(points[missing])[1]]
        count = len(points)
        xy_blocks.append(points)
        value_blocks.append(values)
        label_blocks.append(np.full(count, region_id))
        anchors[region_id] = np.arange(cursor, cursor + count)
        cursor += count
    return dict(xy=np.vstack(xy_blocks), labels=np.concatenate(label_blocks),
                temperature=np.concatenate(value_blocks), anchors=anchors)


def run_fem_reference(definition, output, node_multiplier):
    """Run the FEniCSx reference at level 0 for one node multiplier; return nodal values and its record."""
    from pinn_v3.solvers.fem.adapter import run_problem_fem
    from pinn_v3.solvers.fem.mesh_resolution import validate_node_multiplier
    node_multiplier = validate_node_multiplier(node_multiplier)
    encoded = json.dumps(definition.doc, sort_keys=True, separators=(',', ':'), allow_nan=False)
    cache_key = (hashlib.sha256(encoded.encode('utf-8')).hexdigest(), node_multiplier)
    cached = _FEM_REFERENCE_CACHE.get(cache_key)
    if cached is not None:
        return dict(dof_xy=cached['dof_xy'].copy(), dof_T=cached['dof_T'].copy(),
                    record=dict(cached['record']), wall_s=0.0, cache_hit=True)
    fem_dir = Path(output) / 'fem'
    started = time.perf_counter()
    [record] = run_problem_fem(definition.doc, fem_dir, levels=(0,), node_multipliers=(node_multiplier,))
    wall = time.perf_counter() - started
    stem = f'fem_l0{fem_token(node_multiplier)}'
    arrays = np.load(fem_dir / f'{stem}.npz')
    result = dict(dof_xy=arrays['dof_xy'], dof_T=arrays['dof_T'], record=record, wall_s=wall, cache_hit=False)
    _FEM_REFERENCE_CACHE[cache_key] = dict(dof_xy=result['dof_xy'].copy(), dof_T=result['dof_T'].copy(), record=dict(record))
    return result


def collocation_result(collocation, values):
    """Stack per-region collocation points with their values in the layout the readout expects."""
    xy_blocks, value_blocks, label_blocks, anchors = [], [], [], {}
    cursor = 0
    for region_id in sorted(collocation):
        points = np.asarray(collocation[region_id], dtype=float).reshape(-1, 2)
        xy_blocks.append(points)
        value_blocks.append(np.asarray(values[region_id], dtype=float).reshape(-1))
        label_blocks.append(np.full(len(points), region_id))
        anchors[region_id] = np.arange(cursor, cursor + len(points))
        cursor += len(points)
    return dict(xy=np.vstack(xy_blocks), labels=np.concatenate(label_blocks),
                temperature=np.concatenate(value_blocks), anchors=anchors)


def local_quadratic_fit(definition, node_xy, node_temperature, node_labels, collocation, neighbours=12, node_masks=None):
    """Value and gradient at each collocation point from a local quadratic fit to its region's nodes."""
    fitted = {}
    for region_id in range(len(definition.regions)):
        points = np.asarray(collocation[region_id], dtype=float).reshape(-1, 2)
        mine = node_labels == region_id if node_masks is None else node_masks[region_id]
        nodes_xy, nodes_T = node_xy[mine], node_temperature[mine]
        nearest = cKDTree(nodes_xy).query(points, k=min(neighbours, len(nodes_xy)))[1]
        values, gradients = np.empty(len(points)), np.empty((len(points), 2))
        for row, (point, index) in enumerate(zip(points, nearest)):
            delta = nodes_xy[index] - point
            design = np.column_stack([np.ones(len(delta)), delta[:, 0], delta[:, 1],
                                      delta[:, 0] ** 2, delta[:, 0] * delta[:, 1], delta[:, 1] ** 2])
            coefficients = np.linalg.lstsq(design, nodes_T[index], rcond=None)[0]
            values[row], gradients[row] = coefficients[0], coefficients[1:3]
        fitted[region_id] = (values, gradients)
    return fitted


def fem_gradients(definition, dof_xy, dof_temperature, collocation, neighbours=12, node_masks=None):
    """Gradient of the FEM field at each collocation point from a local quadratic fit to its region's nodes."""
    owners = definition.owner(dof_xy[:, 0], dof_xy[:, 1])
    fitted = local_quadratic_fit(definition, dof_xy, dof_temperature, owners, collocation, neighbours, node_masks)
    return {region_id: gradient for region_id, (_, gradient) in fitted.items()}


def _network_index(model, region):
    network_model = getattr(model, 'base_model', model)
    return 0 if network_model.partition == 'global' else region.id


def _network_inputs(model, region, xy):
    if hasattr(model, 'features'):
        return model.features(region, xy)
    return model.input_coordinates(region, xy)


def readout_transfer(model, result, gradients=None, gradient_weight=1.0, ridge=0.0):
    """Least squares on FEM values and optional gradients for each network's temperature output; no gradient steps."""
    dtype = next(model.parameters()).dtype
    groups = {}
    for region in model.problem.regions:
        groups.setdefault(_network_index(model, region), []).append(region)
    value_errors, gradient_errors, weight_norms = [], [], []
    network_model = getattr(model, 'base_model', model)
    for index, regions in groups.items():
        network = network_model.networks[index]
        last = network[-1]
        blocks, value_rows, value_targets, gradient_rows, gradient_targets = {}, [], [], [], []
        for region in regions:
            indices = result['anchors'][region.id]
            xy = torch.as_tensor(result['xy'][indices], dtype=dtype).requires_grad_(True)
            values = torch.as_tensor(result['temperature'][indices], dtype=torch.float64).reshape(-1, 1)
            captured = []
            hook = last.register_forward_hook(lambda _layer, inputs, output: captured.append((inputs[0], output)))
            try:
                base = model.field(region, xy)[0].double().reshape(-1, 1)
            finally:
                hook.remove()
            value_design = None
            raw_contribution = torch.zeros_like(base)
            for features, raw in captured:
                sensitivity = torch.autograd.grad(base.sum(), raw, create_graph=True,
                    retain_graph=True, allow_unused=True)[0]
                if sensitivity is None:
                    continue
                scale = sensitivity[:, :1].double()
                augmented = torch.cat([features, torch.ones(len(xy), 1, dtype=features.dtype)], dim=1).double()
                contribution = scale * augmented
                value_design = contribution if value_design is None else value_design + contribution
                raw_contribution = raw_contribution + scale * raw[:, :1].double()
            if value_design is None:
                raise ValueError('FEM readout could not trace output-layer sensitivity for a region')
            offset = base - raw_contribution
            value_rows.append(value_design)
            value_targets.append(values - offset)
            block = dict(xy=xy, values=values, gradient=None)
            if gradients is not None:
                length = float(np.sqrt(model.problem.areas[region.id]))
                jacobian = torch.stack([torch.autograd.grad(value_design[:, unit].sum(), xy, retain_graph=True)[0]
                                        for unit in range(value_design.shape[1])], dim=2).double()
                offset_gradient = torch.autograd.grad(offset.sum(), xy, retain_graph=True)[0].double()
                axis_rows, axis_targets = [], []
                for axis in range(2):
                    axis_rows.append(gradient_weight * length * jacobian[:, axis, :])
                    axis_targets.append(gradient_weight * length * torch.as_tensor(
                        gradients[region.id][:, axis], dtype=torch.float64).reshape(-1, 1)
                        - gradient_weight * length * offset_gradient[:, axis:axis + 1])
                block['gradient'] = (torch.cat(axis_rows), torch.cat(axis_targets))
                gradient_rows.append(block['gradient'][0])
                gradient_targets.append(block['gradient'][1])
            blocks[region.id] = block
        design = torch.cat(value_rows + gradient_rows)
        target = torch.cat(value_targets + gradient_targets)
        if ridge > 0:
            width_columns = design.shape[1] - 1
            penalty = torch.cat([float(np.sqrt(ridge)) * torch.eye(width_columns, dtype=torch.float64),
                                 torch.zeros(width_columns, 1, dtype=torch.float64)], dim=1)
            design = torch.cat([design, penalty])
            target = torch.cat([target, torch.zeros(width_columns, 1, dtype=torch.float64)])
        theta = torch.linalg.lstsq(design, target).solution.reshape(-1)
        with torch.no_grad():
            last.weight[0].copy_(theta[:-1].to(last.weight.dtype))
            last.bias[0].copy_(theta[-1].to(last.bias.dtype))
        weight_norms.append(float(theta[:-1].detach().norm()))
        for region in regions:
            block = blocks[region.id]
            with torch.no_grad():
                predicted = model.field(region, block['xy'].detach())[0].double().reshape(-1, 1)
            value_errors.append((predicted - block['values']).reshape(-1))
            if block['gradient'] is not None:
                gradient_design, gradient_target = block['gradient']
                gradient_errors.append((gradient_design @ theta.reshape(-1, 1) - gradient_target).reshape(-1))
    error = torch.cat(value_errors)
    report = dict(steps=0, points=int(len(error)), weight_norm=max(weight_norms),
                  method='readout_gls' if gradients is not None else 'readout',
                  rms_K=float(error.pow(2).mean().sqrt()), max_K=float(error.abs().max()))
    if gradients is not None:
        report['gradient_rms_K'] = float(torch.cat(gradient_errors).detach().pow(2).mean().sqrt())
    return report


def fem_initialization(model, problem, objective, settings, output):
    """Transfer the FEM values at the PINN collocation points into the model and return the report."""
    collocation = {region.id: objective.interior[region.id].detach().cpu().numpy() for region in problem.regions}
    surface_count = 0
    surface_points = {}
    if settings.get('fem_surfaces', False):
        for region in problem.regions:
            interior = collocation[region.id]
            surface = _region_surface_points(objective, region.id)
            surface_points[region.id] = np.unique(np.vstack(surface), axis=0) if surface else np.empty((0, 2))
            collocation[region.id] = np.unique(np.vstack([interior, *surface]), axis=0)
            surface_count += len(collocation[region.id]) - len(np.unique(interior, axis=0))
    fem = run_fem_reference(problem.definition, Path(output) / 'fem_initialization', settings['fem_node_multiplier'])
    masks = _region_node_masks(problem.definition, fem['dof_xy'], include_shared=True) if settings.get('fem_surfaces', False) else None
    result = points_result(problem.definition, fem['dof_xy'], fem['dof_T'], collocation, node_masks=masks)
    if settings['fem_transfer'] in ('readout', 'readout_gls'):
        gradients = None
        if settings['fem_transfer'] == 'readout_gls':
            gradients = fem_gradients(problem.definition, fem['dof_xy'], fem['dof_T'], collocation, node_masks=masks)
        report = readout_transfer(model, result, gradients=gradients, gradient_weight=settings['fem_gradient_weight'], ridge=settings['fem_ridge'])
    else:
        from pinn_v3.solvers.pinn.rbf_fd import pretrain_pinn
        report = pretrain_pinn(model, result, steps=settings['pretrain_steps'],
                               learning_rate=settings['pretrain_lr'], target_points=None)
    if surface_count:
        errors = []
        for region in problem.regions:
            points = surface_points[region.id]
            if not len(points):
                continue
            indices = cKDTree(collocation[region.id]).query(points)[1]
            targets = result['temperature'][result['anchors'][region.id][indices]]
            tensor = torch.as_tensor(points, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
            with torch.no_grad():
                predicted = model.field(region, tensor)[0].detach().cpu().numpy().reshape(-1)
            errors.append(predicted - targets)
        report['surface_rms_K'] = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
    report.update(transfer=settings['fem_transfer'], gradient_weight=settings['fem_gradient_weight'], ridge=settings['fem_ridge'], source='fem', mode=settings['mode'], fem_node_multiplier=settings['fem_node_multiplier'],
                  fem_ndofs=fem['record'].get('ndofs'), fem_h=fem['record'].get('h'),
                  fem_balance_pct=fem['record'].get('balance_consistent_pct', fem['record'].get('balance_pct')), fem_wall_s=fem['wall_s'],
                  fem_cache_hit=fem.get('cache_hit', False),
                  anchor_count=len(result['xy']), fem_surfaces=settings.get('fem_surfaces', False),
                  surface_anchor_count=surface_count)
    return report


def _region_surface_points(objective, region_id):
    surface = []
    for face, xy, _normal, xy_right in getattr(objective, 'faces', ()):
        if region_id == face.left:
            surface.append(xy.detach().cpu().numpy())
        if region_id == face.right:
            surface.append(xy_right.detach().cpu().numpy())
    for piece, xy, _normal in getattr(objective, 'pieces', ()):
        if piece.region == region_id:
            surface.append(xy.detach().cpu().numpy())
    return surface
