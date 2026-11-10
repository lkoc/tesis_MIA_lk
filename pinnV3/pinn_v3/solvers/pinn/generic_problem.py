"""Adapter from a pinn-v3-problem-1 document to the common PINN Problem surface.

Regions are the shapes of the document; interfaces and domain sides come from their geometry.
The adapter never reads a reference: the exact field is used by evaluation only.
"""
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import torch

from pinn_v3.problem.expressions import evaluate
from pinn_v3.problem.fields import evaluate_field
from pinn_v3.problem.build import Problem as ProblemDefinition, build_problem
from pinn_v3.problem.shapes import boundary_distance, interior_margin
from pinn_v3.solvers.pinn.problem import Face, Piece

NORMAL_OFFSET = 1e-7
MEMBERSHIP_TOLERANCE = 1e-8
INTERFACE_MARGIN_FACTOR = 1e-7
GRID = (401, 201)
MAX_ATTEMPTS = 1000
SIDE_NORMALS = {'bottom': (0., -1.), 'top': (0., 1.), 'left': (-1., 0.), 'right': (1., 0.)}


@dataclass(frozen=True)
class GenericRegion:
    """Region record with the attributes the objective reads; id is the index in the document."""
    id: int
    name: str
    kind: str = 'soil'
    axis: int = 0
    lower: float = 0.0
    upper: float = 1.0


@dataclass(frozen=True)
class GenericCableLayer:
    cable_index: int
    region_id: int
    center: tuple[float, float]
    inner_radius: float
    outer_radius: float


@dataclass(frozen=True)
class GenericCable:
    index: int
    center: tuple[float, float]
    radius: float
    layers: tuple[GenericCableLayer, ...]


@dataclass(frozen=True)
class GenericCircularInterface:
    name: str
    left: int
    right: int
    center: tuple[float, float]
    radius: float
    cable_index: int


def torch_field(spec, xy):
    """Field of a spec at the tensor points xy (N, 2), as a (N, 1) tensor; tables interpolate linearly."""
    x, y = xy[:, 0:1], xy[:, 1:2]
    if isinstance(spec, (int, float)):
        return x * 0 + float(spec)
    if 'expression' in spec:
        return x * 0 + evaluate(spec['expression'], x, y, torch)
    table = spec['table']
    coordinate = (x if table['axis'] == 'x' else y)[:, 0]
    knots = torch.as_tensor(table['coordinates'], dtype=xy.dtype, device=xy.device)
    values = torch.as_tensor(table['values'], dtype=xy.dtype, device=xy.device)
    clamped = torch.minimum(torch.maximum(coordinate, knots[0]), knots[-1])
    index = torch.clamp(torch.searchsorted(knots, clamped.contiguous(), right=True) - 1, 0, len(knots) - 2)
    t = (clamped - knots[index]) / (knots[index + 1] - knots[index])
    return (values[index] + t * (values[index + 1] - values[index])).reshape(-1, 1)


class GenericProblem:
    """Region problem with the regions, interfaces, sides and sources of its document."""
    reference_label = 'reference declared in the problem file'

    def __init__(self, definition):
        if isinstance(definition, GenericProblem):
            definition = definition.definition
        if isinstance(definition, dict):
            self.definition = build_problem(definition)
        elif isinstance(definition, ProblemDefinition):
            self.definition = definition
        else:
            raise TypeError('GenericProblem expects a V3 problem document or canonical Problem')
        self.doc = self.definition.doc
        self.name = self.doc['id']
        self.physics_sha256 = hashlib.sha256(json.dumps(self.doc, sort_keys=True).encode()).hexdigest()
        self.bounds = tuple(self.doc['domain']['bounds'])
        self.family = self._infer_family(self.doc)
        self.regions = tuple(self._make_region(i, region) for i, region in enumerate(self.doc['regions']))
        self.cable_layout = self._build_cable_layout()
        self.circular_interfaces = self._build_circular_interfaces()
        self.source_disks = tuple(
            r['shape'] for r in self.doc['regions']
            if r['shape']['type'] == 'disk' and np.any(evaluate_field(
                r.get('source', 0.0), np.array([r['shape']['center'][0]]),
                np.array([r['shape']['center'][1]])) != 0))
        self.T_ref = self._reference_temperature()
        self.T_scale = self._temperature_scale()
        self.T_level = self.T_ref
        grid_xy, cell_area = self._grid()
        owner = self.owner_index(grid_xy)
        exact_areas = self._exact_cable_areas()
        self.k_ref = {}
        self.region_power = {}
        self.source_abs = {}
        self.areas = {}
        absolute = 0.
        for region in self.regions:
            mine = owner == region.id
            self.areas[region.id] = exact_areas.get(region.id, float(np.count_nonzero(mine)) * cell_area)
            points = grid_xy[mine] if mine.any() else self._representative_point(region).reshape(1, 2)
            k_values = evaluate_field(self._spec(region, 'conductivity'), points[:, 0], points[:, 1])
            q = evaluate_field(self._spec(region, 'source'), points[:, 0], points[:, 1])
            self.k_ref[region.id] = float(np.mean(k_values))
            self.region_power[region.id] = float(np.mean(q) * self.areas[region.id])
            self.source_abs[region.id] = float(np.mean(np.abs(q)))
            absolute += self.source_abs[region.id] * self.areas[region.id]
        characteristic = max(self.k_ref.values()) * self.T_scale
        self.power_total = sum(self.region_power.values())
        self.power_scale = absolute if absolute > 0 else characteristic

    # ----- document access -----
    @staticmethod
    def _infer_family(doc):
        declared = doc.get('family')
        if declared:
            return declared
        if doc.get('reference', {}).get('exact') is not None:
            return 'mms'
        for region in doc['regions']:
            shape = region['shape']
            if region.get('kind') == 'conductor':
                return 'cable'
            if shape['type'] == 'disk':
                source = region.get('source', 0.0)
                point = np.asarray(shape['center'], dtype=float).reshape(1, 2)
                if np.any(evaluate_field(source, point[:, 0], point[:, 1]) != 0):
                    return 'cable'
        return 'generic'

    def _make_region(self, index, entry):
        shape = entry['shape']
        if shape['type'] == 'rectangle':
            lower, upper = float(shape['bounds'][0]), float(shape['bounds'][1])
        else:
            lower, upper = 0.0, 1.0
        return GenericRegion(index, entry['id'], self._region_kind(entry), 0, lower, upper)

    def _region_kind(self, region):
        declared = region.get('kind')
        if declared is not None:
            return declared
        shape_type = region.get('shape', {}).get('type')
        if self.family == 'cable' and shape_type in ('disk', 'annulus'):
            return 'conductor'
        return 'soil'

    def _build_cable_layout(self):
        if self.family != 'cable':
            return ()
        grouped = {}
        for region_id, region in enumerate(self.doc['regions']):
            shape = region['shape']
            if shape['type'] == 'disk':
                inner, outer = 0.0, float(shape['radius_m'])
            elif shape['type'] == 'annulus':
                inner, outer = float(shape['inner_radius_m']), float(shape['outer_radius_m'])
            else:
                continue
            center = tuple(float(value) for value in shape['center'])
            grouped.setdefault(center, []).append((inner, outer, region_id))
        cables = []
        for cable_index, (center, entries) in enumerate(sorted(grouped.items())):
            entries.sort(key=lambda item: (item[0], item[1], item[2]))
            layers = tuple(GenericCableLayer(cable_index, region_id, center, inner, outer)
                           for inner, outer, region_id in entries)
            cables.append(GenericCable(cable_index, center, max(layer.outer_radius for layer in layers), layers))
        return tuple(cables)

    @property
    def supports_exact_cable_interfaces(self):
        if self.family != 'cable' or not self.cable_layout:
            return False
        shell_regions = {layer.region_id for cable in self.cable_layout for layer in cable.layers}
        full_bounds = {'type': 'rectangle', 'bounds': list(self.bounds)}
        backgrounds = [index for index, region in enumerate(self.doc['regions'])
                       if region['shape'] == full_bounds and index not in shell_regions]
        return len(backgrounds) == 1 and len(shell_regions) + len(backgrounds) == len(self.regions)

    def _exact_cable_areas(self):
        """Exact owned areas for separated cable shells in one rectangular background."""
        if self.family != 'cable' or not self.cable_layout:
            return {}
        layer_ids = {layer.region_id for cable in self.cable_layout for layer in cable.layers}
        backgrounds = [index for index in range(len(self.doc['regions'])) if index not in layer_ids]
        if len(backgrounds) != 1:
            return {}
        background_id = backgrounds[0]
        bounds = list(self.bounds)
        background_shape = self.doc['regions'][background_id]['shape']
        if background_shape != {'type': 'rectangle', 'bounds': bounds}:
            return {}
        for index, cable in enumerate(self.cable_layout):
            cx, cy = cable.center
            if min(cx - self.bounds[0], self.bounds[1] - cx,
                   cy - self.bounds[2], self.bounds[3] - cy) < cable.radius:
                return {}
            for other in self.cable_layout[index + 1:]:
                distance = math.hypot(cx - other.center[0], cy - other.center[1])
                if distance < cable.radius + other.radius:
                    return {}
        areas = {}
        for cable in self.cable_layout:
            for layer in cable.layers:
                areas[layer.region_id] = math.pi * (layer.outer_radius ** 2 - layer.inner_radius ** 2)
        domain_area = (self.bounds[1] - self.bounds[0]) * (self.bounds[3] - self.bounds[2])
        areas[background_id] = domain_area - sum(math.pi * cable.radius ** 2 for cable in self.cable_layout)
        return areas

    def _representative_point(self, region):
        """Return an interior point so thin regions missed by the raster still get physical fields."""
        shape = self.doc['regions'][region.id]['shape']
        if shape['type'] == 'rectangle':
            x0, x1, y0, y1 = shape['bounds']
            return np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=float)
        if shape['type'] == 'disk':
            return np.asarray(shape['center'], dtype=float)
        radius = math.sqrt((shape['inner_radius_m'] ** 2 + shape['outer_radius_m'] ** 2) / 2)
        return np.asarray(shape['center'], dtype=float) + np.array([radius, 0.])

    def _build_circular_interfaces(self):
        interfaces = []
        span = max(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 1.0)
        external_probe = max(NORMAL_OFFSET, 2 * MEMBERSHIP_TOLERANCE * span)
        for cable in self.cable_layout:
            for layer_index, layer in enumerate(cable.layers):
                right = (cable.layers[layer_index + 1].region_id
                         if layer_index + 1 < len(cable.layers) else None)
                if right is None:
                    probe = np.asarray(cable.center, dtype=float) + np.array([layer.outer_radius + external_probe, 0.])
                    right = int(self.owner_index(probe.reshape(1, 2))[0])
                if right < 0 or self.regions[right].kind == 'conductor' and right == layer.region_id:
                    raise ValueError(f'Cable layer {layer.region_id!r} has no external region')
                interfaces.append(GenericCircularInterface(
                    name=f'cable_{cable.index}_radius_{layer.outer_radius:g}',
                    left=layer.region_id, right=right, center=cable.center,
                    radius=layer.outer_radius, cable_index=cable.index))
        return tuple(interfaces)

    def _spec(self, region, name):
        entry = self.doc['regions'][region.id]
        if name == 'source':
            return entry.get('source', 0.0)
        return entry[name]

    def normalized_xy(self, xy):
        x0, x1, y0, y1 = self.bounds
        return torch.cat([2 * (xy[:, :1] - x0) / (x1 - x0) - 1, 2 * (xy[:, 1:2] - y0) / (y1 - y0) - 1], dim=1)

    def conductors(self):
        return [region.id for region in self.regions if region.kind == 'conductor']

    def owner_index(self, xy):
        """Index of the region owning each point of xy (N, 2); a later region overrides an earlier one."""
        return self.definition.owner(xy[:, 0], xy[:, 1])

    def domain_to_local(self, region, xy):
        """Canonical JSON shape map from physical coordinates to local normalized coordinates."""
        return self.definition.domain_to_local(region.id, xy)

    def local_to_domain(self, region, local):
        """Canonical JSON shape map from local normalized coordinates to physical coordinates."""
        return self.definition.local_to_domain(region.id, local)

    def input_size(self, region, partition):
        shape = self.doc['regions'][region.id]['shape']
        local_size = 3 if shape['type'] == 'annulus' else 2
        if partition == 'global':
            return 2 + 3 + len(self.regions) + len(self.source_disks)
        return local_size + (len(self.source_disks) if shape['type'] == 'rectangle' else 0)

    def _local_tensor_coordinates(self, region, xy):
        shape = self.doc['regions'][region.id]['shape']
        if shape['type'] == 'rectangle':
            x0, x1, y0, y1 = shape['bounds']
            return torch.cat([2 * (xy[:, 0:1] - x0) / (x1 - x0) - 1,
                              2 * (xy[:, 1:2] - y0) / (y1 - y0) - 1], dim=1)
        center = xy.new_tensor(shape['center'])
        delta = xy[:, :2] - center
        radius = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(torch.finfo(xy.dtype).tiny)
        if shape['type'] == 'disk':
            return delta / shape['radius_m']
        radial = 2 * (radius - shape['inner_radius_m']) / (
            shape['outer_radius_m'] - shape['inner_radius_m']) - 1
        return torch.cat([radial, delta / radius], dim=1)

    def _source_distance_features(self, xy, bounds):
        x0, x1, y0, y1 = bounds
        corners = xy.new_tensor([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
        epsilon = torch.finfo(xy.dtype).eps
        features = []
        for disk in self.source_disks:
            center = xy.new_tensor(disk['center'])
            maximum = torch.linalg.vector_norm(corners - center, dim=1).max()
            denominator = torch.log((maximum / disk['radius_m']).clamp_min(1 + epsilon))
            distance = torch.linalg.vector_norm(xy[:, :2] - center, dim=1, keepdim=True)
            log_distance = torch.log((distance / disk['radius_m']).clamp_min(epsilon))
            features.append((2 * log_distance / denominator.clamp_min(epsilon) - 1).clamp(-1, 1))
        return features

    def coordinates(self, region, xy, partition):
        local = self._local_tensor_coordinates(region, xy)
        if partition == 'global':
            if local.shape[1] < 3:
                local = torch.cat([local, xy.new_zeros((len(xy), 3 - local.shape[1]))], dim=1)
            region_code = xy.new_zeros((len(xy), len(self.regions)))
            region_code[:, region.id] = 1.
            return torch.cat([self.normalized_xy(xy), local, region_code,
                              *self._source_distance_features(xy, self.bounds)], dim=1)
        shape = self.doc['regions'][region.id]['shape']
        if shape['type'] == 'rectangle':
            x0, x1, y0, y1 = shape['bounds']
            return torch.cat([local, *self._source_distance_features(xy, (x0, x1, y0, y1))], dim=1)
        return local

    def unit_coordinates(self, region, xy, partition):
        return (self.coordinates(region, xy, partition) + 1) / 2

    def planar_cuts(self):
        cuts = []
        x0, x1, y0, y1 = self.bounds
        for region in self.doc['regions']:
            shape = region['shape']
            if shape['type'] == 'disk':
                cx, cy = shape['center']
                cuts.extend(((0, cx - shape['radius_m']), (0, cx + shape['radius_m']),
                             (1, cy - shape['radius_m']), (1, cy + shape['radius_m'])))
            elif shape['type'] == 'annulus':
                cx, cy = shape['center']
                for radius in (shape['inner_radius_m'], shape['outer_radius_m']):
                    cuts.extend(((0, cx - radius), (0, cx + radius),
                                 (1, cy - radius), (1, cy + radius)))
            elif shape['bounds'] != [x0, x1, y0, y1]:
                a0, a1, b0, b1 = shape['bounds']
                cuts.extend(((0, a0), (0, a1), (1, b0), (1, b1)))
        cuts.extend((0 if item['axis'] == 'x' else 1, float(item['position'])) for item in self.doc.get('interfaces', ())
                    if item['regions'][0] == item['regions'][1])
        return tuple(sorted(set((axis, float(position)) for axis, position in cuts
                                if self.bounds[axis * 2] < position < self.bounds[axis * 2 + 1])))

    def _level_set_specs(self):
        specs = []
        x0, x1, y0, y1 = self.bounds
        for region in self.doc['regions']:
            shape = region['shape']
            if shape['type'] == 'disk':
                specs.append(('circle', tuple(shape['center']), float(shape['radius_m'])))
            elif shape['type'] == 'annulus':
                specs.extend(('circle', tuple(shape['center']), float(radius))
                              for radius in (shape['inner_radius_m'], shape['outer_radius_m']))
            elif shape['bounds'] != [x0, x1, y0, y1]:
                a0, a1, b0, b1 = shape['bounds']
                specs.extend((('x_left', float(a0)), ('x_right', float(a1)),
                              ('y_bottom', float(b0)), ('y_top', float(b1))))
        specs.extend(('plane', 0 if item['axis'] == 'x' else 1, float(item['position']))
                     for item in self.doc.get('interfaces', ()) if item['regions'][0] == item['regions'][1])
        return tuple(specs)

    def level_sets(self, xy):
        """Signed geometric features shared by CUSP and interface-aware variants."""
        columns = []
        for spec in self._level_set_specs():
            kind = spec[0]
            if kind == 'circle':
                columns.append(torch.linalg.vector_norm(xy[:, :2] - xy.new_tensor(spec[1]), dim=1, keepdim=True) - spec[2])
            elif kind == 'x_left':
                columns.append(xy[:, 0:1] - spec[1])
            elif kind == 'x_right':
                columns.append(spec[1] - xy[:, 0:1])
            elif kind == 'y_bottom':
                columns.append(xy[:, 1:2] - spec[1])
            elif kind == 'plane':
                columns.append(xy[:, spec[1]:spec[1] + 1] - spec[2])
            else:
                columns.append(spec[1] - xy[:, 1:2])
        return torch.cat(columns, dim=1) if columns else xy.new_zeros((len(xy), 0))

    def _clear_membership(self, region, points):
        """Interior points must be owned and separated from the selected region boundary."""
        owner = self.owner_index(points)
        shape = self.doc['regions'][region.id]['shape']
        span = max(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 1.0)
        margin = interior_margin(shape, span, INTERFACE_MARGIN_FACTOR)
        distance = boundary_distance(shape, points[:, 0], points[:, 1])
        return (owner == region.id) & (distance > margin)

    def _grid(self):
        x0, x1, y0, y1 = self.bounds
        x, y = np.meshgrid(np.linspace(x0, x1, GRID[0]), np.linspace(y0, y1, GRID[1]))
        cell_area = (x1 - x0) * (y1 - y0) / (GRID[0] * GRID[1])
        return np.c_[x.ravel(), y.ravel()], cell_area

    def _domain_sides(self, n):
        if self.doc['domain']['shape'] == 'annulus':
            center = np.asarray(self.doc['domain']['center'], dtype=float)
            angle = (np.arange(n) + .5) * 2 * np.pi / n
            circle = np.c_[np.cos(angle), np.sin(angle)]
            inner = self.doc['domain']['inner_radius_m']
            outer = self.doc['domain']['outer_radius_m']
            return {
                'inner': dict(xy=center + inner * circle, normal=-circle, length=2 * np.pi * inner),
                'outer': dict(xy=center + outer * circle, normal=circle, length=2 * np.pi * outer),
            }
        x0, x1, y0, y1 = self.bounds
        s = (np.arange(n) + .5) / n
        ones = np.ones(n)
        return {
            'bottom': dict(xy=np.c_[x0 + (x1 - x0) * s, y0 * ones], normal=np.tile(SIDE_NORMALS['bottom'], (n, 1)), length=x1 - x0),
            'top': dict(xy=np.c_[x0 + (x1 - x0) * s, y1 * ones], normal=np.tile(SIDE_NORMALS['top'], (n, 1)), length=x1 - x0),
            'left': dict(xy=np.c_[x0 * ones, y0 + (y1 - y0) * s], normal=np.tile(SIDE_NORMALS['left'], (n, 1)), length=y1 - y0),
            'right': dict(xy=np.c_[x1 * ones, y0 + (y1 - y0) * s], normal=np.tile(SIDE_NORMALS['right'], (n, 1)), length=y1 - y0),
        }

    def _bc(self, condition):
        if condition['type'] == 'dirichlet':
            return dict(type='dirichlet', value=lambda xy, spec=condition['value']: torch_field(spec, xy))
        if condition['type'] == 'neumann':
            return dict(type='neumann', value=lambda xy, spec=condition['q_n_W_m2']: torch_field(spec, xy))
        return dict(type='robin', h=condition['h_W_m2K'],
                    T_inf=lambda xy, spec=condition['T_inf_C']: torch_field(spec, xy))

    def _reference_temperature(self):
        """Lowest boundary reference temperature, so that the offset is never negative."""
        levels = [v for v in (self._reference_on_sides(side) for side in self.doc['boundaries']) if v is not None]
        return float(min(np.min(values) for values in levels))

    def _temperature_scale(self):
        if 'temperature_scale_K' in self.doc:
            return float(self.doc['temperature_scale_K'])
        levels = [v for v in (self._reference_on_sides(side) for side in self.doc['boundaries']) if v is not None]
        return float(max(1., max(np.max(values) - self.T_ref for values in levels)))

    def _reference_on_sides(self, side):
        """Reference temperatures of one side (Dirichlet value or Robin ambient); None for Neumann sides."""
        condition = self.doc['boundaries'][side]
        spec = condition.get('value') if condition['type'] == 'dirichlet' else condition.get('T_inf_C')
        if spec is None:
            return None
        xy = self._domain_sides(33)[side]['xy']
        return evaluate_field(spec, xy[:, 0], xy[:, 1])

    # ----- surface used by the objective -----
    def conductivity(self, region, xy):
        return torch_field(self._spec(region, 'conductivity'), xy)

    def source(self, region, xy):
        return torch_field(self._spec(region, 'source'), xy)

    def scales(self, region):
        """Region scales with L = sqrt(area); the temperature is the local drop q L^2 / k, or T_scale without source."""
        k = self.k_ref[region.id]
        length = max(np.sqrt(self.areas[region.id]), 1e-12)
        source = self.source_abs[region.id]
        local = source * length ** 2 / k if source > 0 else self.T_scale
        return dict(temperature=self.T_scale, local_temperature=local,
            flux=k * local / length, divergence=k * local / length ** 2)

    def amplitude(self, region, partition):
        """Use V2's regional thermal amplitude for sourced regions and the soil scale elsewhere."""
        if self.source_abs[region.id] > 0 or region.kind != 'soil':
            return self.scales(region)['local_temperature']
        return self.T_scale

    def interface_flux_scale(self, face):
        """Scale a transmission jump by expected power per interface length when available."""
        length = max(face.weight * len(face.xy), 1e-12)
        power = max(abs(self.region_power[face.left]), abs(self.region_power[face.right]))
        if power > 0:
            return max(power / length, 1e-12)
        left = self.scales(self.regions[face.left])['flux']
        right = self.scales(self.regions[face.right])['flux']
        return max(left, right, 1e-12)

    def contains_points(self, region, points):
        return self.owner_index(points) == region.id

    def focused_interior_pool(self, region, n, seed, policy='mixed'):
        """Build a V2-style pool: half uniform and half log-radial near source disks."""
        uniform_count = n // 2
        uniform = self.sample_interior(region, uniform_count, seed, policy)
        source_disks = []
        for entry in self.doc['regions']:
            shape = entry['shape']
            if shape['type'] == 'disk' and np.any(evaluate_field(entry.get('source', 0.0),
                    np.array([shape['center'][0]]), np.array([shape['center'][1]])) != 0):
                source_disks.append(shape)
        if not source_disks or self.doc['regions'][region.id]['shape']['type'] != 'rectangle':
            return np.vstack([uniform, self.sample_interior(region, n - uniform_count, seed + 1, policy)])
        rng = np.random.default_rng(seed + 2)
        x0, x1, y0, y1 = self.bounds
        radius_max = max(min(x1 - x0, y1 - y0) * 0.5, 4 * max(d['radius_m'] for d in source_disks))
        focused = []
        accepted = 0
        target = n - uniform_count
        for _ in range(MAX_ATTEMPTS):
            count = max(256, target * 2)
            disk_indices = rng.integers(0, len(source_disks), count)
            disks = [source_disks[index] for index in disk_indices]
            radius = np.array([disk['radius_m'] for disk in disks])
            radius *= np.exp(rng.uniform(1e-6, np.log(radius_max / radius)))
            angle = rng.uniform(0., 2 * np.pi, count)
            candidate = np.c_[np.array([disk['center'][0] for disk in disks]) + radius * np.cos(angle),
                               np.array([disk['center'][1] for disk in disks]) + radius * np.sin(angle)]
            mine = candidate[self._clear_membership(region, candidate)]
            focused.append(mine)
            accepted += len(mine)
            if accepted >= target:
                return np.vstack([uniform, np.vstack(focused)[:target]])
        return np.vstack([uniform, self.sample_interior(region, target, seed + 3, policy)])

    def sample_interior(self, region, n, seed, policy='uniform'):
        """Draw region-local samples through the canonical problem geometry API."""
        if policy == 'paired':
            raise NotImplementedError('paired sampling is not defined for region documents')
        return self.definition.sample_region_interior(region.id, n, np.random.default_rng(seed))

    def faces(self, n):
        """Adapt canonical owner-local JSON interfaces to the PINN surface type."""
        return [Face(interface['name'], interface['left'], interface['right'], interface['xy'],
                     interface['normal'], interface['weight'], xy_right=interface['xy_right'])
                for interface in self.definition.sample_interfaces(n)]

    def pieces(self, n):
        """Boundary samples use the canonical owner-local segments from the JSON problem."""
        result = []
        for sample in self.definition.sample_boundary_regions(n):
            condition = self._bc(self.doc['boundaries'][sample['side']])
            count = len(sample['xy'])
            if count:
                result.append(Piece(sample['side'], sample['region'], sample['xy'], sample['normal'],
                                    sample['length'] / count, condition))
        return result

    def reference(self, xy, labels=None):
        spec = self.doc.get('reference', {}).get('exact')
        return None if spec is None else evaluate_field(spec, xy[:, 0], xy[:, 1])

    def specification(self):
        return dict(kind='region_document', id=self.name)

    def area(self, region):
        return self.areas[region.id]

    def evaluation(self, per_region=1000, seed=20261024):
        """Deterministic interior points of every region with area weights, for evaluation only."""
        xy, labels, weights = [], [], []
        for region in self.regions:
            points = self.sample_interior(region, per_region, seed + region.id)
            xy.append(points)
            labels.append(np.full(per_region, region.id))
            weights.append(np.full(per_region, self.areas[region.id] / per_region))
        return np.vstack(xy), np.concatenate(labels), np.concatenate(weights)
