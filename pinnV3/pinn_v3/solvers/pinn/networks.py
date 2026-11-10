"""M0, M1 and M2 behind one class.

partition  global: one network for the whole domain   | region: one network per material or subdomain
formulation strong: output T, flux from -k grad T       | mixed: output (T, qx, qy), flux is a separate field

  M0 = (global, strong)   control PINN, strong form
  M1 = (region, strong)   per-material PINN, interface conditions as losses
  M2 = (region, mixed)    first-order system T, q with the constitutive law as a loss

No boundary condition is built into the network (no hard envelope): Dirichlet, Neumann and Robin
are all soft losses, so the three kinds are treated alike. The output is T_ref + amplitude * raw.
"""
import math

import torch
from torch import nn
from pinn_v3.problem.build import SIDE_NORMALS

ARCHITECTURES = dict(M0=('global', 'strong'), M1=('region', 'strong'), M2=('region', 'mixed'))
EXACT_CORRECTORS = dict(M0_EXACT='M0', CUSP_EXACT='CUSP', DCSNN_EXACT='DCSNN', FBPINN_EXACT='FBPINN')
DEEPXDE_EXACT_CORRECTORS = dict(DeepXDE_PFNN_EXACT='PFNN', DeepXDE_FNN_EXACT='FNN')
GENERIC_EXACT_ARCHITECTURES = dict(M0_EXACT_BC='M0', M1_EXACT_BC='M1', DCSNN_EXACT_BC='DCSNN')


def mlp(inputs, width, depth, outputs, activation='tanh'):
    act = dict(tanh=nn.Tanh, silu=nn.SiLU)[activation]
    layers, size = [], inputs
    for _ in range(depth):
        linear = nn.Linear(size, width)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend([linear, act()])
        size = width
    last = nn.Linear(size, outputs)
    nn.init.xavier_normal_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class PINNModel(nn.Module):
    kind = 'baseline'

    def __init__(self, problem, architecture='M1', width=32, depth=3, activation='tanh'):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f'Unknown architecture {architecture!r}; expected one of {sorted(ARCHITECTURES)}')
        self._setup(problem, architecture, width, depth, *ARCHITECTURES[architecture])
        representatives = problem.regions[:1] if self.partition == 'global' else problem.regions
        self.networks = nn.ModuleList()
        self.offsets = nn.ParameterList()
        for region in representatives:
            self.networks.append(mlp(problem.input_size(region, self.partition), width, depth, 3 if self.mixed else 1, activation))
            self.offsets.append(nn.Parameter(torch.zeros(())))

    def _setup(self, problem, architecture, width, depth, partition, formulation):
        self.problem = problem
        self.architecture = architecture
        self.partition, self.formulation = partition, formulation
        self.mixed = formulation == 'mixed'
        self.width, self.depth = width, depth
        self.input_normalization = 'region_coordinates'

    def input_coordinates(self, region, xy):
        if self.input_normalization == 'unit_box_v1':
            return self.problem.unit_coordinates(region, xy, self.partition)
        return self.problem.coordinates(region, xy, self.partition)

    def field(self, region, xy):
        """Return (T, q) for a region; q is None in the strong formulation."""
        index = 0 if self.partition == 'global' else region.id
        raw = self.networks[index](self.input_coordinates(region, xy))
        scales = self.problem.scales(region)
        amplitude = self.problem.amplitude(region, self.partition)
        temperature = self.problem.T_ref + self.problem.T_scale * self.offsets[index] + amplitude * raw[:, :1]
        flux = raw[:, 1:3] * scales['flux'] if self.mixed else None
        return temperature, flux

    def temperature(self, xy, labels):
        result = torch.empty((len(xy), 1), dtype=xy.dtype, device=xy.device)
        for region in self.problem.regions:
            mask = labels == region.id
            if mask.any():
                result[mask] = self.field(region, xy[mask])[0]
        return result

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _free_correction(self, region, xy, coordinates):
        if hasattr(self, 'correction_model'):
            temperature, _ = self.correction_model.field(region, xy)
            return (temperature - self.problem.T_ref) / self.problem.T_scale
        return self.networks[region.id](coordinates)[:, :1] + self.offsets[region.id]


class DeepXDECorrector(nn.Module):
    def __init__(self, problem, architecture, width, depth, activation):
        super().__init__()
        import deepxde as dde
        from pinn_v3.solvers.pinn.deepxde_solver import _network
        if dde.backend.backend_name != 'pytorch':
            raise ValueError('Exact DeepXDE correctors require the PyTorch backend')
        self.problem = problem
        self.network = _network(dde, problem, architecture, width, depth, activation,
            input_normalization='unit_box_v1')

    def field(self, region, xy):
        selector = xy.new_full((len(xy), 1), float(region.id))
        raw = self.network(torch.cat((xy, selector), dim=1))[:, region.id:region.id + 1]
        raw = raw + self.network.temperature_offsets[region.id]
        return self.problem.T_ref + self.problem.T_scale * raw, None


class ExactInterfacePINN(PINNModel):
    """M1 field with exact constant outer Dirichlet and circular transmission traces."""
    kind = 'exact_interfaces'
    exact_constraints = True
    fourier_order = 8
    INPUT_NORMALIZATIONS = ('region_coordinates', 'case_log_distance_minus1_plus1_v1', 'unit_box_v1')

    def __init__(self, problem, width=32, depth=3, activation='tanh', input_normalization='case_log_distance_minus1_plus1_v1',
        trace_parameterization='standard', soil_envelope_gain=1.):
        from pinn_v3.solvers.pinn.generic_problem import GenericProblem
        if not isinstance(problem, GenericProblem) or not problem.cable_layout:
            raise ValueError('M1_EXACT requires a generic V3 cable problem')
        outer_temperature = self._outer_reference_temperature(problem)
        super().__init__(problem, 'M1', width, depth, activation)
        self.architecture = 'M1_EXACT'
        if input_normalization not in self.INPUT_NORMALIZATIONS:
            raise ValueError(f'Unsupported M1_EXACT input normalization {input_normalization!r}')
        self.input_normalization = input_normalization
        if trace_parameterization != 'standard' or soil_envelope_gain != 1.:
            raise ValueError('Cable exact engine has one construction: standard traces and unit soil envelope gain')
        self.boundary_mode = 'mms'
        self.boundary_pieces = {}
        for piece in problem.pieces(16):
            previous = self.boundary_pieces.get(piece.side)
            if previous is not None and previous.bc['type'] != piece.bc['type']:
                raise ValueError(f'Exact cable boundary requires one BC type per side: {piece.side}')
            self.boundary_pieces.setdefault(piece.side, piece)
        self.boundary_trace_networks = nn.ModuleDict({
            side: mlp(1, width, depth, 1, activation) for side in self.boundary_pieces
        })
        self.cable_layout = tuple(problem.cable_layout)
        centers = torch.as_tensor([cable.center for cable in self.cable_layout], dtype=torch.float32)
        radii = torch.as_tensor([cable.radius for cable in self.cable_layout], dtype=torch.float32)
        x0, x1, y0, y1 = problem.bounds
        corners = centers.new_tensor([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
        maximum = torch.linalg.vector_norm(corners[None, :, :] - centers[:, None, :], dim=2).max(dim=1).values
        maximum_log = torch.log(maximum / radii).clamp_min(torch.finfo(centers.dtype).eps)
        mode_orders = torch.tensor([0] + [order for order in range(1, self.fourier_order + 1)
                           for _ in range(2)], dtype=torch.float32)
        self.register_buffer('_cable_centers', centers, persistent=False)
        self.register_buffer('_cable_radii', radii, persistent=False)
        self.register_buffer('_cable_max_log_distance', maximum_log, persistent=False)
        self.register_buffer('_angular_orders', torch.arange(1, self.fourier_order + 1, dtype=torch.float32),
                     persistent=False)
        self.register_buffer('_mode_orders', mode_orders, persistent=False)
        self._layers_by_region = {
            layer.region_id: layer
            for cable in self.cable_layout
            for layer in cable.layers
        }
        interfaces = problem.circular_interfaces
        self.interface_index = {interface.name: index for index, interface in enumerate(interfaces)}
        self.interfaces = tuple(interfaces)
        self.trace_coefficients = nn.Parameter(torch.zeros(len(interfaces), 2, 2 * self.fourier_order + 1))
        self.interface_flux_scales = tuple(
            max(problem.scales(problem.regions[item.left])['flux'],
                problem.scales(problem.regions[item.right])['flux'])
            for item in interfaces)
        self.outer_interfaces = tuple(
            item for item in interfaces if problem.regions[item.right].kind == 'soil')
        gaps = []
        for index, left in enumerate(self.outer_interfaces):
            left_center = left.center
            for right in self.outer_interfaces[index + 1:]:
                right_center = right.center
                center_distance = ((left_center[0] - right_center[0]) ** 2 +
                    (left_center[1] - right_center[1]) ** 2) ** .5
                gaps.append(center_distance - left.radius - right.radius)
        x0, x1, y0, y1 = problem.bounds
        for interface in self.outer_interfaces:
            cx, cy = interface.center
            gaps.extend((cx - x0 - interface.radius, x1 - cx - interface.radius,
                cy - y0 - interface.radius, y1 - cy - interface.radius))
        if len(self.outer_interfaces) != len(self.cable_layout) or not gaps or min(gaps) <= 0:
            raise ValueError('M1_EXACT requires separated cable interfaces inside the rectangular soil domain')
        self.collar_width = min(gaps) * .25
        self.outer_temperature = outer_temperature
        self._interfaces_by_cable = {
            cable.index: tuple(sorted((item for item in interfaces if item.cable_index == cable.index),
                key=lambda item: item.radius))
            for cable in self.cable_layout
        }

    @staticmethod
    def _outer_reference_temperature(problem):
        if problem.doc.get('domain', {}).get('shape') != 'rectangle' or not problem.cable_layout:
            raise ValueError('M1_EXACT requires a rectangular generic V3 cable problem')
        pieces = problem.pieces(16)
        if not pieces or any(piece.bc['type'] not in ('dirichlet', 'neumann', 'robin') for piece in pieces):
            raise ValueError('M1_EXACT requires Dirichlet, Neumann or Robin data on every outer side')
        return float(problem.T_ref)

    def _boundary_features(self, side, xy):
        return GenericBoundaryPINN._boundary_features(self, side, xy)

    def _boundary_trace(self, side, xy, region):
        return GenericBoundaryPINN._boundary_trace(self, side, xy, region)

    def _free_boundary_temperature(self, side, xy, learned):
        return GenericBoundaryPINN._free_boundary_temperature(self, side, xy, learned)

    @staticmethod
    def _hermite(coordinate, lower, upper, value_lower, slope_lower, value_upper, slope_upper):
        return GenericBoundaryPINN._hermite(
            coordinate, lower, upper, value_lower, slope_lower, value_upper, slope_upper)

    def _mms_end_data(self, region, high, xy):
        return GenericBoundaryPINN._mms_end_data(self, region, high, xy)

    def _mms_axis_envelopes(self, region, coordinate):
        return GenericBoundaryPINN._mms_axis_envelopes(self, region, coordinate)

    def _mms_strip_lifting(self, region, xy):
        return GenericBoundaryPINN._mms_strip_lifting(self, region, xy)

    def _interface_center(self, interface):
        return self.trace_coefficients.new_tensor(interface.center)

    def _angular_basis(self, xy, center):
        delta = xy[:, :2] - center
        angle = torch.atan2(delta[:, 1:2], delta[:, 0:1])
        orders = self._angular_orders.to(dtype=xy.dtype)
        harmonics = angle * orders.reshape(1, -1)
        paired = torch.stack((torch.cos(harmonics), torch.sin(harmonics)), dim=2).flatten(1)
        return torch.cat((torch.ones_like(angle), paired), dim=1)

    def _trace(self, interface, xy):
        index = self.interface_index[interface.name]
        basis = self._angular_basis(xy, self._interface_center(interface))
        coefficients = self.trace_coefficients[index]
        temperature = self.outer_temperature + basis @ self._temperature_coefficients(interface)[:, None]
        flux = basis @ self._flux_coefficients(interface)[:, None]
        return temperature, flux

    def _flux_coefficients(self, interface):
        index = self.interface_index[interface.name]
        return self.interface_flux_scales[index] * self.trace_coefficients[index, 1]

    def _temperature_coefficients(self, interface):
        index = self.interface_index[interface.name]
        return self.problem.T_scale * self.trace_coefficients[index, 0]

    def _disk_lifting(self, interface, xy, region):
        index = self.interface_index[interface.name]
        center = self._interface_center(interface)
        normalized = (xy[:, :2] - center) / interface.radius
        radial_squared = normalized.square().sum(dim=1, keepdim=True)
        real, imag = torch.ones_like(radial_squared), torch.zeros_like(radial_squared)
        harmonics = [real]
        for _ in range(self.fourier_order):
            real, imag = real * normalized[:, :1] - imag * normalized[:, 1:2], real * normalized[:, 1:2] + imag * normalized[:, :1]
            harmonics.extend((real, imag))
        coefficients = self.trace_coefficients[index]
        temperature_coefficients = self._temperature_coefficients(interface)
        temperature_coefficients = temperature_coefficients + torch.nn.functional.pad(
            temperature_coefficients.new_tensor([self.outer_temperature]),
            (0, 2 * self.fourier_order))
        flux_coefficients = self._flux_coefficients(interface)
        conductivity = self.problem.conductivity(region, xy)
        orders = self._mode_orders.to(dtype=xy.dtype).reshape(1, -1)
        temperature_trace = temperature_coefficients.reshape(1, -1)
        radial_derivative = -flux_coefficients.reshape(1, -1) / conductivity
        correction = (interface.radius * radial_derivative - orders * temperature_trace) / 2
        modes = temperature_trace + correction * (radial_squared - 1)
        return torch.sum(torch.cat(harmonics, dim=1) * modes, dim=1, keepdim=True)

    def _annulus_lifting(self, inner, outer, xy, region):
        center = self._interface_center(outer)
        radius = torch.linalg.vector_norm(xy[:, :2] - center, dim=1, keepdim=True)
        length = outer.radius - inner.radius
        coordinate = (radius - inner.radius) / length
        g0, q0 = self._trace(inner, xy)
        g1, q1 = self._trace(outer, xy)
        conductivity = self.problem.conductivity(region, xy)
        p0, p1 = -q0 / conductivity, -q1 / conductivity
        s2, s3 = coordinate.square(), coordinate.pow(3)
        h00, h10 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + coordinate
        h01, h11 = -2 * s3 + 3 * s2, s3 - s2
        return h00 * g0 + h10 * length * p0 + h01 * g1 + h11 * length * p1

    def _coordinates(self, region, xy):
        if self.input_normalization == 'unit_box_v1':
            return self.problem.unit_coordinates(region, xy, 'region')
        if region.kind != 'soil' or self.input_normalization == 'region_coordinates':
            return self.problem.coordinates(region, xy, 'region')
        coordinates = [self.problem.normalized_xy(xy)]
        centers = self._cable_centers.to(dtype=xy.dtype)
        radii = self._cable_radii.to(dtype=xy.dtype)
        max_log_distance = self._cable_max_log_distance.to(dtype=xy.dtype)
        distance = torch.linalg.vector_norm(xy[:, None, :2] - centers[None, :, :], dim=2)
        log_distance = torch.log((distance / radii[None, :]).clamp_min(1.))
        normalized = (2 * log_distance / max_log_distance[None, :] - 1).clamp(-1, 1)
        coordinates.append(normalized)
        return torch.cat(coordinates, dim=1)

    def field(self, region, xy):
        """Return temperature and no separate flux; traces impose T and q.n exactly."""
        raw = self._free_correction(region, xy, self._coordinates(region, xy))
        amplitude = self.problem.amplitude(region, 'region')
        if region.kind == 'soil':
            with torch.enable_grad():
                points = xy if xy.requires_grad else xy.detach().requires_grad_(True)
                lifting, correction, trace_envelope, network_envelope, tangent_envelope = self._mms_strip_lifting(region, points)
                base = lifting + trace_envelope * correction
            temperature = base
            interface_factors = torch.ones_like(temperature)
            for interface in self.outer_interfaces:
                center = self._interface_center(interface)
                delta = points[:, :2] - center
                radius = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(torch.finfo(points.dtype).tiny)
                distance = radius - interface.radius
                g, q_radial = self._trace(interface, points)
                conductivity = self.problem.conductivity(region, points)
                radial_derivative = -q_radial / conductivity
                scaled_distance = (distance / self.collar_width).clamp(0, 1)
                cutoff = 1 - 10 * scaled_distance.pow(3) + 15 * scaled_distance.pow(4) - 6 * scaled_distance.pow(5)
                temperature = temperature + cutoff * (g - base + radial_derivative * distance)
                interface_factors = interface_factors * (distance / (distance + self.collar_width)).square()
            envelope = network_envelope * tangent_envelope * interface_factors
            return temperature + envelope * amplitude * raw, None

        layer = self._layers_by_region.get(region.id)
        if layer is None:
            raise ValueError(f'M1_EXACT cannot classify region {region.name!r} from generic cable geometry')
        interfaces = self._interfaces_by_cable[layer.cable_index]
        outer = next(item for item in interfaces if abs(item.radius - layer.outer_radius) < 1e-12)
        if layer.inner_radius == 0.:
            lifting = self._disk_lifting(outer, xy, region)
            center = self._interface_center(outer)
            radius_squared = (xy[:, :2] - center).square().sum(dim=1, keepdim=True)
            envelope = ((radius_squared - layer.outer_radius ** 2) / layer.outer_radius ** 2).square()
        else:
            inner = next(item for item in interfaces if abs(item.radius - layer.inner_radius) < 1e-12)
            lifting = self._annulus_lifting(inner, outer, xy, region)
            radius = torch.linalg.vector_norm(xy[:, :2] - self._interface_center(outer), dim=1, keepdim=True)
            length = layer.outer_radius - layer.inner_radius
            envelope = (((radius - layer.inner_radius) * (radius - layer.outer_radius) / length ** 2) * 4).square()
        return lifting + envelope * amplitude * raw, None


class GenericBoundaryPINN(nn.Module):
    """Hard rectangular boundary conditions for the generic region document."""
    kind = 'generic_exact_boundary'
    exact_constraints = True
    exact_boundary = True

    def __init__(self, problem, architecture, width=32, depth=3, activation='tanh'):
        from pinn_v3.solvers.pinn.generic_problem import GenericProblem
        if not isinstance(problem, GenericProblem):
            raise ValueError('Generic exact boundaries require a pinn-v3-problem-1 document')
        if problem.doc['domain']['shape'] not in ('rectangle', 'annulus'):
            raise ValueError('Generic exact boundaries require a rectangular or annular domain')
        super().__init__()
        self.problem = problem
        self.boundary_mode = problem.doc['domain']['shape']
        base_architecture = 'M1' if architecture == 'M1_EXACT' else architecture
        self.base_model = build_model(problem, base_architecture, width, depth, activation)
        if self.base_model.mixed:
            raise ValueError('Generic exact boundaries currently require a strong-temperature network')
        self.mixed = False
        self.input_normalization = 'unit_box_v1'
        self.architecture = architecture if architecture == 'M1_EXACT' else f'{architecture}_EXACT_BC'
        self.width, self.depth = width, depth
        self.boundary_pieces = {}
        for piece in problem.pieces(16):
            previous = self.boundary_pieces.get(piece.side)
            if previous is not None and previous.bc['type'] != piece.bc['type']:
                raise ValueError(f'Exact generic boundary engine requires one BC type per side: {piece.side}')
            self.boundary_pieces.setdefault(piece.side, piece)
        feature_count = 2 if self.boundary_mode == 'annulus' else 1
        self.boundary_trace_networks = nn.ModuleDict({
            side: mlp(feature_count, width, depth, 1, activation)
            for side in self.boundary_pieces
        })

    def input_coordinates(self, region, xy):
        if self.input_normalization == 'unit_box_v1':
            return self.problem.unit_coordinates(region, xy, 'region')
        return self.problem.coordinates(region, xy, 'region')

    def _free_correction(self, region, xy, coordinates):
        return self.base_model.networks[region.id](coordinates)[:, :1] + self.base_model.offsets[region.id]

    def _distances(self, xy):
        x0, x1, y0, y1 = self.problem.bounds
        return dict(left=xy[:, 0:1] - x0, right=x1 - xy[:, 0:1],
                    bottom=xy[:, 1:2] - y0, top=y1 - xy[:, 1:2])

    def _normalized_distances(self, xy):
        x0, x1, y0, y1 = self.problem.bounds
        return dict(left=(xy[:, 0:1] - x0) / (x1 - x0), right=(x1 - xy[:, 0:1]) / (x1 - x0),
                    bottom=(xy[:, 1:2] - y0) / (y1 - y0), top=(y1 - xy[:, 1:2]) / (y1 - y0))

    def _project(self, side, xy):
        x0, x1, y0, y1 = self.problem.bounds
        x, y = xy[:, 0:1], xy[:, 1:2]
        if side == 'left':
            x = x * 0 + x0
        elif side == 'right':
            x = x * 0 + x1
        elif side == 'bottom':
            y = y * 0 + y0
        else:
            y = y * 0 + y1
        return torch.cat((x, y), dim=1)

    def _field_value(self, side, name, xy):
        from pinn_v3.solvers.pinn.generic_problem import torch_field
        return torch_field(self.problem.doc['boundaries'][side][name], xy)

    def _base_field(self, region, xy):
        raw, _ = self.base_model.field(region, xy)
        distances = self._normalized_distances(xy)
        dirichlet = [side for side, bc in self.problem.doc['boundaries'].items()
                     if bc['type'] == 'dirichlet']
        if not dirichlet:
            return raw
        weights = {}
        denominator = xy.new_zeros((len(xy), 1))
        for side in dirichlet:
            weight = xy.new_ones((len(xy), 1))
            for other in dirichlet:
                if other != side:
                    weight = weight * distances[other]
            weights[side] = weight
            denominator = denominator + weight
        lifting = xy.new_zeros((len(xy), 1))
        safe = denominator.clamp_min(torch.finfo(xy.dtype).eps)
        for side in dirichlet:
            lifting = lifting + (weights[side] / safe) * self._field_value(side, 'value', self._project(side, xy))
        envelope = xy.new_ones((len(xy), 1))
        for side in dirichlet:
            envelope = envelope * distances[side]
        return lifting + envelope * (raw - self.problem.T_ref)

    def _normal_bump(self, side, xy, correction):
        distances = self._distances(xy)
        signed = dict(left=-distances['left'], right=-distances['right'],
                      bottom=-distances['bottom'], top=-distances['top'])[side]
        factor = xy.new_ones((len(xy), 1))
        for other in distances:
            if other != side:
                factor = factor * distances[other].square()
        projected = self._project(side, xy)
        projected_distances = self._distances(projected)
        boundary_factor = projected_distances[next(name for name in distances if name != side)].square()
        for other in distances:
            if other != side and other != next(name for name in distances if name != side):
                boundary_factor = boundary_factor * projected_distances[other].square()
        return signed * factor * correction / boundary_factor.clamp_min(torch.finfo(xy.dtype).eps)

    def field(self, region, xy):
        if self.boundary_mode == 'annulus':
            return self._annulus_field(region, xy)
        temperature = self._base_field(region, xy)
        for side, bc in self.problem.doc['boundaries'].items():
            if bc['type'] == 'dirichlet':
                continue
            projected = self._project(side, xy).detach().requires_grad_(True)
            with torch.enable_grad():
                base_boundary = self._base_field(region, projected)
                gradient = torch.autograd.grad(base_boundary, projected, torch.ones_like(base_boundary),
                    create_graph=True, retain_graph=True)[0]
            normal = projected.new_tensor(SIDE_NORMALS[side]).reshape(1, 2)
            normal_derivative = (gradient[:, :2] * normal).sum(1, keepdim=True)
            conductivity = self.problem.conductivity(region, projected)
            if bc['type'] == 'neumann':
                desired = -self._field_value(side, 'q_n_W_m2', projected) / conductivity
            else:
                ambient = self._field_value(side, 'T_inf_C', projected)
                desired = -bc['h_W_m2K'] * (base_boundary - ambient) / conductivity
            temperature = temperature + self._normal_bump(side, xy, desired - normal_derivative)
        return temperature, None

    def temperature(self, xy, labels):
        """Evaluate the wrapped field using the region labels expected by the audit API."""
        result = torch.empty((len(xy), 1), dtype=xy.dtype, device=xy.device)
        for region in self.problem.regions:
            mask = labels == region.id
            if mask.any():
                result[mask] = self.field(region, xy[mask])[0]
        return result

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _legendre_basis(coordinate, order):
        values = [torch.ones_like(coordinate)]
        if order == 0:
            return torch.cat(values, dim=1)
        values.append(coordinate)
        for degree in range(2, order + 1):
            values.append(((2 * degree - 1) * coordinate * values[-1] - (degree - 1) * values[-2]) / degree)
        return torch.cat(values, dim=1)

    def _boundary_features(self, side, xy):
        if self.boundary_mode == 'annulus':
            center = xy.new_tensor(self.problem.doc['domain']['center'])
            radius = torch.linalg.vector_norm(xy[:, :2] - center, dim=1, keepdim=True).clamp_min(torch.finfo(xy.dtype).tiny)
            features = (xy[:, :2] - center) / radius
            return (features + 1) / 2 if self.input_normalization == 'unit_box_v1' else features
        tangent = 1 if side in ('left', 'right') else 0
        lower = self.problem.bounds[0 if tangent == 0 else 2]
        upper = self.problem.bounds[1 if tangent == 0 else 3]
        fraction = (xy[:, tangent:tangent + 1] - lower) / (upper - lower)
        return fraction if self.input_normalization == 'unit_box_v1' else 2 * fraction - 1

    def _boundary_trace(self, side, xy, region):
        piece = self.boundary_pieces[side]
        bc = piece.bc
        learned = self.boundary_trace_networks[side](self._boundary_features(side, xy))
        if bc['type'] == 'dirichlet':
            temperature = bc['value'](xy)
            flux_normal = learned * self.problem.scales(region)['flux']
        elif bc['type'] == 'neumann':
            temperature = self._free_boundary_temperature(side, xy, learned)
            flux_normal = bc['value'](xy)
        else:
            temperature = self._free_boundary_temperature(side, xy, learned)
            flux_normal = bc['h'] * (temperature - bc['T_inf'](xy))
        return temperature, flux_normal

    def _free_boundary_temperature(self, side, xy, learned):
        if self.boundary_mode == 'annulus':
            return self.problem.T_ref + self.problem.T_scale * learned
        adjacent = ('left', 'right') if side in ('top', 'bottom') else ('bottom', 'top')
        axis = 0 if side in ('top', 'bottom') else 1
        lower, upper = (self.problem.bounds[0], self.problem.bounds[1]) if axis == 0 else (self.problem.bounds[2], self.problem.bounds[3])
        endpoint_values = []
        for endpoint, adjacent_side in zip((lower, upper), adjacent):
            points = xy[:1].clone()
            points[:, axis] = endpoint
            normal_axis = 1 - axis
            normal_bound = {
                'top': self.problem.bounds[3], 'bottom': self.problem.bounds[2],
                'right': self.problem.bounds[1], 'left': self.problem.bounds[0],
            }[side]
            points[:, normal_axis] = normal_bound
            neighboring_bc = self.boundary_pieces[adjacent_side].bc
            if neighboring_bc['type'] == 'dirichlet':
                endpoint_values.append(neighboring_bc['value'](points))
            else:
                endpoint_values.append(points.new_full((len(points), 1), self.problem.T_ref))
        coordinate = self._boundary_features(side, xy)
        fraction = coordinate if self.input_normalization == 'unit_box_v1' else (coordinate + 1) / 2
        base = (1 - fraction) * endpoint_values[0] + fraction * endpoint_values[1]
        return base + self.problem.T_scale * 4 * fraction * (1 - fraction) * learned

    def _trace(self, face, xy):
        index = self.interface_index[face.name]
        coefficients = self.trace_coefficients[index]
        radius = getattr(face, 'radius', 0.)
        if radius:
            center = xy.new_tensor(face.center)
            basis = self._angular_basis(xy, center)
            temperature = self.problem.T_ref + self.problem.T_scale * (basis @ coefficients[0:1].T)
            flux_envelope = torch.ones_like(xy[:, :1])
        else:
            axis = face.axis
            tangent = 1 - axis
            lower = self.problem.bounds[0 if tangent == 0 else 2]
            upper = self.problem.bounds[1 if tangent == 0 else 3]
            coordinate = 2 * (xy[:, tangent:tangent + 1] - lower) / (upper - lower) - 1
            basis = self._legendre_basis(coordinate, 2 * self.fourier_order)
            position = self.interface_positions[face.name]
            start = xy.new_zeros((1, 2)); start[:, axis] = position; start[:, tangent] = lower
            end = xy.new_zeros((1, 2)); end[:, axis] = position; end[:, tangent] = upper
            low_side, high_side = (('left', 'right') if axis == 0 else ('bottom', 'top'))
            region = self.problem.regions[face.left]
            start_temperature, _ = self._boundary_trace(low_side, start, region)
            end_temperature, _ = self._boundary_trace(high_side, end, region)
            fraction = (coordinate + 1) / 2
            base = (1 - fraction) * start_temperature + fraction * end_temperature
            temperature = base + self.problem.T_scale * (1 - coordinate.square()) * (basis @ coefficients[0:1].T)
            flux_envelope = 1 - coordinate.square()
        flux = self.interface_flux_scales[index] * flux_envelope * (basis @ coefficients[1:2].T)
        return temperature, flux

    @staticmethod
    def _hermite(coordinate, lower, upper, value_lower, slope_lower, value_upper, slope_upper):
        length = upper - lower
        s = (coordinate - lower) / length
        s2, s3 = s.square(), s.pow(3)
        h00, h10 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s
        h01, h11 = -2 * s3 + 3 * s2, s3 - s2
        return h00 * value_lower + h10 * length * slope_lower + h01 * value_upper + h11 * length * slope_upper

    def _annulus_field(self, region, xy):
        points = xy if xy.requires_grad else xy.detach().requires_grad_(True)
        center = points.new_tensor(self.problem.doc['domain']['center'])
        radius = torch.linalg.vector_norm(points[:, :2] - center, dim=1, keepdim=True)
        ri = float(self.problem.doc['domain']['inner_radius_m'])
        ro = float(self.problem.doc['domain']['outer_radius_m'])
        inner_T, inner_q = self._boundary_trace('inner', points, region)
        outer_T, outer_q = self._boundary_trace('outer', points, region)
        conductivity = self.problem.conductivity(region, points)
        log_extent = math.log(ro / ri)
        coordinate = torch.log(radius / ri) / log_extent
        inner_slope = inner_q / conductivity
        outer_slope = -outer_q / conductivity
        lifting = self._hermite(coordinate, 0., 1., inner_T, inner_slope * ri * log_extent,
            outer_T, outer_slope * ro * log_extent)
        raw = self._free_correction(region, points, self.input_coordinates(region, points))
        envelope = (coordinate * (1 - coordinate)).square()
        return lifting + self.problem.amplitude(region, 'region') * raw * envelope, None

    def _mms_end_data(self, region, high, xy):
        axis = region.axis
        xy = xy.clone()
        xy[:, axis] = region.upper if high else region.lower
        if high:
            face = next((item for item in self.interfaces
                if not getattr(item, 'radius', 0.) and item.left == region.id), None)
            if face is not None:
                temperature, flux = self._trace(face, xy)
                return temperature, -flux / self.problem.conductivity(region, xy)
            side = 'right' if axis == 0 else 'top'
        else:
            face = next((item for item in self.interfaces
                if not getattr(item, 'radius', 0.) and item.right == region.id), None)
            if face is not None:
                temperature, flux = self._trace(face, xy)
                return temperature, -flux / self.problem.conductivity(region, xy)
            side = 'left' if axis == 0 else 'bottom'
        temperature, flux = self._boundary_trace(side, xy, region)
        normal_sign = 1. if side in ('right', 'top') else -1.
        slope = -normal_sign * flux / self.problem.conductivity(region, xy)
        return temperature, slope

    def _mms_strip_lifting(self, region, xy):
        axis, tangent = region.axis, 1 - region.axis
        lower, upper = region.lower, region.upper
        x0, x1, y0, y1 = self.problem.bounds
        lower_points, upper_points = xy.clone(), xy.clone()
        lower_points[:, tangent] = y0 if tangent == 1 else x0
        upper_points[:, tangent] = y1 if tangent == 1 else x1
        t0, p0 = self._mms_end_data(region, False, xy)
        t1, p1 = self._mms_end_data(region, True, xy)
        hx = self._hermite(xy[:, axis:axis + 1], lower, upper, t0, p0, t1, p1)
        tangent_low, tangent_high = (y0, y1) if tangent == 1 else (x0, x1)
        side_low, side_high = (('bottom', 'top') if tangent == 1 else ('left', 'right'))

        def tangential_data(points, side):
            value, flux = self._boundary_trace(side, points, region)
            hx_edge = self._hermite(points[:, axis:axis + 1], lower, upper,
                *self._mms_end_data(region, False, points), *self._mms_end_data(region, True, points))
            hx_tangent = torch.autograd.grad(hx_edge.sum(), points, create_graph=True, retain_graph=True)[0][:, tangent:tangent + 1]
            if self.boundary_pieces[side].bc['type'] == 'dirichlet':
                slope = hx_tangent
            else:
                sign = 1. if side in ('right', 'top') else -1.
                slope = -sign * flux / self.problem.conductivity(region, points)
            boundary_delta = value - hx_edge
            boundary_slope = slope - hx_tangent
            trace_envelope, _ = self._mms_axis_envelopes(region, points[:, axis:axis + 1])
            active = trace_envelope > torch.finfo(points.dtype).eps
            adjusted_delta = torch.where(active, boundary_delta / trace_envelope.clamp_min(torch.finfo(points.dtype).eps), torch.zeros_like(boundary_delta))
            adjusted_slope = torch.where(active, boundary_slope / trace_envelope.clamp_min(torch.finfo(points.dtype).eps), torch.zeros_like(boundary_slope))
            return adjusted_delta, adjusted_slope

        low_delta, low_slope = tangential_data(lower_points, side_low)
        high_delta, high_slope = tangential_data(upper_points, side_high)
        correction = self._hermite(xy[:, tangent:tangent + 1], tangent_low, tangent_high,
            low_delta, low_slope, high_delta, high_slope)
        tangent_fraction = (xy[:, tangent:tangent + 1] - tangent_low) / (tangent_high - tangent_low)
        trace_envelope, network_envelope = self._mms_axis_envelopes(region, xy[:, axis:axis + 1])
        tangent_envelope = (tangent_fraction * (1 - tangent_fraction)).square()
        return hx, correction, trace_envelope, network_envelope, tangent_envelope

    def _mms_axis_envelopes(self, region, coordinate):
        fraction = (coordinate - region.lower) / (region.upper - region.lower)
        lower_interface = any(axis == region.axis and abs(position - region.lower) < 1e-12
            for axis, position in self.problem.planar_cuts())
        upper_interface = any(axis == region.axis and abs(position - region.upper) < 1e-12
            for axis, position in self.problem.planar_cuts())
        smoothstep = 3 * fraction.square() - 2 * fraction.pow(3)
        if lower_interface and upper_interface:
            trace_envelope = 16 * (fraction * (1 - fraction)).square()
        elif lower_interface:
            trace_envelope = smoothstep
        elif upper_interface:
            trace_envelope = 1 - smoothstep
        else:
            trace_envelope = torch.ones_like(fraction)
        network_envelope = (fraction * (1 - fraction)).square()
        return trace_envelope, network_envelope


def build_model(problem, architecture, width=32, depth=3, activation='tanh', input_normalization=None,
    trace_parameterization='standard', soil_envelope_gain=1.):
    """M0-M2 or a Phase 5 variant by name."""
    if architecture in GENERIC_EXACT_ARCHITECTURES:
        return GenericBoundaryPINN(problem, GENERIC_EXACT_ARCHITECTURES[architecture], width, depth, activation)
    if architecture in EXACT_CORRECTORS or architecture in DEEPXDE_EXACT_CORRECTORS:
        if input_normalization not in (None, 'unit_box_v1'):
            raise ValueError('Exact adapted correctors require unit_box_v1')
        model = build_model(problem, 'M1_EXACT', width, depth, activation,
            input_normalization='unit_box_v1', trace_parameterization=trace_parameterization,
            soil_envelope_gain=soil_envelope_gain)
        if architecture in DEEPXDE_EXACT_CORRECTORS:
            corrector = DeepXDECorrector(problem, DEEPXDE_EXACT_CORRECTORS[architecture], width, depth, activation)
        else:
            corrector = build_model(problem, EXACT_CORRECTORS[architecture], width, depth, activation)
            corrector.input_normalization = 'unit_box_v1'
        model.networks = nn.ModuleList()
        model.offsets = nn.ParameterList()
        model.correction_model = corrector
        model.architecture = architecture
        return model
    if architecture == 'DeepXDE_EXACT':
        architecture = 'M1_EXACT'
    if architecture == 'M1_EXACT':
        from pinn_v3.solvers.pinn.generic_problem import GenericProblem
        if not isinstance(problem, GenericProblem):
            raise ValueError('M1_EXACT requires a pinn-v3-problem-1 problem')
        if problem.supports_exact_cable_interfaces:
            return ExactInterfacePINN(problem, width, depth, activation,
                input_normalization or 'case_log_distance_minus1_plus1_v1',
                trace_parameterization=trace_parameterization, soil_envelope_gain=soil_envelope_gain)
        return GenericBoundaryPINN(problem, 'M1_EXACT', width, depth, activation)
    if trace_parameterization != 'standard' or soil_envelope_gain != 1.:
        raise ValueError('Exact trace conditioning requires M1_EXACT')
    from pinn_v3.solvers.pinn.variants import VARIANTS
    if architecture in VARIANTS:
        if input_normalization not in (None, 'region_coordinates'):
            raise ValueError('This variant does not implement the requested input normalization')
        return VARIANTS[architecture](problem, architecture, width, depth, activation)
    if input_normalization not in (None, 'region_coordinates', 'unit_box_v1'):
        raise ValueError(f'Unsupported input normalization {input_normalization!r}')
    model = PINNModel(problem, architecture, width, depth, activation)
    model.input_normalization = input_normalization or 'region_coordinates'
    return model
