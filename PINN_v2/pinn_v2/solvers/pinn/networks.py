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

ARCHITECTURES = dict(M0=('global', 'strong'), M1=('region', 'strong'), M2=('region', 'mixed'))
EXACT_CORRECTORS = dict(M0_EXACT='M0', CUSP_EXACT='CUSP', DCSNN_EXACT='DCSNN', FBPINN_EXACT='FBPINN')
DEEPXDE_EXACT_CORRECTORS = dict(DeepXDE_PFNN_EXACT='PFNN', DeepXDE_FNN_EXACT='FNN')


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
        self.input_normalization = 'legacy_region_coordinates'

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
        from pinn_v2.solvers.pinn.deepxde_solver import _network
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
    INPUT_NORMALIZATIONS = ('legacy_region_coordinates', 'case_log_distance_minus1_plus1_v1', 'unit_box_v1')

    def __init__(self, problem, width=32, depth=3, activation='tanh', input_normalization='case_log_distance_minus1_plus1_v1',
        trace_parameterization='legacy', soil_envelope_gain=1.):
        outer_temperature = self._outer_reference_temperature(problem)
        super().__init__(problem, 'M1', width, depth, activation)
        self.architecture = 'M1_EXACT'
        if input_normalization not in self.INPUT_NORMALIZATIONS:
            raise ValueError(f'Unsupported M1_EXACT input normalization {input_normalization!r}')
        self.input_normalization = input_normalization
        if trace_parameterization != 'legacy' or soil_envelope_gain != 1.:
            raise ValueError('Cable exact engine has one construction: legacy traces and unit soil envelope gain')
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
        interfaces = problem.domain.interfaces
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
            left_center = problem.case['cables'][left.cable]
            for right in self.outer_interfaces[index + 1:]:
                right_center = problem.case['cables'][right.cable]
                center_distance = ((left_center[0] - right_center[0]) ** 2 +
                    (left_center[1] - right_center[1]) ** 2) ** .5
                gaps.append(center_distance - left.radius - right.radius)
        x0, x1, y0, y1 = problem.bounds
        for interface in self.outer_interfaces:
            cx, cy = problem.case['cables'][interface.cable]
            gaps.extend((cx - x0 - interface.radius, x1 - cx - interface.radius,
                cy - y0 - interface.radius, y1 - cy - interface.radius))
        if len(self.outer_interfaces) != len(problem.case['cables']) or not gaps or min(gaps) <= 0:
            raise ValueError('M1_EXACT requires separated cable interfaces inside the rectangular soil domain')
        self.collar_width = min(gaps) * .25
        self.outer_temperature = outer_temperature
        self._interfaces_by_cable = {
            cable: tuple(sorted((item for item in interfaces if item.cable == cable),
                key=lambda item: item.radius))
            for cable in range(len(problem.case['cables']))
        }

    @staticmethod
    def _outer_reference_temperature(problem):
        from pinn_v2.solvers.pinn.problem import CableProblem
        if not isinstance(problem, CableProblem):
            raise ValueError('M1_EXACT requires an explicit cable problem')
        domain = problem.domain
        if domain.mode != 'fixed' or domain.case.get('outer_radius') or domain.nsoil != 1:
            raise ValueError('M1_EXACT currently requires a fixed-source rectangular domain without soil strata')
        if domain.has_imperfect_contact or not domain.interfaces or any(not item.radius for item in domain.interfaces):
            raise ValueError('M1_EXACT requires perfect contact and circular cable-layer interfaces')
        conditions = domain.side_conditions()
        if not conditions or any(item['type'] not in ('dirichlet', 'neumann', 'robin')
            for item in conditions.values()):
            raise ValueError('M1_EXACT requires Dirichlet, Neumann or Robin data on every outer side')
        if len({item.cable for item in domain.interfaces}) != len(domain.case['cables']):
            raise ValueError('M1_EXACT requires circular material interfaces for every cable')
        values = [domain.case['T0'] if item.get('legacy') else item.get('T_C')
            for item in conditions.values() if item['type'] == 'dirichlet']
        return float(next((value for value in values if value is not None), domain.case['T0']))

    def _boundary_features(self, side, xy):
        return ExactBoundaryPINN._boundary_features(self, side, xy)

    def _boundary_trace(self, side, xy, region):
        return ExactBoundaryPINN._boundary_trace(self, side, xy, region)

    def _free_boundary_temperature(self, side, xy, learned):
        return ExactBoundaryPINN._free_boundary_temperature(self, side, xy, learned)

    @staticmethod
    def _hermite(coordinate, lower, upper, value_lower, slope_lower, value_upper, slope_upper):
        return ExactBoundaryPINN._hermite(
            coordinate, lower, upper, value_lower, slope_lower, value_upper, slope_upper)

    def _mms_end_data(self, region, high, xy):
        return ExactBoundaryPINN._mms_end_data(self, region, high, xy)

    def _mms_axis_envelopes(self, region, coordinate):
        return ExactBoundaryPINN._mms_axis_envelopes(self, region, coordinate)

    def _mms_strip_lifting(self, region, xy):
        return ExactBoundaryPINN._mms_strip_lifting(self, region, xy)

    def _interface_center(self, interface):
        return self.trace_coefficients.new_tensor(self.problem.case['cables'][interface.cable])

    def _angular_basis(self, xy, center):
        delta = xy[:, :2] - center
        radius = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(torch.finfo(xy.dtype).tiny)
        cos_theta, sin_theta = delta[:, :1] / radius, delta[:, 1:2] / radius
        terms = [torch.ones_like(radius)]
        cos_n, sin_n = torch.ones_like(radius), torch.zeros_like(radius)
        for _ in range(self.fourier_order):
            cos_n, sin_n = cos_n * cos_theta - sin_n * sin_theta, sin_n * cos_theta + cos_n * sin_theta
            terms.extend((cos_n, sin_n))
        return torch.cat(terms, dim=1)

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
        temperature_coefficients = temperature_coefficients.clone()
        temperature_coefficients[0] = temperature_coefficients[0] + self.outer_temperature
        flux_coefficients = self._flux_coefficients(interface)
        conductivity = float(region.k)
        result = torch.zeros_like(radial_squared)
        for mode in range(2 * self.fourier_order + 1):
            order = 0 if mode == 0 else (mode + 1) // 2
            temperature_trace = temperature_coefficients[mode]
            radial_derivative = -flux_coefficients[mode] / conductivity
            correction = (interface.radius * radial_derivative - order * temperature_trace) / 2
            result = result + harmonics[mode] * (
                temperature_trace + correction * (radial_squared - 1))
        return result

    def _annulus_lifting(self, inner, outer, xy, region):
        center = self._interface_center(outer)
        radius = torch.linalg.vector_norm(xy[:, :2] - center, dim=1, keepdim=True)
        length = outer.radius - inner.radius
        coordinate = (radius - inner.radius) / length
        g0, q0 = self._trace(inner, xy)
        g1, q1 = self._trace(outer, xy)
        p0, p1 = -q0 / float(region.k), -q1 / float(region.k)
        s2, s3 = coordinate.square(), coordinate.pow(3)
        h00, h10 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + coordinate
        h01, h11 = -2 * s3 + 3 * s2, s3 - s2
        return h00 * g0 + h10 * length * p0 + h01 * g1 + h11 * length * p1

    def _coordinates(self, region, xy):
        if self.input_normalization == 'unit_box_v1':
            return self.problem.unit_coordinates(region, xy, 'region')
        if region.kind != 'soil' or self.input_normalization == 'legacy_region_coordinates':
            return self.problem.coordinates(region, xy, 'region')
        coordinates = [self.problem.normalized_xy(xy)]
        x0, x1, y0, y1 = self.problem.bounds
        corners = xy.new_tensor([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
        for center, radius in zip(self.problem.case['cables'], [self.problem.case['radius']] * len(self.problem.case['cables'])):
            center_tensor = xy.new_tensor(center)
            maximum_distance = torch.linalg.vector_norm(corners - center_tensor, dim=1).max()
            maximum_log_distance = torch.log(maximum_distance / radius).clamp_min(torch.finfo(xy.dtype).eps)
            distance = torch.linalg.vector_norm(xy[:, :2] - center_tensor, dim=1, keepdim=True)
            log_distance = torch.log((distance / radius).clamp_min(1.))
            normalized = (2 * log_distance / maximum_log_distance - 1).clamp(-1, 1)
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

        interfaces = self._interfaces_by_cable[region.cable]
        outer = next(item for item in interfaces if abs(item.radius - region.ro) < 1e-12)
        if region.kind == 'conductor':
            lifting = self._disk_lifting(outer, xy, region)
            center = self._interface_center(outer)
            radius_squared = (xy[:, :2] - center).square().sum(dim=1, keepdim=True)
            envelope = ((radius_squared - region.ro ** 2) / region.ro ** 2).square()
        else:
            inner = next(item for item in interfaces if abs(item.radius - region.ri) < 1e-12)
            lifting = self._annulus_lifting(inner, outer, xy, region)
            radius = torch.linalg.vector_norm(xy[:, :2] - self._interface_center(outer), dim=1, keepdim=True)
            length = region.ro - region.ri
            envelope = (((radius - region.ri) * (radius - region.ro) / length ** 2) * 4).square()
        return lifting + envelope * amplitude * raw, None


class ExactBoundaryPINN(PINNModel):
    """Exact BC/transmission lifting for rectangular MMS and analytic annulus problems."""
    kind = 'exact_boundaries'
    exact_constraints = True
    fourier_order = 8

    def __init__(self, problem, width=32, depth=3, activation='tanh'):
        from pinn_v2.solvers.pinn.problem import AnnulusProblem, MMSProblem
        if not isinstance(problem, (AnnulusProblem, MMSProblem)):
            raise ValueError('Exact boundary engine requires an MMS or annulus problem')
        self.boundary_mode = 'annulus' if isinstance(problem, AnnulusProblem) else 'mms'
        boundary_types = {piece.bc['type'] for piece in problem.pieces(8)}
        if self.boundary_mode == 'mms' and not boundary_types <= {'dirichlet', 'neumann', 'robin'}:
            raise ValueError('Exact MMS supports only Dirichlet, Neumann, and Robin boundaries')
        super().__init__(problem, 'M1', width, depth, activation)
        self.architecture = 'M1_EXACT'
        self.interfaces = tuple(problem.faces(16)) if self.boundary_mode == 'mms' else ()
        self.interface_index = {face.name: index for index, face in enumerate(self.interfaces)}
        self.interface_positions = dict(zip(
            (face.name for face in self.interfaces),
            (position for _, position in problem.planar_cuts())))
        self.trace_coefficients = nn.Parameter(torch.zeros(len(self.interfaces), 2, 2 * self.fourier_order + 1))
        self.interface_flux_scales = tuple(
            max(problem.scales(problem.regions[face.left])['flux'],
                problem.scales(problem.regions[face.right])['flux'])
            for face in self.interfaces)
        pieces = problem.pieces(16)
        self.boundary_pieces = {}
        for piece in pieces:
            previous = self.boundary_pieces.get(piece.side)
            if previous is not None and previous.bc['type'] != piece.bc['type']:
                raise ValueError(f'Exact boundary engine requires one BC type per side: {piece.side}')
            self.boundary_pieces.setdefault(piece.side, piece)
        feature_count = 2 if self.boundary_mode == 'annulus' else 1
        self.boundary_trace_networks = nn.ModuleDict({
            side: mlp(feature_count, width, depth, 1, activation)
            for side in self.boundary_pieces
        })

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
            radius = torch.linalg.vector_norm(xy[:, :2], dim=1, keepdim=True).clamp_min(torch.finfo(xy.dtype).tiny)
            features = xy[:, :2] / radius
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
            if neighboring_bc['type'] != 'dirichlet':
                raise ValueError('Exact MMS Neumann/Robin sides require compatible Dirichlet data at both corners')
            endpoint_values.append(neighboring_bc['value'](points))
        coordinate = self._boundary_features(side, xy)
        if self.input_normalization == 'unit_box_v1':
            coordinate = 2 * coordinate - 1
        fraction = (coordinate + 1) / 2
        base = (1 - fraction) * endpoint_values[0] + fraction * endpoint_values[1]
        return base + self.problem.T_scale * (1 - coordinate.square()) * learned

    def _trace(self, face, xy):
        index = self.interface_index[face.name]
        coefficients = self.trace_coefficients[index]
        if getattr(face, 'radius', 0.):
            center = xy.new_tensor(self.problem.case['cables'][face.cable])
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
        radius = torch.linalg.vector_norm(points[:, :2], dim=1, keepdim=True)
        ri, ro = self.problem.case['ri'], self.problem.case['ro']
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

    def _mms_field(self, region, xy):
        with torch.enable_grad():
            points = xy if xy.requires_grad else xy.detach().requires_grad_(True)
            lifting, correction, trace_envelope, network_envelope, tangent_envelope = self._mms_strip_lifting(region, points)
            raw = self._free_correction(region, points, self.input_coordinates(region, points))
            amplitude = self.problem.amplitude(region, 'region')
            return lifting + trace_envelope * correction + network_envelope * tangent_envelope * amplitude * raw, None

    def field(self, region, xy):
        if self.boundary_mode == 'annulus':
            return self._annulus_field(region, xy)
        return self._mms_field(region, xy)


def install_dirichlet_lifting(model, problem):
    """Wrap model.field so soil T = T_ref on the box boundary: T0 + 16 u(1-u) v(1-v) (T_net - T0).

    Valid only when all four box sides are Dirichlet at problem.T_ref; anything else raises ValueError.
    """
    domain = getattr(problem, 'domain', None)
    if domain is None or domain.case.get('outer_radius') or domain.case.get('outer_gradient_K_m'):
        raise ValueError('Dirichlet lifting requires a rectangular cable box with constant T_ref boundaries')
    for side, bc in domain.side_conditions().items():
        if bc['type'] != 'dirichlet' or not (bc.get('legacy') or bc['T_C'] == problem.T_ref):
            raise ValueError(f'Dirichlet lifting requires side {side!r} to be Dirichlet at T_ref')
    x0, x1, y0, y1 = problem.bounds
    t0 = problem.T_ref
    original = model.field

    def field(region, xy):
        temperature, flux = original(region, xy)
        if region.kind != 'soil':
            return temperature, flux
        u = (xy[:, 0:1] - x0) / (x1 - x0)
        v = (xy[:, 1:2] - y0) / (y1 - y0)
        return t0 + 16 * u * (1 - u) * v * (1 - v) * (temperature - t0), flux

    model.field = field
    return model


def build_model(problem, architecture, width=32, depth=3, activation='tanh', input_normalization=None,
    trace_parameterization='legacy', soil_envelope_gain=1.):
    """M0-M2 or a Phase 5 variant by name."""
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
        from pinn_v2.solvers.pinn.problem import AnnulusProblem, CableProblem, MMSProblem
        if isinstance(problem, CableProblem):
            return ExactInterfacePINN(problem, width, depth, activation,
                input_normalization or 'case_log_distance_minus1_plus1_v1',
                trace_parameterization=trace_parameterization, soil_envelope_gain=soil_envelope_gain)
        if isinstance(problem, (AnnulusProblem, MMSProblem)):
            if trace_parameterization != 'legacy' or soil_envelope_gain != 1.:
                raise ValueError('Radial trace conditioning and soil gain require a cable geometry')
            model = ExactBoundaryPINN(problem, width, depth, activation)
            if input_normalization not in (None, 'legacy_region_coordinates', 'unit_box_v1',
                    'log_radius_direction_minus1_plus1_v1', 'problem_coordinates_minus1_plus1_v1'):
                raise ValueError(f'Unsupported input normalization {input_normalization!r}')
            model.input_normalization = input_normalization or 'legacy_region_coordinates'
            return model
        raise ValueError('M1_EXACT supports cable, MMS, and annulus problems only')
    if trace_parameterization != 'legacy' or soil_envelope_gain != 1.:
        raise ValueError('Exact trace conditioning requires M1_EXACT')
    from pinn_v2.solvers.pinn.variants import VARIANTS
    if architecture in VARIANTS:
        if input_normalization not in (None, 'legacy_region_coordinates'):
            raise ValueError('This variant does not implement the requested input normalization')
        return VARIANTS[architecture](problem, architecture, width, depth, activation)
    if input_normalization not in (None, 'legacy_region_coordinates', 'unit_box_v1'):
        raise ValueError(f'Unsupported input normalization {input_normalization!r}')
    model = PINNModel(problem, architecture, width, depth, activation)
    model.input_normalization = input_normalization or 'legacy_region_coordinates'
    return model
