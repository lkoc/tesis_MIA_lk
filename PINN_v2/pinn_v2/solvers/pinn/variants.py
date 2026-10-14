"""Phase 5 model variants. Every one is ADAPTED from the cited idea, not a reproduction of the original code.

 CUSP    global strong net with one-sided |level set| features (adapted from Tseng et al. 2023, DOI 10.1016/j.jcp.2023.112359).
        |d| is written as sign_region * d, the exact one-sided value, so autograd sees a non-zero slope at d = 0.
DCSNN   one shared net on (x, y, region one-hot) (adapted from Hu, Lin, Lai 2022, DOI 10.1016/j.jcp.2022.111576);
        T and q.n continuity stay soft interface losses, as in M1.
FBPINN  Cartesian subdomain nets blended by normalized sigmoid windows (adapted from Moseley et al. 2023,
    DOI 10.1007/s10444-023-10065-9); cuts follow planar interfaces and cable outer radii.
None of them reads a reference solution; boundary and interface conditions are soft losses like M0-M2.
I-PINN and AdaI-PINN are NOT implemented: the original formulation was not verified.
"""
import numpy as np
import torch
from torch import nn

from pinn_v2.solvers.pinn.networks import PINNModel, mlp

WINDOW_SHARPNESS = 0.02
CUSP_OUTPUT_INIT_GAIN = 3e-3


def _cusp_interface_scales(problem, fallback):
    domain = getattr(problem, 'domain', None)
    if domain is not None:
        scales = []
        for interface in domain.interfaces:
            adjacent = (problem.regions[interface.left], problem.regions[interface.right])
            widths = []
            for region in adjacent:
                if interface.radius:
                    width = fallback if region.kind == 'soil' else region.ro - region.ri
                else:
                    width = region.upper - region.lower
                    if not np.isfinite(width):
                        width = fallback
                widths.append(float(width))
            scales.append(min(widths))
        return tuple(scales)

    interface_info = getattr(problem, 'interface_info', ())
    return tuple(min(problem.regions[index].upper - problem.regions[index].lower,
                     problem.regions[index + 1].upper - problem.regions[index + 1].lower)
                 for index in range(len(interface_info)))


class CuspModel(PINNModel):
    """Cusp-capturing global net with bounded, locally scaled one-sided interface features."""
    kind = 'faithful_to_idea_adapted'

    def __init__(self, problem, architecture='CUSP', width=32, depth=3, activation='tanh'):
        nn.Module.__init__(self)
        self._setup(problem, architecture, width, depth, 'global', 'strong')
        extra = problem.level_sets(torch.zeros((1, 2), dtype=torch.float64)).shape[1]
        base_size = problem.input_size(problem.regions[0], 'global')
        x0, x1, y0, y1 = problem.bounds
        self.length = 0.5 * max(x1 - x0, y1 - y0)
        self.interface_scales = _cusp_interface_scales(problem, self.length)
        self.networks = nn.ModuleList([mlp(base_size + extra, width, depth, 1, activation)])
        with torch.no_grad():
            self.networks[0][-1].weight.mul_(CUSP_OUTPUT_INIT_GAIN)
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros(()))])
        self.signs = {}
        for region in problem.regions:
            probe = torch.as_tensor(problem.sample_interior(region, 64, 12345), dtype=torch.float64)
            self.signs[region.id] = torch.sign(torch.median(problem.level_sets(probe), dim=0).values)

    def features(self, region, xy):
        d = self.problem.level_sets(xy)
        sign = self.signs[region.id].to(dtype=d.dtype, device=d.device)
        scale = xy.new_tensor(self.interface_scales)
        level_features = torch.tanh(sign * d / scale)
        return torch.cat([self.problem.coordinates(region, xy, self.partition), level_features], dim=1)

    def field(self, region, xy):
        raw = self.networks[0](self.features(region, xy))
        return self.problem.T_ref + self.problem.T_scale * (self.offsets[0] + raw[:, :1]), None


class DCSNNModel(PINNModel):
    """One shared shallow network with the region indicator as an input feature."""
    kind = 'faithful_to_idea_adapted'

    def __init__(self, problem, architecture='DCSNN', width=32, depth=3, activation='tanh'):
        nn.Module.__init__(self)
        self._setup(problem, architecture, width, depth, 'global', 'strong')
        self.count = len(problem.regions)
        base_size = problem.input_size(problem.regions[0], 'global')
        self.networks = nn.ModuleList([mlp(base_size + self.count, width, depth, 1, activation)])
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros(()))])

    def field(self, region, xy):
        one_hot = xy.new_zeros((len(xy), self.count))
        one_hot[:, region.id] = 1.
        coordinates = self.problem.coordinates(region, xy, self.partition)
        raw = self.networks[0](torch.cat([coordinates, one_hot], dim=1))
        return self.problem.T_ref + self.problem.T_scale * (self.offsets[0] + raw[:, :1]), None


class FBPINNModel(PINNModel):
    """Partition-of-unity blend of local networks on a Cartesian decomposition."""
    kind = 'faithful_to_idea_adapted'
    PARTS = 2

    def __init__(self, problem, architecture='FBPINN', width=16, depth=2, activation='tanh'):
        nn.Module.__init__(self)
        self._setup(problem, architecture, width, depth, 'global', 'strong')
        x0, x1, y0, y1 = problem.bounds
        cuts = [[], []]
        for axis, position in problem.planar_cuts():
            cuts[axis].append(position)
        case = getattr(problem, 'case', {})
        if isinstance(case, dict) and case.get('kind') == 'annulus':
            radius = case['ri']
            for axis, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
                cuts[axis].extend(position for position in (-radius, 0., radius) if lo < position < hi)
        elif isinstance(case, dict) and case.get('cables'):
            radius = case.get('radius')
            if radius is None and case.get('layers'):
                radius = max(layer[1] for layer in case['layers'])
            if radius is not None and radius > 0:
                for center in case['cables']:
                    for axis, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
                        cuts[axis].extend(position for position in
                            (center[axis] - radius, center[axis], center[axis] + radius)
                            if lo < position < hi)
        edges = []
        for axis, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
            inner = sorted(set(cuts[axis])) if cuts[axis] else list(np.linspace(lo, hi, self.PARTS + 1)[1:-1])
            edges.append([lo] + inner + [hi])
        self.edges = edges
        self.boxes = [(a, b, c, d) for a, b in zip(edges[0], edges[0][1:]) for c, d in zip(edges[1], edges[1][1:])]
        self.geometry_features = isinstance(case, dict) and case.get('kind') == 'annulus'
        input_size = problem.input_size(problem.regions[0], 'global') if self.geometry_features else 2
        self.networks = nn.ModuleList([mlp(input_size, width, depth, 1, activation) for _ in self.boxes])
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros(()))])

    def _window(self, coordinate, lo, hi, axis, outer_lo, outer_hi):
        tau = WINDOW_SHARPNESS * (hi - lo)
        left = torch.ones_like(coordinate) if lo <= outer_lo else torch.sigmoid((coordinate - lo) / tau)
        right = torch.ones_like(coordinate) if hi >= outer_hi else torch.sigmoid((hi - coordinate) / tau)
        return left * right

    def field(self, region, xy):
        x0, x1, y0, y1 = self.problem.bounds
        total = torch.zeros_like(xy[:, :1])
        blended = torch.zeros_like(xy[:, :1])
        for net, (a, b, c, d) in zip(self.networks, self.boxes):
            weight = self._window(xy[:, :1], a, b, 0, x0, x1) * self._window(xy[:, 1:2], c, d, 1, y0, y1)
            local = torch.cat([2 * (xy[:, :1] - a) / (b - a) - 1, 2 * (xy[:, 1:2] - c) / (d - c) - 1], dim=1)
            features = self.problem.coordinates(region, xy, self.partition) if self.geometry_features else local
            blended = blended + weight * net(features)
            total = total + weight
        return self.problem.T_ref + self.problem.T_scale * (self.offsets[0] + blended / total), None


VARIANTS = dict(CUSP=CuspModel, DCSNN=DCSNNModel, FBPINN=FBPINNModel)
