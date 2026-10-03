"""Phase 5 model variants. Every one is ADAPTED from the cited idea, not a reproduction of the original code.

CUSP    global strong net with one-sided |level set| features (idea: Tseng et al. 2023, arXiv:2210.08424, DOI por verificar).
        |d| is written as sign_region * d, the exact one-sided value, so autograd sees a non-zero slope at d = 0.
DCSNN   one shared shallow net on (x, y, region one-hot) (idea: Hu, Lin, Lai 2022, 10.1016/j.jcp.2022.111576, por verificar);
        T and q.n continuity stay soft interface losses, as in M1.
FBPINN  overlapping Cartesian subdomain nets blended by a sigmoid partition of unity (idea: Moseley et al. 2023,
        10.1007/s10444-023-10065-9, por verificar); cuts are aligned with planar material interfaces.
None of them reads a reference solution; boundary and interface conditions are soft losses like M0-M2.
I-PINN and AdaI-PINN are NOT implemented: the original formulation was not verified.
"""
import numpy as np
import torch
from torch import nn

from pinn_v2.solvers.pinn.networks import PINNModel, mlp

WINDOW_SHARPNESS = 0.02


class CuspModel(PINNModel):
    """Cusp-capturing variant of M0: same single network, extra features |d_j| for every material interface."""
    kind = 'faithful_to_idea_adapted'

    def __init__(self, problem, architecture='CUSP', width=32, depth=3, activation='tanh'):
        nn.Module.__init__(self)
        self._setup(problem, architecture, width, depth, 'global', 'strong')
        extra = problem.level_sets(torch.zeros((1, 2), dtype=torch.float64)).shape[1]
        x0, x1, y0, y1 = problem.bounds
        self.length = 0.5 * max(x1 - x0, y1 - y0)
        self.networks = nn.ModuleList([mlp(2 + extra, width, depth, 1, activation)])
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros(()))])
        self.signs = {}
        for region in problem.regions:
            probe = torch.as_tensor(problem.sample_interior(region, 64, 12345), dtype=torch.float64)
            self.signs[region.id] = torch.sign(torch.median(problem.level_sets(probe), dim=0).values)

    def features(self, region, xy):
        d = self.problem.level_sets(xy)
        sign = self.signs[region.id].to(dtype=d.dtype, device=d.device)
        return torch.cat([self.problem.normalized_xy(xy), sign * d / self.length], dim=1)

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
        self.networks = nn.ModuleList([mlp(2 + self.count, width * depth, 1, 1, activation)])
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros(()))])

    def field(self, region, xy):
        one_hot = xy.new_zeros((len(xy), self.count))
        one_hot[:, region.id] = 1.
        raw = self.networks[0](torch.cat([self.problem.normalized_xy(xy), one_hot], dim=1))
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
        edges = []
        for axis, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
            inner = sorted(set(cuts[axis])) if cuts[axis] else list(np.linspace(lo, hi, self.PARTS + 1)[1:-1])
            edges.append([lo] + inner + [hi])
        self.edges = edges
        self.tau = [WINDOW_SHARPNESS * (hi - lo) for lo, hi in ((x0, x1), (y0, y1))]
        self.boxes = [(a, b, c, d) for a, b in zip(edges[0], edges[0][1:]) for c, d in zip(edges[1], edges[1][1:])]
        self.networks = nn.ModuleList([mlp(2, width, depth, 1, activation) for _ in self.boxes])
        self.offsets = nn.ParameterList([nn.Parameter(torch.zeros(()))])

    def _window(self, coordinate, lo, hi, axis, outer_lo, outer_hi):
        tau = self.tau[axis]
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
            blended = blended + weight * net(local)
            total = total + weight
        return self.problem.T_ref + self.problem.T_scale * (self.offsets[0] + blended / total), None


VARIANTS = dict(CUSP=CuspModel, DCSNN=DCSNNModel, FBPINN=FBPINNModel)
