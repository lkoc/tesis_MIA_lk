"""One problem interface for every PINN family: regions, faces, boundary pieces, source, scales, reference.

Two adapters share it: MMSProblem (unit-square manufactured cases V0/V1, Robin variant) and
CableProblem (explicit cable cross section through FullDomain). Conventions are those of
cases/boundary.py: n is the outward normal, q = -k grad T, q.n > 0 when heat leaves.
The exact or reference temperature is used by evaluation only, never by a loss.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch

from pinn_v2.cases.boundary import RECT_SIDES, SIDE_NORMALS
from pinn_v2.cases.catalog import cases, field_k, source as case_source, exact as case_exact
from pinn_v2.cases.catalog import interfaces as case_interfaces, evaluation_points as case_evaluation_points, fingerprint as case_fingerprint
from pinn_v2.cases.verification import coaxial_robin, coaxial_dirichlet, symmetric_pair
from pinn_v2.geometry.domain import Region, FullDomain, radial_exact, coaxial_case
from pinn_v2.sampling.collocation import sample as stratified_sample

PROBE = 1e-7


@dataclass(frozen=True)
class Face:
    name: str
    left: int
    right: int
    xy: np.ndarray
    normal: np.ndarray
    weight: float
    axis: int = -1


@dataclass(frozen=True)
class Piece:
    side: str
    region: int
    xy: np.ndarray
    normal: np.ndarray
    weight: float
    bc: dict


def _dirichlet(value):
    return dict(type='dirichlet', value=value)


def _probe(zone, xy):
    """Coordinate along the stratification axis kept strictly inside the zone, so k is the one-sided value."""
    coordinate = xy[:, zone.axis:zone.axis + 1]
    inside = coordinate.detach().clamp(zone.lower + PROBE, zone.upper - PROBE)
    shifted = coordinate + (inside - coordinate.detach())
    columns = [xy[:, 0:1], xy[:, 1:2]]
    columns[zone.axis] = shifted
    return columns


class Problem:
    """Common surface; subclasses fill regions, faces, pieces and the physical callbacks."""
    name = ''
    regions = ()
    bounds = (0., 1., 0., 1.)
    T_ref = 0.
    T_scale = 1.
    T_level = 0.
    reference_label = None

    def conductors(self):
        return [r.id for r in self.regions if r.kind == 'conductor']

    def amplitude(self, region, partition):
        return self.T_scale

    def normalized_xy(self, xy):
        x0, x1, y0, y1 = self.bounds
        return torch.cat([2 * (xy[:, :1] - x0) / (x1 - x0) - 1, 2 * (xy[:, 1:2] - y0) / (y1 - y0) - 1], dim=1)

    def input_size(self, region, partition):
        return 2

    def coordinates(self, region, xy, partition):
        return self.normalized_xy(xy)

    def mixed_region_split(self):
        return len(self.regions) > 1

    def level_sets(self, xy):
        """Signed distance of every point to each material interface, shape (N, m); empty when there is none."""
        return xy.new_zeros((len(xy), 0))


class MMSProblem(Problem):
    """Manufactured unit-square cases. Dirichlet sides carry the exact trace; mms_robin has a Robin top (h=10)."""
    ROBIN_H = 10.
    reference_label = 'exact manufactured solution'

    def __init__(self, case):
        if not case['kind'].startswith('mms'):
            raise ValueError('MMSProblem needs a manufactured case')
        self.case = case
        self.name = case['id']
        self.bounds = tuple(case['bounds'])
        self.T_ref = case['T0']
        self.T_level = case['T0']
        self.T_scale = case['scale']
        cuts = case_interfaces(case)
        axes = {0 if i['axis'] == 'x' else 1 for i in cuts}
        if len(axes) > 1:
            raise ValueError('Interfaces must share one axis')
        axis = next(iter(axes), 0)
        lows, highs = (self.bounds[0], self.bounds[1]) if axis == 0 else (self.bounds[2], self.bounds[3])
        limits = [lows] + [i['position'] for i in cuts] + [highs]
        self.axis = axis
        self.regions = tuple(Region(j, f'mms_{j}', 'soil', axis=axis, lower=lo, upper=hi) for j, (lo, hi) in enumerate(zip(limits, limits[1:])))
        self.interface_info = [dict(position=i['position']) for i in cuts]
        self.robin = case['kind'] == 'mms_robin'
        self._k_ref = {r.id: self._reference_k(r) for r in self.regions}
        self.physics_sha256 = case_fingerprint(case)
        self.power_total, self.power_scale = self._integrate_source()

    def _box(self, zone):
        x0, x1, y0, y1 = self.bounds
        return (zone.lower, zone.upper, y0, y1) if self.axis == 0 else (x0, x1, zone.lower, zone.upper)

    def _grid(self, zone, n):
        a, b, c, d = self._box(zone)
        z, w = np.polynomial.legendre.leggauss(n)
        x, y = (a + (b - a) * (z + 1) / 2), (c + (d - c) * (z + 1) / 2)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return X, Y, np.outer(w, w) * (b - a) * (d - c) / 4

    def _reference_k(self, zone):
        X, Y, W = self._grid(zone, 8)
        return float(np.sum(field_k(self.case, X, Y) * W) / np.sum(W))

    def _integrate_source(self):
        total = absolute = 0.
        self.region_power = {}
        for zone in self.regions:
            X, Y, W = self._grid(zone, 48)
            q = case_source(self.case, X, Y)
            self.region_power[zone.id] = float(np.sum(q * W))
            total += self.region_power[zone.id]
            absolute += float(np.sum(np.abs(q) * W))
        return total, absolute

    def conductivity(self, zone, xy):
        columns = _probe(zone, xy) if len(self.regions) > 1 else [xy[:, 0:1], xy[:, 1:2]]
        return field_k(self.case, columns[0], columns[1], torch)

    def source(self, zone, xy):
        columns = _probe(zone, xy) if len(self.regions) > 1 else [xy[:, 0:1], xy[:, 1:2]]
        return case_source(self.case, columns[0], columns[1], torch)

    def scales(self, zone):
        k = self._k_ref[zone.id]
        return dict(temperature=self.T_scale, flux=k * self.T_scale, divergence=k * self.T_scale)

    def sample_interior(self, zone, n, seed, policy='uniform'):
        a, b, c, d = self._box(zone)
        return np.random.default_rng(seed).uniform([a, c], [b, d], (n, 2))

    def faces(self, n):
        s = (np.arange(n) + .5) / n
        result = []
        for j, info in enumerate(self.interface_info):
            if self.axis == 0:
                xy = np.c_[np.full(n, info['position']), self.bounds[2] + (self.bounds[3] - self.bounds[2]) * s]
                normal = np.tile([1., 0.], (n, 1))
                length = self.bounds[3] - self.bounds[2]
            else:
                xy = np.c_[self.bounds[0] + (self.bounds[1] - self.bounds[0]) * s, np.full(n, info['position'])]
                normal = np.tile([0., 1.], (n, 1))
                length = self.bounds[1] - self.bounds[0]
            result.append(Face(f'mms_interface_{j}', j, j + 1, xy, normal, length / n, self.axis))
        return result

    def level_sets(self, xy):
        columns = [xy[:, self.axis:self.axis + 1] - info['position'] for info in self.interface_info]
        return torch.cat(columns, dim=1) if columns else xy.new_zeros((len(xy), 0))

    def _exact(self, xy):
        return case_exact(self.case, xy[:, 0:1], xy[:, 1:2], torch)

    def _bc(self, side):
        if self.robin and side == 'top':
            return dict(type='robin', h=self.ROBIN_H, T_inf=lambda xy: self._exact(xy) + 3 * xy[:, 0:1] * (1 - xy[:, 0:1]))
        return _dirichlet(self._exact)

    def pieces(self, n):
        x0, x1, y0, y1 = self.bounds
        s = (np.arange(n) + .5) / n
        coordinates = dict(bottom=np.c_[x0 + (x1 - x0) * s, np.full(n, y0)], top=np.c_[x0 + (x1 - x0) * s, np.full(n, y1)],
            left=np.c_[np.full(n, x0), y0 + (y1 - y0) * s], right=np.c_[np.full(n, x1), y0 + (y1 - y0) * s])
        lengths = dict(bottom=x1 - x0, top=x1 - x0, left=y1 - y0, right=y1 - y0)
        result = []
        for side in RECT_SIDES:
            xy = coordinates[side]
            normal = np.tile(SIDE_NORMALS[side], (n, 1))
            for zone in self.regions:
                # Half-open ownership: a point on a cut belongs to the lower zone, like the k expression.
                mask = (xy[:, zone.axis] > zone.lower) & (xy[:, zone.axis] <= zone.upper)
                if zone.id == 0:
                    mask |= xy[:, zone.axis] == zone.lower
                if np.any(mask):
                    result.append(Piece(side, zone.id, xy[mask], normal[mask], lengths[side] / n, self._bc(side)))
        return result

    def evaluation(self):
        xy = case_evaluation_points(self.case)
        labels = np.zeros(len(xy), dtype=int)
        for zone in self.regions:
            owned = (xy[:, zone.axis] > zone.lower) & (xy[:, zone.axis] <= zone.upper)
            labels[owned] = zone.id
        weights = np.full(len(xy), 1. / len(xy))
        return xy, labels, weights

    def reference(self, xy, labels=None):
        return np.asarray(case_exact(self.case, xy[:, 0], xy[:, 1])).ravel()

    def area(self, zone):
        a, b, c, d = self._box(zone)
        return (b - a) * (d - c)

    def specification(self):
        return dict(kind='mms', case=self.case, robin_h=self.ROBIN_H if self.robin else None, boundary='exact trace on Dirichlet sides')


class CableProblem(Problem):
    """Explicit cable cross section: every layer and the soil solve -div(k grad T) = Q."""
    reference_label = 'radial_exact (concentric verification only)'

    def __init__(self, source, mode='fixed'):
        if mode != 'fixed':
            raise NotImplementedError('Phase 4 supports fixed prescribed sources only; dc_temperature coupling is not implemented here')
        domain = source.domain(mode) if hasattr(source, 'domain') else FullDomain(source, mode)
        if domain.has_imperfect_contact:
            raise NotImplementedError('Imperfect contact is declared in the schema but no solver implements the temperature jump [T] = R q.n yet')
        self.domain = domain
        self.case = domain.case
        self.name = self.case['id']
        self.bounds = tuple(self.case['bounds'])
        self.regions = tuple(domain.regions)
        self.T_scale = self.case['scale']
        self.T_ref = self.case['T0']
        self.physics_sha256 = domain.fingerprint
        levels = []
        for bc in domain.side_conditions().values():
            levels.append(self.case['T0'] if bc.get('legacy') else bc['T_C'] if bc['type'] == 'dirichlet' else bc['T_inf_C'] if bc['type'] == 'robin' else None)
        levels = [v for v in levels if v is not None]
        self.T_level = min(levels) if levels else self.case['T0']
        self.region_power = {r.id: float(domain.source_integral(r)) for r in domain.regions}
        self.power_total = float(sum(self.region_power.values()))
        self.power_scale = abs(self.power_total)
        self.soil_strata = domain.nsoil > 1

    def amplitude(self, region, partition):
        if partition == 'global' or region.kind == 'soil':
            return self.T_scale
        return self.domain.scales(region)['temperature']

    def input_size(self, region, partition):
        if partition == 'global':
            return 2
        if region.kind == 'soil':
            return 2 + len(self.case['cables'])
        return 2 if region.kind == 'conductor' else 3

    def coordinates(self, region, xy, partition):
        if partition == 'global':
            return self.normalized_xy(xy)
        c = self.case
        if region.kind == 'soil':
            extras = [torch.log(torch.linalg.vector_norm(xy[:, :2] - xy.new_tensor(center), dim=1, keepdim=True) / c['radius']) for center in c['cables']]
            return torch.cat([self.normalized_xy(xy), *extras], dim=1)
        delta = xy[:, :2] - xy.new_tensor(c['cables'][region.cable])
        if region.kind == 'conductor':
            return delta / region.ro
        radius = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        eta = torch.log(radius / region.ri) / math.log(region.ro / region.ri)
        return torch.cat([2 * eta - 1, delta / radius], dim=1)

    def level_sets(self, xy):
        columns = []
        for interface in self.domain.interfaces:
            if interface.radius:
                center = xy.new_tensor(self.case['cables'][interface.cable])
                columns.append(torch.linalg.vector_norm(xy[:, :2] - center, dim=1, keepdim=True) - interface.radius)
            else:
                columns.append(xy[:, interface.axis:interface.axis + 1] - interface.position)
        return torch.cat(columns, dim=1)

    def conductivity(self, zone, xy):
        if zone.kind != 'soil':
            return xy[:, 0:1] * 0 + zone.k
        columns = _probe(zone, xy) if self.soil_strata else [xy[:, 0:1], xy[:, 1:2]]
        return field_k(self.case, columns[0], columns[1], torch)

    def source(self, zone, xy):
        return self.domain.source(zone, xy)

    def scales(self, zone):
        return self.domain.scales(zone)

    def sample_interior(self, zone, n, seed, policy='mixed'):
        return stratified_sample(self.domain, zone, n, seed, policy)

    def faces(self, n):
        result = []
        for interface in self.domain.interfaces:
            xy, normal, length = self.domain.interface_points(interface, n)
            result.append(Face(interface.name, interface.left, interface.right, xy, normal, length / n, -1 if interface.radius else interface.axis))
        return result

    def _bc(self, side, bc):
        if bc['type'] == 'dirichlet':
            return _dirichlet(lambda xy, side=side: self.domain.dirichlet_value(side, xy))
        if bc['type'] == 'neumann':
            return dict(type='neumann', value=lambda xy, v=bc['q_n_W_m2']: xy[:, 0:1] * 0 + v)
        return dict(type='robin', h=bc['h_W_m2K'], T_inf=lambda xy, v=bc['T_inf_C']: xy[:, 0:1] * 0 + v)

    def pieces(self, n):
        result = []
        for side in self.domain.boundary_sides(n):
            bc = self._bc(side['name'], side['bc'])
            xy, normal, weight = side['xy'], side['normal'], side['length'] / n
            if side['name'] == 'outer':
                result.append(Piece('outer', 0, xy, normal, weight, bc))
                continue
            for zone in self.regions[:self.domain.nsoil]:
                mask = (xy[:, zone.axis] > zone.lower) & (xy[:, zone.axis] <= zone.upper)
                if zone.id == 0:
                    mask |= xy[:, zone.axis] == zone.lower
                if np.any(mask):
                    result.append(Piece(side['name'], zone.id, xy[mask], normal[mask], weight, bc))
        return result

    def evaluation(self):
        return self.domain.evaluation_points()

    def reference(self, xy, labels=None):
        try:
            return radial_exact(self.domain, np.linalg.norm(xy, axis=1))
        except ValueError:
            return None

    def area(self, zone):
        return self.domain.area(zone)

    def specification(self):
        return self.domain.specification()


def named_case(name):
    """Catalog case, the concentric verification case, or a CaseDefinition builder from cases/verification.py."""
    builders = dict(coaxial_full=coaxial_case, coaxial_robin=coaxial_robin, coaxial_bc_dirichlet=coaxial_dirichlet,
        symmetric_full=lambda: symmetric_pair(False), symmetric_half=lambda: symmetric_pair(True))
    if name in builders:
        return builders[name]()
    catalog = cases()
    if name not in catalog:
        raise KeyError(f'Unknown case {name!r}')
    return catalog[name]


def load_problem(source, mode='fixed'):
    """Build a Problem from a catalog id, a legacy dictionary, a CaseDefinition or a ready Problem."""
    if isinstance(source, Problem):
        return source
    if isinstance(source, str):
        source = named_case(source)
    if isinstance(source, dict):
        return MMSProblem(source) if source['kind'].startswith('mms') else CableProblem(source, mode)
    if hasattr(source, 'physical') and not source.physical['kind'].startswith('mms'):
        return CableProblem(source, mode)
    if hasattr(source, 'physical'):
        return MMSProblem(source.physical)
    raise TypeError('Unsupported problem source')
