"""Problem built from a validated pinn-v3-problem-1 document: region ownership, fields, boundaries and points."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pinn_v3.problem.fields import evaluate_field
from pinn_v3.problem.schema import validate_problem
from pinn_v3.problem.shapes import (boundary_distance, boundary_offset, contains, domain_to_local, interior_margin,
    local_to_domain, ring_offset, sample_local)

SIDE_NORMALS = {'bottom': (0., -1.), 'top': (0., 1.), 'left': (-1., 0.), 'right': (1., 0.)}
MAX_ATTEMPTS = 1000
MEMBERSHIP_TOLERANCE = 1e-8
INTERFACE_MARGIN_FACTOR = 1e-7


@dataclass(frozen=True)
class Problem:
    doc: dict

    @property
    def id(self):
        return self.doc['id']

    @property
    def bounds(self):
        return tuple(self.doc['domain']['bounds'])

    @property
    def regions(self):
        return self.doc['regions']

    def region_shape(self, region):
        """Geometry record for a region index or engine region carrying an integer id."""
        index = region if isinstance(region, (int, np.integer)) else region.id
        return self.regions[int(index)]['shape']

    def domain_to_local(self, region, xy):
        """Map physical points to normalized coordinates local to the selected region."""
        return domain_to_local(self.region_shape(region), xy)

    def local_to_domain(self, region, local):
        """Map normalized region coordinates back to physical domain coordinates."""
        return local_to_domain(self.region_shape(region), local)

    def owner(self, x, y):
        """Index of the owning region of each point; a later region overrides an earlier one."""
        index = np.full(np.shape(x), -1)
        x0, x1, y0, y1 = self.bounds
        tolerance = MEMBERSHIP_TOLERANCE * max(max(x1 - x0, y1 - y0), 1.0)
        order = sorted(range(len(self.regions)), key=lambda item: self.regions[item]['priority'])
        for position in order:
            region = self.regions[position]
            index[contains(region['shape'], x, y, tolerance)] = position
        return index

    def field(self, name, x, y):
        """Values of a field (conductivity, source, heat_capacity) at the points, from the owning region."""
        owner = self.owner(x, y)
        values = np.full(np.shape(x), np.nan)
        for position, region in enumerate(self.regions):
            mine = owner == position
            if not np.any(mine):
                continue
            spec = region.get(name, 0.0 if name == 'source' else None)
            if spec is None:
                raise KeyError(f"region {region['id']!r} has no {name}")
            values[mine] = evaluate_field(spec, x[mine], y[mine])
        return values

    def reference(self, x, y):
        """Exact field for comparison only; None when the problem declares no reference."""
        spec = self.doc.get('reference', {}).get('exact')
        return None if spec is None else evaluate_field(spec, x, y)

    def sample_interior(self, n_per_region, seed=0):
        """Sample each region in local coordinates; returns (xy, owner index)."""
        if n_per_region < 1:
            raise ValueError('n_per_region must be positive')
        rng = np.random.default_rng(seed)
        blocks, owners = [], []
        for position, region in enumerate(self.regions):
            blocks.append(self.sample_region_interior(position, n_per_region, rng))
            owners.append(np.full(n_per_region, position))
        return np.vstack(blocks), np.concatenate(owners)

    def sample_region_interior(self, region, count, rng):
        """Draw from the region's local shape and reject only points owned by higher-priority overlaps."""
        shape = self.region_shape(region)
        span = max(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 1.0)
        margin = interior_margin(shape, span, INTERFACE_MARGIN_FACTOR)
        index = region if isinstance(region, (int, np.integer)) else region.id
        accepted = []
        remaining = int(count)
        for _ in range(MAX_ATTEMPTS):
            candidates = sample_local(shape, max(64, 2 * remaining), rng, margin)
            owners = self.owner(candidates[:, 0], candidates[:, 1])
            owned = candidates[owners == index]
            if len(owned):
                take = owned[:remaining]
                accepted.append(take)
                remaining -= len(take)
                if remaining == 0:
                    return np.vstack(accepted)
        raise RuntimeError(f"region {self.regions[int(index)]['id']!r} owns too little of its local shape to sample {count} points")

    def sample_boundary(self, n_per_side):
        """Aggregate locally sampled boundary segments by side."""
        result = {}
        for side in self.doc['boundaries']:
            pieces = [piece for piece in self.sample_boundary_regions(n_per_side) if piece['side'] == side]
            if not pieces:
                result[side] = dict(xy=np.empty((0, 2)), normal=np.empty((0, 2)),
                                    owner=np.empty(0, dtype=int), condition=self.doc['boundaries'][side])
                continue
            result[side] = dict(
                xy=np.vstack([piece['xy'] for piece in pieces]),
                normal=np.vstack([piece['normal'] for piece in pieces]),
                owner=np.concatenate([np.full(len(piece['xy']), piece['region']) for piece in pieces]),
                condition=self.doc['boundaries'][side])
        return result

    def sample_boundary_regions(self, n_per_piece, rows=2):
        """Sample each locally owned boundary segment separately, returning its physical length and owner.

        rows=2 (default, used by the PINN loss) adds one row inside the owning region for each segment;
        rows=1 returns only the physical boundary (used by the RBF-FD solver).
        """
        if rows not in (1, 2):
            raise ValueError('rows must be 1 or 2')
        if n_per_piece < 1:
            raise ValueError('n_per_piece must be positive')
        if self.doc['domain']['shape'] == 'annulus':
            domain = self.doc['domain']
            center = np.asarray(domain['center'], dtype=float)
            result = []
            for side, radius, sign in (('inner', domain['inner_radius_m'], -1),
                                       ('outer', domain['outer_radius_m'], 1)):
                angle = 2 * np.pi * (np.arange(n_per_piece) + 0.5) / n_per_piece
                directions = np.c_[np.cos(angle), np.sin(angle)]
                xy = center + radius * directions
                normals = sign * directions
                owners = self.owner(xy[:, 0], xy[:, 1])
                for region_id in np.unique(owners[owners >= 0]):
                    selected = owners == region_id
                    rows_xy, rows_normal = self._boundary_rows(int(region_id), xy[selected], normals[selected], rows)
                    result.append(dict(side=side, region=int(region_id), xy=rows_xy,
                        normal=rows_normal, length=2 * np.pi * radius * np.count_nonzero(selected) / n_per_piece,
                        condition=self.doc['boundaries'][side]))
            return result

        x0, x1, y0, y1 = self.bounds
        result = {}
        side_geometry = {
            'bottom': (1, y0, x0, x1), 'top': (1, y1, x0, x1),
            'left': (0, x0, y0, y1), 'right': (0, x1, y0, y1),
        }
        for side, (fixed_axis, fixed, tangent0, tangent1) in side_geometry.items():
            cuts = {float(tangent0), float(tangent1)}
            for region in self.regions:
                shape = region['shape']
                if shape['type'] == 'rectangle':
                    a0, a1, b0, b1 = shape['bounds']
                    touches = (fixed_axis == 0 and np.isclose(fixed, a0 if side == 'left' else a1)
                               if fixed_axis == 0 else
                               np.isclose(fixed, b0 if side == 'bottom' else b1))
                    if touches:
                        cuts.update((max(tangent0, b0 if fixed_axis == 0 else a0),
                                     min(tangent1, b1 if fixed_axis == 0 else a1)))
                else:
                    center = shape['center']
                    fixed_center = center[fixed_axis]
                    tangent_center = center[1 - fixed_axis]
                    offset = fixed - fixed_center
                    radii = ([shape['radius_m']] if shape['type'] == 'disk' else
                             [shape['inner_radius_m'], shape['outer_radius_m']])
                    for radius in radii:
                        if abs(offset) <= radius:
                            half = np.sqrt(max(radius ** 2 - offset ** 2, 0.0))
                            cuts.update((max(tangent0, tangent_center - half),
                                         min(tangent1, tangent_center + half)))
            ordered = sorted(value for value in cuts if tangent0 <= value <= tangent1)
            for lower, upper in zip(ordered, ordered[1:]):
                if upper - lower <= np.finfo(float).eps * max(abs(lower), abs(upper), 1.0):
                    continue
                midpoint = (lower + upper) / 2
                probe = np.array([[fixed, midpoint]]) if fixed_axis == 0 else np.array([[midpoint, fixed]])
                region_id = int(self.owner(probe[:, 0], probe[:, 1])[0])
                if region_id < 0:
                    continue
                tangent = lower + (np.arange(n_per_piece) + 0.5) * (upper - lower) / n_per_piece
                if fixed_axis == 0:
                    xy = np.c_[np.full(n_per_piece, fixed), tangent]
                else:
                    xy = np.c_[tangent, np.full(n_per_piece, fixed)]
                rows_xy, rows_normal = self._boundary_rows(region_id, xy, np.tile(SIDE_NORMALS[side], (n_per_piece, 1)), rows)
                result.setdefault(side, []).append(dict(side=side, region=region_id, xy=rows_xy,
                    normal=rows_normal, length=upper - lower,
                    condition=self.doc['boundaries'][side]))
        return [piece for pieces in result.values() for piece in pieces]

    def region_boundary_curves(self, region, n_per_curve):
        """Return local shape boundary samples as (xy, outward-normal, physical-length) curves."""
        if n_per_curve < 1:
            raise ValueError('n_per_curve must be positive')
        shape = self.region_shape(region)
        if shape['type'] == 'rectangle':
            x0, x1, y0, y1 = shape['bounds']
            s = (np.arange(n_per_curve) + 0.5) / n_per_curve
            ones = np.ones(n_per_curve)
            return [
                (np.c_[x0 + (x1 - x0) * s, y0 * ones], np.tile((0., -1.), (n_per_curve, 1)), x1 - x0),
                (np.c_[x0 + (x1 - x0) * s, y1 * ones], np.tile((0., 1.), (n_per_curve, 1)), x1 - x0),
                (np.c_[x0 * ones, y0 + (y1 - y0) * s], np.tile((-1., 0.), (n_per_curve, 1)), y1 - y0),
                (np.c_[x1 * ones, y0 + (y1 - y0) * s], np.tile((1., 0.), (n_per_curve, 1)), y1 - y0),
            ]
        center = np.asarray(shape['center'], dtype=float)
        angle = 2 * np.pi * (np.arange(n_per_curve) + 0.5) / n_per_curve
        radial = np.c_[np.cos(angle), np.sin(angle)]
        if shape['type'] == 'disk':
            radius = shape['radius_m']
            return [(center + radius * radial, radial, 2 * np.pi * radius)]
        inner, outer = shape['inner_radius_m'], shape['outer_radius_m']
        return [(center + outer * radial, radial, 2 * np.pi * outer),
                (center + inner * radial, -radial, 2 * np.pi * inner)]

    def _boundary_rows(self, region_id, xy, normal, rows=2):
        """Boundary segment rows: the physical boundary, plus (if rows == 2) one row inside the owning region."""
        if rows == 1:
            return xy, normal
        inner = xy - boundary_offset(self.region_shape(region_id)) * normal
        keep = self.owner(inner[:, 0], inner[:, 1]) == region_id
        return np.vstack([xy, inner[keep]]), np.vstack([normal, normal[keep]])

    def _declared_interfaces(self, n_per_curve):
        """Planar interfaces declared with the same region on both sides: two rings inside that region."""
        index_of = {region['id']: index for index, region in enumerate(self.regions)}
        result = []
        for declared in self.doc.get('interfaces', ()):
            left_id, right_id = declared['regions']
            if left_id != right_id:
                continue
            region_id = index_of[left_id]
            shape = self.region_shape(region_id)
            if shape['type'] != 'rectangle':
                raise ValueError(f"declared interface {declared['id']!r} requires a rectangular region")
            x0, x1, y0, y1 = shape['bounds']
            axis = 0 if declared['axis'] == 'x' else 1
            lower, upper = (y0, y1) if axis == 0 else (x0, x1)
            tangent = lower + (upper - lower) * (np.arange(n_per_curve) + 0.5) / n_per_curve
            position = np.full(n_per_curve, float(declared['position']))
            if axis == 0:
                xy, normal = np.c_[position, tangent], np.tile([1., 0.], (n_per_curve, 1))
            else:
                xy, normal = np.c_[tangent, position], np.tile([0., 1.], (n_per_curve, 1))
            delta = ring_offset(shape)
            left_ring, right_ring = xy - delta * normal, xy + delta * normal
            keep = ((self.owner(left_ring[:, 0], left_ring[:, 1]) == region_id)
                    & (self.owner(right_ring[:, 0], right_ring[:, 1]) == region_id))
            if not np.any(keep):
                raise ValueError(f"declared interface {declared['id']!r} lies outside its region")
            result.append(dict(name=f"{declared['id']}@declared", left=region_id, right=region_id,
                xy=left_ring[keep], xy_right=right_ring[keep], normal=normal[keep],
                weight=(upper - lower) / n_per_curve))
        return result

    def sample_interfaces(self, n_per_curve):
        """Two concentric rings per interface, one inside each adjacent region at equal distance from the shared curve.

        Each interface is emitted from the curve(s) that carry it: a segment that is a boundary of both regions
        is emitted once, from the lower-index side. xy lies inside the left region and xy_right inside the right
        region, so each side is evaluated in its own region.
        """
        span = max(self.bounds[1] - self.bounds[0], self.bounds[3] - self.bounds[2], 1.0)
        tolerance = MEMBERSHIP_TOLERANCE * span
        probe = max(1e-7, 2 * tolerance)
        result = []
        for region_id, region in enumerate(self.regions):
            for curve_index, (xy, normal, length) in enumerate(self.region_boundary_curves(region_id, n_per_curve)):
                outside = self.owner(*(xy + probe * normal).T)
                for neighbor in np.unique(outside[(outside >= 0) & (outside != region_id)]):
                    neighbor = int(neighbor)
                    # A segment that is a boundary of both regions is emitted once, from the lower-index side;
                    # a segment that is a boundary of only this region (e.g. a layer inside a larger region) is emitted here.
                    on_neighbor = boundary_distance(self.region_shape(neighbor), xy[:, 0], xy[:, 1]) <= tolerance
                    selected = (outside == neighbor) & (~on_neighbor | (region_id < neighbor))
                    delta = min(ring_offset(self.region_shape(region_id)), ring_offset(self.region_shape(neighbor)))
                    left_ring = xy - delta * normal
                    right_ring = xy + delta * normal
                    keep = (selected
                        & (self.owner(left_ring[:, 0], left_ring[:, 1]) == region_id)
                        & (self.owner(right_ring[:, 0], right_ring[:, 1]) == neighbor))
                    if not np.any(keep):
                        continue
                    result.append(dict(name=f"{region['id']}->{self.regions[neighbor]['id']}@{curve_index}",
                        left=region_id, right=neighbor, xy=left_ring[keep], xy_right=right_ring[keep],
                        normal=normal[keep], weight=length / n_per_curve))
        result.extend(self._declared_interfaces(n_per_curve))
        return result


def build_problem(doc):
    """Validate a document and wrap it as a Problem."""
    return Problem(validate_problem(doc))


def load_problem(path):
    """Read problems/<id>/problem.json and build the Problem."""
    return build_problem(json.loads(Path(path).read_text(encoding='utf-8')))
