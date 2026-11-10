"""Geometry of pinn-v3-problem-1: region shapes as vectorised membership tests."""
import math

import numpy as np

SHAPES = ('rectangle', 'disk', 'annulus')


def contains(shape, x, y, tolerance=0.0):
    """Boolean mask of the points (x, y) that lie inside the shape (boundary included)."""
    kind = shape['type']
    if kind == 'rectangle':
        x0, x1, y0, y1 = shape['bounds']
        return (x >= x0 - tolerance) & (x <= x1 + tolerance) & (y >= y0 - tolerance) & (y <= y1 + tolerance)
    cx, cy = shape['center']
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    if kind == 'disk':
        return r2 <= (shape['radius_m'] + tolerance) ** 2
    return (r2 >= max(shape['inner_radius_m'] - tolerance, 0.0) ** 2) & (r2 <= (shape['outer_radius_m'] + tolerance) ** 2)


def boundary_distance(shape, x, y):
    """Distance to the nearest boundary of a region, in the same units as x and y."""
    kind = shape['type']
    if kind == 'rectangle':
        x0, x1, y0, y1 = shape['bounds']
        return np.minimum.reduce((abs(x - x0), abs(x1 - x), abs(y - y0), abs(y1 - y)))
    radius = np.sqrt((x - shape['center'][0]) ** 2 + (y - shape['center'][1]) ** 2)
    if kind == 'disk':
        return abs(radius - shape['radius_m'])
    return np.minimum(abs(radius - shape['inner_radius_m']), abs(radius - shape['outer_radius_m']))


def local_to_domain(shape, local):
    """Map normalized region coordinates to physical (x, y) coordinates.

    Rectangles use (u, v) in [0, 1]^2; disks use Cartesian coordinates in the
    unit disk; annuli use normalized radius and normalized angle in [0, 1]^2.
    """
    local = np.asarray(local, dtype=float)
    kind = shape['type']
    if kind == 'rectangle':
        x0, x1, y0, y1 = shape['bounds']
        return np.c_[x0 + local[:, 0] * (x1 - x0), y0 + local[:, 1] * (y1 - y0)]
    center = np.asarray(shape['center'], dtype=float)
    if kind == 'disk':
        return center + shape['radius_m'] * local[:, :2]
    radius = shape['inner_radius_m'] + local[:, 0] * (
        shape['outer_radius_m'] - shape['inner_radius_m'])
    angle = 2 * math.pi * local[:, 1]
    return center + np.c_[radius * np.cos(angle), radius * np.sin(angle)]


def domain_to_local(shape, xy):
    """Map physical (x, y) points to each shape's normalized region coordinates."""
    xy = np.asarray(xy, dtype=float)
    kind = shape['type']
    if kind == 'rectangle':
        x0, x1, y0, y1 = shape['bounds']
        return np.c_[(xy[:, 0] - x0) / (x1 - x0), (xy[:, 1] - y0) / (y1 - y0)]
    delta = xy - np.asarray(shape['center'], dtype=float)
    radius = np.linalg.norm(delta, axis=1)
    if kind == 'disk':
        return delta / shape['radius_m']
    angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2 * math.pi)
    radial = (radius - shape['inner_radius_m']) / (
        shape['outer_radius_m'] - shape['inner_radius_m'])
    return np.c_[radial, angle / (2 * math.pi)]


def interior_margin(shape, span, factor=1e-7):
    """Scale an interface exclusion margin locally and cap it for thin regions."""
    margin = factor * max(float(span), 1.0)
    kind = shape['type']
    if kind == 'rectangle':
        x0, x1, y0, y1 = shape['bounds']
        return min(margin, 0.1 * min(x1 - x0, y1 - y0))
    if kind == 'disk':
        return min(margin, 0.1 * shape['radius_m'])
    thickness = shape['outer_radius_m'] - shape['inner_radius_m']
    return min(margin, 0.25 * thickness)


RING_FRACTION = 0.05
BOUNDARY_FRACTION = 1e-3


def characteristic_size(shape):
    """Smallest interior distance of a region: side length, radius, or annulus thickness."""
    kind = shape['type']
    if kind == 'rectangle':
        x0, x1, y0, y1 = shape['bounds']
        return float(min(x1 - x0, y1 - y0))
    if kind == 'disk':
        return float(shape['radius_m'])
    return float(shape['outer_radius_m'] - shape['inner_radius_m'])


def ring_offset(shape):
    """Distance of the second interface ring from the shared curve, relative to the region size."""
    return RING_FRACTION * characteristic_size(shape)


def boundary_offset(shape):
    """Distance of the second boundary row inside the owning region, relative to the region size."""
    return BOUNDARY_FRACTION * characteristic_size(shape)


def sample_local(shape, count, rng, margin=0.0):
    """Sample a shape directly in its local coordinates, uniformly by physical area."""
    count = int(count)
    if count < 0:
        raise ValueError('Sample count must be non-negative')
    if count == 0:
        return np.empty((0, 2), dtype=float)
    kind = shape['type']
    if kind == 'rectangle':
        x0, x1, y0, y1 = shape['bounds']
        mx = min(float(margin) / (x1 - x0), 0.1)
        my = min(float(margin) / (y1 - y0), 0.1)
        local = np.c_[rng.uniform(mx, 1 - mx, count), rng.uniform(my, 1 - my, count)]
    elif kind == 'disk':
        radius = shape['radius_m'] - float(margin)
        if radius <= 0:
            raise RuntimeError('Disk has no interior after the local interface margin')
        radial = np.sqrt(rng.uniform(0.0, 1.0, count))
        angle = rng.uniform(0.0, 2 * math.pi, count)
        local = np.c_[radial * np.cos(angle), radial * np.sin(angle)]
        local *= radius / shape['radius_m']
    else:
        inner = shape['inner_radius_m'] + float(margin)
        outer = shape['outer_radius_m'] - float(margin)
        if outer <= inner:
            raise RuntimeError('Annulus has no interior after the local interface margin')
        radius = np.sqrt(rng.uniform(inner ** 2, outer ** 2, count))
        radial = (radius - shape['inner_radius_m']) / (
            shape['outer_radius_m'] - shape['inner_radius_m'])
        angle = rng.uniform(0.0, 2 * math.pi, count) / (2 * math.pi)
        local = np.c_[radial, angle]
    return local_to_domain(shape, local)


def inside_bounds(shape, bounds):
    """True when the whole shape lies inside the rectangle bounds [x0, x1, y0, y1]."""
    x0, x1, y0, y1 = bounds
    if shape['type'] == 'rectangle':
        a0, a1, b0, b1 = shape['bounds']
        return a0 >= x0 and a1 <= x1 and b0 >= y0 and b1 <= y1
    cx, cy = shape['center']
    radius = shape['radius_m'] if shape['type'] == 'disk' else shape['outer_radius_m']
    return cx - radius >= x0 and cx + radius <= x1 and cy - radius >= y0 and cy + radius <= y1
