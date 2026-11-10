"""Local polyharmonic RBF-FD weights for two-dimensional stencils."""

import numpy as np
import torch
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from pinn_v3.problem.fields import evaluate_field


def derivative_weights(center, nodes):
    """Return weights for x, y and Laplacian derivatives at center.

    Uses the cubic polyharmonic spline and all polynomials through degree two.
    The caller is responsible for selecting nodes from the appropriate material.
    """
    center = np.asarray(center, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    if center.shape != (2,) or nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError('center and nodes must have shapes (2,) and (n, 2)')
    if len(nodes) < 6 or not np.isfinite(nodes).all() or not np.isfinite(center).all():
        raise ValueError('at least six finite stencil nodes are required')
    offsets = nodes - center
    _, singular_values, axes = np.linalg.svd(offsets, full_matrices=False)
    scales = singular_values / np.sqrt(len(nodes))
    if scales[0] == 0 or scales[-1] <= scales[0] * 1e-12:
        raise ValueError('stencil cannot reproduce quadratic polynomials')
    transform = axes / scales[:, None]
    offsets = offsets @ transform.T
    x, y = offsets[:, 0], offsets[:, 1]
    polynomial = np.column_stack((np.ones(len(nodes)), x, y, x*x, x*y, y*y))
    if np.linalg.matrix_rank(polynomial) < 6:
        raise ValueError('stencil cannot reproduce quadratic polynomials')
    distances = cdist(offsets, offsets)
    system = np.block([[distances**3, polynomial],
                       [polynomial.T, np.zeros((6, 6))]])
    radius = np.linalg.norm(offsets, axis=1)
    rhs = np.zeros((len(nodes) + 6, 3))
    metric = transform @ transform.T
    rhs[:len(nodes), :2] = (-3 * radius[:, None] * offsets) @ transform
    quadratic = np.einsum('ni,ij,nj->n', offsets, metric, offsets)
    rhs[:len(nodes), 2] = 3 * radius * np.trace(metric) + np.divide(
        3 * quadratic, radius, out=np.zeros_like(radius), where=radius > 0)
    rhs[len(nodes) + 1:len(nodes) + 3, :2] = transform
    rhs[len(nodes) + 3:len(nodes) + 6, 2] = [2 * metric[0, 0], 2 * metric[0, 1], 2 * metric[1, 1]]
    weights = solve(system, rhs, assume_a='sym')[:len(nodes)]
    return weights


def annulus_derivative_weights(problem, region_id, center, nodes):
    """Use normalized polar coordinates and return physical x/y/Laplacian weights."""
    shape = problem.regions[region_id]['shape']
    local_center = problem.domain_to_local(region_id, center[None, :])[0]
    local_nodes = problem.domain_to_local(region_id, nodes)
    offsets = local_nodes - local_center
    offsets[:, 1] = (offsets[:, 1] + 0.5) % 1. - 0.5
    scale = np.linalg.norm(offsets, axis=1).max()
    if scale <= 0:
        raise ValueError('stencil has zero extent')
    offsets = offsets / scale
    x, y = offsets.T
    polynomial = np.column_stack((np.ones(len(nodes)), x, y, x*x, x*y, y*y))
    if np.linalg.matrix_rank(polynomial) < 6:
        raise ValueError('stencil cannot reproduce quadratic polynomials')
    distances = cdist(offsets, offsets)
    system = np.block([[distances**3, polynomial], [polynomial.T, np.zeros((6, 6))]])
    radius = np.linalg.norm(offsets, axis=1)
    inverse = np.divide(1., radius, out=np.zeros_like(radius), where=radius > 0)
    rhs = np.zeros((len(nodes) + 6, 5))
    rhs[:len(nodes), 0] = -3 * radius * x
    rhs[:len(nodes), 1] = -3 * radius * y
    rhs[:len(nodes), 2] = 3 * radius + 3 * x*x * inverse
    rhs[:len(nodes), 3] = 3 * x*y * inverse
    rhs[:len(nodes), 4] = 3 * radius + 3 * y*y * inverse
    rhs[len(nodes) + 1, 0] = rhs[len(nodes) + 2, 1] = 1.
    rhs[len(nodes) + 3, 2] = rhs[len(nodes) + 5, 4] = 2.
    rhs[len(nodes) + 4, 3] = 1.
    chart = solve(system, rhs, assume_a='sym')[:len(nodes)]
    chart /= np.array([scale, scale, scale**2, scale**2, scale**2])
    displacement = center - np.asarray(shape['center'], dtype=float)
    physical_radius = np.linalg.norm(displacement)
    radial = displacement / physical_radius
    tangent = np.array([-radial[1], radial[0]])
    thickness = shape['outer_radius_m'] - shape['inner_radius_m']
    gradient = chart[:, 0, None] * radial / thickness + chart[:, 1, None] * tangent / (2 * np.pi * physical_radius)
    laplacian = (chart[:, 2] / thickness**2 + chart[:, 0] / (thickness * physical_radius) +
        chart[:, 4] / (2 * np.pi * physical_radius)**2)
    return np.column_stack((gradient, laplacian))


def build_region_nodes(problem, interior_per_region=100, surface_count=24, seed=11, interior_points=None):
    """Build interior nodes and paired, region-labelled interface nodes."""
    if interior_per_region < 6 or surface_count < 6:
        raise ValueError('at least six interior and surface points are required')
    coordinates, labels, kinds = [], [], []
    boundaries, interfaces, anchors = [], [], {}
    rng = np.random.default_rng(seed)

    def append(points, region_id, kind):
        first = len(coordinates)
        coordinates.extend(np.asarray(points, dtype=float))
        labels.extend([region_id] * len(points))
        kinds.extend([kind] * len(points))
        return np.arange(first, len(coordinates), dtype=int)

    for region_id, region in enumerate(problem.regions):
        if interior_points is None:
            points = problem.sample_region_interior(region_id, interior_per_region, rng)
        else:
            points = np.asarray(interior_points[region_id], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError(f'invalid collocation points for region {region_id}')
            if np.any(problem.owner(points[:, 0], points[:, 1]) != region_id):
                raise ValueError(f'collocation points do not belong to region {region_id}')
        anchors[region_id] = append(points, region_id, 'interior')
        shape = region['shape']
        if shape['type'] == 'annulus':
            center = np.asarray(shape['center'], dtype=float)
            inner, outer = shape['inner_radius_m'], shape['outer_radius_m']
            for layer in range(1, 5):
                radius = inner + (outer - inner) * layer / 5
                angle = 2 * np.pi * (np.arange(surface_count) + 0.5 * (layer % 2)) / surface_count
                layer_points = center + radius * np.column_stack((np.cos(angle), np.sin(angle)))
                owned = problem.owner(layer_points[:, 0], layer_points[:, 1]) == region_id
                append(layer_points[owned], region_id, 'interior')

    for piece in problem.sample_boundary_regions(surface_count, rows=1):
        indices = append(piece['xy'], piece['region'], 'boundary')
        boundaries.append((indices, piece))
    for face in problem.sample_interfaces(surface_count):
        # The two rings are symmetric about the shared curve; RBF-FD enforces continuity on that curve.
        shared_xy = 0.5 * (face['xy'] + face['xy_right'])
        span = max(problem.bounds[1] - problem.bounds[0], problem.bounds[3] - problem.bounds[2])
        for region_id, direction in ((face['left'], -1.), (face['right'], 1.)):
            shape = problem.regions[region_id]['shape']
            distance = span * 0.002
            if shape['type'] == 'annulus':
                distance = min(distance, (shape['outer_radius_m'] - shape['inner_radius_m']) / 15)
            elif shape['type'] == 'disk':
                distance = min(distance, shape['radius_m'] / 15)
            factors = (1, 2, 4, 8, 16, 32, 64, 128) if shape['type'] == 'rectangle' else (1, 2, 4, 5)
            for factor in factors:
                collar = shared_xy + direction * distance * factor * face['normal']
                owned = problem.owner(collar[:, 0], collar[:, 1]) == region_id
                append(collar[owned], region_id, 'collar')
        left = append(shared_xy, face['left'], 'interface')
        right = append(shared_xy, face['right'], 'interface')
        interfaces.append((left, right, face['normal']))
    return dict(xy=np.asarray(coordinates), labels=np.asarray(labels, dtype=int),
                kinds=np.asarray(kinds), boundaries=boundaries, interfaces=interfaces, anchors=anchors)


def check_maximum_principle(problem, temperature):
    """Reject gross violations where constant-sign sources and Dirichlet data imply bounds."""
    boundaries = list(problem.doc['boundaries'].values())
    sources = [region.get('source', 0.) for region in problem.regions]
    if any(bc['type'] != 'dirichlet' or not isinstance(bc['value'], (int, float)) for bc in boundaries):
        return
    if not all(isinstance(source, (int, float)) for source in sources):
        return
    values = np.asarray(temperature, dtype=float)
    bounds = [float(bc['value']) for bc in boundaries]
    tolerance = 1e-3 * max(1., max(abs(value) for value in bounds),
                           float(problem.doc.get('temperature_scale_K', 1.)))
    if all(source >= 0 for source in sources) and values.min() < min(bounds) - tolerance:
        raise ValueError(f'RBF-FD violates the lower temperature bound: {values.min():.6g}')
    if all(source <= 0 for source in sources) and values.max() > max(bounds) + tolerance:
        raise ValueError(f'RBF-FD violates the upper temperature bound: {values.max():.6g}')


def solve_regions(problem, interior_per_region=100, surface_count=24, stencil_size=21, seed=11, interior_points=None):
    """Solve a V3 problem using one-sided material stencils and interface rows."""
    if stencil_size < 6:
        raise ValueError('stencil_size must be at least six')
    nodes = build_region_nodes(problem, interior_per_region, surface_count, seed, interior_points)
    points, labels, kinds = nodes['xy'], nodes['labels'], nodes['kinds']
    matrix = lil_matrix((len(points), len(points)), dtype=float)
    right = np.zeros(len(points))
    region_indices = [np.flatnonzero(labels == region_id) for region_id in range(len(problem.regions))]
    def local_features(region_id, xy):
        local = problem.domain_to_local(region_id, xy)
        shape = problem.regions[region_id]['shape']
        if shape['type'] == 'annulus':
            angle = 2 * np.pi * local[:, 1]
            return np.column_stack((local[:, 0], np.cos(angle), np.sin(angle)))
        return local

    trees = [cKDTree(local_features(region_id, points[indices]))
        for region_id, indices in enumerate(region_indices)]
    core_indices = [indices[kinds[indices] == 'interior'] if problem.regions[region_id]['shape']['type'] == 'rectangle'
                    else indices[kinds[indices] != 'collar']
                    for region_id, indices in enumerate(region_indices)]
    core_trees = [cKDTree(local_features(region_id, points[indices]))
        for region_id, indices in enumerate(core_indices)]

    def stencil(point, region_id):
        region_nodes = region_indices[region_id]
        count = min(stencil_size, len(region_nodes))
        if count < 6:
            raise ValueError(f'region {region_id} has too few nodes')
        query = local_features(region_id, point[None, :])[0]
        nearest = np.atleast_1d(region_nodes[trees[region_id].query(query, k=count)[1]])
        core_count = min(6, count - 1, len(core_indices[region_id]))
        core = np.atleast_1d(core_indices[region_id][core_trees[region_id].query(query, k=core_count)[1]])
        required = list(dict.fromkeys([int(nearest[0]), *map(int, core)]))
        indices = np.array((required + [int(index) for index in nearest if int(index) not in required])[:count])
        if problem.regions[region_id]['shape']['type'] == 'annulus':
            weights = annulus_derivative_weights(problem, region_id, point, points[indices])
        else:
            weights = derivative_weights(point, points[indices])
        return indices, weights

    def field(spec, point):
        return float(evaluate_field(spec, point[:1], point[1:])[0])

    span = max(problem.bounds[1] - problem.bounds[0], problem.bounds[3] - problem.bounds[2])
    delta = np.sqrt(np.finfo(float).eps) * span
    for index in np.flatnonzero((kinds == 'interior') | (kinds == 'collar')):
        region = problem.regions[labels[index]]
        point = points[index]
        indices, weights = stencil(point, labels[index])
        conductivity = region['conductivity']
        local_k = field(conductivity, point)
        gradient = np.array([(field(conductivity, point + delta * axis) -
            field(conductivity, point - delta * axis)) / (2 * delta) for axis in np.eye(2)])
        matrix[index, indices] = -(local_k * weights[:, 2] + weights[:, :2] @ gradient)
        right[index] = field(region.get('source', 0.), point)
    for indices, piece in nodes['boundaries']:
        condition = piece['condition']
        for index, normal in zip(indices, piece['normal']):
            point = points[index]
            kind = condition['type']
            if kind == 'dirichlet':
                matrix[index, index] = 1.
                right[index] = field(condition['value'], point)
            else:
                neighbors, weights = stencil(point, labels[index])
                local_k = field(problem.regions[labels[index]]['conductivity'], point)
                matrix[index, neighbors] = -local_k * (weights[:, :2] @ normal)
                if kind == 'neumann':
                    right[index] = field(condition['q_n_W_m2'], point)
                elif kind == 'robin':
                    h = condition['h_W_m2K']
                    matrix[index, index] -= h
                    right[index] = -h * field(condition['T_inf_C'], point)
                else:
                    raise ValueError(f'Unsupported boundary type: {kind}')
    for left, right_indices, normals in nodes['interfaces']:
        for left_index, right_index, normal in zip(left, right_indices, normals):
            matrix[left_index, left_index] = 1.
            matrix[left_index, right_index] = -1.
            for index, sign in ((left_index, 1.), (right_index, -1.)):
                region_id = labels[index]
                neighbors, weights = stencil(points[index], region_id)
                local_k = field(problem.regions[region_id]['conductivity'], points[index])
                matrix[right_index, neighbors] = sign * local_k * (weights[:, :2] @ normal)
    matrix = matrix.tocsr()
    row_scale = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    if np.any(row_scale == 0):
        raise ValueError('RBF-FD system has an empty equation')
    scaled = matrix.multiply((1. / row_scale)[:, None]).tocsc()
    temperature = splu(scaled).solve(right / row_scale)
    residual = np.linalg.norm(scaled @ temperature - right / row_scale, ord=np.inf)
    if not np.isfinite(temperature).all() or not np.isfinite(residual):
        raise ValueError('RBF-FD solve returned non-finite results')
    if residual > 1e-8:
        raise ValueError(f'RBF-FD scaled residual exceeds tolerance: {residual:.6g}')
    check_maximum_principle(problem, temperature)
    return dict(**nodes, temperature=temperature, scaled_residual_inf=float(residual))


def pretrain_pinn(model, result, steps=100, learning_rate=1e-3, target_points=None):
    """Fit temperature anchors uniformly, optionally extending them within each material."""
    if int(steps) != steps or steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('pretraining requires positive steps and learning rate')
    reference = next(model.parameters())
    region_data = []
    for region in model.problem.regions:
        indices = result['anchors'][region.id]
        if not len(indices):
            raise ValueError(f'no RBF-FD anchors for region {region.id}')
        anchor_xy = result['xy'][indices]
        anchor_temperature = result['temperature'][indices]
        if target_points is None:
            xy, values = anchor_xy, anchor_temperature
        else:
            xy = np.asarray(target_points[region.id], dtype=float)
            region_indices = np.flatnonzero(result['labels'] == region.id)
            nearest = cKDTree(result['xy'][region_indices]).query(xy)[1]
            values = result['temperature'][region_indices[nearest]]
        region_data.append((region, torch.as_tensor(xy, device=reference.device, dtype=reference.dtype),
            torch.as_tensor(values, device=reference.device, dtype=reference.dtype).reshape(-1, 1)))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        terms = []
        for region, xy, values in region_data:
            points = xy.detach().requires_grad_(True)
            temperature, _ = model.field(region, points)
            terms.append(((temperature - values) / model.problem.T_scale).square().sum())
        loss = sum(terms) / sum(len(xy) for _, xy, _ in region_data)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('non-finite RBF-FD pretraining loss')
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    return dict(steps=int(steps), points=sum(len(xy) for _, xy, _ in region_data),
                initial_loss=losses[0], final_loss=losses[-1])


def solve_single_rectangle(problem, interior_per_axis=14, boundary_per_side=16, stencil_size=21):
    """Solve a single-material rectangular V3 problem with RBF-FD on CPU.

    This initial solver deliberately rejects multi-region and annular domains.
    """
    doc = problem.doc
    if doc['domain']['shape'] != 'rectangle' or len(doc['regions']) != 1:
        raise NotImplementedError('single rectangular region required')
    region = doc['regions'][0]
    shape = region['shape']
    if shape['type'] != 'rectangle' or list(shape['bounds']) != list(problem.bounds):
        raise NotImplementedError('region must cover the rectangular domain')
    if interior_per_axis < 3 or boundary_per_side < 3 or stencil_size < 6:
        raise ValueError('insufficient nodes or stencil size')
    x0, x1, y0, y1 = problem.bounds
    xs = x0 + (np.arange(interior_per_axis) + 0.5) * (x1 - x0) / interior_per_axis
    ys = y0 + (np.arange(interior_per_axis) + 0.5) * (y1 - y0) / interior_per_axis
    grid_x, grid_y = np.meshgrid(xs, ys)
    interior = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    boundaries = problem.sample_boundary_regions(boundary_per_side, rows=1)
    points = np.vstack([interior] + [piece['xy'] for piece in boundaries])
    if stencil_size > len(points):
        raise ValueError('stencil exceeds the number of nodes')
    tree = cKDTree(points)
    nearest = tree.query(points, k=stencil_size)[1]
    matrix = lil_matrix((len(points), len(points)), dtype=float)
    right = np.zeros(len(points))
    conductivity = region['conductivity']
    x, y = interior.T
    k = evaluate_field(conductivity, x, y)
    source = evaluate_field(region.get('source', 0.), x, y)
    delta = np.sqrt(np.finfo(float).eps) * max(x1 - x0, y1 - y0)
    grad_x = (evaluate_field(conductivity, x + delta, y) -
              evaluate_field(conductivity, x - delta, y)) / (2 * delta)
    grad_y = (evaluate_field(conductivity, x, y + delta) -
              evaluate_field(conductivity, x, y - delta)) / (2 * delta)
    for index, point in enumerate(interior):
        indices = nearest[index]
        weights = derivative_weights(point, points[indices])
        matrix[index, indices] = -(k[index] * weights[:, 2] +
            grad_x[index] * weights[:, 0] + grad_y[index] * weights[:, 1])
        right[index] = source[index]
    offset = len(interior)
    for piece in boundaries:
        condition = piece['condition']
        for local_index, (point, normal) in enumerate(zip(piece['xy'], piece['normal'])):
            index = offset + local_index
            bc_type = condition['type']
            if bc_type == 'dirichlet':
                matrix[index, index] = 1.
                right[index] = evaluate_field(condition['value'], point[:1], point[1:])[0]
                continue
            indices = nearest[index]
            weights = derivative_weights(point, points[indices])
            local_k = evaluate_field(conductivity, point[:1], point[1:])[0]
            matrix[index, indices] = -local_k * (normal[0] * weights[:, 0] + normal[1] * weights[:, 1])
            if bc_type == 'neumann':
                right[index] = evaluate_field(condition['q_n_W_m2'], point[:1], point[1:])[0]
            elif bc_type == 'robin':
                h = condition['h_W_m2K']
                matrix[index, index] -= h
                right[index] = -h * evaluate_field(condition['T_inf_C'], point[:1], point[1:])[0]
            else:
                raise ValueError(f'Unsupported boundary type: {bc_type}')
        offset += len(piece['xy'])
    matrix = matrix.tocsr()
    row_scale = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    if np.any(row_scale == 0):
        raise ValueError('RBF-FD system has an empty equation')
    scaled = matrix.multiply((1. / row_scale)[:, None]).tocsc()
    solution = splu(scaled).solve(right / row_scale)
    residual = np.linalg.norm(scaled @ solution - right / row_scale, ord=np.inf)
    if not np.isfinite(solution).all() or not np.isfinite(residual):
        raise ValueError('RBF-FD solve returned non-finite results')
    return dict(xy=points, temperature=solution, labels=np.zeros(len(points), dtype=int),
                scaled_residual_inf=float(residual), interior_count=len(interior))