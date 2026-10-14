"""FEniCSx reference for the manufactured unit-square benchmarks V0 (homogeneous), V1 (flat interface)
and the Robin variant. Run inside the FEniCSx environment; the exact solution is used only for the error.

Dirichlet data are the exact trace on every side, except mms_robin where the top side carries
q.n = h (T - T_inf) with h = 10 (never turned into a Dirichlet condition).
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time
import numpy as np
import dolfinx
import gmsh
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from pinn_v2.cases.catalog import field_k, source, exact, evaluation_points, fingerprint, interfaces
from pinn_v2.cases.boundary import parse_annulus_boundary_conditions
from pinn_v2.cases.definition import CaseDefinition, load_all
from pinn_v2.solvers.fem.backend import UFLBackend
from pinn_v2.manifest.provenance import archive_sources
from pinn_v2.solvers.fem.mesh_resolution import mesh_size_factor, scaled_cells_per_side, validate_node_multiplier

ROBIN_H = 10.0
TOP_TAG, OTHER_TAG = 3, 1


def evaluate(function, xy, vector=False):
    points = np.c_[xy, np.zeros(len(xy))]
    msh = function.function_space.mesh
    tree = geometry.bb_tree(msh, 2)
    colliding = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, points), points)
    cells = np.array([colliding.links(i)[0] if len(colliding.links(i)) else -1 for i in range(len(xy))], dtype=np.int32)
    if np.any(cells < 0):
        raise ValueError(f'{int(np.sum(cells < 0))} evaluation points outside the FEM mesh')
    values = function.eval(points, cells)
    return values if vector else values.reshape(-1)


def solve_annulus(source_case, c, level, output, node_multiplier=1.):
    if MPI.COMM_WORLD.size != 1:
        raise ValueError('Reference currently requires one MPI rank')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{c['id']}_l{level}"
    if output.joinpath(stem + '.json').exists():
        raise FileExistsError('Use a new directory for a new reference run')
    raw_bcs = getattr(source_case, 'boundary_conditions', None)
    if raw_bcs is None and isinstance(source_case, dict) and 'physical' in source_case:
        raw_bcs = source_case.get('boundary_conditions')
    if raw_bcs is None:
        flux = -c['power'] / (2 * np.pi * c['ri'])
        raw_bcs = dict(inner=dict(type='neumann', heat_flux_out=dict(value=flux, unit='W/m2')),
                       outer=dict(type='dirichlet', temperature=dict(value=c['T0'], unit='degC')))
    conditions = parse_annulus_boundary_conditions(raw_bcs)

    node_multiplier = validate_node_multiplier(node_multiplier)
    size_factor = mesh_size_factor(node_multiplier)
    start = time.perf_counter()
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add(c['id'] + '_annulus')
        occ = gmsh.model.occ
        outer = occ.addDisk(0, 0, 0, c['ro'], c['ro'])
        inner = occ.addDisk(0, 0, 0, c['ri'], c['ri'])
        occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)
        occ.synchronize()
        surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
        if len(surfaces) != 1:
            raise RuntimeError(f'Expected one annular surface, got {len(surfaces)}')
        curves = [tag for dim, tag in gmsh.model.getBoundary([(2, surfaces[0])], combined=True, oriented=False) if dim == 1]
        inner_curves, outer_curves = [], []
        radius_tolerance = 1e-6 * max(c['ro'], 1.)
        for tag in curves:
            bounds = gmsh.model.getBoundingBox(1, tag)
            radius = max(abs(bounds[0]), abs(bounds[1]), abs(bounds[3]), abs(bounds[4]))
            if np.isclose(radius, c['ri'], rtol=0, atol=radius_tolerance):
                inner_curves.append(tag)
            elif np.isclose(radius, c['ro'], rtol=0, atol=radius_tolerance):
                outer_curves.append(tag)
        if not inner_curves or not outer_curves:
            raise RuntimeError('Could not identify the inner and outer annulus boundaries')
        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, 'homogeneous_material')
        gmsh.model.addPhysicalGroup(1, inner_curves, 11)
        gmsh.model.addPhysicalGroup(1, outer_curves, 12)
        distance = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(distance, 'CurvesList', inner_curves)
        gmsh.model.mesh.field.setNumber(distance, 'Sampling', 256)
        threshold = gmsh.model.mesh.field.add('Threshold')
        for key, value in [('InField', distance), ('SizeMin', c['ri'] / (3 * 2 ** level) * size_factor),
                   ('SizeMax', c['ro'] / (12 * 2 ** level) * size_factor), ('DistMin', c['ri']),
                           ('DistMax', c['ro'] / 4)]:
            gmsh.model.mesh.field.setNumber(threshold, key, value)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        msh, facets = data.mesh, data.facet_tags
    finally:
        gmsh.finalize()

    V = fem.functionspace(msh, ('Lagrange', 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    dx = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': 8})
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facets, metadata={'quadrature_degree': 8})
    kval = c['k'] + 0 * x[0]
    a = kval * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    inner_bc, outer_bc = conditions['inner'], conditions['outer']
    if inner_bc['type'] != 'neumann' or outer_bc['type'] != 'dirichlet':
        raise ValueError('The analytic annulus FEM requires inner Neumann flux and outer Dirichlet temperature')
    L = -inner_bc['q_n_W_m2'] * v * ds(11)
    trace = fem.Function(V)
    trace.interpolate(lambda points: np.full(points.shape[1], outer_bc['T_C']))
    outer_dofs = fem.locate_dofs_topological(V, 1, facets.find(12))
    solution = LinearProblem(a, L, bcs=[fem.dirichletbc(trace, outer_dofs)],
        petsc_options_prefix=f"annulus_{c['id']}_{level}_",
        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps', 'ksp_error_if_not_converged': True}).solve()
    solution.x.scatter_forward()
    ex = exact(c, x[0], x[1], UFLBackend)
    l2 = float(np.sqrt(fem.assemble_scalar(fem.form((solution - ex) ** 2 * dx))))
    h1 = float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(solution - ex), ufl.grad(solution - ex)) * dx))))
    xy = evaluation_points(c)
    values, truth = evaluate(solution, xy), exact(c, xy[:, 0], xy[:, 1])
    normal = ufl.FacetNormal(msh)
    inner_outflow = float(fem.assemble_scalar(fem.form(-kval * ufl.dot(ufl.grad(solution), normal) * ds(11))))
    outer_outflow = float(fem.assemble_scalar(fem.form(-kval * ufl.dot(ufl.grad(solution), normal) * ds(12))))
    net_outflow = inner_outflow + outer_outflow
    power = c['power']
    record = dict(case=c['id'], kind='annulus', level=level, node_multiplier=node_multiplier,
        nominal_size_factor=size_factor, ndofs=V.dofmap.index_map.size_global, degree=2,
        l2_error_K=l2, h1_seminorm_error_K_per_m=h1,
        rmse_eval_K=float(np.sqrt(np.mean((values - truth) ** 2))),
        max_eval_error_K=float(np.max(np.abs(values - truth))),
        inner_outflow_W_m=inner_outflow, outer_outflow_W_m=outer_outflow,
        expected_inner_outflow_W_m=-power, expected_outer_outflow_W_m=power,
        outflow_W_m=net_outflow, generated_W_m=0., balance_pct=100 * abs(net_outflow) / power,
        inner_temperature_exact_C=float(exact(c, c['ri'], 0.)), physics_sha256=getattr(source_case, 'sha256', fingerprint(c)))
    np.savez_compressed(output.joinpath(stem + '.npz'), xy=xy, T=values, T_exact=truth)
    record.update(elapsed_s=time.perf_counter() - start, dolfinx=dolfinx.__version__,
        completed_utc=datetime.now(timezone.utc).isoformat(), source_sha256=archive_sources(output))
    output.joinpath(stem + '.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({key: record[key] for key in ('case', 'level', 'ndofs', 'l2_error_K', 'balance_pct', 'elapsed_s')}), flush=True)
    return record


def solve(source_case, level, output, node_multiplier=1.):
    node_multiplier = validate_node_multiplier(node_multiplier)
    if MPI.COMM_WORLD.size != 1:
        raise ValueError('Reference currently requires one MPI rank')
    c = source_case.physical if isinstance(source_case, CaseDefinition) else source_case
    if c['kind'] == 'annulus':
        return solve_annulus(source_case, c, level, output, node_multiplier)
    if not c['kind'].startswith('mms'):
        raise ValueError('mms_fem solves manufactured cases only')
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{c['id']}_l{level}"
    if output.joinpath(stem + '.json').exists():
        raise FileExistsError('Use a new directory for a new reference run')
    n = scaled_cells_per_side(16 * 2 ** level, node_multiplier)
    msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = fem.functionspace(msh, ('Lagrange', 2))
    u, v, x = ufl.TrialFunction(V), ufl.TestFunction(V), ufl.SpatialCoordinate(msh)
    k, q = field_k(c, x[0], x[1], UFLBackend), source(c, x[0], x[1], UFLBackend)
    dx = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': 8})
    boundary = mesh.locate_entities_boundary(msh, 1, lambda z: np.full(z.shape[1], True))
    tags = np.full(len(boundary), OTHER_TAG, dtype=np.int32)
    tags[np.isclose(mesh.compute_midpoints(msh, 1, boundary)[:, 1], 1.0)] = TOP_TAG
    order = np.argsort(boundary)
    facet_tags = mesh.meshtags(msh, 1, boundary[order], tags[order])
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': 8})
    a = ufl.inner(k * ufl.grad(u), ufl.grad(v)) * dx
    L = q * v * dx
    trace = fem.Function(V)
    trace.interpolate(lambda z: exact(c, z[0], z[1]))
    robin = c['kind'] == 'mms_robin'
    if robin:
        t_inf = exact(c, x[0], x[1], UFLBackend) + 3 * x[0] * (1 - x[0])
        a += ROBIN_H * u * v * ds(TOP_TAG)
        L += ROBIN_H * t_inf * v * ds(TOP_TAG)
        dirichlet_facets = facet_tags.find(OTHER_TAG)
    else:
        dirichlet_facets = np.r_[facet_tags.find(OTHER_TAG), facet_tags.find(TOP_TAG)]
    bc = fem.dirichletbc(trace, fem.locate_dofs_topological(V, 1, dirichlet_facets))
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f"mms_{c['id']}_{level}_",
        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps', 'ksp_error_if_not_converged': True})
    solution = problem.solve()
    solution.x.scatter_forward()
    ex = exact(c, x[0], x[1], UFLBackend)
    l2 = float(np.sqrt(fem.assemble_scalar(fem.form((solution - ex) ** 2 * dx))))
    h1 = float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(solution - ex), ufl.grad(solution - ex)) * dx))))
    nodal = fem.Function(V)
    nodal.interpolate(lambda z: exact(c, z[0], z[1]))
    xy = evaluation_points(c)
    values, truth = evaluate(solution, xy), exact(c, xy[:, 0], xy[:, 1])
    normal = ufl.FacetNormal(msh)
    outflow = float(fem.assemble_scalar(fem.form(-k * ufl.dot(ufl.grad(solution), normal) * ds)))
    generated = float(fem.assemble_scalar(fem.form(q * dx)))
    record = dict(case=c['id'], kind=c['kind'], level=level, node_multiplier=node_multiplier,
        nominal_size_factor=mesh_size_factor(node_multiplier), cells_per_side=n, h=1.0 / n,
        ndofs=V.dofmap.index_map.size_global, degree=2,
        l2_error_K=l2, h1_seminorm_error_K_per_m=h1, max_nodal_error_K=float(np.max(np.abs(solution.x.array - nodal.x.array))),
        rmse_eval_K=float(np.sqrt(np.mean((values - truth) ** 2))), max_eval_error_K=float(np.max(np.abs(values - truth))),
        generated_W_m=generated, outflow_W_m=outflow, balance_pct=100 * abs(outflow - generated) / abs(generated),
        robin_h_W_m2K=ROBIN_H if robin else None, physics_sha256=fingerprint(c))
    if interfaces(c):
        W = fem.functionspace(msh, ('DG', 1, (2,)))
        gradient = fem.Function(W)
        gradient.interpolate(fem.Expression(ufl.grad(solution), W.element.interpolation_points))
        y = (np.arange(64) + .5) / 64
        info = interfaces(c)[0]
        axis = 0 if info['axis'] == 'x' else 1
        left, right = [np.c_[y, np.full(64, info['position'] + s * 1e-6)] if axis else
                       np.c_[np.full(64, info['position'] + s * 1e-6), y] for s in (-1, 1)]
        flux_left = -field_k(c, left[:, 0], left[:, 1])[:] * evaluate(gradient, left, vector=True)[:, axis]
        flux_right = -field_k(c, right[:, 0], right[:, 1])[:] * evaluate(gradient, right, vector=True)[:, axis]
        jump_T = np.abs(evaluate(solution, left) - evaluate(solution, right))
        record.update(interface_flux_jump_max=float(np.max(np.abs(flux_left - flux_right))), interface_flux_scale=float(np.max(np.abs(flux_left))),
            interface_T_jump_max_K=float(np.max(jump_T)))
        record['interface_flux_jump_rel'] = record['interface_flux_jump_max'] / record['interface_flux_scale']
    np.savez_compressed(output.joinpath(stem + '.npz'), xy=xy, T=values, T_exact=truth)
    record.update(elapsed_s=time.perf_counter() - start, dolfinx=dolfinx.__version__, completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=archive_sources(output))
    output.joinpath(stem + '.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({key: record[key] for key in ('case', 'level', 'ndofs', 'l2_error_K', 'balance_pct', 'elapsed_s')}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case')
    parser.add_argument('--definition', type=Path)
    parser.add_argument('--levels', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--node-multiplier', type=float, default=1.0)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.definition:
        case = CaseDefinition.load(arguments.definition)
    elif arguments.case:
        case = load_all()[arguments.case]
    else:
        parser.error('Give --case ID or --definition FILE')
    for level in arguments.levels:
        solve(case, level, arguments.output, arguments.node_multiplier)


if __name__ == '__main__':
    main()