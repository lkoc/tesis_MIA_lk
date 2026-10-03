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
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from pinn_v2.cases.catalog import cases, field_k, source, exact, evaluation_points, fingerprint, interfaces
from pinn_v2.solvers.fem.backend import UFLBackend
from pinn_v2.manifest.provenance import archive_sources

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


def solve(c, level, output):
    if MPI.COMM_WORLD.size != 1:
        raise ValueError('Reference currently requires one MPI rank')
    if not c['kind'].startswith('mms'):
        raise ValueError('mms_fem solves manufactured cases only')
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{c['id']}_l{level}"
    if output.joinpath(stem + '.json').exists():
        raise FileExistsError('Use a new directory for a new reference run')
    n = 16 * 2 ** level
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
    record = dict(case=c['id'], kind=c['kind'], level=level, cells_per_side=n, h=1.0 / n, ndofs=V.dofmap.index_map.size_global, degree=2,
        l2_error_K=l2, h1_seminorm_error_K_per_m=h1, max_nodal_error_K=float(np.max(np.abs(solution.x.array - nodal.x.array))),
        rmse_eval_K=float(np.sqrt(np.mean((values - truth) ** 2))), max_eval_error_K=float(np.max(np.abs(values - truth))),
        generated_W_m=generated, outflow_W_m=outflow, balance_pct=100 * abs(outflow - generated) / abs(generated),
        robin_h_W_m2K=ROBIN_H if robin else None, physics_sha256=fingerprint(c))
    if interfaces(c):
        W = fem.functionspace(msh, ('DG', 1, (2,)))
        gradient = fem.Function(W)
        gradient.interpolate(fem.Expression(ufl.grad(solution), W.element.interpolation_points))
        y = (np.arange(64) + .5) / 64
        left, right = [np.c_[np.full(64, info['position'] + s * 1e-6), y] for info in interfaces(c)[:1] for s in (-1, 1)]
        flux_left = -field_k(c, left[:, 0], left[:, 1])[:] * evaluate(gradient, left, vector=True)[:, 0]
        flux_right = -field_k(c, right[:, 0], right[:, 1])[:] * evaluate(gradient, right, vector=True)[:, 0]
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
    parser.add_argument('--case', required=True)
    parser.add_argument('--levels', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    case = cases()[arguments.case]
    for level in arguments.levels:
        solve(case, level, arguments.output)


if __name__ == '__main__':
    main()