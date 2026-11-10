"""FEniCSx reference for any pinn-v3-problem-1 document: mesh from its regions, solve and energy balance.

Run inside the FEniCSx environment. Region tags are index + 1; side tags are 1001 to 1004.
"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import dolfinx
import gmsh
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from pinn_v3.problem.expressions import evaluate
from pinn_v3.problem.build import load_problem
from pinn_v3.problem.fields import evaluate_field
from pinn_v3.solvers.fem.backend import UFLBackend
from pinn_v3.solvers.fem.mesh_resolution import mesh_size_factor, validate_node_multiplier

SIDE_TAGS = {'bottom': 1001, 'top': 1002, 'left': 1003, 'right': 1004}
REGION_TAG_OFFSET = 1
BASE_SIZE = 0.25


def solve_annulus(problem, level, output):
    """Solve the true annulus problem with inner Neumann and outer Dirichlet boundaries."""
    domain = problem.doc['domain']
    inner_radius = domain['inner_radius_m']
    outer_radius = domain['outer_radius_m']
    condition_inner = problem.doc['boundaries']['inner']
    condition_outer = problem.doc['boundaries']['outer']
    if condition_inner['type'] != 'neumann' or condition_outer['type'] != 'dirichlet':
        raise ValueError('Annulus FEM requires inner Neumann and outer Dirichlet conditions')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f'fem_l{level}'
    if (output / f'{stem}.json').exists():
        raise FileExistsError('Use a new directory for a new annulus reference run')
    start = time.perf_counter()
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add(f'{problem.id}_annulus')
        occ = gmsh.model.occ
        outer = occ.addDisk(0, 0, 0, outer_radius, outer_radius)
        inner = occ.addDisk(0, 0, 0, inner_radius, inner_radius)
        occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)
        occ.synchronize()
        surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
        if len(surfaces) != 1:
            raise RuntimeError(f'Expected one annular surface, got {len(surfaces)}')
        curves = [tag for dim, tag in gmsh.model.getBoundary([(2, surfaces[0])], combined=True, oriented=False) if dim == 1]
        inner_curves, outer_curves = [], []
        tolerance = 1e-6 * max(outer_radius, 1.0)
        for tag in curves:
            box = gmsh.model.getBoundingBox(1, tag)
            radius = max(abs(box[0]), abs(box[1]), abs(box[3]), abs(box[4]))
            if np.isclose(radius, inner_radius, rtol=0, atol=tolerance):
                inner_curves.append(tag)
            elif np.isclose(radius, outer_radius, rtol=0, atol=tolerance):
                outer_curves.append(tag)
        if not inner_curves or not outer_curves:
            raise RuntimeError('Could not identify annulus boundaries')
        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.addPhysicalGroup(1, inner_curves, 11)
        gmsh.model.addPhysicalGroup(1, outer_curves, 12)
        distance = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(distance, 'CurvesList', inner_curves)
        gmsh.model.mesh.field.setNumber(distance, 'Sampling', 256)
        threshold = gmsh.model.mesh.field.add('Threshold')
        factor = mesh_size_factor(1.0)
        for key, value in [('InField', distance), ('SizeMin', inner_radius / (3 * 2 ** level) * factor),
                           ('SizeMax', outer_radius / (12 * 2 ** level) * factor),
                           ('DistMin', inner_radius), ('DistMax', outer_radius / 4)]:
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
    conductivity = float(problem.doc['regions'][0]['conductivity'])
    a_form = conductivity * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    rhs = -float(condition_inner['q_n_W_m2']) * v * ds(11)
    trace = fem.Function(V)
    trace.x.array[:] = float(condition_outer['value'])
    dofs = fem.locate_dofs_topological(V, 1, facets.find(12))
    solution = LinearProblem(a_form, rhs, bcs=[fem.dirichletbc(trace, dofs)],
        petsc_options_prefix=f'annulus_{problem.id}_{level}_',
        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps',
                       'ksp_error_if_not_converged': True}).solve()
    solution.x.scatter_forward()
    dof_xy = V.tabulate_dof_coordinates()[:, :2]
    reference = problem.reference(dof_xy[:, 0], dof_xy[:, 1])
    error = solution.x.array - reference
    normal = ufl.FacetNormal(msh)
    inner_flux = float(fem.assemble_scalar(fem.form(-conductivity * ufl.dot(ufl.grad(solution), normal) * ds(11))))
    outer_flux = float(fem.assemble_scalar(fem.form(-conductivity * ufl.dot(ufl.grad(solution), normal) * ds(12))))
    power = abs(float(condition_inner['q_n_W_m2'])) * 2 * math.pi * inner_radius
    record = dict(id=problem.id, level=level, method='annulus_FEniCSx', ndofs=V.dofmap.index_map.size_global,
        physics_sha256=hashlib.sha256(json.dumps(problem.doc, sort_keys=True).encode()).hexdigest(),
        inner_outflow_W_m=inner_flux, outer_outflow_W_m=outer_flux, expected_power_W_m=power,
        balance_pct=100 * abs(outer_flux - power) / power,
        l2_dof_error_K=float(np.sqrt(np.mean(error ** 2))), max_dof_error_K=float(np.max(np.abs(error))),
        Tmin_C=float(np.min(solution.x.array)), Tmax_C=float(np.max(solution.x.array)),
        elapsed_s=time.perf_counter() - start, dolfinx=dolfinx.__version__)
    np.savez_compressed(output / f'{stem}.npz', dof_xy=dof_xy, dof_T=solution.x.array.copy(), T_exact=reference)
    (output / f'{stem}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def make_shape(occ, shape):
    """OpenCASCADE surface of one region; an annulus is a disk with a hole."""
    kind = shape['type']
    if kind == 'rectangle':
        x0, x1, y0, y1 = shape['bounds']
        return occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)
    cx, cy = shape['center']
    if kind == 'disk':
        return occ.addDisk(cx, cy, 0, shape['radius_m'], shape['radius_m'])
    outer = occ.addDisk(cx, cy, 0, shape['outer_radius_m'], shape['outer_radius_m'])
    inner = occ.addDisk(cx, cy, 0, shape['inner_radius_m'], shape['inner_radius_m'])
    cut, _ = occ.cut([(2, outer)], [(2, inner)])
    return cut[0][1]


def build_mesh(problem, h, size_factor=1.0):
    """Conforming mesh; each surface belongs to the last region that contains it."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add(problem.id)
        occ = gmsh.model.occ
        x0, x1, y0, y1 = problem.bounds
        outer = occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)
        tools = [make_shape(occ, region['shape']) for region in problem.regions]
        _, mapping = occ.fragment([(2, outer)], [(2, tag) for tag in tools])
        occ.synchronize()
        owner = {}
        order = sorted(range(len(tools)), key=lambda item: problem.regions[item]['priority'])
        for position in order:
            for dim, tag in mapping[1 + position]:
                if dim == 2:
                    owner[tag] = position
        surfaces = [tag for dim, tag in mapping[0] if dim == 2]
        if any(tag not in owner for tag in surfaces):
            raise RuntimeError('mesh has a surface that no region owns')
        for position, region in enumerate(problem.regions):
            faces = [tag for tag, p in owner.items() if p == position]
            if not faces:
                raise RuntimeError(f"region {region['id']!r} is fully covered by later regions")
            gmsh.model.addPhysicalGroup(2, faces, position + REGION_TAG_OFFSET)
            gmsh.model.setPhysicalName(2, position + REGION_TAG_OFFSET, region['id'])
        # Curves whose centre lies on one rectangle side form that side's physical group.
        tol = 1e-9 * max(x1 - x0, y1 - y0)
        curves = {side: [] for side in SIDE_TAGS}
        for dim, tag in gmsh.model.getBoundary([(2, t) for t in surfaces], combined=True, oriented=False):
            if dim != 1:
                continue
            mx, my, _ = occ.getCenterOfMass(1, tag)
            for side, coordinate, target in (('bottom', my, y0), ('top', my, y1), ('left', mx, x0), ('right', mx, x1)):
                if abs(coordinate - target) < tol:
                    curves[side].append(tag)
        for side, tags in curves.items():
            if not tags:
                raise RuntimeError(f'side {side} has no boundary curves')
            gmsh.model.addPhysicalGroup(1, tags, SIDE_TAGS[side])
        gmsh.option.setNumber('Mesh.MeshSizeMax', h)
        gmsh.option.setNumber('Mesh.MeshSizeFactor', size_factor)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        return data.mesh, data.cell_tags, data.facet_tags
    finally:
        gmsh.finalize()


def ufl_field(spec, x, V):
    """Field inside the weak form: a constant, a UFL expression, or an interpolated table."""
    if isinstance(spec, (int, float)):
        return float(spec)
    if 'expression' in spec:
        return evaluate(spec['expression'], x[0], x[1], UFLBackend)
    values = fem.Function(V)
    values.interpolate(lambda z: evaluate_field(spec, z[0], z[1]).ravel())
    return values


def solve(problem, level, output, node_multiplier=1.0):
    """Solve the problem at one mesh level and node multiplier; write its JSON record and nodal values."""
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    node_multiplier = validate_node_multiplier(node_multiplier)
    token = '' if node_multiplier == 1.0 else f'_m{node_multiplier:g}'.replace('.', 'p')
    stem = f'fem_l{level}{token}'
    if output.joinpath(stem + '.json').exists():
        raise FileExistsError('Use a new directory for a new reference run')
    h = BASE_SIZE / 2 ** level * mesh_size_factor(node_multiplier)
    msh, cells, facets = build_mesh(problem, BASE_SIZE / 2 ** level, mesh_size_factor(node_multiplier))
    V = fem.functionspace(msh, ('Lagrange', 2))
    u, v, x = ufl.TrialFunction(V), ufl.TestFunction(V), ufl.SpatialCoordinate(msh)
    dx = ufl.Measure('dx', domain=msh, subdomain_data=cells, metadata={'quadrature_degree': 8})
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facets, metadata={'quadrature_degree': 8})

    a_terms, L_terms, sources, regions = [], [0.0 * v * dx], [], []
    for position, region in enumerate(problem.regions):
        tag = position + REGION_TAG_OFFSET
        a_terms.append(ufl_field(region['conductivity'], x, V) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(tag))
        q = ufl_field(region.get('source', 0.0), x, V)
        if not (isinstance(q, float) and q == 0.0):
            L_terms.append(q * v * dx(tag))
            sources.append(q * dx(tag))
        regions.append((region['id'], tag))

    bcs, prescribed_sides, robin_sides, dirichlet_tags = [], [], [], []
    for side, condition in problem.doc['boundaries'].items():
        tag = SIDE_TAGS[side]
        if condition['type'] == 'dirichlet':
            values = fem.Function(V)
            values.interpolate(lambda z, spec=condition['value']: evaluate_field(spec, z[0], z[1]).ravel())
            bcs.append(fem.dirichletbc(values, fem.locate_dofs_topological(V, 1, facets.find(tag))))
            dirichlet_tags.append(tag)
        elif condition['type'] == 'neumann':
            L_terms.append(-ufl_field(condition['q_n_W_m2'], x, V) * v * ds(tag))
            prescribed_sides.append((tag, condition['q_n_W_m2']))
        else:
            h_side = condition['h_W_m2K']
            a_terms.append(h_side * u * v * ds(tag))
            L_terms.append(h_side * ufl_field(condition['T_inf_C'], x, V) * v * ds(tag))
            robin_sides.append((tag, h_side, condition['T_inf_C']))

    a_form, L_form = sum(a_terms), sum(L_terms)
    problem_linear = LinearProblem(a_form, L_form, bcs=bcs, petsc_options_prefix=f'generic_{problem.id}_{level}{token}_',
        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps', 'ksp_error_if_not_converged': True})
    solution = problem_linear.solve()
    solution.x.scatter_forward()

    # Energy balance: Dirichlet reaction, Robin outflow and prescribed Neumann flux against the generated heat.
    psi = fem.Function(V)
    psi.x.array[:] = 0.0
    for tag in dirichlet_tags:
        psi.x.array[fem.locate_dofs_topological(V, 1, facets.find(tag))] = 1.0
    reaction = -float(fem.assemble_scalar(fem.form(ufl.replace(a_form, {u: solution, v: psi}) - ufl.replace(L_form, {v: psi}))))
    constitutive = sum(float(fem.assemble_scalar(fem.form(h_side * (solution - ufl_field(t_inf, x, V)) * ds(tag))))
                       for tag, h_side, t_inf in robin_sides)
    prescribed = sum(float(fem.assemble_scalar(fem.form(ufl_field(q, x, V) * ds(tag)))) for tag, q in prescribed_sides)
    generated = sum(float(fem.assemble_scalar(fem.form(term))) for term in sources)
    consistent = reaction + constitutive + prescribed

    region_records = []
    for region_id, tag in regions:
        area = float(fem.assemble_scalar(fem.form(1.0 * dx(tag))))
        region_records.append(dict(id=region_id, area_mesh_m2=area))
    sha = hashlib.sha256(json.dumps(problem.doc, sort_keys=True).encode()).hexdigest()
    record = dict(id=problem.id, level=level, node_multiplier=node_multiplier, h=h, ndofs=V.dofmap.index_map.size_global, degree=2,
        physics_sha256=sha, regions=region_records,
        generated_W_m=generated, dirichlet_reaction_W_m=reaction, constitutive_outflow_W_m=constitutive,
        prescribed_outflow_W_m=prescribed, consistent_outflow_W_m=consistent,
        balance_consistent_pct=100 * abs(consistent - generated) / max(abs(generated), 1e-30),
        Tmin_C=float(np.min(solution.x.array)), Tmax_C=float(np.max(solution.x.array)),
        elapsed_s=time.perf_counter() - start, dolfinx=dolfinx.__version__,
        completed_utc=datetime.now(timezone.utc).isoformat())
    np.savez_compressed(output.joinpath(stem + '.npz'), dof_xy=V.tabulate_dof_coordinates()[:, :2],
        dof_T=solution.x.array.copy())
    output.joinpath(stem + '.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({key: record[key] for key in ('id', 'level', 'ndofs', 'Tmax_C', 'balance_consistent_pct')}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', type=Path, required=True, help='problem.json path')
    parser.add_argument('--levels', nargs='+', type=int, default=[0])
    parser.add_argument('--node-multipliers', nargs='+', type=float, default=[1.0])
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    problem = load_problem(arguments.problem)
    for level in arguments.levels:
        for node_multiplier in arguments.node_multipliers:
            if problem.doc['domain']['shape'] == 'annulus':
                if node_multiplier != 1.0:
                    raise NotImplementedError('node multipliers other than 1.0 are not implemented for annulus')
                solve_annulus(problem, level, arguments.output)
            else:
                solve(problem, level, arguments.output, node_multiplier)


if __name__ == '__main__':
    main()
