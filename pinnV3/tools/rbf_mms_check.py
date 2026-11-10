"""MMS check of the RBF-FD field: PDE consistency, refinement, and interface flux, without the PINN."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.build import build_problem
from pinn_v3.problem.fields import evaluate_field
from pinn_v3.solvers.pinn.rbf_fd import derivative_weights, solve_regions

LEVELS = ((50, 12), (100, 24), (200, 48))
STENCIL = 21
REL_TOL = 0.02
FLUX_TOL = 0.01
OUTPUT = ROOT / 'results' / 'rbf_mms_check_20261030.json'


def load_doc(case):
    return json.loads((ROOT / 'problems' / case / 'problem.json').read_text(encoding='utf-8'))


def exact_spec(doc):
    return doc['reference']['exact']


def scalar(spec, x, y):
    return float(np.ravel(evaluate_field(spec, np.array([x], dtype=float), np.array([y], dtype=float)))[0])


def fd_laplacian(function, x, y, h=1e-4):
    return ((function(x + h, y) - 2 * function(x, y) + function(x - h, y)) / h**2 +
            (function(x, y + h) - 2 * function(x, y) + function(x, y - h)) / h**2)


def fd_gradient(function, x, y, h=1e-5):
    return ((function(x + h, y) - function(x - h, y)) / (2 * h),
            (function(x, y + h) - function(x, y - h)) / (2 * h))


def pde_mismatch(problem, samples):
    x, y = samples.T
    spec = exact_spec(problem.doc)
    exact = lambda a, b: np.asarray(evaluate_field(spec, a, b), dtype=float)
    conductivity = problem.field('conductivity', x, y)
    kappa = lambda a, b: problem.field('conductivity', a, b)
    kx, ky = fd_gradient(kappa, x, y)
    tx, ty = fd_gradient(exact, x, y)
    source = problem.field('source', x, y)
    residual = -conductivity * fd_laplacian(exact, x, y) - (kx * tx + ky * ty) - source
    return dict(max_residual=float(np.abs(residual).max()), max_source=float(np.abs(source).max()),
                relative=float(np.abs(residual).max() / np.abs(source).max()))


def side_flux(problem, result, index, normal, temperature=None):
    xy, labels = result['xy'], result['labels']
    temperature = result['temperature'] if temperature is None else temperature
    region = int(labels[index])
    nodes = np.flatnonzero(labels == region)
    nearest = nodes[cKDTree(xy[nodes]).query(xy[index], k=STENCIL)[1]]
    weights = derivative_weights(xy[index], xy[nearest])
    gradient = weights[:, :2].T @ temperature[nearest]
    return scalar(problem.regions[region]['conductivity'], xy[index][0], xy[index][1]) * float(gradient @ normal)


def exact_flux(problem, point, normal, side, region):
    spec = exact_spec(problem.doc)
    p = point + side * 1e-5 * normal
    h = 1e-7
    gx = (scalar(spec, p[0] + h, p[1]) - scalar(spec, p[0] - h, p[1])) / (2 * h)
    gy = (scalar(spec, p[0], p[1] + h) - scalar(spec, p[0], p[1] - h)) / (2 * h)
    return scalar(problem.regions[region]['conductivity'], p[0], p[1]) * (gx * normal[0] + gy * normal[1])


def interface_metrics(problem, result):
    xy, labels = result['xy'], result['labels']
    exact_nodes = np.asarray(evaluate_field(exact_spec(problem.doc), xy[:, 0], xy[:, 1]), dtype=float)
    numeric, exact, exact_stencil, jumps = [], [], [], []
    for left, right, normals in result['interfaces']:
        for li, ri, normal in zip(left, right, normals):
            numeric += [side_flux(problem, result, li, normal), side_flux(problem, result, ri, normal)]
            exact += [exact_flux(problem, xy[li], normal, -1., int(labels[li])),
                      exact_flux(problem, xy[ri], normal, 1., int(labels[ri]))]
            exact_stencil += [side_flux(problem, result, li, normal, exact_nodes),
                              side_flux(problem, result, ri, normal, exact_nodes)]
            jumps.append(numeric[-2] - numeric[-1])
    numeric, exact, exact_stencil, jumps = np.asarray(numeric), np.asarray(exact), np.asarray(exact_stencil), np.asarray(jumps)
    scale = np.abs(exact).max()
    interface = result['kinds'] == 'interface'
    unique = np.unique(np.round(xy[interface], 12), axis=0)
    return dict(pairs=len(jumps), flux_scale=float(scale),
                flux_jump_rel=float(np.abs(jumps).max() / scale),
                flux_error_rel=float(np.abs(numeric - exact).max() / scale),
                flux_error_stencil_only_rel=float(np.abs(exact_stencil - exact).max() / scale),
                flux_error_solution_rel=float(np.abs(numeric - exact_stencil).max() / scale),
                interface_nodes=int(interface.sum()), unique_interface_xy=len(unique))


def evaluate_levels(problem, doc, with_interfaces):
    spec = exact_spec(doc)
    scale = float(doc['temperature_scale_K'])
    rows = []
    for interior, surface in LEVELS:
        row = dict(interior_per_region=interior, surface_count=surface)
        try:
            started = time.perf_counter()
            result = solve_regions(problem, interior_per_region=interior, surface_count=surface,
                                   stencil_size=STENCIL, seed=11)
            row['solve_s'] = time.perf_counter() - started
            xy = result['xy']
            error = result['temperature'] - np.asarray(evaluate_field(spec, xy[:, 0], xy[:, 1]), dtype=float)
            interior_mask = result['kinds'] == 'interior'
            row.update(nodes=len(xy), max_abs_err_K=float(np.abs(error).max()),
                       rel_max=float(np.abs(error).max() / scale),
                       rel_rms_interior=float(np.sqrt(np.mean(error[interior_mask] ** 2)) / scale),
                       scaled_residual=float(result['scaled_residual_inf']))
            if with_interfaces:
                row.update(interface_metrics(problem, result))
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    return rows


def verdict(rows, with_interfaces):
    valid = [row for row in rows if 'rel_max' in row]
    last = valid[-1]
    passed = last['rel_max'] < REL_TOL
    if with_interfaces:
        passed = passed and last['flux_jump_rel'] < FLUX_TOL and last['flux_error_rel'] < FLUX_TOL
    decreasing = all(b['rel_max'] < a['rel_max'] for a, b in zip(valid, valid[1:]))
    return dict(passed_finest=bool(passed), error_decreases_with_refinement=bool(decreasing))


def split_interface_doc(doc):
    split = json.loads(json.dumps(doc))
    split['id'] = 'mms_interface_split'
    split.pop('interfaces', None)
    split['regions'] = [
        {'id': 'left', 'priority': 0, 'shape': {'type': 'rectangle', 'bounds': [0., 0.5, 0., 1.]},
         'conductivity': 0.5, 'source': {'expression': 'pi**2*0.5*20*x*sin(pi*y)'}},
        {'id': 'right', 'priority': 1, 'shape': {'type': 'rectangle', 'bounds': [0.5, 1., 0., 1.]},
         'conductivity': 2., 'source': {'expression': 'pi**2*2*10*(1+(x-0.5)/2)*sin(pi*y)'}},
    ]
    return split


def run_case(name, doc, with_interfaces):
    problem = build_problem(doc)
    samples = problem.sample_interior(200, seed=1)[0]
    rows = evaluate_levels(problem, doc, with_interfaces)
    return dict(pde=pde_mismatch(problem, samples), levels=rows, verdict=verdict(rows, with_interfaces))


def main():
    report = {
        'mms_sine': run_case('mms_sine', load_doc('mms_sine'), False),
        'mms_interface_as_declared': run_case('declared', load_doc('mms_interface'), False),
        'mms_interface_split': run_case('split', split_interface_doc(load_doc('mms_interface')), True),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for name, case in report.items():
        print(name, case['verdict'], flush=True)


if __name__ == '__main__':
    main()
