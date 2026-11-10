"""Audit manufactured problems: PDE residual and Dirichlet data, committed vs regenerated from cases/."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.build import build_problem
from pinn_v3.problem.fields import evaluate_field
from tools import convert_local_cases_to_problems as converter
from tools.rbf_mms_check import pde_mismatch

TOL = 1e-4
OUTPUT = ROOT / 'results' / 'audit_manufactured_problems_20261030.json'


def boundary_mismatch(doc, problem):
    if doc['domain']['shape'] != 'rectangle':
        return None
    x0, x1, y0, y1 = problem.bounds
    t = np.linspace(0., 1., 201)
    along_x = x0 + (x1 - x0) * t
    along_y = y0 + (y1 - y0) * t
    sides = {'left': (np.full_like(t, x0), along_y), 'right': (np.full_like(t, x1), along_y),
             'bottom': (along_x, np.full_like(t, y0)), 'top': (along_x, np.full_like(t, y1))}
    exact = np.asarray(evaluate_field(doc['reference']['exact'], along_x, along_y), dtype=float)
    worst = 0.
    for side, condition in doc['boundaries'].items():
        if condition['type'] != 'dirichlet' or side not in sides:
            continue
        x, y = sides[side]
        value = condition['value']
        given = (np.asarray(evaluate_field(value, x, y), dtype=float) if isinstance(value, dict)
                 else np.full_like(x, float(value)))
        reference = np.asarray(evaluate_field(doc['reference']['exact'], x, y), dtype=float)
        worst = max(worst, float(np.abs(given - reference).max()))
    return worst


def audit_doc(doc):
    problem = build_problem(doc)
    samples = problem.sample_interior(200, seed=1)[0]
    pde = pde_mismatch(problem, samples)
    boundary = boundary_mismatch(doc, problem)
    pde_ok = pde['max_residual'] <= TOL * max(pde['max_source'], 1.)
    boundary_ok = boundary is None or boundary <= TOL * max(1., float(doc.get('temperature_scale_K', 1.)))
    return dict(max_residual=pde['max_residual'], max_source=pde['max_source'], pde_ok=bool(pde_ok),
                boundary_max_abs=boundary, boundary_ok=bool(boundary_ok), ok=bool(pde_ok and boundary_ok))


def committed_docs():
    for path in sorted((ROOT / 'problems').glob('*/problem.json')):
        yield json.loads(path.read_text(encoding='utf-8'))


def regenerated_docs():
    for path in sorted(converter.SOURCE.glob('*/case.json')):
        case = json.loads(path.read_text(encoding='utf-8'))
        try:
            yield case['id'], converter.convert(case), None
        except Exception as exc:
            yield case['id'], None, f'{type(exc).__name__}: {exc}'


def safe_audit(doc):
    try:
        return audit_doc(doc)
    except Exception as exc:
        return dict(error=f'{type(exc).__name__}: {exc}', ok=False)


def main():
    committed_by_id = {doc['id']: doc for doc in committed_docs()}
    committed = {doc_id: safe_audit(doc) for doc_id, doc in committed_by_id.items()
                 if 'exact' in doc.get('reference', {})}
    regenerated, differences = {}, {}
    for case_id, doc, error in regenerated_docs():
        if error:
            differences[case_id] = f'convert_error: {error}'
            continue
        if committed_by_id.get(case_id) != doc:
            differences[case_id] = 'regenerated problem.json differs from committed'
        if 'exact' in doc.get('reference', {}):
            regenerated[case_id] = safe_audit(doc)
    OUTPUT.write_text(json.dumps(dict(committed=committed, regenerated=regenerated,
                                      regeneration_differences=differences), indent=2), encoding='utf-8')
    for label, rows in (('committed', committed), ('regenerated', regenerated)):
        for case_id, row in rows.items():
            print(label, case_id, 'ok' if row.get('ok') else 'FAIL', row.get('error', ''), flush=True)
    print('regeneration differences:', sorted(differences), flush=True)


if __name__ == '__main__':
    main()
