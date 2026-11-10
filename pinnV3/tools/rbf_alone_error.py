"""RBF-FD nodal field against the cached FEM level-1 reference, with the evaluator's NRMSE formula and no WSL call."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.rbf_fd import solve_regions

REFERENCE = ROOT / 'results' / 'fem_generic' / 'cable_disk' / 'fem_l1.npz'
OUTPUT = ROOT / 'results' / 'rbf_alone_error_20261030.json'
LEVEL_C = 20.0


def main():
    definition = GenericProblem(load_problem(ROOT / 'problems' / 'cable_disk' / 'problem.json')).definition
    field = solve_regions(definition, interior_per_region=100, surface_count=24, stencil_size=21, seed=11)
    reference = np.load(REFERENCE)
    reference_xy, reference_T = reference['dof_xy'], reference['dof_T']
    owners = definition.owner(reference_xy[:, 0], reference_xy[:, 1])
    values = np.full(len(reference_T), np.nan)
    for region_id in range(len(definition.regions)):
        at_reference = owners == region_id
        source = field['labels'] == region_id
        source_xy, source_T = field['xy'][source], field['temperature'][source]
        interpolated = np.asarray(LinearNDInterpolator(source_xy, source_T)(reference_xy[at_reference]), dtype=float)
        missing = ~np.isfinite(interpolated)
        if missing.any():
            interpolated[missing] = NearestNDInterpolator(source_xy, source_T)(reference_xy[at_reference][missing])
        values[at_reference] = interpolated
    error = values - reference_T
    rise = reference_T.max() - LEVEL_C
    report = dict(rbf_nodes=int(len(field['xy'])), scaled_residual=float(field['scaled_residual_inf']),
                  Tmax_rbf_C=float(values.max()), Tmax_ref_C=float(reference_T.max()),
                  nrmse_pct=float(100 * np.sqrt(np.mean(error ** 2)) / rise),
                  error_Tmax_abs_K=float(abs(values.max() - reference_T.max())),
                  max_error_K=float(np.abs(error).max()))
    OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
