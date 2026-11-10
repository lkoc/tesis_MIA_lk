"""FEM-alone error of cached nodal solutions against the cable reference (FEM level 1), with the evaluator's NRMSE formula."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'results' / 'fem_generic' / 'cable_disk' / 'fem_l1.npz'
LEVEL_C = 20.0


def values_at(source, xy):
    values = np.asarray(LinearNDInterpolator(source['dof_xy'], source['dof_T'])(xy), dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = NearestNDInterpolator(source['dof_xy'], source['dof_T'])(xy[missing])
    return values


def main(paths):
    reference = np.load(REFERENCE)
    xy, reference_T = reference['dof_xy'], reference['dof_T']
    rise = reference_T.max() - LEVEL_C
    report = {}
    for path in paths:
        candidate = np.load(path)
        error = values_at(candidate, xy) - reference_T
        report[str(path)] = dict(ndofs=int(len(candidate['dof_T'])),
            nrmse_pct=float(100 * np.sqrt(np.mean(error ** 2)) / rise),
            error_Tmax_abs_K=float(abs(values_at(candidate, xy).max() - reference_T.max())),
            max_error_K=float(np.abs(error).max()))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main([Path(p) for p in sys.argv[1:]])
