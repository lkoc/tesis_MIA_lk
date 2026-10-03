"""Load a reference temperature field (for example a FEM result) for evaluation only.

The file is an NPZ with the evaluation points `xy` and the temperatures `T`, the same layout
that the FEM runner writes. The points must match the problem's own evaluation points; if they
do not, no reference is built, because comparing on different points would be meaningless.
The loss never reads this field.
"""
import numpy as np


def load_reference_npz(path, problem):
    data = np.load(path)
    xy, _, _ = problem.evaluation()
    if 'xy' not in data or 'T' not in data:
        raise ValueError(f'{path} needs the arrays xy and T')
    if data['xy'].shape != xy.shape or not np.allclose(data['xy'], xy, atol=1e-12):
        raise ValueError(f'Evaluation points in {path} differ from the problem {problem.name}; no reference is built')
    return np.asarray(data['T'], dtype=float).ravel()
