"""Load a reference temperature field (for example a FEM result) for evaluation only.

The file is an NPZ with the evaluation points `xy` and the temperatures `T`, the same layout
that the FEM runner writes. The points must match the problem's own evaluation points; if they
do not, no reference is built, because comparing on different points would be meaningless.
The loss never reads this field.
"""
import json
from pathlib import Path

import numpy as np


def load_reference_npz(path, problem):
    metadata_path = Path(path).with_suffix('.json')
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        declared_physics = metadata.get('physics_sha256')
        if declared_physics is not None and declared_physics != problem.physics_sha256:
            raise ValueError(f'Reference physics in {metadata_path} differs from the problem {problem.name}')
    data = np.load(path)
    xy, _, _ = problem.evaluation()
    if 'xy' not in data or 'T' not in data:
        raise ValueError(f'{path} needs the arrays xy and T')
    if data['xy'].shape != xy.shape or not np.allclose(data['xy'], xy, atol=1e-12):
        raise ValueError(f'Evaluation points in {path} differ from the problem {problem.name}; no reference is built')
    return np.asarray(data['T'], dtype=float).ravel()
