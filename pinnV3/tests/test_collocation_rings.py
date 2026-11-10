"""Two collocation rings per interface (one inside each region) and two rows per boundary segment."""
from pathlib import Path

import numpy as np
import pytest

from pinn_v3.problem.build import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.objective import DEFAULT_WEIGHTS

PROBLEMS = Path(__file__).parents[1] / 'problems'
CIRCULAR = PROBLEMS / 'cable_disk' / 'problem.json'
LINEAR = PROBLEMS / 'two_layer_slab' / 'problem.json'


@pytest.mark.parametrize('path', (CIRCULAR, LINEAR))
def test_interface_rings_lie_inside_their_own_regions_at_equal_distance(path):
    problem = GenericProblem(load_problem(path).doc)
    faces = problem.faces(32)
    assert faces
    for face in faces:
        assert face.xy_right is not None
        assert np.all(problem.owner_index(face.xy) == face.left)
        assert np.all(problem.owner_index(face.xy_right) == face.right)
        gap = np.linalg.norm(face.xy_right - face.xy, axis=1)
        assert np.allclose(gap, gap[0], rtol=1e-9)


@pytest.mark.parametrize('path', (CIRCULAR, LINEAR))
def test_each_interface_is_emitted_once(path):
    problem = GenericProblem(load_problem(path).doc)
    names = [face.name for face in problem.faces(32)]
    assert len(names) == len(set(names))


def test_boundary_segments_have_two_rows_owned_by_their_region():
    problem = GenericProblem(load_problem(LINEAR).doc)
    pieces = problem.pieces(16)
    assert pieces
    for piece in pieces:
        assert len(piece.xy) == 32
        assert np.all(problem.owner_index(piece.xy) == piece.region)


def test_interface_flux_is_not_a_default_loss_term():
    assert 'interface_flux' not in DEFAULT_WEIGHTS
    assert DEFAULT_WEIGHTS['energy'] == 0.
