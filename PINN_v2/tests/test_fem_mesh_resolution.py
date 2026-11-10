import math

import pytest

from pinn_v2.solvers.fem.mesh_resolution import (
    NODE_MULTIPLIERS,
    mesh_size_factor,
    scaled_cells_per_side,
    validate_node_multiplier,
)


@pytest.mark.parametrize('multiplier', NODE_MULTIPLIERS)
def test_node_multiplier_scales_target_mesh_density(multiplier):
    assert validate_node_multiplier(multiplier) == multiplier
    assert mesh_size_factor(multiplier) == pytest.approx(1 / math.sqrt(multiplier))
    assert scaled_cells_per_side(16, multiplier) >= 16


@pytest.mark.parametrize('multiplier', [0, -1, 1.25, float('nan'), float('inf'), None])
def test_node_multiplier_rejects_unsupported_values(multiplier):
    with pytest.raises(ValueError, match='node_multiplier'):
        validate_node_multiplier(multiplier)


def test_scaled_cells_per_side_increases_monotonically():
    counts = [scaled_cells_per_side(16, multiplier) for multiplier in NODE_MULTIPLIERS]

    assert counts == sorted(counts)
    assert counts == [16, 20, 23, 28]