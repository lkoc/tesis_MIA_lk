"""Shared target-density controls for the FEM mesh generators."""
import math


NODE_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)


def validate_node_multiplier(value):
    try:
        multiplier = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'node_multiplier must be one of {NODE_MULTIPLIERS}') from exc
    if not math.isfinite(multiplier) or not any(math.isclose(multiplier, allowed) for allowed in NODE_MULTIPLIERS):
        raise ValueError(f'node_multiplier must be one of {NODE_MULTIPLIERS}')
    return multiplier


def mesh_size_factor(node_multiplier):
    """Scale element length so 2D element count grows approximately by the requested factor."""
    return 1.0 / math.sqrt(validate_node_multiplier(node_multiplier))


def scaled_cells_per_side(base_cells, node_multiplier):
    """Scale a structured 2D grid to at least the requested approximate cell-count factor."""
    return int(math.ceil(int(base_cells) / mesh_size_factor(node_multiplier)))