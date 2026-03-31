"""Physics sub-package: analytical background temperature, IEC 60287, and k-fields."""

from pinn_cables.physics.iec60287 import Q_lin_from_I, compute_iec60287_Q
from pinn_cables.physics.k_field import PhysicsParams, k_scalar, k_tensor, load_physics_params
from pinn_cables.physics.kennelly import multilayer_T_multi

__all__ = [
    "Q_lin_from_I",
    "PhysicsParams",
    "compute_iec60287_Q",
    "k_scalar",
    "k_tensor",
    "load_physics_params",
    "multilayer_T_multi",
]
