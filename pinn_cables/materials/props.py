"""Material property helpers built from CSV-loaded cable layers.

Provides region-based thermal-conductivity and volumetric-heat-capacity
look-ups used by the PDE residual computation.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from pinn_cables.io.readers import CableLayer, SoilProperties


# ---------------------------------------------------------------------------
# Spatially-variable soil conductivity
# ---------------------------------------------------------------------------

def k_soil_variable(
    xy: torch.Tensor,
    k0: float,
    amp: float = 0.3,
) -> torch.Tensor:
    """Sinusoidal spatially-variable soil conductivity.

    .. math::
        k(x, y) = k_0 \\bigl(1 + A \\sin(2\\pi x)\\cos(2\\pi y)\\bigr)

    Args:
        xy:  Coordinates tensor of shape ``(N, 2)``.
        k0:  Base conductivity [W/(m K)].
        amp: Perturbation amplitude (fraction of *k0*).

    Returns:
        Conductivity tensor of shape ``(N, 1)``.
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    two_pi = 2.0 * math.pi
    return k0 * (1.0 + amp * torch.sin(two_pi * x) * torch.cos(two_pi * y))


# ---------------------------------------------------------------------------
# Region look-up
# ---------------------------------------------------------------------------

def get_k(
    layer: CableLayer | None,
    xy: torch.Tensor,
    soil: SoilProperties,
) -> torch.Tensor | float:
    """Return thermal conductivity for a cable layer or the soil.

    Args:
        layer: Cable layer (``None`` means soil region).
        xy:    Spatial coordinates ``(N, 2)`` — used only for variable soil *k*.
        soil:  Soil properties.

    Returns:
        Scalar float (constant *k*) or ``(N, 1)`` tensor (variable soil *k*).
    """
    if layer is not None:
        return layer.k
    if soil.variable:
        return k_soil_variable(xy, soil.k, soil.amp)
    return soil.k


def get_rho_c(
    layer: CableLayer | None,
    soil: SoilProperties,
) -> float:
    """Return volumetric heat capacity for a cable layer or the soil.

    Args:
        layer: Cable layer (``None`` means soil region).
        soil:  Soil properties.

    Returns:
        Scalar value [J/(m^3 K)].
    """
    if layer is not None:
        return layer.rho_c
    return soil.rho_c


def get_Q(
    layer: CableLayer | None,
    Q_scale: float = 1.0,
) -> float:
    """Return volumetric source term for a cable layer or the soil.

    Args:
        layer:   Cable layer (``None`` means soil — no internal heat source).
        Q_scale: Multiplicative factor from the scenario.

    Returns:
        Volumetric source [W/m^3].
    """
    if layer is not None:
        return layer.Q * Q_scale
    return 0.0
