"""Ground temperature profiles for domain boundary conditions.

Provides a small hierarchy of callable objects that map ``(N, 2)``
coordinate tensors to ``(N, 1)`` temperature tensors [K].  These are
intended to be attached to :class:`~pinn_cables.io.readers.BoundaryCondition`
instances via :meth:`~pinn_cables.io.readers.BoundaryCondition.T_target`,
replacing the fixed scalar ``value`` field with a spatially-varying profile.

Classes
-------
GroundTempProfile
    Abstract base class (callable protocol).
ConstantProfile
    Wraps a scalar temperature — backward-compatible default.
CosineGroundProfile
    Kusuda-Achenbach periodic ground temperature distribution
    (Eq. 29 in Kim et al. 2024), suitable for lateral and bottom
    boundary conditions of underground cable FEM/PINN domains.

Usage example
-------------
.. code-block:: python

    from pinn_cables.physics.ground_temp import CosineGroundProfile
    import dataclasses

    summer = CosineGroundProfile(
        T_g=288.35,   # 15.2 °C mean annual ground temp [K]
        A_s=10.9,     # surface amplitude [K]
        tp=217,       # Aug 5th (summer peak)
    )

    # Non-destructively attach to an existing BC loaded from CSV
    bc_left_new  = dataclasses.replace(bc_left,  profile=summer)
    bc_right_new = dataclasses.replace(bc_right, profile=summer)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class GroundTempProfile(ABC):
    """Abstract callable: ``(N, 2)`` xy → ``(N, 1)`` T_K.

    Subclasses implement :meth:`__call__` so that a profile can be
    used anywhere a constant scalar temperature was used before.
    """

    @abstractmethod
    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        """Return temperature [K] at each point in *xy*.

        Args:
            xy: ``(N, 2)`` tensor of (x, y) coordinates [m],
                where ``y = 0`` is the ground surface and ``y < 0``
                is depth (positive depth = ``-y``).

        Returns:
            ``(N, 1)`` temperature tensor [K].
        """


# ---------------------------------------------------------------------------
# Constant (scalar) profile — backward-compatible default
# ---------------------------------------------------------------------------

class ConstantProfile(GroundTempProfile):
    """Uniform temperature profile wrapping a scalar value.

    Produces the same result as the legacy ``bc.value`` scalar path,
    making it a drop-in replacement when no spatial variation is needed.

    Args:
        T_K: Prescribed temperature [K].
    """

    def __init__(self, T_K: float) -> None:
        self.T_K = float(T_K)

    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        return xy.new_full((xy.shape[0], 1), self.T_K)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ConstantProfile(T={self.T_K - 273.15:.2f}°C)"


# ---------------------------------------------------------------------------
# Kusuda-Achenbach cosine ground temperature profile
# ---------------------------------------------------------------------------

class CosineGroundProfile(GroundTempProfile):
    r"""Kusuda-Achenbach periodic ground temperature distribution.

    Implements the analytical solution for 1-D periodic heat diffusion
    in a semi-infinite homogeneous soil (Eq. 29, Kim et al. 2024)::

        T(z, t) = T_g
                  + A_s · exp(-z / D)
                  · cos(2π/τ · (tp - t0) − z / D)

    where ``z = max(0, -y)`` is the depth below the surface [m] and
    ``D = sqrt(τ · α / π)`` is the characteristic damping depth [m].

    The profile is purely a function of depth (y-coordinate), so it
    can be applied consistently to the left, right, and bottom edges
    of a rectangular domain.

    Args:
        T_g:   Mean annual ground temperature [K].
        A_s:   Annual amplitude of surface temperature variation [K].
               Estimated from weather data as half the annual range.
        alpha: Soil thermal diffusivity [m²/day].  Default 0.0425
               (Kim et al. 2024, Gwangju, Korea).
        tau:   Period of the annual cycle [days].  Default 365.
        t0:    Day of the annual temperature maximum (hottest day).
               Default 217 (August 5th, per Kim 2024).
        tp:    Day of the simulation (day of year).  Default 217
               (summer peak — same as ``t0`` → maximum surface temperature).

    Examples
    --------
    Summer peak profile used in Kim 2024 (Gwangju, Korea):

    .. code-block:: python

        summer = CosineGroundProfile(
            T_g=288.35,   # 15.2 °C
            A_s=10.9,     # K
            tp=217,       # Aug 5th
        )
        # At surface (z=0): T = 15.2 + 10.9*cos(0) = 26.1 °C ✓
        # At z=1.4 m (cable depth): T ≈ 21 °C  (instead of 15.2 °C constant)
        # At z→∞:  T → T_g = 15.2 °C ✓
    """

    def __init__(
        self,
        T_g: float,
        A_s: float,
        alpha: float = 0.0425,
        tau: float = 365.0,
        t0: float = 217.0,
        tp: float = 217.0,
    ) -> None:
        self.T_g = float(T_g)
        self.A_s = float(A_s)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.t0 = float(t0)
        self.tp = float(tp)

        # Damping depth D [m] — precomputed for speed
        self._D: float = math.sqrt(tau * alpha / math.pi)
        # Time-phase shift (scalar, constant for a given simulation day tp)
        self._phase: float = 2.0 * math.pi / tau * (tp - t0)

    # ------------------------------------------------------------------
    # Core callable
    # ------------------------------------------------------------------

    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        """Return T(z) [K] for each row of *xy*.

        Points above the surface (y > 0) are clamped to z = 0 so the
        formula remains well-defined in all geometries.
        """
        # depth z ≥ 0; y is negative below surface
        z = (-xy[:, 1:2]).clamp(min=0.0)
        z_over_D = z / self._D
        T = self.T_g + self.A_s * torch.exp(-z_over_D) * torch.cos(
            self._phase - z_over_D
        )
        return T

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def T_surface(self) -> float:
        """Surface temperature T(z=0, tp) [K]."""
        return self.T_g + self.A_s * math.cos(self._phase)

    def T_at_depth(self, z_m: float) -> float:
        """Temperature [K] at a specific depth *z_m* [m]."""
        z_over_D = z_m / self._D
        return self.T_g + self.A_s * math.exp(-z_over_D) * math.cos(
            self._phase - z_over_D
        )

    def damping_depth(self) -> float:
        """Characteristic damping depth D [m]."""
        return self._D

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CosineGroundProfile("
            f"Tg={self.T_g - 273.15:.1f}°C, "
            f"As={self.A_s:.1f} K, "
            f"tp={self.tp:.0f}d, "
            f"D={self._D:.2f} m, "
            f"T_surf={self.T_surface() - 273.15:.1f}°C)"
        )


# ---------------------------------------------------------------------------
# Piecewise-linear profile — CSV-loadable, simplest option
# ---------------------------------------------------------------------------

class PiecewiseLinearProfile(GroundTempProfile):
    """Piecewise-linear temperature profile over depth, read from CSV.

    Knots are ``(depth_m, T_K)`` pairs where ``depth_m = 0`` is the
    surface and values increase with depth.  Temperature is linearly
    interpolated between knots; beyond the shallowest / deepest knot
    the profile is clamped (constant extrapolation).

    This is the **simplest CSV-parameterizable profile**: add a file
    ``boundary_profiles.csv`` beside ``boundary_conditions.csv`` and
    :func:`~pinn_cables.io.readers.load_problem` will attach it
    automatically.

    Args:
        depths: Depth values [m] (``z ≥ 0``), need not be sorted.
        temps:  Corresponding temperatures [K].

    Raises:
        ValueError: If fewer than 2 knots are provided.

    Example CSV (``boundary_profiles.csv``)::

        boundary,depth_m,T_K
        left,0.0,299.15
        left,1.4,294.26
        left,5.0,288.35
        right,0.0,299.15
        right,1.4,294.26
        right,5.0,288.35
        bottom,0.0,288.35
    """

    def __init__(
        self,
        depths: Sequence[float],
        temps: Sequence[float],
    ) -> None:
        if len(depths) < 2:
            raise ValueError(
                "PiecewiseLinearProfile requires at least 2 knots; "
                f"got {len(depths)}"
            )
        paired = sorted(zip(depths, temps), key=lambda p: p[0])
        z_vals = [float(p[0]) for p in paired]
        t_vals = [float(p[1]) for p in paired]
        # kept as Python lists for repr; tensors for computation
        self._z_list = z_vals
        self._T_list = t_vals
        # CPU float32 tensors; lazily moved to device in __call__
        self._z = torch.tensor(z_vals, dtype=torch.float32)
        self._T = torch.tensor(t_vals, dtype=torch.float32)

    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        """Return T(z) [K] for each row in *xy* via linear interpolation."""
        # depth z ≥ 0  (y ≤ 0 → depth = -y)
        z = (-xy[:, 1:2]).clamp(min=0.0)  # (N, 1)
        z_flat = z.squeeze(1)             # (N,)

        knot_z = self._z.to(xy.device)   # (K,)
        knot_T = self._T.to(xy.device)   # (K,)

        # Index of the right bracket (1 … K-1 after clamp)
        idx = torch.searchsorted(knot_z.contiguous(), z_flat.contiguous())
        idx = idx.clamp(1, knot_z.shape[0] - 1)

        z0, z1 = knot_z[idx - 1], knot_z[idx]
        T0, T1 = knot_T[idx - 1], knot_T[idx]

        # Linear weight; clamp handles both extrapolation edges
        t = ((z_flat - z0) / (z1 - z0).clamp(min=1e-12)).clamp(0.0, 1.0)
        T = T0 + t * (T1 - T0)
        return T.unsqueeze(1)

    @property
    def knots(self) -> list[tuple[float, float]]:
        """Return the list of ``(depth_m, T_K)`` knot pairs."""
        return list(zip(self._z_list, self._T_list))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PiecewiseLinearProfile("
            f"knots={len(self._z_list)}, "
            f"z=[{self._z_list[0]:.1f}…{self._z_list[-1]:.1f}] m)"
        )
