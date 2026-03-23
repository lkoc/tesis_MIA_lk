"""Material property helpers built from CSV-loaded cable layers.

Provides region-based thermal-conductivity and volumetric-heat-capacity
look-ups used by the PDE residual computation.  Also includes the
**XLPE cable catalog** for standard cable sections (95 – 600 mm²) in
copper and aluminium, enabling automatic layer generation from the
``cables_placement.csv`` parameters ``section_mm2``,
``conductor_material`` and ``current_A``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from pinn_cables.io.readers import CableLayer, SoilProperties


# ---------------------------------------------------------------------------
# XLPE cable catalog (12/20 kV)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CableSpec:
    """Internal specification for one standard XLPE cable size."""
    section_mm2: int
    r_conductor: float        # [m]
    r_xlpe_outer: float       # [m]
    r_screen_outer: float     # [m]
    r_sheath_outer: float     # [m]
    R_dc_20_cu: float         # DC resistance at 20 °C, copper  [Ω/m]
    R_dc_20_al: float         # DC resistance at 20 °C, aluminium [Ω/m]


# Geometry based on typical 12/20 kV XLPE single-core cables (IEC 60502-2).
_XLPE_SPECS: dict[int, _CableSpec] = {
    95: _CableSpec(
        section_mm2=95,
        r_conductor=0.0055, r_xlpe_outer=0.0120,
        r_screen_outer=0.0130, r_sheath_outer=0.0150,
        R_dc_20_cu=0.000193, R_dc_20_al=0.000320,
    ),
    150: _CableSpec(
        section_mm2=150,
        r_conductor=0.0070, r_xlpe_outer=0.0135,
        r_screen_outer=0.0145, r_sheath_outer=0.0165,
        R_dc_20_cu=0.000124, R_dc_20_al=0.000206,
    ),
    240: _CableSpec(
        section_mm2=240,
        r_conductor=0.0088, r_xlpe_outer=0.0155,
        r_screen_outer=0.0165, r_sheath_outer=0.0185,
        R_dc_20_cu=0.0000754, R_dc_20_al=0.000125,
    ),
    400: _CableSpec(
        section_mm2=400,
        r_conductor=0.0113, r_xlpe_outer=0.0180,
        r_screen_outer=0.0190, r_sheath_outer=0.0210,
        R_dc_20_cu=0.0000470, R_dc_20_al=0.0000778,
    ),
    600: _CableSpec(
        section_mm2=600,
        r_conductor=0.0142, r_xlpe_outer=0.0210,
        r_screen_outer=0.0220, r_sheath_outer=0.0240,
        R_dc_20_cu=0.0000283, R_dc_20_al=0.0000469,
    ),
}

# Material constants
_CONDUCTOR_PROPS: dict[str, dict[str, float]] = {
    "cu": {"k": 400.0, "rho_c": 3.45e6, "alpha_R": 0.00393},
    "al": {"k": 237.0, "rho_c": 2.42e6, "alpha_R": 0.00403},
}

# Fixed layer thermal properties
_XLPE_K = 0.286       # W/(m·K)
_XLPE_RHO_C = 1.9e6   # J/(m³·K)
_SCREEN_K = 380.0
_SCREEN_RHO_C = 3.4e6
_SHEATH_K = 0.45
_SHEATH_RHO_C = 1.9e6


def available_sections() -> list[int]:
    """Return the list of supported cable sections [mm²]."""
    return sorted(_XLPE_SPECS.keys())


def get_cable_spec(section_mm2: int) -> _CableSpec:
    """Look up a cable specification by section.

    Raises:
        ValueError: If *section_mm2* is not in the catalog.
    """
    if section_mm2 not in _XLPE_SPECS:
        raise ValueError(
            f"Cable section {section_mm2} mm² not in catalog.  "
            f"Available: {available_sections()}"
        )
    return _XLPE_SPECS[section_mm2]


def get_R_dc_20(section_mm2: int, material: str) -> float:
    """DC resistance at 20 °C [Ω/m] for a given section and material."""
    spec = get_cable_spec(section_mm2)
    mat = material.strip().lower()
    if mat == "cu":
        return spec.R_dc_20_cu
    if mat == "al":
        return spec.R_dc_20_al
    raise ValueError(f"Unknown conductor material '{material}'; use 'cu' or 'al'.")


def get_alpha_R(material: str) -> float:
    """Temperature coefficient of resistance [1/K]."""
    mat = material.strip().lower()
    if mat not in _CONDUCTOR_PROPS:
        raise ValueError(f"Unknown conductor material '{material}'; use 'cu' or 'al'.")
    return _CONDUCTOR_PROPS[mat]["alpha_R"]


def generate_cable_layers(
    section_mm2: int,
    material: str,
    current_A: float,
    T_ref: float = 293.15,
) -> list[CableLayer]:
    """Generate :class:`CableLayer` list for a standard XLPE cable.

    The volumetric heat source ``Q`` of the conductor is computed as:

    .. math::
        Q_{vol} = \\frac{I^2 R_{dc,20}}{\\pi r_{cond}^2}

    where :math:`R_{dc,20}` is the DC resistance at 20 °C for the
    chosen material and section.  Temperature-dependent resistance
    effects (R(T)) can be handled at the example / scenario level.

    Args:
        section_mm2: Nominal cross-section [mm²] (95, 150, 240, 400, 600).
        material:    ``"cu"`` (copper) or ``"al"`` (aluminium).
        current_A:   Operating current [A].
        T_ref:       Reference temperature for R [K] (default 20 °C).

    Returns:
        List of four :class:`CableLayer` objects (conductor, xlpe,
        screen, sheath) with correct geometry, thermal properties and Q.
    """
    spec = get_cable_spec(section_mm2)
    mat = material.strip().lower()
    if mat not in _CONDUCTOR_PROPS:
        raise ValueError(f"Unknown conductor material '{material}'; use 'cu' or 'al'.")

    R_dc = get_R_dc_20(section_mm2, mat)
    cond_props = _CONDUCTOR_PROPS[mat]

    # I²R [W/m] → volumetric [W/m³]
    Q_lin = current_A * current_A * R_dc
    area_cond = math.pi * spec.r_conductor ** 2
    Q_vol = Q_lin / area_cond if area_cond > 0 else 0.0

    return [
        CableLayer(
            name="conductor",
            r_inner=0.0,
            r_outer=spec.r_conductor,
            k=cond_props["k"],
            rho_c=cond_props["rho_c"],
            Q=Q_vol,
        ),
        CableLayer(
            name="xlpe",
            r_inner=spec.r_conductor,
            r_outer=spec.r_xlpe_outer,
            k=_XLPE_K,
            rho_c=_XLPE_RHO_C,
            Q=0.0,
        ),
        CableLayer(
            name="screen",
            r_inner=spec.r_xlpe_outer,
            r_outer=spec.r_screen_outer,
            k=_SCREEN_K,
            rho_c=_SCREEN_RHO_C,
            Q=0.0,
        ),
        CableLayer(
            name="sheath",
            r_inner=spec.r_screen_outer,
            r_outer=spec.r_sheath_outer,
            k=_SHEATH_K,
            rho_c=_SHEATH_RHO_C,
            Q=0.0,
        ),
    ]


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
