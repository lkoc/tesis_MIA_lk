"""CSV readers for problem data files.

Each function reads one CSV file and returns typed dataclasses.
Uses only the ``csv`` module from the standard library.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CableLayer:
    """Single concentric layer of a power cable.

    Attributes:
        name:    Human-readable identifier (e.g. ``"conductor"``).
        r_inner: Inner radius [m].
        r_outer: Outer radius [m].
        k:       Thermal conductivity [W/(m K)].
        rho_c:   Volumetric heat capacity [J/(m^3 K)].
        Q:       Volumetric heat-source density [W/m^3].
    """
    name: str
    r_inner: float
    r_outer: float
    k: float
    rho_c: float
    Q: float

    def __post_init__(self) -> None:
        if self.r_outer <= self.r_inner:
            raise ValueError(
                f"Layer '{self.name}': r_outer ({self.r_outer}) must be > "
                f"r_inner ({self.r_inner})"
            )
        if self.k <= 0:
            raise ValueError(f"Layer '{self.name}': k must be > 0, got {self.k}")
        if self.rho_c <= 0:
            raise ValueError(
                f"Layer '{self.name}': rho_c must be > 0, got {self.rho_c}"
            )


@dataclass(frozen=True)
class Domain2D:
    """Rectangular 2-D computational domain.

    Attributes:
        xmin, xmax: Horizontal extent [m].
        ymin, ymax: Vertical extent [m] (ymax = surface, ymin = deep).
    """
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmax <= self.xmin:
            raise ValueError("xmax must be > xmin")
        if self.ymax <= self.ymin:
            raise ValueError("ymax must be > ymin")


@dataclass(frozen=True)
class CablePlacement:
    """Centre position of one cable in the domain.

    Attributes:
        cable_id: Integer identifier (used when several cables exist).
        cx: Horizontal centre [m].
        cy: Vertical centre [m].
    """
    cable_id: int
    cx: float
    cy: float


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition on one edge of the rectangular domain.

    Attributes:
        boundary: Edge name (``"top"`` / ``"bottom"`` / ``"left"`` / ``"right"``).
        bc_type:  ``"dirichlet"`` | ``"neumann"`` | ``"robin"``.
        value:    Prescribed temperature [K] (Dirichlet) or flux [W/m^2] (Neumann).
        h:        Convection coefficient [W/(m^2 K)] (Robin only).
    """
    boundary: str
    bc_type: str
    value: float
    h: float

    _VALID_TYPES = {"dirichlet", "neumann", "robin"}

    def __post_init__(self) -> None:
        if self.bc_type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown BC type '{self.bc_type}'; "
                f"valid: {self._VALID_TYPES}"
            )


@dataclass(frozen=True)
class SoilProperties:
    """Thermal properties of the surrounding soil.

    Attributes:
        k:        Base thermal conductivity [W/(m K)].
        rho_c:    Volumetric heat capacity [J/(m^3 K)].
        variable: Whether *k* varies spatially.
        amp:      Amplitude of sinusoidal variation (fraction of *k*).
    """
    k: float
    rho_c: float
    variable: bool
    amp: float


@dataclass(frozen=True)
class Scenario:
    """One simulation scenario (row in ``scenarios.csv``).

    Attributes:
        scenario_id: Unique name.
        mode:        ``"steady"`` or ``"transient"``.
        Q_scale:     Multiplier applied to all layer Q values.
        k_soil:      Override for the base soil conductivity [W/(m K)].
        T_amb:       Ambient / boundary temperature [K].
        t_end:       End time [s] (0 for steady-state).
    """
    scenario_id: str
    mode: str
    Q_scale: float
    k_soil: float
    T_amb: float
    t_end: float

    _VALID_MODES = {"steady", "transient"}

    def __post_init__(self) -> None:
        if self.mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown mode '{self.mode}'; valid: {self._VALID_MODES}"
            )


@dataclass
class ProblemDefinition:
    """Complete problem assembled from all CSV files.

    Attributes:
        layers:     Ordered cable layers (inner → outer).
        domain:     Computational domain.
        placements: Cable positions.
        bcs:        Boundary conditions keyed by edge name.
        soil:       Soil thermal properties.
        scenarios:  Available simulation scenarios.
    """
    layers: list[CableLayer]
    domain: Domain2D
    placements: list[CablePlacement]
    bcs: dict[str, BoundaryCondition]
    soil: SoilProperties
    scenarios: list[Scenario]


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def _read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file and return a list of row dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_cable_layers(path: str | Path) -> list[CableLayer]:
    """Load cable layer definitions from *cable_layers.csv*.

    Expected columns: ``layer_name, r_inner, r_outer, k, rho_c, Q``.
    """
    rows = _read_csv(path)
    layers: list[CableLayer] = []
    for r in rows:
        layers.append(CableLayer(
            name=r["layer_name"].strip(),
            r_inner=float(r["r_inner"]),
            r_outer=float(r["r_outer"]),
            k=float(r["k"]),
            rho_c=float(r["rho_c"]),
            Q=float(r["Q"]),
        ))
    return layers


def load_domain(path: str | Path) -> Domain2D:
    """Load domain extents from *domain.csv*.

    Expected columns: ``param, value`` with rows for
    ``xmin``, ``xmax``, ``ymin``, ``ymax``.
    """
    rows = _read_csv(path)
    d: dict[str, float] = {}
    for r in rows:
        d[r["param"].strip()] = float(r["value"])
    required = {"xmin", "xmax", "ymin", "ymax"}
    missing = required - d.keys()
    if missing:
        raise ValueError(f"Missing domain parameters: {missing}")
    return Domain2D(**{k: d[k] for k in ("xmin", "xmax", "ymin", "ymax")})


def load_placements(path: str | Path) -> list[CablePlacement]:
    """Load cable centre positions from *cables_placement.csv*.

    Expected columns: ``cable_id, cx, cy``.
    """
    rows = _read_csv(path)
    return [
        CablePlacement(
            cable_id=int(r["cable_id"]),
            cx=float(r["cx"]),
            cy=float(r["cy"]),
        )
        for r in rows
    ]


def load_boundary_conditions(
    path: str | Path,
) -> dict[str, BoundaryCondition]:
    """Load boundary conditions from *boundary_conditions.csv*.

    Expected columns: ``boundary, type, value, h``.
    Returns a dict keyed by boundary name (``"top"`` etc.).
    """
    rows = _read_csv(path)
    bcs: dict[str, BoundaryCondition] = {}
    for r in rows:
        name = r["boundary"].strip()
        bcs[name] = BoundaryCondition(
            boundary=name,
            bc_type=r["type"].strip(),
            value=float(r["value"]),
            h=float(r["h"]),
        )
    return bcs


def load_soil_properties(path: str | Path) -> SoilProperties:
    """Load soil properties from *soil_properties.csv*.

    Expected columns: ``param, value`` with rows for
    ``k``, ``rho_c``, ``variable``, ``amp``.
    """
    rows = _read_csv(path)
    d: dict[str, str] = {}
    for r in rows:
        d[r["param"].strip()] = r["value"].strip()

    return SoilProperties(
        k=float(d["k"]),
        rho_c=float(d["rho_c"]),
        variable=d["variable"].lower() in ("true", "1", "yes"),
        amp=float(d["amp"]),
    )


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load simulation scenarios from *scenarios.csv*.

    Expected columns:
    ``scenario_id, mode, Q_scale, k_soil, T_amb, t_end``.
    """
    rows = _read_csv(path)
    return [
        Scenario(
            scenario_id=r["scenario_id"].strip(),
            mode=r["mode"].strip(),
            Q_scale=float(r["Q_scale"]),
            k_soil=float(r["k_soil"]),
            T_amb=float(r["T_amb"]),
            t_end=float(r["t_end"]),
        )
        for r in rows
    ]


def load_problem(data_dir: str | Path) -> ProblemDefinition:
    """Load all problem CSV files from *data_dir* into a single object.

    Args:
        data_dir: Directory containing the six CSV files.

    Returns:
        Fully-populated :class:`ProblemDefinition`.
    """
    d = Path(data_dir)
    return ProblemDefinition(
        layers=load_cable_layers(d / "cable_layers.csv"),
        domain=load_domain(d / "domain.csv"),
        placements=load_placements(d / "cables_placement.csv"),
        bcs=load_boundary_conditions(d / "boundary_conditions.csv"),
        soil=load_soil_properties(d / "soil_properties.csv"),
        scenarios=load_scenarios(d / "scenarios.csv"),
    )
