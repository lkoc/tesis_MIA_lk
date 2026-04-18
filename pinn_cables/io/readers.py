"""CSV readers for problem data files.

Each function reads one CSV file and returns typed dataclasses.
Uses only the ``csv`` module from the standard library.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from pinn_cables.physics.ground_temp import GroundTempProfile


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
        section_mm2: Nominal conductor cross-section [mm²]
            (e.g. 95, 150, 240, 400, 600).  ``0`` means "use
            cable_layers.csv" (backward compatibility).
        conductor_material: ``"cu"`` (copper) or ``"al"`` (aluminium).
            Ignored when *section_mm2* is 0.
        current_A: Operating current [A].  When a non-zero
            *section_mm2* is given the volumetric heat source *Q* is
            computed as ``I² R_dc / A_cond``.
    """
    cable_id: int
    cx: float
    cy: float
    section_mm2: int = 0
    conductor_material: str = "cu"
    current_A: float = 0.0


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition on one edge of the rectangular domain.

    Attributes:
        boundary: Edge name (``"top"`` / ``"bottom"`` / ``"left"`` / ``"right"``).
        bc_type:  ``"dirichlet"`` | ``"neumann"`` | ``"robin"``.
        value:    Prescribed temperature [K] (Dirichlet) or flux [W/m^2] (Neumann).
                  Used only when *profile* is ``None``.
        h:        Convection coefficient [W/(m^2 K)] (Robin only).
        profile:  Optional spatially-varying temperature profile
                  (:class:`~pinn_cables.physics.ground_temp.GroundTempProfile`).
                  When set, overrides the scalar *value* for Dirichlet targets.
                  Not loaded from CSV — set programmatically via
                  ``dataclasses.replace(bc, profile=my_profile)``.
    """
    boundary: str
    bc_type: str
    value: float
    h: float
    profile: "GroundTempProfile | None" = field(
        default=None, compare=False, hash=False, repr=False
    )

    _VALID_TYPES = {"dirichlet", "neumann", "robin"}

    def __post_init__(self) -> None:
        if self.bc_type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown BC type '{self.bc_type}'; "
                f"valid: {self._VALID_TYPES}"
            )

    def T_target(self, xy: "torch.Tensor", T_amb: float = 0.0) -> "torch.Tensor":
        """Return the target temperature tensor ``(N, 1)`` [K] at *xy*.

        If a *profile* is attached, delegates to it (spatially-varying).
        Otherwise falls back to the scalar *value* (or *T_amb* when
        *value* is 0, preserving legacy behaviour).

        Args:
            xy:    ``(N, 2)`` coordinate tensor [m].
            T_amb: Fallback ambient temperature [K] used when ``value == 0``.

        Returns:
            ``(N, 1)`` temperature tensor [K].
        """
        if self.profile is not None:
            return self.profile(xy)
        val = self.value if self.value != 0 else T_amb
        return xy.new_full((xy.shape[0], 1), val)


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


@dataclass(frozen=True)
class SolverParams:
    """ML/solver hyperparameters loaded from ``solver_params.csv``.

    Flat dataclass that mirrors every knob exposed in ``solver.yaml``.
    Call :meth:`to_solver_cfg` to get the nested dict expected by
    ``build_model``, ``TrainConfig.from_config``, etc.

    Attributes — model:
        model_width:           Hidden-layer width.
        model_depth:           Number of hidden layers.
        model_activation:      Activation function name.
        model_fourier_features: Use Fourier feature mapping.
        model_fourier_scale:   Scale of Fourier frequencies.
        model_fourier_mapping_size: Size of Fourier mapping.

    Attributes — training:
        lr:              Adam learning rate.
        adam_steps:      Adam iterations.
        lbfgs_steps:     L-BFGS outer iterations (0 = disabled).
        lbfgs_history:   L-BFGS history size.
        print_every:     Logging frequency (Adam steps).
        save_every:      Checkpoint frequency (0 = none).
        resample_every:  Re-sample collocation points every N steps.
        use_compile:     Apply ``torch.compile``.
        checkpoint_dir:  Checkpoint directory path.

    Attributes — sampling:
        n_interior:   Interior collocation points.
        n_interface:  Interface collocation points.
        n_boundary:   Boundary collocation points.
        oversample:   Over-sampling factor for rejection sampling.

    Attributes — other:
        normalize_coords: Normalize spatial coordinates to [-1,1].
        w_pde, w_bc_dirichlet, w_bc_neumann, w_bc_robin,
        w_interface_T, w_interface_flux, w_ic: Loss weights.
        n_time:   Time collocation points (transient mode).
        device:   Torch device string (``"auto"``, ``"cpu"``, ``"cuda"``).
        seed:     Random seed.
        log_dir:  Directory for training logs.
    """
    # model
    model_width: int = 128
    model_depth: int = 6
    model_activation: str = "tanh"
    model_fourier_features: bool = False
    model_fourier_scale: float = 1.0
    model_fourier_mapping_size: int = 64
    # training
    lr: float = 1e-3
    adam_steps: int = 20_000
    lbfgs_steps: int = 5_000
    lbfgs_history: int = 50
    adam2_steps: int = 0
    adam2_lr: float = 1e-5
    print_every: int = 200
    save_every: int = 0
    resample_every: int = 0
    use_compile: bool = False
    checkpoint_dir: str = "checkpoints/"
    # sampling
    n_interior: int = 8_000
    n_interface: int = 500
    n_boundary: int = 400
    oversample: int = 5
    min_per_region: int = 20
    # normalisation
    normalize_coords: bool = True
    # loss weights
    w_pde: float = 1.0
    w_bc_dirichlet: float = 10.0
    w_bc_neumann: float = 1.0
    w_bc_robin: float = 10.0
    w_interface_T: float = 10.0
    w_interface_flux: float = 10.0
    w_ic: float = 10.0
    w_cable_flux: float = 5.0
    # time (transient)
    n_time: int = 200
    # execution
    device: str = "auto"
    seed: int = 42
    log_dir: str = "runs/"

    def to_solver_cfg(self) -> dict[str, Any]:
        """Convert to the nested dict consumed by existing solver code.

        Returns a structure identical to the ``solver.yaml`` layout so
        that :func:`pinn_cables.pinn.model.build_model`,
        :class:`pinn_cables.pinn.train.TrainConfig`, etc. work without
        any changes.
        """
        return {
            "device": self.device,
            "seed": self.seed,
            "model": {
                "width": self.model_width,
                "depth": self.model_depth,
                "activation": self.model_activation,
                "fourier_features": self.model_fourier_features,
                "fourier_scale": self.model_fourier_scale,
                "fourier_mapping_size": self.model_fourier_mapping_size,
            },
            "sampling": {
                "n_interior": self.n_interior,
                "n_interface": self.n_interface,
                "n_boundary": self.n_boundary,
                "oversample": self.oversample,
                "min_per_region": self.min_per_region,
            },
            "normalization": {
                "normalize_coords": self.normalize_coords,
            },
            "training": {
                "lr": self.lr,
                "adam_steps": self.adam_steps,
                "lbfgs_steps": self.lbfgs_steps,
                "lbfgs_history": self.lbfgs_history,
                "adam2_steps": self.adam2_steps,
                "adam2_lr": self.adam2_lr,
                "print_every": self.print_every,
                "save_every": self.save_every,
                "resample_every": self.resample_every,
                "use_compile": self.use_compile,
                "checkpoint_dir": self.checkpoint_dir,
            },
            "loss_weights": {
                "pde": self.w_pde,
                "bc_dirichlet": self.w_bc_dirichlet,
                "bc_neumann": self.w_bc_neumann,
                "bc_robin": self.w_bc_robin,
                "interface_T": self.w_interface_T,
                "interface_flux": self.w_interface_flux,
                "ic": self.w_ic,
                "cable_flux": self.w_cable_flux,
            },
            "time": {"n_time": self.n_time},
            "logging": {"log_dir": self.log_dir},
        }


# ---------------------------------------------------------------------------
# Helpers para modificar capas de cable
# ---------------------------------------------------------------------------

def override_conductor_Q(
    layers: list[CableLayer],
    Q_total_lin: float,
) -> list[CableLayer]:
    """Reemplazar Q volumetrico del conductor para inyectar *Q_total_lin*.

    Recalcula ``Q = Q_total_lin / A_conductor`` y devuelve una nueva
    lista de capas con el conductor modificado.  Las demas capas se
    mantienen iguales.

    Args:
        layers:       Lista original de CableLayer (la primera es el conductor).
        Q_total_lin:  Calor total lineal a evacuar [W/m].

    Returns:
        Nueva lista de CableLayer.
    """
    conductor = layers[0]
    A_cond = math.pi * conductor.r_outer ** 2
    return [
        CableLayer(
            name=conductor.name,
            r_inner=conductor.r_inner,
            r_outer=conductor.r_outer,
            k=conductor.k,
            rho_c=conductor.rho_c,
            Q=Q_total_lin / A_cond,
        ),
    ] + list(layers[1:])


@dataclass
class ProblemDefinition:
    """Complete problem assembled from all CSV files.

    Attributes:
        layers:        Default cable layers (inner → outer) loaded from
                       ``cable_layers.csv``.  Used as fallback when
                       *layers_per_cable* is empty.
        domain:        Computational domain.
        placements:    Cable positions (may include per-cable type info).
        bcs:           Boundary conditions keyed by edge name.
        soil:          Soil thermal properties.
        scenarios:     Available simulation scenarios.
        solver_params: Solver/ML hyperparameters (``None`` if no
                       ``solver_params.csv`` present in data dir).
        layers_per_cable: Per-cable layer lists keyed by *cable_id*.
                       Populated automatically when ``cables_placement.csv``
                       contains ``section_mm2`` / ``conductor_material`` /
                       ``current_A`` columns.
    """
    layers: list[CableLayer]
    domain: Domain2D
    placements: list[CablePlacement]
    bcs: dict[str, BoundaryCondition]
    soil: SoilProperties
    scenarios: list[Scenario]
    solver_params: SolverParams | None = None
    layers_per_cable: dict[int, list[CableLayer]] | None = None

    def get_layers(self, cable_id: int) -> list[CableLayer]:
        """Return layers for a specific cable.

        Falls back to the default ``layers`` when no per-cable entry
        exists.
        """
        if self.layers_per_cable and cable_id in self.layers_per_cable:
            return self.layers_per_cable[cable_id]
        return self.layers


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

    Accepted formats:

    * **Wide** (one row): columns ``xmin, xmax, ymin, ymax``.
    * **Long** (multiple rows): columns ``param, value`` with rows for
      ``xmin``, ``xmax``, ``ymin``, ``ymax``.
    """
    rows = _read_csv(path)
    d: dict[str, float] = {}
    if rows and "param" in rows[0] and "value" in rows[0]:
        # Long / key-value format
        for r in rows:
            d[r["param"].strip()] = float(r["value"])
    else:
        # Wide / columnar format - each column name is a parameter
        if len(rows) != 1:
            raise ValueError(
                f"Wide-format domain.csv must have exactly 1 data row, got {len(rows)}"
            )
        for k, v in rows[0].items():
            d[k.strip()] = float(v)
    required = {"xmin", "xmax", "ymin", "ymax"}
    missing = required - d.keys()
    if missing:
        raise ValueError(f"Missing domain parameters: {missing}")
    return Domain2D(**{k: d[k] for k in ("xmin", "xmax", "ymin", "ymax")})


def load_placements(path: str | Path) -> list[CablePlacement]:
    """Load cable centre positions from *cables_placement.csv*.

    Expected columns: ``cable_id, cx, cy``.
    Optional columns (for per-cable cable-type selection):
    ``section_mm2, conductor_material, current_A``.
    """
    rows = _read_csv(path)
    return [
        CablePlacement(
            cable_id=int(r["cable_id"]),
            cx=float(r["cx"]),
            cy=float(r["cy"]),
            section_mm2=int(r["section_mm2"]) if "section_mm2" in r else 0,
            conductor_material=r.get("conductor_material", "cu").strip().lower(),
            current_A=float(r["current_A"]) if "current_A" in r else 0.0,
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


def load_boundary_profiles(
    path: str | Path,
) -> "dict[str, PiecewiseLinearProfile]":
    """Load piecewise-linear BC profiles from *boundary_profiles.csv*.

    Expected columns: ``boundary, depth_m, T_K``.

    Each boundary can have an arbitrary number of rows (knots), and rows
    do not need to be sorted — the loader sorts them by depth.
    Returns a dict keyed by boundary name.

    Example file::

        boundary,depth_m,T_K
        left,0.0,299.15
        left,1.4,294.26
        left,5.0,288.35
        right,0.0,299.15
        right,1.4,294.26
        right,5.0,288.35
        bottom,0.0,288.35
    """
    from pinn_cables.physics.ground_temp import PiecewiseLinearProfile

    rows = _read_csv(path)
    knots: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        name = r["boundary"].strip()
        knots.setdefault(name, []).append(
            (float(r["depth_m"]), float(r["T_K"]))
        )
    return {
        name: PiecewiseLinearProfile(
            depths=[k[0] for k in pts],
            temps=[k[1] for k in pts],
        )
        for name, pts in knots.items()
    }


def with_profiles(
    bcs: dict[str, BoundaryCondition],
    profiles: "dict[str, GroundTempProfile]",
) -> dict[str, BoundaryCondition]:
    """Return a new BC dict with temperature profiles attached.

    Uses :func:`dataclasses.replace` so the original dict is unchanged.
    Boundaries whose name is not in *profiles* are returned as-is.

    Args:
        bcs:      Original BC dict (e.g. from :func:`load_boundary_conditions`).
        profiles: Dict mapping boundary name to a
                  :class:`~pinn_cables.physics.ground_temp.GroundTempProfile`.

    Returns:
        New dict with profiles attached to matching boundaries.
    """
    import dataclasses

    return {
        name: dataclasses.replace(bc, profile=profiles[name])
        if name in profiles
        else bc
        for name, bc in bcs.items()
    }


def load_soil_properties(path: str | Path) -> SoilProperties:
    """Load soil properties from *soil_properties.csv*.

    Required row: ``rho_c``.  Optional rows (with defaults):
    ``k`` (default 1.0), ``variable`` (default false), ``amp`` (default 0.0).

    Keeping the CSV minimal (``rho_c`` only) is recommended; the soil thermal
    conductivity is better specified in ``scenarios.csv`` via ``k_soil``.
    """
    rows = _read_csv(path)
    d: dict[str, str] = {}
    for r in rows:
        d[r["param"].strip()] = r["value"].strip()

    return SoilProperties(
        k=float(d.get("k", "1.0")),
        rho_c=float(d["rho_c"]),
        variable=d.get("variable", "false").lower() in ("true", "1", "yes"),
        amp=float(d.get("amp", "0.0")),
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


def load_solver_params(path: str | Path) -> SolverParams:
    """Load ML/solver hyperparameters from *solver_params.csv*.

    The CSV uses a ``param, value`` layout (same as ``domain.csv``).
    Unknown parameter names are silently ignored so new parameters can
    be added without breaking older CSV files.

    Args:
        path: Path to ``solver_params.csv``.

    Returns:
        :class:`SolverParams` with all available fields populated.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    rows = _read_csv(path)
    raw: dict[str, str] = {r["param"].strip(): r["value"].strip() for r in rows}

    # Explicit type mapping for every field in SolverParams.
    _INT_FIELDS = {
        "model_width", "model_depth", "model_fourier_mapping_size",
        "adam_steps", "lbfgs_steps", "lbfgs_history", "adam2_steps",
        "print_every", "save_every", "resample_every",
        "n_interior", "n_interface", "n_boundary", "oversample", "min_per_region",
        "n_time", "seed",
    }
    _FLOAT_FIELDS = {
        "model_fourier_scale", "lr", "adam2_lr",
        "w_pde", "w_bc_dirichlet", "w_bc_neumann", "w_bc_robin",
        "w_interface_T", "w_interface_flux", "w_ic", "w_cable_flux",
    }
    _BOOL_FIELDS = {"model_fourier_features", "use_compile", "normalize_coords"}
    # All remaining fields are strings: model_activation, checkpoint_dir, device, log_dir

    def _parse_bool(s: str) -> bool:
        return s.lower() in ("true", "1", "yes")

    kwargs: dict[str, Any] = {}
    for name, val in raw.items():
        if name in _INT_FIELDS:
            kwargs[name] = int(float(val))   # float() first handles "2.0e3"
        elif name in _FLOAT_FIELDS:
            kwargs[name] = float(val)
        elif name in _BOOL_FIELDS:
            kwargs[name] = _parse_bool(val)
        else:
            # string field (or unknown — will be ignored by dataclass)
            # only store if it's actually a known field name
            kwargs[name] = val

    # Filter out any unrecognised keys to avoid TypeError in the dataclass
    valid = {f.name for f in __import__("dataclasses").fields(SolverParams)}
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return SolverParams(**kwargs)


def load_problem(data_dir: str | Path) -> ProblemDefinition:
    """Load all problem CSV files from *data_dir* into a single object.

    The six physical-data CSVs are mandatory.  ``solver_params.csv`` is
    optional: when present it is loaded as a :class:`SolverParams`
    instance and stored in :attr:`ProblemDefinition.solver_params`.

    When ``cables_placement.csv`` contains the optional columns
    ``section_mm2``, ``conductor_material`` and ``current_A``, per-cable
    layers are generated automatically from the built-in XLPE cable
    catalog (see :mod:`pinn_cables.materials.props`).

    Args:
        data_dir: Directory containing the CSV files.

    Returns:
        Fully-populated :class:`ProblemDefinition`.
    """
    d = Path(data_dir)
    sp_path = d / "solver_params.csv"
    solver_params = load_solver_params(sp_path) if sp_path.exists() else None

    layers = load_cable_layers(d / "cable_layers.csv")
    placements = load_placements(d / "cables_placement.csv")

    # Auto-generate per-cable layers from catalog when section_mm2 > 0
    layers_per_cable: dict[int, list[CableLayer]] | None = None
    if any(p.section_mm2 > 0 for p in placements):
        from pinn_cables.materials.props import generate_cable_layers
        layers_per_cable = {}
        for p in placements:
            if p.section_mm2 > 0:
                layers_per_cable[p.cable_id] = generate_cable_layers(
                    p.section_mm2, p.conductor_material, p.current_A,
                )
            # else: will fall back to default `layers`

    bcs = load_boundary_conditions(d / "boundary_conditions.csv")
    bp_path = d / "boundary_profiles.csv"
    if bp_path.exists():
        bcs = with_profiles(bcs, load_boundary_profiles(bp_path))

    return ProblemDefinition(
        layers=layers,
        domain=load_domain(d / "domain.csv"),
        placements=placements,
        bcs=bcs,
        soil=load_soil_properties(d / "soil_properties.csv"),
        scenarios=load_scenarios(d / "scenarios.csv"),
        solver_params=solver_params,
        layers_per_cable=layers_per_cable,
    )
