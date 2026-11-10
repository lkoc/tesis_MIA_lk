"""Spatially-variable thermal conductivity k(x,y).

Architecture:
- :class:`SoilLayerBand`   — one horizontal soil stratum (y_top, y_bottom, k, rho_c).
- :class:`PhysicsParams`   — R(T) config + rectangular PAC/bedding zone params.
- :class:`KFieldModel`     — **central reusable class** that encapsulates ALL k(x,y)
                             logic: uniform, multilayer, PAC zone, or any combination.
                             Callable as PyTorch tensor function for the PDE, and has
                             Python scalar methods for IEC/Kennelly background use.
- :func:`load_physics_params` — read PhysicsParams from CSV.
- :func:`load_soil_layers`    — read list[SoilLayerBand] from CSV.
- :func:`make_k_functions`    — backward-compatible wrapper (returns tuple of closures).
- :func:`k_scalar`, :func:`k_tensor` — legacy module-level helpers (delegate to KFieldModel).
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# SoilLayerBand — one horizontal soil stratum
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SoilLayerBand:
    """One horizontal soil stratum with constant thermal properties.

    The domain is divided into horizontal bands from y_top (shallower)
    to y_bottom (deeper), where y = 0 is the surface and y < 0 is depth.
    Bands must be contiguous and non-overlapping when used in a list.

    Attributes:
        y_top:    Top boundary of the band [m], inclusive (y ≥ y_top).
        y_bottom: Bottom boundary [m], exclusive (y < y_bottom).
        k:        Thermal conductivity [W/(m·K)].
        rho_c:    Volumetric heat capacity [J/(m³·K)]. 0 means unspecified.
    """
    y_top: float
    y_bottom: float
    k: float
    rho_c: float = 0.0

    def __post_init__(self) -> None:
        if self.y_bottom >= self.y_top:
            raise ValueError(
                f"SoilLayerBand: y_bottom ({self.y_bottom}) must be < y_top ({self.y_top})"
            )
        if self.k <= 0:
            raise ValueError(f"SoilLayerBand: k must be > 0, got {self.k}")


def load_soil_layers(path: Path) -> list[SoilLayerBand]:
    """Load soil layer bands from a ``layer_id,y_top,y_bottom,k[,rho_c]`` CSV file.

    Rows are sorted by y_top descending (shallowest first) after loading.

    Args:
        path: Path to the CSV file.

    Returns:
        List of :class:`SoilLayerBand`, sorted shallow → deep.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"soil_layers CSV not found: {path}")
    bands: list[SoilLayerBand] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#"))
        )
        for row in reader:
            bands.append(SoilLayerBand(
                y_top=float(row["y_top"]),
                y_bottom=float(row["y_bottom"]),
                k=float(row["k"]),
                rho_c=float(row.get("rho_c", 0.0) or 0.0),
            ))
    # Sort from surface (y_top closest to 0) to deepest
    bands.sort(key=lambda b: b.y_top, reverse=True)
    return bands


# ---------------------------------------------------------------------------
# PhysicsParams — R(T) and rectangular PAC/bedding zone
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PhysicsParams:
    """Extra physics: temperature-dependent resistance R(T) and sigmoid k(x,y).

    R(T): conductor resistance increases linearly with temperature,
    raising dissipated power via Q_lin = I² R(T).

    k(x,y): soil thermal conductivity varies with distance to a
    'good' region centred at (k_cx, k_cy) of size k_width × k_height.
    Smooth sigmoid transition avoids gradient discontinuities.

    Formula::

        d = max(|x − k_cx| − k_width/2, |y − k_cy| − k_height/2)
        k = k_bad + (k_good − k_bad) × σ(−d / k_transition)

    Note:
        ``k_bad`` is only used when there are no :class:`SoilLayerBand` s.
        When a ``KFieldModel`` has both soil bands and a PAC zone, ``k_bad``
        is replaced by the layer k evaluated at each point.
    """
    # --- R(T) ---
    I_A: float = 0.0
    R_ref: float = 0.0
    T_ref_R_K: float = 293.15
    alpha_R: float = 0.00393
    n_R_iter: int = 2

    # --- k(x,y) sigmoid (PAC/bedding zone) ---
    k_variable: bool = True
    k_good: float = 1.5
    k_bad: float = 0.8
    k_cx: float = 0.0
    k_cy: float = -0.70
    k_width: float = 0.5
    k_height: float = 0.5
    k_transition: float = 0.05


def load_physics_params(path: Path) -> PhysicsParams:
    """Load :class:`PhysicsParams` from a ``param,value`` CSV file.

    Parameters absent from the file use the dataclass defaults.
    """
    if not path.exists():
        return PhysicsParams()
    raw: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw[row["param"].strip()] = row["value"].strip()

    fields = {f.name: f for f in dataclasses.fields(PhysicsParams)}
    kwargs: dict = {}
    for key, val_str in raw.items():
        if key not in fields:
            continue
        ft = fields[key].type
        if ft in (bool, "bool"):
            kwargs[key] = val_str.lower() in ("true", "1", "yes")
        elif ft in (int, "int"):
            kwargs[key] = int(val_str)
        else:
            try:
                kwargs[key] = float(val_str)
            except ValueError:
                kwargs[key] = val_str
    return PhysicsParams(**kwargs)


# ---------------------------------------------------------------------------
# KFieldModel — central reusable k(x,y) class
# ---------------------------------------------------------------------------

class KFieldModel:
    """Encapsulates the full spatially-variable thermal conductivity field k(x,y).

    Supports three orthogonal features that can be combined freely:

    1. **Uniform soil** — single constant k everywhere (default).
    2. **Multilayer soil** — horizontal bands with different k; smooth
       sigmoid transitions at each interface prevents gradient discontinuities.
    3. **PAC/bedding zone** — rectangular high-conductivity zone around the
       cables (e.g. Prepacked Aggregate Concrete).  Described by
       :class:`PhysicsParams`.

    When both multilayer soil *and* a PAC zone are active, the combined formula
    is:

    .. math::

        k(x,y) = k_L(y) + [k_{PAC} - k_L(y)] \\cdot \\sigma_{PAC}(x,y)

    where :math:`k_L(y)` is the native-soil conductivity at depth y and
    :math:`\\sigma_{PAC}` is the sigmoid for the PAC rectangle.  This ensures
    that inside the PAC zone :math:`k \\to k_{PAC}`, and outside
    :math:`k \\to k_L(y)` regardless of depth.

    Usage::

        # 1. Uniform (no files needed)
        k_model = KFieldModel.uniform(k_soil=1.55)

        # 2. CSV factories
        k_model = KFieldModel.from_csvs(
            k_soil=1.55,
            soil_layers_path=Path("data/soil_layers.csv"),          # optional
            physics_params_path=Path("data/physics_params.csv"),    # optional
        )

        # 3. Use directly as k_fn in the PDE trainer
        k_vals = k_model(xy_tensor)   # returns (N, 1) tensor

        # 4. Scalar evaluation (for IEC / Kennelly background)
        k_at_point = k_model.k_scalar(x, y)
        k_bg = k_model.k_eff_bg(placements)

        # 5. Importance sampling hints (for the collocation point sampler)
        hints = k_model.transition_hints()

    Args:
        k_soil:         Base soil conductivity used when no soil bands are given,
                        and as the far-field reference [W/(m·K)].
        soil_bands:     Optional list of :class:`SoilLayerBand` (shallow → deep).
        pac_params:     Optional :class:`PhysicsParams` describing a PAC/bedding
                        rectangular zone (only used when ``pac_params.k_variable``
                        is ``True``).
        layer_transition: Sigmoid half-width [m] for soil-layer interfaces.
    """

    def __init__(
        self,
        k_soil: float,
        soil_bands: list[SoilLayerBand] | None = None,
        pac_params: PhysicsParams | None = None,
        layer_transition: float = 0.05,
    ) -> None:
        self._k_soil = k_soil
        self._bands = soil_bands or []
        self._pac = pac_params
        self._layer_tr = layer_transition

        # Pre-validate that pac_params has k_variable=True when provided
        self._has_pac = (
            pac_params is not None and pac_params.k_variable
        )
        self._has_layers = len(self._bands) > 0

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def uniform(cls, k_soil: float) -> "KFieldModel":
        """Create a uniform (homogeneous) k field.

        Args:
            k_soil: Constant thermal conductivity [W/(m·K)].
        """
        return cls(k_soil=k_soil)

    @classmethod
    def from_csvs(
        cls,
        k_soil: float,
        soil_layers_path: Path | None = None,
        physics_params_path: Path | None = None,
        layer_transition: float = 0.05,
    ) -> "KFieldModel":
        """Build a KFieldModel from optional CSV files.

        Any file that is ``None`` or does not exist is simply skipped.

        Args:
            k_soil:              Base soil conductivity [W/(m·K)] (used as
                                 fallback when no soil bands cover a point).
            soil_layers_path:    Path to ``soil_layers.csv``.
            physics_params_path: Path to ``physics_params.csv`` (PAC zone).
            layer_transition:    Sigmoid half-width at soil layer interfaces [m].
        """
        bands: list[SoilLayerBand] | None = None
        if soil_layers_path is not None and Path(soil_layers_path).exists():
            bands = load_soil_layers(Path(soil_layers_path))

        pac: PhysicsParams | None = None
        if physics_params_path is not None and Path(physics_params_path).exists():
            pac = load_physics_params(Path(physics_params_path))

        return cls(
            k_soil=k_soil,
            soil_bands=bands,
            pac_params=pac,
            layer_transition=layer_transition,
        )

    # ------------------------------------------------------------------
    # Core k(y) — multilayer soil, Python scalar
    # ------------------------------------------------------------------

    def _k_layer_scalar(self, y: float) -> float:
        """Return k of the soil layer at depth y (Python scalar, no autograd).

        Falls back to ``k_soil`` when no bands are defined or y is outside all bands.
        """
        if not self._has_layers:
            return self._k_soil
        for band in self._bands:          # sorted shallow → deep
            if band.y_bottom <= y <= band.y_top:
                return band.k
        # Outside all bands: use nearest-band extrapolation or fallback
        return self._k_soil

    def _k_layer_tensor(self, xy: torch.Tensor) -> torch.Tensor:
        """Return k of the soil layer per point, as a differentiable tensor (N,1).

        Uses smooth sigmoid blending at each interface to avoid discontinuities
        that would cause large PDE residuals near band boundaries.

        The blending formula between adjacent bands with conductivities k_a
        (shallower) and k_b (deeper), with interface at y_interface, is::

            weight_b = sigmoid((y_interface - y) / layer_transition)
            k = k_a + (k_b - k_a) * weight_b

        This gives k → k_a for y > y_interface (above) and k → k_b for
        y < y_interface (below), with a smooth sigmoid transition of width
        ~4 * layer_transition centred at the interface.
        """
        if not self._has_layers:
            return torch.full(
                (xy.shape[0], 1), self._k_soil,
                device=xy.device, dtype=xy.dtype,
            )

        y = xy[:, 1:2]          # (N, 1)
        # Start with the shallowest (top) layer k
        k_out = torch.full(
            (xy.shape[0], 1), self._bands[0].k,
            device=xy.device, dtype=xy.dtype,
        )

        # Walk from top to bottom, blending in each interface
        for i in range(len(self._bands) - 1):
            k_above = self._bands[i].k
            k_below = self._bands[i + 1].k
            y_iface = self._bands[i].y_bottom   # = self._bands[i+1].y_top
            # weight_below approaches 1 as y -> -inf (deeper)
            weight_below = torch.sigmoid(
                (y_iface - y) / self._layer_tr
            )
            k_out = k_out + (k_below - k_above) * weight_below

        return k_out

    # ------------------------------------------------------------------
    # Core PAC sigmoid — Python and tensor
    # ------------------------------------------------------------------

    def _pac_sigma_scalar(self, x: float, y: float) -> float:
        """PAC sigmoid weight at (x,y): ~1 inside the zone, ~0 outside."""
        pp = self._pac
        dx = abs(x - pp.k_cx) - pp.k_width / 2.0
        dy = abs(y - pp.k_cy) - pp.k_height / 2.0
        d = max(dx, dy)
        return 1.0 / (1.0 + math.exp(d / pp.k_transition))

    def _pac_sigma_tensor(self, xy: torch.Tensor) -> torch.Tensor:
        """PAC sigmoid weight per point (N,1), differentiable."""
        pp = self._pac
        dx = (xy[:, 0:1] - pp.k_cx).abs() - pp.k_width / 2.0
        dy = (xy[:, 1:2] - pp.k_cy).abs() - pp.k_height / 2.0
        d = torch.max(dx, dy)
        return torch.sigmoid(-d / pp.k_transition)

    # ------------------------------------------------------------------
    # Public interface — PyTorch callable (for PDE trainer)
    # ------------------------------------------------------------------

    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        """Compute k(x,y) as a differentiable tensor of shape (N, 1).

        This is the main entry point for the PDE loss computation.  The
        returned tensor supports autograd.

        Combined formula when both multilayer soil and PAC zone are active::

            k(x,y) = k_layer(y) + [k_PAC − k_layer(y)] * sigma_PAC(x,y)

        Args:
            xy: Physical coordinates ``(N, 2)`` in metres.

        Returns:
            Thermal conductivity ``(N, 1)`` tensor.
        """
        if not self._has_layers and not self._has_pac:
            # Uniform case: constant tensor, fastest path
            return torch.full(
                (xy.shape[0], 1), self._k_soil,
                device=xy.device, dtype=xy.dtype,
            )

        k_layer = self._k_layer_tensor(xy)   # (N,1)

        if not self._has_pac:
            return k_layer

        # PAC override: k = k_L + (k_PAC - k_L) * sigma_PAC
        sigma = self._pac_sigma_tensor(xy)   # (N,1), 0..1
        k_pac = self._pac.k_good
        return k_layer + (k_pac - k_layer) * sigma

    # ------------------------------------------------------------------
    # Public interface — Python scalar (for IEC / Kennelly background)
    # ------------------------------------------------------------------

    def k_scalar(self, x: float, y: float) -> float:
        """Evaluate k(x,y) as a Python float (no autograd).

        Use for IEC 60287 estimates and the Kennelly analytical background
        where PyTorch differentiation is not needed.

        Args:
            x: Horizontal coordinate [m].
            y: Vertical coordinate [m] (negative below surface).

        Returns:
            Thermal conductivity [W/(m·K)].
        """
        k_layer = self._k_layer_scalar(y)

        if not self._has_pac:
            return k_layer

        sigma = self._pac_sigma_scalar(x, y)
        k_pac = self._pac.k_good
        return k_layer + (k_pac - k_layer) * sigma

    # ------------------------------------------------------------------
    # Effective background k for Kennelly (centroid of cable group)
    # ------------------------------------------------------------------

    def k_eff_bg(self, placements: list | None = None) -> float:
        """Effective k at the centroid of the cable group.

        Used by the Kennelly analytical background to set the far-field
        soil conductivity for temperature superposition.

        Args:
            placements: Cable placements.  If ``None``, evaluates at the
                        PAC zone centre (if available) or returns ``k_soil``.

        Returns:
            Effective conductivity [W/(m·K)] at the group centroid.
        """
        if placements is not None and len(placements) > 0:
            cx = sum(pl.cx for pl in placements) / len(placements)
            cy = sum(pl.cy for pl in placements) / len(placements)
        elif self._has_pac:
            cx, cy = self._pac.k_cx, self._pac.k_cy
        else:
            return self._k_soil

        return self.k_scalar(cx, cy)

    # ------------------------------------------------------------------
    # Transition hints for importance sampling
    # ------------------------------------------------------------------

    def transition_hints(self) -> list[dict]:
        """Return regions that need extra collocation points (importance sampling).

        Each hint is a dict with keys describing a strip or rectangle where
        the conductivity gradient is large:

        - ``type``      : ``"horizontal_strip"`` or ``"pac_boundary"``
        - ``y_centre``  : (horizontal_strip) y coordinate of the interface [m]
        - ``half_width``: (horizontal_strip) half-thickness of the sampling band [m]
        - ``x_lo``, ``x_hi``, ``y_lo``, ``y_hi``: (pac_boundary) bounding box [m]

        Returns:
            List of hint dicts, one per transition region.
        """
        hints: list[dict] = []

        # Soil layer interfaces
        if self._has_layers:
            margin = max(4.0 * self._layer_tr, 0.10)
            for band in self._bands[:-1]:
                hints.append({
                    "type": "horizontal_strip",
                    "y_centre": band.y_bottom,
                    "half_width": margin,
                })

        # PAC zone boundary
        if self._has_pac:
            pp = self._pac
            margin = max(4.0 * pp.k_transition, 0.15)
            hints.append({
                "type": "pac_boundary",
                "x_lo": pp.k_cx - pp.k_width / 2.0 - margin,
                "x_hi": pp.k_cx + pp.k_width / 2.0 + margin,
                "y_lo": pp.k_cy - pp.k_height / 2.0 - margin,
                "y_hi": pp.k_cy + pp.k_height / 2.0 + margin,
            })

        return hints

    # ------------------------------------------------------------------
    # Properties for introspection / plotting
    # ------------------------------------------------------------------

    @property
    def k_soil(self) -> float:
        """Base soil conductivity [W/(m·K)]."""
        return self._k_soil

    @property
    def soil_bands(self) -> list[SoilLayerBand]:
        """List of soil layer bands (may be empty)."""
        return list(self._bands)

    @property
    def pac_params(self) -> PhysicsParams | None:
        """PAC/bedding zone params, or None if no PAC zone is active."""
        return self._pac if self._has_pac else None

    @property
    def has_layers(self) -> bool:
        """True when multilayer soil bands are configured."""
        return self._has_layers

    @property
    def has_pac(self) -> bool:
        """True when a PAC/bedding zone is active."""
        return self._has_pac

    def __repr__(self) -> str:
        parts = [f"KFieldModel(k_soil={self._k_soil}"]
        if self._has_layers:
            parts.append(f", {len(self._bands)} soil bands")
        if self._has_pac:
            parts.append(
                f", PAC k={self._pac.k_good} at "
                f"({self._pac.k_cx},{self._pac.k_cy})"
            )
        parts.append(")")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Backward-compatible module-level helpers
# ---------------------------------------------------------------------------

def k_scalar(x: float, y: float, pp: PhysicsParams) -> float:
    """Scalar k(x,y) — backward-compatible wrapper (delegates to KFieldModel).

    .. deprecated::
        Create a :class:`KFieldModel` directly instead.
    """
    model = KFieldModel(k_soil=pp.k_bad, pac_params=pp if pp.k_variable else None)
    return model.k_scalar(x, y)


def k_tensor(xy: torch.Tensor, pp: PhysicsParams) -> torch.Tensor:
    """Differentiable k(x,y) tensor (N,1) — backward-compatible wrapper.

    .. deprecated::
        Use :class:`KFieldModel` directly instead.
    """
    model = KFieldModel(k_soil=pp.k_bad, pac_params=pp if pp.k_variable else None)
    return model(xy)


def make_k_functions(
    pp: PhysicsParams,
    k_soil: float,
    placements: list | None = None,
) -> tuple:
    """Backward-compatible factory returning ``(k_fn_pde, k_eff_fn_iec, k_eff_bg)``.

    Builds a :class:`KFieldModel` internally and returns three callables
    with the same signatures as the original implementation.

    Args:
        pp:         PhysicsParams with PAC zone config.
        k_soil:     Base soil conductivity [W/(m·K)].
        placements: Cable placements (for centroid k_eff_bg evaluation).

    Returns:
        ``(k_fn_pde, k_eff_fn_iec, k_eff_bg_float)``
    """
    model = KFieldModel(
        k_soil=k_soil,
        pac_params=pp if pp.k_variable else None,
    )

    def k_fn_pde(xy_phys: torch.Tensor) -> torch.Tensor:
        return model(xy_phys)

    def k_eff_fn_iec(x: float, y: float) -> float:
        return model.k_scalar(x, y)

    k_eff_bg_float = model.k_eff_bg(placements)
    return k_fn_pde, k_eff_fn_iec, k_eff_bg_float
