"""Spatially-variable thermal conductivity k(x,y) via sigmoid transition.

Provides:
- :class:`PhysicsParams` — dataclass with R(T) + k-sigmoid configuration.
- :func:`load_physics_params` — read from CSV.
- :func:`k_scalar` — Python-only k(x,y) (for analytical background).
- :func:`k_tensor` — differentiable k(x,y) (for PDE loss).
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import torch


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
    """
    # --- R(T) ---
    I_A: float = 0.0
    R_ref: float = 0.0
    T_ref_R_K: float = 293.15
    alpha_R: float = 0.00393
    n_R_iter: int = 2

    # --- k(x,y) sigmoid ---
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
# k(x,y): scalar (Python) and tensor (differentiable)
# ---------------------------------------------------------------------------

def k_scalar(x: float, y: float, pp: PhysicsParams) -> float:
    """Scalar k(x,y) — NOT differentiable by PyTorch.

    Use for the analytical Kennelly background temperature.
    """
    if not pp.k_variable:
        return pp.k_bad
    dx = abs(x - pp.k_cx) - pp.k_width / 2.0
    dy = abs(y - pp.k_cy) - pp.k_height / 2.0
    d = max(dx, dy)
    sig = 1.0 / (1.0 + math.exp(d / pp.k_transition))
    return pp.k_bad + (pp.k_good - pp.k_bad) * sig


def k_tensor(xy: torch.Tensor, pp: PhysicsParams) -> torch.Tensor:
    """Differentiable k(x,y) tensor (N,1) — use in PDE loss.

    Args:
        xy: Coordinates in physical space, shape ``(N, 2)``.
        pp: Physics parameters with sigmoid k-field config.

    Returns:
        Thermal conductivity ``(N, 1)``.
    """
    dx = (xy[:, 0:1] - pp.k_cx).abs() - pp.k_width / 2.0
    dy = (xy[:, 1:2] - pp.k_cy).abs() - pp.k_height / 2.0
    d = torch.max(dx, dy)
    sig = torch.sigmoid(-d / pp.k_transition)
    return pp.k_bad + (pp.k_good - pp.k_bad) * sig


# ---------------------------------------------------------------------------
# Construccion automatica de funciones k para PDE e IEC
# ---------------------------------------------------------------------------

def make_k_functions(
    pp: PhysicsParams,
    k_soil: float,
    placements: list | None = None,
) -> tuple:
    """Construir funciones k(x,y) para PDE, IEC y background Kennelly.

    A partir de los *PhysicsParams* genera tres objetos:

    * **k_fn_pde** — ``Callable[[Tensor], Tensor]`` para la PDE
      ``div(k·grad T)=0``.
    * **k_eff_fn_iec** — ``Callable[[float, float], float]`` escalar
      para estimacion IEC con k variable.
    * **k_eff_bg** — escalar ``float`` con la k en el centroide del
      grupo de cables (para background Kennelly).

    Si ``pp.k_variable`` es ``False``, todas las funciones devuelven
    ``k_soil`` constante.

    Args:
        pp:         Parametros fisicos con config de zona k(x,y).
        k_soil:     Conductividad termica del suelo base [W/(mK)].
        placements: Posiciones de cables (para calcular centroide).
                    Si es ``None``, ``k_eff_bg`` se evalua en
                    ``(pp.k_cx, pp.k_cy)``.

    Returns:
        ``(k_fn_pde, k_eff_fn_iec, k_eff_bg)``
    """
    # Centroide para el background
    if placements is not None and len(placements) > 0:
        n = len(placements)
        cx_mean = sum(pl.cx for pl in placements) / n
        cy_mean = sum(pl.cy for pl in placements) / n
    else:
        cx_mean, cy_mean = pp.k_cx, pp.k_cy

    if pp.k_variable:
        k_eff_bg = k_scalar(cx_mean, cy_mean, pp)

        def k_fn_pde(xy_phys: torch.Tensor) -> torch.Tensor:
            return k_tensor(xy_phys, pp)

        def k_eff_fn_iec(x: float, y: float) -> float:
            return k_scalar(x, y, pp)
    else:
        k_eff_bg = k_soil

        def k_fn_pde(xy_phys: torch.Tensor) -> torch.Tensor:
            return torch.full((xy_phys.shape[0], 1), k_soil,
                              device=xy_phys.device, dtype=xy_phys.dtype)

        def k_eff_fn_iec(x: float, y: float) -> float:
            return k_soil

    return k_fn_pde, k_eff_fn_iec, k_eff_bg
