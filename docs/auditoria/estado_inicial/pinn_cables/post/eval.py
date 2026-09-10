"""Métricas de evaluación, predicción en grilla y helpers de benchmark.

Provee:
- Errores L2/L-inf.
- Evaluación en grilla 2-D para graficación.
- Evaluación puntual de temperatura en centros de conductores.
- Funciones analíticas de benchmark (MMS, Laplace, cilindro radial).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from pinn_cables.io.readers import Domain2D
from pinn_cables.pinn.utils import normalize_coords


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def l2_relative_error(T_pred: torch.Tensor, T_exact: torch.Tensor) -> float:
    """Relative L2 error between predicted and exact temperature fields.

    .. math:: \\frac{\\|T_{pred} - T_{exact}\\|_2}{\\|T_{exact}\\|_2}

    Args:
        T_pred:  Predicted values ``(N,)`` or ``(N, 1)``.
        T_exact: Reference values, same shape.

    Returns:
        Scalar relative error.
    """
    diff = (T_pred.view(-1) - T_exact.view(-1)).float()
    ref = T_exact.view(-1).float()
    denom = torch.norm(ref)
    if denom < 1e-15:
        return torch.norm(diff).item()
    return (torch.norm(diff) / denom).item()


def linf_error(T_pred: torch.Tensor, T_exact: torch.Tensor) -> float:
    """Maximum absolute error.

    Args:
        T_pred:  Predicted values.
        T_exact: Reference values.

    Returns:
        Scalar max |T_pred - T_exact|.
    """
    return (T_pred.view(-1) - T_exact.view(-1)).abs().max().item()


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_grid(
    model: nn.Module,
    domain: Domain2D,
    nx: int = 100,
    ny: int = 100,
    device: torch.device | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the trained model on a regular 2-D grid.

    Args:
        model:     Trained PINN (steady state, input dim 2).
        domain:    Computational domain.
        nx, ny:    Grid resolution.
        device:    Torch device.
        normalize: Whether to normalise coordinates to [-1, 1].

    Returns:
        ``(X, Y, T)`` — NumPy arrays of shape ``(ny, nx)`` suitable for
        ``matplotlib.pyplot.contourf``.
    """
    model.eval()
    xs = torch.linspace(domain.xmin, domain.xmax, nx, device=device)
    ys = torch.linspace(domain.ymin, domain.ymax, ny, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    if normalize:
        mins = torch.tensor([domain.xmin, domain.ymin], device=device)
        maxs = torch.tensor([domain.xmax, domain.ymax], device=device)
        xy_in = normalize_coords(xy, mins, maxs)
    else:
        xy_in = xy

    T = model(xy_in).cpu().numpy().reshape(ny, nx)
    return X.cpu().numpy(), Y.cpu().numpy(), T


@torch.no_grad()
def evaluate_on_grid_transient(
    model: nn.Module,
    domain: Domain2D,
    t_val: float,
    t_end: float,
    nx: int = 100,
    ny: int = 100,
    device: torch.device | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a transient model at a single time snapshot.

    Args:
        model:  Trained PINN (transient, input dim 3).
        domain: Computational domain.
        t_val:  Time value for the snapshot [s].
        t_end:  Maximum time (for normalisation).
        nx, ny: Grid resolution.
        device: Torch device.
        normalize: Normalise coordinates.

    Returns:
        ``(X, Y, T)`` NumPy arrays.
    """
    model.eval()
    xs = torch.linspace(domain.xmin, domain.xmax, nx, device=device)
    ys = torch.linspace(domain.ymin, domain.ymax, ny, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    flat = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    t_col = torch.full((flat.shape[0], 1), t_val, device=device)
    xyt = torch.cat([flat, t_col], dim=1)

    if normalize:
        mins = torch.tensor([domain.xmin, domain.ymin, 0.0], device=device)
        maxs = torch.tensor([domain.xmax, domain.ymax, t_end], device=device)
        xyt = normalize_coords(xyt, mins, maxs)

    T = model(xyt).cpu().numpy().reshape(ny, nx)
    return X.cpu().numpy(), Y.cpu().numpy(), T


# ---------------------------------------------------------------------------
# Evaluación puntual en centros de conductores
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_conductor_temps(
    model: nn.Module,
    placements: list,
    domain: Domain2D,
    device: torch.device,
    normalize: bool = True,
) -> list[float]:
    """Evaluar temperatura PINN en el centro de cada conductor.

    Normaliza coordenadas si ``normalize=True`` y retorna una lista de
    temperaturas [K], una por cable (en el orden de *placements*).

    Args:
        model:      Modelo PINN entrenado.
        placements: Lista de :class:`CablePlacement`.
        domain:     Dominio computacional.
        device:     Dispositivo de torch.
        normalize:  Normalizar coordenadas a [-1, 1].

    Returns:
        Lista de temperaturas [K] en el centro de cada conductor.
    """
    coord_mins = torch.tensor(
        [domain.xmin, domain.ymin], device=device, dtype=torch.float32,
    )
    coord_maxs = torch.tensor(
        [domain.xmax, domain.ymax], device=device, dtype=torch.float32,
    )
    model.eval()
    T_list: list[float] = []
    for p in placements:
        pt = torch.tensor([[p.cx, p.cy]], device=device, dtype=torch.float32)
        if normalize:
            pt = 2.0 * (pt - coord_mins) / (coord_maxs - coord_mins) - 1.0
        T_list.append(float(model(pt).item()))
    return T_list


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def mms_source_constant_k(
    xy: torch.Tensor, k: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manufactured-solution pair for constant *k*.

    Exact solution: :math:`T^*(x,y) = \\sin(\\pi x)\\sin(\\pi y)`.

    Source term:    :math:`Q = 2 k \\pi^2 \\sin(\\pi x)\\sin(\\pi y)`.

    Args:
        xy: Coordinates ``(N, 2)``.
        k:  Constant thermal conductivity.

    Returns:
        ``(T_exact, Q)`` each of shape ``(N, 1)``.
    """
    x, y = xy[:, 0:1], xy[:, 1:2]
    T_exact = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    Q = 2.0 * k * math.pi**2 * T_exact
    return T_exact, Q


def mms_source_variable_k(
    xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MMS triplet for variable *k(x,y)*.

    - :math:`T^* = x^2 + y^2`
    - :math:`k   = 1 + 0.5\\,x`
    - :math:`Q   = -(3x + 4)`  (derived so that the PDE is satisfied exactly)

    Args:
        xy: Coordinates ``(N, 2)``.

    Returns:
        ``(T_exact, k, Q)`` each ``(N, 1)``.
    """
    x, y = xy[:, 0:1], xy[:, 1:2]
    T_exact = x**2 + y**2
    k = 1.0 + 0.5 * x
    Q = -(3.0 * x + 4.0)
    return T_exact, k, Q


def analytical_radial(
    r: torch.Tensor,
    r_in: float,
    r_out: float,
    T_in: float,
    T_out: float,
) -> torch.Tensor:
    """Analytical steady-state temperature for radial conduction in a cylinder.

    .. math:: T(r) = T_{in} + (T_{out} - T_{in}) \\frac{\\ln(r/r_{in})}{\\ln(r_{out}/r_{in})}

    Args:
        r:     Radial distance ``(N, 1)``.
        r_in:  Inner radius.
        r_out: Outer radius.
        T_in:  Temperature at inner radius.
        T_out: Temperature at outer radius.

    Returns:
        Temperature ``(N, 1)``.
    """
    log_ratio = torch.log(r / r_in) / math.log(r_out / r_in)
    return T_in + (T_out - T_in) * log_ratio
