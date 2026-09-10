"""Evaluación de resultados del FNO: extracción de T_max, mapas de temperatura.

Funciones
---------
eval_fno_field
    Evalúa el FNO en la malla completa y devuelve T(x,y) en Kelvin.
fno_T_max
    Encuentra la temperatura máxima en los conductores (T_max).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from pinn_cables.fno.dataset import make_input_channels


def eval_fno_field(
    model: nn.Module,
    domain,
    placements: list,
    layers_list: list,
    k_fn,
    Q_lins: list[float],
    T_amb: float,
    N_g: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluar FNO en malla completa y devolver T(x,y) en Kelvin.

    Args:
        model:        FNO entrenado.
        domain:       Dominio 2D.
        placements:   Posiciones de cables.
        layers_list:  Capas de cable.
        k_fn:         Función k(x,y) o escalar.
        Q_lins:       Calor lineal por cable [W/m].
        T_amb:        Temperatura ambiente [K].
        N_g:          Resolución de la malla.
        device:       Dispositivo.

    Returns:
        (X_grid, Y_grid, T_grid):
          - X_grid: ``(N_g, N_g)`` coordenadas x [m]
          - Y_grid: ``(N_g, N_g)`` coordenadas y [m]
          - T_grid: ``(N_g, N_g)`` temperatura [K]
    """
    model.eval()
    with torch.no_grad():
        input_grid, coord_grid = make_input_channels(
            domain, placements, layers_list,
            k_fn, Q_lins, T_amb, N_g, device,
        )
        # FNO espera batch dimension: (1, C_in, N_g, N_g)
        x_in = input_grid.unsqueeze(0)          # (1, 3, N_g, N_g)
        dT_pred = model(x_in).squeeze(0).squeeze(0)  # (N_g, N_g) — delta T

    T_grid = dT_pred + T_amb  # volver a Kelvin absoluto
    X_grid = coord_grid[0]    # (N_g, N_g)
    Y_grid = coord_grid[1]    # (N_g, N_g)
    return X_grid, Y_grid, T_grid


def fno_T_max(
    model: nn.Module,
    domain,
    placements: list,
    layers_list: list,
    k_fn,
    Q_lins: list[float],
    T_amb: float,
    N_g: int,
    device: torch.device,
    n_ring: int = 64,
) -> dict[str, float]:
    """Encontrar temperatura máxima en las superficies de los conductores.

    Muestrea ``n_ring`` puntos en el borde del conductor de cada cable,
    evalúa el FNO interpolando desde la malla y devuelve el máximo global.

    Args:
        model:      FNO entrenado.
        domain:     Dominio 2D.
        placements: Posiciones de cables.
        layers_list: Capas de cable.
        k_fn:       Función de conductividad.
        Q_lins:     Calor lineal [W/m].
        T_amb:      Temperatura ambiente [K].
        N_g:        Resolución de la malla.
        device:     Dispositivo.
        n_ring:     Puntos de muestreo en el anillo conductor.

    Returns:
        Diccionario con ``"T_max_K"``, ``"T_max_C"`` y ``"T_per_cable"``
        (lista con T máx por cable [K]).
    """
    _, _, T_grid = eval_fno_field(
        model, domain, placements, layers_list,
        k_fn, Q_lins, T_amb, N_g, device,
    )

    # Interpolación bilineal: mapear coordenadas físicas a índices de malla
    Nx = N_g
    Ny = N_g
    dx = (domain.xmax - domain.xmin) / (Nx - 1)
    dy = (domain.ymax - domain.ymin) / (Ny - 1)

    T_per_cable: list[float] = []
    for pl, layers in zip(placements, layers_list):
        r_cond = layers[0].r_outer  # radio del conductor
        angles = torch.linspace(0, 2 * math.pi, n_ring, device=device)
        xr = pl.cx + r_cond * torch.cos(angles)
        yr = pl.cy + r_cond * torch.sin(angles)

        # Índices fraccionarios en la malla
        xi = (xr - domain.xmin) / dx
        yi = (yr - domain.ymin) / dy

        # Bilineal: clamp a límites de la malla
        xi0 = xi.long().clamp(0, Nx - 2)
        yi0 = yi.long().clamp(0, Ny - 2)
        xi1 = (xi0 + 1).clamp(0, Nx - 1)
        yi1 = (yi0 + 1).clamp(0, Ny - 1)
        fx = (xi - xi0.float()).clamp(0, 1)
        fy = (yi - yi0.float()).clamp(0, 1)

        # T_grid indexado como [y, x] (row = y, col = x)
        T00 = T_grid[yi0, xi0]
        T10 = T_grid[yi1, xi0]
        T01 = T_grid[yi0, xi1]
        T11 = T_grid[yi1, xi1]
        T_ring = (
            (1 - fy) * (1 - fx) * T00
            + fy * (1 - fx) * T10
            + (1 - fy) * fx * T01
            + fy * fx * T11
        )
        T_per_cable.append(float(T_ring.max().item()))

    T_max_K = max(T_per_cable)
    return {
        "T_max_K":      T_max_K,
        "T_max_C":      T_max_K - 273.15,
        "T_per_cable":  T_per_cable,
    }
