"""PDE operators for 2-D heat conduction via PyTorch autograd.

All functions operate on tensors with ``requires_grad=True`` and build a
computation graph suitable for second-order differentiation
(``create_graph=True``).

Key operators
-------------
- :func:`gradients` — first-order partial derivatives.
- :func:`laplace_variable_k` — divergence form :math:`\\nabla\\cdot(k \\nabla T)`.
- :func:`pde_residual_steady` — steady-state residual.
- :func:`pde_residual_transient` — transient residual.
- :func:`neumann_residual` — Neumann BC residual.
- :func:`robin_residual` — Robin (convection) BC residual.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Autograd helpers
# ---------------------------------------------------------------------------

def gradients(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute :math:`\\partial u / \\partial x_i` for every column of *x*.

    Args:
        u: Scalar field ``(N, 1)``.
        x: Independent variables ``(N, D)`` with ``requires_grad=True``.

    Returns:
        Gradient tensor ``(N, D)``.
    """
    assert x.requires_grad, (
        "Input tensor must have requires_grad=True for autograd differentiation"
    )
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]


# ---------------------------------------------------------------------------
# Divergence-form Laplacian with variable k
# ---------------------------------------------------------------------------

def _ensure_tensor(
    v: torch.Tensor | float,
    ref: torch.Tensor,
) -> torch.Tensor:
    """Promote a scalar / 0-d tensor to shape ``(N, 1)`` matching *ref*."""
    if not torch.is_tensor(v):
        v = torch.tensor(v, device=ref.device, dtype=ref.dtype)
    if v.ndim == 0:
        v = v.view(1, 1)
    if v.shape[0] == 1 and ref.shape[0] > 1:
        v = v.expand(ref.shape[0], 1)
    return v


def laplace_variable_k(
    T: torch.Tensor,
    inputs: torch.Tensor,
    k: torch.Tensor | float,
    spatial_cols: tuple[int, int] = (0, 1),
) -> torch.Tensor:
    """Compute :math:`\\nabla\\cdot(k\\,\\nabla T)` in 2-D.

    The derivatives are taken *with respect to the full* ``inputs`` tensor
    (which may include a time column for transient problems) but only the
    spatial columns specified by *spatial_cols* are used.

    Args:
        T:            Temperature field ``(N, 1)``.
        inputs:       Independent-variable tensor ``(N, D)`` (``D >= 2``);
                      must have ``requires_grad=True``.
        k:            Thermal conductivity — scalar or ``(N, 1)``.
        spatial_cols: Indices of the two spatial columns in *inputs*.

    Returns:
        ``(N, 1)`` tensor equal to
        :math:`\\partial_x(k\\,\\partial_x T) + \\partial_y(k\\,\\partial_y T)`.
    """
    c0, c1 = spatial_cols
    k_ = _ensure_tensor(k, inputs)

    gT = gradients(T, inputs)  # (N, D)
    dTdx = gT[:, c0:c0+1]
    dTdy = gT[:, c1:c1+1]

    # Flux components: k * dT/dx , k * dT/dy
    flux_x = k_ * dTdx
    flux_y = k_ * dTdy

    # div(flux) = d(flux_x)/dx + d(flux_y)/dy
    g_flux_x = gradients(flux_x, inputs)
    g_flux_y = gradients(flux_y, inputs)

    return g_flux_x[:, c0:c0+1] + g_flux_y[:, c1:c1+1]


# ---------------------------------------------------------------------------
# PDE residuals
# ---------------------------------------------------------------------------

def pde_residual_steady(
    T: torch.Tensor,
    inputs: torch.Tensor,
    k: torch.Tensor | float,
    Q: torch.Tensor | float,
) -> torch.Tensor:
    """Steady-state residual :math:`\\nabla\\cdot(k\\nabla T) + Q = 0`.

    Args:
        T:      Predicted temperature ``(N, 1)``.
        inputs: Coordinates ``(N, 2)`` with ``requires_grad=True``.
        k:      Thermal conductivity.
        Q:      Volumetric heat source [W/m^3].

    Returns:
        Residual ``(N, 1)`` — should be driven to zero.
    """
    Q_ = _ensure_tensor(Q, inputs)
    return laplace_variable_k(T, inputs, k) + Q_


def pde_residual_transient(
    T: torch.Tensor,
    inputs: torch.Tensor,
    k: torch.Tensor | float,
    rho_c: torch.Tensor | float,
    Q: torch.Tensor | float,
) -> torch.Tensor:
    """Transient residual: :math:`\\nabla\\cdot(k\\nabla T) + Q - \\rho c\\,\\partial T/\\partial t = 0`.

    Args:
        T:      Predicted temperature ``(N, 1)``.
        inputs: ``(N, 3)`` tensor ``[x, y, t]`` with ``requires_grad=True``.
        k:      Thermal conductivity.
        rho_c:  Volumetric heat capacity [J/(m^3 K)].
        Q:      Volumetric heat source [W/m^3].

    Returns:
        Residual ``(N, 1)``.
    """
    Q_ = _ensure_tensor(Q, inputs)
    rho_c_ = _ensure_tensor(rho_c, inputs)

    # Spatial part: div(k grad T) w.r.t. columns 0,1 of the full input
    div_term = laplace_variable_k(T, inputs, k, spatial_cols=(0, 1))

    # Temporal derivative
    gT = gradients(T, inputs)
    dTdt = gT[:, 2:3]

    return div_term + Q_ - rho_c_ * dTdt


# ---------------------------------------------------------------------------
# Boundary-condition residuals
# ---------------------------------------------------------------------------

def neumann_residual(
    T: torch.Tensor,
    inputs: torch.Tensor,
    normal: torch.Tensor,
    target_flux: float = 0.0,
) -> torch.Tensor:
    """Neumann BC residual: :math:`\\nabla T \\cdot \\mathbf{n} - q_n = 0`.

    Args:
        T:           Predicted temperature ``(N, 1)``.
        inputs:      Coordinates ``(N, 2)`` with ``requires_grad=True``.
        normal:      Outward unit normal ``(N, 2)`` or ``(1, 2)``.
        target_flux: Prescribed normal heat flux (0 for insulated).

    Returns:
        Residual ``(N, 1)``.
    """
    gT = gradients(T, inputs)      # (N, 2)
    dTdn = (gT * normal).sum(dim=1, keepdim=True)  # dot product
    return dTdn - target_flux


def robin_residual(
    T: torch.Tensor,
    inputs: torch.Tensor,
    normal: torch.Tensor,
    k_val: float,
    h: float,
    T_inf: float,
) -> torch.Tensor:
    """Robin BC residual: :math:`-k\\,\\nabla T \\cdot \\mathbf{n} = h(T - T_\\infty)`.

    Returned as :math:`-k\\,\\nabla T \\cdot \\mathbf{n} - h(T - T_\\infty)`.

    Args:
        T:      Predicted temperature ``(N, 1)``.
        inputs: Coordinates ``(N, 2)`` with ``requires_grad=True``.
        normal: Outward unit normal.
        k_val:  Thermal conductivity at the boundary.
        h:      Convection heat-transfer coefficient [W/(m^2 K)].
        T_inf:  Far-field temperature [K].

    Returns:
        Residual ``(N, 1)``.
    """
    gT = gradients(T, inputs)
    dTdn = (gT * normal).sum(dim=1, keepdim=True)
    return -k_val * dTdn - h * (T - T_inf)
