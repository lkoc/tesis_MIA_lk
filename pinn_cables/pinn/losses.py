"""Loss functions for the PINN solver.

Every loss function returns a scalar tensor so it can be back-propagated.
The :func:`weighted_total_loss` helper aggregates named components with
user-configured weights from the solver YAML.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Elementary losses
# ---------------------------------------------------------------------------

def mse(x: torch.Tensor) -> torch.Tensor:
    """Mean squared error of a residual tensor.

    Args:
        x: Residual tensor of any shape.

    Returns:
        Scalar mean(x^2).
    """
    return torch.mean(x * x)


def dirichlet_loss(T_pred: torch.Tensor, T_target: torch.Tensor) -> torch.Tensor:
    """Dirichlet BC loss: MSE between predicted and prescribed temperature.

    Args:
        T_pred:   Predicted temperature ``(N, 1)``.
        T_target: Target temperature ``(N, 1)`` or scalar broadcast.

    Returns:
        Scalar loss.
    """
    return mse(T_pred - T_target)


def neumann_loss(
    dTdn_pred: torch.Tensor, target: float = 0.0,
) -> torch.Tensor:
    """Neumann BC loss: MSE of the normal gradient residual.

    Args:
        dTdn_pred: Predicted normal gradient ``(N, 1)``.
        target:    Prescribed value (default 0 = insulated).

    Returns:
        Scalar loss.
    """
    return mse(dTdn_pred - target)


def robin_loss(residual: torch.Tensor) -> torch.Tensor:
    """Robin BC loss from a pre-computed residual (see :func:`pde.robin_residual`).

    Args:
        residual: Robin residual ``(N, 1)``.

    Returns:
        Scalar loss.
    """
    return mse(residual)


# ---------------------------------------------------------------------------
# Interface losses
# ---------------------------------------------------------------------------

def interface_T_loss(
    T_inner: torch.Tensor, T_outer: torch.Tensor,
) -> torch.Tensor:
    """Temperature-continuity loss at an interface.

    Args:
        T_inner: Temperature from the inner region ``(N, 1)``.
        T_outer: Temperature from the outer region ``(N, 1)``.

    Returns:
        Scalar loss.
    """
    return mse(T_inner - T_outer)


def interface_flux_loss(
    flux_inner: torch.Tensor, flux_outer: torch.Tensor,
) -> torch.Tensor:
    """Heat-flux-continuity loss at an interface.

    Args:
        flux_inner: :math:`-k_i \\nabla T_i \\cdot n` from the inner side.
        flux_outer: :math:`-k_j \\nabla T_j \\cdot n` from the outer side.

    Returns:
        Scalar loss.
    """
    return mse(flux_inner - flux_outer)


# ---------------------------------------------------------------------------
# Initial condition (transient)
# ---------------------------------------------------------------------------

def initial_condition_loss(
    T_pred: torch.Tensor, T_init: float,
) -> torch.Tensor:
    """IC loss for transient problems.

    Args:
        T_pred: Predicted temperature at t = 0, shape ``(N, 1)``.
        T_init: Uniform initial temperature [K].

    Returns:
        Scalar loss.
    """
    return mse(T_pred - T_init)


# ---------------------------------------------------------------------------
# Weighted aggregation
# ---------------------------------------------------------------------------

def weighted_total_loss(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    """Combine named loss components with configurable weights.

    Only components whose name appears in *weights* with a weight > 0 are
    included.  Extra names in *weights* that are not in *losses* are ignored.

    Args:
        losses:  Dict of named scalar loss tensors.
        weights: Dict of weights keyed by the same names.

    Returns:
        Weighted sum (scalar tensor).
    """
    total = torch.tensor(0.0, device=next(iter(losses.values())).device)
    for name, loss_val in losses.items():
        w = weights.get(name, 0.0)
        if w > 0.0:
            total = total + w * loss_val
    return total
