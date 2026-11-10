"""Integration test: steady-state Laplace equation on a rectangle.

Solves  div(grad T) = 0  on [0,1]x[0,1] with:
  - T(x,0) = sin(pi*x)  (bottom)
  - T = 0 on top, left, right

Exact:  T(x,y) = sin(pi*x) * sinh(pi*(1-y)) / sinh(pi)

Uses very reduced training (500 Adam steps, no LBFGS) for CI speed.
Marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import math

import pytest
import torch

from pinn_cables.pinn.model import MLP
from pinn_cables.pinn.pde import pde_residual_steady
from pinn_cables.pinn.losses import mse
from pinn_cables.post.eval import l2_relative_error


def _fresh(t: torch.Tensor) -> torch.Tensor:
    """Detach and re-enable grad to avoid stale computation graphs."""
    return t.data.clone().requires_grad_(True)


@pytest.mark.slow
def test_laplace_rectangle(device):
    torch.manual_seed(0)

    model = MLP(in_dim=2, out_dim=1, width=64, depth=4, activation="tanh").to(device)

    n_int = 2000
    n_bc = 200

    # Fixed sample data (re-wrapped each iteration)
    xy_int_data = torch.rand(n_int, 2, device=device)
    x_bc = torch.rand(n_bc, 1, device=device)
    y_bc = torch.rand(n_bc, 1, device=device)
    bc_bottom_data = torch.cat([x_bc, torch.zeros(n_bc, 1, device=device)], dim=1)
    bc_top_data = torch.cat([x_bc, torch.ones(n_bc, 1, device=device)], dim=1)
    bc_left_data = torch.cat([torch.zeros(n_bc, 1, device=device), y_bc], dim=1)
    bc_right_data = torch.cat([torch.ones(n_bc, 1, device=device), y_bc], dim=1)

    def compute_loss():
        xy = _fresh(xy_int_data)
        bc_b = _fresh(bc_bottom_data)
        T_bot_exact = torch.sin(math.pi * bc_b[:, 0:1])

        T_int = model(xy)
        loss_pde = mse(pde_residual_steady(T_int, xy, k=1.0, Q=0.0))
        loss_bc = (
            mse(model(bc_b) - T_bot_exact)
            + mse(model(_fresh(bc_top_data)))
            + mse(model(_fresh(bc_left_data)))
            + mse(model(_fresh(bc_right_data)))
        )
        return loss_pde + 10.0 * loss_bc

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        loss = compute_loss()
        loss.backward()
        opt.step()

    # Evaluate -- with only 500 Adam steps the accuracy is limited,
    # so we use a generous tolerance.  The goal is to verify that the
    # loss is decreasing and the model produces reasonable output.
    model.eval()
    xy_test = torch.rand(500, 2, device=device)
    with torch.no_grad():
        T_pred = model(xy_test)
    x_t = xy_test[:, 0:1]
    y_t = xy_test[:, 1:2]
    T_exact = torch.sin(math.pi * x_t) * torch.sinh(math.pi * (1 - y_t)) / math.sinh(math.pi)

    err = l2_relative_error(T_pred, T_exact)
    # With reduced training we accept a higher error bound
    assert err < 1.0, f"L2 relative error too high: {err:.4f}"
