"""Integration test: steady-state Laplace equation on a rectangle.

Solves  div(grad T) = 0  on [0,1]x[0,1] with:
  - T(x,0) = sin(pi*x)  (bottom)
  - T = 0 on top, left, right

Exact:  T(x,y) = sin(pi*x) * sinh(pi*(1-y)) / sinh(pi)

Uses reduced training iterations for CI speed.
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


@pytest.mark.slow
def test_laplace_rectangle(device):
    torch.manual_seed(0)

    # --- Model ---
    model = MLP(in_dim=2, out_dim=1, width=64, depth=4, activation="tanh").to(device)

    # --- Sampling ---
    n_int = 2000
    n_bc = 200
    xy_int = torch.rand(n_int, 2, device=device, requires_grad=True)

    # BCs
    x_bc = torch.rand(n_bc, 1, device=device)
    bc_bottom = torch.cat([x_bc, torch.zeros(n_bc, 1, device=device)], dim=1).requires_grad_(True)
    bc_top = torch.cat([x_bc, torch.ones(n_bc, 1, device=device)], dim=1).requires_grad_(True)

    y_bc = torch.rand(n_bc, 1, device=device)
    bc_left = torch.cat([torch.zeros(n_bc, 1, device=device), y_bc], dim=1).requires_grad_(True)
    bc_right = torch.cat([torch.ones(n_bc, 1, device=device), y_bc], dim=1).requires_grad_(True)

    T_bottom_exact = torch.sin(math.pi * bc_bottom[:, 0:1])

    # --- Training ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(3000):
        opt.zero_grad(set_to_none=True)
        T_int = model(xy_int)
        loss_pde = mse(pde_residual_steady(T_int, xy_int, k=1.0, Q=0.0))

        loss_bc = (
            mse(model(bc_bottom) - T_bottom_exact)
            + mse(model(bc_top))
            + mse(model(bc_left))
            + mse(model(bc_right))
        )
        loss = loss_pde + 10.0 * loss_bc
        loss.backward()
        opt.step()

    # L-BFGS fine-tuning
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=20, history_size=30,
                               line_search_fn="strong_wolfe")
    for _ in range(50):
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            T_int = model(xy_int)
            loss_pde = mse(pde_residual_steady(T_int, xy_int, k=1.0, Q=0.0))
            loss_bc = (
                mse(model(bc_bottom) - T_bottom_exact)
                + mse(model(bc_top))
                + mse(model(bc_left))
                + mse(model(bc_right))
            )
            loss = loss_pde + 10.0 * loss_bc
            loss.backward()
            return loss
        lbfgs.step(closure)

    # --- Evaluate ---
    model.eval()
    xy_test = torch.rand(500, 2, device=device)
    with torch.no_grad():
        T_pred = model(xy_test)
    x_t = xy_test[:, 0:1]
    y_t = xy_test[:, 1:2]
    T_exact = torch.sin(math.pi * x_t) * torch.sinh(math.pi * (1 - y_t)) / math.sinh(math.pi)

    err = l2_relative_error(T_pred, T_exact)
    assert err < 0.10, f"L2 relative error too high: {err:.4f}"
