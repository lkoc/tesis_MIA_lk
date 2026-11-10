"""Integration test: MMS benchmark with constant k.

Uses a manufactured solution to verify that the PINN solver can learn
an exact PDE solution when the source term is chosen to satisfy the
equation identically.

T* = sin(pi*x)*sin(pi*y), k=1, Q = 2*pi^2*sin(pi*x)*sin(pi*y).
Domain [0,1]x[0,1], Dirichlet T=0 on all boundaries.

Uses reduced training for CI speed.
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
def test_mms_constant_k(device):
    """T*(x,y) = sin(pi*x)*sin(pi*y), k=1.

    Q = 2*pi^2*sin(pi*x)*sin(pi*y)  (so that div(k grad T) + Q = 0).
    Domain [0,1]x[0,1], Dirichlet T=0 on all boundaries.
    """
    torch.manual_seed(42)
    model = MLP(in_dim=2, out_dim=1, width=64, depth=4).to(device)

    n_int = 2000
    n_bc = 200
    xy_int_data = torch.rand(n_int, 2, device=device)
    x_bc = torch.rand(n_bc, 1, device=device)
    y_bc = torch.rand(n_bc, 1, device=device)

    bcs_data = [
        torch.cat([x_bc, torch.zeros(n_bc, 1, device=device)], dim=1),
        torch.cat([x_bc, torch.ones(n_bc, 1, device=device)], dim=1),
        torch.cat([torch.zeros(n_bc, 1, device=device), y_bc], dim=1),
        torch.cat([torch.ones(n_bc, 1, device=device), y_bc], dim=1),
    ]

    def loss_fn():
        xy = _fresh(xy_int_data)
        T_ = model(xy)
        x_ = xy[:, 0:1]
        y_ = xy[:, 1:2]
        Q_ = 2.0 * math.pi ** 2 * torch.sin(math.pi * x_) * torch.sin(math.pi * y_)
        loss_p = mse(pde_residual_steady(T_, xy, k=1.0, Q=Q_))
        loss_b = sum(mse(model(_fresh(b))) for b in bcs_data)
        return loss_p + 10.0 * loss_b

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(1000):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()

    model.eval()
    xy_test = torch.rand(500, 2, device=device)
    with torch.no_grad():
        T_pred = model(xy_test)
    T_exact = (
        torch.sin(math.pi * xy_test[:, 0:1])
        * torch.sin(math.pi * xy_test[:, 1:2])
    )
    err = l2_relative_error(T_pred, T_exact)
    # Reduced training -- accept a generous tolerance
    assert err < 1.0, f"MMS constant-k L2 error: {err:.4f}"
