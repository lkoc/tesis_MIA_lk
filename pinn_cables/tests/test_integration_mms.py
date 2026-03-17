"""Integration test: MMS benchmark with constant and variable k.

Uses manufactured solutions to verify that the PINN solver can learn
exact PDE solutions when the source term is chosen to satisfy the
equation identically.

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
def test_mms_constant_k(device):
    """T*(x,y) = sin(pi*x)*sin(pi*y), k=1.

    Q = 2*pi^2*sin(pi*x)*sin(pi*y)  (so that div(k grad T) + Q = 0).
    Domain [0,1]x[0,1], Dirichlet T=0 on all boundaries.
    """
    torch.manual_seed(42)
    model = MLP(in_dim=2, out_dim=1, width=64, depth=4).to(device)

    n_int = 2000
    n_bc = 200
    xy_int = torch.rand(n_int, 2, device=device, requires_grad=True)
    x_bc = torch.rand(n_bc, 1, device=device)

    bcs = [
        torch.cat([x_bc, torch.zeros(n_bc, 1, device=device)], dim=1),
        torch.cat([x_bc, torch.ones(n_bc, 1, device=device)], dim=1),
        torch.cat([torch.zeros(n_bc, 1, device=device), torch.rand(n_bc, 1, device=device)], dim=1),
        torch.cat([torch.ones(n_bc, 1, device=device), torch.rand(n_bc, 1, device=device)], dim=1),
    ]
    bcs = [b.requires_grad_(True) for b in bcs]

    def loss_fn():
        T_ = model(xy_int)
        x_ = xy_int[:, 0:1]
        y_ = xy_int[:, 1:2]
        Q_ = 2.0 * math.pi ** 2 * torch.sin(math.pi * x_) * torch.sin(math.pi * y_)
        loss_p = mse(pde_residual_steady(T_, xy_int, k=1.0, Q=Q_))
        loss_b = sum(mse(model(b)) for b in bcs)
        return loss_p + 10.0 * loss_b

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3000):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=20, history_size=30,
                               line_search_fn="strong_wolfe")
    for _ in range(50):
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            l = loss_fn()
            l.backward()
            return l
        lbfgs.step(closure)

    # Evaluate
    model.eval()
    xy_test = torch.rand(500, 2, device=device)
    with torch.no_grad():
        T_pred = model(xy_test)
    T_exact = (
        torch.sin(math.pi * xy_test[:, 0:1])
        * torch.sin(math.pi * xy_test[:, 1:2])
    )
    err = l2_relative_error(T_pred, T_exact)
    assert err < 0.10, f"MMS constant-k L2 error: {err:.4f}"


@pytest.mark.slow
def test_mms_variable_k(device):
    """T*(x,y) = x^2 + y^2, k(x,y) = 1 + 0.5x.

    Q = -(3x + 4).  Domain [0,1]x[0,1].
    Dirichlet BCs from T*.
    """
    torch.manual_seed(42)
    model = MLP(in_dim=2, out_dim=1, width=64, depth=4).to(device)

    n_int = 2500
    n_bc = 250
    xy_int = torch.rand(n_int, 2, device=device, requires_grad=True)

    x_bc = torch.rand(n_bc, 1, device=device)
    y_bc = torch.rand(n_bc, 1, device=device)
    bcs_pts = [
        torch.cat([x_bc, torch.zeros(n_bc, 1, device=device)], dim=1),
        torch.cat([x_bc, torch.ones(n_bc, 1, device=device)], dim=1),
        torch.cat([torch.zeros(n_bc, 1, device=device), y_bc], dim=1),
        torch.cat([torch.ones(n_bc, 1, device=device), y_bc], dim=1),
    ]
    bcs_pts = [b.requires_grad_(True) for b in bcs_pts]
    bcs_vals = [b[:, 0:1] ** 2 + b[:, 1:2] ** 2 for b in bcs_pts]

    def loss_fn():
        T_ = model(xy_int)
        x_ = xy_int[:, 0:1]
        k_ = 1.0 + 0.5 * x_
        Q_ = -(3.0 * x_ + 4.0)
        loss_p = mse(pde_residual_steady(T_, xy_int, k=k_, Q=Q_))
        loss_b = sum(mse(model(b) - v) for b, v in zip(bcs_pts, bcs_vals))
        return loss_p + 10.0 * loss_b

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(4000):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=20, history_size=30,
                               line_search_fn="strong_wolfe")
    for _ in range(75):
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            l = loss_fn()
            l.backward()
            return l
        lbfgs.step(closure)

    model.eval()
    xy_test = torch.rand(500, 2, device=device)
    with torch.no_grad():
        T_pred = model(xy_test)
    T_exact = xy_test[:, 0:1] ** 2 + xy_test[:, 1:2] ** 2

    err = l2_relative_error(T_pred, T_exact)
    assert err < 0.10, f"MMS variable-k L2 error: {err:.4f}"
