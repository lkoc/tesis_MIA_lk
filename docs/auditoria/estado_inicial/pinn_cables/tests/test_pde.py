"""Tests for pinn_cables.pinn.pde -- PDE operators via autograd."""

from __future__ import annotations

import math

import torch

from pinn_cables.pinn.pde import (
    gradients,
    laplace_variable_k,
    neumann_residual,
    pde_residual_steady,
    pde_residual_transient,
)


def test_gradients_linear(device):
    """dT/dx = 2, dT/dy = 3 for T = 2x + 3y."""
    xy = torch.randn(30, 2, device=device, requires_grad=True)
    T = 2.0 * xy[:, 0:1] + 3.0 * xy[:, 1:2]
    g = gradients(T, xy)
    assert torch.allclose(g[:, 0], torch.tensor(2.0, device=device), atol=1e-5)
    assert torch.allclose(g[:, 1], torch.tensor(3.0, device=device), atol=1e-5)


def test_laplace_constant_k_harmonic(device):
    """Laplacian of sin(pi*x)*sin(pi*y) with k=1 is -2*pi^2 * T."""
    xy = torch.rand(50, 2, device=device, requires_grad=True) * 0.8 + 0.1
    T = torch.sin(math.pi * xy[:, 0:1]) * torch.sin(math.pi * xy[:, 1:2])
    lap = laplace_variable_k(T, xy, k=1.0)
    expected = -2.0 * math.pi ** 2 * T
    assert torch.allclose(lap, expected, atol=1e-3)


def test_laplace_variable_k_mms(device):
    """MMS: T = x^2 + y^2, k = 1 + 0.5x.

    div(k grad T) = d/dx(k*2x) + d/dy(k*2y)
                   = d/dx((1+0.5x)*2x) + d/dy((1+0.5x)*2y)
                   = d/dx(2x + x^2) + 2*(1+0.5x)
                   = (2 + 2x) + (2 + x)
                   = 4 + 3x
    So Q = -(4 + 3x) to satisfy div(k grad T) + Q = 0.
    """
    xy = torch.rand(100, 2, device=device, requires_grad=True)
    x = xy[:, 0:1]
    T = xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2
    k = 1.0 + 0.5 * x
    lap = laplace_variable_k(T, xy, k)
    expected = 4.0 + 3.0 * x
    assert torch.allclose(lap, expected, atol=5e-3)


def test_steady_residual_zero(device):
    """For the MMS pair above, residual should be ~0."""
    xy = torch.rand(100, 2, device=device, requires_grad=True)
    x = xy[:, 0:1]
    T = xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2
    k = 1.0 + 0.5 * x
    Q = -(4.0 + 3.0 * x)
    res = pde_residual_steady(T, xy, k, Q)
    assert res.abs().max() < 5e-3


def test_transient_residual_zero(device):
    """T(x,y,t) = exp(-t)*sin(pi*x)*sin(pi*y), k=1, rho_c=1.

    dT/dt = -T
    Laplacian = -2*pi^2 * T
    So residual = -2*pi^2*T + Q - 1*(-T) = 0  =>  Q = (2*pi^2 - 1)*T
    """
    xyt = torch.rand(80, 3, device=device, requires_grad=True)
    x_ = xyt[:, 0:1]
    y_ = xyt[:, 1:2]
    t_ = xyt[:, 2:3]
    T = torch.exp(-t_) * torch.sin(math.pi * x_) * torch.sin(math.pi * y_)
    Q = (2.0 * math.pi ** 2 - 1.0) * T
    res = pde_residual_transient(T, xyt, k=1.0, rho_c=1.0, Q=Q)
    assert res.abs().max() < 5e-2


def test_neumann_residual(device):
    """dT/dn = 0 for T = const."""
    xy = torch.randn(20, 2, device=device, requires_grad=True)
    T = torch.ones(20, 1, device=device, requires_grad=True)
    # Need T to depend on xy for autograd
    T = T + 0.0 * xy[:, 0:1]
    normal = torch.tensor([[0.0, 1.0]], device=device).expand(20, 2)
    res = neumann_residual(T, xy, normal, target_flux=0.0)
    assert res.abs().max() < 1e-5
