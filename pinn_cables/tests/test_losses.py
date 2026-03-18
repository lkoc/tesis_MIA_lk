"""Tests for pinn_cables.pinn.losses -- loss functions."""

from __future__ import annotations

import torch

from pinn_cables.pinn.losses import (
    dirichlet_loss,
    initial_condition_loss,
    interface_T_loss,
    interface_flux_loss,
    mse,
    neumann_loss,
    robin_loss,
    weighted_total_loss,
)


def test_mse_zero():
    x = torch.zeros(10, 1)
    assert mse(x).item() == 0.0


def test_mse_known_value():
    x = torch.ones(4, 1) * 2.0
    assert abs(mse(x).item() - 4.0) < 1e-6


def test_dirichlet_loss_zero():
    t = torch.randn(10, 1)
    assert dirichlet_loss(t, t).item() < 1e-12


def test_dirichlet_loss_nonzero():
    pred = torch.ones(5, 1)
    target = torch.zeros(5, 1)
    assert abs(dirichlet_loss(pred, target).item() - 1.0) < 1e-6


def test_neumann_loss_zero_flux():
    dTdn = torch.zeros(10, 1)
    assert neumann_loss(dTdn, target=0.0).item() < 1e-12


def test_robin_loss_zero_residual():
    residual = torch.zeros(8, 1)
    assert robin_loss(residual).item() < 1e-12


def test_interface_T_loss_symmetry():
    a = torch.randn(10, 1)
    b = torch.randn(10, 1)
    assert abs(interface_T_loss(a, b).item() - interface_T_loss(b, a).item()) < 1e-6


def test_interface_flux_loss():
    f1 = torch.ones(10, 1) * 5.0
    f2 = torch.ones(10, 1) * 3.0
    assert abs(interface_flux_loss(f1, f2).item() - 4.0) < 1e-6


def test_initial_condition_loss():
    pred = torch.ones(10, 1) * 300.0
    assert abs(initial_condition_loss(pred, 293.15).item() - (300.0 - 293.15) ** 2) < 1e-3


def test_weighted_total_loss():
    losses = {
        "pde": torch.tensor(1.0),
        "bc": torch.tensor(2.0),
    }
    weights = {"pde": 1.0, "bc": 10.0}
    total = weighted_total_loss(losses, weights)
    assert abs(total.item() - 21.0) < 1e-6


def test_weighted_total_loss_ignores_zero_weight():
    losses = {
        "pde": torch.tensor(1.0),
        "bc": torch.tensor(2.0),
    }
    weights = {"pde": 1.0, "bc": 0.0}
    total = weighted_total_loss(losses, weights)
    assert abs(total.item() - 1.0) < 1e-6
