"""Tests for pinn_cables.materials.props -- material property look-ups."""

from __future__ import annotations

import math

import torch

from pinn_cables.io.readers import CableLayer, SoilProperties
from pinn_cables.materials.props import get_Q, get_k, get_rho_c, k_soil_variable


def test_k_soil_variable_shape(device):
    xy = torch.randn(50, 2, device=device)
    k = k_soil_variable(xy, k0=1.0, amp=0.3)
    assert k.shape == (50, 1)


def test_k_soil_variable_range(device):
    xy = torch.randn(500, 2, device=device)
    k0, amp = 1.0, 0.3
    k = k_soil_variable(xy, k0, amp)
    assert k.min() >= k0 * (1.0 - amp) - 1e-6
    assert k.max() <= k0 * (1.0 + amp) + 1e-6


def test_k_soil_variable_at_origin(device):
    xy = torch.zeros(1, 2, device=device)
    k = k_soil_variable(xy, k0=1.0, amp=0.3)
    # sin(0)*cos(0) = 0, so k = k0
    assert abs(k.item() - 1.0) < 1e-6


def test_get_k_layer():
    layer = CableLayer("cond", 0.0, 0.01, 400.0, 3e6, 30.0)
    soil = SoilProperties(1.0, 2e6, False, 0.3)
    xy = torch.randn(5, 2)
    assert get_k(layer, xy, soil) == 400.0


def test_get_k_soil_constant():
    soil = SoilProperties(1.5, 2e6, False, 0.3)
    xy = torch.randn(5, 2)
    assert get_k(None, xy, soil) == 1.5


def test_get_k_soil_variable():
    soil = SoilProperties(1.0, 2e6, True, 0.3)
    xy = torch.randn(10, 2)
    k = get_k(None, xy, soil)
    assert isinstance(k, torch.Tensor)
    assert k.shape == (10, 1)


def test_get_rho_c():
    layer = CableLayer("cond", 0.0, 0.01, 400.0, 3.45e6, 30.0)
    soil = SoilProperties(1.0, 2e6, False, 0.3)
    assert get_rho_c(layer, soil) == 3.45e6
    assert get_rho_c(None, soil) == 2e6


def test_get_Q_conductor():
    layer = CableLayer("cond", 0.0, 0.01, 400.0, 3e6, 30.0)
    assert get_Q(layer, Q_scale=2.0) == 60.0


def test_get_Q_soil():
    assert get_Q(None) == 0.0
    assert get_Q(None, Q_scale=5.0) == 0.0
