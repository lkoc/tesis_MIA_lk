"""Tests for pinn_cables.materials.props -- material property look-ups."""

from __future__ import annotations

import math

import pytest
import torch

from pinn_cables.io.readers import CableLayer, SoilProperties
from pinn_cables.materials.props import (
    available_sections,
    generate_cable_layers,
    get_alpha_R,
    get_cable_spec,
    get_kim2024_cable_layers,
    get_Q,
    get_R_dc_20,
    get_k,
    get_rho_c,
    k_soil_variable,
)


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


# ---------------------------------------------------------------------------
# Cable catalog helpers
# ---------------------------------------------------------------------------

class TestAvailableSections:
    def test_returns_list_of_ints(self):
        secs = available_sections()
        assert isinstance(secs, list)
        assert all(isinstance(s, int) for s in secs)

    def test_contains_standard_sections(self):
        secs = available_sections()
        for s in (95, 150, 240, 400, 600, 1200):
            assert s in secs

    def test_sorted(self):
        secs = available_sections()
        assert secs == sorted(secs)


class TestGetCableSpec:
    def test_known_section(self):
        spec = get_cable_spec(240)
        assert spec.section_mm2 == 240
        assert spec.r_conductor > 0
        assert spec.r_sheath_outer > spec.r_xlpe_outer

    def test_unknown_section_raises(self):
        with pytest.raises(ValueError, match="not in catalog"):
            get_cable_spec(999)


class TestGetRDc20:
    def test_cu_less_than_al(self):
        R_cu = get_R_dc_20(240, "cu")
        R_al = get_R_dc_20(240, "al")
        assert R_cu < R_al

    def test_larger_section_lower_R(self):
        R_small = get_R_dc_20(95, "cu")
        R_large = get_R_dc_20(600, "cu")
        assert R_small > R_large

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError, match="Unknown conductor"):
            get_R_dc_20(240, "gold")

    def test_case_insensitive(self):
        assert get_R_dc_20(240, "CU") == get_R_dc_20(240, "cu")
        assert get_R_dc_20(240, "Al") == get_R_dc_20(240, "al")


class TestGetAlphaR:
    def test_cu_alpha(self):
        assert get_alpha_R("cu") == pytest.approx(0.00393)

    def test_al_alpha(self):
        assert get_alpha_R("al") == pytest.approx(0.00403)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_alpha_R("silver")


# ---------------------------------------------------------------------------
# generate_cable_layers
# ---------------------------------------------------------------------------

class TestGenerateCableLayers:
    def test_returns_four_layers(self):
        layers = generate_cable_layers(240, "cu", 400.0)
        assert len(layers) == 4

    def test_layer_names(self):
        layers = generate_cable_layers(240, "cu", 400.0)
        names = [la.name for la in layers]
        assert "conductor" in names
        assert "xlpe" in names

    def test_conductor_has_heat_source(self):
        layers = generate_cable_layers(240, "cu", 500.0)
        cond = layers[0]
        assert cond.Q > 0.0

    def test_non_conductor_zero_Q(self):
        layers = generate_cable_layers(240, "cu", 500.0)
        for la in layers[1:]:
            assert la.Q == 0.0

    def test_zero_current_zero_Q(self):
        layers = generate_cable_layers(240, "cu", 0.0)
        assert layers[0].Q == 0.0

    def test_radii_ordered(self):
        layers = generate_cable_layers(400, "al", 300.0)
        for i in range(len(layers) - 1):
            assert layers[i].r_outer == pytest.approx(layers[i + 1].r_inner)

    def test_aluminium_higher_R(self):
        """Al conductor dissipates more than Cu at the same current."""
        layers_cu = generate_cable_layers(240, "cu", 400.0)
        layers_al = generate_cable_layers(240, "al", 400.0)
        assert layers_al[0].Q > layers_cu[0].Q

    def test_unknown_section_raises(self):
        with pytest.raises(ValueError):
            generate_cable_layers(999, "cu", 400.0)

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError):
            generate_cable_layers(240, "brass", 400.0)


# ---------------------------------------------------------------------------
# get_kim2024_cable_layers
# ---------------------------------------------------------------------------

class TestGetKim2024CableLayers:
    def test_returns_nine_layers(self):
        layers = get_kim2024_cable_layers(Q_lin=50.0)
        assert len(layers) == 9

    def test_layer_names_include_clsm_and_casing(self):
        layers = get_kim2024_cable_layers(Q_lin=50.0)
        names = [la.name for la in layers]
        assert "clsm" in names
        assert "pe_casing" in names
        assert "conductor" in names

    def test_clsm_k_value(self):
        layers = get_kim2024_cable_layers(Q_lin=50.0)
        clsm = next(la for la in layers if la.name == "clsm")
        assert clsm.k == pytest.approx(2.150)

    def test_pe_casing_outer_radius(self):
        layers = get_kim2024_cable_layers(Q_lin=50.0)
        casing = next(la for la in layers if la.name == "pe_casing")
        assert casing.r_outer == pytest.approx(0.110)

    def test_conductor_heat_source_proportional(self):
        q1 = get_kim2024_cable_layers(Q_lin=100.0)[0].Q
        q2 = get_kim2024_cable_layers(Q_lin=200.0)[0].Q
        assert q2 == pytest.approx(2.0 * q1)

    def test_zero_Q_lin(self):
        layers = get_kim2024_cable_layers(Q_lin=0.0)
        assert layers[0].Q == 0.0

    def test_radii_contiguous(self):
        layers = get_kim2024_cable_layers(Q_lin=50.0)
        for i in range(len(layers) - 1):
            assert layers[i].r_outer == pytest.approx(layers[i + 1].r_inner)

    def test_non_conductor_zero_Q(self):
        layers = get_kim2024_cable_layers(Q_lin=60.0)
        for la in layers[1:]:
            assert la.Q == 0.0


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
