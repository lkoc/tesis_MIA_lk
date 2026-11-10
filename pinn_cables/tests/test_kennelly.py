"""Tests for pinn_cables.physics.kennelly -- Kennelly superposition + IEC 60287 estimate."""

from __future__ import annotations

import math

import pytest
import torch

from pinn_cables.io.readers import CableLayer, CablePlacement, Domain2D
from pinn_cables.physics.kennelly import iec60287_estimate, multilayer_T_multi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_cable_layers():
    """Two-layer cable: conductor + insulation."""
    return [
        CableLayer("conductor", 0.0, 0.0125, 400.0, 3.45e6, 30.0),
        CableLayer("xlpe", 0.0125, 0.028, 0.286, 2.4e6, 0.0),
    ]


@pytest.fixture
def single_placement():
    return [CablePlacement(cable_id=1, cx=0.0, cy=-1.0)]


# ---------------------------------------------------------------------------
# multilayer_T_multi
# ---------------------------------------------------------------------------

class TestMultilayerTMulti:
    def test_output_shape(self, single_cable_layers, single_placement, device):
        xy = torch.rand(50, 2, device=device) * 2.0 - 1.0
        xy[:, 1] = -torch.rand(50, device=device) * 2.0  # negative y
        T = multilayer_T_multi(
            xy, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
        )
        assert T.shape == (50, 1)

    def test_far_field_near_ambient(self, single_cable_layers, single_placement, device):
        """Points far from the cable should be close to T_amb."""
        xy = torch.tensor([[0.9, -0.01]], device=device)  # near surface, far from cable
        T = multilayer_T_multi(
            xy, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
        )
        assert T.item() < 300.0  # should be close to 293.15

    def test_cable_centre_hottest(self, single_cable_layers, single_placement, device):
        """Cable centre should be hotter than any soil point."""
        xy_centre = torch.tensor([[0.0, -1.0]], device=device)
        xy_soil = torch.tensor([[0.3, -1.0]], device=device)
        T_c = multilayer_T_multi(
            xy_centre, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
        )
        T_s = multilayer_T_multi(
            xy_soil, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
        )
        assert T_c.item() > T_s.item()

    def test_zero_Q_gives_ambient(self, single_cable_layers, single_placement, device):
        """No heat source -> temperature everywhere equals T_amb."""
        xy = torch.rand(20, 2, device=device) * 2.0 - 1.0
        xy[:, 1] = -torch.rand(20, device=device) * 2.0
        T = multilayer_T_multi(
            xy, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[0.0],
        )
        assert torch.allclose(T, torch.full_like(T, 293.15), atol=1e-3)

    def test_higher_Q_higher_T(self, single_cable_layers, single_placement, device):
        """Increasing Q should raise temperatures."""
        xy = torch.tensor([[0.0, -1.0]], device=device)
        T_low = multilayer_T_multi(
            xy, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[10.0],
        )
        T_high = multilayer_T_multi(
            xy, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[50.0],
        )
        assert T_high.item() > T_low.item()

    def test_enable_grad_allows_gradient(self, single_cable_layers, single_placement, device):
        """enable_grad=True should allow autograd through T_bg."""
        xy = torch.tensor([[0.5, -0.1]], device=device, requires_grad=True)
        T = multilayer_T_multi(
            xy, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0], enable_grad=True,
        )
        loss = T.sum()
        loss.backward()
        assert xy.grad is not None

    def test_disable_grad_no_gradient(self, single_cable_layers, single_placement, device):
        """enable_grad=False (default) should block gradient flow."""
        xy = torch.rand(10, 2, device=device, requires_grad=True)
        xy_soil = xy.clone()
        xy_soil[:, 1] = -0.01
        T = multilayer_T_multi(
            xy_soil, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0], enable_grad=False,
        )
        # T_bg has no grad, so cannot backprop through it
        assert not T.requires_grad

    def test_two_cables_mutual_heating(self, single_cable_layers, device):
        """A cable with a neighbour should be hotter than without."""
        placements_1 = [CablePlacement(1, 0.0, -1.0)]
        placements_2 = [
            CablePlacement(1, 0.0, -1.0),
            CablePlacement(2, 0.3, -1.0),
        ]
        # Measure at the centre of cable 1 which exists in both configs
        xy = torch.tensor([[0.0, -1.0]], device=device)
        T1 = multilayer_T_multi(
            xy, [single_cable_layers], placements_1,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
        )
        T2 = multilayer_T_multi(
            xy, [single_cable_layers], placements_2,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0, 30.0],
        )
        assert T2.item() > T1.item()

    def test_layers_broadcast(self, single_cable_layers, device):
        """Passing a single layers_list with 2 placements should broadcast."""
        placements = [CablePlacement(1, -0.2, -1.0), CablePlacement(2, 0.2, -1.0)]
        xy = torch.tensor([[0.0, -0.5]], device=device)
        T = multilayer_T_multi(
            xy, [single_cable_layers], placements,
            k_soil=1.0, T_amb=293.15, Q_lins=[20.0, 20.0],
        )
        assert T.shape == (1, 1)
        assert T.item() > 293.15

    def test_dielectric_losses(self, single_cable_layers, single_placement, device):
        """Q_d redistributes heat: soil T unchanged, cable-interior profile shifts."""
        # Soil point: T depends on Q_lins only, Q_d should not change it
        xy_soil = torch.tensor([[0.5, -0.5]], device=device)
        T_soil_no_Qd = multilayer_T_multi(
            xy_soil, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0], Q_d=0.0,
        )
        T_soil_with_Qd = multilayer_T_multi(
            xy_soil, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0], Q_d=3.0,
        )
        assert abs(T_soil_no_Qd.item() - T_soil_with_Qd.item()) < 0.01

        # Conductor centre: Q_cond_eff = Q - Q_d => less heat in conductor => cooler
        xy_cond = torch.tensor([[0.0, -1.0]], device=device)
        T_cond_no_Qd = multilayer_T_multi(
            xy_cond, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0], Q_d=0.0,
        )
        T_cond_with_Qd = multilayer_T_multi(
            xy_cond, [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0], Q_d=3.0,
        )
        assert T_cond_with_Qd.item() < T_cond_no_Qd.item()


# ---------------------------------------------------------------------------
# iec60287_estimate
# ---------------------------------------------------------------------------

class TestIEC60287Estimate:
    def test_returns_expected_keys(self, single_cable_layers, single_placement):
        result = iec60287_estimate(
            [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
        )
        assert "cables" in result
        assert "hottest_idx" in result
        assert "T_cond_ref" in result
        assert "Q_lins_W_per_m" in result
        assert "dT_by_layer" in result
        assert "dT_cable" in result

    def test_single_cable_T_above_ambient(self, single_cable_layers, single_placement):
        result = iec60287_estimate(
            [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
        )
        assert result["T_cond_ref"] > 293.15

    def test_zero_Q_gives_ambient(self, single_cable_layers, single_placement):
        result = iec60287_estimate(
            [single_cable_layers], single_placement,
            k_soil=1.0, T_amb=293.15, Q_lins=[0.0],
        )
        assert abs(result["T_cond_ref"] - 293.15) < 0.01

    def test_two_cables_hottest_index(self, single_cable_layers):
        """With symmetric placement and equal Q, both should be equally hot."""
        placements = [CablePlacement(1, -0.3, -1.0), CablePlacement(2, 0.3, -1.0)]
        result = iec60287_estimate(
            [single_cable_layers], placements,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0, 30.0],
        )
        assert len(result["cables"]) == 2
        # Symmetric: both T_cond should be equal
        T1 = result["cables"][0]["T_cond"]
        T2 = result["cables"][1]["T_cond"]
        assert abs(T1 - T2) < 0.01

    def test_higher_k_soil_lower_T(self, single_cable_layers, single_placement):
        """Higher soil conductivity should reduce conductor temperature."""
        r1 = iec60287_estimate(
            [single_cable_layers], single_placement,
            k_soil=0.5, T_amb=293.15, Q_lins=[30.0],
        )
        r2 = iec60287_estimate(
            [single_cable_layers], single_placement,
            k_soil=2.0, T_amb=293.15, Q_lins=[30.0],
        )
        assert r1["T_cond_ref"] > r2["T_cond_ref"]
