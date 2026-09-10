"""Tests for pinn_cables.physics.iec60287 -- IEC 60287 heat computation."""

from __future__ import annotations

import pytest

from pinn_cables.physics.iec60287 import Q_lin_from_I, compute_iec60287_Q


# ---------------------------------------------------------------------------
# compute_iec60287_Q
# ---------------------------------------------------------------------------

class TestComputeIEC60287Q:
    def test_returns_expected_keys(self):
        result = compute_iec60287_Q(
            section_mm2=1200, material="cu", current_A=1000.0, T_op=343.75,
        )
        expected_keys = {
            "R_dc_20", "R_dc_T", "ys", "R_ac",
            "Q_cond_W_per_m", "W_d", "Q_total_W_per_m", "ratio_vs_Rdc20",
        }
        assert set(result.keys()) == expected_keys

    def test_Q_total_positive(self):
        result = compute_iec60287_Q(
            section_mm2=1200, material="cu", current_A=1000.0, T_op=343.75,
        )
        assert result["Q_total_W_per_m"] > 0.0

    def test_zero_current_zero_Q_cond(self):
        result = compute_iec60287_Q(
            section_mm2=400, material="cu", current_A=0.0, T_op=293.15,
        )
        assert result["Q_cond_W_per_m"] == 0.0

    def test_dielectric_losses_added(self):
        r_no_wd = compute_iec60287_Q(
            section_mm2=1200, material="cu", current_A=1000.0, T_op=343.75, W_d=0.0,
        )
        r_with_wd = compute_iec60287_Q(
            section_mm2=1200, material="cu", current_A=1000.0, T_op=343.75, W_d=5.0,
        )
        assert r_with_wd["Q_total_W_per_m"] == pytest.approx(
            r_no_wd["Q_total_W_per_m"] + 5.0, abs=1e-6,
        )

    def test_R_ac_greater_than_R_dc(self):
        """Skin effect should make R_ac > R_dc_T."""
        result = compute_iec60287_Q(
            section_mm2=1200, material="cu", current_A=1000.0, T_op=343.75,
        )
        assert result["R_ac"] >= result["R_dc_T"]
        assert result["ys"] >= 0.0

    def test_higher_T_higher_R_dc(self):
        """R_dc_T should increase with temperature (positive alpha_R for Cu)."""
        r_cold = compute_iec60287_Q(
            section_mm2=400, material="cu", current_A=500.0, T_op=293.15,
        )
        r_hot = compute_iec60287_Q(
            section_mm2=400, material="cu", current_A=500.0, T_op=363.15,
        )
        assert r_hot["R_dc_T"] > r_cold["R_dc_T"]

    def test_aluminium_higher_R_than_copper(self):
        """Al should have higher R_dc_20 than Cu for the same section."""
        r_cu = compute_iec60287_Q(
            section_mm2=400, material="cu", current_A=500.0, T_op=293.15,
        )
        r_al = compute_iec60287_Q(
            section_mm2=400, material="al", current_A=500.0, T_op=293.15,
        )
        assert r_al["R_dc_20"] > r_cu["R_dc_20"]

    def test_ratio_vs_Rdc20(self):
        result = compute_iec60287_Q(
            section_mm2=1200, material="cu", current_A=1000.0, T_op=343.75,
        )
        assert result["ratio_vs_Rdc20"] == pytest.approx(
            result["R_ac"] / result["R_dc_20"], rel=1e-10,
        )


# ---------------------------------------------------------------------------
# Q_lin_from_I
# ---------------------------------------------------------------------------

class TestQLinFromI:
    def test_zero_current(self):
        assert Q_lin_from_I(I=0.0, R_ref=1e-4, alpha_R=0.004, T_cond=350.0, T_ref=293.15) == 0.0

    def test_known_value(self):
        """Q = I² × R_ref × (1 + alpha × (T - T_ref))."""
        I = 1000.0
        R_ref = 1.5e-5
        alpha_R = 0.00393
        T_cond = 343.75
        T_ref = 293.15
        expected = I**2 * R_ref * (1.0 + alpha_R * (T_cond - T_ref))
        assert Q_lin_from_I(I, R_ref, alpha_R, T_cond, T_ref) == pytest.approx(expected, rel=1e-10)

    def test_higher_T_higher_Q(self):
        """Q should increase with conductor temperature."""
        Q_cold = Q_lin_from_I(I=500.0, R_ref=5e-5, alpha_R=0.004, T_cond=293.15, T_ref=293.15)
        Q_hot = Q_lin_from_I(I=500.0, R_ref=5e-5, alpha_R=0.004, T_cond=363.15, T_ref=293.15)
        assert Q_hot > Q_cold
