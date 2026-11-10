"""Tests for pinn_cables.physics.ground_temp and related readers helpers."""

from __future__ import annotations

import dataclasses
import math
import textwrap
import tempfile
from pathlib import Path

import pytest
import torch

from pinn_cables.physics.ground_temp import (
    ConstantProfile,
    CosineGroundProfile,
    GroundTempProfile,
    PiecewiseLinearProfile,
)
from pinn_cables.io.readers import (
    BoundaryCondition,
    load_boundary_conditions,
    load_boundary_profiles,
    with_profiles,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _xy(y_values: list[float]) -> torch.Tensor:
    """Build (N, 2) tensor with x=0 and given y values."""
    n = len(y_values)
    x = torch.zeros(n, 1)
    y = torch.tensor(y_values, dtype=torch.float32).unsqueeze(1)
    return torch.cat([x, y], dim=1)


def _make_bc(**kwargs) -> BoundaryCondition:
    defaults = dict(boundary="left", bc_type="dirichlet", value=288.35, h=0.0)
    defaults.update(kwargs)
    return BoundaryCondition(**defaults)


# ---------------------------------------------------------------------------
# ConstantProfile
# ---------------------------------------------------------------------------

class TestConstantProfile:
    def test_returns_correct_temperature(self):
        p = ConstantProfile(300.0)
        xy = _xy([-1.0, -2.0, 0.0])
        out = p(xy)
        assert out.shape == (3, 1)
        assert torch.allclose(out, torch.full((3, 1), 300.0))

    def test_output_dtype_matches_input(self):
        p = ConstantProfile(288.35)
        xy = _xy([-1.0]).double()
        out = p(xy)
        assert out.dtype == torch.float64

    def test_is_ground_temp_profile(self):
        assert isinstance(ConstantProfile(300.0), GroundTempProfile)


# ---------------------------------------------------------------------------
# CosineGroundProfile
# ---------------------------------------------------------------------------

class TestCosineGroundProfile:
    def _summer(self) -> CosineGroundProfile:
        """Kim 2024 summer peak: Tg=288.35 K, As=10.9 K, tp=t0=217."""
        return CosineGroundProfile(T_g=288.35, A_s=10.9, tp=217.0)

    def test_surface_temperature(self):
        """At z=0 and tp=t0: T = Tg + As (phase=0 → cos(0)=1)."""
        p = self._summer()
        expected = 288.35 + 10.9
        assert math.isclose(p.T_surface(), expected, rel_tol=1e-6)

    def test_deep_temperature_approaches_Tg(self):
        """At z >> D the exponential kills the oscillation → T → Tg."""
        p = self._summer()
        xy = _xy([-50.0])              # z = 50 m >> D ≈ 3.65 m
        out = p(xy).item()
        assert abs(out - p.T_g) < 0.01

    def test_output_shape(self):
        p = self._summer()
        xy = _xy([-0.0, -1.0, -2.0, -5.0])
        out = p(xy)
        assert out.shape == (4, 1)

    def test_positive_y_treated_as_surface(self):
        """Points above ground (y > 0) should give surface temperature."""
        p = self._summer()
        xy_above = _xy([1.0, 2.0])    # y > 0 → z = 0
        xy_surf  = _xy([0.0, 0.0])
        assert torch.allclose(p(xy_above), p(xy_surf))

    def test_T_at_depth_matches_tensor_call(self):
        p = self._summer()
        z = 1.4
        scalar = p.T_at_depth(z)
        xy = _xy([-z])
        tensor = p(xy).item()
        assert math.isclose(scalar, tensor, rel_tol=1e-5)

    def test_damping_depth_formula(self):
        p = CosineGroundProfile(T_g=288.35, A_s=10.9, alpha=0.0425, tau=365.0)
        expected_D = math.sqrt(365.0 * 0.0425 / math.pi)
        assert math.isclose(p.damping_depth(), expected_D, rel_tol=1e-6)

    def test_is_ground_temp_profile(self):
        assert isinstance(self._summer(), GroundTempProfile)


# ---------------------------------------------------------------------------
# PiecewiseLinearProfile
# ---------------------------------------------------------------------------

class TestPiecewiseLinearProfile:
    def _default(self) -> PiecewiseLinearProfile:
        return PiecewiseLinearProfile(
            depths=[0.0, 1.0, 5.0],
            temps=[300.0, 295.0, 288.35],
        )

    def test_output_shape(self):
        p = self._default()
        xy = _xy([0.0, -1.0, -3.0])
        assert p(xy).shape == (3, 1)

    def test_exact_at_knots(self):
        p = self._default()
        xy = _xy([0.0, -1.0, -5.0])   # z = 0, 1, 5
        out = p(xy).squeeze(1)
        assert torch.allclose(out, torch.tensor([300.0, 295.0, 288.35]), atol=1e-4)

    def test_interpolation_midpoint(self):
        p = PiecewiseLinearProfile(depths=[0.0, 2.0], temps=[300.0, 290.0])
        xy = _xy([-1.0])              # z = 1.0  → midpoint
        expected = 295.0
        assert abs(p(xy).item() - expected) < 1e-4

    def test_clamp_below_surface(self):
        """Points above surface (y > 0) are clamped to z = 0."""
        p = self._default()
        xy_above = _xy([2.0])         # z = max(0, -2) = 0
        xy_surf  = _xy([0.0])
        assert torch.allclose(p(xy_above), p(xy_surf))

    def test_clamp_beyond_deepest_knot(self):
        """Depths beyond last knot return last knot temperature."""
        p = self._default()           # last knot at z=5 → T=288.35
        xy = _xy([-100.0])
        assert abs(p(xy).item() - 288.35) < 1e-4

    def test_unsorted_input_sorted(self):
        """Knots supplied out of order should give the same result."""
        p1 = PiecewiseLinearProfile(depths=[0.0, 1.0], temps=[300.0, 295.0])
        p2 = PiecewiseLinearProfile(depths=[1.0, 0.0], temps=[295.0, 300.0])
        xy = _xy([-0.5])
        assert torch.allclose(p1(xy), p2(xy))

    def test_two_knot_minimum(self):
        with pytest.raises(ValueError, match="at least 2"):
            PiecewiseLinearProfile(depths=[0.0], temps=[300.0])

    def test_knots_property(self):
        p = self._default()
        k = p.knots
        assert k[0] == (0.0, 300.0)
        assert k[-1] == (5.0, 288.35)

    def test_is_ground_temp_profile(self):
        assert isinstance(self._default(), GroundTempProfile)


# ---------------------------------------------------------------------------
# BoundaryCondition.T_target
# ---------------------------------------------------------------------------

class TestBoundaryConditionTTarget:
    def test_scalar_fallback_value(self):
        bc = _make_bc(value=300.0)
        xy = _xy([-1.0, -2.0])
        out = bc.T_target(xy, T_amb=280.0)
        assert torch.allclose(out, xy.new_full((2, 1), 300.0))

    def test_scalar_fallback_T_amb_when_value_zero(self):
        bc = _make_bc(value=0.0)
        xy = _xy([-1.0])
        out = bc.T_target(xy, T_amb=295.0)
        assert abs(out.item() - 295.0) < 1e-5

    def test_profile_overrides_scalar(self):
        profile = ConstantProfile(310.0)
        bc = dataclasses.replace(_make_bc(value=288.0), profile=profile)
        xy = _xy([-1.0, -2.0])
        out = bc.T_target(xy)
        assert torch.allclose(out, xy.new_full((2, 1), 310.0))

    def test_piecewise_profile_via_T_target(self):
        profile = PiecewiseLinearProfile(depths=[0.0, 2.0], temps=[300.0, 290.0])
        bc = dataclasses.replace(_make_bc(), profile=profile)
        xy = _xy([-1.0])              # z=1 → T=295
        assert abs(bc.T_target(xy).item() - 295.0) < 1e-4


# ---------------------------------------------------------------------------
# load_boundary_profiles + with_profiles
# ---------------------------------------------------------------------------

class TestLoadBoundaryProfiles:
    def _write_csv(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "boundary_profiles.csv"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    def test_basic_load(self, tmp_path):
        csv = """\
            boundary,depth_m,T_K
            left,0.0,299.15
            left,1.4,294.26
            left,5.0,288.35
        """
        path = self._write_csv(tmp_path, csv)
        profiles = load_boundary_profiles(path)
        assert "left" in profiles
        p = profiles["left"]
        assert isinstance(p, PiecewiseLinearProfile)
        assert len(p.knots) == 3

    def test_multiple_boundaries(self, tmp_path):
        csv = """\
            boundary,depth_m,T_K
            left,0.0,300.0
            left,5.0,288.35
            right,0.0,300.0
            right,5.0,288.35
            bottom,0.0,288.35
            bottom,1.0,288.35
        """
        path = self._write_csv(tmp_path, csv)
        profiles = load_boundary_profiles(path)
        assert set(profiles) == {"left", "right", "bottom"}


class TestWithProfiles:
    def test_replaces_specified_boundaries(self):
        bcs = {
            "left":  _make_bc(boundary="left"),
            "right": _make_bc(boundary="right"),
            "top":   _make_bc(boundary="top",   bc_type="robin",     value=300.0),
        }
        profiles = {"left": ConstantProfile(299.0), "right": ConstantProfile(299.0)}
        new_bcs = with_profiles(bcs, profiles)

        assert new_bcs["left"].profile is not None
        assert new_bcs["right"].profile is not None
        assert new_bcs["top"].profile is None     # untouched

    def test_original_dict_unchanged(self):
        bcs = {"left": _make_bc(boundary="left")}
        with_profiles(bcs, {"left": ConstantProfile(299.0)})
        assert bcs["left"].profile is None        # original unaffected

    def test_empty_profiles_is_identity(self):
        bcs = {"left": _make_bc(boundary="left")}
        new_bcs = with_profiles(bcs, {})
        assert new_bcs == bcs


# ---------------------------------------------------------------------------
# load_problem auto-detects boundary_profiles.csv
# ---------------------------------------------------------------------------

class TestLoadProblemAutoProfiles:
    """Integration test: load_problem applies boundary_profiles.csv if present."""

    def test_profiles_applied_when_file_exists(self, tmp_path):
        """Copy the aras_2005 data dir and inject a boundary_profiles.csv."""
        import shutil

        src = Path(__file__).parent.parent.parent / "examples" / "aras_2005_154kv" / "data"
        if not src.exists():
            pytest.skip("aras_2005_154kv data not available")

        dst = tmp_path / "data"
        shutil.copytree(src, dst)

        # Write a minimal boundary_profiles.csv for left/right edges
        bp_csv = dst / "boundary_profiles.csv"
        bp_csv.write_text(
            "boundary,depth_m,T_K\n"
            "left,0.0,299.15\n"
            "left,5.0,288.35\n"
            "right,0.0,299.15\n"
            "right,5.0,288.35\n",
            encoding="utf-8",
        )

        from pinn_cables.io.readers import load_problem
        problem = load_problem(dst)

        assert problem.bcs["left"].profile is not None
        assert problem.bcs["right"].profile is not None
        # top should remain untouched (not in boundary_profiles.csv)
        assert problem.bcs["top"].profile is None

    def test_no_profiles_file_leaves_bcs_unchanged(self, tmp_path):
        src = Path(__file__).parent.parent.parent / "examples" / "aras_2005_154kv" / "data"
        if not src.exists():
            pytest.skip("aras_2005_154kv data not available")

        import shutil
        dst = tmp_path / "data"
        shutil.copytree(src, dst)

        from pinn_cables.io.readers import load_problem
        problem = load_problem(dst)
        for bc in problem.bcs.values():
            assert bc.profile is None
