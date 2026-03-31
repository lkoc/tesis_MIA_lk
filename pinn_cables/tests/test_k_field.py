"""Tests for pinn_cables.physics.k_field -- PhysicsParams, load, k_scalar, k_tensor."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from pinn_cables.physics.k_field import PhysicsParams, k_scalar, k_tensor, load_physics_params


# ---------------------------------------------------------------------------
# PhysicsParams
# ---------------------------------------------------------------------------

class TestPhysicsParams:
    def test_defaults(self):
        pp = PhysicsParams()
        assert pp.k_variable is True
        assert pp.k_good == 1.5
        assert pp.k_bad == 0.8
        assert pp.k_transition == 0.05
        assert pp.alpha_R == pytest.approx(0.00393)

    def test_custom_values(self):
        pp = PhysicsParams(k_good=2.0, k_bad=1.0, k_cx=0.5, k_cy=-0.5)
        assert pp.k_good == 2.0
        assert pp.k_cx == 0.5

    def test_frozen(self):
        pp = PhysicsParams()
        with pytest.raises(AttributeError):
            pp.k_good = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_physics_params
# ---------------------------------------------------------------------------

def _write_physics_csv(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "value"])
        for r in rows:
            w.writerow(r)


class TestLoadPhysicsParams:
    def test_loads_from_csv(self, tmp_path):
        p = tmp_path / "physics_params.csv"
        _write_physics_csv(p, [
            ["k_variable", "true"],
            ["k_good", "2.094"],
            ["k_bad", "1.55"],
            ["k_cx", "0.0"],
            ["k_cy", "-1.3"],
            ["k_width", "0.80"],
            ["k_height", "0.60"],
            ["k_transition", "0.10"],
        ])
        pp = load_physics_params(p)
        assert pp.k_variable is True
        assert pp.k_good == pytest.approx(2.094)
        assert pp.k_bad == pytest.approx(1.55)
        assert pp.k_width == pytest.approx(0.80)
        assert pp.k_transition == pytest.approx(0.10)

    def test_missing_file_returns_defaults(self, tmp_path):
        pp = load_physics_params(tmp_path / "nonexistent.csv")
        assert pp == PhysicsParams()

    def test_partial_csv_uses_defaults(self, tmp_path):
        p = tmp_path / "physics_params.csv"
        _write_physics_csv(p, [["k_good", "3.0"]])
        pp = load_physics_params(p)
        assert pp.k_good == 3.0
        assert pp.k_bad == 0.8  # default

    def test_bool_parsing(self, tmp_path):
        p = tmp_path / "physics_params.csv"
        _write_physics_csv(p, [["k_variable", "false"]])
        pp = load_physics_params(p)
        assert pp.k_variable is False

    def test_int_parsing(self, tmp_path):
        p = tmp_path / "physics_params.csv"
        _write_physics_csv(p, [["n_R_iter", "5"]])
        pp = load_physics_params(p)
        assert pp.n_R_iter == 5
        assert isinstance(pp.n_R_iter, int)


# ---------------------------------------------------------------------------
# k_scalar
# ---------------------------------------------------------------------------

class TestKScalar:
    def test_centre_of_good_zone(self):
        pp = PhysicsParams(k_good=2.0, k_bad=1.0, k_cx=0.0, k_cy=-1.0,
                           k_width=1.0, k_height=1.0, k_transition=0.05)
        k = k_scalar(0.0, -1.0, pp)
        # Inside zone centre -> d < 0 -> sigmoid ~ 1 -> k ~ k_good
        assert abs(k - 2.0) < 0.05

    def test_far_from_zone(self):
        pp = PhysicsParams(k_good=2.0, k_bad=1.0, k_cx=0.0, k_cy=-1.0,
                           k_width=0.5, k_height=0.5, k_transition=0.05)
        k = k_scalar(10.0, 10.0, pp)
        # Far away -> d >> 0 -> sigmoid ~ 0 -> k ~ k_bad
        assert abs(k - 1.0) < 0.01

    def test_transition_monotone(self):
        """k should transition smoothly from k_good to k_bad."""
        pp = PhysicsParams(k_good=2.0, k_bad=1.0, k_cx=0.0, k_cy=0.0,
                           k_width=1.0, k_height=1.0, k_transition=0.1)
        k_inside = k_scalar(0.0, 0.0, pp)
        k_edge = k_scalar(0.5, 0.0, pp)  # exactly at edge
        k_outside = k_scalar(2.0, 0.0, pp)
        assert k_inside > k_edge > k_outside

    def test_k_variable_false(self):
        pp = PhysicsParams(k_variable=False, k_bad=1.5)
        assert k_scalar(0.0, 0.0, pp) == 1.5
        assert k_scalar(100.0, -100.0, pp) == 1.5


# ---------------------------------------------------------------------------
# k_tensor
# ---------------------------------------------------------------------------

class TestKTensor:
    def test_output_shape(self, device):
        pp = PhysicsParams()
        xy = torch.randn(50, 2, device=device)
        k = k_tensor(xy, pp)
        assert k.shape == (50, 1)

    def test_range_between_k_bad_and_k_good(self, device):
        pp = PhysicsParams(k_good=2.0, k_bad=1.0, k_transition=0.05)
        xy = torch.randn(500, 2, device=device) * 5.0
        k = k_tensor(xy, pp)
        assert k.min().item() >= 1.0 - 1e-4
        assert k.max().item() <= 2.0 + 1e-4

    def test_matches_k_scalar(self, device):
        """k_tensor should give the same values as k_scalar (within float tolerance)."""
        pp = PhysicsParams(k_good=2.0, k_bad=1.0, k_cx=0.0, k_cy=-1.0,
                           k_width=0.5, k_height=0.5, k_transition=0.1)
        test_pts = [(0.0, -1.0), (0.0, 0.0), (1.5, -1.0), (-0.25, -1.0)]
        for x, y in test_pts:
            k_s = k_scalar(x, y, pp)
            k_t = k_tensor(torch.tensor([[x, y]], device=device), pp)
            assert abs(k_t.item() - k_s) < 1e-5, f"Mismatch at ({x}, {y})"

    def test_differentiable(self, device):
        """k_tensor should support autograd."""
        pp = PhysicsParams()
        xy = torch.randn(10, 2, device=device, requires_grad=True)
        k = k_tensor(xy, pp)
        k.sum().backward()
        assert xy.grad is not None
