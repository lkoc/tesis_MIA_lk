"""Tests for pinn_cables.physics.k_field -- PhysicsParams, load, k_scalar, k_tensor."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from pinn_cables.physics.k_field import (
    KFieldModel,
    PhysicsParams,
    SoilLayerBand,
    k_scalar,
    k_tensor,
    load_physics_params,
    load_soil_layers,
)


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


# ---------------------------------------------------------------------------
# SoilLayerBand
# ---------------------------------------------------------------------------

class TestSoilLayerBand:
    def test_valid_band(self):
        b = SoilLayerBand(y_top=0.0, y_bottom=-1.0, k=1.5)
        assert b.k == 1.5
        assert b.rho_c == 0.0

    def test_invalid_y_order(self):
        with pytest.raises(ValueError, match="y_bottom"):
            SoilLayerBand(y_top=-1.0, y_bottom=0.0, k=1.5)

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k must be"):
            SoilLayerBand(y_top=0.0, y_bottom=-1.0, k=0.0)

    def test_frozen(self):
        b = SoilLayerBand(y_top=0.0, y_bottom=-1.0, k=1.5)
        with pytest.raises(AttributeError):
            b.k = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_soil_layers
# ---------------------------------------------------------------------------

def _write_soil_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer_id", "y_top", "y_bottom", "k", "rho_c"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestLoadSoilLayers:
    def test_loads_three_layers(self, tmp_path):
        p = tmp_path / "soil_layers.csv"
        _write_soil_csv(p, [
            {"layer_id": 1, "y_top": 0.0,    "y_bottom": -0.56, "k": 1.804, "rho_c": 3182000},
            {"layer_id": 2, "y_top": -0.56,  "y_bottom": -1.76, "k": 1.351, "rho_c": 3645000},
            {"layer_id": 3, "y_top": -1.76,  "y_bottom": -45.5, "k": 1.517, "rho_c": 3051000},
        ])
        bands = load_soil_layers(p)
        assert len(bands) == 3
        assert bands[0].y_top == pytest.approx(0.0)           # shallowest first
        assert bands[0].k == pytest.approx(1.804)
        assert bands[1].k == pytest.approx(1.351)
        assert bands[2].k == pytest.approx(1.517)

    def test_sorted_shallow_to_deep(self, tmp_path):
        p = tmp_path / "soil_layers.csv"
        _write_soil_csv(p, [
            {"layer_id": 3, "y_top": -1.76, "y_bottom": -45.5, "k": 1.517, "rho_c": 0},
            {"layer_id": 1, "y_top":  0.0,  "y_bottom":  -0.56, "k": 1.804, "rho_c": 0},
            {"layer_id": 2, "y_top": -0.56, "y_bottom":  -1.76, "k": 1.351, "rho_c": 0},
        ])
        bands = load_soil_layers(p)
        assert bands[0].y_top > bands[1].y_top > bands[2].y_top

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_soil_layers(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# KFieldModel — uniform factory
# ---------------------------------------------------------------------------

class TestKFieldModelUniform:
    def test_uniform_scalar(self):
        m = KFieldModel.uniform(k_soil=1.55)
        assert m.k_scalar(0.0, -1.0) == pytest.approx(1.55)
        assert m.k_scalar(5.0, -20.0) == pytest.approx(1.55)

    def test_uniform_tensor(self, device):
        m = KFieldModel.uniform(k_soil=1.55)
        xy = torch.randn(20, 2, device=device)
        k = m(xy)
        assert k.shape == (20, 1)
        assert k.allclose(torch.full_like(k, 1.55))

    def test_uniform_no_layers_no_pac(self):
        m = KFieldModel.uniform(1.0)
        assert not m.has_layers
        assert not m.has_pac
        assert m.transition_hints() == []

    def test_k_eff_bg_fallback(self):
        m = KFieldModel.uniform(1.5)
        assert m.k_eff_bg() == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# KFieldModel — multilayer soil
# ---------------------------------------------------------------------------

class TestKFieldModelMultilayer:
    @pytest.fixture()
    def three_bands(self) -> list[SoilLayerBand]:
        return [
            SoilLayerBand(y_top=0.0,   y_bottom=-0.56, k=1.804),
            SoilLayerBand(y_top=-0.56, y_bottom=-1.76, k=1.351),
            SoilLayerBand(y_top=-1.76, y_bottom=-45.5, k=1.517),
        ]

    def test_scalar_in_each_layer(self, three_bands):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands)
        # Layer 1: y = -0.30  → k ≈ 1.804
        assert m.k_scalar(0.0, -0.30) == pytest.approx(1.804, abs=0.02)
        # Layer 2: y = -1.20  → k ≈ 1.351 (where cables are)
        assert m.k_scalar(0.0, -1.20) == pytest.approx(1.351, abs=0.02)
        # Layer 3: y = -5.00  → k ≈ 1.517
        assert m.k_scalar(0.0, -5.00) == pytest.approx(1.517, abs=0.02)

    def test_tensor_shape(self, three_bands, device):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands)
        xy = torch.randn(100, 2, device=device)
        k = m(xy)
        assert k.shape == (100, 1)

    def test_tensor_range(self, three_bands, device):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands)
        y_vals = torch.linspace(-40.0, 0.0, 200, device=device)
        x_vals = torch.zeros_like(y_vals)
        xy = torch.stack([x_vals, y_vals], dim=1)
        k = m(xy)
        # All k values should be in [min_band_k - eps, max_band_k + eps]
        k_min = min(b.k for b in three_bands)
        k_max = max(b.k for b in three_bands)
        assert k.min().item() >= k_min - 0.05
        assert k.max().item() <= k_max + 0.05

    def test_differentiable(self, three_bands, device):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands)
        xy = torch.randn(10, 2, device=device, requires_grad=True)
        k = m(xy)
        k.sum().backward()
        assert xy.grad is not None

    def test_has_layers_flag(self, three_bands):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands)
        assert m.has_layers is True

    def test_transition_hints_has_interfaces(self, three_bands):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands)
        hints = m.transition_hints()
        # Two interfaces between three bands
        assert len([h for h in hints if h["type"] == "horizontal_strip"]) == 2

    def test_k_eff_bg_at_cable_depth(self, three_bands):
        """Centroid of cables at y=-1.4 is in Layer 2 (k=1.351)."""
        from pinn_cables.io.readers import CablePlacement
        pls = [CablePlacement(cx=0.0, cy=-1.2, cable_id=0,
                               section_mm2=1200, conductor_material="cu", current_A=1026.0),
               CablePlacement(cx=0.0, cy=-1.6, cable_id=0,
                               section_mm2=1200, conductor_material="cu", current_A=1026.0)]
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands)
        k_bg = m.k_eff_bg(pls)
        # centroid at y=-1.4, which is in Layer 2 (k=1.351); at the sigmoid transition
        assert 1.30 < k_bg < 1.55


# ---------------------------------------------------------------------------
# KFieldModel — PAC zone only
# ---------------------------------------------------------------------------

class TestKFieldModelPAC:
    @pytest.fixture()
    def pac_pp(self) -> PhysicsParams:
        return PhysicsParams(
            k_variable=True, k_good=2.094, k_bad=1.55,
            k_cx=0.0, k_cy=-1.40, k_width=1.30, k_height=0.90,
            k_transition=0.10,
        )

    def test_inside_pac_zone(self, pac_pp):
        m = KFieldModel(k_soil=pac_pp.k_bad, pac_params=pac_pp)
        k = m.k_scalar(0.0, -1.40)
        assert abs(k - pac_pp.k_good) < 0.05

    def test_outside_pac_zone(self, pac_pp):
        m = KFieldModel(k_soil=pac_pp.k_bad, pac_params=pac_pp)
        k = m.k_scalar(10.0, -10.0)
        assert abs(k - pac_pp.k_bad) < 0.01

    def test_has_pac_flag(self, pac_pp):
        m = KFieldModel(k_soil=pac_pp.k_bad, pac_params=pac_pp)
        assert m.has_pac is True

    def test_pac_transition_hint(self, pac_pp):
        m = KFieldModel(k_soil=pac_pp.k_bad, pac_params=pac_pp)
        hints = m.transition_hints()
        pac_hints = [h for h in hints if h["type"] == "pac_boundary"]
        assert len(pac_hints) == 1

    def test_k_variable_false_ignored(self):
        pp = PhysicsParams(k_variable=False, k_good=2.0, k_bad=1.0)
        m = KFieldModel(k_soil=1.0, pac_params=pp)
        assert not m.has_pac
        # k_variable=False: should return k_soil everywhere
        assert m.k_scalar(0.0, 0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# KFieldModel — combined multilayer + PAC
# ---------------------------------------------------------------------------

class TestKFieldModelCombined:
    @pytest.fixture()
    def three_bands(self) -> list[SoilLayerBand]:
        return [
            SoilLayerBand(y_top=0.0,   y_bottom=-0.56, k=1.804),
            SoilLayerBand(y_top=-0.56, y_bottom=-1.76, k=1.351),
            SoilLayerBand(y_top=-1.76, y_bottom=-45.5, k=1.517),
        ]

    @pytest.fixture()
    def pac_pp(self) -> PhysicsParams:
        return PhysicsParams(
            k_variable=True, k_good=2.094, k_bad=1.55,
            k_cx=0.0, k_cy=-1.40, k_width=1.30, k_height=0.90,
            k_transition=0.05,
        )

    def test_inside_pac_overrides_layer2(self, three_bands, pac_pp):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands, pac_params=pac_pp)
        # Inside PAC zone (which is in Layer 2 region) → k near k_PAC
        k = m.k_scalar(0.0, -1.40)
        assert k > 1.9   # closer to k_PAC=2.094 than Layer2=1.351

    def test_outside_pac_in_layer1(self, three_bands, pac_pp):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands, pac_params=pac_pp)
        # Outside PAC zone and in Layer 1 (y=-0.30) → k near Layer1 k
        k = m.k_scalar(10.0, -0.30)
        assert abs(k - 1.804) < 0.02

    def test_all_hints_present(self, three_bands, pac_pp):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands, pac_params=pac_pp)
        hints = m.transition_hints()
        types = {h["type"] for h in hints}
        assert "horizontal_strip" in types
        assert "pac_boundary" in types

    def test_combined_tensor_differentiable(self, three_bands, pac_pp, device):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands, pac_params=pac_pp)
        xy = torch.randn(20, 2, device=device, requires_grad=True)
        k = m(xy)
        k.sum().backward()
        assert xy.grad is not None

    def test_repr_contains_info(self, three_bands, pac_pp):
        m = KFieldModel(k_soil=1.55, soil_bands=three_bands, pac_params=pac_pp)
        r = repr(m)
        assert "1.55" in r
        assert "soil bands" in r
        assert "PAC" in r


# ---------------------------------------------------------------------------
# KFieldModel — from_csvs factory
# ---------------------------------------------------------------------------

class TestKFieldModelFromCsvs:
    def test_from_csvs_no_files(self, tmp_path):
        m = KFieldModel.from_csvs(k_soil=1.55)
        assert m.k_scalar(0.0, -1.0) == pytest.approx(1.55)

    def test_from_csvs_with_soil_layers(self, tmp_path):
        p = tmp_path / "soil_layers.csv"
        _write_soil_csv(p, [
            {"layer_id": 1, "y_top": 0.0,   "y_bottom": -1.0,  "k": 2.0, "rho_c": 0},
            {"layer_id": 2, "y_top": -1.0,  "y_bottom": -10.0, "k": 1.0, "rho_c": 0},
        ])
        m = KFieldModel.from_csvs(k_soil=1.5, soil_layers_path=p)
        assert m.has_layers
        assert m.k_scalar(0.0, -0.5) == pytest.approx(2.0, abs=0.05)
        assert m.k_scalar(0.0, -5.0) == pytest.approx(1.0, abs=0.05)

    def test_from_csvs_with_physics_params(self, tmp_path):
        pp_path = tmp_path / "physics_params.csv"
        _write_physics_csv(pp_path, [
            ["k_variable", "true"],
            ["k_good", "2.0"], ["k_bad", "1.0"],
            ["k_cx", "0.0"], ["k_cy", "-1.0"],
            ["k_width", "0.5"], ["k_height", "0.5"],
            ["k_transition", "0.05"],
        ])
        m = KFieldModel.from_csvs(k_soil=1.0, physics_params_path=pp_path)
        assert m.has_pac
        assert m.k_scalar(0.0, -1.0) > 1.9  # inside PAC

