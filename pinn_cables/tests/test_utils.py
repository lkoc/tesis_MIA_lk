"""Tests for pinn_cables.pinn.utils -- device, seed, coordinate normalisation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinn_cables.pinn.utils import (
    denormalize_coords,
    get_device,
    normalize_coords,
    set_seed,
)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_cpu(self):
        d = get_device("cpu")
        assert d == torch.device("cpu")

    def test_auto_returns_device(self):
        d = get_device("auto")
        # Should be either cpu or cuda — just verify it's a valid torch.device
        assert isinstance(d, torch.device)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown device"):
            get_device("tpu")


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

class TestSetSeed:
    def test_same_seed_same_output(self):
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_different_seed_different_output(self):
        set_seed(1)
        t1 = torch.randn(10)
        set_seed(2)
        t2 = torch.randn(10)
        assert not torch.allclose(t1, t2)

    def test_numpy_seeded(self):
        set_seed(99)
        a1 = np.random.rand(5)
        set_seed(99)
        a2 = np.random.rand(5)
        assert np.allclose(a1, a2)


# ---------------------------------------------------------------------------
# normalize_coords / denormalize_coords
# ---------------------------------------------------------------------------

class TestNormalizeCoords:
    def test_maps_to_minus1_plus1(self, device):
        mins = torch.tensor([0.0, -2.0], device=device)
        maxs = torch.tensor([2.0, 0.0], device=device)
        coords = torch.tensor([[0.0, -2.0], [2.0, 0.0], [1.0, -1.0]],
                               device=device)
        out = normalize_coords(coords, mins, maxs)
        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                                  device=device)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_shape_preserved(self, device):
        mins = torch.zeros(3, device=device)
        maxs = torch.ones(3, device=device)
        coords = torch.randn(20, 3, device=device)
        out = normalize_coords(coords, mins, maxs)
        assert out.shape == coords.shape

    def test_round_trip(self, device):
        mins = torch.tensor([-5.0, -10.0], device=device)
        maxs = torch.tensor([5.0, 10.0], device=device)
        coords = torch.randn(50, 2, device=device) * 3.0
        normed = normalize_coords(coords, mins, maxs)
        back = denormalize_coords(normed, mins, maxs)
        assert torch.allclose(back, coords, atol=1e-4)


class TestDenormalizeCoords:
    def test_maps_from_minus1_plus1(self, device):
        mins = torch.tensor([0.0, 0.0], device=device)
        maxs = torch.tensor([4.0, 8.0], device=device)
        normed = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
                               device=device)
        out = denormalize_coords(normed, mins, maxs)
        expected = torch.tensor([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]],
                                  device=device)
        assert torch.allclose(out, expected, atol=1e-5)
