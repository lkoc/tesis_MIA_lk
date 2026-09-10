"""Tests for pinn_cables.post.eval -- error metrics, grid evaluation, conductor temps."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinn_cables.io.readers import CablePlacement, Domain2D
from pinn_cables.post.eval import (
    eval_conductor_temps,
    evaluate_on_grid,
    l2_relative_error,
    linf_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_domain():
    return Domain2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=0.0)


class _ConstModel(nn.Module):
    """Trivial model that always returns a constant value."""
    def __init__(self, val: float):
        super().__init__()
        self.val = val
        # one parameter so .to(device) works
        self._p = nn.Parameter(torch.tensor(val), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._p.expand(x.shape[0], 1)


class _LinearModel(nn.Module):
    """Returns T = x * scale (useful for non-trivial grid checks)."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self._p = nn.Parameter(torch.tensor(scale), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._p * x[:, 0:1]


# ---------------------------------------------------------------------------
# l2_relative_error
# ---------------------------------------------------------------------------

class TestL2RelativeError:
    def test_zero_error(self, device):
        T = torch.ones(100, device=device) * 300.0
        assert l2_relative_error(T, T) == pytest.approx(0.0, abs=1e-6)

    def test_known_error(self, device):
        # T_pred = 2*T_exact  → |error|/|exact| = |T_exact|/|T_exact| = 1.0
        T_exact = torch.ones(100, device=device)
        T_pred = 2.0 * T_exact
        err = l2_relative_error(T_pred, T_exact)
        assert err == pytest.approx(1.0, rel=1e-5)

    def test_two_d_input(self, device):
        T_exact = torch.ones(10, 1, device=device) * 5.0
        T_pred = T_exact.clone()
        assert l2_relative_error(T_pred, T_exact) == pytest.approx(0.0, abs=1e-6)

    def test_zero_reference_returns_abs_norm(self, device):
        """When reference is zero, should return ||T_pred||."""
        T_exact = torch.zeros(50, device=device)
        T_pred = torch.ones(50, device=device)
        err = l2_relative_error(T_pred, T_exact)
        assert err == pytest.approx(math.sqrt(50), rel=1e-4)

    def test_larger_error_larger_value(self, device):
        T_exact = torch.ones(50, device=device)
        T_pred_small = T_exact + 0.01
        T_pred_large = T_exact + 1.0
        err_small = l2_relative_error(T_pred_small, T_exact)
        err_large = l2_relative_error(T_pred_large, T_exact)
        assert err_large > err_small


# ---------------------------------------------------------------------------
# linf_error
# ---------------------------------------------------------------------------

class TestLinfError:
    def test_zero_error(self, device):
        T = torch.randn(100, device=device)
        assert linf_error(T, T) == pytest.approx(0.0, abs=1e-6)

    def test_known_error(self, device):
        T_exact = torch.zeros(10, device=device)
        T_pred = torch.arange(10, dtype=torch.float32, device=device)
        # max |T_pred - T_exact| = 9
        assert linf_error(T_pred, T_exact) == pytest.approx(9.0, rel=1e-5)

    def test_returns_float(self, device):
        T = torch.ones(20, device=device)
        result = linf_error(T, T)
        assert isinstance(result, float)

    def test_consistent_with_l2(self, device):
        """L-inf ≤ L2 norm for unit signals."""
        T_exact = torch.zeros(50, device=device)
        T_pred = T_exact + 0.5
        linf = linf_error(T_pred, T_exact)
        l2 = l2_relative_error(T_pred, T_exact)
        # linf = 0.5, l2_rel = 0.5*sqrt(50)/sqrt(50) = ... depends on normalisation
        assert linf == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# evaluate_on_grid
# ---------------------------------------------------------------------------

class TestEvaluateOnGrid:
    def test_output_shapes(self, small_domain, device):
        model = _ConstModel(300.0).to(device)
        X, Y, T = evaluate_on_grid(model, small_domain, nx=10, ny=8,
                                    device=device, normalize=False)
        assert X.shape == (8, 10)
        assert Y.shape == (8, 10)
        assert T.shape == (8, 10)

    def test_constant_model_returns_constant(self, small_domain, device):
        model = _ConstModel(42.0).to(device)
        _, _, T = evaluate_on_grid(model, small_domain, nx=20, ny=15,
                                    device=device, normalize=False)
        assert np.allclose(T, 42.0, atol=1e-4)

    def test_returns_numpy_arrays(self, small_domain, device):
        model = _ConstModel(0.0).to(device)
        X, Y, T = evaluate_on_grid(model, small_domain, nx=5, ny=5,
                                    device=device, normalize=False)
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert isinstance(T, np.ndarray)

    def test_grid_spans_domain(self, small_domain, device):
        model = _ConstModel(1.0).to(device)
        X, Y, T = evaluate_on_grid(model, small_domain, nx=10, ny=10,
                                    device=device, normalize=False)
        assert X.min() == pytest.approx(small_domain.xmin, abs=1e-4)
        assert X.max() == pytest.approx(small_domain.xmax, abs=1e-4)
        assert Y.min() == pytest.approx(small_domain.ymin, abs=1e-4)
        assert Y.max() == pytest.approx(small_domain.ymax, abs=1e-4)

    def test_normalize_does_not_crash(self, small_domain, device):
        model = _ConstModel(5.0).to(device)
        X, Y, T = evaluate_on_grid(model, small_domain, nx=8, ny=8,
                                    device=device, normalize=True)
        assert T.shape == (8, 8)


# ---------------------------------------------------------------------------
# eval_conductor_temps
# ---------------------------------------------------------------------------

class TestEvalConductorTemps:
    def test_returns_list_of_floats(self, small_domain, device):
        model = _ConstModel(350.0).to(device)
        pls = [CablePlacement(cable_id=0, cx=0.0, cy=-0.5,
                               section_mm2=240, conductor_material="cu", current_A=300.0)]
        T_list = eval_conductor_temps(model, pls, small_domain, device, normalize=False)
        assert isinstance(T_list, list)
        assert len(T_list) == 1
        assert isinstance(T_list[0], float)

    def test_constant_model_returns_constant(self, small_domain, device):
        model = _ConstModel(310.0).to(device)
        pls = [
            CablePlacement(cable_id=0, cx=-0.3, cy=-0.5,
                           section_mm2=240, conductor_material="cu", current_A=300.0),
            CablePlacement(cable_id=1, cx=0.3, cy=-0.7,
                           section_mm2=240, conductor_material="cu", current_A=300.0),
        ]
        T_list = eval_conductor_temps(model, pls, small_domain, device, normalize=False)
        assert len(T_list) == 2
        for T in T_list:
            assert T == pytest.approx(310.0, abs=1e-3)

    def test_multiple_cables_count(self, small_domain, device):
        model = _ConstModel(0.0).to(device)
        n = 4
        pls = [
            CablePlacement(cable_id=i, cx=float(i) * 0.1, cy=-0.5,
                           section_mm2=240, conductor_material="cu", current_A=300.0)
            for i in range(n)
        ]
        T_list = eval_conductor_temps(model, pls, small_domain, device, normalize=False)
        assert len(T_list) == n

    def test_normalize_doesnt_change_constant_model(self, small_domain, device):
        """For a constant model, normalization should not change the output."""
        model = _ConstModel(273.15).to(device)
        pls = [CablePlacement(cable_id=0, cx=0.0, cy=-0.5,
                               section_mm2=240, conductor_material="cu", current_A=300.0)]
        T_no_norm = eval_conductor_temps(model, pls, small_domain, device, normalize=False)
        T_norm = eval_conductor_temps(model, pls, small_domain, device, normalize=True)
        assert T_no_norm[0] == pytest.approx(T_norm[0], abs=1e-3)
