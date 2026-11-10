"""Tests for pinn_cables.pinn.model -- MLP, Fourier features, build_model, and ResidualPINNModel."""

from __future__ import annotations

import torch

from pinn_cables.io.readers import CableLayer, CablePlacement, Domain2D
from pinn_cables.pinn.model import (
    FourierFeatureMapping,
    FourierFeatureNet,
    MLP,
    ResidualPINNModel,
    build_model,
)


def test_mlp_output_shape(device):
    model = MLP(in_dim=2, out_dim=1, width=32, depth=3).to(device)
    x = torch.randn(50, 2, device=device)
    y = model(x)
    assert y.shape == (50, 1)


def test_mlp_xavier_init(device):
    model = MLP(in_dim=2, width=64, depth=4).to(device)
    for m in model.net:
        if isinstance(m, torch.nn.Linear):
            # Xavier uniform: weights should be in a reasonable range
            w = m.weight.data
            fan_in = w.shape[1]
            fan_out = w.shape[0]
            limit = (6.0 / (fan_in + fan_out)) ** 0.5
            assert w.abs().max() <= limit + 0.01


def test_fourier_feature_mapping_output_dim(device):
    ff = FourierFeatureMapping(in_dim=2, mapping_size=32, scale=1.0).to(device)
    assert ff.out_dim == 64
    x = torch.randn(10, 2, device=device)
    out = ff(x)
    assert out.shape == (10, 64)


def test_fourier_feature_net_output_shape(device):
    model = FourierFeatureNet(
        in_dim=2, out_dim=1, width=32, depth=3,
        fourier_mapping_size=16,
    ).to(device)
    x = torch.randn(20, 2, device=device)
    y = model(x)
    assert y.shape == (20, 1)


def test_build_model_plain(device):
    cfg = {"width": 32, "depth": 3, "activation": "tanh", "fourier_features": False}
    model = build_model(cfg, in_dim=2, device=device)
    assert isinstance(model, MLP)


def test_build_model_fourier(device):
    cfg = {
        "width": 32, "depth": 3, "activation": "tanh",
        "fourier_features": True, "fourier_scale": 1.0, "fourier_mapping_size": 16,
    }
    model = build_model(cfg, in_dim=2, device=device)
    assert isinstance(model, FourierFeatureNet)


def test_gradients_flow(device):
    model = MLP(in_dim=2, width=16, depth=2).to(device)
    x = torch.randn(5, 2, device=device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_deterministic_with_seed(device):
    torch.manual_seed(123)
    m1 = MLP(in_dim=2, width=16, depth=2).to(device)
    x = torch.randn(5, 2, device=device)

    torch.manual_seed(123)
    m2 = MLP(in_dim=2, width=16, depth=2).to(device)

    y1 = m1(x)
    y2 = m2(x)
    assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# ResidualPINNModel
# ---------------------------------------------------------------------------

class TestResidualPINNModel:
    """Tests for the residual model T = T_bg(Kennelly) + u(NN)."""

    @staticmethod
    def _make_model(device, enable_grad_Tbg=False):
        layers = [
            CableLayer("conductor", 0.0, 0.0125, 400.0, 3.45e6, 30.0),
            CableLayer("xlpe", 0.0125, 0.028, 0.286, 2.4e6, 0.0),
        ]
        placements = [CablePlacement(cable_id=1, cx=0.0, cy=-1.0)]
        domain = Domain2D(xmin=-1.0, xmax=1.0, ymin=-2.0, ymax=0.0)
        base = MLP(in_dim=2, out_dim=1, width=16, depth=2).to(device)
        model = ResidualPINNModel(
            base, [layers], placements,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
            domain=domain, normalize=True,
            enable_grad_Tbg=enable_grad_Tbg,
        ).to(device)
        return model

    def test_output_shape(self, device):
        model = self._make_model(device)
        xy_n = torch.rand(20, 2, device=device) * 2.0 - 1.0  # normalised
        T = model(xy_n)
        assert T.shape == (20, 1)

    def test_output_above_ambient(self, device):
        """With Q > 0, predictions should be above ambient (at least T_bg)."""
        model = self._make_model(device)
        # A point near the cable in normalised coords
        xy_n = torch.tensor([[0.0, 0.0]], device=device)  # maps to centre of domain
        with torch.no_grad():
            T = model(xy_n)
        assert T.item() > 290.0  # should be near or above T_amb

    def test_mutable_Q_lins(self, device):
        """_Q_lins should be mutable for iterative R(T) updates."""
        model = self._make_model(device)
        xy_n = torch.tensor([[0.0, 0.0]], device=device)
        with torch.no_grad():
            T1 = model(xy_n).item()
        model._Q_lins[0] = 60.0
        with torch.no_grad():
            T2 = model(xy_n).item()
        assert T2 > T1  # doubling Q should raise T

    def test_gradient_flow_with_enable_grad(self, device):
        model = self._make_model(device, enable_grad_Tbg=True)
        xy_n = torch.zeros(5, 2, device=device, requires_grad=True)
        T = model(xy_n)
        T.sum().backward()
        # Input should have gradients (leaf tensor)
        assert xy_n.grad is not None
        for p in model.base.parameters():
            assert p.grad is not None

    def test_denormalize_roundtrip(self, device):
        """_denormalize should map [-1, 1] back to physical domain."""
        model = self._make_model(device)
        corners_n = torch.tensor([
            [-1.0, -1.0],
            [1.0, 1.0],
        ], device=device)
        phys = model._denormalize(corners_n)
        assert abs(phys[0, 0].item() - (-1.0)) < 1e-5  # xmin
        assert abs(phys[0, 1].item() - (-2.0)) < 1e-5  # ymin
        assert abs(phys[1, 0].item() - 1.0) < 1e-5     # xmax
        assert abs(phys[1, 1].item() - 0.0) < 1e-5     # ymax

    def test_dielectric_losses(self, device):
        """Q_d changes the cable-interior temperature profile."""
        layers = [
            CableLayer("conductor", 0.0, 0.0125, 400.0, 3.45e6, 30.0),
            CableLayer("xlpe", 0.0125, 0.028, 0.286, 2.4e6, 0.0),
        ]
        placements = [CablePlacement(cable_id=1, cx=0.0, cy=-1.0)]
        domain = Domain2D(xmin=-1.0, xmax=1.0, ymin=-2.0, ymax=0.0)

        torch.manual_seed(42)
        base1 = MLP(in_dim=2, out_dim=1, width=16, depth=2).to(device)
        m1 = ResidualPINNModel(
            base1, [layers], placements, k_soil=1.0, T_amb=293.15,
            Q_lins=[30.0], domain=domain, Q_d=0.0,
        ).to(device)

        torch.manual_seed(42)
        base2 = MLP(in_dim=2, out_dim=1, width=16, depth=2).to(device)
        m2 = ResidualPINNModel(
            base2, [layers], placements, k_soil=1.0, T_amb=293.15,
            Q_lins=[30.0], domain=domain, Q_d=3.0,
        ).to(device)

        # At cable centre (normalised: x_n = 2*(0-(-1))/2-1 = 0, y_n = 2*(-1-(-2))/2-1 = 0)
        # Q_d redistributes heat -> conductor centre cooler (Q_cond_eff = Q - Q_d)
        xy_n = torch.tensor([[0.0, 0.0]], device=device)
        with torch.no_grad():
            T1 = m1(xy_n).item()
            T2 = m2(xy_n).item()
        assert T2 < T1
