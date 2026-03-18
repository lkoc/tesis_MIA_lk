"""Tests for pinn_cables.pinn.model -- MLP, Fourier features, and build_model."""

from __future__ import annotations

import torch

from pinn_cables.pinn.model import (
    FourierFeatureMapping,
    FourierFeatureNet,
    MLP,
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
