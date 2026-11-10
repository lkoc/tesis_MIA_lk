"""Tests for pinn_cables.geom.sampler -- geometry and collocation-point sampling."""

from __future__ import annotations

import math

import torch

from pinn_cables.geom.sampler import (
    append_time,
    sample_boundary_points,
    sample_domain_points,
    sample_initial_condition,
    sample_time,
)
from pinn_cables.io.readers import CableLayer, CablePlacement, Domain2D


def test_interior_points_non_empty(cable_layers, domain, placement, device):
    pts, _ = sample_domain_points(domain, cable_layers, placement, 2000, 100, device=device)
    for name, t in pts.items():
        assert t.shape[0] > 0, f"Region '{name}' has no points"
        assert t.shape[1] == 2


def test_interior_points_in_correct_region(cable_layers, domain, placement, device):
    pts, _ = sample_domain_points(domain, cable_layers, placement, 4000, 100, oversample=10, device=device)
    cx, cy = placement.cx, placement.cy

    # Conductor: r < r_outer
    cond = pts["conductor"]
    r2 = (cond[:, 0] - cx) ** 2 + (cond[:, 1] - cy) ** 2
    assert (r2 < cable_layers[0].r_outer ** 2 + 1e-6).all()

    # Soil: r >= outer-most layer radius
    soil = pts["soil"]
    r2_soil = (soil[:, 0] - cx) ** 2 + (soil[:, 1] - cy) ** 2
    r_max = cable_layers[-1].r_outer
    assert (r2_soil >= r_max ** 2 - 1e-6).all()


def test_interface_points_near_radius(cable_layers, domain, placement, device):
    _, ifc = sample_domain_points(domain, cable_layers, placement, 2000, 200, device=device)
    cx, cy = placement.cx, placement.cy
    eps = 0.002 * cable_layers[-1].r_outer * 2  # generous tolerance

    for layer in cable_layers:
        key = f"r_{layer.name}"
        pts = ifc[key]
        r = torch.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        assert (r - layer.r_outer).abs().max() < eps


def test_boundary_points_on_edges(domain, device):
    bnd = sample_boundary_points(domain, 200, device=device)
    assert torch.allclose(bnd["top"][:, 1], torch.tensor(domain.ymax))
    assert torch.allclose(bnd["bottom"][:, 1], torch.tensor(domain.ymin))
    assert torch.allclose(bnd["left"][:, 0], torch.tensor(domain.xmin))
    assert torch.allclose(bnd["right"][:, 0], torch.tensor(domain.xmax))


def test_boundary_within_domain(domain, device):
    bnd = sample_boundary_points(domain, 200, device=device)
    for _, pts in bnd.items():
        assert (pts[:, 0] >= domain.xmin - 1e-6).all()
        assert (pts[:, 0] <= domain.xmax + 1e-6).all()
        assert (pts[:, 1] >= domain.ymin - 1e-6).all()
        assert (pts[:, 1] <= domain.ymax + 1e-6).all()


def test_requires_grad(cable_layers, domain, placement, device):
    pts, ifc = sample_domain_points(domain, cable_layers, placement, 500, 50, device=device)
    for t in pts.values():
        assert t.requires_grad
    for t in ifc.values():
        assert t.requires_grad

    bnd = sample_boundary_points(domain, 100, device=device)
    for t in bnd.values():
        assert t.requires_grad


def test_sample_time_range(device):
    t = sample_time(100, 0.0, 3600.0, device=device)
    assert t.shape == (100, 1)
    assert t.min() >= 0.0
    assert t.max() <= 3600.0
    assert t.requires_grad


def test_sample_initial_condition(domain, device):
    ic = sample_initial_condition(domain, 200, t0=0.0, device=device)
    assert ic.shape == (200, 3)
    assert torch.allclose(ic[:, 2], torch.tensor(0.0))
    assert ic.requires_grad


def test_append_time_output_shape(device):
    xy = torch.rand(50, 2, device=device)
    t = torch.rand(1, 1, device=device)
    out = append_time(xy, t)
    assert out.shape == (50, 3)
    assert out.requires_grad
    # The time column should be broadcast from the single time value
    assert torch.allclose(out[:, 2], t.squeeze().expand(50))
