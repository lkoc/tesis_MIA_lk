"""Geometry sampling for concentric cable layers inside a rectangular domain.

Generates collocation points for:
- interior regions (each cable layer + soil) via rejection sampling,
- interfaces (angular bands around each layer boundary),
- domain boundaries (rectangle edges),
- space-time points (transient mode).

All returned tensors have ``requires_grad=True`` so that PyTorch autograd can
compute PDE residuals.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from pinn_cables.io.readers import CableLayer, CablePlacement, Domain2D


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _rand_uniform(
    n: int, lo: float, hi: float, device: torch.device | None = None,
) -> torch.Tensor:
    """Uniform random column vector in [lo, hi)."""
    return lo + (hi - lo) * torch.rand((n, 1), device=device)


def _in_annulus(
    xy: torch.Tensor, cx: float, cy: float, r_in: float, r_out: float,
) -> torch.Tensor:
    """Boolean mask: points inside annulus ``r_in <= r < r_out``."""
    dx = xy[:, 0:1] - cx
    dy = xy[:, 1:2] - cy
    r2 = dx * dx + dy * dy
    return (r2 >= r_in * r_in) & (r2 < r_out * r_out)


def _in_circle(
    xy: torch.Tensor, cx: float, cy: float, r: float,
) -> torch.Tensor:
    """Boolean mask: points inside circle of radius *r*."""
    dx = xy[:, 0:1] - cx
    dy = xy[:, 1:2] - cy
    return (dx * dx + dy * dy) < (r * r)


def _take(
    xy: torch.Tensor, mask: torch.Tensor, n: int, device: torch.device | None,
) -> torch.Tensor:
    """Select up to *n* points where *mask* is True (with replacement if needed)."""
    idx = torch.where(mask.squeeze(1))[0]
    if idx.numel() == 0:
        return torch.empty((0, 2), device=device)
    if idx.numel() < n:
        ridx = idx[torch.randint(0, idx.numel(), (n,), device=device)]
    else:
        ridx = idx[torch.randperm(idx.numel(), device=device)[:n]]
    return xy[ridx]


# ---------------------------------------------------------------------------
# Allocation of interior points by region area
# ---------------------------------------------------------------------------

def _compute_region_counts(
    layers: Sequence[CableLayer],
    domain: Domain2D,
    cable_r_outer: float,
    n_total: int,
    min_per_region: int = 20,
) -> list[int]:
    """Distribute *n_total* interior points proportionally to region area.

    Returns a list of counts: one per cable layer plus one for soil.
    """
    domain_area = (domain.xmax - domain.xmin) * (domain.ymax - domain.ymin)
    areas: list[float] = []
    for layer in layers:
        areas.append(math.pi * (layer.r_outer**2 - layer.r_inner**2))
    soil_area = domain_area - math.pi * cable_r_outer**2
    areas.append(max(soil_area, 0.01))

    total_area = sum(areas)
    counts = [max(min_per_region, int(a / total_area * n_total)) for a in areas]

    # Adjust last (soil) to match n_total exactly
    counts[-1] = max(min_per_region, n_total - sum(counts[:-1]))
    return counts


# ---------------------------------------------------------------------------
# Direct annulus sampling for thin layers
# ---------------------------------------------------------------------------

def _sample_annulus_direct(
    cx: float, cy: float,
    r_inner: float, r_outer: float,
    n: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample *n* points uniformly inside an annulus using polar coordinates.

    For layers that are too thin for rejection sampling to be efficient.
    """
    angles = 2.0 * math.pi * torch.rand(n, 1, device=device)
    # Uniform in area ⇒ r = sqrt(U * (r_out^2 - r_in^2) + r_in^2)
    u = torch.rand(n, 1, device=device)
    r = torch.sqrt(u * (r_outer**2 - r_inner**2) + r_inner**2)
    x = cx + r * torch.cos(angles)
    y = cy + r * torch.sin(angles)
    return torch.cat([x, y], dim=1)


# ---------------------------------------------------------------------------
# Main sampling functions
# ---------------------------------------------------------------------------

def sample_domain_points(
    domain: Domain2D,
    layers: Sequence[CableLayer],
    placement: CablePlacement,
    n_interior: int,
    n_interface: int,
    oversample: int = 5,
    device: torch.device | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Sample interior and interface collocation points (single cable).

    Args:
        domain:      Rectangular computational domain.
        layers:      Cable layers from inner to outer.
        placement:   Cable centre position.
        n_interior:  Total number of interior points across all regions.
        n_interface: Points per interface ring.
        oversample:  Over-sampling factor for rejection sampling.
        device:      Torch device.

    Returns:
        ``(interior_pts, interface_pts)`` where each is a dict mapping
        region/interface names to ``(N, 2)`` tensors with
        ``requires_grad=True``.
    """
    cx, cy = placement.cx, placement.cy
    r_outer_max = layers[-1].r_outer

    counts = _compute_region_counts(
        layers, domain, r_outer_max, n_interior,
    )

    # Generate candidate pool
    n_try = n_interior * oversample
    x = _rand_uniform(n_try, domain.xmin, domain.xmax, device=device)
    y = _rand_uniform(n_try, domain.ymin, domain.ymax, device=device)
    xy = torch.cat([x, y], dim=1)

    # Build masks per region
    masks: list[torch.Tensor] = []
    for layer in layers:
        if layer.r_inner == 0.0:
            masks.append(_in_circle(xy, cx, cy, layer.r_outer))
        else:
            masks.append(_in_annulus(xy, cx, cy, layer.r_inner, layer.r_outer))
    masks.append(~_in_circle(xy, cx, cy, r_outer_max))  # soil

    interior: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(layers):
        got = _take(xy, masks[i], counts[i], device)
        if got.shape[0] < counts[i]:
            # Thin-layer fallback: sample directly in annulus using polar coords
            extra = _sample_annulus_direct(
                cx, cy, layer.r_inner, layer.r_outer, counts[i], device,
            )
            got = extra
        interior[layer.name] = got.clone().detach().requires_grad_(True)
    soil_pts = _take(xy, masks[-1], counts[-1], device)
    interior["soil"] = soil_pts.clone().detach().requires_grad_(True)

    # Interface rings (angular sampling + radial perturbation)
    eps = 0.002 * r_outer_max
    angles = 2.0 * math.pi * torch.rand((n_interface, 1), device=device)

    interfaces: dict[str, torch.Tensor] = {}
    for layer in layers:
        r = layer.r_outer
        rr = r + eps * (2.0 * torch.rand((n_interface, 1), device=device) - 1.0)
        ix = cx + rr * torch.cos(angles)
        iy = cy + rr * torch.sin(angles)
        pts = torch.cat([ix, iy], dim=1)
        interfaces[f"r_{layer.name}"] = pts.clone().detach().requires_grad_(True)

    return interior, interfaces


def sample_boundary_points(
    domain: Domain2D,
    n_boundary: int,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Sample points on the four edges of the rectangular domain.

    Args:
        domain:     Rectangular domain.
        n_boundary: Total number of boundary points (split among edges).
        device:     Torch device.

    Returns:
        Dict mapping edge names to ``(n, 2)`` tensors with
        ``requires_grad=True``.
    """
    n_per = max(1, n_boundary // 4)
    result: dict[str, torch.Tensor] = {}

    # Top & bottom (horizontal edges)
    xh = _rand_uniform(n_per, domain.xmin, domain.xmax, device=device)
    result["top"] = torch.cat(
        [xh, torch.full_like(xh, domain.ymax)], dim=1,
    ).clone().detach().requires_grad_(True)

    xh2 = _rand_uniform(n_per, domain.xmin, domain.xmax, device=device)
    result["bottom"] = torch.cat(
        [xh2, torch.full_like(xh2, domain.ymin)], dim=1,
    ).clone().detach().requires_grad_(True)

    # Left & right (vertical edges)
    yv = _rand_uniform(n_per, domain.ymin, domain.ymax, device=device)
    result["left"] = torch.cat(
        [torch.full_like(yv, domain.xmin), yv], dim=1,
    ).clone().detach().requires_grad_(True)

    yv2 = _rand_uniform(n_per, domain.ymin, domain.ymax, device=device)
    result["right"] = torch.cat(
        [torch.full_like(yv2, domain.xmax), yv2], dim=1,
    ).clone().detach().requires_grad_(True)

    return result


def sample_time(
    n_t: int,
    t0: float,
    t1: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Uniformly sample time values in ``[t0, t1]``.

    Returns:
        Tensor of shape ``(n_t, 1)`` with ``requires_grad=True``.
    """
    return _rand_uniform(n_t, t0, t1, device=device).requires_grad_(True)


def sample_initial_condition(
    domain: Domain2D,
    n_ic: int,
    t0: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample points for the initial-condition loss (t = t0).

    Returns:
        Tensor of shape ``(n_ic, 3)`` with columns ``[x, y, t0]`` and
        ``requires_grad=True``.
    """
    x = _rand_uniform(n_ic, domain.xmin, domain.xmax, device=device)
    y = _rand_uniform(n_ic, domain.ymin, domain.ymax, device=device)
    t = torch.full((n_ic, 1), t0, device=device)
    return torch.cat([x, y, t], dim=1).requires_grad_(True)


def append_time(
    xy: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Broadcast-concatenate spatial points with a time column.

    Args:
        xy: ``(N, 2)`` spatial points.
        t:  ``(M, 1)`` time values **or** ``(N, 1)`` matched times.

    Returns:
        ``(N, 3)`` tensor ``[x, y, t]`` with ``requires_grad=True``.
    """
    if t.shape[0] == 1:
        t = t.expand(xy.shape[0], 1)
    elif t.shape[0] != xy.shape[0]:
        # Random assignment: pair each spatial point with a random time
        idx = torch.randint(0, t.shape[0], (xy.shape[0],), device=xy.device)
        t = t[idx]
    return torch.cat([xy[:, :2], t], dim=1).requires_grad_(True)
