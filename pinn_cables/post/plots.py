"""Visualisation utilities for PINN results.

All ``plot_*`` functions accept an optional *save_path*; when provided the
figure is saved and ``plt.close()`` is called (useful in non-interactive
scripts).  When *save_path* is ``None`` the figure is displayed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from pinn_cables.io.readers import CableLayer, CablePlacement, Domain2D


def _finish(save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Temperature field
# ---------------------------------------------------------------------------

def plot_temperature_field(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    title: str = "Temperature field [K]",
    save_path: str | Path | None = None,
) -> None:
    """Filled-contour plot of a 2-D temperature field.

    Args:
        X, Y: Mesh-grid coordinate arrays ``(ny, nx)``.
        T:    Temperature array ``(ny, nx)``.
        title: Plot title.
        save_path: File path to save the figure (``None`` to show).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(X, Y, T, levels=50, cmap="hot")
    fig.colorbar(cf, ax=ax, label="T [K]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    _finish(save_path)


# ---------------------------------------------------------------------------
# Loss history
# ---------------------------------------------------------------------------

def plot_loss_history(
    history: dict[str, list[float]],
    title: str = "Training loss",
    save_path: str | Path | None = None,
) -> None:
    """Semi-log plot of loss components vs.\ iteration.

    Args:
        history: Dict of loss lists (key = component name).
        title:   Plot title.
        save_path: Save path or ``None``.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, vals in history.items():
        ax.semilogy(vals, label=name, linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    _finish(save_path)


# ---------------------------------------------------------------------------
# Comparison & error
# ---------------------------------------------------------------------------

def plot_comparison(
    X: np.ndarray, Y: np.ndarray,
    T_pinn: np.ndarray, T_ref: np.ndarray,
    title: str = "PINN vs Reference",
    save_path: str | Path | None = None,
) -> None:
    """Side-by-side contour comparison of PINN prediction and reference.

    Args:
        X, Y:    Grid arrays.
        T_pinn:  PINN temperature.
        T_ref:   Reference temperature.
        title:   Super-title.
        save_path: Save path or ``None``.
    """
    vmin = min(T_pinn.min(), T_ref.min())
    vmax = max(T_pinn.max(), T_ref.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cf1 = ax1.contourf(X, Y, T_pinn, levels=50, cmap="hot", vmin=vmin, vmax=vmax)
    fig.colorbar(cf1, ax=ax1, label="T [K]")
    ax1.set_title("PINN")
    ax1.set_aspect("equal")

    cf2 = ax2.contourf(X, Y, T_ref, levels=50, cmap="hot", vmin=vmin, vmax=vmax)
    fig.colorbar(cf2, ax=ax2, label="T [K]")
    ax2.set_title("Reference")
    ax2.set_aspect("equal")

    fig.suptitle(title)
    _finish(save_path)


def plot_error_field(
    X: np.ndarray, Y: np.ndarray,
    T_pinn: np.ndarray, T_ref: np.ndarray,
    title: str = "Absolute error",
    save_path: str | Path | None = None,
) -> None:
    """Contour map of the absolute point-wise error.

    Args:
        X, Y:    Grid arrays.
        T_pinn:  PINN temperature.
        T_ref:   Reference temperature.
        title:   Plot title.
        save_path: Save path or ``None``.
    """
    err = np.abs(T_pinn - T_ref)
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(X, Y, err, levels=50, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="|Error| [K]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    _finish(save_path)


# ---------------------------------------------------------------------------
# Cable geometry
# ---------------------------------------------------------------------------

def plot_cable_geometry(
    layers: Sequence[CableLayer],
    placement: CablePlacement,
    domain: Domain2D,
    title: str = "Cable geometry",
    save_path: str | Path | None = None,
) -> None:
    """Draw the cable cross-section inside the domain rectangle.

    Args:
        layers:    Cable layers.
        placement: Cable centre.
        domain:    Computational domain.
        title:     Plot title.
        save_path: Save path or ``None``.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    # Domain rectangle
    rect = plt.Rectangle(
        (domain.xmin, domain.ymin),
        domain.xmax - domain.xmin,
        domain.ymax - domain.ymin,
        linewidth=1.5, edgecolor="black", facecolor="bisque", alpha=0.3,
    )
    ax.add_patch(rect)

    # Layers as circles
    colors = plt.cm.Set2(np.linspace(0, 1, len(layers)))  # type: ignore[attr-defined]
    for layer, color in zip(layers, colors):
        circle = plt.Circle(
            (placement.cx, placement.cy), layer.r_outer,
            linewidth=1.2, edgecolor=color, facecolor="none", label=layer.name,
        )
        ax.add_patch(circle)

    ax.set_xlim(domain.xmin - 0.1, domain.xmax + 0.1)
    ax.set_ylim(domain.ymin - 0.1, domain.ymax + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    _finish(save_path)
