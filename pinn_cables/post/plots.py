"""Utilidades de visualización para resultados PINN.

Todas las funciones ``plot_*`` aceptan un *save_path* opcional;
si se proporciona, la figura se guarda y se cierra ``plt.close()``
(útil en scripts no interactivos).  Sin *save_path* se muestra la figura.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    """Semi-log plot of loss components vs. iteration.

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


# ---------------------------------------------------------------------------
# Zoom de temperatura alrededor de conductores
# ---------------------------------------------------------------------------

def plot_zoom_temperature(
    model,
    domain: Domain2D,
    placements: list,
    layers_list: list[list],
    device: torch.device,
    normalize: bool,
    save_path: str | Path | None = None,
    *,
    margin: float = 0.20,
    nx: int = 300,
    ny: int = 300,
    zoom: tuple[float, float, float, float] | None = None,
    pp=None,
    celsius: bool = False,
    annotate_max: bool = False,
    cable_labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Mapa de temperatura con zoom alrededor de los conductores.

    Si *zoom* es ``None``, los limites se calculan automaticamente a
    partir de las posiciones de los cables ± *margin*.

    Args:
        model:       Modelo PINN entrenado.
        domain:      Dominio computacional.
        placements:  Posiciones de los cables.
        layers_list: Capas por cable (para dibujar circulos).
        device:      Dispositivo torch.
        normalize:   Si las coordenadas se normalizan a [-1, 1].
        save_path:   Ruta para guardar la figura.
        margin:      Margen alrededor de los cables [m].
        zoom:        Limites fijos (xmin, xmax, ymin, ymax).
        pp:          PhysicsParams — si se proporciona, dibuja la zona
                     PAC/suelo mejorado.
        celsius:     Mostrar temperatura en °C en vez de K.
        annotate_max: Anotar el punto de T maxima con flecha.
        cable_labels: Etiquetas para cada cable.
        title:       Titulo del grafico.
    """
    # --- Limites del zoom ---
    if zoom is not None:
        x0, x1, y0, y1 = zoom
    else:
        x_centers = [pl.cx for pl in placements]
        y_centers = [pl.cy for pl in placements]
        x0 = min(x_centers) - margin
        x1 = max(x_centers) + margin
        y0 = min(y_centers) - margin
        y1 = max(y_centers) + margin

    # --- Evaluar modelo en la grilla de zoom ---
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    xy_flat = torch.tensor(
        np.column_stack([X.ravel(), Y.ravel()]),
        dtype=torch.float32, device=device,
    )
    if normalize:
        cmins = torch.tensor([domain.xmin, domain.ymin], device=device, dtype=torch.float32)
        cmaxs = torch.tensor([domain.xmax, domain.ymax], device=device, dtype=torch.float32)
        xy_in = 2.0 * (xy_flat - cmins) / (cmaxs - cmins) - 1.0
    else:
        xy_in = xy_flat

    with torch.no_grad():
        T_raw = model(xy_in).cpu().numpy().reshape(ny, nx)

    T_plot = T_raw - 273.15 if celsius else T_raw
    t_label = "T [°C]" if celsius else "T [K]"

    fig, ax = plt.subplots(figsize=(10, 8) if pp else (8, 7))
    cmap = "hot" if celsius else "hot_r"
    cf = ax.contourf(X, Y, T_plot, levels=60, cmap=cmap)
    fig.colorbar(cf, ax=ax, label=t_label, shrink=0.85)

    # Isotermas suaves
    if celsius:
        t_min_c, t_max_c = T_plot.min(), T_plot.max()
        step = max(1, int((t_max_c - t_min_c) / 30))
        lvls = np.arange(int(t_min_c), int(t_max_c) + 1, step)
        if len(lvls) > 2:
            cs = ax.contour(X, Y, T_plot, levels=lvls, colors="white",
                            linewidths=0.4, alpha=0.5)
            ax.clabel(cs, cs.levels[::2], fontsize=6, fmt="%.0f", colors="white")

    # --- Zona PAC/suelo mejorado (si se proporciona pp) ---
    if pp is not None and getattr(pp, "k_variable", False):
        rect = plt.Rectangle(
            (pp.k_cx - pp.k_width / 2.0, pp.k_cy - pp.k_height / 2.0),
            pp.k_width, pp.k_height,
            linewidth=2, edgecolor="lime", facecolor="none", linestyle="--",
            label="k=%.3f W/(m·K)" % pp.k_good,
        )
        ax.add_patch(rect)

    # --- Cables ---
    for idx, pl in enumerate(placements):
        r_sh = layers_list[idx][-1].r_outer
        if pp is not None:
            # Modo PAC: solo circulo de sheath
            circle = plt.Circle(
                (pl.cx, pl.cy), r_sh, facecolor="none",
                edgecolor="cyan", linewidth=1, linestyle="--",
            )
            ax.add_patch(circle)
        else:
            # Modo detallado: dibujar todas las capas
            layer_colors = ["white", "#aaddff", "#88bbff", "#4472C4"]
            for li, layer in enumerate(reversed(layers_list[idx])):
                lc = layer_colors[li % len(layer_colors)]
                lw = 1.2 if li == 0 else 0.6
                c = plt.Circle(
                    (pl.cx, pl.cy), layer.r_outer,
                    fill=False, edgecolor=lc, linewidth=lw, alpha=0.9, zorder=5,
                )
                ax.add_patch(c)
            ax.plot(pl.cx, pl.cy, "w+", ms=5, zorder=6)

        # Etiquetas de cable
        if cable_labels is not None and idx < len(cable_labels):
            ax.text(
                pl.cx, pl.cy + r_sh + 0.03, cable_labels[idx],
                fontsize=7, ha="center", va="bottom",
                color="cyan" if pp else "white", fontweight="bold",
            )

    # --- Anotacion T_max ---
    if annotate_max:
        idx_max = np.unravel_index(T_plot.argmax(), T_plot.shape)
        T_max_v = T_plot[idx_max]
        x_max, y_max = X[idx_max], Y[idx_max]
        ax.plot(x_max, y_max, "v", color="lime", markersize=8, markeredgecolor="white")
        unit = "°C" if celsius else "K"
        ax.annotate(
            "T_max = %.1f %s" % (T_max_v, unit), (x_max, y_max),
            xytext=(x_max + 0.3, y_max + 0.15), fontsize=8, fontweight="bold",
            color="lime", arrowprops=dict(arrowstyle="->", color="lime", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title, fontsize=11)
    if pp is not None and getattr(pp, "k_variable", False):
        ax.legend(loc="upper right", fontsize=8)
    elif pp is None and layers_list:
        layer_names = [la.name for la in reversed(layers_list[0])]
        layer_colors = ["white", "#aaddff", "#88bbff", "#4472C4"]
        patches = [
            mpatches.Patch(facecolor="none", edgecolor=layer_colors[i], label=layer_names[i])
            for i in range(min(len(layer_names), len(layer_colors)))
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8)
    plt.tight_layout()
    _finish(save_path)


# ---------------------------------------------------------------------------
# Campo de conductividad termica k(x,y)
# ---------------------------------------------------------------------------

def plot_k_field(
    domain: Domain2D,
    pp,
    placements: list,
    layers_list: list[list],
    save_path: str | Path | None = None,
    *,
    k_soil_base: float | None = None,
    title: str | None = None,
) -> None:
    """Mapa del campo de conductividad termica k(x,y).

    Dibuja la zona de suelo mejorado (PAC/backfill) definida por
    *pp.k_variable=True* con transicion sigmoide, junto con las
    posiciones de los cables.

    Args:
        domain:      Dominio computacional.
        pp:          PhysicsParams con config de zona k(x,y).
        placements:  Posiciones de cables.
        layers_list: Capas por cable.
        save_path:   Ruta para guardar.
        k_soil_base: k del suelo base (para caso no-variable).
        title:       Titulo del grafico.
    """
    if k_soil_base is None:
        k_soil_base = pp.k_bad

    nx, ny = 300, 200
    xs = np.linspace(domain.xmin, domain.xmax, nx)
    ys = np.linspace(domain.ymin, domain.ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    if pp.k_variable:
        dx_np = np.abs(X - pp.k_cx) - pp.k_width / 2.0
        dy_np = np.abs(Y - pp.k_cy) - pp.k_height / 2.0
        d_np = np.maximum(dx_np, dy_np)
        sig = 1.0 / (1.0 + np.exp(d_np / pp.k_transition))
        K_field = pp.k_bad + (pp.k_good - pp.k_bad) * sig
    else:
        K_field = np.full_like(X, k_soil_base)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin = min(pp.k_bad, k_soil_base) * 0.95
    vmax = max(pp.k_good, k_soil_base) * 1.05
    cf = ax.contourf(X, Y, K_field, levels=40, cmap="YlOrRd_r", vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax, label="k suelo [W/(m·K)]")

    # Cables como circulos solidos negros
    for idx, pl in enumerate(placements):
        r_sh = layers_list[idx][-1].r_outer
        circle = plt.Circle(
            (pl.cx, pl.cy), r_sh,
            fill=True, facecolor="black", edgecolor="white", linewidth=1.5, zorder=5,
        )
        ax.add_patch(circle)

    # Rectangulo de la zona mejorada
    if pp.k_variable:
        box = plt.Rectangle(
            (pp.k_cx - pp.k_width / 2.0, pp.k_cy - pp.k_height / 2.0),
            pp.k_width, pp.k_height,
            fill=False, edgecolor="white", linestyle="--", linewidth=1.5, zorder=6,
        )
        ax.add_patch(box)
        ax.text(
            pp.k_cx, pp.k_cy + pp.k_height / 2.0 + 0.08,
            "k = %.1f W/(m·K)" % pp.k_good,
            ha="center", va="bottom", color="white", fontsize=9, zorder=7,
        )
        ax.text(
            pp.k_cx, domain.ymin + 0.15,
            "k = %.1f W/(m·K)" % pp.k_bad,
            ha="center", va="bottom", color="black", fontsize=9, zorder=7,
        )

    ax.set_xlim(domain.xmin, domain.xmax)
    ax.set_ylim(domain.ymin, domain.ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or "k(x,y) suelo [W/(m·K)]")
    plt.tight_layout()
    _finish(save_path)


# ---------------------------------------------------------------------------
# Geometria multi-cable (trefoil, flat, two-flat, etc.)
# ---------------------------------------------------------------------------

def plot_geometry_multicable(
    layers_list: list[list],
    placements: list,
    domain: Domain2D,
    title: str = "Sección transversal de cables",
    save_path: str | Path | None = None,
    *,
    circuit_size: int = 1,
) -> None:
    """Sección transversal mostrando N cables con sus capas.

    Agrupa cables en circuitos de *circuit_size* para asignar colores.
    Con ``circuit_size=1`` cada cable tiene su propio color.
    Con ``circuit_size=3`` se agrupan en trios (típico trefoil).

    Args:
        layers_list: Capas por cable.
        placements:  Posiciones de cables.
        domain:      Dominio para el rectángulo exterior.
        title:       Titulo.
        save_path:   Ruta para guardar.
        circuit_size: Cantidad de cables por circuito para coloreo.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red",
               "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    layer_fill = ["#888888", "#aaaaaa", "#cccccc", "#4472C4"]

    for idx, pl in enumerate(placements):
        layers_i = layers_list[idx]
        # Color por circuito
        circ_idx = idx // max(1, circuit_size)
        color = palette[circ_idx % len(palette)]

        for li, (layer, lcol) in enumerate(zip(reversed(layers_i), reversed(layer_fill))):
            ring = plt.Circle((pl.cx, pl.cy), layer.r_outer, color=lcol, zorder=5)
            ax.add_patch(ring)

        # Etiqueta
        r_outer = layers_i[-1].r_outer
        if circuit_size > 1:
            circ_num = circ_idx + 1
            label = "C%d-%d" % (circ_num, pl.cable_id)
        else:
            label = "Cable %d" % pl.cable_id
        ax.text(
            pl.cx, pl.cy + r_outer * 1.8,
            label, ha="center", va="bottom", fontsize=8, color=color, zorder=10,
        )

    # Rectangulo del dominio
    rect = plt.Rectangle(
        (domain.xmin, domain.ymin),
        domain.xmax - domain.xmin, domain.ymax - domain.ymin,
        fill=False, edgecolor="black", linewidth=1.5, zorder=1,
    )
    ax.add_patch(rect)

    # Zoom automatico
    margin = 0.15
    cx_min = min(pl.cx for pl in placements) - margin
    cx_max = max(pl.cx for pl in placements) + margin
    cy_min = min(pl.cy for pl in placements) - margin
    cy_max = max(pl.cy for pl in placements) + margin
    ax.set_xlim(cx_min, cx_max)
    ax.set_ylim(cy_min, cy_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)

    legend_patches = [
        mpatches.Patch(color="#4472C4", label="Conductor Cu"),
        mpatches.Patch(color="#cccccc", label="Aislante XLPE"),
        mpatches.Patch(color="#aaaaaa", label="Pantalla"),
        mpatches.Patch(color="#888888", label="Cubierta (sheath)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    plt.tight_layout()
    _finish(save_path)


# ---------------------------------------------------------------------------
# Grafico de barras comparativo entre metodos
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    results: dict[str, float],
    save_path: str | Path | None = None,
    *,
    ref_line: tuple[float, str] | None = None,
    limit_line: tuple[float, str] | None = None,
    title: str = "Comparación T_max entre métodos",
) -> None:
    """Grafico de barras horizontales comparando T_max [°C] de distintos metodos.

    Args:
        results:    Dict ``{nombre_metodo: T_K}`` (temperatura en K).
        save_path:  Ruta para guardar.
        ref_line:   (valor_C, etiqueta) para linea de referencia.
        limit_line: (valor_C, etiqueta) para linea de limite.
        title:      Titulo del grafico.
    """
    labels = list(results.keys())
    temps = [v - 273.15 for v in results.values()]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4",
              "#795548", "#607D8B"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, temps,
                   color=[colors[i % len(colors)] for i in range(len(labels))],
                   edgecolor="black", height=0.5)

    if ref_line is not None:
        ax.axvline(ref_line[0], color="red", linestyle="--", linewidth=1.5,
                   label=ref_line[1])
    if limit_line is not None:
        ax.axvline(limit_line[0], color="darkred", linestyle=":", linewidth=1.5,
                   label=limit_line[1])

    for bar, t in zip(bars, temps):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                "%.1f °C" % t, va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("T_max conductor [°C]")
    ax.set_title(title)
    if ref_line or limit_line:
        ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _finish(save_path)
