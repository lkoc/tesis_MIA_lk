"""Gráficos de publicación — Comparación FEM FEniCSx vs modelos PINN.

Kim 2024 Caso B: 6 cables XLPE 154 kV, suelo multicapa + zona PAC.
Zona de análisis (zoom): x ∈ [−1.5, 1.5] m, y ∈ [−2.5, −0.5] m.

Figuras generadas (PDF + PNG, 300 dpi) en results_fem/:
  fig01_T_field_zoom.pdf/png   — Campos de temperatura T [°C]
  fig02_dT_signed_zoom.pdf/png — Diferencia con signo ΔT = T_PINN − T_FEM [K]
  fig03_dT_abs_zoom.pdf/png    — Error absoluto |ΔT| [K]

Características:
  • Interpolación bilineal del campo FEM sobre grilla regular 500×350
  • Isotermas del campo FEM cada 5 °C (líneas finas, etiquetadas, no dominantes)
  • Contornos punteados de límites de bandas de suelo (y = −0.56 m, −1.76 m)
  • Rectángulo punteado de la zona PAC
  • Circunferencias del conductor Cu (dorado) y cubierta exterior (cian)
  • Anotación de máximo error local en figuras ΔT
  • Leyenda geométrica compartida al pie de cada figura
  • rcParams de publicación: serif 9 pt, ejes limpios, PDF Type-42

Uso (desde la raíz del proyecto)::

    python examples/kim_2024_154kv_optim_C/results_fem/plot_zoom_comparison.py
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.interpolate import LinearNDInterpolator

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
HERE      = Path(__file__).resolve().parent           # results_fem/
OPT_C_DIR = HERE.parent                               # kim_2024_154kv_optim_C/
ROOT      = HERE.parent.parent.parent                 # raíz del proyecto

DATA_DIR     = HERE.parent.parent / "kim_2024_154kv_optim_B" / "data"
FEM_NPZ      = HERE / "fem_field_kim2024B.npz"

# Default: original T_amb=290.15 K models
MODEL_64x4   = (HERE.parent.parent / "kim_2024_154kv_optim_B"
                / "results_optim" / "model_best_64x4.pt")
MODEL_128x5  = OPT_C_DIR / "results_optim"  / "model_best_128x5.pt"
MODEL_DISTIL = OPT_C_DIR / "results_distil" / "model_best_128x5_distil.pt"

# Corrected T_amb=300.30 K models
MODEL_64x4_CORR   = (HERE.parent.parent / "kim_2024_154kv_optim_B"
                     / "results_corrected" / "model_best_64x4.pt")
MODEL_128x5_CORR  = OPT_C_DIR / "results_corrected" / "model_best_128x5.pt"
MODEL_DISTIL_CORR = OPT_C_DIR / "results_corrected" / "model_best_128x5_distil.pt"

sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constantes geométricas (Kim 2024, Caso B)
# ---------------------------------------------------------------------------
CABLES = [
    (-0.40, -1.60), (0.00, -1.60), (0.40, -1.60),
    (-0.40, -1.20), (0.00, -1.20), (0.40, -1.20),
]
R_CONDUCTOR = 0.01885   # m
R_CABLE     = 0.05335   # m

SOIL_BANDS = [
    ( 0.000, -0.560, 1.804),
    (-0.560, -1.760, 1.351),
    (-1.760, -45.50, 1.517),
]

PAC_CX, PAC_CY = 0.0, -1.40
PAC_W,  PAC_H  = 1.30, 0.90
PAC_K          = 2.094

ZOOM_XMIN, ZOOM_XMAX = -1.50,  1.50
ZOOM_YMIN, ZOOM_YMAX = -2.50, -0.50

FEM_REF_C       = 70.6
PAPER_T_MAX_PAC = 273.15 + FEM_REF_C
PAPER_W_D       = 3.57
PAPER_FREQ      = 60.0

# Derived PAC bounds
PAC_X0 = PAC_CX - PAC_W / 2   # -0.65 m
PAC_Y0 = PAC_CY - PAC_H / 2   # -1.85 m

# ---------------------------------------------------------------------------
# Estilo de publicación
# ---------------------------------------------------------------------------
def _set_pub_style() -> None:
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":        9,
        "axes.titlesize":   9,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  7.5,
        "lines.linewidth":  1.0,
        "axes.linewidth":   0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "axes.grid":        False,
        "pdf.fonttype":     42,   # TrueType → editable en Illustrator/Inkscape
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "figure.dpi":       100,
    })


# ---------------------------------------------------------------------------
# 1. Carga del campo FEM
# ---------------------------------------------------------------------------
def load_fem() -> dict:
    if not FEM_NPZ.exists():
        raise FileNotFoundError(
            f"Campo FEM no encontrado: {FEM_NPZ}\n"
            "Ejecuta fem_fenicsx_colab.ipynb (Celda 8) y copia el .npz aquí."
        )
    d = np.load(FEM_NPZ)
    print(f"  ✓ FEM: {d['coords_xy'].shape[0]:,d} nodos, "
          f"{d['cells'].shape[0]:,d} triángulos")
    print(f"    T_C ∈ [{d['T_C'].min():.2f}, {d['T_C'].max():.2f}] °C")
    return dict(
        coords = d["coords_xy"],           # (N, 2)  float64
        T_C    = d["T_C"],                 # (N,)    °C
        cells  = d["cells"].astype(np.int32),  # (M, 3)
    )


# ---------------------------------------------------------------------------
# 2. Grilla regular en zona zoom
# ---------------------------------------------------------------------------
def build_zoom_grid(nx: int = 480, ny: int = 320):
    x = np.linspace(ZOOM_XMIN, ZOOM_XMAX, nx)
    y = np.linspace(ZOOM_YMIN, ZOOM_YMAX, ny)
    XX, YY = np.meshgrid(x, y)
    return x, y, XX, YY


# ---------------------------------------------------------------------------
# 3. Interpolación FEM en grilla (sólo nodos en zona zoom → rápido)
# ---------------------------------------------------------------------------
def interpolate_fem_on_grid(fem: dict, XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
    """Interpola el campo FEM en la grilla regular del zoom.

    Estrategia eficiente: extrae sólo los nodos dentro de la caja zoom
    (más un buffer de 0.3 m) y construye el interpolador sólo sobre ese
    subconjunto, en lugar de usar los ~millones de nodos del dominio completo.
    """
    print("  Interpolando FEM en grilla …", end=" ", flush=True)
    t0 = time.time()
    buf = 0.30   # buffer [m] para evitar artefactos de borde
    x, y, T = fem["coords"][:, 0], fem["coords"][:, 1], fem["T_C"]
    mask = (
        (x >= ZOOM_XMIN - buf) & (x <= ZOOM_XMAX + buf) &
        (y >= ZOOM_YMIN - buf) & (y <= ZOOM_YMAX + buf)
    )
    n_sub = int(mask.sum())
    print(f"  ({n_sub:,d}/{len(x):,d} nodos en zoom+buffer) …", end=" ", flush=True)

    pts_sub = np.stack([x[mask], y[mask]], axis=1)
    T_sub   = T[mask]

    # LinearNDInterpolator construye una triangulación Delaunay local
    interp  = LinearNDInterpolator(pts_sub, T_sub)
    T_grid  = np.asarray(interp(XX, YY), dtype=np.float64)

    print(f"OK ({time.time()-t0:.1f}s)  "
          f"T ∈ [{np.nanmin(T_grid):.1f}, {np.nanmax(T_grid):.1f}] °C")
    return T_grid


# ---------------------------------------------------------------------------
# 4. Preparar problema PINN compartido
# ---------------------------------------------------------------------------
def build_shared_problem() -> dict:
    from pinn_cables.io.readers import load_problem
    from pinn_cables.materials.props import get_kim2024_cable_layers
    from pinn_cables.physics.iec60287 import compute_iec60287_Q
    from pinn_cables.physics.k_field import load_physics_params, load_soil_layers

    problem         = load_problem(DATA_DIR)
    all_placements  = problem.placements
    n_cables        = len(all_placements)
    soil_bands      = load_soil_layers(DATA_DIR / "soil_layers.csv")
    pac_params_base = load_physics_params(DATA_DIR / "physics_params.csv")

    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A   = all_placements[0].current_A
    section_mm2 = all_placements[0].section_mm2

    iec_q  = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=PAPER_T_MAX_PAC, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_lin  = iec_q["Q_cond_W_per_m"] + PAPER_W_D
    layers = get_kim2024_cable_layers(Q_lin)

    return dict(
        problem         = problem,
        all_placements  = all_placements,
        n_cables        = n_cables,
        soil_bands      = soil_bands,
        pac_params_base = pac_params_base,
        layers_list     = [layers] * n_cables,
        Q_lins          = [Q_lin]  * n_cables,
        Q_d             = PAPER_W_D,
    )


# ---------------------------------------------------------------------------
# 5. Evaluación de modelos PINN en grilla
# ---------------------------------------------------------------------------
def eval_pinn_on_grid(
    ckpt_path: Path,
    shared: dict,
    device: torch.device,
    width: int,
    depth: int,
    XX: np.ndarray,
    YY: np.ndarray,
    batch_size: int = 60_000,
) -> np.ndarray | None:
    if not ckpt_path.exists():
        print(f"  AVISO: modelo no encontrado: {ckpt_path}")
        return None

    from pinn_cables.physics.k_field import KFieldModel
    from pinn_cables.pinn.model import ResidualPINNModel, build_model

    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = ckpt["params"]

    pac_params = dataclasses.replace(
        shared["pac_params_base"],
        k_transition=params.get("pac_transition", 0.20),
    )
    k_model = KFieldModel(
        k_soil=1.351,
        soil_bands=shared["soil_bands"],
        pac_params=pac_params,
        layer_transition=params.get("layer_transition", 0.10),
    )
    T_amb     = shared["problem"].scenarios[0].T_amb
    k_soil_bg = k_model.k_eff_bg(shared["all_placements"])

    use_ff = params.get("fourier_mapping_size", 0) > 0
    cfg = {
        "architecture":         "mlp",
        "width":                width,
        "depth":                depth,
        "activation":           "tanh",
        "fourier_features":     use_ff,
        "fourier_mapping_size": params.get("fourier_mapping_size", 64),
        "fourier_scale":        params.get("fourier_scale", 1.0),
    }
    base  = build_model(cfg, in_dim=2, device=device)
    model = ResidualPINNModel(
        base, shared["layers_list"], shared["all_placements"],
        k_soil_bg, T_amb, shared["Q_lins"],
        shared["problem"].domain, normalize=True, Q_d=shared["Q_d"],
        enable_grad_Tbg=True,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    dom   = shared["problem"].domain
    x_n   = (2.0 * (XX.ravel() - dom.xmin) / (dom.xmax - dom.xmin) - 1.0).astype(np.float32)
    y_n   = (2.0 * (YY.ravel() - dom.ymin) / (dom.ymax - dom.ymin) - 1.0).astype(np.float32)
    xy_np = np.stack([x_n, y_n], axis=1)

    T_out = np.empty(len(xy_np), dtype=np.float32)
    for start in range(0, len(xy_np), batch_size):
        end = min(start + batch_size, len(xy_np))
        with torch.no_grad():
            T_out[start:end] = (
                model(torch.from_numpy(xy_np[start:end]).to(device))
                .cpu().numpy().ravel()
            )

    T_grid = (T_out.astype(np.float64) - 273.15).reshape(XX.shape)
    print(f"    T_PINN ∈ [{T_grid.min():.2f}, {T_grid.max():.2f}] °C")
    return T_grid


# ---------------------------------------------------------------------------
# Helpers de decoración — isotermas, zonas, cables
# ---------------------------------------------------------------------------

def _isotherm_levels(T_grid: np.ndarray, step: float = 5.0) -> np.ndarray:
    tmin = np.nanmin(T_grid)
    tmax = np.nanmax(T_grid)
    return np.arange(np.ceil(tmin / step) * step, tmax, step)


def add_isotherms(
    ax,
    XX: np.ndarray,
    YY: np.ndarray,
    T_grid: np.ndarray,
    levels: np.ndarray,
    color: str = "white",
    lw: float = 0.45,
    alpha: float = 0.65,
    label_every: int = 2,
    fmt: str = "%d °C",
    fontsize: float = 5.5,
) -> None:
    """Dibuja isotermas sobre el eje; etiqueta 1 de cada `label_every`."""
    valid = levels[(levels > np.nanmin(T_grid)) & (levels < np.nanmax(T_grid))]
    if len(valid) < 2:
        return
    try:
        cs = ax.contour(
            XX, YY, T_grid, levels=valid,
            colors=color, linewidths=lw, alpha=alpha, zorder=3,
        )
        label_lev = valid[::label_every]
        ax.clabel(
            cs, levels=label_lev,
            inline=True, fmt=fmt,
            fontsize=fontsize, inline_spacing=1, zorder=4,
        )
    except Exception:
        pass


def add_zone_decorations(
    ax,
    add_labels: bool = True,
    alpha_lines: float = 0.65,
    lw_zone: float = 0.8,
) -> None:
    """Añade al eje: límites de suelo, PAC, circunferencias de cables."""

    # ── Colores ─────────────────────────────────────────────────────────────
    C_SOIL  = "#555555"
    C_PAC   = "#1a6faf"   # azul acero
    C_COND  = "#FFD700"   # dorado
    C_OUTER = "#29B6F6"   # cian

    # ── Bandas de suelo: líneas horizontales ─────────────────────────────────
    for y_bd, (y_top, y_bot, k_val) in zip(
        [SOIL_BANDS[0][1], SOIL_BANDS[1][1]],
        [(SOIL_BANDS[0][0], SOIL_BANDS[0][1], SOIL_BANDS[0][2]),
         (SOIL_BANDS[1][0], SOIL_BANDS[1][1], SOIL_BANDS[1][2])],
    ):
        if ZOOM_YMIN < y_bd < ZOOM_YMAX:
            ax.axhline(
                y=y_bd, color=C_SOIL, linestyle="--",
                linewidth=lw_zone, alpha=alpha_lines, zorder=5,
            )

    # Etiquetas de banda de suelo (margen izquierdo)
    if add_labels:
        soil_labels = [
            ((-0.56 + 0.00) / 2, f"Suelo 1\nk={SOIL_BANDS[0][2]:.3f} W/(m·K)"),
            ((-1.76 + -0.56) / 2, f"Suelo 2\nk={SOIL_BANDS[1][2]:.3f} W/(m·K)"),
            ((-2.50 + -1.76) / 2, f"Suelo 3\nk={SOIL_BANDS[2][2]:.3f} W/(m·K)"),
        ]
        for y_mid, lbl in soil_labels:
            if ZOOM_YMIN + 0.1 <= y_mid <= ZOOM_YMAX - 0.08:
                ax.text(
                    ZOOM_XMIN + 0.07, y_mid, lbl,
                    fontsize=5.8, color=C_SOIL, va="center", ha="left",
                    zorder=8, linespacing=1.35,
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white",
                        ec="none", alpha=0.60,
                    ),
                )

    # ── Zona PAC ─────────────────────────────────────────────────────────────
    pac_rect = mpatches.Rectangle(
        (PAC_X0, PAC_Y0), PAC_W, PAC_H,
        linewidth=lw_zone + 0.3,
        edgecolor=C_PAC,
        facecolor="none",
        linestyle=(0, (4, 2)),   # dash-dot fino
        alpha=alpha_lines + 0.1,
        zorder=6,
    )
    ax.add_patch(pac_rect)
    if add_labels:
        ax.text(
            PAC_CX, PAC_Y0 + 0.06,
            f"PAC  k = {PAC_K} W/(m·K)",
            fontsize=5.8, color=C_PAC, va="bottom", ha="center",
            zorder=9,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.65),
        )

    # ── Cables: conductor Cu + cubierta exterior ──────────────────────────────
    for idx, (cx, cy) in enumerate(CABLES):
        # Cubierta exterior (punteada, cian)
        outer = mpatches.Circle(
            (cx, cy), R_CABLE,
            linewidth=0.8, edgecolor=C_OUTER,
            facecolor="none", linestyle="--",
            alpha=0.75, zorder=7,
        )
        ax.add_patch(outer)
        # Conductor Cu (sólida, dorado)
        cond = mpatches.Circle(
            (cx, cy), R_CONDUCTOR,
            linewidth=1.0, edgecolor=C_COND,
            facecolor="none", linestyle="-",
            alpha=0.90, zorder=8,
        )
        ax.add_patch(cond)

    # Etiqueta de cable (solo para los dos de la izquierda como ejemplo)
    if add_labels:
        ax.text(
            CABLES[0][0], CABLES[0][1] - R_CABLE - 0.07,
            "Cu conductor", fontsize=5.2, color=C_COND,
            ha="center", va="top", zorder=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.60),
        )


def _style_ax(ax, xlabel: bool = False, ylabel: bool = False) -> None:
    ax.set_xlim(ZOOM_XMIN, ZOOM_XMAX)
    ax.set_ylim(ZOOM_YMIN, ZOOM_YMAX)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    if xlabel:
        ax.set_xlabel("$x$ [m]")
    if ylabel:
        ax.set_ylabel("$y$ [m]")


def _geometry_legend_handles() -> list:
    """Manejadores de leyenda para anotaciones geométricas."""
    return [
        mlines.Line2D([], [], color="#FFD700", lw=1.2, ls="-",
                      label=f"Conductor Cu (r = {R_CONDUCTOR*1e3:.1f} mm)"),
        mlines.Line2D([], [], color="#29B6F6", lw=0.9, ls="--",
                      label=f"Cubierta cable (r = {R_CABLE*1e3:.1f} mm)"),
        mlines.Line2D([], [], color="#1a6faf", lw=1.0, ls=(0, (4, 2)),
                      label=f"Zona PAC (k = {PAC_K} W/(m·K))"),
        mlines.Line2D([], [], color="#555555", lw=0.8, ls="--",
                      label="Límite banda de suelo"),
        mlines.Line2D([], [], color="white", lw=0.6, ls="-",
                      label="Isoterma FEM (c/5 °C)"),
    ]


def _add_footer_legend(fig) -> None:
    fig.legend(
        handles=_geometry_legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=True,
        framealpha=0.88,
        edgecolor="#bbbbbb",
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )


def _save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        p = HERE / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  ✓ {p.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURA 1 — Campos de temperatura T [°C]
# ---------------------------------------------------------------------------
def fig_T_fields(
    XX: np.ndarray, YY: np.ndarray,
    T_fem_grid: np.ndarray,
    pinn_results: list[dict],
    T_fem_max: float,
    suffix: str = "",
) -> None:
    valid = [r for r in pinn_results if r["T_grid"] is not None]
    n     = 1 + len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.0))
    if n == 1:
        axes = [axes]

    # Escala común de color
    all_T  = [T_fem_grid] + [r["T_grid"] for r in valid]
    vmin_T = min(np.nanmin(t) for t in all_T)
    vmax_T = max(np.nanmax(t) for t in all_T)
    iso    = _isotherm_levels(T_fem_grid, step=5.0)

    def _plot_panel(ax, T_grid, title, show_y: bool = False):
        pm = ax.pcolormesh(
            XX, YY, T_grid,
            cmap="plasma", vmin=vmin_T, vmax=vmax_T,
            shading="gouraud", rasterized=True, zorder=1,
        )
        add_isotherms(ax, XX, YY, T_grid, iso,
                      color="white", lw=0.45, alpha=0.65, label_every=2)
        add_zone_decorations(ax, add_labels=(ax is axes[0]))
        _style_ax(ax, xlabel=True, ylabel=show_y)
        ax.set_title(title, pad=5, fontsize=9)
        return pm

    T_fem_max_zoom = float(np.nanmax(T_fem_grid))
    pm = _plot_panel(
        axes[0], T_fem_grid,
        f"FEM FEniCSx  (ref.)\n$T_\\mathrm{{max}}$ = {T_fem_max_zoom:.2f} °C",
        show_y=True,
    )
    for i, r in enumerate(valid):
        T_max_r = float(np.nanmax(r["T_grid"]))
        dT_max  = T_max_r - FEM_REF_C
        _plot_panel(
            axes[i + 1], r["T_grid"],
            f"{r['label']}\n$T_\\mathrm{{max}}$ = {T_max_r:.2f} °C "
            f"($\\Delta$ = {dT_max:+.2f} K)",
        )

    cbar = fig.colorbar(pm, ax=list(axes), shrink=0.88, pad=0.015, aspect=30)
    cbar.set_label("$T$ [°C]", labelpad=5)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_major_locator(MultipleLocator(10))

    fig.suptitle(
        "Campo de temperatura — zona cables (Kim 2024 Caso B)",
        fontsize=10, y=1.01,
    )
    _add_footer_legend(fig)
    _save_fig(fig, f"fig01_T_field_zoom{suffix}")


# ---------------------------------------------------------------------------
# FIGURA 2 — ΔT = T_PINN − T_FEM con signo [K]
# ---------------------------------------------------------------------------
def fig_dT_signed(
    XX: np.ndarray, YY: np.ndarray,
    T_fem_grid: np.ndarray,
    pinn_results: list[dict],
    suffix: str = "",
) -> None:
    valid = [r for r in pinn_results if r["T_grid"] is not None]
    if not valid:
        return
    n     = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0))
    if n == 1:
        axes = [axes]

    dT_all   = [r["T_grid"] - T_fem_grid for r in valid]
    abs_max  = max(np.nanmax(np.abs(d)) for d in dT_all)
    clim     = float(np.ceil(min(abs_max * 0.95, 7.0) * 2) / 2)  # round to 0.5

    iso_fem = _isotherm_levels(T_fem_grid, step=5.0)

    for ax, r, dT in zip(axes, valid, dT_all):
        pm = ax.pcolormesh(
            XX, YY, dT,
            cmap="RdBu_r", vmin=-clim, vmax=clim,
            shading="gouraud", rasterized=True, zorder=1,
        )
        # Isotermas del campo FEM de referencia en gris
        add_isotherms(
            ax, XX, YY, T_fem_grid, iso_fem,
            color="#888888", lw=0.40, alpha=0.55, label_every=2,
            fmt="%d °C", fontsize=5.2,
        )
        add_zone_decorations(ax, add_labels=True)
        _style_ax(ax, xlabel=True, ylabel=(ax is axes[0]))

        # Estadísticas en la zona visible
        zoom_mask = (
            (XX >= ZOOM_XMIN) & (XX <= ZOOM_XMAX) &
            (YY >= ZOOM_YMIN) & (YY <= ZOOM_YMAX)
        )
        dT_zoom  = dT[zoom_mask]
        rmse_z   = float(np.sqrt(np.nanmean(dT_zoom**2)))
        bias_z   = float(np.nanmean(dT_zoom))
        emax_z   = float(np.nanmax(np.abs(dT_zoom)))

        ax.set_title(
            f"{r['label']}\n"
            f"RMSE$_{{\\mathrm{{zoom}}}}$ = {rmse_z:.3f} K  |  "
            f"bias = {bias_z:+.3f} K  |  "
            f"$|\\Delta T|_{{\\max}}$ = {emax_z:.2f} K",
            pad=5, fontsize=8.5,
        )

        # Anotar punto de máximo error absoluto
        idx_max  = np.unravel_index(np.nanargmax(np.abs(dT)), dT.shape)
        x_max    = float(XX[idx_max])
        y_max    = float(YY[idx_max])
        dT_val   = float(dT[idx_max])
        # Posición de la anotación (al lado del punto para evitar solapamiento)
        xt, yt   = (0.60, 0.07) if x_max < 0 else (0.05, 0.07)
        ax.annotate(
            f"$\\Delta T_{{\\max}}$ = {dT_val:+.1f} K\n"
            f"({x_max:.2f}, {y_max:.2f}) m",
            xy=(x_max, y_max),
            xytext=(xt, yt), textcoords="axes fraction",
            fontsize=6, color="black", zorder=10,
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8,
                            connectionstyle="arc3,rad=0.2"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#aaaaaa", alpha=0.88),
        )

        cbar = fig.colorbar(pm, ax=ax, shrink=0.88, pad=0.015, aspect=30)
        cbar.set_label(
            "$\\Delta T = T_\\mathrm{PINN} - T_\\mathrm{FEM}$ [K]",
            labelpad=4,
        )
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.yaxis.set_major_locator(MultipleLocator(1.0))

    # Línea de referencia en 0 en cada colorbar
    for ax in axes:
        for child in ax.get_children():
            if isinstance(child, matplotlib.colorbar.Colorbar):
                child.ax.axhline(y=0, color="black", lw=0.8, ls="--")

    fig.suptitle(
        "Diferencia con signo $\\Delta T = T_\\mathrm{PINN} - T_\\mathrm{FEM}$ "
        "— zona cables (Kim 2024 Caso B)\n"
        "Rojo: PINN sobreestima  |  Azul: PINN subestima  |  "
        "Contornos grises = isotermas FEM ref.",
        fontsize=8.5, y=1.02,
    )
    _add_footer_legend(fig)
    _save_fig(fig, f"fig02_dT_signed_zoom{suffix}")


# ---------------------------------------------------------------------------
# FIGURA 3 — |ΔT| error absoluto [K]
# ---------------------------------------------------------------------------
def fig_dT_abs(
    XX: np.ndarray, YY: np.ndarray,
    T_fem_grid: np.ndarray,
    pinn_results: list[dict],
    suffix: str = "",
) -> None:
    valid = [r for r in pinn_results if r["T_grid"] is not None]
    if not valid:
        return
    n     = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0))
    if n == 1:
        axes = [axes]

    abs_grids = [np.abs(r["T_grid"] - T_fem_grid) for r in valid]
    abs_max   = max(np.nanmax(a) for a in abs_grids)
    clim      = float(np.ceil(min(abs_max * 0.95, 8.0)))

    iso_fem = _isotherm_levels(T_fem_grid, step=5.0)

    for ax, r, dT_abs in zip(axes, valid, abs_grids):
        pm = ax.pcolormesh(
            XX, YY, dT_abs,
            cmap="YlOrRd", vmin=0, vmax=clim,
            shading="gouraud", rasterized=True, zorder=1,
        )
        add_isotherms(
            ax, XX, YY, T_fem_grid, iso_fem,
            color="#333333", lw=0.40, alpha=0.50, label_every=2,
            fmt="%d °C", fontsize=5.2,
        )
        add_zone_decorations(ax, add_labels=True)
        _style_ax(ax, xlabel=True, ylabel=(ax is axes[0]))

        zoom_mask = (
            (XX >= ZOOM_XMIN) & (XX <= ZOOM_XMAX) &
            (YY >= ZOOM_YMIN) & (YY <= ZOOM_YMAX)
        )
        abs_zoom = dT_abs[zoom_mask]
        rmse_z   = float(np.sqrt(np.nanmean((r["T_grid"] - T_fem_grid)[zoom_mask]**2)))
        mae_z    = float(np.nanmean(abs_zoom))
        emax_z   = float(np.nanmax(abs_zoom))
        p95_z    = float(np.nanpercentile(abs_zoom, 95))

        ax.set_title(
            f"{r['label']}\n"
            f"RMSE = {rmse_z:.3f} K  |  MAE = {mae_z:.3f} K  |  "
            f"P95 = {p95_z:.2f} K  |  max = {emax_z:.2f} K",
            pad=5, fontsize=8.5,
        )

        # Contornos del error absoluto para destacar zonas de alto error
        try:
            error_levels = np.array([1.0, 2.0, 4.0, 6.0])
            error_levels = error_levels[error_levels < clim]
            if len(error_levels) >= 1:
                ce = ax.contour(
                    XX, YY, dT_abs, levels=error_levels,
                    colors=["#444444"], linewidths=[0.5], alpha=0.75,
                    linestyles=["--"], zorder=3,
                )
                ax.clabel(ce, inline=True, fmt="%.0f K", fontsize=5.5,
                          inline_spacing=1, zorder=4)
        except Exception:
            pass

        # Anotar la región de máximo error
        idx_max = np.unravel_index(np.nanargmax(dT_abs), dT_abs.shape)
        x_max   = float(XX[idx_max])
        y_max   = float(YY[idx_max])
        xt, yt  = (0.58, 0.06) if x_max < 0 else (0.04, 0.06)
        ax.annotate(
            f"$|\\Delta T|_{{\\max}}$ = {emax_z:.2f} K\n({x_max:.2f}, {y_max:.2f}) m",
            xy=(x_max, y_max),
            xytext=(xt, yt), textcoords="axes fraction",
            fontsize=6, color="black", zorder=10,
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8,
                            connectionstyle="arc3,rad=0.2"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#aaaaaa", alpha=0.88),
        )

        cbar = fig.colorbar(pm, ax=ax, shrink=0.88, pad=0.015, aspect=30)
        cbar.set_label("$|\\Delta T|$ [K]", labelpad=4)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.yaxis.set_major_locator(MultipleLocator(1.0))

    fig.suptitle(
        "Error absoluto $|\\Delta T| = |T_\\mathrm{PINN} - T_\\mathrm{FEM}|$ "
        "— zona cables (Kim 2024 Caso B)\n"
        "Contornos grises = isotermas FEM ref.  |  "
        "Contornos punteados negros = isolíneas de error [K]",
        fontsize=8.5, y=1.02,
    )
    _add_footer_legend(fig)
    _save_fig(fig, f"fig03_dT_abs_zoom{suffix}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Gráficos publicación FEM vs PINN")
    ap.add_argument("--corrected", action="store_true",
                    help="Usar modelos con T_amb=300.30 K (corregido)")
    args = ap.parse_args()

    use_corrected = args.corrected
    if use_corrected:
        m64   = MODEL_64x4_CORR
        m128  = MODEL_128x5_CORR
        mdist = MODEL_DISTIL_CORR
        tag   = "(T_amb corregido)"
    else:
        m64   = MODEL_64x4
        m128  = MODEL_128x5
        mdist = MODEL_DISTIL
        tag   = "(T_amb original)"

    _set_pub_style()
    device = torch.device("cpu")

    print("=" * 65)
    print("  Gráficos de publicación — FEM vs PINN  (Kim 2024 Caso B)")
    print(f"  {tag}")
    print("=" * 65)

    print("\n1. Cargando campo FEM …")
    fem = load_fem()

    print("\n2. Construyendo grilla zoom (480×320) …")
    x_g, y_g, XX, YY = build_zoom_grid(nx=480, ny=320)
    print(f"   {XX.shape[1]}×{XX.shape[0]} = {XX.size:,d} puntos en "
          f"x∈[{ZOOM_XMIN},{ZOOM_XMAX}], y∈[{ZOOM_YMIN},{ZOOM_YMAX}] m")
    print("\n3. Interpolando FEM en grilla …")
    T_fem_grid = interpolate_fem_on_grid(fem, XX, YY)

    print("\n4. Cargando problema PINN …")
    shared = build_shared_problem()

    suffix = "_corrected" if use_corrected else ""
    MODELS = [
        (f"PINN 64×4\n{tag}",   m64,   64, 4),
        (f"PINN 128×5 baseline\n{tag}", m128, 128, 5),
        (f"PINN 128×5 destilado\n{tag}", mdist, 128, 5),
    ]

    pinn_results: list[dict] = []
    for i, (label, ckpt, width, depth) in enumerate(MODELS):
        print(f"\n5.{i+1} Evaluando {label.replace(chr(10), ' ')} …")
        T_g = eval_pinn_on_grid(ckpt, shared, device, width, depth, XX, YY)
        pinn_results.append(dict(label=label, T_grid=T_g))

    T_fem_max = float(np.nanmax(T_fem_grid))
    print(f"\n   T_max FEM (zona zoom) = {T_fem_max:.4f} °C")

    print("\n6. Generando figuras de publicación …")

    print("\n  ➜ fig01 — Campos T")
    fig_T_fields(XX, YY, T_fem_grid, pinn_results, T_fem_max, suffix=suffix)

    print("\n  ➜ fig02 — ΔT con signo")
    fig_dT_signed(XX, YY, T_fem_grid, pinn_results, suffix=suffix)

    print("\n  ➜ fig03 — |ΔT| absoluto")
    fig_dT_abs(XX, YY, T_fem_grid, pinn_results, suffix=suffix)

    print(f"\n✓ Figuras guardadas en: {HERE}")
    print("  (PDF: editable en Illustrator/Inkscape | PNG: 300 dpi)\n")


if __name__ == "__main__":
    main()

