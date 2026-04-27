"""Análisis espacial FEM vs PINN — Kim 2024 Caso B.

Carga el campo FEM de FEniCSx (results_fem/fem_field_kim2024B.npz) y
evalúa los tres modelos PINN en los mismos nodos de la malla.

Métricas calculadas
-------------------
  - RMSE espacial (dominio completo, en K y °C)
  - RMSE espacial (zona zoom cables: x ∈ [-1.5, 1.5] m, y ∈ [-2.5, -0.5] m)
  - Error máximo absoluto
  - T_max conductor (6 cables) vs paper FEM vs FEM FEniCSx
  - Tabla Markdown lista para copiar en la tesis

Outputs guardados en results_fem/
  - comparison_spatial_errors.csv
  - comparison_T_field_spatial.png   (campo FEM + campos PINN + mapas de error)
  - comparison_error_map.png         (solo los 3 mapas |T_PINN - T_FEM|)

Uso::

    python examples/kim_2024_154kv_optim_C/analyze_fem_vs_pinn.py
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE        = Path(__file__).resolve().parent
DATA_DIR    = HERE.parent / "kim_2024_154kv_optim_B" / "data"
FEM_NPZ     = HERE / "results_fem" / "fem_field_kim2024B.npz"

# Paths are resolved at runtime depending on --corrected flag (see main())
# Default: original T_amb=290.15 K models
MODEL_64x4   = HERE.parent / "kim_2024_154kv_optim_B" / "results_optim"     / "model_best_64x4.pt"
MODEL_128x5  = HERE / "results_optim"  / "model_best_128x5.pt"
MODEL_DISTIL = HERE / "results_distil" / "model_best_128x5_distil.pt"

# Corrected T_amb=300.30 K models
MODEL_64x4_CORR   = HERE.parent / "kim_2024_154kv_optim_B" / "results_corrected" / "model_best_64x4.pt"
MODEL_128x5_CORR  = HERE / "results_corrected" / "model_best_128x5.pt"
MODEL_DISTIL_CORR = HERE / "results_corrected" / "model_best_128x5_distil.pt"

RESULTS_DIR = HERE / "results_fem"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constantes del problema (Kim 2024, Caso B)
# ---------------------------------------------------------------------------
FEM_REF_C       = 70.6   # °C — referencia paper COMSOL
PAPER_T_MAX_PAC = 273.15 + FEM_REF_C   # K
PAPER_W_D       = 3.57
PAPER_FREQ      = 60.0
CABLES = [
    (-0.40, -1.60), ( 0.00, -1.60), ( 0.40, -1.60),
    (-0.40, -1.20), ( 0.00, -1.20), ( 0.40, -1.20),
]
R_CONDUCTOR = 0.01885   # m  (kim 2024 optim_B data — 4-layer geometry)
R_CABLE     = 0.05335   # m

# Zona "zoom cables" para métricas locales
ZOOM_XMIN, ZOOM_XMAX = -1.5,  1.5
ZOOM_YMIN, ZOOM_YMAX = -2.5, -0.5


# ---------------------------------------------------------------------------
# Cargar el campo FEM
# ---------------------------------------------------------------------------
def load_fem_field() -> dict:
    if not FEM_NPZ.exists():
        raise FileNotFoundError(
            f"Campo FEM no encontrado: {FEM_NPZ}\n"
            "Ejecuta primero fem_fenicsx_colab.ipynb (Celda 8) y copia el .npz aquí."
        )
    d = np.load(FEM_NPZ)
    print(f"✓ FEM cargado:  {FEM_NPZ.name}")
    print(f"  nodos : {d['coords_xy'].shape[0]:,d}")
    print(f"  celdas: {d['cells'].shape[0]:,d}")
    T_C = d["T_C"]
    print(f"  T_C ∈ [{T_C.min():.2f}, {T_C.max():.2f}] °C")
    return dict(
        coords_xy = d["coords_xy"],   # (N, 2)  float64
        T_C       = T_C,              # (N,)    °C
        cells     = d["cells"],       # (M, 3)  int32
    )


# ---------------------------------------------------------------------------
# Preparar el "shared" problem para cargar los modelos PINN
# ---------------------------------------------------------------------------
def build_shared_problem() -> dict:
    from pinn_cables.io.readers import load_problem
    from pinn_cables.materials.props import get_kim2024_cable_layers
    from pinn_cables.physics.iec60287 import compute_iec60287_Q
    from pinn_cables.physics.k_field import load_physics_params, load_soil_layers

    problem        = load_problem(DATA_DIR)
    all_placements = problem.placements
    n_cables       = len(all_placements)
    soil_bands     = load_soil_layers(DATA_DIR / "soil_layers.csv")
    pac_params_base = load_physics_params(DATA_DIR / "physics_params.csv")

    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A   = all_placements[0].current_A
    section_mm2 = all_placements[0].section_mm2

    iec_q   = compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=PAPER_T_MAX_PAC, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_cond  = iec_q["Q_cond_W_per_m"]
    Q_lin   = Q_cond + PAPER_W_D
    layers  = get_kim2024_cable_layers(Q_lin)

    return dict(
        problem         = problem,
        all_placements  = all_placements,
        n_cables        = n_cables,
        soil_bands      = soil_bands,
        pac_params_base = pac_params_base,
        layers_list     = [layers] * n_cables,
        Q_lins          = [Q_lin] * n_cables,
        Q_d             = PAPER_W_D,
    )


# ---------------------------------------------------------------------------
# Cargar un modelo PINN y evaluarlo en coords_xy arbitrarias (nodos FEM)
# ---------------------------------------------------------------------------
def load_pinn_at_nodes(
    ckpt_path: Path,
    shared: dict,
    device: torch.device,
    width: int,
    depth: int,
    coords_xy: np.ndarray,   # (N, 2)  — coordenadas físicas [m]
    batch_size: int = 50_000,
) -> np.ndarray | None:
    """Evalúa el PINN en los puntos `coords_xy` (en K → devuelve °C)."""
    if not ckpt_path.exists():
        print(f"  AVISO: modelo no encontrado: {ckpt_path}")
        return None

    from pinn_cables.pinn.model import ResidualPINNModel, build_model
    from pinn_cables.physics.k_field import KFieldModel, load_soil_layers

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

    use_fourier = params.get("fourier_mapping_size", 0) > 0
    cfg = {
        "architecture":         "mlp",
        "width":                width,
        "depth":                depth,
        "activation":           "tanh",
        "fourier_features":     use_fourier,
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

    # Normalizar coordenadas al rango [-1, 1]
    dom   = shared["problem"].domain
    x_n   = (2.0 * (coords_xy[:, 0] - dom.xmin) / (dom.xmax - dom.xmin) - 1.0).astype(np.float32)
    y_n   = (2.0 * (coords_xy[:, 1] - dom.ymin) / (dom.ymax - dom.ymin) - 1.0).astype(np.float32)
    xy_np = np.stack([x_n, y_n], axis=1)

    # Evaluación por lotes (evita OOM en CPU)
    T_out = np.empty(len(xy_np), dtype=np.float32)
    for start in range(0, len(xy_np), batch_size):
        end  = min(start + batch_size, len(xy_np))
        xy_t = torch.from_numpy(xy_np[start:end]).to(device)
        with torch.no_grad():
            T_out[start:end] = model(xy_t).cpu().numpy().ravel()

    return T_out.astype(np.float64) - 273.15   # → °C


# ---------------------------------------------------------------------------
# Métricas espaciales
# ---------------------------------------------------------------------------
def spatial_metrics(T_pinn: np.ndarray, T_fem: np.ndarray, mask: np.ndarray | None = None
                    ) -> dict:
    """RMSE, MAE y error máximo sobre la región definida por `mask`."""
    if mask is not None:
        t_p = T_pinn[mask]
        t_f = T_fem[mask]
    else:
        t_p, t_f = T_pinn, T_fem
    err   = t_p - t_f
    rmse  = float(np.sqrt(np.mean(err**2)))
    mae   = float(np.mean(np.abs(err)))
    e_max = float(np.max(np.abs(err)))
    bias  = float(np.mean(err))
    return dict(rmse=rmse, mae=mae, e_max=e_max, bias=bias, n=int(mask.sum() if mask is not None else len(T_pinn)))


# ---------------------------------------------------------------------------
# T_max conductor para cada modelo evaluado en los nodos FEM
# ---------------------------------------------------------------------------
def tmax_from_pinn_at_nodes(T_pinn_C: np.ndarray, coords_xy: np.ndarray) -> float:
    """T máxima dentro de los conductores Cu (r ≤ R_CONDUCTOR de cualquier cable)."""
    x, y = coords_xy[:, 0], coords_xy[:, 1]
    mask = np.zeros(len(x), bool)
    for cx, cy in CABLES:
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask |= r <= R_CONDUCTOR * 1.05   # pequeña tolerancia para nodos de borde
    if mask.sum() == 0:
        return float(np.max(T_pinn_C))
    return float(np.max(T_pinn_C[mask]))


def tmax_fem_from_nodes(T_fem_C: np.ndarray, coords_xy: np.ndarray) -> float:
    x, y = coords_xy[:, 0], coords_xy[:, 1]
    mask = np.zeros(len(x), bool)
    for cx, cy in CABLES:
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask |= r <= R_CONDUCTOR * 1.05
    if mask.sum() == 0:
        return float(np.max(T_fem_C))
    return float(np.max(T_fem_C[mask]))


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def make_plots(fem: dict, results: list[dict], T_fem_max: float, suffix: str = "") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from matplotlib.colors import Normalize

    coords = fem["coords_xy"]
    cells  = fem["cells"]
    T_fem  = fem["T_C"]
    tri    = Triangulation(coords[:, 0], coords[:, 1], cells)

    # ── Figura 1: campo T completo (FEM + 3 PINNs + zoom) ─────────────────
    ncols = len(results) + 1   # FEM + PINNs
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 10))

    vmin = min(T_fem.min(), *(r["T_C"].min() for r in results if r["T_C"] is not None))
    vmax = max(T_fem.max(), *(r["T_C"].max() for r in results if r["T_C"] is not None))
    norm = Normalize(vmin=vmin, vmax=vmax)

    def _plot_field(ax, T, title, zoom=False):
        tc = ax.tripcolor(tri, T, shading="flat", cmap="hot_r", norm=norm)
        plt.colorbar(tc, ax=ax, label="T [°C]", fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        if zoom:
            ax.set_xlim(ZOOM_XMIN, ZOOM_XMAX)
            ax.set_ylim(ZOOM_YMIN, ZOOM_YMAX)
            for cx, cy in CABLES:
                ax.add_patch(plt.Circle((cx, cy), R_CABLE, fill=False, color="cyan", lw=0.8))
                ax.add_patch(plt.Circle((cx, cy), R_CONDUCTOR, fill=False, color="yellow", lw=0.5))

    _plot_field(axes[0, 0], T_fem, f"FEM FEniCSx\nT_max={T_fem_max:.2f}°C")
    for i, r in enumerate(results):
        if r["T_C"] is not None:
            _plot_field(axes[0, i+1], r["T_C"], f"{r['label']}\nT_max={r['T_max']:.2f}°C")
        else:
            axes[0, i+1].text(0.5, 0.5, "N/A", transform=axes[0, i+1].transAxes, ha="center")

    # Zoom
    _plot_field(axes[1, 0], T_fem, "FEM FEniCSx — zoom", zoom=True)
    for i, r in enumerate(results):
        if r["T_C"] is not None:
            _plot_field(axes[1, i+1], r["T_C"], f"{r['label']} — zoom", zoom=True)

    fig.suptitle("Campos de temperatura — FEM FEniCSx vs modelos PINN (Kim 2024 Caso B)", fontsize=12)
    plt.tight_layout()
    out1 = RESULTS_DIR / f"comparison_T_field_spatial{suffix}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Figura campo T: {out1}")

    # ── Figura 2: mapas de error |T_PINN - T_FEM| ─────────────────────────
    fig2, axes2 = plt.subplots(2, len(results), figsize=(5 * len(results), 9))
    if len(results) == 1:
        axes2 = axes2.reshape(2, 1)

    e_global_max = max(
        np.max(np.abs(r["T_C"] - T_fem))
        for r in results if r["T_C"] is not None
    )
    enorm = Normalize(vmin=0, vmax=min(e_global_max, 2.0))

    for i, r in enumerate(results):
        if r["T_C"] is None:
            continue
        err = np.abs(r["T_C"] - T_fem)

        # Dominio completo
        tc = axes2[0, i].tripcolor(tri, err, shading="flat", cmap="plasma", norm=enorm)
        plt.colorbar(tc, ax=axes2[0, i], label="|ΔT| [°C]", fraction=0.046, pad=0.04)
        axes2[0, i].set_title(f"|T_PINN - T_FEM| — {r['label']}\nRMSE={r['rmse_total']:.3f} K", fontsize=9)
        axes2[0, i].set_xlabel("x [m]"); axes2[0, i].set_ylabel("y [m]")
        axes2[0, i].set_aspect("equal")

        # Zoom cables
        tc2 = axes2[1, i].tripcolor(tri, err, shading="flat", cmap="plasma", norm=enorm)
        plt.colorbar(tc2, ax=axes2[1, i], label="|ΔT| [°C]", fraction=0.046, pad=0.04)
        axes2[1, i].set_xlim(ZOOM_XMIN, ZOOM_XMAX); axes2[1, i].set_ylim(ZOOM_YMIN, ZOOM_YMAX)
        axes2[1, i].set_title(
            f"|T_PINN - T_FEM| zoom — {r['label']}\nRMSE_zoom={r['rmse_zoom']:.3f} K", fontsize=9
        )
        axes2[1, i].set_xlabel("x [m]"); axes2[1, i].set_ylabel("y [m]")
        axes2[1, i].set_aspect("equal")
        for cx, cy in CABLES:
            axes2[1, i].add_patch(plt.Circle((cx, cy), R_CABLE, fill=False, color="cyan", lw=0.8))

    fig2.suptitle("Mapas de error espacial |T_PINN − T_FEM| (Kim 2024 Caso B)", fontsize=12)
    plt.tight_layout()
    out2 = RESULTS_DIR / f"comparison_error_map{suffix}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"✓ Figura error map: {out2}")


# ---------------------------------------------------------------------------
# Tabla de resultados
# ---------------------------------------------------------------------------
def print_table(results: list[dict], T_fem_max: float) -> None:
    w1, w2 = 30, 10
    sep = "─" * (w1 + 6 * w2 + 7)

    print()
    print("=" * len(sep))
    print("  COMPARACIÓN ESPACIAL FEM FEniCSx vs PINN — Kim 2024 Caso B")
    print("=" * len(sep))

    header = (
        f"  {'Método':<{w1}}  {'T_max':>{w2}}  {'ΔT_max':>{w2}}  "
        f"{'RMSE_tot':>{w2}}  {'RMSE_zoom':>{w2}}  {'MAE_tot':>{w2}}  {'|ΔT|_max':>{w2}}"
    )
    print(header)
    print(sep)

    # Fila FEM paper (referencia puntual)
    dT_fem_paper = T_fem_max - FEM_REF_C
    print(f"  {'FEM paper COMSOL (Kim 2024)':<{w1}}  {FEM_REF_C:>{w2}.2f}  {'—':>{w2}}  {'—':>{w2}}  {'—':>{w2}}  {'—':>{w2}}  {'—':>{w2}}")
    print(f"  {'FEM FEniCSx (ref. espacial)':<{w1}}  {T_fem_max:>{w2}.2f}  {dT_fem_paper:>{w2}.2f}  {'0.000':>{w2}}  {'0.000':>{w2}}  {'0.000':>{w2}}  {'0.000':>{w2}}")
    print(sep)

    for r in results:
        dT   = r["T_max"] - FEM_REF_C
        rmse = r["rmse_total"]
        rmsz = r["rmse_zoom"]
        mae  = r["mae_total"]
        emax = r["emax_total"]
        if r["T_C"] is None:
            print(f"  {r['label']:<{w1}}  {'N/A':>{w2}}  {'N/A':>{w2}}  {'N/A':>{w2}}  {'N/A':>{w2}}  {'N/A':>{w2}}  {'N/A':>{w2}}")
        else:
            print(
                f"  {r['label']:<{w1}}  {r['T_max']:>{w2}.2f}  {dT:>{w2}.2f}  "
                f"{rmse:>{w2}.4f}  {rmsz:>{w2}.4f}  {mae:>{w2}.4f}  {emax:>{w2}.4f}"
            )
    print(sep)
    print("  Unidades: T en [°C], errores en [K = °C]\n")
    print("  RMSE_tot  = √(Σ|T_PINN-T_FEM|²/N)  sobre todo el dominio")
    print("  RMSE_zoom = ídem, solo nodos en zona cables (x∈[-1.5,1.5], y∈[-2.5,-0.5])")
    print("  ΔT_max    = T_max_modelo − 70.6 °C (paper COMSOL)")
    print()


def save_csv(results: list[dict], T_fem_max: float, suffix: str = "") -> None:
    import csv
    csv_path = RESULTS_DIR / f"comparison_spatial_errors{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "T_max_C", "dT_vs_paper_K", "RMSE_total_K",
                    "RMSE_zoom_K", "MAE_total_K", "emax_total_K",
                    "bias_total_K", "n_nodos_total", "n_nodos_zoom"])
        w.writerow(["FEM_paper_COMSOL", FEM_REF_C, 0.0, 0, 0, 0, 0, 0, 0, 0])
        w.writerow(["FEM_FEniCSx", round(T_fem_max, 4), round(T_fem_max - FEM_REF_C, 4),
                    0, 0, 0, 0, 0, 0, 0])
        for r in results:
            if r["T_C"] is None:
                w.writerow([r["label"]] + ["N/A"] * 9)
            else:
                w.writerow([
                    r["label"],
                    round(r["T_max"], 4),
                    round(r["T_max"] - FEM_REF_C, 4),
                    round(r["rmse_total"], 6),
                    round(r["rmse_zoom"],  6),
                    round(r["mae_total"],  6),
                    round(r["emax_total"], 6),
                    round(r["bias_total"], 6),
                    r["n_total"],
                    r["n_zoom"],
                ])
    print(f"✓ CSV guardado: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Comparación espacial FEM vs PINN")
    ap.add_argument("--corrected", action="store_true",
                    help="Usar modelos con T_amb=300.30 K (corregido)")
    args = ap.parse_args()

    use_corrected = args.corrected
    suffix = "_corrected" if use_corrected else ""

    if use_corrected:
        m64   = MODEL_64x4_CORR
        m128  = MODEL_128x5_CORR
        mdist = MODEL_DISTIL_CORR
        tag   = "T_amb=300.30 K (corregido)"
    else:
        m64   = MODEL_64x4
        m128  = MODEL_128x5
        mdist = MODEL_DISTIL
        tag   = "T_amb=290.15 K (original)"

    device = torch.device("cpu")

    # 1. Cargar campo FEM
    print("─" * 60)
    print("1. Cargando campo FEM …")
    fem = load_fem_field()
    coords_xy = fem["coords_xy"]
    T_fem_C   = fem["T_C"]

    # Máscara zona zoom cables
    x, y = coords_xy[:, 0], coords_xy[:, 1]
    zoom_mask = (x >= ZOOM_XMIN) & (x <= ZOOM_XMAX) & (y >= ZOOM_YMIN) & (y <= ZOOM_YMAX)
    print(f"  Nodos dominio completo: {len(T_fem_C):,d}")
    print(f"  Nodos zona zoom cables: {zoom_mask.sum():,d}")

    T_fem_max = tmax_fem_from_nodes(T_fem_C, coords_xy)
    print(f"  T_max conductor FEM:    {T_fem_max:.4f} °C")

    # 2. Cargar el problema PINN
    print()
    print("─" * 60)
    print("2. Cargando problema PINN (Data dir: %s) …" % DATA_DIR.name)
    shared = build_shared_problem()

    # 3. Evaluar cada modelo
    MODELS = [
        (f"PINN 64×4  ({tag})",      m64,   64,  4),
        (f"PINN 128×5 baseline ({tag})",  m128, 128,  5),
        (f"PINN 128×5 destilado ({tag})", mdist,128,  5),
    ]

    results = []
    for label, ckpt_path, width, depth in MODELS:
        print()
        print("─" * 60)
        print(f"3. Evaluando {label} …")
        T_pinn_C = load_pinn_at_nodes(ckpt_path, shared, device, width, depth, coords_xy)

        if T_pinn_C is not None:
            T_max    = tmax_from_pinn_at_nodes(T_pinn_C, coords_xy)
            m_total  = spatial_metrics(T_pinn_C, T_fem_C)
            m_zoom   = spatial_metrics(T_pinn_C, T_fem_C, zoom_mask)
            print(f"  T_max conductor: {T_max:.4f} °C  (FEM ref: {T_fem_max:.4f})")
            print(f"  RMSE total:      {m_total['rmse']:.4f} K")
            print(f"  RMSE zoom:       {m_zoom['rmse']:.4f} K")
            results.append(dict(
                label       = label,
                T_C         = T_pinn_C,
                T_max       = T_max,
                rmse_total  = m_total["rmse"],
                rmse_zoom   = m_zoom["rmse"],
                mae_total   = m_total["mae"],
                emax_total  = m_total["e_max"],
                bias_total  = m_total["bias"],
                n_total     = m_total["n"],
                n_zoom      = m_zoom["n"],
            ))
        else:
            results.append(dict(
                label=label, T_C=None, T_max=float("nan"),
                rmse_total=float("nan"), rmse_zoom=float("nan"),
                mae_total=float("nan"), emax_total=float("nan"),
                bias_total=float("nan"), n_total=0, n_zoom=0,
            ))

    # 4. Tabla y CSV
    print()
    print(f"  [{tag}]")
    print_table(results, T_fem_max)

    # Pass suffix for file naming
    save_csv(results, T_fem_max, suffix=suffix)

    # 5. Gráficas
    print("─" * 60)
    print("5. Generando figuras …")
    try:
        make_plots(fem, results, T_fem_max, suffix=suffix)
    except ImportError:
        print("  AVISO: matplotlib no disponible — figuras omitidas.")


if __name__ == "__main__":
    main()
