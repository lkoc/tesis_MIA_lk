"""Comparación PINN vs FNO — Kim et al. (2024) benchmark 154 kV.

Script que:
1. Genera un dataset paramétrico variando k_soil, k_pac e I_A
2. Entrena un FNO 2D sobre ese dataset
3. Evalúa el FNO en el caso nominal del paper (k_pac=2.094, I=1026 A)
4. Compara con resultados PINN previamente entrenados y con referencia FEM
5. Genera gráficas comparativas

La solución analítica de Kennelly se usa como ground-truth de entrenamiento
(exacta para k homogéneo, aproximada para zona PAC). El FNO aprende la
solución del operador −∇·(k∇T)=0 para distintas combinaciones de parámetros.

Uso::

    python examples/kim_2024_154kv_bedding/run_fno_comparison.py
    python examples/kim_2024_154kv_bedding/run_fno_comparison.py --n-samples 200 --epochs 300

Referencia FNO:
    Li et al. (2021) "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
    https://arxiv.org/abs/2010.08895
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results_fno"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pinn_cables.io.readers import (  # noqa: E402
    load_problem,
    load_solver_params,
    override_conductor_Q,
)
from pinn_cables.physics.iec60287 import compute_iec60287_Q  # noqa: E402
from pinn_cables.physics.k_field import (  # noqa: E402
    load_physics_params,
    make_k_functions,
)
from pinn_cables.physics.kennelly import iec60287_estimate  # noqa: E402
from pinn_cables.pinn.model import ResidualPINNModel, build_model  # noqa: E402
from pinn_cables.pinn.utils import get_device, set_seed  # noqa: E402
from pinn_cables.post.eval import eval_conductor_temps, evaluate_on_grid  # noqa: E402
from pinn_cables.fno.model import CableFNO2d  # noqa: E402
from pinn_cables.fno.dataset import CableParametricDataset, make_input_channels  # noqa: E402
from pinn_cables.fno.train import train_fno, FNOTrainConfig  # noqa: E402
from pinn_cables.fno.eval import eval_fno_field, fno_T_max  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes del paper
# ---------------------------------------------------------------------------
PAPER_T_MAX_PAC  = 273.15 + 70.6   # K — FEM T_max con PAC bedding
PAPER_T_MAX_SAND = 273.15 + 77.6   # K — FEM T_max con sand bedding
PAPER_W_D  = 3.57                   # W/m — pérdidas dieléctricas
PAPER_FREQ = 60.0                   # Hz
PAPER_K_PAC = 2.094                 # W/(mK)
PAPER_I_A   = 1026.0                # A  — corriente nominal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Comparación PINN vs FNO — Kim et al. (2024)"
    )
    ap.add_argument("--n-samples", type=int, default=150,
                    help="Número de muestras para entrenar FNO (default: 150)")
    ap.add_argument("--epochs", type=int, default=300,
                    help="Épocas de entrenamiento FNO (default: 300)")
    ap.add_argument("--n-grid", type=int, default=64,
                    help="Resolución de la malla FNO N_g×N_g (default: 64)")
    ap.add_argument("--d-model", type=int, default=32,
                    help="Canales internos del FNO (default: 32)")
    ap.add_argument("--modes", type=int, default=12,
                    help="Modos de Fourier (default: 12)")
    ap.add_argument("--n-layers", type=int, default=4,
                    help="Bloques FNO (default: 4)")
    ap.add_argument("--no-pinn-compare", action="store_true",
                    help="No cargar modelo PINN para comparar")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Utilidades de graficación
# ---------------------------------------------------------------------------

def _plot_fno_field(X, Y, T, placements, title, save_path):
    """Mapa de temperatura del FNO con contornos."""
    T_c = T.cpu().numpy() - 273.15   # °C
    X_n = X.cpu().numpy()
    Y_n = Y.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    levels = np.linspace(T_c.min(), T_c.max(), 20)
    cf = ax.contourf(X_n, Y_n, T_c, levels=levels, cmap="hot_r")
    ax.contour(X_n, Y_n, T_c, levels=levels[::3], colors="white", linewidths=0.4)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label("T [°C]", fontsize=11)

    for pl in placements:
        circle = plt.Circle((pl.cx, pl.cy), 0.05, color="cyan", fill=False, lw=1.5)
        ax.add_patch(circle)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(results: dict, save_path: Path) -> None:
    """Gráfica de barras comparando T_max entre métodos."""
    labels = list(results.keys())
    values = [v - 273.15 for v in results.values()]  # → °C

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    colors = colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", height=0.5)
    ax.bar_label(bars, fmt="%.1f °C", padding=3, fontsize=10)
    ax.axvline(x=PAPER_T_MAX_PAC - 273.15, color="red", linestyle="--",
               linewidth=1.5, label="FEM PAC (%.1f °C)" % (PAPER_T_MAX_PAC - 273.15))
    ax.set_xlabel("T_max [°C]", fontsize=11)
    ax.set_title("Comparación T_max — Kim et al. (2024) 154 kV, PAC bedding")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_training_loss(history: dict, save_path: Path) -> None:
    """Curva de pérdida del FNO."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_total"]) + 1)
    ax.semilogy(epochs, history["train_total"], label="Train total")
    ax.semilogy(epochs, history["train_data"], label="Train data", linestyle="--")
    if any(not np.isnan(v) for v in history["val_total"]):
        ax.semilogy(epochs, history["val_total"], label="Validation", linestyle=":")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE Loss [K²]")
    ax.set_title("FNO — Historial de entrenamiento")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 72
    print(SEP)
    print("  COMPARACIÓN PINN vs FNO — Kim et al. (2024) 154 kV PAC bedding")
    print("  Referencia: Li et al. (2021) arXiv:2010.08895")
    print(SEP)

    # --- Cargar problema ---
    problem = load_problem(DATA_DIR)
    scenario = problem.scenarios[0]
    all_placements = problem.placements
    n_cables = len(all_placements)
    pp = load_physics_params(DATA_DIR / "physics_params.csv")

    pl0 = all_placements[0]
    material_lc = pl0.conductor_material.strip().lower()

    # Calcular Q nominal
    T_op_K = PAPER_T_MAX_PAC
    iec_q = compute_iec60287_Q(
        pl0.section_mm2, material_lc, PAPER_I_A,
        T_op=T_op_K, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )
    Q_cond_nominal = iec_q["Q_cond_W_per_m"]
    Q_d = PAPER_W_D
    Q_total_nominal = Q_cond_nominal + Q_d

    layers_template = problem.get_layers(pl0.cable_id)
    layers_nominal = override_conductor_Q(layers_template, Q_total_nominal)
    Q_lins_nominal = [Q_total_nominal] * n_cables
    layers_list_nominal = [layers_nominal] * n_cables
    r_sheaths = [layers_nominal[-1].r_outer] * n_cables
    k_soil_nominal = scenario.k_soil
    T_amb = scenario.T_amb

    # k(x,y) nominal (zona PAC)
    k_fn_nominal, _, _ = make_k_functions(pp, k_soil_nominal, placements=all_placements)

    device = get_device("auto")
    set_seed(42)

    print("\n  Parámetros FNO:")
    print("    Muestras de entrenamiento : %d" % args.n_samples)
    print("    Épocas                    : %d" % args.epochs)
    print("    Malla                     : %dx%d" % (args.n_grid, args.n_grid))
    print("    d_model / modos / bloques : %d / %d / %d" % (
        args.d_model, args.modes, args.n_layers))

    # -----------------------------------------------------------------------
    # 1. Generar dataset paramétrico
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  GENERANDO DATASET PARAMÉTRICO (%d muestras)..." % args.n_samples)
    print("-" * 72)

    def _iec_Q_fn(section_mm2, material, I_A):
        try:
            res = compute_iec60287_Q(
                section_mm2, material.strip().lower(), I_A,
                T_op=T_op_K, W_d=PAPER_W_D, freq=PAPER_FREQ,
            )
            return res["Q_cond_W_per_m"]
        except Exception:
            import math
            R_dc = 0.0151e-3   # Ω/m @ 20°C, 1200 mm² Cu
            return I_A**2 * R_dc

    t_ds_start = time.time()
    n_train = max(1, int(args.n_samples * 0.85))
    n_val   = args.n_samples - n_train

    full_dataset = CableParametricDataset(
        domain=problem.domain,
        placements=all_placements,
        layers_template=layers_template,
        T_amb=T_amb,
        n_samples=args.n_samples,
        N_g=args.n_grid,
        k_soil_range=(0.8, 2.0),
        k_pac_range=(1.5, 2.5),
        I_range=(600.0, 1100.0),
        Q_d=Q_d,
        device=device,
        seed=42,
        iec_Q_fn=_iec_Q_fn,
        pp=pp,
    )

    from torch.utils.data import random_split
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    t_ds = time.time() - t_ds_start
    print("  Dataset generado en %.1f s  (train=%d / val=%d)" % (t_ds, n_train, n_val))

    # -----------------------------------------------------------------------
    # 2. Construir y entrenar FNO
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  ENTRENANDO FNO...")
    print("-" * 72)

    fno_model = CableFNO2d(
        in_channels=3,
        d_model=args.d_model,
        n_layers=args.n_layers,
        modes1=args.modes,
        modes2=args.modes,
    ).to(device)
    print("  FNO parámetros: %d" % fno_model.n_params)

    fno_cfg = FNOTrainConfig(
        epochs=args.epochs,
        batch_size=min(16, n_train),
        lr=1e-3,
        lr_decay_step=max(1, args.epochs // 3),
        lr_decay_gamma=0.5,
        w_data=1.0,
        w_pde=0.0,
        print_every=max(1, args.epochs // 10),
        device="auto",
        seed=42,
    )

    t_train_start = time.time()
    history = train_fno(
        model=fno_model,
        dataset=train_ds,
        cfg=fno_cfg,
        val_dataset=val_ds if n_val > 0 else None,
    )
    t_train = time.time() - t_train_start
    print("  FNO entrenado en %.1f s" % t_train)

    # Guardar modelo FNO
    fno_path = RESULTS_DIR / "fno_model.pt"
    torch.save(fno_model.state_dict(), fno_path)
    print("  FNO guardado: %s" % fno_path)

    # Gráfica de pérdida
    _plot_training_loss(history, RESULTS_DIR / "fno_loss_history.png")

    # -----------------------------------------------------------------------
    # 3. Evaluar FNO en caso nominal (k_pac=2.094, I=1026 A)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  EVALUANDO FNO — caso nominal")
    print("-" * 72)

    t_inf_start = time.time()
    fno_model.eval()
    result_fno = fno_T_max(
        model=fno_model,
        domain=problem.domain,
        placements=all_placements,
        layers_list=layers_list_nominal,
        k_fn=k_fn_nominal,
        Q_lins=Q_lins_nominal,
        T_amb=T_amb,
        N_g=args.n_grid,
        device=device,
    )
    t_inf = (time.time() - t_inf_start) * 1000  # ms
    T_fno_max = result_fno["T_max_K"]
    print("  FNO T_max = %.2f K (%.2f °C)" % (T_fno_max, T_fno_max - 273.15))
    print("  Tiempo inferencia FNO: %.2f ms" % t_inf)

    # Mapa de temperatura FNO
    X_g, Y_g, T_grid = eval_fno_field(
        fno_model, problem.domain, all_placements, layers_list_nominal,
        k_fn_nominal, Q_lins_nominal, T_amb, args.n_grid, device,
    )
    _plot_fno_field(
        X_g, Y_g, T_grid, all_placements,
        title="T [°C] — FNO 2D — Kim et al. (2024) PAC bedding",
        save_path=RESULTS_DIR / "fno_temperature_field.png",
    )

    # -----------------------------------------------------------------------
    # 4. Cargar T_max del PINN (si existe)
    # -----------------------------------------------------------------------
    results_comparison: dict[str, float] = {}
    results_comparison["FEM (paper)"] = PAPER_T_MAX_PAC

    pinn_T_max = None
    if not args.no_pinn_compare:
        pinn_paths = {
            "PINN quick": HERE / "results" / "model_final.pt",
            "PINN research": HERE / "results_research" / "model_final.pt",
            "PINN PAC quick": HERE / "results_pac_quick" / "model_final.pt",
            "PINN PAC research": HERE / "results_pac_research" / "model_final.pt",
        }
        for label, pt_path in pinn_paths.items():
            if not pt_path.exists():
                continue
            try:
                # Detectar perfil por tamaño de modelo
                is_research = "research" in label.lower()
                solver_csv = "solver_params_research.csv" if is_research else "solver_params.csv"
                sp = load_solver_params(DATA_DIR / solver_csv)
                cfg = sp.to_solver_cfg()
                normalize = cfg.get("normalization", {}).get("normalize_coords", True)
                base = build_model(cfg["model"], in_dim=2, device=device)
                pinn_mdl = ResidualPINNModel(
                    base, layers_list_nominal, all_placements,
                    k_soil_nominal, T_amb, Q_lins_nominal,
                    problem.domain, normalize=normalize, Q_d=Q_d,
                    enable_grad_Tbg="pac" in label.lower(),
                )
                pinn_mdl.load_state_dict(
                    torch.load(pt_path, map_location=device, weights_only=True)
                )
                pinn_mdl.eval()
                T_vals = eval_conductor_temps(
                    pinn_mdl, all_placements, problem.domain, device, normalize,
                )
                T_max_pinn = max(T_vals)
                results_comparison[label] = T_max_pinn
                if pinn_T_max is None:
                    pinn_T_max = T_max_pinn
                print("  %s T_max = %.2f K (%.2f °C)" % (label, T_max_pinn, T_max_pinn - 273.15))
            except Exception as e:
                print("  (No se pudo cargar %s: %s)" % (label, e))

    results_comparison["FNO (este trabajo)"] = T_fno_max
    _plot_comparison(results_comparison, RESULTS_DIR / "comparison_T_max.png")

    # -----------------------------------------------------------------------
    # 5. Tabla resumen
    # -----------------------------------------------------------------------
    print("\n" + SEP)
    print("  TABLA RESUMEN — T_max COMPARACIÓN")
    print(SEP)
    print("  %-30s  %7s  %7s  %8s" % ("Método", "T_K", "T_°C", "Error K"))
    print("  " + "-" * 60)
    T_ref = PAPER_T_MAX_PAC
    for label, T_val in results_comparison.items():
        err = T_val - T_ref if label != "FEM (paper)" else float("nan")
        err_str = "%+.2f" % err if not (err != err) else "   ref"
        print("  %-30s  %7.2f  %7.2f  %8s" % (label, T_val, T_val - 273.15, err_str))
    print(SEP)

    print("\n  Velocidad comparada:")
    pinn_info = "" if pinn_T_max is None else "(PINN entrenamiento ~10-30 min)"
    print("    FNO  dataset+entrenamiento  : %.1f s total" % (t_ds + t_train))
    print("    FNO  inferencia por caso    : %.2f ms %s" % (t_inf, pinn_info))
    print("    PINN inferencia por caso    : ~1-5 ms")
    print("\n  Archivos generados en:", RESULTS_DIR)
    print(SEP)


if __name__ == "__main__":
    main()
