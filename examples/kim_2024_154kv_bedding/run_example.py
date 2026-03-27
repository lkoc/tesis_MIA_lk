"""Benchmark Kim, Nguyen Cong, Dinh & Kim (2024) — 6 cables XLPE 154 kV
con PAC cable bedding material.

Referencia:
    Kim, Y.-S., Nguyen Cong, H., Dinh, B.H. & Kim, H.-K. (2024).
    "Effect of ambient air and ground temperatures on heat transfer in
    underground power cable system buried in newly developed cable
    bedding material."  Geothermics 125, 103151.
    DOI: 10.1016/j.geothermics.2024.103151

Sistema modelado (caso critico verano — caso 5 del paper):
- 6 cables XLPE 154 kV 1200 mm² Cu en formacion dos-plano (two-flat)
  3 columnas × 2 filas dentro de un duct-bank
- I = 1026 A (ampacidad critica determinada por temperaturas reales)
- Cable bedding: PAC (prepacked aggregate concrete) λ = 2.094 W/(mK)
- Relleno entre cable y casing pipe: CLSM λ = 2.150 W/(mK)
- Suelo natural: 3 capas con λ promedio ~1.55 W/(mK)
- T_amb_air = 27.2 °C, T_sur = 17.0 °C, T_g = 15.2 °C (fondo)
- Conveccion forzada en superficie: h = 7.371 W/(m²K) (v = 1.17 m/s)

Simplificaciones vs. el modelo FEM original (COMSOL):
- Los cables tienen las mismas 4 capas internas (conductor, XLPE, screen,
  sheath) sin modelar casing pipe/CLSM explicitamente; su efecto se
  captura en el calor total back-calculated.
- Suelo efectivo uniforme k = 1.55 W/(mK) (promedio ponderado de las 3 capas)
- Dominio 91×45.5 m (Db = 44 m segun convergencia del paper, Fig. 8)
- Condiciones de frontera simplificadas:
  - Top: Robin (conveccion forzada, h=7.371, T_amb_air=27.2°C=300.35K)
  - Bottom: Dirichlet T_g = 15.2°C = 288.35 K
  - Left/Right: Dirichlet T ~ T_g = 288.35 K (aproximacion; paper usa T(z,t))

Resultados de referencia FEM (COMSOL, PAC, verano):
- T_max conductor (cable B1 central fondo) = 70.6 °C
- T_max conductor (cable T1 central arriba) ≈ 70.6 - 1.4 = 69.2 °C

Formulacion residual: T = T_bg + u
  T_bg : superposicion de Kennelly para 6 cables + perfil cilindr. por cable
  u    : correccion aprendida por la red neuronal

Soporta dos perfiles de ejecucion:

- **quick**    (~15-20 min CPU): 5 000 Adam + 500 L-BFGS, red 64x4
- **research** (~60-90 min CPU): 10 000 Adam + 1 000 L-BFGS, red 128x5

Uso::

    python examples/kim_2024_154kv_bedding/run_example.py
    python examples/kim_2024_154kv_bedding/run_example.py --profile research
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pinn_cables.io.readers import (  # noqa: E402
    CableLayer,
    load_problem,
    load_solver_params,
)
from pinn_cables.materials.props import (  # noqa: E402
    get_R_dc_20,
    get_alpha_R,
)
from pinn_cables.pinn.model import build_model  # noqa: E402
from pinn_cables.pinn.train import SteadyStatePINNTrainer  # noqa: E402
from pinn_cables.pinn.utils import (  # noqa: E402
    get_device,
    set_seed,
    setup_logging,
)
from pinn_cables.post.eval import evaluate_on_grid  # noqa: E402
from pinn_cables.post.plots import (  # noqa: E402
    plot_cable_geometry,
    plot_loss_history,
    plot_temperature_field,
)


# ---------------------------------------------------------------------------
# Paper constants — Kim et al. (2024) Table 10 case 5 (summer critical)
# ---------------------------------------------------------------------------
PAPER_T_MAX_PAC = 343.75     # K  (70.6 degC)  — max cable B1, PAC summer
PAPER_T_MAX_SAND = 350.75    # K  (77.6 degC)  — max cable B1, sand summer
PAPER_T_MAX_PAC_W = 323.05   # K  (49.9 degC)  — max cable B1, PAC winter
PAPER_T_MAX_SAND_W = 330.25  # K  (57.1 degC)  — max cable B1, sand winter
PAPER_T_MAX = 363.15         # K  (90 degC)  — XLPE limit
PAPER_W_D = 3.57             # W/m  — dielectric losses in XLPE
PAPER_K_XLPE = 0.2857        # W/(mK)
PAPER_K_SCREEN = 384.6       # W/(mK)
PAPER_FREQ = 50.0            # Hz
PAPER_CURRENT = 1026.0       # A — critical ampacity (case 5)
PAPER_T_AMB_AIR = 300.35     # K  (27.2 degC)
PAPER_T_SUR = 290.15         # K  (17.0 degC) — cable-surrounding ground temp
PAPER_T_G = 288.35           # K  (15.2 degC) — constant ground temp at depth
PAPER_K_PAC = 2.094           # W/(mK) — PAC thermal conductivity
PAPER_K_NC = 2.093            # W/(mK) — normal concrete
PAPER_K_SAND = 1.365          # W/(mK) — natural sand
PAPER_K_CLSM = 2.150          # W/(mK) — CLSM filling casing pipe
PAPER_K_SOIL_L1 = 1.804       # W/(mK) — soil layer 1 (SC)
PAPER_K_SOIL_L2 = 1.351       # W/(mK) — soil layer 2 (CL)
PAPER_K_SOIL_L3 = 1.517       # W/(mK) — soil layer 3 (CL)
PAPER_LOAD_LOSS_FACTOR = 0.8  # KEPCO DS-6210
PAPER_WIND_SPEED_SUMMER = 1.17  # m/s
# h_c = 7.371 + 6.43 * v^0.75 ≈ 7.371 + 7.52 ≈ 14.9 ... paper uses Eq. 27
# Actually: h_c = 7.371 * v^0.632 + 6.43 ... simplify to ~7.371 for low v

# Posiciones de los 6 cables (Fig. 7b del paper):
# Two-flat: fila inferior (B1, B2, B3) y fila superior (T1, T2, T3)
# Separacion entre centros de columna ≈ 0.40 m
# Separacion vertical entre filas ≈ 0.40 m
# Profundidad eje inferior ≈ 1.6 m, superior ≈ 1.2 m
CABLE_SEP_H = 0.40   # m — separacion horizontal centro a centro
CABLE_SEP_V = 0.40   # m — separacion vertical entre filas
DEPTH_BOTTOM = 1.6    # m — profundidad fila inferior
DEPTH_TOP = 1.2       # m — profundidad fila superior


def _compute_iec60287_Q(
    section_mm2: int,
    material: str,
    current_A: float,
    T_op: float,
    W_d: float,
    freq: float = 50.0,
) -> dict:
    """Calculo de calor total segun IEC 60287 simplificado.

    Incluye:
    - R(T) a la temperatura de operacion
    - Efecto piel (skin effect) para conductores solidos redondos
    - Perdidas dielectricas
    """
    R_dc_20 = get_R_dc_20(section_mm2, material)
    alpha_R = get_alpha_R(material)

    R_dc_T = R_dc_20 * (1.0 + alpha_R * (T_op - 293.15))

    xs_sq = 8.0 * math.pi * freq / (R_dc_T * 1e7)
    xs_4 = xs_sq ** 2
    ys = xs_4 / (192.0 + 0.8 * xs_4)

    R_ac = R_dc_T * (1.0 + ys)
    Q_cond = current_A ** 2 * R_ac
    Q_total = Q_cond + W_d

    return {
        "R_dc_20": R_dc_20,
        "R_dc_T": R_dc_T,
        "ys": ys,
        "R_ac": R_ac,
        "Q_cond_W_per_m": Q_cond,
        "W_d": W_d,
        "Q_total_W_per_m": Q_total,
        "ratio_vs_Rdc20": R_ac / R_dc_20,
    }


def _init_output_bias(model: nn.Module, value: float) -> None:
    """Inicializar la ultima capa lineal en *value* (warm-start)."""
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        last_linear.bias.data.fill_(value)


# ---------------------------------------------------------------------------
# T_bg analitico para 6 cables (superposicion Kennelly)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _multilayer_T_multi(
    xy: torch.Tensor,
    layers: list,
    placements: list,
    k_soil: float,
    T_amb: float,
    Q_lins: list[float],
    Q_d: float = 0.0,
) -> torch.Tensor:
    """Temperatura analitica multi-cable con superposicion de Kennelly.

    En el suelo: superpone las contribuciones de Kennelly de cada cable.
    Dentro de cada cable: perfil cilindrico 1D mas contribucion mutua.
    Q_d se distribuye volumetricamente en la capa XLPE.
    """
    N = xy.shape[0]
    r_sheath = layers[-1].r_outer

    # Fuente volumetrica dielectrica en XLPE
    q_vol_d = 0.0
    xlpe_ri = xlpe_ro = 0.0
    if Q_d > 0.0:
        for lyr in layers:
            if lyr.name == 'xlpe':
                xlpe_ri = lyr.r_inner
                xlpe_ro = lyr.r_outer
                q_vol_d = Q_d / (math.pi * (xlpe_ro ** 2 - xlpe_ri ** 2))
                break

    # --- 1) Contribucion del suelo por TODOS los cables ---
    dT_soil_total = torch.zeros(N, 1, device=xy.device, dtype=xy.dtype)
    for p, Q_lin in zip(placements, Q_lins):
        cx, cy = p.cx, p.cy
        d = abs(cy)
        dx = xy[:, 0:1] - cx
        dy_r = xy[:, 1:2] - cy
        dy_img = xy[:, 1:2] - d  # imagen respecto a y=0

        r_img_sq = (dx * dx + dy_img * dy_img).clamp(min=1e-20)
        r_sq_clamped = (dx * dx + dy_r * dy_r).clamp(min=r_sheath ** 2)
        dT_cable = Q_lin / (4.0 * math.pi * k_soil) * torch.log(
            r_img_sq / r_sq_clamped
        )
        dT_soil_total = dT_soil_total + dT_cable.clamp(min=0.0)

    T_soil = T_amb + dT_soil_total

    # --- 2) Dentro de cada cable: perfil cilindrico + mutuo ---
    result = T_soil

    for cable_idx, (p, Q_lin) in enumerate(zip(placements, Q_lins)):
        cx, cy = p.cx, p.cy
        d = abs(cy)
        dx = xy[:, 0:1] - cx
        dy_r = xy[:, 1:2] - cy
        r = torch.sqrt(dx * dx + dy_r * dy_r).clamp(min=1e-9)

        # dT mutuo por otros cables en la posicion de este cable
        dT_mutual = 0.0
        for j, (p2, Q2) in enumerate(zip(placements, Q_lins)):
            if j == cable_idx:
                continue
            cx2, cy2 = p2.cx, p2.cy
            d2 = abs(cy2)
            dx_m = cx - cx2
            dy_m = cy - cy2
            dy_img_m = cy - d2
            r_real = math.sqrt(dx_m ** 2 + dy_m ** 2)
            r_image = math.sqrt(dx_m ** 2 + dy_img_m ** 2)
            r_eff = max(r_real, r_sheath)
            dT_mutual += Q2 / (4.0 * math.pi * k_soil) * math.log(
                r_image ** 2 / r_eff ** 2
            )

        # T en la superficie exterior del sheath
        T_sheath_outer = (
            T_amb
            + Q_lin / (2.0 * math.pi * k_soil) * math.log(2.0 * d / r_sheath)
            + dT_mutual
        )

        # Temperaturas en el borde exterior de cada capa
        layer_T_outer: dict[str, float] = {}
        T_curr = T_sheath_outer
        for layer in reversed(layers):
            layer_T_outer[layer.name] = T_curr
            r_out = layer.r_outer
            r_in = max(layer.r_inner, 1e-9)
            if layer.name == 'xlpe' and Q_d > 0.0:
                Q_cond_eff = Q_lin - Q_d
                T_curr += (Q_cond_eff / (2.0 * math.pi * layer.k)
                           * math.log(r_out / r_in)
                           + q_vol_d / (2.0 * layer.k) * (
                               (r_out ** 2 - r_in ** 2) / 2.0
                               - r_in ** 2 * math.log(r_out / r_in)))
            elif layer.r_inner == 0.0 and Q_lin > 0.0:
                Q_cond_eff = Q_lin - Q_d
                Q_vol = Q_cond_eff / (math.pi * r_out ** 2)
                T_curr += Q_vol / (4.0 * layer.k) * r_out ** 2
            else:
                T_curr += Q_lin / (2.0 * math.pi * layer.k) * math.log(
                    r_out / r_in)

        # Asignar T dentro de cada capa de este cable
        for layer in reversed(layers):
            r_out = layer.r_outer
            r_in = max(layer.r_inner, 1e-9)
            T_out_layer = layer_T_outer[layer.name]
            mask = (r >= layer.r_inner) & (r < r_out)
            if layer.name == 'xlpe' and Q_d > 0.0:
                Q_cond_eff = Q_lin - Q_d
                r_c = r.clamp(min=r_in)
                log_ro_r = torch.log(r_out / r_c)
                T_layer = (T_out_layer
                           + Q_cond_eff / (2.0 * math.pi * layer.k) * log_ro_r
                           + q_vol_d / (2.0 * layer.k) * (
                               (r_out ** 2 - r_c ** 2) / 2.0
                               - r_in ** 2 * log_ro_r))
            elif layer.r_inner == 0.0 and Q_lin > 0.0:
                Q_cond_eff = Q_lin - Q_d
                Q_vol = Q_cond_eff / (math.pi * r_out ** 2)
                T_layer = T_out_layer + Q_vol / (4.0 * layer.k) * (
                    r_out ** 2 - r.clamp(min=r_in) ** 2
                )
            else:
                T_layer = T_out_layer + Q_lin / (2.0 * math.pi * layer.k) * torch.log(
                    r_out / r.clamp(min=r_in)
                )
            result = torch.where(mask, T_layer, result)

    return result


def _pretrain_multi_cable(
    model: nn.Module,
    placements: list,
    domain,
    layers: list,
    Q_lins: list[float],
    k_soil: float,
    T_amb: float,
    device: torch.device,
    normalize: bool,
    Q_d: float = 0.0,
    n_per_cable: int = 1500,
    n_bc: int = 200,
    n_steps: int = 500,
    lr: float = 1e-3,
) -> float:
    """Pre-entrenar con: interior de cada cable (T analitica) + contornos."""
    r_sheath = layers[-1].r_outer
    cable_pts = []

    for p in placements:
        angles = 2.0 * math.pi * torch.rand(
            n_per_cable, 1, device=device, dtype=torch.float32)
        us = torch.rand(n_per_cable, 1, device=device, dtype=torch.float32)
        rs = torch.sqrt(us) * r_sheath
        x_c = p.cx + rs * torch.cos(angles)
        y_c = p.cy + rs * torch.sin(angles)
        cable_pts.append(torch.cat([x_c, y_c], dim=1))

    xy_cable = torch.cat(cable_pts, dim=0)
    T_cable = _multilayer_T_multi(
        xy_cable, layers, placements, k_soil, T_amb, Q_lins, Q_d=Q_d)

    n_per = max(1, n_bc // 4)
    xmin, xmax = domain.xmin, domain.xmax
    ymin, ymax = domain.ymin, domain.ymax
    xh = xmin + (xmax - xmin) * torch.rand(
        n_per, 1, device=device, dtype=torch.float32)
    xh2 = xmin + (xmax - xmin) * torch.rand(
        n_per, 1, device=device, dtype=torch.float32)
    yv = ymin + (ymax - ymin) * torch.rand(
        n_per, 1, device=device, dtype=torch.float32)
    yv2 = ymin + (ymax - ymin) * torch.rand(
        n_per, 1, device=device, dtype=torch.float32)
    xy_bc = torch.cat([
        torch.cat([xh, torch.full_like(xh, ymax)], dim=1),
        torch.cat([xh2, torch.full_like(xh2, ymin)], dim=1),
        torch.cat([torch.full_like(yv, xmin), yv], dim=1),
        torch.cat([torch.full_like(yv2, xmax), yv2], dim=1),
    ], dim=0)
    T_bc = torch.full(
        (xy_bc.shape[0], 1), T_amb, device=device, dtype=torch.float32)

    xy_all = torch.cat([xy_cable, xy_bc], dim=0)
    T_all = torch.cat([T_cable, T_bc], dim=0)

    coord_mins = torch.tensor(
        [xmin, ymin], device=device, dtype=torch.float32)
    coord_maxs = torch.tensor(
        [xmax, ymax], device=device, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        xy_in = (2.0 * (xy_all - coord_mins) / (coord_maxs - coord_mins) - 1.0
                 if normalize else xy_all)
        loss = torch.mean((model(xy_in) - T_all) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        xy_in = (2.0 * (xy_all - coord_mins) / (coord_maxs - coord_mins) - 1.0
                 if normalize else xy_all)
        T_pred = model(xy_in)
        rmse = float(torch.sqrt(torch.mean((T_pred - T_all) ** 2)).item())
    return rmse


class ResidualPINNModel(nn.Module):
    """PINN que aprende la correccion u = T - T_analitico (multi-cable)."""

    def __init__(
        self,
        base: nn.Module,
        layers: list,
        placements: list,
        k_soil: float,
        T_amb: float,
        Q_lins: list[float],
        domain,
        normalize: bool = True,
        Q_d: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self._layers = layers
        self._placements = placements
        self._k_soil = k_soil
        self._T_amb = T_amb
        self._Q_lins = Q_lins
        self._Q_d = Q_d
        self._normalize = normalize
        self._xmin = domain.xmin
        self._xmax = domain.xmax
        self._ymin = domain.ymin
        self._ymax = domain.ymax

    def _denormalize(self, xy_n: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor(
            [self._xmin, self._ymin], device=xy_n.device, dtype=xy_n.dtype)
        hi = torch.tensor(
            [self._xmax, self._ymax], device=xy_n.device, dtype=xy_n.dtype)
        return (xy_n + 1.0) * 0.5 * (hi - lo) + lo

    def forward(self, xy_in: torch.Tensor) -> torch.Tensor:
        xy_phys = self._denormalize(xy_in) if self._normalize else xy_in
        T_bg = _multilayer_T_multi(
            xy_phys,
            self._layers,
            self._placements,
            self._k_soil,
            self._T_amb,
            self._Q_lins,
            Q_d=self._Q_d,
        )
        u = self.base(xy_in)
        return T_bg + u


def _iec60287_estimate_6cable(
    layers, placements, k_soil: float, Q_lins: list[float], T_amb: float,
    Q_d: float = 0.0,
) -> dict:
    """Estimacion analitica IEC 60287 para 6 cables (two-flat).

    Calcula temperatura del conductor con maxima interferencia termica
    mutua (cable B1 — central fila inferior, indice 1).
    """
    r_sheath = layers[-1].r_outer

    # Cable B1 central bottom es el peor caso (indice 1: cx=0, cy=-1.6)
    worst_idx = 1
    p_worst = placements[worst_idx]
    Q_worst = Q_lins[worst_idx]
    d = abs(p_worst.cy)

    # dT por capas cilindricas
    dT_layers: dict[str, float] = {}
    for layer in layers:
        r_in = max(layer.r_inner, 1e-9)
        r_out = layer.r_outer
        if r_out <= r_in:
            continue
        if layer.name == 'xlpe' and Q_d > 0.0:
            Q_cond_eff = Q_worst - Q_d
            q_vol_d = Q_d / (math.pi * (r_out ** 2 - r_in ** 2))
            dT_layers[layer.name] = (
                Q_cond_eff / (2.0 * math.pi * layer.k) * math.log(r_out / r_in)
                + q_vol_d / (2.0 * layer.k) * (
                    (r_out ** 2 - r_in ** 2) / 2.0
                    - r_in ** 2 * math.log(r_out / r_in)))
        elif layer.r_inner == 0.0:
            Q_cond_eff = Q_worst - Q_d
            Q_vol_c = Q_cond_eff / (math.pi * r_out ** 2)
            dT_layers[layer.name] = Q_vol_c / (4.0 * layer.k) * r_out ** 2
        else:
            dT_layers[layer.name] = Q_worst / (2.0 * math.pi * layer.k) * math.log(
                r_out / r_in)

    # dT propio en suelo (Kennelly)
    dT_soil_self = Q_worst / (2.0 * math.pi * k_soil) * math.log(
        2.0 * d / r_sheath)

    # dT mutuo por los otros 5 cables
    dT_mutual = 0.0
    for j, (pj, Qj) in enumerate(zip(placements, Q_lins)):
        if j == worst_idx:
            continue
        cx_j, cy_j = pj.cx, pj.cy
        d_j = abs(cy_j)
        dx = p_worst.cx - cx_j
        dy = p_worst.cy - cy_j
        r_real = math.sqrt(dx ** 2 + dy ** 2)
        r_image = math.sqrt(dx ** 2 + (p_worst.cy - d_j) ** 2)
        dT_mutual += Qj / (4.0 * math.pi * k_soil) * math.log(
            r_image ** 2 / max(r_real, r_sheath) ** 2)

    dT_total = sum(dT_layers.values()) + dT_soil_self + dT_mutual
    T_cond = T_amb + dT_total
    return {
        "worst_idx": worst_idx,
        "Q_lin_W_per_m": Q_worst,
        "dT_by_layer": dT_layers,
        "dT_soil_self": dT_soil_self,
        "dT_mutual": dT_mutual,
        "dT_total": dT_total,
        "T_cond_K": T_cond,
    }


def main() -> None:
    """Cargar datos, entrenar PINN y comparar con Kim et al. (2024)."""
    parser = argparse.ArgumentParser(
        description="Benchmark PINN: Kim et al. (2024) — 6 cables XLPE 154 kV two-flat, PAC bedding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Perfiles disponibles:\n"
            "  quick    : 5 000 Adam + 500 L-BFGS, red 64x4\n"
            "  research : 10 000 Adam + 1 000 L-BFGS, red 128x5\n"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "research"],
        default="quick",
        help="Perfil de ejecucion (default: quick)",
    )
    args = parser.parse_args()
    profile = args.profile

    RESULTS_DIR = HERE / ("results" if profile == "quick" else "results_research")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar problema fisico (CSV)
    problem = load_problem(DATA_DIR)
    scenario = problem.scenarios[0]
    all_placements = problem.placements

    # Capas del cable (compartidas — mismo tipo de cable 1200mm² Cu)
    layers_template = problem.get_layers(all_placements[0].cable_id)

    section_mm2 = all_placements[0].section_mm2
    material = all_placements[0].conductor_material.upper()
    material_lc = all_placements[0].conductor_material.strip().lower()
    current_A = all_placements[0].current_A
    n_cables = len(all_placements)

    # -----------------------------------------------------------------
    # Calculo de calor del paper
    # -----------------------------------------------------------------
    # IEC 60287 a T_op = 70.6°C (temperatura max del conductor con PAC)
    T_op_K = PAPER_T_MAX_PAC
    iec_q = _compute_iec60287_Q(
        section_mm2, material_lc, current_A,
        T_op=T_op_K, W_d=PAPER_W_D, freq=PAPER_FREQ,
    )

    # Usar Q_cond del IEC para el caso de 1026A @ 70.6°C
    Q_cond_iec = iec_q["Q_cond_W_per_m"]
    Q_d = PAPER_W_D
    Q_total_lin = Q_cond_iec + Q_d

    # Sobreescribir Q_vol del conductor
    conductor = layers_template[0]
    A_cond = math.pi * conductor.r_outer ** 2
    Q_vol_corrected = Q_total_lin / A_cond

    layers = [
        CableLayer(
            name=conductor.name,
            r_inner=conductor.r_inner,
            r_outer=conductor.r_outer,
            k=conductor.k,
            rho_c=conductor.rho_c,
            Q=Q_vol_corrected,
        ),
    ] + list(layers_template[1:])

    Q_lins = [Q_total_lin] * n_cables

    SEP = "=" * 72
    print(SEP)
    print("  BENCHMARK: Kim, Nguyen Cong, Dinh & Kim (2024)")
    print("  154 kV XLPE 6 cables two-flat + PAC bedding — PINN vs FEM")
    print("  Cable XLPE 1200 mm2 %s / I = %.0f A (critical ampacity)" % (
        material, current_A))
    print("  Perfil de ejecucion : %s" % profile.upper())
    print(SEP)

    # Cargar config
    params_csv = DATA_DIR / (
        "solver_params.csv" if profile == "quick" else "solver_params_research.csv"
    )
    solver_params = load_solver_params(params_csv)
    solver_cfg = solver_params.to_solver_cfg()

    adam_n = solver_cfg["training"]["adam_steps"]
    lbfgs_n = solver_cfg["training"]["lbfgs_steps"]
    print_ev = solver_cfg["training"]["print_every"]
    width = solver_cfg["model"]["width"]
    depth = solver_cfg["model"]["depth"]
    n_int = solver_cfg["sampling"]["n_interior"]
    n_ifc = solver_cfg["sampling"]["n_interface"]
    n_bnd = solver_cfg["sampling"]["n_boundary"]

    print("\n  Problema fisico:")
    print("  Escenario       : %s  (%s)" % (scenario.scenario_id, scenario.mode))
    print("  Cables          : %d x XLPE 1200 mm2 %s  I=%.0f A" % (
        n_cables, material, current_A))
    print("  Formacion       : two-flat (3 col x 2 filas), s_h=%.2f m, s_v=%.2f m" % (
        CABLE_SEP_H, CABLE_SEP_V))
    print("  Cable bedding   : PAC (lambda=%.3f W/(mK))" % PAPER_K_PAC)
    print()
    print("  Fuentes de calor (IEC 60287 @ T_op=%.1f degC):" % (T_op_K - 273.15))
    print("  Q_cond (I^2Rac) : %.2f W/m" % Q_cond_iec)
    print("  Q_d XLPE (vol.) : %.2f W/m" % Q_d)
    print("  Q_TOTAL (x1)    : %.2f W/m  (x%d = %.2f W/m)" % (
        Q_total_lin, n_cables, Q_total_lin * n_cables))
    print()
    print("  k_suelo efectivo: %.2f W/(m*K)" % scenario.k_soil)
    print("  T_amb (suelo)   : %.1f degC  (%.2f K)" % (
        scenario.T_amb - 273.15, scenario.T_amb))
    print("  T_amb_air       : %.1f degC  (conveccion top)" % (
        PAPER_T_AMB_AIR - 273.15))
    print("  T_g (fondo)     : %.1f degC = %.2f K" % (
        PAPER_T_G - 273.15, PAPER_T_G))
    print("  Posiciones:")
    for i, p in enumerate(all_placements):
        row = "B" if abs(p.cy) > 1.3 else "T"
        col = {-0.40: "izq", 0.00: "cen", 0.40: "der"}.get(round(p.cx, 2), "?")
        print("    Cable %d (%s%s): cx=%+.2f, cy=%.2f m" % (
            i + 1, row, col, p.cx, p.cy))
    print("  Dominio         : [%.1f, %.1f] x [%.1f, %.0f] m" % (
        problem.domain.xmin, problem.domain.xmax,
        problem.domain.ymin, problem.domain.ymax))

    # Estimacion analitica
    iec = _iec60287_estimate_6cable(
        layers, all_placements, scenario.k_soil, Q_lins, scenario.T_amb,
        Q_d=Q_d,
    )
    T_ref_K = iec["T_cond_K"]

    print("\n  Referencia analitica (cable B1 central bottom, peor caso):")
    print("  Q total (lin.)   : %.2f W/m (cond %.2f + diel %.2f)" % (
        iec["Q_lin_W_per_m"], Q_cond_iec, Q_d))
    for name, dT in iec["dT_by_layer"].items():
        print("  dT %-10s   : %+.2f K" % (name, dT))
    print("  dT suelo propio  : %+.2f K" % iec["dT_soil_self"])
    print("  dT mutual (otros): %+.2f K" % iec["dT_mutual"])
    print("  dT TOTAL         : %+.2f K  -->  T_cond = %.1f K (%.1f degC)" % (
        iec["dT_total"], T_ref_K, T_ref_K - 273.15))

    print("\n  Referencia FEM Kim et al. (2024):")
    print("  T_max cable B1 (PAC, verano)  : %.1f degC" % (
        PAPER_T_MAX_PAC - 273.15))
    print("  T_max cable B1 (sand, verano) : %.1f degC" % (
        PAPER_T_MAX_SAND - 273.15))
    print("  T_max cable B1 (PAC, invierno): %.1f degC" % (
        PAPER_T_MAX_PAC_W - 273.15))

    # Configuracion y modelo
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="example_kim2024_" + profile)

    normalize = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    base_model = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModel(
        base_model,
        layers,
        all_placements,
        scenario.k_soil,
        scenario.T_amb,
        Q_lins,
        problem.domain,
        normalize=normalize,
        Q_d=Q_d,
    )
    _init_output_bias(model.base, 0.0)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n  Pre-entrenando en perfil multicable (500 pasos)...", flush=True)
    rmse_pre = _pretrain_multi_cable(
        model, all_placements, problem.domain,
        layers, Q_lins, scenario.k_soil, scenario.T_amb,
        device=device, normalize=normalize, Q_d=Q_d,
        n_per_cable=1500, n_bc=200, n_steps=500, lr=1e-3,
    )
    print("  Pre-entrenamiento OK: RMSE = %.3f K" % rmse_pre, flush=True)

    print("\n  Configuracion del solver:")
    print("  Red neuronal : MLP %dx%d  (%d params)" % (width, depth, n_params))
    print("  Puntos       : int=%d  ifc=%d  bnd=%d" % (n_int, n_ifc, n_bnd))
    print("  Entrenamiento: %d Adam + %d L-BFGS" % (adam_n, lbfgs_n))
    print("  Avance cada  : %d pasos Adam" % print_ev)
    logger.info(
        "Device=%s | Perfil=%s | red MLP%dx%d (%d params)",
        device, profile, width, depth, n_params,
    )

    # Geometria
    geo_path = RESULTS_DIR / "geometry.png"
    plot_cable_geometry(
        layers, all_placements[0], problem.domain,
        title="6 Cables XLPE 154 kV Two-Flat + PAC — Kim et al. (2024)",
        save_path=geo_path,
    )

    # Entrenamiento PINN
    print("\n" + "-" * 72)
    print("  ENTRENAMIENTO  (Adam --> L-BFGS)")
    print("  Columnas del log:  [fase paso/total pct%%] loss  pde  bc  ifc")
    print("-" * 72)
    trainer = SteadyStatePINNTrainer(
        model=model,
        layers=layers,
        placement=all_placements[0],
        domain=problem.domain,
        soil=problem.soil,
        bcs=problem.bcs,
        scenario=scenario,
        solver_cfg=solver_cfg,
        device=device,
        logger=logger,
    )
    history = trainer.train()
    print("-" * 72)

    # Guardar modelo
    model_path = RESULTS_DIR / "model_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo guardado: %s", model_path)

    # Graficas
    loss_path = RESULTS_DIR / "loss_history.png"
    T_map_path = RESULTS_DIR / "temperature_field.png"
    plot_loss_history(
        history,
        title="Perdida — Kim et al. (2024) 154 kV Two-Flat PAC (%s)" % profile,
        save_path=loss_path,
    )
    X, Y, T = evaluate_on_grid(
        model, problem.domain, nx=300, ny=300,
        device=device, normalize=normalize,
    )
    plot_temperature_field(
        X, Y, T,
        title="T(x,y) [K] — 6 x XLPE 1200 mm2 %s %.0f A PAC [%s]" % (
            material, current_A, profile),
        save_path=T_map_path,
    )

    # -----------------------------------------------------------------
    # Resumen comparativo
    # -----------------------------------------------------------------
    T_max_pinn = float(T.max())
    T_min_pinn = float(T.min())

    # Evaluar T en el centro de CADA conductor
    T_cond_pinns = []
    with torch.no_grad():
        model.eval()
        for p in all_placements:
            pt = torch.tensor(
                [[p.cx, p.cy]], device=device, dtype=torch.float32)
            if normalize:
                mins = torch.tensor(
                    [problem.domain.xmin, problem.domain.ymin],
                    device=device, dtype=torch.float32)
                maxs = torch.tensor(
                    [problem.domain.xmax, problem.domain.ymax],
                    device=device, dtype=torch.float32)
                pt_in = 2.0 * (pt - mins) / (maxs - mins) - 1.0
            else:
                pt_in = pt
            T_cond_pinns.append(float(model(pt_in).item()))

    # Cable B1 (central bottom, indice 1) es el peor caso
    T_cond_worst = T_cond_pinns[iec["worst_idx"]]
    T_cond_max = max(T_cond_pinns)
    loss_final = history["total"][-1]
    error_K = T_cond_worst - T_ref_K
    error_vs_fem = T_cond_worst - PAPER_T_MAX_PAC

    C = 42
    print("\n" + SEP)
    print("  RESULTADOS — BENCHMARK Kim et al. (2024)")
    print("  6 x Cable XLPE 154 kV 1200 mm2 %s / I = %d A / Two-Flat + PAC" % (
        material, int(current_A)))
    print(SEP)

    print("\n  --- Temperaturas por cable ---")
    for i, (p, Tc) in enumerate(zip(all_placements, T_cond_pinns)):
        row = "B" if abs(p.cy) > 1.3 else "T"
        col = {-0.40: "izq", 0.00: "cen", 0.40: "der"}.get(round(p.cx, 2), "?")
        tag = " (*)" if i == iec["worst_idx"] else ""
        print("  Cable %d (%s%s, cx=%+.2f cy=%.2f)%s : T = %.2f K (%.1f degC)" % (
            i + 1, row, col, p.cx, p.cy, tag, Tc, Tc - 273.15))

    print("\n  %-*s  %10s  %10s  %9s" % (C, "Magnitud", "PINN", "Referencia", "Error"))
    print("  " + "-" * 70)

    print("\n  --- PINN vs IEC 60287 analitico (cable B1) ---")
    print("  %-*s  %10.2f  %10.2f  %+8.2f K" % (
        C, "T conductor (K)", T_cond_worst, T_ref_K, error_K))
    print("  %-*s  %10.1f  %10.1f  %+8.1f K" % (
        C, "T conductor (degC)",
        T_cond_worst - 273.15, T_ref_K - 273.15, error_K))

    print("\n  --- PINN vs FEM COMSOL (Kim et al., PAC verano) ---")
    print("  %-*s  %10.2f  %10.2f  %+8.2f K" % (
        C, "T conductor (K)", T_cond_worst, PAPER_T_MAX_PAC, error_vs_fem))
    print("  %-*s  %10.1f  %10.1f  %+8.1f K" % (
        C, "T conductor (degC)",
        T_cond_worst - 273.15, PAPER_T_MAX_PAC - 273.15, error_vs_fem))

    print("\n  --- Comparacion entre metodos ---")
    print("  %-*s  %10s  %10s  %10s" % (C, "Metodo", "I (A)", "T_max", "Bedding"))
    print("  " + "-" * 70)
    print("  %-*s  %10d  %9.1f degC  %7s" % (
        C, "FEM COMSOL (PAC, verano)", int(current_A),
        PAPER_T_MAX_PAC - 273.15, "PAC"))
    print("  %-*s  %10d  %9.1f degC  %7s" % (
        C, "FEM COMSOL (sand, verano)", int(current_A),
        PAPER_T_MAX_SAND - 273.15, "sand"))
    print("  %-*s  %10d  %9.1f degC  %7s" % (
        C, "IEC analitico (este script)", int(current_A),
        T_ref_K - 273.15, "homog."))
    print("  %-*s  %10d  %9.1f degC  %7s" % (
        C, "PINN (cable B1)", int(current_A),
        T_cond_worst - 273.15, "homog."))

    print("\n  " + "-" * 70)
    print("  %-*s  %10.2f" % (C, "T max dominio (K)", T_max_pinn))
    print("  %-*s  %10.2f" % (C, "T min dominio (K)", T_min_pinn))
    print("  %-*s  %10.4e" % (C, "Perdida final", loss_final))
    print("  " + "-" * 70)
    print("  Limite IEC XLPE: 363 K (90 degC)")
    print("  Margen termico PINN (B1): %.1f K" % (363.0 - T_cond_worst))
    print("  Reduccion PAC vs sand (paper): %.1f K" % (
        PAPER_T_MAX_SAND - PAPER_T_MAX_PAC))

    if profile == "quick":
        print()
        print("  Para resultados de investigacion:")
        print("    python examples/kim_2024_154kv_bedding/run_example.py --profile research")
    print()
    print("  Graficas guardadas en: %s" % RESULTS_DIR)
    print("    - geometry.png  |  loss_history.png  |  temperature_field.png")
    print("    - model_final.pt  (pesos de la red entrenada)")


if __name__ == "__main__":
    main()
