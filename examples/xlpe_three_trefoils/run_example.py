"""Ejemplo tres circuitos XLPE 12/20 kV en formacion trefoil separados.

Nueve cables (3 circuitos de 3 cables cada uno) enterrados en
agrupacion trefoil tocante por circuito, con una separacion tipica de 0.30 m
entre los centroides de circuito adyacente y el centroide del conjunto a
70 cm de profundidad.  Soporta cables de diferente seccion (95, 150, 240,
400, 600 mm2), material (cu/al) y corriente individual, especificados
en ``cables_placement.csv``.

Efectos modelados:
- Calentamiento mutuo entre los 9 cables (superposicion Kennelly)
- Resistencia electrica R(T): R aumenta con temperatura, incrementando Q
- k(x,y) espacialmente variable: region buena (zona del conjunto de 9 cables)
  rodeada de suelo con k menor; transicion suave sigmoide

Formulacion residual: T_total = T_bg + u
- T_bg : superposicion de Kennelly (N cables) + perfil multicapa por cable
- u    : correccion de dominio finito aprendida por la red neuronal
- El PINN solo aprende u -- la solucion trivialmente correcta es u = 0.

Archivos de datos requeridos (directorio examples/xlpe_three_trefoils/data/):
  cable_layers.csv          -- geometria y propiedades termicas del cable
  cables_placement.csv      -- posicion de los 9 cables (3 circuitos x 3)
  boundary_conditions.csv   -- CCF del dominio
  domain.csv                -- limites del dominio
  scenarios.csv             -- escenarios de operacion
  soil_properties.csv       -- propiedades del suelo
  physics_params.csv        -- parametros R(T) y k(x,y) sigmoide
  solver_params.csv         -- hiperparametros del solver (perfil quick)
  solver_params_research.csv-- hiperparametros del solver (perfil research)

Circuitos y cables (separacion entre centroides = 0.30 m):
  Circuito 1 (x=-0.30 m): cables 1 (top), 2 (bot-izq), 3 (bot-der)
  Circuito 2 (x= 0.00 m): cables 4 (top), 5 (bot-izq), 6 (bot-der)
  Circuito 3 (x=+0.30 m): cables 7 (top), 8 (bot-izq), 9 (bot-der)

Separacion libre entre circuitos adyacentes: ~0.24 m (superficie a superficie).
Suelo mejorado: region de k=1.5 W/(mK) alrededor del conjunto 0.90 x 0.55 m,
resto del suelo k=0.8 W/(mK).

Soporta dos perfiles de ejecucion:

- **quick**    (~15-20 min CPU) : 5 000 Adam, red 64x4
- **research** (~60-90 min CPU): 10 000 Adam + 500 L-BFGS, red 128x5

Uso::

    python examples/xlpe_three_trefoils/run_example.py
    python examples/xlpe_three_trefoils/run_example.py --profile research

Referencia IEC 60287: T_max admisible XLPE = 90 degC (363 K).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pinn_cables.io.readers import load_problem, load_solver_params  # noqa: E402
from pinn_cables.pinn.model import build_model  # noqa: E402
from pinn_cables.pinn.pde import pde_residual_steady  # noqa: E402
from pinn_cables.pinn.utils import get_device, set_seed, setup_logging  # noqa: E402
from pinn_cables.post.eval import evaluate_on_grid  # noqa: E402
from pinn_cables.post.plots import plot_loss_history, plot_temperature_field  # noqa: E402


# ---------------------------------------------------------------------------
# Parametros fisicos adicionales: R(T) + k sigmoide
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PhysicsParams:
    """Parametros fisicos adicionales: resistencia R(T) y conductividad sigmoide k(x,y).

    R(T): la resistencia del conductor aumenta linealmente con la temperatura,
    incrementando la potencia disipada segun Q_lin = I^2 * R(T).

    k(x,y): la conductividad termica del suelo varía segun la distancia a una
    region 'buena' centrada en (k_cx, k_cy) de dimension k_width x k_height.
    La transicion usa una funcion sigmoide para evitar discontinuidades.
    Formula:  d = max(|x-k_cx| - k_width/2, |y-k_cy| - k_height/2)
              k = k_bad + (k_good - k_bad) * sigmoid(-d / k_transition)
    """
    # --- R(T) ---
    I_A: float = 0.0            # Corriente fallback [A] (0 = usar current_A del CSV)
    R_ref: float = 0.0          # R_dc fallback [Ohm/m] (0 = usar catalogo)
    T_ref_R_K: float = 293.15   # Temperatura de referencia de R [K]
    alpha_R: float = 0.00393    # Coeficiente de temperatura fallback [1/K]
    n_R_iter: int = 2           # Iteraciones para la auto-consistencia R(T) (0 = sin R(T))

    # --- k(x,y) sigmoide ---
    k_variable: bool = True     # Activar k(x,y) variable
    k_good: float = 1.5         # k en region buena [W/(m*K)]
    k_bad: float = 0.8          # k en region mala  [W/(m*K)]
    k_cx: float = 0.0           # Centro x de la region buena [m]
    k_cy: float = -0.70         # Centro y de la region buena [m]
    k_width: float = 0.5        # Ancho de la region buena [m]
    k_height: float = 0.5       # Alto  de la region buena [m]
    k_transition: float = 0.05  # Escala de transicion sigmoide [m]


def load_physics_params(path: Path) -> PhysicsParams:
    """Cargar PhysicsParams desde archivo CSV de pares param,value.

    Parametros no presentes en el archivo usan los valores por defecto de la dataclass.
    Soporta: float (int, float), bool (true/false), int.
    """
    if not path.exists():
        return PhysicsParams()
    raw: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw[row["param"].strip()] = row["value"].strip()

    fields = {f.name: f for f in dataclasses.fields(PhysicsParams)}
    kwargs: dict = {}
    for key, val_str in raw.items():
        if key not in fields:
            continue
        ft = fields[key].type
        if ft in (bool, "bool"):
            kwargs[key] = val_str.lower() in ("true", "1", "yes")
        elif ft in (int, "int"):
            kwargs[key] = int(val_str)
        else:
            try:
                kwargs[key] = float(val_str)
            except ValueError:
                kwargs[key] = val_str
    return PhysicsParams(**kwargs)


# ---------------------------------------------------------------------------
# k(x,y) sigmoide: version escalar y tensor
# ---------------------------------------------------------------------------

def k_scalar(x: float, y: float, pp: PhysicsParams) -> float:
    """k(x, y) escalar Python; NO diferenciable por PyTorch (uso en fondo analitico)."""
    if not pp.k_variable:
        return pp.k_bad  # sera sobreescrito por k_soil del escenario si no k_variable
    dx = abs(x - pp.k_cx) - pp.k_width  / 2.0
    dy = abs(y - pp.k_cy) - pp.k_height / 2.0
    d  = max(dx, dy)
    sig = 1.0 / (1.0 + math.exp(d / pp.k_transition))
    return pp.k_bad + (pp.k_good - pp.k_bad) * sig


def k_tensor(xy: torch.Tensor, pp: PhysicsParams) -> torch.Tensor:
    """k(x, y) como tensor diferenciable (N,1); correcto para la perdida PDE."""
    dx = (xy[:, 0:1] - pp.k_cx).abs() - pp.k_width  / 2.0
    dy = (xy[:, 1:2] - pp.k_cy).abs() - pp.k_height / 2.0
    d  = torch.max(dx, dy)
    sig = torch.sigmoid(-d / pp.k_transition)
    return pp.k_bad + (pp.k_good - pp.k_bad) * sig


# ---------------------------------------------------------------------------
# Potencia lineal con resistencia R(T)
# ---------------------------------------------------------------------------

def Q_lin_from_I(I: float, R_ref: float, alpha_R: float,
                  T_cond: float, T_ref: float) -> float:
    """Q lineal [W/m] = I^2 * R(T);  R(T) = R_ref * (1 + alpha_R*(T-T_ref))."""
    R_T = R_ref * (1.0 + alpha_R * (T_cond - T_ref))
    return I * I * R_T


# ---------------------------------------------------------------------------
# Temperatura analitica multicable (formulacion residual)
# ---------------------------------------------------------------------------

def _init_output_bias(model: nn.Module, value: float) -> None:
    """Inicializar la ultima capa lineal en *value* (warm-start cerca de cero)."""
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        last_linear.bias.data.fill_(value)


@torch.no_grad()
def _multilayer_T_multi(
    xy: torch.Tensor,
    layers_list: list[list],
    placements: list,
    k_soil: float,
    T_amb: float,
    Q_lins: list[float],
) -> torch.Tensor:
    """Temperatura analitica para N cables: superposicion Kennelly + multicapa.

    Soporta cables de diferente tipo (diferentes capas) y diferente corriente
    (diferentes Q_lin por cable).

    Args:
        layers_list: Lista de listas de capas, una por cable.
        Q_lins:      Lista de potencias lineales [W/m], una por cable.

    Returns:
        Tensor (N, 1) de temperaturas en K.
    """
    N = xy.shape[0]
    result = torch.full((N, 1), T_amb, device=xy.device, dtype=xy.dtype)
    r_sheaths = [ls[-1].r_outer for ls in layers_list]

    # T_sheath_outer_i (escalar) por cable
    T_sheath_outers: list[float] = []
    for i, pl_i in enumerate(placements):
        d_i = abs(pl_i.cy)
        Q_i = Q_lins[i]
        r_sh_i = r_sheaths[i]
        T_s_i = T_amb + Q_i / (2.0 * math.pi * k_soil) * math.log(2.0 * d_i / r_sh_i)
        for j, pl_j in enumerate(placements):
            if i == j:
                continue
            Q_j = Q_lins[j]
            d_j = abs(pl_j.cy)
            dist_ij = math.sqrt((pl_i.cx - pl_j.cx) ** 2 + (pl_i.cy - pl_j.cy) ** 2)
            r_img_sq = (pl_i.cx - pl_j.cx) ** 2 + (pl_i.cy - d_j) ** 2
            T_s_i += Q_j / (4.0 * math.pi * k_soil) * math.log(r_img_sq / dist_ij ** 2)
        T_sheath_outers.append(T_s_i)

    # Temperaturas en interfaces de capa de cada cable (exterior -> interior)
    layer_T_outer_list: list[dict[str, float]] = []
    for i in range(len(placements)):
        layers = layers_list[i]
        Q_i = Q_lins[i]
        T_curr = T_sheath_outers[i]
        layer_T_outer: dict[str, float] = {}
        for layer in reversed(layers):
            layer_T_outer[layer.name] = T_curr
            r_out = layer.r_outer
            r_in  = max(layer.r_inner, 1e-9)
            if layer.r_inner == 0.0 and Q_i > 0.0:
                Q_vol = Q_i / (math.pi * r_out ** 2)
                T_curr += Q_vol / (4.0 * layer.k) * r_out ** 2
            else:
                T_curr += Q_i / (2.0 * math.pi * layer.k) * math.log(r_out / r_in)
        layer_T_outer_list.append(layer_T_outer)

    # Distancias radiales por cable
    r_per_cable: list[torch.Tensor] = []
    for pl in placements:
        dx = xy[:, 0:1] - pl.cx
        dy = xy[:, 1:2] - pl.cy
        r_per_cable.append(torch.sqrt(dx * dx + dy * dy).clamp(min=1e-9))

    # Mascara suelo: puntos fuera de TODOS los cables
    soil_mask = torch.ones(N, dtype=torch.bool, device=xy.device)
    for i, r_i in enumerate(r_per_cable):
        soil_mask &= (r_i.squeeze(1) >= r_sheaths[i])

    # Suelo: superposicion de Kennelly de todos los cables
    dT_soil = torch.zeros(N, 1, device=xy.device, dtype=xy.dtype)
    for i, pl in enumerate(placements):
        Q_i = Q_lins[i]
        r_sh_i = r_sheaths[i]
        dx     = xy[:, 0:1] - pl.cx
        d_pl   = abs(pl.cy)
        dy_img = xy[:, 1:2] - d_pl
        dy_r   = xy[:, 1:2] - pl.cy
        r_img_sq = (dx * dx + dy_img * dy_img).clamp(min=1e-20)
        r_sq     = (dx * dx + dy_r  * dy_r).clamp(min=r_sh_i ** 2)
        dT_soil += Q_i / (4.0 * math.pi * k_soil) * torch.log(r_img_sq / r_sq)
    result[soil_mask] = (T_amb + dT_soil[soil_mask].clamp(min=0.0))

    # Interior de cada cable: perfil cilindrico 1D
    for i, pl in enumerate(placements):
        layers = layers_list[i]
        Q_i = Q_lins[i]
        r = r_per_cable[i]
        r_sh_i = r_sheaths[i]
        layer_T_outer = layer_T_outer_list[i]
        cable_mask = r.squeeze(1) < r_sh_i

        for layer in reversed(layers):
            r_out = layer.r_outer
            r_in  = max(layer.r_inner, 1e-9)
            T_out_layer = layer_T_outer[layer.name]
            mask = ((r >= layer.r_inner) & (r < r_out)).squeeze(1) & cable_mask
            if not mask.any():
                continue
            r_pts = r[mask, 0]
            if layer.r_inner == 0.0 and Q_i > 0.0:
                Q_vol = Q_i / (math.pi * r_out ** 2)
                dT = Q_vol / (4.0 * layer.k) * (r_out ** 2 - r_pts ** 2)
            else:
                dT = Q_i / (2.0 * math.pi * layer.k) * torch.log(
                    r_out / r_pts.clamp(min=r_in)
                )
            result[mask, 0] = T_out_layer + dT

    return result


# ---------------------------------------------------------------------------
# Pre-entrenamiento: todos los interiores de cable + contornos del dominio
# ---------------------------------------------------------------------------

def _pretrain_cables(
    model: nn.Module,
    placements: list,
    domain,
    layers_list: list[list],
    Q_lins: list[float],
    k_soil: float,
    T_amb: float,
    device: torch.device,
    normalize: bool,
    n_cable_per: int = 1000,
    n_bc: int = 200,
    n_steps: int = 800,
    lr: float = 1e-3,
) -> float:
    """Pre-entrenar en: interiores de los N cables (T analitica) + 4 bordes (T_amb).

    Con la formulacion residual, el objetivo para el modelo completo es T_analitica
    (= T_bg + 0), lo que implica que la correccion u debe converger a cero.
    Devuelve el RMSE final (K).
    """
    xmin, xmax = domain.xmin, domain.xmax
    ymin, ymax = domain.ymin, domain.ymax

    # Puntos dentro de cada cable
    cable_pts_list: list[torch.Tensor] = []
    cable_T_list:   list[torch.Tensor] = []
    for idx, pl in enumerate(placements):
        r_sheath_i = layers_list[idx][-1].r_outer
        angles = 2.0 * math.pi * torch.rand(n_cable_per, 1, device=device, dtype=torch.float32)
        us     = torch.rand(n_cable_per, 1, device=device, dtype=torch.float32)
        rs     = torch.sqrt(us) * r_sheath_i
        x_c    = pl.cx + rs * torch.cos(angles)
        y_c    = pl.cy + rs * torch.sin(angles)
        xy_c   = torch.cat([x_c, y_c], dim=1)
        T_c    = _multilayer_T_multi(xy_c, layers_list, placements, k_soil, T_amb, Q_lins)
        cable_pts_list.append(xy_c)
        cable_T_list.append(T_c)

    # Puntos sobre los cuatro bordes del dominio: T_amb
    n_per = max(1, n_bc // 4)
    xh    = xmin + (xmax - xmin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    xh2   = xmin + (xmax - xmin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    yv    = ymin + (ymax - ymin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    yv2   = ymin + (ymax - ymin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    xy_bc = torch.cat([
        torch.cat([xh,  torch.full_like(xh,  ymax)], dim=1),
        torch.cat([xh2, torch.full_like(xh2, ymin)], dim=1),
        torch.cat([torch.full_like(yv,  xmin), yv ],  dim=1),
        torch.cat([torch.full_like(yv2, xmax), yv2],  dim=1),
    ], dim=0)
    T_bc = torch.full((xy_bc.shape[0], 1), T_amb, device=device, dtype=torch.float32)

    xy_all = torch.cat(cable_pts_list + [xy_bc], dim=0)
    T_all  = torch.cat(cable_T_list  + [T_bc],  dim=0)

    coord_mins = torch.tensor([xmin, ymin], device=device, dtype=torch.float32)
    coord_maxs = torch.tensor([xmax, ymax], device=device, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        xy_in = (
            2.0 * (xy_all - coord_mins) / (coord_maxs - coord_mins) - 1.0
            if normalize else xy_all
        )
        loss = torch.mean((model(xy_in) - T_all) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        xy_in = (
            2.0 * (xy_all - coord_mins) / (coord_maxs - coord_mins) - 1.0
            if normalize else xy_all
        )
        T_pred = model(xy_in)
        rmse = float(torch.sqrt(torch.mean((T_pred - T_all) ** 2)).item())
    return rmse


# ---------------------------------------------------------------------------
# Modelo residual multicable
# ---------------------------------------------------------------------------

class ResidualPINNModelMulti(nn.Module):
    """PINN que aprende la correccion u = T_total - T_analitico_multicable.

    T_total(x,y) = T_bg(x,y) + u(x,y)
      T_bg  = superposicion Kennelly + perfil cilindrico por cable   (no diff.)
      u     = red neuronal base (correccion de dominio finito)

    Dado que T_bg ya satisface: PDE en suelo, T=T_amb en y=0 y el flujo de
    calor en cada vaina, la red converge partiendo de u=0 sin minimos espurios.

    Nota: _Q_lins es mutable (no frozen) para soportar la iteracion R(T).
    """

    def __init__(
        self,
        base: nn.Module,
        layers_list: list[list],
        placements: list,
        k_soil: float,
        T_amb: float,
        Q_lins: list[float],
        domain,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self._layers_list = layers_list
        self._placements = placements
        self._k_soil = k_soil
        self._T_amb = T_amb
        self._Q_lins = list(Q_lins)   # mutable: se actualiza en iteracion R(T)
        self._normalize = normalize
        self._xmin = domain.xmin
        self._xmax = domain.xmax
        self._ymin = domain.ymin
        self._ymax = domain.ymax

    def _denormalize(self, xy_n: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor([self._xmin, self._ymin], device=xy_n.device, dtype=xy_n.dtype)
        hi = torch.tensor([self._xmax, self._ymax], device=xy_n.device, dtype=xy_n.dtype)
        return (xy_n + 1.0) * 0.5 * (hi - lo) + lo

    def forward(self, xy_in: torch.Tensor) -> torch.Tensor:
        xy_phys = self._denormalize(xy_in) if self._normalize else xy_in
        T_bg = _multilayer_T_multi(
            xy_phys,
            self._layers_list,
            self._placements,
            self._k_soil,
            self._T_amb,
            self._Q_lins,
        )
        u = self.base(xy_in)
        return T_bg + u


# ---------------------------------------------------------------------------
# Estimacion analitica IEC 60287 para trefoil
# ---------------------------------------------------------------------------

def _iec60287_trefoil(
    layers_list: list[list],
    placements: list,
    k_soil: float,
    T_amb: float,
    Q_scale: float,
    k_eff_fn: Callable[[float, float], float] | None = None,
    q_lin_overrides: list[float] | None = None,
) -> dict:
    """Estimacion IEC 60287 para N cables en trefoil: resistencias en serie + calentamiento mutuo.

    Soporta cables de diferente tipo y corriente (Q_lin diferente por cable).

    Args:
        layers_list:     Lista de listas de capas, una por cable.
        k_eff_fn:        si se pasa, k(x,y) escalar para la estimacion Kennelly.
        q_lin_overrides: si se pasa, lista de Q_lin ya calculados (uno por cable).
    """
    n_cables = len(placements)

    # Q_lin por cable
    Q_lins: list[float] = []
    for i in range(n_cables):
        layers = layers_list[i]
        conductor = layers[0]
        r_cond    = conductor.r_outer
        if q_lin_overrides is not None:
            Q_lins.append(q_lin_overrides[i])
        else:
            Q_lins.append(conductor.Q * Q_scale * math.pi * r_cond ** 2)

    # Resistencia termica de las capas de cada cable
    dT_layers_list: list[dict[str, float]] = []
    dT_cable_totals: list[float] = []
    for i in range(n_cables):
        layers = layers_list[i]
        Q_i = Q_lins[i]
        dT_layers: dict[str, float] = {}
        for layer in layers:
            r_in  = max(layer.r_inner, 1e-9)
            r_out = layer.r_outer
            if r_out <= r_in:
                continue
            dT_layers[layer.name] = Q_i / (2.0 * math.pi * layer.k) * math.log(r_out / r_in)
        dT_layers_list.append(dT_layers)
        dT_cable_totals.append(sum(dT_layers.values()))

    # dT_soil por cable: Kennelly self + mutuo de los demas
    cable_results = []
    for i, pl_i in enumerate(placements):
        layers = layers_list[i]
        r_sheath = layers[-1].r_outer
        d_i   = abs(pl_i.cy)
        k_i   = k_eff_fn(pl_i.cx, pl_i.cy) if k_eff_fn is not None else k_soil
        dT_soil_i = Q_lins[i] / (2.0 * math.pi * k_i) * math.log(2.0 * d_i / r_sheath)
        for j, pl_j in enumerate(placements):
            if i == j:
                continue
            d_j      = abs(pl_j.cy)
            dist_ij  = math.sqrt((pl_i.cx - pl_j.cx) ** 2 + (pl_i.cy - pl_j.cy) ** 2)
            r_img_sq = (pl_i.cx - pl_j.cx) ** 2 + (pl_i.cy - d_j) ** 2
            dT_soil_i += Q_lins[j] / (4.0 * math.pi * k_i) * math.log(r_img_sq / dist_ij ** 2)
        T_cond_i = T_amb + dT_soil_i + dT_cable_totals[i]
        cable_results.append({
            "cable_id": i + 1,
            "cx": pl_i.cx,
            "cy": pl_i.cy,
            "dT_soil": dT_soil_i,
            "T_cond": T_cond_i,
        })

    hottest_idx = max(range(len(cable_results)), key=lambda k: cable_results[k]["T_cond"])
    return {
        "Q_lins_W_per_m": Q_lins,
        "Q_lin_W_per_m":  max(Q_lins),      # backward compat
        "dT_by_layer":    dT_layers_list[hottest_idx],  # del cable mas caliente
        "dT_cable":       dT_cable_totals[hottest_idx],
        "cables":         cable_results,
        "hottest_idx":    hottest_idx,
        "T_cond_ref":     cable_results[hottest_idx]["T_cond"],
    }


# ---------------------------------------------------------------------------
# Muestreo de puntos (suelo + contornos)
# ---------------------------------------------------------------------------

def _sample_soil_pts(
    domain,
    placements: list,
    r_sheaths: list[float],
    n: int,
    device: torch.device,
    oversample: int = 8,
) -> torch.Tensor:
    """Muestreo uniforme en el dominio excluyendo el interior de TODOS los cables."""
    collected: list[torch.Tensor] = []
    need = n
    while need > 0:
        xs = (domain.xmin + (domain.xmax - domain.xmin)
              * torch.rand(need * oversample, 1, device=device, dtype=torch.float32))
        ys = (domain.ymin + (domain.ymax - domain.ymin)
              * torch.rand(need * oversample, 1, device=device, dtype=torch.float32))
        pts = torch.cat([xs, ys], dim=1)
        in_any = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)
        for idx, pl in enumerate(placements):
            dx = pts[:, 0] - pl.cx
            dy = pts[:, 1] - pl.cy
            in_any |= (dx * dx + dy * dy < r_sheaths[idx] ** 2)
        valid = pts[~in_any]
        collected.append(valid)
        need = max(0, n - sum(v.shape[0] for v in collected))
    return torch.cat(collected, dim=0)[:n]


def _sample_bnd_pts(domain, n: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Muestreo aleatorio de n//4 puntos por cada borde del dominio."""
    n_per = max(1, n // 4)
    xmin, xmax = domain.xmin, domain.xmax
    ymin, ymax = domain.ymin, domain.ymax
    xr = torch.rand(n_per, 1, device=device, dtype=torch.float32)
    yr = torch.rand(n_per, 1, device=device, dtype=torch.float32)
    return {
        "top":    torch.cat([xmin + (xmax - xmin) * xr, torch.full_like(xr, ymax)], dim=1),
        "bottom": torch.cat([xmin + (xmax - xmin) * xr.clone(), torch.full_like(xr, ymin)], dim=1),
        "left":   torch.cat([torch.full_like(yr, xmin), ymin + (ymax - ymin) * yr], dim=1),
        "right":  torch.cat([torch.full_like(yr, xmax), ymin + (ymax - ymin) * yr.clone()], dim=1),
    }


# ---------------------------------------------------------------------------
# Funcion de perdida (con k(x,y) variable)
# ---------------------------------------------------------------------------

def _compute_losses(
    model: nn.Module,
    xy_soil_fixed: torch.Tensor,
    bnd_pts: dict[str, torch.Tensor],
    bcs: dict,
    T_amb: float,
    norm_fn: Callable,
    normalize: bool,
    k_fn: Callable[[torch.Tensor], torch.Tensor] | float,
    w_pde: float,
    w_bc: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcula perdida PDE (suelo) + CC Dirichlet; devuelve (total, pde, bc).

    k_fn puede ser:
    - float: k uniforme (comportamiento original)
    - Callable(xy_phys: Tensor) -> Tensor(N,1): k espacialmente variable

    Con formulacion residual: autograd ve dT_bg/dx = 0 (boolean mask no diff.),
    por lo que pde_residual_steady(T_total, pts, k, 0) = laplaciano(k*grad(u)) = 0.
    Con k variable: laplaciano(k(x,y)*grad(u)) = 0 se resuelve correctamente.
    """
    pts = xy_soil_fixed.clone().detach().requires_grad_(True)
    pts_in = norm_fn(pts) if normalize else pts
    T_pred = model(pts_in)
    k_vals = k_fn(pts) if callable(k_fn) else k_fn
    res_pde = pde_residual_steady(T_pred, pts, k_vals, 0.0)
    loss_pde = torch.mean(res_pde ** 2)

    # BC Dirichlet (solo la parte superior: T = T_amb)
    loss_bc = torch.tensor(0.0, device=xy_soil_fixed.device)
    for edge, pts_b in bnd_pts.items():
        bc = bcs.get(edge)
        if bc is None or bc.bc_type != "dirichlet":
            continue
        val = bc.value if bc.value > 1.0 else T_amb   # 0.0 se reemplaza por T_amb
        T_b = model(norm_fn(pts_b) if normalize else pts_b)
        loss_bc = loss_bc + torch.mean((T_b - val) ** 2)

    total = w_pde * loss_pde + w_bc * loss_bc
    return total, loss_pde.detach(), loss_bc.detach()


# ---------------------------------------------------------------------------
# Bucle de entrenamiento Adam + L-BFGS
# ---------------------------------------------------------------------------

def _train_adam_lbfgs(
    model: nn.Module,
    domain,
    placements: list,
    bcs: dict,
    T_amb: float,
    r_sheaths: list[float],
    k_fn: Callable[[torch.Tensor], torch.Tensor] | float,
    adam_steps: int,
    lbfgs_steps: int,
    n_int: int,
    n_bnd: int,
    oversample: int,
    w_pde: float,
    w_bc: float,
    lr: float,
    print_every: int,
    normalize: bool,
    device: torch.device,
    logger,
    step_offset: int = 0,
    total_adam_budget: int = 0,
) -> dict[str, list[float]]:
    """Adam (+ L-BFGS opcional) con remuestreo periodico de puntos de colecacion.

    k_fn puede ser float (k uniforme) o callable(xy_phys)->Tensor (k variable).
    step_offset / total_adam_budget: para imprimir progreso correcto en iteracion R(T).
    """
    coord_mins = torch.tensor(
        [domain.xmin, domain.ymin], device=device, dtype=torch.float32
    )
    coord_maxs = torch.tensor(
        [domain.xmax, domain.ymax], device=device, dtype=torch.float32
    )

    def norm_fn(xy: torch.Tensor) -> torch.Tensor:
        return 2.0 * (xy - coord_mins) / (coord_maxs - coord_mins) - 1.0

    history: dict[str, list[float]] = {"total": [], "pde": [], "bc": []}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    xy_soil: torch.Tensor | None = None
    bnd_pts: dict | None = None
    total_for_pct = total_adam_budget if total_adam_budget > 0 else adam_steps

    # Adam
    for step in range(1, adam_steps + 1):
        if xy_soil is None or (step - 1) % print_every == 0:
            xy_soil = _sample_soil_pts(domain, placements, r_sheaths, n_int, device, oversample)
            bnd_pts = _sample_bnd_pts(domain, n_bnd, device)

        optimizer.zero_grad()
        total, l_pde, l_bc = _compute_losses(
            model, xy_soil, bnd_pts, bcs, T_amb,  # type: ignore[arg-type]
            norm_fn, normalize, k_fn, w_pde, w_bc,
        )
        total.backward()
        optimizer.step()

        history["total"].append(float(total.detach()))
        history["pde"].append(float(l_pde))
        history["bc"].append(float(l_bc))

        if step % print_every == 0:
            global_step = step_offset + step
            pct = 100.0 * global_step / total_for_pct
            logger.info(
                "[Adam %d/%d  %.1f%%] loss=%.4e  pde=%.3e  bc=%.3e",
                global_step, total_for_pct, pct, float(total), float(l_pde), float(l_bc),
            )

    # L-BFGS
    if lbfgs_steps > 0:
        max_iter  = 20
        n_events  = max(1, lbfgs_steps // max_iter)
        print_ev_l = max(1, n_events // 25)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=max_iter,
            history_size=100,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        for event in range(1, n_events + 1):
            xy_s = _sample_soil_pts(domain, placements, r_sheaths, n_int, device, oversample)
            bd_p = _sample_bnd_pts(domain, n_bnd, device)

            def closure_fn(xs=xy_s, bp=bd_p) -> torch.Tensor:
                lbfgs.zero_grad()
                tot, _, _ = _compute_losses(
                    model, xs, bp, bcs, T_amb, norm_fn, normalize, k_fn, w_pde, w_bc,
                )
                tot.backward()
                return tot

            loss_v = lbfgs.step(closure_fn)
            loss_val = float(loss_v) if loss_v is not None else float("nan")
            history["total"].append(loss_val)

            if event % print_ev_l == 0:
                pct = 100.0 * (step_offset + adam_steps + event) / (total_for_pct + n_events)
                logger.info("[LBFGS %d/%d  %.1f%%] loss=%.4e", event, n_events, pct, loss_val)

    return history


# ---------------------------------------------------------------------------
# Graficas adicionales: zoom en conductores y campo k(x,y)
# ---------------------------------------------------------------------------

def _plot_zoom_temperature(
    model: nn.Module,
    domain,
    placements: list,
    layers_list: list[list],
    device: torch.device,
    normalize: bool,
    profile: str,
    save_path: Path,
    margin: float = 0.20,
    nx: int = 300,
    ny: int = 300,
) -> None:
    """Mapa de temperatura con zoom alrededor del grupo de conductores."""
    r_sheath = max(ls[-1].r_outer for ls in layers_list)
    x_centers = [pl.cx for pl in placements]
    y_centers = [pl.cy for pl in placements]
    x0 = min(x_centers) - margin
    x1 = max(x_centers) + margin
    y0 = min(y_centers) - margin
    y1 = max(y_centers) + margin

    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    xy_flat = torch.tensor(
        np.column_stack([X.ravel(), Y.ravel()]),
        dtype=torch.float32, device=device,
    )

    if normalize:
        coord_mins = torch.tensor([domain.xmin, domain.ymin], dtype=torch.float32, device=device)
        coord_maxs = torch.tensor([domain.xmax, domain.ymax], dtype=torch.float32, device=device)
        xy_in = 2.0 * (xy_flat - coord_mins) / (coord_maxs - coord_mins) - 1.0
    else:
        xy_in = xy_flat

    with torch.no_grad():
        T_flat = model(xy_in).cpu().numpy().reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(X, Y, T_flat, levels=60, cmap="hot_r")
    plt.colorbar(cf, ax=ax, label="T [K]")

    # Contornos de las capas de cada cable
    layer_colors = ["white", "#aaddff", "#88bbff", "#4472C4"]
    layer_names  = [la.name for la in reversed(layers_list[0])]
    for idx, pl in enumerate(placements):
        layers_i = layers_list[idx]
        for li, layer in enumerate(reversed(layers_i)):
            lc = layer_colors[li % len(layer_colors)]
            lw = 1.2 if li == 0 else 0.6
            circle = plt.Circle(
                (pl.cx, pl.cy), layer.r_outer,
                fill=False, edgecolor=lc, linewidth=lw, alpha=0.9, zorder=5,
            )
            ax.add_patch(circle)
        ax.plot(pl.cx, pl.cy, "w+", ms=5, zorder=6)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]  (profundidad negativa)")
    ax.set_title("T(x,y) [K] -- Zoom conductores  [%s]" % profile)

    legend_patches = [
        mpatches.Patch(facecolor="none", edgecolor=layer_colors[i], label=layer_names[i])
        for i in range(len(layers_list[0]))
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_k_field(
    domain,
    pp: PhysicsParams,
    placements: list,
    layers_list: list[list],
    k_soil_base: float,
    profile: str,
    save_path: Path,
) -> None:
    """Visualizar el campo de conductividad termica k(x,y) del suelo."""
    nx, ny = 300, 200
    xs = np.linspace(domain.xmin, domain.xmax, nx)
    ys = np.linspace(domain.ymin, domain.ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    if pp.k_variable:
        dx_np = np.abs(X - pp.k_cx) - pp.k_width  / 2.0
        dy_np = np.abs(Y - pp.k_cy) - pp.k_height / 2.0
        d_np  = np.maximum(dx_np, dy_np)
        sig   = 1.0 / (1.0 + np.exp(d_np / pp.k_transition))
        K_field = pp.k_bad + (pp.k_good - pp.k_bad) * sig
    else:
        K_field = np.full_like(X, k_soil_base)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin = min(pp.k_bad, k_soil_base) * 0.95
    vmax = max(pp.k_good, k_soil_base) * 1.05
    cf = ax.contourf(X, Y, K_field, levels=40, cmap="YlOrRd_r", vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax, label="k suelo [W/(m·K)]")

    for idx, pl in enumerate(placements):
        r_sheath_i = layers_list[idx][-1].r_outer
        circle = plt.Circle(
            (pl.cx, pl.cy), r_sheath_i,
            fill=True, facecolor="black", edgecolor="white", linewidth=1.5, zorder=5,
        )
        ax.add_patch(circle)

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
    ax.set_ylabel("y [m]  (profundidad negativa)")
    ax.set_title("k(x,y) suelo [W/(m·K)]  --  Trefoil  [%s]" % profile)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grafica de geometria personalizada para trefoil
# ---------------------------------------------------------------------------

def _plot_geometry_trefoil(
    layers_list: list[list],
    placements: list,
    domain,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """Visualizar dominio con los cables del trefoil (seccion transversal)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    circuit_colors = ["tab:blue", "tab:orange", "tab:green"]
    colores_cable = [circuit_colors[(pl.cable_id - 1) // 3] for pl in placements]
    circuit_num = [(pl.cable_id - 1) // 3 + 1 for pl in placements]
    names = [f"C{circuit_num[i]}-{pl.cable_id}" for i, pl in enumerate(placements)]

    for idx, (pl, color) in enumerate(zip(placements, colores_cable)):
        layers_i = layers_list[idx]
        layer_colors = ["#888888", "#aaaaaa", "#cccccc", "#4472C4"]
        for li, (layer, lcol) in enumerate(zip(reversed(layers_i), reversed(layer_colors))):
            ring = plt.Circle((pl.cx, pl.cy), layer.r_outer, color=lcol, zorder=5)
            ax.add_patch(ring)
        ax.text(
            pl.cx, pl.cy + layers_i[-1].r_outer * 1.8,
            names[idx], ha="center", va="bottom", fontsize=8, color=color, zorder=10,
        )

    rect = plt.Rectangle(
        (domain.xmin, domain.ymin),
        domain.xmax - domain.xmin,
        domain.ymax - domain.ymin,
        fill=False, edgecolor="black", linewidth=1.5, zorder=1,
    )
    ax.add_patch(rect)

    margin = 0.15
    cx_min = min(pl.cx for pl in placements) - margin
    cx_max = max(pl.cx for pl in placements) + margin
    cy_min = min(pl.cy for pl in placements) - margin
    cy_max = max(pl.cy for pl in placements) + margin
    ax.set_xlim(cx_min, cx_max)
    ax.set_ylim(cy_min, cy_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]  (profundidad negativa)")
    if title:
        ax.set_title(title)

    legend_patches = [
        mpatches.Patch(color="#4472C4", label="Conductor Cu"),
        mpatches.Patch(color="#cccccc", label="Aislante XLPE"),
        mpatches.Patch(color="#aaaaaa", label="Pantalla"),
        mpatches.Patch(color="#888888", label="Cubierta (sheath)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Cargar datos, entrenar PINN trefoil y comparar con estimacion analitica."""
    parser = argparse.ArgumentParser(
        description="Ejemplo PINN: tres circuitos de 3 cables XLPE en trefoil separado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Perfiles disponibles:\n"
            "  quick    : 5 000 Adam,               red 64x4  (~10-15 min CPU)\n"
            "  research : 10 000 Adam + 500 L-BFGS, red 128x5 (~40-70 min CPU)\n"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "research"],
        default="quick",
        help="Perfil de ejecucion (default: quick)",
    )
    args   = parser.parse_args()
    profile = args.profile

    RESULTS_DIR = HERE / ("results" if profile == "quick" else "results_research")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    SEP = "=" * 66
    print(SEP)
    print("  PINN -- 3 circuitos XLPE / Trefoil separado (9 cables)")
    print("  Perfil de ejecucion : %s" % profile.upper())
    print(SEP)

    # ------------------------------------------------------------------
    # Cargar parametros fisicos y del solver
    # ------------------------------------------------------------------
    problem  = load_problem(DATA_DIR)
    scenario = problem.scenarios[0]

    pp = load_physics_params(DATA_DIR / "physics_params.csv")

    params_csv = DATA_DIR / (
        "solver_params.csv" if profile == "quick" else "solver_params_research.csv"
    )
    solver_params = load_solver_params(params_csv)
    solver_cfg    = solver_params.to_solver_cfg()

    adam_n    = solver_cfg["training"]["adam_steps"]
    lbfgs_n   = solver_cfg["training"]["lbfgs_steps"]
    print_ev  = solver_cfg["training"]["print_every"]
    width     = solver_cfg["model"]["width"]
    depth     = solver_cfg["model"]["depth"]
    n_int     = solver_cfg["sampling"]["n_interior"]
    n_bnd     = solver_cfg["sampling"]["n_boundary"]
    oversamp  = solver_cfg["sampling"]["oversample"]
    normalize = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    w_pde     = solver_cfg["loss_weights"].get("pde", 1.0)
    w_bc      = solver_cfg["loss_weights"].get("bc_dirichlet", 10.0)
    k_soil    = scenario.k_soil     # k base del escenario
    T_amb     = scenario.T_amb
    n_cables  = len(problem.placements)

    # ------------------------------------------------------------------
    # Capas y Q_lin por cable (del catalogo si section_mm2 > 0)
    # ------------------------------------------------------------------
    from pinn_cables.materials.props import get_R_dc_20, get_alpha_R

    layers_list: list[list] = []
    I_per_cable: list[float] = []
    R_per_cable: list[float] = []
    alpha_per_cable: list[float] = []
    for pl in problem.placements:
        layers_list.append(problem.get_layers(pl.cable_id))
        I_i = pl.current_A if pl.current_A > 0.0 else pp.I_A
        if pl.section_mm2 > 0:
            R_i = get_R_dc_20(pl.section_mm2, pl.conductor_material)
            a_i = get_alpha_R(pl.conductor_material)
        else:
            R_i = pp.R_ref
            a_i = pp.alpha_R
        I_per_cable.append(I_i)
        R_per_cable.append(R_i)
        alpha_per_cable.append(a_i)

    r_sheaths = [ls[-1].r_outer for ls in layers_list]

    # ------------------------------------------------------------------
    # k(x,y): funcion escalar (para fondo analitico) y tensor (para PDE)
    # k_eff_bg: k representativa para el fondo Kennelly (k en el centroide)
    # ------------------------------------------------------------------
    cx_mean = sum(pl.cx for pl in problem.placements) / n_cables
    cy_mean = sum(pl.cy for pl in problem.placements) / n_cables

    if pp.k_variable:
        k_eff_bg = k_scalar(cx_mean, cy_mean, pp)

        def k_fn_pde(xy_phys: torch.Tensor) -> torch.Tensor:
            return k_tensor(xy_phys, pp)

        def k_eff_fn_iec(x: float, y: float) -> float:
            return k_scalar(x, y, pp)
    else:
        k_eff_bg = k_soil

        def k_fn_pde(xy_phys: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
            return torch.full((xy_phys.shape[0], 1), k_soil,
                              device=xy_phys.device, dtype=xy_phys.dtype)

        def k_eff_fn_iec(x: float, y: float) -> float:  # type: ignore[misc]
            return k_soil

    # ------------------------------------------------------------------
    # R(T) iteracion para la referencia IEC
    # ------------------------------------------------------------------
    use_R_T = (pp.n_R_iter > 0)
    if use_R_T:
        Q_lins_iec = [Q_lin_from_I(I_per_cable[i], R_per_cable[i],
                                    alpha_per_cable[i], T_amb, pp.T_ref_R_K)
                       for i in range(n_cables)]
        T_cond_iec_est = T_amb + 50.0
        for _ in range(15):
            Q_lins_iec = [Q_lin_from_I(I_per_cable[i], R_per_cable[i],
                                        alpha_per_cable[i], T_cond_iec_est, pp.T_ref_R_K)
                           for i in range(n_cables)]
            iec_tmp = _iec60287_trefoil(
                layers_list, problem.placements, k_soil, T_amb, scenario.Q_scale,
                k_eff_fn=k_eff_fn_iec, q_lin_overrides=Q_lins_iec,
            )
            T_new = iec_tmp["T_cond_ref"]
            if abs(T_new - T_cond_iec_est) < 0.01:
                break
            T_cond_iec_est = T_new
        iec = iec_tmp
    else:
        Q_lins_iec = []
        for i in range(n_cables):
            cond_i = layers_list[i][0]
            Q_lins_iec.append(cond_i.Q * scenario.Q_scale * math.pi * cond_i.r_outer ** 2)
        iec = _iec60287_trefoil(
            layers_list, problem.placements, k_soil, T_amb, scenario.Q_scale,
            k_eff_fn=k_eff_fn_iec,
        )
    T_ref_K = iec["T_cond_ref"]
    Q_lins  = list(iec["Q_lins_W_per_m"])   # Q_lins iniciales para el PINN

    # ------------------------------------------------------------------
    # Informacion del problema
    # ------------------------------------------------------------------
    print("\n  Configuracion fisica:")
    print("  Escenario   : %s  (%s)" % (scenario.scenario_id, scenario.mode))
    n_circuits = n_cables // 3
    print("  Cables      : %d en %d circuitos trefoil (sep. 0.30 m)" % (n_cables, n_circuits))
    for pl in problem.placements:
        sec = "%d mm2" % pl.section_mm2 if pl.section_mm2 > 0 else "CSV"
        mat = pl.conductor_material.upper() if pl.section_mm2 > 0 else "CSV"
        cur = "%.0f A" % pl.current_A if pl.current_A > 0 else "%.0f A (pp)" % pp.I_A
        print("    Cable %d: seccion=%s  conductor=%s  corriente=%s" % (
            pl.cable_id, sec, mat, cur))
    print("  R(T)        : %s" % ("SI" if use_R_T else "No"))
    if use_R_T:
        print("  n_R_iter    : %d  (pasos Adam por iter: %d)" % (
            pp.n_R_iter, adam_n // max(1, pp.n_R_iter)))
    if pp.k_variable:
        print("  k(x,y)      : variable  (k_good=%.1f, k_bad=%.1f W/mK)" % (
            pp.k_good, pp.k_bad))
        print("  Region buena: cx=%.2f  cy=%.2f  w=%.2f  h=%.2f  s=%.3f m" % (
            pp.k_cx, pp.k_cy, pp.k_width, pp.k_height, pp.k_transition))
        print("  k_eff_bg    : %.3f W/(m*K)  (en centroide del conjunto)" % k_eff_bg)
    else:
        print("  k_suelo     : %.1f W/(m*K)  (uniforme)" % k_soil)
    print("  T_ambiente  : %.1f degC  (%.2f K)" % (T_amb - 273.15, T_amb))

    # ------------------------------------------------------------------
    # Estimacion analitica IEC 60287 trefoil (con efectos R(T) y k variable)
    # ------------------------------------------------------------------
    print("\n  Referencia analitica (Kennelly + IEC, con R(T) y k variable):")
    for i, q_i in enumerate(iec["Q_lins_W_per_m"]):
        print("  Q_lin cable %d    : %.2f W/m lineal" % (i + 1, q_i))
    for name, dT in iec["dT_by_layer"].items():
        print("  dT %-10s : %+.2f K" % (name, dT))
    print("  dT cable total  : %+.2f K" % iec["dT_cable"])
    print()
    for circ in range(1, 4):
        first = (circ - 1) * 3 + 1
        cx_circ = iec["cables"][first - 1]["cx"]  # top cable cx = circuit centroid x
        print("  Circuito %d (x=%+.2f m):" % (circ, cx_circ))
        for cr in iec["cables"][first - 1 : first + 2]:
            hot = "  <-- mas caliente" if cr["cable_id"] == iec["hottest_idx"] + 1 else ""
            print("    Cable %d (%.3f, %.3f m): dT_suelo=%+.2f K  T_cond=%.1f K (%.1f degC)%s" % (
                cr["cable_id"], cr["cx"], cr["cy"],
                cr["dT_soil"], cr["T_cond"], cr["T_cond"] - 273.15, hot,
            ))
    print("  T_cond ref. (max) : %.1f K  (%.1f degC)" % (T_ref_K, T_ref_K - 273.15))

    # ------------------------------------------------------------------
    # Configuracion del solver y modelo
    # ------------------------------------------------------------------
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="trefoil_" + profile)

    base_model = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModelMulti(
        base_model,
        layers_list,
        problem.placements,
        k_eff_bg,
        T_amb,
        Q_lins,
        problem.domain,
        normalize=normalize,
    )
    _init_output_bias(model.base, 0.0)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n  Configuracion del solver:")
    print("  Red neuronal  : MLP %dx%d  (%d params)" % (width, depth, n_params))
    print("  Puntos suelo  : %d  |  contorno: %d" % (n_int, n_bnd))
    print("  Entrenamiento : %d Adam + %d L-BFGS" % (adam_n, lbfgs_n))
    if use_R_T and pp.n_R_iter > 1:
        print("  R(T) iters    : %d x ~%d pasos Adam" % (pp.n_R_iter, adam_n // pp.n_R_iter))
    print("  Avance cada   : %d pasos Adam" % print_ev)
    logger.info(
        "Device=%s | Perfil=%s | red MLP%dx%d (%d params)",
        device, profile, width, depth, n_params,
    )

    # Grafica de geometria
    _plot_geometry_trefoil(
        layers_list, problem.placements, problem.domain,
        title="Cables XLPE -- 3 circuitos Trefoil (detalle)",
        save_path=RESULTS_DIR / "geometry.png",
    )

    # Grafica campo k(x,y)
    _plot_k_field(
        problem.domain, pp, problem.placements, layers_list,
        k_soil, profile, RESULTS_DIR / "k_field.png",
    )

    # ------------------------------------------------------------------
    # Pre-entrenamiento
    # ------------------------------------------------------------------
    print("\n  Pre-entrenando en perfiles cilindricos de los %d cables (800 pasos)..." % n_cables, flush=True)
    rmse_pre = _pretrain_cables(
        model, problem.placements, problem.domain, layers_list,
        Q_lins, k_eff_bg, T_amb, device=device, normalize=normalize,
        n_cable_per=1000, n_bc=300, n_steps=800, lr=1e-3,
    )
    print("  Pre-entrenamiento OK: RMSE = %.3f K" % rmse_pre, flush=True)

    # ------------------------------------------------------------------
    # Entrenamiento con iteracion R(T)
    # ------------------------------------------------------------------
    n_iters = max(1, pp.n_R_iter) if use_R_T else 1
    adam_per_iter = adam_n // n_iters
    history_all: dict[str, list[float]] = {"total": [], "pde": [], "bc": []}

    print("\n" + "-" * 66)
    print("  ENTRENAMIENTO  (Adam --> L-BFGS)")
    if use_R_T:
        print("  R(T) iteracion: %d rondas, ~%d pasos cada una" % (n_iters, adam_per_iter))
    print("  Columnas del log: [fase paso/total pct%%] loss  pde  bc")
    print("-" * 66)

    T_cond_pinns: list[float] = []
    coord_mins_t = torch.tensor(
        [problem.domain.xmin, problem.domain.ymin], device=device, dtype=torch.float32
    )
    coord_maxs_t = torch.tensor(
        [problem.domain.xmax, problem.domain.ymax], device=device, dtype=torch.float32
    )

    def norm_fn_eval(xy: torch.Tensor) -> torch.Tensor:
        return 2.0 * (xy - coord_mins_t) / (coord_maxs_t - coord_mins_t) - 1.0

    for iter_i in range(n_iters):
        is_last = (iter_i == n_iters - 1)
        steps_adam = adam_per_iter if not is_last else (adam_n - adam_per_iter * (n_iters - 1))
        lbfgs_this = lbfgs_n if is_last else 0

        if use_R_T and n_iters > 1:
            print("\n  -- R(T) iteracion %d/%d -- Q_lins=[%s] W/m" % (
                iter_i + 1, n_iters,
                ", ".join("%.3f" % q for q in model._Q_lins)))

        h = _train_adam_lbfgs(
            model=model,
            domain=problem.domain,
            placements=problem.placements,
            bcs=problem.bcs,
            T_amb=T_amb,
            r_sheaths=r_sheaths,
            k_fn=k_fn_pde,
            adam_steps=steps_adam,
            lbfgs_steps=lbfgs_this,
            n_int=n_int,
            n_bnd=n_bnd,
            oversample=oversamp,
            w_pde=w_pde,
            w_bc=w_bc,
            lr=solver_cfg["training"]["lr"],
            print_every=print_ev,
            normalize=normalize,
            device=device,
            logger=logger,
            step_offset=iter_i * adam_per_iter,
            total_adam_budget=adam_n,
        )
        for k in history_all:
            history_all[k].extend(h.get(k, []))

        # Evaluar T_cond en el centro de cada cable
        with torch.no_grad():
            model.eval()
            T_cond_pinns = []
            for pl in problem.placements:
                pt    = torch.tensor([[pl.cx, pl.cy]], device=device, dtype=torch.float32)
                pt_in = norm_fn_eval(pt) if normalize else pt
                T_cond_pinns.append(float(model(pt_in).item()))
            model.train()

        # Actualizar Q_lins para la proxima iteracion
        if use_R_T and not is_last:
            T_cond_hottest = max(T_cond_pinns)
            Q_lins_new = [Q_lin_from_I(I_per_cable[i], R_per_cable[i],
                                        alpha_per_cable[i], T_cond_hottest, pp.T_ref_R_K)
                           for i in range(n_cables)]
            model._Q_lins = Q_lins_new
            print("  --> T_cond estimada = %.1f K (%.1f degC)  =>  Q_lins nuevo = [%s] W/m" % (
                T_cond_hottest, T_cond_hottest - 273.15,
                ", ".join("%.3f" % q for q in Q_lins_new),
            ))

    print("-" * 66)

    # ------------------------------------------------------------------
    # Guardar modelo
    # ------------------------------------------------------------------
    model_path = RESULTS_DIR / "model_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo guardado: %s", model_path)

    # ------------------------------------------------------------------
    # Graficas
    # ------------------------------------------------------------------
    coord_mins_t2 = torch.tensor(
        [problem.domain.xmin, problem.domain.ymin], device=device, dtype=torch.float32
    )
    coord_maxs_t2 = torch.tensor(
        [problem.domain.xmax, problem.domain.ymax], device=device, dtype=torch.float32
    )

    plot_loss_history(
        history_all,
        title="Historia de perdida (%s) -- trefoil  R(T)+k_var" % profile,
        save_path=RESULTS_DIR / "loss_history.png",
    )
    X, Y, T_grid = evaluate_on_grid(
        model, problem.domain, nx=300, ny=300,
        device=device, normalize=normalize,
    )
    plot_temperature_field(
        X, Y, T_grid,
        title="T(x,y) [K] -- Cables XLPE Trefoil  [%s]" % profile,
        save_path=RESULTS_DIR / "temperature_field.png",
    )
    _plot_zoom_temperature(
        model, problem.domain, problem.placements, layers_list,
        device, normalize, profile,
        save_path=RESULTS_DIR / "temperature_zoom.png",
        margin=0.12,
    )

    # ------------------------------------------------------------------
    # Tabla de resultados
    # ------------------------------------------------------------------
    with torch.no_grad():
        model.eval()
        T_cond_pinns = []
        for pl in problem.placements:
            pt    = torch.tensor([[pl.cx, pl.cy]], device=device, dtype=torch.float32)
            pt_in = norm_fn_eval(pt) if normalize else pt
            T_cond_pinns.append(float(model(pt_in).item()))

    T_max_pinn = float(T_grid.max())
    T_min_pinn = float(T_grid.min())
    loss_final = history_all["total"][-1]

    print("\n" + "=" * 66)
    print("  RESULTADOS FINALES  [%s]" % profile.upper())
    print("=" * 66)
    if use_R_T:
        Q_lins_final = model._Q_lins
        print("  Q_lins final (R(T))  : [%s] W/m" % ", ".join("%.3f" % q for q in Q_lins_final))
    print("  %-34s  %10s  %10s  %8s" % ("Magnitud", "PINN", "Ref.", "Error"))
    print("  " + "-" * 62)
    for i, (pl, T_pinn_i, cr) in enumerate(
        zip(problem.placements, T_cond_pinns, iec["cables"])
    ):
        if i % 3 == 0:
            circ = i // 3 + 1
            print("  -- Circuito %d --" % circ)
        T_ref_i = cr["T_cond"]
        err_i   = T_pinn_i - T_ref_i
        hot     = " *" if i == iec["hottest_idx"] else "  "
        print("  T_cond cable %d (K)%s           %10.2f  %10.2f  %+7.2f K" % (
            pl.cable_id, hot, T_pinn_i, T_ref_i, err_i,
        ))
    print("  " + "-" * 62)
    T_cond_max_pinn = max(T_cond_pinns)
    err_max = T_cond_max_pinn - T_ref_K
    print("  %-34s  %10.2f  %10.2f  %+7.2f K" % (
        "T_cond max (K)", T_cond_max_pinn, T_ref_K, err_max
    ))
    print("  %-34s  %10.1f  %10.1f" % (
        "T_cond max (degC)", T_cond_max_pinn - 273.15, T_ref_K - 273.15
    ))
    print("  %-34s  %10.2f" % ("T max dominio (K)", T_max_pinn))
    print("  %-34s  %10.2f" % ("T min dominio (K)", T_min_pinn))
    print("  %-34s  %10.4e" % ("Perdida final", loss_final))
    print("  " + "-" * 62)
    print("  Limite IEC 60287 XLPE : 363 K (90 degC)")
    print("  Margen termico PINN   : %.1f K  (ref: %.1f K)" % (
        363.0 - T_cond_max_pinn, 363.0 - T_ref_K
    ))
    print("  [*] Cable mas caliente segun referencia analitica")
    if profile == "quick":
        print()
        print("  Para resultados de investigacion (~40-70 min):")
        print("    python examples/xlpe_three_trefoils/run_example.py --profile research")
    print()
    print("  Graficas guardadas en: %s" % RESULTS_DIR)
    print("    - geometry.png  |  loss_history.png  |  temperature_field.png")
    print("    - temperature_zoom.png  |  k_field.png")
    print("    - model_final.pt  (pesos de la red entrenada)")


if __name__ == "__main__":
    main()
