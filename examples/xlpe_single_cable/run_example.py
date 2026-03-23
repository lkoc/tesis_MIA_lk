"""Ejemplo cable XLPE a 12/20 kV enterrado a 70 cm.

Soporta cables XLPE de 95, 150, 240, 400 y 600 mm2 con conductor de
cobre (cu) o aluminio (al).  La seccion, material y corriente se
especifican en ``cables_placement.csv`` (columnas ``section_mm2``,
``conductor_material``, ``current_A``).

Soporta dos perfiles de ejecucion:

- **quick**    (~5-8 min CPU) : 5 000 Adam + 500 L-BFGS, red 64x4
- **research** (~25-35 min CPU): 15 000 Adam + 3 000 L-BFGS, red 128x5

Uso::

    python examples/xlpe_single_cable/run_example.py
    python examples/xlpe_single_cable/run_example.py --profile research

Referencia IEC 60287: T_max admisible XLPE = 90 degC (363 K).
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pinn_cables.io.readers import load_problem, load_solver_params  # noqa: E402
from pinn_cables.pinn.model import build_model  # noqa: E402
from pinn_cables.pinn.train import SteadyStatePINNTrainer  # noqa: E402
from pinn_cables.pinn.utils import get_device, set_seed, setup_logging  # noqa: E402
from pinn_cables.post.eval import evaluate_on_grid  # noqa: E402
from pinn_cables.post.plots import (  # noqa: E402
    plot_cable_geometry,
    plot_loss_history,
    plot_temperature_field,
)


def _init_output_bias(model: nn.Module, value: float) -> None:
    """Inicializar la ultima capa lineal en *value* (warm-start cerca de T_amb)."""
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        last_linear.bias.data.fill_(value)


@torch.no_grad()
def _multilayer_T(
    xy: torch.Tensor,
    layers: list,
    placement,
    k_soil: float,
    T_amb: float,
    Q_lin: float,
) -> torch.Tensor:
    """Temperatura analitica: perfil 1D cilindrico en capas + imagen en suelo.

    Dentro de cada capa del cable: T(r) = T_out + Q_lin/(2*pi*k) * log(r_out/r).
    En el suelo: formula de imagen de Kennelly (satisface T=T_amb en y=0).

    Returns:
        Tensor de forma (N, 1) con las temperaturas en K.
    """
    cx, cy = placement.cx, placement.cy
    d = abs(cy)
    r_sheath = layers[-1].r_outer

    dx = xy[:, 0:1] - cx
    dy_r = xy[:, 1:2] - cy
    r = torch.sqrt(dx * dx + dy_r * dy_r).clamp(min=1e-9)

    # T en la superficie del cable (borde exterior del sheath)
    T_sheath_outer = T_amb + Q_lin / (2.0 * math.pi * k_soil) * math.log(2.0 * d / r_sheath)

    # T en el borde exterior de cada capa (de suelo hacia el interior)
    layer_T_outer: dict[str, float] = {}
    T_curr = T_sheath_outer
    for layer in reversed(layers):
        layer_T_outer[layer.name] = T_curr
        r_out = layer.r_outer
        r_in = max(layer.r_inner, 1e-9)
        if layer.r_inner == 0.0 and Q_lin > 0.0:
            # Conductor: dT desde superficie hasta centro (perfil parabolico)
            Q_vol = Q_lin / (math.pi * r_out ** 2)
            T_curr += Q_vol / (4.0 * layer.k) * r_out ** 2  # = Q_lin/(4*pi*k)
        else:
            T_curr += Q_lin / (2.0 * math.pi * layer.k) * math.log(r_out / r_in)

    result = torch.full((xy.shape[0], 1), T_amb, device=xy.device, dtype=xy.dtype)

    # Suelo: formula de imagen completa
    # Fuente real en (cx, cy=-d); fuente imagen en (cx, +d).
    # r_image = distancia a (cx, +d) => dy_imagen = y - d
    dy_img = xy[:, 1:2] - d
    r_img_sq = (dx * dx + dy_img * dy_img).clamp(min=1e-20)
    r_sq_clamped = (dx * dx + dy_r * dy_r).clamp(min=r_sheath ** 2)
    dT_soil = Q_lin / (4.0 * math.pi * k_soil) * torch.log(r_img_sq / r_sq_clamped)
    T_soil = T_amb + dT_soil.clamp(min=0.0)
    mask_soil = (r >= r_sheath).squeeze(1)
    result[mask_soil] = T_soil[mask_soil]

    # Capas del cable (de exterior a interior)
    for layer in reversed(layers):
        r_out = layer.r_outer
        r_in = max(layer.r_inner, 1e-9)
        T_out_layer = layer_T_outer[layer.name]
        mask = ((r >= layer.r_inner) & (r < r_out)).squeeze(1)
        if not mask.any():
            continue
        r_pts = r[mask, 0]
        if layer.r_inner == 0.0 and Q_lin > 0.0:
            # Conductor solido con fuente volumetrica: perfil parabolico exacto
            # T(r) = T_out + Q_vol/(4k) * (r_out^2 - r^2), Q_vol = Q_lin/(pi*r_out^2)
            Q_vol = Q_lin / (math.pi * r_out ** 2)
            dT = Q_vol / (4.0 * layer.k) * (r_out ** 2 - r_pts ** 2)
        else:
            # Capa sin fuente (XLPE, pantalla, cubierta): perfil logaritmico
            dT = Q_lin / (2.0 * math.pi * layer.k) * torch.log(r_out / r_pts.clamp(min=r_in))
        result[mask, 0] = T_out_layer + dT

    return result


def _pretrain_cable_plus_bc(
    model: nn.Module,
    placement,
    domain,
    layers: list,
    Q_lin: float,
    k_soil: float,
    T_amb: float,
    device: torch.device,
    normalize: bool,
    n_cable: int = 2000,
    n_bc: int = 200,
    n_steps: int = 500,
    lr: float = 1e-3,
) -> float:
    """Pre-entrenar con: interior del cable (T analitica) + contornos del dominio (T_amb).

    Esta estrategia:
      - Proporciona T >> T_amb dentro del cable (hasta ~310 K en el conductor),
        evitando el minimo trivial T = T_amb.
      - Fuerza T = T_amb en los cuatro bordes del dominio,
        SIN violar las CCs Dirichlet como hacia el perfil de Kennelly (semi-espacio).
    Devuelve el RMSE final (K) del ajuste.
    """
    cx, cy = placement.cx, placement.cy
    r_sheath = layers[-1].r_outer

    # Puntos dentro del cable: muestreo uniforme en disco r < r_sheath
    angles = 2.0 * math.pi * torch.rand(n_cable, 1, device=device, dtype=torch.float32)
    us = torch.rand(n_cable, 1, device=device, dtype=torch.float32)
    rs = torch.sqrt(us) * r_sheath
    x_c = cx + rs * torch.cos(angles)
    y_c = cy + rs * torch.sin(angles)
    xy_cable = torch.cat([x_c, y_c], dim=1)
    T_cable = _multilayer_T(xy_cable, layers, placement, k_soil, T_amb, Q_lin)

    # Puntos sobre los cuatro bordes del dominio: objetivo T = T_amb
    n_per = max(1, n_bc // 4)
    xmin, xmax = domain.xmin, domain.xmax
    ymin, ymax = domain.ymin, domain.ymax
    xh = xmin + (xmax - xmin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    xh2 = xmin + (xmax - xmin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    yv = ymin + (ymax - ymin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    yv2 = ymin + (ymax - ymin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    xy_bc = torch.cat([
        torch.cat([xh,  torch.full_like(xh,  ymax)], dim=1),
        torch.cat([xh2, torch.full_like(xh2, ymin)], dim=1),
        torch.cat([torch.full_like(yv,  xmin), yv],  dim=1),
        torch.cat([torch.full_like(yv2, xmax), yv2], dim=1),
    ], dim=0)
    T_bc = torch.full((xy_bc.shape[0], 1), T_amb, device=device, dtype=torch.float32)

    # Combinar interior del cable + contornos
    xy_all = torch.cat([xy_cable, xy_bc], dim=0)
    T_all  = torch.cat([T_cable,  T_bc],  dim=0)

    coord_mins = torch.tensor([xmin, ymin], device=device, dtype=torch.float32)
    coord_maxs = torch.tensor([xmax, ymax], device=device, dtype=torch.float32)

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
    """PINN que aprende la correccion u = T - T_analitico.

    El modelo base solo aprende la correccion de dominio finito respecto al
    perfil de Kennelly (semiplano infinito).  Dado que T_analitico ya satisface
    la ecuacion de Laplace, las CCs y el flujo en la vaina, la red converge
    desde u=0 (solucion trivialmente correcta) sin minimos locales espurios.
    """

    def __init__(
        self,
        base: nn.Module,
        layers: list,
        placement,
        k_soil: float,
        T_amb: float,
        Q_lin: float,
        domain,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self._layers = layers
        self._placement = placement
        self._k_soil = k_soil
        self._T_amb = T_amb
        self._Q_lin = Q_lin
        self._normalize = normalize
        self._xmin = domain.xmin
        self._xmax = domain.xmax
        self._ymin = domain.ymin
        self._ymax = domain.ymax

    def _denormalize(self, xy_n: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor(
            [self._xmin, self._ymin], device=xy_n.device, dtype=xy_n.dtype
        )
        hi = torch.tensor(
            [self._xmax, self._ymax], device=xy_n.device, dtype=xy_n.dtype
        )
        return (xy_n + 1.0) * 0.5 * (hi - lo) + lo

    def forward(self, xy_in: torch.Tensor) -> torch.Tensor:
        xy_phys = self._denormalize(xy_in) if self._normalize else xy_in
        T_bg = _multilayer_T(
            xy_phys,
            self._layers,
            self._placement,
            self._k_soil,
            self._T_amb,
            self._Q_lin,
        )
        u = self.base(xy_in)
        return T_bg + u


def _iec60287_estimate(layers, placement, k_soil: float, Q_scale: float) -> dict:
    """Estimacion analitica IEC 60287 (resistencias termicas en serie).

    Calcula la elevacion de temperatura desde el conductor hasta la
    superficie del terreno mediante formulas de resistencia termica cilindrica
    y la formula de Kennelly para el suelo.
    """
    conductor = layers[0]
    r_cond = conductor.r_outer
    Q_lin = conductor.Q * Q_scale * math.pi * r_cond ** 2  # W/m lineal

    dT_layers: dict[str, float] = {}
    for layer in layers:
        r_in = max(layer.r_inner, 1e-9)
        r_out = layer.r_outer
        if r_out <= r_in:
            continue
        dT_layers[layer.name] = Q_lin / (2.0 * math.pi * layer.k) * math.log(r_out / r_in)

    # Resistencia termica del suelo (formula de Kennelly: d >> r_ext)
    d = abs(placement.cy)
    r_ext = layers[-1].r_outer
    dT_soil = Q_lin / (2.0 * math.pi * k_soil) * math.log(2.0 * d / r_ext)

    dT_total = sum(dT_layers.values()) + dT_soil
    return {
        "Q_lin_W_per_m": Q_lin,
        "dT_by_layer": dT_layers,
        "dT_soil": dT_soil,
        "dT_total": dT_total,
    }


def main() -> None:
    """Cargar datos, entrenar PINN y mostrar resumen comparativo con IEC 60287."""
    parser = argparse.ArgumentParser(
        description="Ejemplo PINN: cable XLPE en zanja estandar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Perfiles disponibles:\n"
            "  quick    : 5 000 Adam + 500 L-BFGS, red 64x4  (~5-8 min CPU)\n"
            "  research : 15 000 Adam + 3 000 L-BFGS, red 128x5 (~25-35 min CPU)\n"
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
    placement = problem.placements[0]

    # Obtener capas del cable (per-cable del catalogo o fallback a cable_layers.csv)
    layers = problem.get_layers(placement.cable_id)

    # Informacion del tipo de cable
    section_mm2 = placement.section_mm2
    material = placement.conductor_material.upper()
    current_A = placement.current_A

    SEP = "=" * 62
    print(SEP)
    if section_mm2 > 0:
        print("  PINN -- Cable XLPE %d mm2 %s / 12-20 kV / %.0f A" % (
            section_mm2, material, current_A))
    else:
        print("  PINN -- Cable XLPE / 12-20 kV / zanja estandar")
    print("  Perfil de ejecucion : %s" % profile.upper())
    print(SEP)

    # Cargar config del perfil seleccionado
    params_csv = DATA_DIR / (
        "solver_params.csv" if profile == "quick" else "solver_params_research.csv"
    )
    solver_params = load_solver_params(params_csv)
    solver_cfg = solver_params.to_solver_cfg()

    adam_n   = solver_cfg["training"]["adam_steps"]
    lbfgs_n  = solver_cfg["training"]["lbfgs_steps"]
    print_ev = solver_cfg["training"]["print_every"]
    width    = solver_cfg["model"]["width"]
    depth    = solver_cfg["model"]["depth"]
    n_int    = solver_cfg["sampling"]["n_interior"]
    n_ifc    = solver_cfg["sampling"]["n_interface"]
    n_bnd    = solver_cfg["sampling"]["n_boundary"]

    conductor = layers[0]
    q_kw  = conductor.Q * scenario.Q_scale / 1000
    q_lin = conductor.Q * scenario.Q_scale * math.pi * conductor.r_outer ** 2
    print("\n  Problema fisico:")
    print("  Escenario   : %s  (%s)" % (scenario.scenario_id, scenario.mode))
    if section_mm2 > 0:
        print("  Cable       : XLPE %d mm2  %s  I=%.0f A" % (
            section_mm2, material, current_A))
    print("  Q_conductor : %.0f kW/m3  (%.2f W/m lineal)" % (q_kw, q_lin))
    print("  k_suelo     : %.1f W/(m*K)  (suelo humedo tipico)" % scenario.k_soil)
    print("  T_ambiente  : %.1f degC  (%.2f K)" % (scenario.T_amb - 273.15, scenario.T_amb))

    # Estimacion analitica IEC 60287 para comparacion
    iec = _iec60287_estimate(
        layers, placement, scenario.k_soil, scenario.Q_scale
    )
    T_ref_K = scenario.T_amb + iec["dT_total"]
    print("\n  Referencia analitica IEC 60287 (resistencias en serie):")
    print("  Calor total    : %.2f W/m lineal" % iec["Q_lin_W_per_m"])
    for name, dT in iec["dT_by_layer"].items():
        print("  dT %-10s   : %+.2f K" % (name, dT))
    print("  dT suelo       : %+.2f K" % iec["dT_soil"])
    print("  dT TOTAL       : %+.2f K" % iec["dT_total"])
    print("  T_cond ref.    : %.1f K  (%.1f degC)" % (T_ref_K, T_ref_K - 273.15))

    # Configuracion del modelo
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(str(RESULTS_DIR), name="example_" + profile)

    normalize = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    base_model = build_model(solver_cfg["model"], in_dim=2, device=device)
    model = ResidualPINNModel(
        base_model,
        layers,
        placement,
        scenario.k_soil,
        scenario.T_amb,
        q_lin,
        problem.domain,
        normalize=normalize,
    )
    _init_output_bias(model.base, 0.0)  # correccion inicia en cero
    n_params = sum(p.numel() for p in model.parameters())

    print("\n  Pre-entrenando en perfil cilindrico multicapa (500 pasos)...", flush=True)
    rmse_pre = _pretrain_cable_plus_bc(
        model, placement, problem.domain,
        layers, q_lin, scenario.k_soil, scenario.T_amb,
        device=device, normalize=normalize,
        n_cable=2000, n_bc=200, n_steps=500, lr=1e-3,
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

    # Geometria del cable
    geo_path = RESULTS_DIR / "geometry.png"
    cable_title = "Cable XLPE %d mm2 %s" % (section_mm2, material) if section_mm2 > 0 else "Cable XLPE"
    plot_cable_geometry(
        layers, placement, problem.domain,
        title="%s -- seccion transversal" % cable_title,
        save_path=geo_path,
    )

    # Entrenamiento PINN
    print("\n" + "-" * 62)
    print("  ENTRENAMIENTO  (Adam --> L-BFGS)")
    print("  Columnas del log:  [fase paso/total pct%%] loss  pde  bc  ifc")
    print("-" * 62)
    trainer = SteadyStatePINNTrainer(
        model=model,
        layers=layers,
        placement=placement,
        domain=problem.domain,
        soil=problem.soil,
        bcs=problem.bcs,
        scenario=scenario,
        solver_cfg=solver_cfg,
        device=device,
        logger=logger,
    )
    history = trainer.train()
    print("-" * 62)

    # Guardar modelo entrenado
    model_path = RESULTS_DIR / "model_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo guardado: %s", model_path)

    # Graficas de perdida y campo de temperatura
    loss_path  = RESULTS_DIR / "loss_history.png"
    T_map_path = RESULTS_DIR / "temperature_field.png"
    plot_loss_history(
        history,
        title="Historia de perdida (%s) -- Adam + L-BFGS" % profile,
        save_path=loss_path,
    )
    normalize = solver_cfg.get("normalization", {}).get("normalize_coords", True)
    X, Y, T = evaluate_on_grid(
        model, problem.domain, nx=300, ny=300,
        device=device, normalize=normalize,
    )
    cable_label = "XLPE %d mm2 %s %.0f A" % (section_mm2, material, current_A) if section_mm2 > 0 else "XLPE"
    plot_temperature_field(
        X, Y, T,
        title="T(x,y) [K] -- %s  [%s]" % (cable_label, profile),
        save_path=T_map_path,
    )

    # Resumen comparativo PINN vs IEC 60287
    T_max_pinn  = float(T.max())
    T_min_pinn  = float(T.min())
    T_amb_K     = scenario.T_amb

    # Evaluacion directa en el centro del conductor (mas precisa que grid)
    with torch.no_grad():
        model.eval()
        pt = torch.tensor([[placement.cx, placement.cy]], device=device, dtype=torch.float32)
        if normalize:
            mins = torch.tensor(
                [problem.domain.xmin, problem.domain.ymin],
                device=device, dtype=torch.float32,
            )
            maxs = torch.tensor(
                [problem.domain.xmax, problem.domain.ymax],
                device=device, dtype=torch.float32,
            )
            pt_in = 2.0 * (pt - mins) / (maxs - mins) - 1.0
        else:
            pt_in = pt
        T_cond_pinn = float(model(pt_in).item())
    loss_final  = history["total"][-1]
    error_K     = T_cond_pinn - T_ref_K

    C = 32  # column width
    print("\n" + "=" * 62)
    print("  RESULTADOS FINALES  [%s]" % profile.upper())
    if section_mm2 > 0:
        print("  Cable: XLPE %d mm2  %s  I=%.0f A" % (section_mm2, material, current_A))
    print("=" * 62)
    print("  %-*s  %10s  %10s  %8s" % (C, "Magnitud", "PINN", "IEC ref.", "Error"))
    print("  " + "-" * 58)
    print("  %-*s  %10.2f  %10.2f  %+7.2f K" % (
        C, "T conductor (K)", T_cond_pinn, T_ref_K, error_K))
    print("  %-*s  %10.1f  %10.1f" % (
        C, "T conductor (degC)", T_cond_pinn - 273.15, T_ref_K - 273.15))
    print("  %-*s  %10.2f  %10.2f" % (
        C, "dT conductor-amb (K)", T_cond_pinn - T_amb_K, iec["dT_total"]))
    print("  %-*s  %10.2f" % (C, "T max dominio (K)", T_max_pinn))
    print("  %-*s  %10.2f" % (C, "T min dominio (K)", T_min_pinn))
    print("  %-*s  %10.4e" % (C, "Perdida final", loss_final))
    print("  " + "-" * 58)
    print("  Limite IEC 60287 XLPE : 363 K (90 degC)")
    print("  Margen termico PINN   : %.1f K  (ref: %.1f K)" % (
        363.0 - T_cond_pinn, 363.0 - T_ref_K))
    if profile == "quick":
        print()
        print("  Para resultados de investigacion (~25-35 min):")
        print("    python examples/xlpe_single_cable/run_example.py --profile research")
    print()
    print("  Graficas guardadas en: %s" % RESULTS_DIR)
    print("    - geometry.png  |  loss_history.png  |  temperature_field.png")
    print("    - model_final.pt  (pesos de la red entrenada)")


if __name__ == "__main__":
    main()
