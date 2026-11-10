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

from pinn_cables.io.readers import load_problem, load_solver_params  # noqa: E402
from pinn_cables.pinn.model import build_model, ResidualPINNModel  # noqa: E402
from pinn_cables.pinn.train_custom import (  # noqa: E402
    init_output_bias,
    pretrain_multicable,
    train_adam_lbfgs,
)
from pinn_cables.pinn.utils import get_device, set_seed, setup_logging  # noqa: E402
from pinn_cables.physics.iec60287 import Q_lin_from_I, iterate_R_T  # noqa: E402
from pinn_cables.physics.k_field import (  # noqa: E402
    load_physics_params,
    make_k_functions,
)
from pinn_cables.physics.kennelly import iec60287_estimate  # noqa: E402
from pinn_cables.post.eval import (  # noqa: E402
    eval_conductor_temps,
    evaluate_on_grid,
)
from pinn_cables.post.plots import (  # noqa: E402
    plot_geometry_multicable,
    plot_k_field,
    plot_loss_history,
    plot_temperature_field,
    plot_zoom_temperature,
)


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
            "  quick    : 5 000 Adam,               red 64x4  (~15-20 min CPU)\n"
            "  research : 10 000 Adam + 500 L-BFGS, red 128x5 (~60-90 min CPU)\n"
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

    # k(x,y): construye tres funciones de conductividad termica del suelo:
    #  - k_fn_pde   : tensor (para la PDE, soporta autograd)
    #  - k_eff_fn_iec: escalar (para la referencia IEC 60287)
    #  - k_eff_bg   : constante (para el fondo Kennelly T_bg)
    k_fn_pde, k_eff_fn_iec, k_eff_bg = make_k_functions(
        pp, k_soil, placements=problem.placements,
    )

    # Iteracion auto-consistente R(T): la resistencia electrica depende de
    # la temperatura, lo que cambia Q = I^2*R(T). Se iteran IEC 60287 hasta
    # que la temperatura converja (tol=0.01 K).
    use_R_T = (pp.n_R_iter > 0)
    if use_R_T:
        iec, Q_lins_iec = iterate_R_T(
            layers_list, problem.placements, k_soil, T_amb,
            I_per_cable, R_per_cable, alpha_per_cable,
            T_ref_R_K=pp.T_ref_R_K,
            n_iter=15, tol=0.01,
            k_eff_fn=k_eff_fn_iec,
        )
    else:
        Q_lins_iec = []
        for i in range(n_cables):
            cond_i = layers_list[i][0]
            Q_lins_iec.append(cond_i.Q * scenario.Q_scale * math.pi * cond_i.r_outer ** 2)
        iec = iec60287_estimate(
            layers_list, problem.placements, k_soil, T_amb,
            Q_lins=Q_lins_iec, k_eff_fn=k_eff_fn_iec,
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
    model = ResidualPINNModel(
        base_model,
        layers_list,
        problem.placements,
        k_eff_bg,
        T_amb,
        Q_lins,
        problem.domain,
        normalize=normalize,
    )
    init_output_bias(model.base, 0.0)
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

    # Grafica de geometria (3 cables por circuito)
    plot_geometry_multicable(
        layers_list, problem.placements, problem.domain,
        title="Cables XLPE -- 3 circuitos Trefoil (detalle)",
        save_path=RESULTS_DIR / "geometry.png",
        circuit_size=3,
    )

    # Grafica campo k(x,y)
    plot_k_field(
        problem.domain, pp, problem.placements, layers_list,
        RESULTS_DIR / "k_field.png",
        k_soil_base=k_soil,
        title="k(x,y) suelo [W/(m\u00b7K)]  --  Trefoil  [%s]" % profile,
    )

    # ------------------------------------------------------------------
    # Pre-entrenamiento
    # ------------------------------------------------------------------
    print("\n  Pre-entrenando en perfiles cilindricos de los %d cables (800 pasos)..." % n_cables, flush=True)
    rmse_pre = pretrain_multicable(
        model, problem.placements, problem.domain, layers_list,
        Q_lins, k_eff_bg, T_amb, device=device, normalize=normalize,
        n_per_cable=1000, n_bc=300, n_steps=800, lr=1e-3,
    )
    print("  Pre-entrenamiento OK: RMSE = %.3f K" % rmse_pre, flush=True)

    # Entrenamiento con iteracion R(T): se reparten los pasos Adam en
    # n_iters rondas.  Tras cada ronda se re-evalua T_cond y se actualiza
    # Q_lin = I^2*R(T_cond) para la siguiente. L-BFGS solo en la ronda final.
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

    for iter_i in range(n_iters):
        is_last = (iter_i == n_iters - 1)
        steps_adam = adam_per_iter if not is_last else (adam_n - adam_per_iter * (n_iters - 1))
        lbfgs_this = lbfgs_n if is_last else 0

        if use_R_T and n_iters > 1:
            print("\n  -- R(T) iteracion %d/%d -- Q_lins=[%s] W/m" % (
                iter_i + 1, n_iters,
                ", ".join("%.3f" % q for q in model._Q_lins)))

        h = train_adam_lbfgs(
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

        # Evaluar T_cond y recalcular Q si hay mas rondas R(T)
        T_cond_pinns = eval_conductor_temps(
            model, problem.placements, problem.domain, device, normalize,
        )
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
    plot_zoom_temperature(
        model, problem.domain, problem.placements, layers_list,
        device, normalize, RESULTS_DIR / "temperature_zoom.png",
        margin=0.12,
        title="T(x,y) [K] -- Zoom conductores  [%s]" % profile,
    )

    # ------------------------------------------------------------------
    # Tabla de resultados
    # ------------------------------------------------------------------
    T_cond_pinns = eval_conductor_temps(
        model, problem.placements, problem.domain, device, normalize,
    )

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
        print("  Para resultados de investigacion (~60-90 min):")
        print("    python examples/xlpe_three_trefoils/run_example.py --profile research")
    print()
    print("  Graficas guardadas en: %s" % RESULTS_DIR)
    print("    - geometry.png  |  loss_history.png  |  temperature_field.png")
    print("    - temperature_zoom.png  |  k_field.png")
    print("    - model_final.pt  (pesos de la red entrenada)")


if __name__ == "__main__":
    main()
