"""IEC 60287 heat-loss computation and temperature-dependent resistance.

Provides:
- :func:`compute_iec60287_Q` — full IEC 60287 heat calculation with skin effect.
- :func:`Q_lin_from_I` — linear heat [W/m] from current and R(T).
"""

from __future__ import annotations

import math

from pinn_cables.materials.props import get_R_dc_20, get_alpha_R


def compute_iec60287_Q(
    section_mm2: int,
    material: str,
    current_A: float,
    T_op: float,
    W_d: float = 0.0,
    freq: float = 50.0,
) -> dict:
    """IEC 60287 simplified heat calculation.

    Includes R(T), skin-effect factor for solid round conductors,
    and dielectric losses.

    Args:
        section_mm2: Nominal conductor cross-section [mm²].
        material:    ``"cu"`` or ``"al"``.
        current_A:   Operating current [A].
        T_op:        Operating temperature [K].
        W_d:         Dielectric losses [W/m].
        freq:        Network frequency [Hz].

    Returns:
        Dict with ``R_dc_20``, ``R_dc_T``, ``ys``, ``R_ac``,
        ``Q_cond_W_per_m``, ``W_d``, ``Q_total_W_per_m``,
        ``ratio_vs_Rdc20``.
    """
    R_dc_20 = get_R_dc_20(section_mm2, material)
    alpha_R = get_alpha_R(material)

    R_dc_T = R_dc_20 * (1.0 + alpha_R * (T_op - 293.15))

    # Skin-effect factor (IEC 60287-1-1, solid round conductors)
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


def Q_lin_from_I(
    I: float,
    R_ref: float,
    alpha_R: float,
    T_cond: float,
    T_ref: float,
) -> float:
    """Linear heat [W/m] = I² R(T); R(T) = R_ref × (1 + α(T − T_ref)).

    Args:
        I:       Current [A].
        R_ref:   DC resistance at *T_ref* [Ω/m].
        alpha_R: Temperature coefficient of resistance [1/K].
        T_cond:  Conductor temperature [K].
        T_ref:   Reference temperature for *R_ref* [K].
    """
    R_T = R_ref * (1.0 + alpha_R * (T_cond - T_ref))
    return I * I * R_T


# ---------------------------------------------------------------------------
# Iteracion R(T) para estimacion IEC con dependencia termica
# ---------------------------------------------------------------------------

def iterate_R_T(
    layers_list: list[list],
    placements: list,
    k_soil: float,
    T_amb: float,
    I_per_cable: list[float],
    R_per_cable: list[float],
    alpha_per_cable: list[float],
    T_ref_R_K: float,
    n_iter: int = 15,
    tol: float = 0.01,
    k_eff_fn=None,
    Q_d: float = 0.0,
) -> tuple[dict, list[float]]:
    """Iteracion R(T) autoconsistente para estimacion IEC.

    Dado que la resistencia del conductor depende de la temperatura
    (R aumenta con T → Q aumenta → T sube), se itera hasta convergencia:

    1. Estimar Q_lin(T) para cada cable.
    2. Calcular T_cond con iec60287_estimate.
    3. Verificar convergencia (|ΔT| < tol).
    4. Repetir.

    Args:
        layers_list:     Capas por cable.
        placements:      Posiciones de cables.
        k_soil:          k base del suelo [W/(mK)].
        T_amb:           Temperatura ambiente [K].
        I_per_cable:     Corriente [A] por cable.
        R_per_cable:     R_dc a T_ref [Ω/m] por cable.
        alpha_per_cable: Coeficiente α_R [1/K] por cable.
        T_ref_R_K:       Temperatura de referencia para R [K].
        n_iter:          Iteraciones máximas.
        tol:             Tolerancia de convergencia [K].
        k_eff_fn:        Función k(x,y) escalar para IEC (opcional).
        Q_d:             Pérdidas dieléctricas [W/m] (opcional).

    Returns:
        Tupla ``(iec_result, Q_lins)`` donde *iec_result* es el dict
        de :func:`~pinn_cables.physics.kennelly.iec60287_estimate` y
        *Q_lins* la lista de calor lineal final [W/m].
    """
    # Importar aqui para evitar dependencia circular
    from pinn_cables.physics.kennelly import iec60287_estimate

    n_cables = len(placements)
    T_cond_est = T_amb + 50.0  # estimacion inicial

    Q_lins = [Q_lin_from_I(I_per_cable[i], R_per_cable[i],
                           alpha_per_cable[i], T_amb, T_ref_R_K)
              for i in range(n_cables)]

    iec = None
    for _ in range(n_iter):
        Q_lins = [Q_lin_from_I(I_per_cable[i], R_per_cable[i],
                               alpha_per_cable[i], T_cond_est, T_ref_R_K)
                  for i in range(n_cables)]
        iec = iec60287_estimate(
            layers_list, placements, k_soil, T_amb,
            Q_lins=Q_lins, k_eff_fn=k_eff_fn, Q_d=Q_d,
        )
        T_new = iec["T_cond_ref"]
        if abs(T_new - T_cond_est) < tol:
            break
        T_cond_est = T_new

    assert iec is not None
    return iec, Q_lins
