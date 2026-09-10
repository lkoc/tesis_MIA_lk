"""Analytical background temperature: Kennelly superposition + cylindrical multilayer.

Provides:
- :func:`multilayer_T_multi` — T_bg for N cables (generalised).
- :func:`iec60287_estimate` — IEC 60287 analytical estimate for N cables.

The *multilayer_T_multi* function is the unified background temperature used
by :class:`~pinn_cables.pinn.model.ResidualPINNModel`.
"""

from __future__ import annotations

import math

import torch


def multilayer_T_multi(
    xy: torch.Tensor,
    layers_list: list[list],
    placements: list,
    k_soil: float,
    T_amb: float,
    Q_lins: list[float],
    Q_d: float = 0.0,
    *,
    enable_grad: bool = False,
) -> torch.Tensor:
    """Analytical temperature for N cables: Kennelly superposition + multilayer.

    Supports cables of different type (different layer stacks) and different
    Q_lin per cable.  Optional dielectric losses ``Q_d`` are applied as a
    volumetric source in the XLPE layer (same Q_d for every cable).

    When *enable_grad* is ``False`` (default) the function runs inside
    ``torch.no_grad()`` — appropriate for the fixed analytical background.
    Set *enable_grad*=True when gradient flow through T_bg is needed
    (e.g. for a variable-k PDE where T_bg depends on spatial coordinates).

    Args:
        xy:          Physical coordinates ``(N, 2)`` in metres.
        layers_list: ``layers_list[i]`` is the list of CableLayer for cable *i*.
                     If all cables share the same type, you may pass a single
                     list and it will be broadcast.
        placements:  List of CablePlacement (one per cable).
        k_soil:      Effective soil thermal conductivity [W/(m K)].
        T_amb:       Ambient temperature [K] at the surface (y = 0).
        Q_lins:      Linear heat [W/m] per cable.
        Q_d:         Dielectric losses [W/m] in XLPE (0 to disable).
        enable_grad: Allow gradient flow through this function.

    Returns:
        Temperature tensor ``(N, 1)`` in K.
    """
    if enable_grad:
        return _multilayer_T_multi_impl(
            xy, layers_list, placements, k_soil, T_amb, Q_lins, Q_d,
        )
    with torch.no_grad():
        return _multilayer_T_multi_impl(
            xy, layers_list, placements, k_soil, T_amb, Q_lins, Q_d,
        )


def _multilayer_T_multi_impl(
    xy: torch.Tensor,
    layers_list: list[list],
    placements: list,
    k_soil: float,
    T_amb: float,
    Q_lins: list[float],
    Q_d: float,
) -> torch.Tensor:
    """Core implementation (called with or without torch.no_grad)."""
    n_cables = len(placements)

    # Broadcast shared layers to all cables if needed
    if len(layers_list) == 1 and n_cables > 1:
        layers_list = layers_list * n_cables

    N = xy.shape[0]
    r_sheaths = [ls[-1].r_outer for ls in layers_list]

    # Dielectric volumetric source in XLPE (if Q_d > 0)
    q_vol_d = 0.0
    xlpe_ri = xlpe_ro = 0.0
    if Q_d > 0.0:
        for lyr in layers_list[0]:
            if lyr.name == "xlpe":
                xlpe_ri = lyr.r_inner
                xlpe_ro = lyr.r_outer
                q_vol_d = Q_d / (math.pi * (xlpe_ro**2 - xlpe_ri**2))
                break

    # ---- T_sheath_outer per cable (scalar) ----
    T_sheath_outers: list[float] = []
    for i, pl_i in enumerate(placements):
        d_i = abs(pl_i.cy)
        Q_i = Q_lins[i]
        r_sh_i = r_sheaths[i]
        T_s_i = T_amb + Q_i / (2.0 * math.pi * k_soil) * math.log(2.0 * d_i / r_sh_i)
        # Mutual heating from other cables
        for j, pl_j in enumerate(placements):
            if i == j:
                continue
            Q_j = Q_lins[j]
            d_j = abs(pl_j.cy)
            dist_ij = math.sqrt((pl_i.cx - pl_j.cx) ** 2 + (pl_i.cy - pl_j.cy) ** 2)
            r_img_sq = (pl_i.cx - pl_j.cx) ** 2 + (pl_i.cy - d_j) ** 2
            T_s_i += Q_j / (4.0 * math.pi * k_soil) * math.log(r_img_sq / dist_ij**2)
        T_sheath_outers.append(T_s_i)

    # ---- Layer‑interface temperatures (outer→inner), per cable ----
    layer_T_outer_list: list[dict[str, float]] = []
    for i in range(n_cables):
        layers = layers_list[i]
        Q_i = Q_lins[i]
        T_curr = T_sheath_outers[i]
        layer_T_outer: dict[str, float] = {}
        for layer in reversed(layers):
            layer_T_outer[layer.name] = T_curr
            r_out = layer.r_outer
            r_in = max(layer.r_inner, 1e-9)
            if layer.name == "xlpe" and Q_d > 0.0:
                Q_cond_eff = Q_i - Q_d
                T_curr += (
                    Q_cond_eff / (2.0 * math.pi * layer.k) * math.log(r_out / r_in)
                    + q_vol_d
                    / (2.0 * layer.k)
                    * ((r_out**2 - r_in**2) / 2.0 - r_in**2 * math.log(r_out / r_in))
                )
            elif layer.r_inner == 0.0 and Q_i > 0.0:
                Q_cond_eff = Q_i - Q_d
                Q_vol = Q_cond_eff / (math.pi * r_out**2)
                T_curr += Q_vol / (4.0 * layer.k) * r_out**2
            else:
                T_curr += Q_i / (2.0 * math.pi * layer.k) * math.log(r_out / r_in)
        layer_T_outer_list.append(layer_T_outer)

    # ---- Radial distances per cable ----
    r_per_cable: list[torch.Tensor] = []
    for pl in placements:
        dx = xy[:, 0:1] - pl.cx
        dy = xy[:, 1:2] - pl.cy
        r_per_cable.append(torch.sqrt(dx * dx + dy * dy).clamp(min=1e-9))

    # ---- Soil mask: outside ALL cables ----
    soil_mask = torch.ones(N, dtype=torch.bool, device=xy.device)
    for i, r_i in enumerate(r_per_cable):
        soil_mask &= r_i.squeeze(1) >= r_sheaths[i]

    # ---- Soil: Kennelly superposition from all cables ----
    dT_soil = torch.zeros(N, 1, device=xy.device, dtype=xy.dtype)
    for i, pl in enumerate(placements):
        Q_i = Q_lins[i]
        r_sh_i = r_sheaths[i]
        dx = xy[:, 0:1] - pl.cx
        d_pl = abs(pl.cy)
        dy_img = xy[:, 1:2] - d_pl
        dy_r = xy[:, 1:2] - pl.cy
        r_img_sq = (dx * dx + dy_img * dy_img).clamp(min=1e-20)
        r_sq = (dx * dx + dy_r * dy_r).clamp(min=r_sh_i**2)
        dT_soil += Q_i / (4.0 * math.pi * k_soil) * torch.log(r_img_sq / r_sq)

    result = torch.full((N, 1), T_amb, device=xy.device, dtype=xy.dtype)
    result[soil_mask] = T_amb + dT_soil[soil_mask].clamp(min=0.0)

    # ---- Interior of each cable: cylindrical 1-D profile ----
    for i, pl in enumerate(placements):
        layers = layers_list[i]
        Q_i = Q_lins[i]
        r = r_per_cable[i]
        r_sh_i = r_sheaths[i]
        layer_T_outer = layer_T_outer_list[i]
        cable_mask = r.squeeze(1) < r_sh_i

        for layer in reversed(layers):
            r_out = layer.r_outer
            r_in = max(layer.r_inner, 1e-9)
            T_out_layer = layer_T_outer[layer.name]
            mask = ((r >= layer.r_inner) & (r < r_out)).squeeze(1) & cable_mask
            if not mask.any():
                continue
            r_pts = r[mask, 0]
            if layer.name == "xlpe" and Q_d > 0.0:
                Q_cond_eff = Q_i - Q_d
                r_c = r_pts.clamp(min=r_in)
                log_ro_r = torch.log(r_out / r_c)
                dT = (
                    Q_cond_eff / (2.0 * math.pi * layer.k) * log_ro_r
                    + q_vol_d
                    / (2.0 * layer.k)
                    * ((r_out**2 - r_c**2) / 2.0 - r_in**2 * log_ro_r)
                )
            elif layer.r_inner == 0.0 and Q_i > 0.0:
                Q_cond_eff = Q_i - Q_d
                Q_vol = Q_cond_eff / (math.pi * r_out**2)
                dT = Q_vol / (4.0 * layer.k) * (r_out**2 - r_pts**2)
            else:
                dT = Q_i / (2.0 * math.pi * layer.k) * torch.log(
                    r_out / r_pts.clamp(min=r_in)
                )
            result[mask, 0] = T_out_layer + dT

    return result


# ---------------------------------------------------------------------------
# IEC 60287 analytical estimate for N cables
# ---------------------------------------------------------------------------

def iec60287_estimate(
    layers_list: list[list],
    placements: list,
    k_soil: float,
    T_amb: float,
    Q_lins: list[float],
    Q_d: float = 0.0,
    k_eff_fn=None,
) -> dict:
    """IEC 60287 analytical estimate for N cables.

    Computes conductor temperature with cylindrical layer resistances
    plus Kennelly mutual heating.

    Args:
        layers_list: Layer stacks per cable (broadcast if single list).
        placements:  Cable centre positions.
        k_soil:      Effective soil conductivity [W/(m K)].
        T_amb:       Ambient temperature [K].
        Q_lins:      Linear heat [W/m] per cable.
        Q_d:         Dielectric losses [W/m] in XLPE.
        k_eff_fn:    Optional ``k(x, y)`` scalar callable for Kennelly dT_soil.

    Returns:
        Dict with ``cables``, ``hottest_idx``, ``T_cond_ref``, ``Q_lins_W_per_m``,
        ``dT_by_layer``, ``dT_cable``.
    """
    n_cables = len(placements)
    if len(layers_list) == 1 and n_cables > 1:
        layers_list = layers_list * n_cables

    # Dielectric source density
    q_vol_d = 0.0
    xlpe_ri = xlpe_ro = 0.0
    if Q_d > 0.0:
        for lyr in layers_list[0]:
            if lyr.name == "xlpe":
                xlpe_ri = lyr.r_inner
                xlpe_ro = lyr.r_outer
                q_vol_d = Q_d / (math.pi * (xlpe_ro**2 - xlpe_ri**2))
                break

    # Layer dT per cable
    dT_layers_list: list[dict[str, float]] = []
    dT_cable_totals: list[float] = []
    for i in range(n_cables):
        layers = layers_list[i]
        Q_i = Q_lins[i]
        dT_layers: dict[str, float] = {}
        for layer in layers:
            r_in = max(layer.r_inner, 1e-9)
            r_out = layer.r_outer
            if r_out <= r_in:
                continue
            if layer.name == "xlpe" and Q_d > 0.0:
                Q_cond_eff = Q_i - Q_d
                dT_layers[layer.name] = (
                    Q_cond_eff / (2.0 * math.pi * layer.k) * math.log(r_out / r_in)
                    + q_vol_d
                    / (2.0 * layer.k)
                    * ((r_out**2 - r_in**2) / 2.0 - r_in**2 * math.log(r_out / r_in))
                )
            elif layer.r_inner == 0.0:
                Q_cond_eff = Q_i - Q_d
                Q_vol_c = Q_cond_eff / (math.pi * r_out**2)
                dT_layers[layer.name] = Q_vol_c / (4.0 * layer.k) * r_out**2
            else:
                dT_layers[layer.name] = (
                    Q_i / (2.0 * math.pi * layer.k) * math.log(r_out / r_in)
                )
        dT_layers_list.append(dT_layers)
        dT_cable_totals.append(sum(dT_layers.values()))

    # dT_soil per cable: Kennelly self + mutual
    cable_results = []
    for i, pl_i in enumerate(placements):
        layers = layers_list[i]
        r_sheath = layers[-1].r_outer
        d_i = abs(pl_i.cy)
        k_i = k_eff_fn(pl_i.cx, pl_i.cy) if k_eff_fn is not None else k_soil
        dT_soil_i = Q_lins[i] / (2.0 * math.pi * k_i) * math.log(2.0 * d_i / r_sheath)
        for j, pl_j in enumerate(placements):
            if i == j:
                continue
            d_j = abs(pl_j.cy)
            dist_ij = math.sqrt((pl_i.cx - pl_j.cx) ** 2 + (pl_i.cy - pl_j.cy) ** 2)
            r_img_sq = (pl_i.cx - pl_j.cx) ** 2 + (pl_i.cy - d_j) ** 2
            dT_soil_i += Q_lins[j] / (4.0 * math.pi * k_i) * math.log(
                r_img_sq / dist_ij**2
            )
        T_cond_i = T_amb + dT_soil_i + dT_cable_totals[i]
        cable_results.append(
            {
                "cable_id": i + 1,
                "cx": pl_i.cx,
                "cy": pl_i.cy,
                "dT_soil": dT_soil_i,
                "T_cond": T_cond_i,
            }
        )

    hottest_idx = max(range(len(cable_results)), key=lambda k: cable_results[k]["T_cond"])
    return {
        "Q_lins_W_per_m": list(Q_lins),
        "Q_lin_W_per_m": max(Q_lins),
        "dT_by_layer": dT_layers_list[hottest_idx],
        "dT_cable": dT_cable_totals[hottest_idx],
        "cables": cable_results,
        "hottest_idx": hottest_idx,
        "T_cond_ref": cable_results[hottest_idx]["T_cond"],
    }
