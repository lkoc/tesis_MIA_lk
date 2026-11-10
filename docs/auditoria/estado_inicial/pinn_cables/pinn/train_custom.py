"""Custom training loop for multi-cable residual PINNs.

Provides:
- :func:`pretrain_multicable`  — warm-start on analytical T_bg + boundary.
- :func:`sample_soil_pts`      — rejection sampling (optional PAC importance).
- :func:`sample_bnd_pts`       — boundary edge sampling.
- :func:`compute_pde_bc_loss`  — PDE (soil) + Dirichlet/Robin BC loss.
- :func:`train_adam_lbfgs`     — Adam → L-BFGS with curriculum + safeguard.
- :func:`init_output_bias`     — set last-layer bias to a given value.
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import torch
import torch.nn as nn

from pinn_cables.physics.k_field import KFieldModel, PhysicsParams
from pinn_cables.physics.kennelly import multilayer_T_multi
from pinn_cables.pinn.pde import pde_residual_steady


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_output_bias(model: nn.Module, value: float) -> None:
    """Set the bias of the last ``nn.Linear`` layer to *value*."""
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        last_linear.bias.data.fill_(value)


def _norm_fn_factory(domain, device: torch.device):
    """Return a normalisation closure [-1, 1]."""
    coord_mins = torch.tensor(
        [domain.xmin, domain.ymin], device=device, dtype=torch.float32,
    )
    coord_maxs = torch.tensor(
        [domain.xmax, domain.ymax], device=device, dtype=torch.float32,
    )

    def norm_fn(xy: torch.Tensor) -> torch.Tensor:
        return 2.0 * (xy - coord_mins) / (coord_maxs - coord_mins) - 1.0

    return norm_fn


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_soil_pts(
    domain,
    placements: list,
    r_sheaths: list[float],
    n: int,
    device: torch.device,
    oversample: int = 8,
    pp: PhysicsParams | None = None,
    frac_pac_bnd: float = 0.30,
    k_model: "KFieldModel | None" = None,
) -> torch.Tensor:
    """Sample *n* points in the soil, excluding cable interiors.

    Transition-region importance sampling adapts to the active k field:

    * When *k_model* is given, transition hints are read from
      :meth:`KFieldModel.transition_hints` and a fraction
      ``frac_pac_bnd`` of points are distributed across **all** hints
      (soil-layer interfaces + PAC boundary).  This supersedes *pp*.

    * When only *pp* is given (backward-compat path), the original
      PAC-boundary sampling is used.

    Args:
        domain:       Computational domain.
        placements:   List of cable placements.
        r_sheaths:    Outer radius of each cable's outermost layer [m].
        n:            Total number of points to return.
        device:       PyTorch device.
        oversample:   Over-sampling factor for rejection sampling.
        pp:           Legacy PhysicsParams (used when k_model is None).
        frac_pac_bnd: Fraction of *n* to allocate to transition regions.
        k_model:      :class:`KFieldModel` instance (preferred over *pp*).
    """
    def _reject_cables(pts: torch.Tensor) -> torch.Tensor:
        in_any = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)
        for idx, pl in enumerate(placements):
            dx = pts[:, 0] - pl.cx
            dy = pts[:, 1] - pl.cy
            in_any |= (dx * dx + dy * dy < r_sheaths[idx] ** 2)
        return pts[~in_any]

    # --- Build list of transition bounding boxes ---
    hint_boxes: list[tuple[float, float, float, float]] = []   # (x_lo, x_hi, y_lo, y_hi)

    if k_model is not None and frac_pac_bnd > 0:
        for hint in k_model.transition_hints():
            if hint["type"] == "horizontal_strip":
                yc = hint["y_centre"]
                hw = hint["half_width"]
                hint_boxes.append((domain.xmin, domain.xmax, yc - hw, yc + hw))
            elif hint["type"] == "pac_boundary":
                hint_boxes.append((hint["x_lo"], hint["x_hi"], hint["y_lo"], hint["y_hi"]))
    elif pp is not None and pp.k_variable and frac_pac_bnd > 0:
        # Backward-compat: keep original PAC sampling
        margin = max(4.0 * pp.k_transition, 0.15)
        hint_boxes.append((
            max(pp.k_cx - pp.k_width / 2.0 - margin, domain.xmin),
            min(pp.k_cx + pp.k_width / 2.0 + margin, domain.xmax),
            max(pp.k_cy - pp.k_height / 2.0 - margin, domain.ymin),
            min(pp.k_cy + pp.k_height / 2.0 + margin, domain.ymax),
        ))

    # Clamp hint boxes to domain
    hint_boxes = [
        (max(xl, domain.xmin), min(xh, domain.xmax),
         max(yl, domain.ymin), min(yh, domain.ymax))
        for xl, xh, yl, yh in hint_boxes
        if xl < xh and yl < yh
    ]

    # Distribute transition fraction evenly across hints
    n_per_hint = int(n * frac_pac_bnd / len(hint_boxes)) if hint_boxes else 0
    n_hints_total = n_per_hint * len(hint_boxes)

    hint_pts_all: list[torch.Tensor] = []
    for (x_lo, x_hi, y_lo, y_hi) in hint_boxes:
        collected_h: list[torch.Tensor] = []
        need_h = n_per_hint
        while need_h > 0:
            xs = x_lo + (x_hi - x_lo) * torch.rand(
                need_h * oversample, 1, device=device, dtype=torch.float32)
            ys = y_lo + (y_hi - y_lo) * torch.rand(
                need_h * oversample, 1, device=device, dtype=torch.float32)
            cands = _reject_cables(torch.cat([xs, ys], dim=1))
            collected_h.append(cands)
            need_h = max(0, n_per_hint - sum(v.shape[0] for v in collected_h))
        hint_pts_all.append(torch.cat(collected_h, dim=0)[:n_per_hint])

    pac_pts_t = (
        torch.cat(hint_pts_all, dim=0)
        if hint_pts_all
        else torch.empty(0, 2, device=device, dtype=torch.float32)
    )

    # --- Uniform points ---
    n_uniform = n - pac_pts_t.shape[0]
    collected: list[torch.Tensor] = []
    need = n_uniform
    while need > 0:
        xs = (domain.xmin + (domain.xmax - domain.xmin)
              * torch.rand(need * oversample, 1, device=device, dtype=torch.float32))
        ys = (domain.ymin + (domain.ymax - domain.ymin)
              * torch.rand(need * oversample, 1, device=device, dtype=torch.float32))
        pts = torch.cat([xs, ys], dim=1)
        valid = _reject_cables(pts)
        collected.append(valid)
        need = max(0, n_uniform - sum(v.shape[0] for v in collected))
    uniform_pts = torch.cat(collected, dim=0)[:n_uniform]

    return torch.cat([pac_pts_t, uniform_pts], dim=0)


def sample_bnd_pts(
    domain, n: int, device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample n/4 random points on each domain edge."""
    n_per = max(1, n // 4)
    xmin, xmax = domain.xmin, domain.xmax
    ymin, ymax = domain.ymin, domain.ymax
    xr = torch.rand(n_per, 1, device=device, dtype=torch.float32)
    yr = torch.rand(n_per, 1, device=device, dtype=torch.float32)
    return {
        "top":    torch.cat([xmin + (xmax - xmin) * xr,                      torch.full_like(xr, ymax)], dim=1),
        "bottom": torch.cat([xmin + (xmax - xmin) * xr.clone(),              torch.full_like(xr, ymin)], dim=1),
        "left":   torch.cat([torch.full_like(yr, xmin), ymin + (ymax - ymin) * yr],                      dim=1),
        "right":  torch.cat([torch.full_like(yr, xmax), ymin + (ymax - ymin) * yr.clone()],              dim=1),
    }


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_pde_bc_loss(
    model: nn.Module,
    xy_soil: torch.Tensor,
    bnd_pts: dict[str, torch.Tensor],
    bcs: dict,
    T_amb: float,
    norm_fn: Callable,
    normalize: bool,
    k_fn: Callable[[torch.Tensor], torch.Tensor] | float,
    w_pde: float,
    w_bc: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PDE residual (soil) + Dirichlet/Robin BC loss.

    Returns ``(total, pde_detached, bc_detached)``.
    """
    pts = xy_soil.clone().detach().requires_grad_(True)
    pts_in = norm_fn(pts) if normalize else pts
    T_pred = model(pts_in)
    k_vals = k_fn(pts) if callable(k_fn) else k_fn
    res_pde = pde_residual_steady(T_pred, pts, k_vals, 0.0)
    loss_pde = torch.mean(res_pde ** 2)

    loss_bc = torch.tensor(0.0, device=xy_soil.device)
    for edge, pts_b in bnd_pts.items():
        bc = bcs.get(edge)
        if bc is None:
            continue
        T_b = model(norm_fn(pts_b) if normalize else pts_b)
        if bc.bc_type in ("dirichlet", "robin"):
            T_tgt = bc.T_target(pts_b, T_amb)
            loss_bc = loss_bc + torch.mean((T_b - T_tgt) ** 2)

    total = w_pde * loss_pde + w_bc * loss_bc
    return total, loss_pde.detach(), loss_bc.detach()


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def pretrain_multicable(
    model: nn.Module,
    placements: list,
    domain,
    layers_list: list[list],
    Q_lins: list[float],
    k_soil: float,
    T_amb: float,
    device: torch.device,
    normalize: bool = True,
    Q_d: float = 0.0,
    n_per_cable: int = 1000,
    n_bc: int = 200,
    n_steps: int = 800,
    lr: float = 1e-3,
    bc_temps: dict | None = None,
) -> float:
    """Pre-train on cable interiors (analytical T_bg) + domain boundaries.

    The target for the full residual model is T_bg (i.e. u → 0).

    *bc_temps* optionally overrides the boundary target temperature for each
    edge ('top', 'bottom', 'left', 'right').  When None (default) all edges
    use *T_amb*, preserving backward-compatible behaviour.  Providing actual
    Dirichlet BC values here is important when those values differ significantly
    from *T_amb* (e.g. a cold bottom isotherm with a warm surface), so that
    the NN starts main training with a correctly-initialized residual.

    Returns the final RMSE in K.
    """
    norm_fn = _norm_fn_factory(domain, device)

    # Sample inside each cable
    cable_pts_list: list[torch.Tensor] = []
    cable_T_list: list[torch.Tensor] = []
    all_layers = (
        layers_list if len(layers_list) == len(placements)
        else layers_list * len(placements)
    )
    for idx, pl in enumerate(placements):
        r_sheath_i = all_layers[idx][-1].r_outer
        angles = 2.0 * math.pi * torch.rand(n_per_cable, 1, device=device, dtype=torch.float32)
        us = torch.rand(n_per_cable, 1, device=device, dtype=torch.float32)
        rs = torch.sqrt(us) * r_sheath_i
        x_c = pl.cx + rs * torch.cos(angles)
        y_c = pl.cy + rs * torch.sin(angles)
        xy_c = torch.cat([x_c, y_c], dim=1)
        T_c = multilayer_T_multi(
            xy_c, all_layers, placements, k_soil, T_amb, Q_lins, Q_d=Q_d,
        )
        cable_pts_list.append(xy_c)
        cable_T_list.append(T_c)

    # Boundary points — per-edge target temperatures
    n_per = max(1, n_bc // 4)
    xmin, xmax = domain.xmin, domain.xmax
    ymin, ymax = domain.ymin, domain.ymax
    xh = xmin + (xmax - xmin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    xh2 = xmin + (xmax - xmin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    yv = ymin + (ymax - ymin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    yv2 = ymin + (ymax - ymin) * torch.rand(n_per, 1, device=device, dtype=torch.float32)
    # Four edges: top (ymax), bottom (ymin), left (xmin), right (xmax)
    _T_top    = bc_temps.get("top",    T_amb) if bc_temps else T_amb
    _T_bottom = bc_temps.get("bottom", T_amb) if bc_temps else T_amb
    _T_left   = bc_temps.get("left",   T_amb) if bc_temps else T_amb
    _T_right  = bc_temps.get("right",  T_amb) if bc_temps else T_amb
    xy_bc = torch.cat([
        torch.cat([xh,  torch.full_like(xh,  ymax)], dim=1),   # top
        torch.cat([xh2, torch.full_like(xh2, ymin)], dim=1),   # bottom
        torch.cat([torch.full_like(yv,  xmin), yv ], dim=1),   # left
        torch.cat([torch.full_like(yv2, xmax), yv2], dim=1),   # right
    ], dim=0)
    T_bc = torch.cat([
        torch.full((n_per, 1), _T_top,    device=device, dtype=torch.float32),
        torch.full((n_per, 1), _T_bottom, device=device, dtype=torch.float32),
        torch.full((n_per, 1), _T_left,   device=device, dtype=torch.float32),
        torch.full((n_per, 1), _T_right,  device=device, dtype=torch.float32),
    ], dim=0)

    xy_all = torch.cat(cable_pts_list + [xy_bc], dim=0)
    T_all = torch.cat(cable_T_list + [T_bc], dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        xy_in = norm_fn(xy_all) if normalize else xy_all
        loss = torch.mean((model(xy_in) - T_all) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        xy_in = norm_fn(xy_all) if normalize else xy_all
        T_pred = model(xy_in)
        rmse = float(torch.sqrt(torch.mean((T_pred - T_all) ** 2)).item())
    return rmse


# ---------------------------------------------------------------------------
# Adam + L-BFGS training loop
# ---------------------------------------------------------------------------

def train_adam_lbfgs(
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
    *,
    step_offset: int = 0,
    total_adam_budget: int = 0,
    k_fn_warmup: Callable[[torch.Tensor], torch.Tensor] | float | None = None,
    warmup_frac: float = 0.0,
    pp: PhysicsParams | None = None,
    lbfgs_history: int = 50,
    k_model: "KFieldModel | None" = None,
    adam2_steps: int = 0,
    adam2_lr: float = 1e-5,
    frac_pac_bnd: float = 0.30,
) -> dict[str, list[float]]:
    """Adam (+ L-BFGS) with optional curriculum warm-up and best-state safeguard.

    **Curriculum training** (when *k_fn_warmup* is given and *warmup_frac* > 0):
    the first ``warmup_frac × adam_steps`` steps use *k_fn_warmup* (homogeneous k),
    then switch to the variable *k_fn*.

    **L-BFGS fixed sample**: collocation points are sampled once before the event
    loop and reused throughout, giving L-BFGS a consistent objective so its
    quasi-Newton Hessian accumulates correctly.  Re-sampling inside the loop
    (the previous behaviour) corrupts the Hessian and causes the 100-1000x loss
    jumps observed in the logs.

    **L-BFGS safeguard**: saves the best model state during L-BFGS and restores
    it if L-BFGS diverges (NaN/Inf).

    **Adam2 fine-tuning** (when *adam2_steps* > 0): a second Adam phase at
    *adam2_lr* after L-BFGS, useful as a fallback when L-BFGS is unstable.

    Returns a history dict with keys ``"total"``, ``"pde"``, ``"bc"``.
    """
    import logging as _logging
    if logger is None:
        logger = _logging.getLogger("train_null")
        logger.addHandler(_logging.NullHandler())
    if print_every <= 0:
        print_every = 10 ** 9

    norm_fn = _norm_fn_factory(domain, device)
    history: dict[str, list[float]] = {"total": [], "pde": [], "bc": []}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    xy_soil: torch.Tensor | None = None
    bnd_pts: dict | None = None
    total_for_pct = total_adam_budget if total_adam_budget > 0 else adam_steps

    # Curriculum threshold
    warmup_step = int(adam_steps * warmup_frac) if k_fn_warmup is not None else 0
    if warmup_step > 0:
        logger.info("Curriculum: %d steps with homogeneous k, then variable k", warmup_step)

    # ---- Adam phase ----
    for step in range(1, adam_steps + 1):
        # Select k function (curriculum)
        use_k = k_fn_warmup if (step <= warmup_step and k_fn_warmup is not None) else k_fn
        # PhysicsParams for importance sampling: None during warmup
        use_pp = pp if (step > warmup_step) else None

        # Force resample at curriculum transition
        if step == warmup_step + 1 and warmup_step > 0:
            logger.info(">>> Curriculum switch: homogeneous k -> variable k")
            xy_soil = None

        if xy_soil is None or (step - 1) % print_every == 0:
            xy_soil = sample_soil_pts(
                domain, placements, r_sheaths, n_int, device, oversample,
                pp=use_pp, k_model=k_model if step > warmup_step else None,
                frac_pac_bnd=frac_pac_bnd,
            )
            bnd_pts = sample_bnd_pts(domain, n_bnd, device)

        optimizer.zero_grad()
        total, l_pde, l_bc = compute_pde_bc_loss(
            model, xy_soil, bnd_pts, bcs, T_amb,
            norm_fn, normalize, use_k, w_pde, w_bc,
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
                global_step, total_for_pct, pct, float(total.detach()), float(l_pde), float(l_bc),
            )

    # ---- L-BFGS phase with best-state safeguard ----
    if lbfgs_steps > 0:
        max_iter = 20
        n_events = max(1, lbfgs_steps // max_iter)
        print_ev_l = max(1, n_events // 25)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=max_iter,
            history_size=lbfgs_history,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        # Sample ONCE — fixed objective for the entire L-BFGS phase so the
        # quasi-Newton Hessian accumulates on a consistent loss landscape.
        xy_s_lbfgs = sample_soil_pts(
            domain, placements, r_sheaths, n_int, device, oversample,
            pp=pp, k_model=k_model,
        )
        bd_p_lbfgs = sample_bnd_pts(domain, n_bnd, device)

        def closure_lbfgs() -> torch.Tensor:
            lbfgs.zero_grad()
            tot, _, _ = compute_pde_bc_loss(
                model, xy_s_lbfgs, bd_p_lbfgs, bcs, T_amb, norm_fn, normalize, k_fn, w_pde, w_bc,
            )
            tot.backward()
            return tot

        # Use final Adam loss as the initial best-loss baseline for safeguard.
        best_loss = history["total"][-1] if history["total"] else float("inf")
        best_state = copy.deepcopy(model.state_dict())
        logger.info("L-BFGS start — Adam final loss (baseline): %.4e", best_loss)
        nan_streak = 0

        for event in range(1, n_events + 1):
            loss_v = lbfgs.step(closure_lbfgs)
            current_loss = float(loss_v.detach()) if loss_v is not None else float("nan")

            if math.isnan(current_loss) or math.isinf(current_loss):
                nan_streak += 1
                logger.warning(
                    "L-BFGS step %d: NaN/Inf — restoring best state", event,
                )
                model.load_state_dict(best_state)
                if nan_streak >= 3:
                    logger.warning("L-BFGS diverged %d times, stopping early", nan_streak)
                    break
                continue

            nan_streak = 0
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = copy.deepcopy(model.state_dict())

            history["total"].append(current_loss)

            if event % print_ev_l == 0:
                pct = 100.0 * (step_offset + adam_steps + event) / (total_for_pct + n_events)
                logger.info("[LBFGS %d/%d  %.1f%%] loss=%.4e", event, n_events, pct, current_loss)

        # Restore best state found during L-BFGS
        model.load_state_dict(best_state)
        logger.info("L-BFGS done — restored best state (loss=%.4e)", best_loss)

    # ---- Optional second Adam fine-tuning phase ----
    if adam2_steps > 0:
        logger.info("Adam2 fine-tuning: %d steps at lr=%.1e", adam2_steps, adam2_lr)
        optimizer2 = torch.optim.Adam(model.parameters(), lr=adam2_lr)
        xy_soil = None
        bnd_pts = None
        for step in range(1, adam2_steps + 1):
            if xy_soil is None or (step - 1) % print_every == 0:
                xy_soil = sample_soil_pts(
                    domain, placements, r_sheaths, n_int, device, oversample,
                    pp=pp, k_model=k_model,
                )
                bnd_pts = sample_bnd_pts(domain, n_bnd, device)
            optimizer2.zero_grad()
            total, l_pde, l_bc = compute_pde_bc_loss(
                model, xy_soil, bnd_pts, bcs, T_amb,
                norm_fn, normalize, k_fn, w_pde, w_bc,
            )
            total.backward()
            optimizer2.step()
            history["total"].append(float(total.detach()))
            history["pde"].append(float(l_pde))
            history["bc"].append(float(l_bc))
            if step % print_every == 0:
                logger.info(
                    "[Adam2 %d/%d] loss=%.4e  pde=%.3e  bc=%.3e",
                    step, adam2_steps, float(total.detach()), float(l_pde), float(l_bc),
                )

    return history
