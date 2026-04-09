"""Training orchestration for steady-state and transient PINN solvers.

Implements the Adam → L-BFGS two-phase training strategy with optional
periodic re-sampling of collocation points and ``torch.compile`` support.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from pinn_cables.geom.sampler import (
    append_time,
    sample_boundary_points,
    sample_domain_points,
    sample_initial_condition,
    sample_time,
)
from pinn_cables.io.readers import (
    BoundaryCondition,
    CableLayer,
    CablePlacement,
    Domain2D,
    Scenario,
    SoilProperties,
)
from pinn_cables.materials.props import get_Q, get_k, get_rho_c
from pinn_cables.pinn.losses import (
    dirichlet_loss,
    initial_condition_loss,
    interface_T_loss,
    interface_flux_loss,
    mse,
    weighted_total_loss,
)
from pinn_cables.pinn.pde import (
    gradients,
    neumann_residual,
    pde_residual_steady,
    pde_residual_transient,
    robin_residual,
)
from pinn_cables.pinn.utils import normalize_coords


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Hyper-parameters for the training loop.

    Attributes:
        lr:              Initial learning rate for Adam.
        adam_steps:       Number of Adam iterations.
        lbfgs_steps:     Number of L-BFGS outer iterations.
        lbfgs_history:   L-BFGS history size.
        adam2_steps:     Second Adam fine-tuning steps after L-BFGS (0 = disabled).
        adam2_lr:        Learning rate for the second Adam phase.
        print_every:     Logging frequency (Adam steps).
        save_every:      Checkpoint frequency (0 = no checkpointing).
        resample_every:  Re-sample collocation points every N Adam steps.
        use_compile:     Apply ``torch.compile`` to the model.
        checkpoint_dir:  Directory for saved checkpoints.
    """
    lr: float = 1e-3
    adam_steps: int = 20_000
    lbfgs_steps: int = 5_000
    lbfgs_history: int = 50
    adam2_steps: int = 0
    adam2_lr: float = 1e-5
    print_every: int = 500
    save_every: int = 0
    resample_every: int = 0
    use_compile: bool = False
    checkpoint_dir: str = "checkpoints/"

    @classmethod
    def from_config(cls, cfg: dict) -> "TrainConfig":
        """Build from the ``training`` section of the solver YAML."""
        return cls(
            lr=cfg.get("lr", 1e-3),
            adam_steps=cfg.get("adam_steps", 20_000),
            lbfgs_steps=cfg.get("lbfgs_steps", 5_000),
            lbfgs_history=cfg.get("lbfgs_history", 50),
            adam2_steps=cfg.get("adam2_steps", 0),
            adam2_lr=cfg.get("adam2_lr", 1e-5),
            print_every=cfg.get("print_every", 500),
            save_every=cfg.get("save_every", 0),
            resample_every=cfg.get("resample_every", 0),
            use_compile=cfg.get("use_compile", False),
            checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NORMALS: dict[str, list[float]] = {
    "top":    [0.0,  1.0],
    "bottom": [0.0, -1.0],
    "left":   [-1.0, 0.0],
    "right":  [1.0,  0.0],
}


def _normal_tensor(
    boundary: str, n: int, device: torch.device,
) -> torch.Tensor:
    """Unit outward normal for a named rectangular edge."""
    nv = _NORMALS[boundary]
    return torch.tensor(nv, device=device).unsqueeze(0).expand(n, 2)


def _maybe_compile(model: nn.Module, use: bool) -> nn.Module:
    if not use:
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")  # type: ignore[attr-defined]
    except Exception:
        return model


# ---------------------------------------------------------------------------
# Steady-state trainer
# ---------------------------------------------------------------------------

class SteadyStatePINNTrainer:
    """Trainer for the steady-state heat-conduction PINN.

    Args:
        model:     Neural network T_theta(x, y).
        layers:    Cable layers (inner → outer).
        placement: Cable centre position.
        domain:    Computational domain.
        soil:      Soil thermal properties.
        bcs:       Boundary conditions dict.
        scenario:  Active scenario.
        solver_cfg: Full solver YAML dict.
        device:    Torch device.
        logger:    Python logger.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[CableLayer],
        placement: CablePlacement,
        domain: Domain2D,
        soil: SoilProperties,
        bcs: dict[str, BoundaryCondition],
        scenario: Scenario,
        solver_cfg: dict,
        device: torch.device,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.layers = list(layers)
        self.placement = placement
        self.domain = domain
        self.soil = SoilProperties(
            k=scenario.k_soil, rho_c=soil.rho_c,
            variable=soil.variable, amp=soil.amp,
        )
        self.bcs = bcs
        self.scenario = scenario
        self.device = device
        self.logger = logger or logging.getLogger("pinn")
        self.train_cfg = TrainConfig.from_config(solver_cfg.get("training", {}))
        self.weights = solver_cfg.get("loss_weights", {})
        self.samp_cfg = solver_cfg.get("sampling", {})

        self._do_normalize = solver_cfg.get("normalization", {}).get(
            "normalize_coords", True,
        )
        self._coord_mins = torch.tensor(
            [domain.xmin, domain.ymin], device=device, dtype=torch.float32,
        )
        self._coord_maxs = torch.tensor(
            [domain.xmax, domain.ymax], device=device, dtype=torch.float32,
        )

        # Q reference for dimensionless PDE residuals — prevents the large
        # heat-source term (e.g. 150 kW/m³) from swamping the BC losses.
        self._Q_ref = float(max(
            (abs(layer.Q * scenario.Q_scale) for layer in layers),
            default=1.0,
        )) or 1.0  # fallback when every layer has Q=0

        # Cable-surface Neumann condition: at the cable outer surface (sheath),
        # the radial heat flux into the soil must equal Q_lin (W/m).
        # This breaks the trivial T=T_amb local minimum by coupling the
        # conductor heat source to the soil temperature field.
        conductor = self.layers[0]
        self._Q_lin = float(
            conductor.Q * scenario.Q_scale * math.pi * conductor.r_outer ** 2
        )
        self._r_cable_outer = float(self.layers[-1].r_outer)
        # Expected radial gradient at cable surface (k_soil × ∂T/∂r = -Q_lin/(2πr))
        if self._r_cable_outer > 0 and self._Q_lin != 0:
            self._expected_flux_at_cable = float(
                -self._Q_lin / (2.0 * math.pi * self._r_cable_outer)
            )  # < 0: T decreases outward
        else:
            self._expected_flux_at_cable = 0.0

        self.model = _maybe_compile(self.model, self.train_cfg.use_compile)

        # Pre-sample points
        self._sample_all()

    # -- sampling -----------------------------------------------------------

    def _sample_all(self) -> None:
        n_int = self.samp_cfg.get("n_interior", 8000)
        n_ifc = self.samp_cfg.get("n_interface", 500)
        n_bnd = self.samp_cfg.get("n_boundary", 400)

        self.pts_int, self.pts_ifc = sample_domain_points(
            self.domain, self.layers, self.placement,
            n_int, n_ifc,
            oversample=self.samp_cfg.get("oversample", 5),
            min_per_region=self.samp_cfg.get("min_per_region", 20),
            device=self.device,
        )
        self.pts_bnd = sample_boundary_points(
            self.domain, n_bnd, device=self.device,
        )

    # -- normalise helper ---------------------------------------------------

    @staticmethod
    def _fresh(t: torch.Tensor) -> torch.Tensor:
        """Detach and clone so autograd graphs don't persist across iterations."""
        return t.data.clone().requires_grad_(True)

    def _norm(self, xy: torch.Tensor) -> torch.Tensor:
        if self._do_normalize:
            return normalize_coords(xy, self._coord_mins, self._coord_maxs)
        return xy

    # -- loss computation ---------------------------------------------------

    def _compute_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}

        # PDE residual per region — equal weighting per region so that the
        # small-area conductor region (Q ≠ 0) has the same influence as soil.
        pde_region_losses: list[torch.Tensor] = []
        for layer in self.layers:
            xy = self._fresh(self.pts_int[layer.name])
            if xy.shape[0] == 0:
                continue
            T = self.model(self._norm(xy))
            k = get_k(layer, xy, self.soil)
            Q = get_Q(layer, self.scenario.Q_scale)
            pde_region_losses.append(mse(pde_residual_steady(T, xy, k, Q) / self._Q_ref))

        # Soil
        xy_soil = self._fresh(self.pts_int["soil"])
        if xy_soil.shape[0] > 0:
            T_soil = self.model(self._norm(xy_soil))
            k_soil = get_k(None, xy_soil, self.soil)
            pde_region_losses.append(mse(pde_residual_steady(T_soil, xy_soil, k_soil, 0.0) / self._Q_ref))

        if pde_region_losses:
            losses["pde"] = sum(pde_region_losses) / len(pde_region_losses)  # type: ignore[assignment]

        # Boundary conditions
        bc_parts: list[torch.Tensor] = []
        for edge, bc in self.bcs.items():
            pts_raw = self.pts_bnd.get(edge)
            if pts_raw is None or pts_raw.shape[0] == 0:
                continue
            pts = self._fresh(pts_raw)
            T_b = self.model(self._norm(pts))
            if bc.bc_type == "dirichlet":
                target = bc.value if bc.value != 0 else self.scenario.T_amb
                bc_parts.append(
                    (T_b - target).view(-1, 1)
                )
            elif bc.bc_type == "neumann":
                normal = _normal_tensor(edge, pts.shape[0], self.device)
                bc_parts.append(neumann_residual(T_b, pts, normal, bc.value))
            elif bc.bc_type == "robin":
                normal = _normal_tensor(edge, pts.shape[0], self.device)
                bc_parts.append(
                    robin_residual(T_b, pts, normal, self.soil.k, bc.h, bc.value)
                )
        if bc_parts:
            all_bc = torch.cat(bc_parts, dim=0)
            losses["bc_dirichlet"] = mse(all_bc)

        # Interface flux-continuity losses.
        # For each shared interface at r = layers[i].r_outer, enforce
        #   k_inner * (dT/dr) == k_outer * (dT/dr)
        # where k_inner is from layer[i] and k_outer is from layer[i+1]
        # (or soil for the outermost layer).
        #
        # NOTE: with a *global* single-network PINN the gradient dT/dr is
        # continuous, so this loss effectively penalises (k_inner - k_outer)
        # * dT/dr.  For the residual formulation (T = T_bg + u) this is
        # typically disabled (w_interface_flux = 0) because T_bg already
        # captures the inter-layer physics.
        ifc_flux_losses: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            key = f"r_{layer.name}"
            pts_raw = self.pts_ifc.get(key)
            if pts_raw is None or pts_raw.shape[0] == 0:
                continue
            pts = self._fresh(pts_raw)
            T_at_ifc = self.model(self._norm(pts))

            # Radial unit normal at this interface
            gT = gradients(T_at_ifc, pts)
            dx = pts[:, 0:1] - self.placement.cx
            dy = pts[:, 1:2] - self.placement.cy
            r = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-12)
            nr = torch.cat([dx / r, dy / r], dim=1)
            dTdr = (gT * nr).sum(dim=1, keepdim=True)

            # Inner-side conductivity (from layer i)
            k_in = get_k(layer, pts, self.soil)
            if not torch.is_tensor(k_in):
                k_in = torch.tensor(k_in, device=self.device)

            # Outer-side conductivity (from layer i+1 or soil)
            if i + 1 < len(self.layers):
                k_out = get_k(self.layers[i + 1], pts, self.soil)
            else:
                k_out = get_k(None, pts, self.soil)
            if not torch.is_tensor(k_out):
                k_out = torch.tensor(k_out, device=self.device)

            ifc_flux_losses.append(mse(k_in * dTdr - k_out * dTdr))

        if ifc_flux_losses:
            losses["interface_flux"] = sum(ifc_flux_losses) / len(ifc_flux_losses)  # type: ignore[assignment]

        # Cable outer surface flux condition: k_soil × ∂T/∂r = -Q_lin/(2π×r)
        # This couples the cable heat source to the soil temperature field and
        # prevents the trivial solution T ≈ T_amb (which has zero radial flux).
        if self._expected_flux_at_cable != 0.0:
            key = f"r_{self.layers[-1].name}"
            pts_raw = self.pts_ifc.get(key)
            if pts_raw is not None and pts_raw.shape[0] > 0:
                pts_c = self._fresh(pts_raw)
                T_surf = self.model(self._norm(pts_c))
                gT_surf = gradients(T_surf, pts_c)
                dx_c = pts_c[:, 0:1] - self.placement.cx
                dy_c = pts_c[:, 1:2] - self.placement.cy
                r_c = torch.sqrt(dx_c * dx_c + dy_c * dy_c).clamp(min=1e-12)
                nr_c = torch.cat([dx_c / r_c, dy_c / r_c], dim=1)
                dTdr_c = (gT_surf * nr_c).sum(dim=1, keepdim=True)
                expected = self._expected_flux_at_cable  # negative (< 0)
                # Normalise by |expected| so the loss is dimensionless (1.0 for trivial, 0 for correct)
                losses["cable_flux"] = mse(
                    (self.soil.k * dTdr_c - expected) / abs(expected)
                )

        total = weighted_total_loss(losses, self.weights)
        return total, losses

    # -- training loop ------------------------------------------------------

    def train(self) -> dict[str, list[float]]:
        """Run the full Adam → L-BFGS training loop.

        Returns:
            Dict of loss histories (list per component name + ``"total"``).
        """
        cfg = self.train_cfg
        self.model.train()
        history: dict[str, list[float]] = {"total": []}

        total_steps = cfg.adam_steps + cfg.lbfgs_steps
        completed = 0

        # --- Adam phase ---
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        for it in range(1, cfg.adam_steps + 1):
            opt.zero_grad(set_to_none=True)
            total, parts = self._compute_loss()
            total.backward()
            opt.step()

            completed += 1
            history["total"].append(total.item())
            for k, v in parts.items():
                history.setdefault(k, []).append(v.item())

            if it % cfg.print_every == 0:
                pct = completed / total_steps * 100
                self.logger.info(
                    "[Adam %d/%d  %.1f%%] loss=%.4e  %s",
                    it, cfg.adam_steps, pct, total.item(),
                    "  ".join(f"{k}={v.item():.3e}" for k, v in parts.items()),
                )

            if cfg.resample_every > 0 and it % cfg.resample_every == 0:
                self._sample_all()

            if cfg.save_every > 0 and it % cfg.save_every == 0:
                self._save_checkpoint(it)

        # --- L-BFGS phase ---
        if cfg.lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=20,
                history_size=cfg.lbfgs_history,
                line_search_fn="strong_wolfe",
            )
            outer_iters = max(1, cfg.lbfgs_steps // 20)
            steps_per_outer = cfg.lbfgs_steps / outer_iters

            # Save checkpoint so we can restore if L-BFGS diverges (NaN/Inf)
            best_lbfgs_loss = history["total"][-1] if history["total"] else float("inf")
            best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            nan_streak = 0

            for oi in range(1, outer_iters + 1):
                def closure() -> torch.Tensor:
                    lbfgs.zero_grad(set_to_none=True)
                    total, _ = self._compute_loss()
                    total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    return total

                loss_val = lbfgs.step(closure)
                completed += int(steps_per_outer)

                current_loss = float(loss_val) if loss_val is not None else float("nan")
                if math.isnan(current_loss) or math.isinf(current_loss):
                    nan_streak += 1
                    self.logger.warning(
                        "L-BFGS step %d: NaN/Inf detected, restoring last good state", oi,
                    )
                    self.model.load_state_dict(best_state)
                    if nan_streak >= 3:
                        self.logger.warning("L-BFGS diverged %d times, stopping early", nan_streak)
                        break
                    continue
                nan_streak = 0
                if current_loss < best_lbfgs_loss:
                    best_lbfgs_loss = current_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                history["total"].append(current_loss)
                if oi % max(1, outer_iters // 10) == 0:
                    pct = min(completed / total_steps * 100, 100.0)
                    self.logger.info(
                        "[LBFGS %d/%d  %.1f%%] loss=%.4e", oi, outer_iters, pct, current_loss,
                    )

        # --- Optional second Adam fine-tuning phase ---
        if cfg.adam2_steps > 0:
            self.logger.info(
                "Adam2 fine-tuning: %d steps at lr=%.1e", cfg.adam2_steps, cfg.adam2_lr,
            )
            opt2 = torch.optim.Adam(self.model.parameters(), lr=cfg.adam2_lr)
            for step2 in range(1, cfg.adam2_steps + 1):
                if cfg.resample_every > 0 and step2 % cfg.resample_every == 0:
                    self._sample_all()
                opt2.zero_grad()
                total2, _ = self._compute_loss()
                total2.backward()
                opt2.step()
                history["total"].append(float(total2.detach()))
                if step2 % cfg.print_every == 0:
                    self.logger.info(
                        "[Adam2 %d/%d] loss=%.4e", step2, cfg.adam2_steps, float(total2.detach()),
                    )

        self.logger.info("Training complete (100%%). Final loss=%.4e", history["total"][-1])
        return history

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = Path(self.train_cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"model_step_{step}.pt"
        torch.save(self.model.state_dict(), path)
        self.logger.info("Checkpoint saved: %s", path)


# ---------------------------------------------------------------------------
# Transient trainer
# ---------------------------------------------------------------------------

class TransientPINNTrainer(SteadyStatePINNTrainer):
    """Trainer for the transient heat-conduction PINN.

    Extends :class:`SteadyStatePINNTrainer` with a time dimension,
    initial-condition loss, and space-time sampling.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002
        super().__init__(*args, **kwargs)
        self.t_end = self.scenario.t_end
        self.T_init = self.scenario.T_amb
        time_cfg = kwargs.get("solver_cfg", args[7] if len(args) > 7 else {}).get("time", {})
        self.n_time = time_cfg.get("n_time", 200)

        # Adjust normalisation bounds for 3-D (x, y, t)
        self._coord_mins = torch.tensor(
            [self.domain.xmin, self.domain.ymin, 0.0],
            device=self.device, dtype=torch.float32,
        )
        self._coord_maxs = torch.tensor(
            [self.domain.xmax, self.domain.ymax, self.t_end],
            device=self.device, dtype=torch.float32,
        )

        # IC points
        self.pts_ic = sample_initial_condition(
            self.domain,
            n_ic=self.samp_cfg.get("n_interior", 8000) // 4,
            t0=0.0,
            device=self.device,
        )

        # Pre-sample time values
        self.t_samples = sample_time(
            self.n_time, 0.0, self.t_end, device=self.device,
        )

    def _compute_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}

        # PDE residual per region — equal weighting per region
        pde_region_losses: list[torch.Tensor] = []
        for layer in self.layers:
            xy = self._fresh(self.pts_int[layer.name])
            if xy.shape[0] == 0:
                continue
            xyt = append_time(xy, self._fresh(self.t_samples))
            T = self.model(self._norm(xyt))
            k = get_k(layer, xyt[:, :2], self.soil)
            Q = get_Q(layer, self.scenario.Q_scale)
            rho_c = get_rho_c(layer, self.soil)
            pde_region_losses.append(mse(pde_residual_transient(T, xyt, k, rho_c, Q) / self._Q_ref))

        xy_soil = self._fresh(self.pts_int["soil"])
        if xy_soil.shape[0] > 0:
            xyt_s = append_time(xy_soil, self._fresh(self.t_samples))
            T_s = self.model(self._norm(xyt_s))
            k_s = get_k(None, xyt_s[:, :2], self.soil)
            rho_c_s = get_rho_c(None, self.soil)
            pde_region_losses.append(mse(pde_residual_transient(T_s, xyt_s, k_s, rho_c_s, 0.0) / self._Q_ref))

        if pde_region_losses:
            losses["pde"] = sum(pde_region_losses) / len(pde_region_losses)  # type: ignore[assignment]

        # Boundary conditions (at random times)
        bc_parts: list[torch.Tensor] = []
        for edge, bc in self.bcs.items():
            pts_raw = self.pts_bnd.get(edge)
            if pts_raw is None or pts_raw.shape[0] == 0:
                continue
            xyt_b = append_time(self._fresh(pts_raw), self._fresh(self.t_samples))
            T_b = self.model(self._norm(xyt_b))
            if bc.bc_type == "dirichlet":
                target = bc.value if bc.value != 0 else self.scenario.T_amb
                bc_parts.append((T_b - target).view(-1, 1))
            elif bc.bc_type == "neumann":
                normal_2d = _normal_tensor(edge, xyt_b.shape[0], self.device)
                gT = gradients(T_b, xyt_b)
                dTdn = (gT[:, :2] * normal_2d).sum(dim=1, keepdim=True)
                bc_parts.append(dTdn - bc.value)
        if bc_parts:
            losses["bc_dirichlet"] = mse(torch.cat(bc_parts, dim=0))

        # Initial condition
        pts_ic = self._fresh(self.pts_ic)
        T_ic = self.model(self._norm(pts_ic))
        losses["ic"] = initial_condition_loss(T_ic, self.T_init)

        total = weighted_total_loss(losses, self.weights)
        return total, losses
