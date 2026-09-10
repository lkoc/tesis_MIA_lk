"""Ciclo de entrenamiento para el FNO de cables subterráneos.

Soporta dos modalidades:
    - **Data-driven puro**: minimiza el MSE entre predicción y ground-truth.
    - **Physics-informed (PINO)**: añade residuo PDE como pérdida adicional.

La pérdida total es:
    L = w_data * MSE(T_pred, T_gt) + w_pde * PDE_residual_loss

Referencias
-----------
Li et al. (2021) "Fourier Neural Operator..." arXiv:2010.08895
Herde et al. (2024) "Poseidon: Efficient Foundation Models for PDEs" arXiv:2405.19101
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuración de entrenamiento
# ---------------------------------------------------------------------------

@dataclass
class FNOTrainConfig:
    """Hiperparámetros para entrenar el FNO.

    Attributes:
        epochs:         Número de épocas de entrenamiento.
        batch_size:     Tamaño de lote.
        lr:             Tasa de aprendizaje inicial (Adam).
        lr_decay_step:  Cada cuántas épocas reducir lr por *lr_decay_gamma*.
        lr_decay_gamma: Factor de reducción de lr (scheduler StepLR).
        w_data:         Peso de la pérdida de datos (MSE).
        w_pde:          Peso de la pérdida de PDE (0 = sólo data-driven).
        n_pde_pts:      Puntos de colocación PDE por epoch (cuando w_pde > 0).
        print_every:    Imprimir pérdida cada N épocas.
        device:         Dispositivo torch.
        seed:           Semilla aleatoria.
    """
    epochs: int = 500
    batch_size: int = 16
    lr: float = 1e-3
    lr_decay_step: int = 100
    lr_decay_gamma: float = 0.5
    w_data: float = 1.0
    w_pde: float = 0.0
    n_pde_pts: int = 2000
    print_every: int = 50
    device: str = "auto"
    seed: int = 42


# ---------------------------------------------------------------------------
# Ciclo de entrenamiento
# ---------------------------------------------------------------------------

def train_fno(
    model: nn.Module,
    dataset: Dataset,
    cfg: FNOTrainConfig,
    *,
    val_dataset: Dataset | None = None,
    pde_residual_fn: Callable | None = None,
    domain=None,
    logger=None,
) -> dict[str, list[float]]:
    """Entrenar un FNO con pérdida de datos y opcionalmente de PDE.

    Args:
        model:           Instancia de :class:`CableFNO2d`.
        dataset:         Dataset de entrenamiento.
        cfg:             Configuración de entrenamiento.
        val_dataset:     Dataset de validación (opcional).
        pde_residual_fn: Función ``(xy_pts, T_pred) → residuo`` para pérdida PDE.
                         Requerida cuando ``cfg.w_pde > 0``.
        domain:          Dominio 2D (para muestreo de puntos PDE).
        logger:          Logger estándar de Python.

    Returns:
        Diccionario con históricos:
        ``{"train_total", "train_data", "train_pde", "val_total"}``.
    """
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    torch.manual_seed(cfg.seed)
    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(dataset) > cfg.batch_size,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
        if val_dataset is not None else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_gamma,
    )

    history: dict[str, list[float]] = {
        "train_total": [], "train_data": [], "train_pde": [], "val_total": [],
    }

    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_data, epoch_pde, epoch_total = 0.0, 0.0, 0.0
        n_batches = 0

        for x_batch, T_batch in loader:
            x_batch = x_batch.to(device)   # (B, C_in, N_g, N_g)
            T_batch = T_batch.to(device)   # (B, 1, N_g, N_g)

            optimizer.zero_grad()

            T_pred = model(x_batch)        # (B, 1, N_g, N_g)
            loss_data = torch.mean((T_pred - T_batch) ** 2)

            loss_pde = torch.tensor(0.0, device=device)
            if cfg.w_pde > 0 and pde_residual_fn is not None and domain is not None:
                loss_pde = _compute_pde_loss(
                    model, pde_residual_fn, domain, cfg.n_pde_pts, device,
                )

            loss = cfg.w_data * loss_data + cfg.w_pde * loss_pde
            loss.backward()
            optimizer.step()

            epoch_data  += loss_data.item()
            epoch_pde   += loss_pde.item() if isinstance(loss_pde, torch.Tensor) else float(loss_pde)
            epoch_total += loss.item()
            n_batches   += 1

        scheduler.step()

        avg_data  = epoch_data  / max(n_batches, 1)
        avg_pde   = epoch_pde   / max(n_batches, 1)
        avg_total = epoch_total / max(n_batches, 1)

        history["train_total"].append(avg_total)
        history["train_data"].append(avg_data)
        history["train_pde"].append(avg_pde)

        # Validación
        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, device)
            history["val_total"].append(val_loss)
        else:
            history["val_total"].append(float("nan"))

        if epoch % cfg.print_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            msg = (
                "[FNO epoch %d/%d  %.0fs] "
                "total=%.4e  data=%.4e  pde=%.4e  val=%.4e"
            ) % (epoch, cfg.epochs, elapsed,
                 avg_total, avg_data, avg_pde,
                 history["val_total"][-1])
            if logger is not None:
                logger.info(msg)
            else:
                print(msg, flush=True)

    return history


def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x_batch, T_batch in loader:
            x_batch = x_batch.to(device)
            T_batch = T_batch.to(device)
            T_pred = model(x_batch)
            total += torch.mean((T_pred - T_batch) ** 2).item()
            n += 1
    model.train()
    return total / max(n, 1)


def _compute_pde_loss(
    model: nn.Module,
    pde_residual_fn: Callable,
    domain,
    n_pts: int,
    device: torch.device,
) -> torch.Tensor:
    """Calcular pérdida de PDE muestreando puntos aleatorios del dominio."""
    x_pts = torch.rand(n_pts, device=device) * (domain.xmax - domain.xmin) + domain.xmin
    y_pts = torch.rand(n_pts, device=device) * (domain.ymax - domain.ymin) + domain.ymin
    xy_pts = torch.stack([x_pts, y_pts], dim=1).requires_grad_(True)
    residuals = pde_residual_fn(xy_pts)
    return torch.mean(residuals ** 2)
