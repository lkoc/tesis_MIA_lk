"""Utility helpers: seeding, device selection, YAML loading, coordinate normalisation."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for ``torch``, ``numpy``, and ``random``.

    Args:
        seed: Integer seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(device_str: str = "auto") -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Args:
        device_str: ``"cpu"``, ``"cuda"``, or ``"auto"`` (CUDA when available).

    Returns:
        Resolved device.

    Raises:
        ValueError: If *device_str* is not recognised.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str in ("cpu", "cuda"):
        return torch.device(device_str)
    raise ValueError(f"Unknown device string: '{device_str}'")


# ---------------------------------------------------------------------------
# YAML configuration
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------

def normalize_coords(
    coords: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
) -> torch.Tensor:
    """Map physical coordinates to the range [-1, 1].

    Args:
        coords: Tensor of shape ``(N, D)``.
        mins:   Per-dimension minima, shape ``(D,)`` or ``(1, D)``.
        maxs:   Per-dimension maxima, shape ``(D,)`` or ``(1, D)``.

    Returns:
        Normalised tensor with the same shape.
    """
    return 2.0 * (coords - mins) / (maxs - mins) - 1.0


def denormalize_coords(
    coords_norm: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
) -> torch.Tensor:
    """Map normalised [-1, 1] coordinates back to physical space.

    Args:
        coords_norm: Normalised tensor.
        mins:        Per-dimension minima.
        maxs:        Per-dimension maxima.

    Returns:
        Physical coordinates tensor.
    """
    return (coords_norm + 1.0) / 2.0 * (maxs - mins) + mins


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "runs/", name: str = "pinn") -> logging.Logger:
    """Configure a logger with console and optional file output.

    Args:
        log_dir: Directory for log files.  Created if it does not exist.
        name:    Logger name.

    Returns:
        Configured :class:`logging.Logger`.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
