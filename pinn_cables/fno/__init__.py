"""Fourier Neural Operator (FNO) para conducción de calor 2D.

Módulo `pinn_cables.fno`:

- :mod:`pinn_cables.fno.model`   — arquitectura FNO 2D
- :mod:`pinn_cables.fno.dataset` — generación de datos paramétricos
- :mod:`pinn_cables.fno.train`   — ciclo de entrenamiento (data-driven + physics)
- :mod:`pinn_cables.fno.eval`    — evaluación y extracción de T_max

Referencias
-----------
Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential
Equations", ICLR 2021. https://arxiv.org/abs/2010.08895
"""

from pinn_cables.fno.model import CableFNO2d, SpectralConv2d
from pinn_cables.fno.dataset import CableParametricDataset, make_input_channels
from pinn_cables.fno.train import train_fno, FNOTrainConfig
from pinn_cables.fno.eval import fno_T_max, eval_fno_field

__all__ = [
    "CableFNO2d",
    "SpectralConv2d",
    "CableParametricDataset",
    "make_input_channels",
    "train_fno",
    "FNOTrainConfig",
    "fno_T_max",
    "eval_fno_field",
]
