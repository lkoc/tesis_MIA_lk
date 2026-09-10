"""Arquitectura Fourier Neural Operator 2D para campos de temperatura.

Implementa la arquitectura FNO descrita en:

    Li et al. (2021) "Fourier Neural Operator for Parametric Partial
    Differential Equations", ICLR 2021.  arXiv:2010.08895

Para el problema de cables subterráneos, el FNO aprende el operador solución:

    G : (k_field, Q_sources, BC_mask) → T(x, y)

donde todas las funciones están discretizadas en una malla uniforme N_g × N_g.

Clases
------
SpectralConv2d
    Capa de convolución espectral en espacio de Fourier.
FNOBlock
    Bloque FNO: SpectralConv2d + convolución 1×1 de salto + activación.
CableFNO2d
    Red FNO completa: lifting → L bloques FNO → projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Convolución espectral 2D (núcleo de Fourier aprendible).

    Multiplica los ``modes1 × modes2`` modos de Fourier de baja frecuencia
    por pesos complejos aprendibles, luego aplica FFT inversa.

    La integral en espacio de Fourier aproxima:
        (K · v)(x) = F^{-1}[R · F[v](ξ)](x)
    donde R ∈ C^{C_in × C_out × modes1 × modes2} es la matriz de pesos.

    Args:
        in_channels:  Canales de entrada.
        out_channels: Canales de salida.
        modes1:       Modos de Fourier retenidos en dimensión x.
        modes2:       Modos de Fourier retenidos en dimensión y.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(
        x: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Multiplicación matricial compleja: (batch, C_in, m1, m2) × (C_in, C_out, m1, m2)."""
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica la convolución espectral.

        Args:
            x: ``(batch, C_in, N_x, N_y)`` tensor de características.

        Returns:
            ``(batch, C_out, N_x, N_y)`` después de convolución espectral.
        """
        bsz, _, N_x, N_y = x.shape

        # FFT 2D → espacio de Fourier (solo frecuencias reales)
        x_ft = torch.fft.rfft2(x)  # (batch, C_in, N_x, N_y//2+1)

        out_ft = torch.zeros(
            bsz, self.out_channels, N_x, N_y // 2 + 1,
            device=x.device, dtype=torch.cfloat,
        )

        # Esquina superior-izquierda (frecuencias bajas positivas)
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        # Esquina superior-derecha (frecuencias negativas en x → modes1 desde el final)
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        # FFT inversa → espacio físico
        return torch.fft.irfft2(out_ft, s=(N_x, N_y))


class FNOBlock(nn.Module):
    """Bloque FNO: convolución espectral + skip 1×1 + activación.

    Implementa la actualización:
        v_{l+1}(x) = σ( F^{-1}[R_l · F[v_l]] + W_l v_l )

    donde W_l es una convolución 1×1 (transformación lineal punto-a-punto).

    Args:
        channels: Dimensión de canal de los features en todo el bloque.
        modes1:   Modos de Fourier en dimensión x.
        modes2:   Modos de Fourier en dimensión y.
        activation: Función de activación (``"gelu"`` recomendado para FNO).
    """

    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.skip = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.skip(x))


class CableFNO2d(nn.Module):
    """FNO 2D para mapeo (k_field, Q_src, BC) → T(x,y) en cables subterráneos.

    Arquitectura:
        1. **Lifting** (P): proyección lineal puntual C_in → d_model
        2. **L bloques FNO**: cada bloque mezcla espectral + skip de features
        3. **Projection** (Q): proyección no-lineal d_model → 1

    La resolución de la malla N_g × N_g es arbitraria en inferencia (zero-shot
    super-resolution si N_g cambia entre entrenamiento y evaluación).

    Args:
        in_channels:  Número de canales de entrada (default 3: k, Q, BC_mask).
        d_model:      Dimensión de canal interna.
        n_layers:     Número de bloques FNO.
        modes1:       Modos de Fourier retenidos en dimensión x.
        modes2:       Modos de Fourier retenidos en dimensión y.
        activation:   Activación en bloques FNO.

    Example::

        fno = CableFNO2d(in_channels=3, d_model=32, n_layers=4, modes1=12, modes2=12)
        x = torch.randn(4, 3, 64, 64)   # batch=4, 3 canales, malla 64×64
        T = fno(x)                        # (4, 1, 64, 64)
    """

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 32,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        # P: lifting
        self.lifting = nn.Conv2d(in_channels, d_model, kernel_size=1)

        # L bloques FNO
        self.blocks = nn.ModuleList([
            FNOBlock(d_model, modes1, modes2, activation)
            for _ in range(n_layers)
        ])

        # Q: projection (no lineal)
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU() if activation == "gelu" else nn.Tanh(),
            nn.Conv2d(d_model, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.is_floating_point():
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del FNO.

        Args:
            x: ``(batch, C_in, N_x, N_y)`` — canales de entrada en la malla.

        Returns:
            ``(batch, 1, N_x, N_y)`` — campo de temperatura predicho [K].
        """
        v = self.lifting(x)       # (batch, d_model, N_x, N_y)
        for block in self.blocks:
            v = block(v)           # (batch, d_model, N_x, N_y)
        return self.proj(v)        # (batch, 1, N_x, N_y)

    @property
    def n_params(self) -> int:
        """Número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters())
