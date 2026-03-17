"""Neural-network architectures for the PINN solver.

Provides:
- :class:`MLP` — plain multi-layer perceptron.
- :class:`FourierFeatureMapping` — random Fourier feature layer.
- :class:`FourierFeatureNet` — MLP preceded by Fourier encoding.
- :func:`build_model` — factory that reads the ``model`` section of the
  solver YAML and returns the appropriate network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation look-up
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _get_activation(name: str) -> type[nn.Module]:
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'; "
            f"choose from {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Multi-layer perceptron with configurable width, depth, and activation.

    Weights are initialised with Xavier uniform; biases with zeros.

    Args:
        in_dim:     Input dimensionality (2 for steady, 3 for transient).
        out_dim:    Output dimensionality (1 = temperature).
        width:      Hidden-layer width.
        depth:      Number of hidden layers.
        activation: Name of the activation function (``"tanh"`` etc.).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        width: int = 128,
        depth: int = 6,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        act_cls = _get_activation(activation)

        layers: list[nn.Module] = [nn.Linear(in_dim, width), act_cls()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act_cls()]
        layers.append(nn.Linear(width, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(N, in_dim)``.

        Returns:
            Output tensor of shape ``(N, out_dim)``.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Fourier features
# ---------------------------------------------------------------------------

class FourierFeatureMapping(nn.Module):
    """Random Fourier-feature mapping (Tancik et al., 2020).

    Maps each input ``x`` to ``[cos(2 pi B x), sin(2 pi B x)]`` where
    ``B`` is a fixed random matrix drawn once at initialisation.

    Args:
        in_dim:       Input dimensionality.
        mapping_size: Number of Fourier basis functions (output dim = 2 * mapping_size).
        scale:        Standard deviation of the random frequencies.
    """

    def __init__(
        self,
        in_dim: int,
        mapping_size: int = 64,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return self.B.shape[1] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping.

        Args:
            x: Input ``(N, in_dim)``.

        Returns:
            Encoded ``(N, 2 * mapping_size)``.
        """
        proj = 2.0 * torch.pi * x @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)


class FourierFeatureNet(nn.Module):
    """MLP preceded by a random Fourier-feature encoding.

    Args:
        in_dim:              Physical input dimension.
        out_dim:             Output dimension.
        width:               MLP hidden width.
        depth:               MLP hidden depth.
        activation:          Activation name.
        fourier_scale:       Frequency scale for Fourier features.
        fourier_mapping_size: Number of Fourier basis functions.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        width: int = 128,
        depth: int = 6,
        activation: str = "tanh",
        fourier_scale: float = 1.0,
        fourier_mapping_size: int = 64,
    ) -> None:
        super().__init__()
        self.ff = FourierFeatureMapping(in_dim, fourier_mapping_size, fourier_scale)
        self.mlp = MLP(self.ff.out_dim, out_dim, width, depth, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ff(x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    cfg: dict,
    in_dim: int,
    device: torch.device | None = None,
) -> nn.Module:
    """Build a PINN model from the ``model`` section of the solver config.

    Args:
        cfg:    The ``model`` sub-dictionary from solver YAML.
        in_dim: Input dimensionality (2 = steady, 3 = transient).
        device: Target device.

    Returns:
        Constructed neural network, moved to *device*.
    """
    use_fourier = cfg.get("fourier_features", False)
    width = cfg.get("width", 128)
    depth = cfg.get("depth", 6)
    activation = cfg.get("activation", "tanh")

    if use_fourier:
        model: nn.Module = FourierFeatureNet(
            in_dim=in_dim,
            out_dim=1,
            width=width,
            depth=depth,
            activation=activation,
            fourier_scale=cfg.get("fourier_scale", 1.0),
            fourier_mapping_size=cfg.get("fourier_mapping_size", 64),
        )
    else:
        model = MLP(
            in_dim=in_dim,
            out_dim=1,
            width=width,
            depth=depth,
            activation=activation,
        )

    if device is not None:
        model = model.to(device)
    return model
