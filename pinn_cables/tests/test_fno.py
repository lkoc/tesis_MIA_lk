"""Tests for pinn_cables.fno — SpectralConv2d, FNOBlock, CableFNO2d,
make_input_channels, FNOTrainConfig, train_fno, eval_fno_field, fno_T_max."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from pinn_cables.fno.model import SpectralConv2d, FNOBlock, CableFNO2d
from pinn_cables.fno.dataset import make_input_channels
from pinn_cables.fno.train import FNOTrainConfig, train_fno
from pinn_cables.fno.eval import eval_fno_field, fno_T_max
from pinn_cables.io.readers import CableLayer, CablePlacement, Domain2D


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def small_domain() -> Domain2D:
    return Domain2D(xmin=-5.0, xmax=5.0, ymin=-4.0, ymax=0.0)


@pytest.fixture
def single_placement() -> CablePlacement:
    return CablePlacement(cable_id=1, cx=0.0, cy=-1.5,
                          section_mm2=400, conductor_material="cu", current_A=800.0)


@pytest.fixture
def simple_layers() -> list[CableLayer]:
    return [
        CableLayer("conductor", 0.0, 0.012, 400.0, 3.45e6, 1e5),
        CableLayer("insulation", 0.012, 0.022, 0.286, 2.4e6, 0.0),
        CableLayer("sheath", 0.022, 0.025, 0.2, 1.5e6, 0.0),
    ]


@pytest.fixture
def tiny_fno(device) -> CableFNO2d:
    return CableFNO2d(
        in_channels=3, d_model=8, n_layers=2, modes1=4, modes2=4,
    ).to(device)


# ---------------------------------------------------------------------------
# SpectralConv2d
# ---------------------------------------------------------------------------

class TestSpectralConv2d:
    """Tests for SpectralConv2d — inherits nn.Module."""

    def test_is_nn_module(self, device):
        layer = SpectralConv2d(4, 4, modes1=4, modes2=4).to(device)
        assert isinstance(layer, nn.Module)

    def test_output_shape_preserves_spatial(self, device):
        """Output spatial dims must equal input spatial dims (inverse FFT)."""
        layer = SpectralConv2d(in_channels=4, out_channels=8, modes1=4, modes2=4).to(device)
        x = torch.randn(2, 4, 16, 16, device=device)
        y = layer(x)
        assert y.shape == (2, 8, 16, 16)

    def test_output_shape_non_square(self, device):
        layer = SpectralConv2d(2, 2, modes1=3, modes2=5).to(device)
        x = torch.randn(3, 2, 12, 20, device=device)
        y = layer(x)
        assert y.shape == (3, 2, 12, 20)

    def test_output_is_real(self, device):
        """irfft2 must return real-valued tensor."""
        layer = SpectralConv2d(2, 2, 4, 4).to(device)
        x = torch.randn(1, 2, 8, 8, device=device)
        y = layer(x)
        assert y.is_floating_point()
        assert not y.is_complex()

    def test_learnable_parameters(self, device):
        layer = SpectralConv2d(4, 4, 6, 6).to(device)
        params = list(layer.parameters())
        assert len(params) == 2            # weights1, weights2
        assert params[0].is_complex()      # cfloat parameters
        assert params[0].requires_grad

    def test_parameter_shapes(self, device):
        C_in, C_out, m1, m2 = 4, 8, 5, 6
        layer = SpectralConv2d(C_in, C_out, m1, m2).to(device)
        assert layer.weights1.shape == (C_in, C_out, m1, m2)
        assert layer.weights2.shape == (C_in, C_out, m1, m2)

    def test_gradients_flow(self, device):
        layer = SpectralConv2d(2, 2, 4, 4).to(device)
        x = torch.randn(1, 2, 8, 8, device=device)
        loss = layer(x).sum()
        loss.backward()
        assert layer.weights1.grad is not None
        assert layer.weights2.grad is not None

    def test_modes_not_exceed_half_resolution(self, device):
        """modes must be <= N//2 for rfft2; verify no out-of-bound indexing."""
        layer = SpectralConv2d(1, 1, modes1=8, modes2=8).to(device)
        x = torch.randn(1, 1, 16, 16, device=device)   # N//2 = 8 exactly
        y = layer(x)
        assert y.shape == (1, 1, 16, 16)


# ---------------------------------------------------------------------------
# FNOBlock
# ---------------------------------------------------------------------------

class TestFNOBlock:
    """Tests for FNOBlock residual layer."""

    def test_is_nn_module(self, device):
        block = FNOBlock(channels=8, modes1=4, modes2=4).to(device)
        assert isinstance(block, nn.Module)

    def test_output_shape_equals_input(self, device):
        """FNOBlock is channel-preserving."""
        block = FNOBlock(channels=16, modes1=4, modes2=4).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        assert block(x).shape == x.shape

    def test_activation_gelu(self, device):
        block = FNOBlock(8, 4, 4, activation="gelu").to(device)
        assert isinstance(block.act, nn.GELU)

    def test_activation_tanh(self, device):
        block = FNOBlock(8, 4, 4, activation="tanh").to(device)
        assert isinstance(block.act, nn.Tanh)

    def test_has_spectral_and_skip(self, device):
        block = FNOBlock(8, 4, 4).to(device)
        assert hasattr(block, "spectral")
        assert hasattr(block, "skip")
        assert isinstance(block.spectral, SpectralConv2d)
        assert isinstance(block.skip, nn.Conv2d)

    def test_gradients_flow(self, device):
        block = FNOBlock(8, 4, 4).to(device)
        x = torch.randn(1, 8, 8, 8, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# CableFNO2d
# ---------------------------------------------------------------------------

class TestCableFNO2d:
    """Tests for CableFNO2d — complete FNO."""

    def test_is_nn_module(self, tiny_fno):
        assert isinstance(tiny_fno, nn.Module)

    def test_output_shape(self, device):
        fno = CableFNO2d(in_channels=3, d_model=8, n_layers=2,
                         modes1=4, modes2=4).to(device)
        x = torch.randn(4, 3, 16, 16, device=device)
        y = fno(x)
        assert y.shape == (4, 1, 16, 16)

    def test_output_shape_different_resolution(self, device):
        """FNO should run at any grid resolution (zero-shot)."""
        fno = CableFNO2d(in_channels=3, d_model=8, n_layers=2,
                         modes1=4, modes2=4).to(device)
        for N in (16, 32, 64):
            y = fno(torch.randn(1, 3, N, N, device=device))
            assert y.shape == (1, 1, N, N)

    def test_output_is_real(self, tiny_fno, device):
        x = torch.randn(2, 3, 16, 16, device=device)
        y = tiny_fno(x)
        assert y.is_floating_point() and not y.is_complex()

    def test_n_params_positive(self, tiny_fno):
        assert tiny_fno.n_params > 0

    def test_n_params_scales_with_d_model(self, device):
        small = CableFNO2d(d_model=8, n_layers=2, modes1=4, modes2=4).to(device)
        large = CableFNO2d(d_model=32, n_layers=2, modes1=4, modes2=4).to(device)
        assert large.n_params > small.n_params

    def test_n_layers_count(self, device):
        fno = CableFNO2d(d_model=8, n_layers=3, modes1=4, modes2=4).to(device)
        assert len(fno.blocks) == 3

    def test_weights_initialised(self, tiny_fno):
        """Xavier init: conv weights must not be all zeros."""
        for m in tiny_fno.modules():
            if isinstance(m, nn.Conv2d) and m.weight.is_floating_point():
                assert m.weight.abs().sum().item() > 0

    def test_gradients_end_to_end(self, tiny_fno, device):
        x = torch.randn(2, 3, 8, 8, device=device)
        loss = tiny_fno(x).sum()
        loss.backward()
        for p in tiny_fno.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_state_dict_save_load(self, tiny_fno, device, tmp_path):
        """Save/load via state_dict must preserve predictions."""
        x = torch.randn(1, 3, 8, 8, device=device)
        with torch.no_grad():
            y_before = tiny_fno(x).clone()
        path = tmp_path / "fno.pt"
        torch.save(tiny_fno.state_dict(), path)
        fno2 = CableFNO2d(in_channels=3, d_model=8, n_layers=2,
                           modes1=4, modes2=4).to(device)
        fno2.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        with torch.no_grad():
            y_after = fno2(x)
        assert torch.allclose(y_before, y_after)

    def test_lifting_projection_shapes(self, device):
        fno = CableFNO2d(in_channels=3, d_model=16, n_layers=1,
                         modes1=4, modes2=4).to(device)
        assert fno.lifting.in_channels == 3
        assert fno.lifting.out_channels == 16

    def test_batch_size_one(self, tiny_fno, device):
        x = torch.randn(1, 3, 8, 8, device=device)
        assert tiny_fno(x).shape == (1, 1, 8, 8)


# ---------------------------------------------------------------------------
# make_input_channels
# ---------------------------------------------------------------------------

class TestMakeInputChannels:
    """Tests for the grid input builder."""

    def test_output_shapes(self, device, small_domain, single_placement, simple_layers):
        N_g = 16
        placements = [single_placement]
        layers_list = [simple_layers]
        Q_lins = [50.0]

        input_grid, coord_grid = make_input_channels(
            small_domain, placements, layers_list,
            k_fn=1.0, Q_lins=Q_lins, T_amb=300.0,
            N_g=N_g, device=device,
        )
        assert input_grid.shape == (3, N_g, N_g)
        assert coord_grid.shape == (2, N_g, N_g)

    def test_k_channel_normalised(self, device, small_domain, single_placement, simple_layers):
        """Canal k debe estar en [0, 1]."""
        input_grid, _ = make_input_channels(
            small_domain, [single_placement], [simple_layers],
            k_fn=1.5, Q_lins=[50.0], T_amb=300.0, N_g=8, device=device,
        )
        k_ch = input_grid[0]
        assert k_ch.min() >= 0.0 - 1e-6
        assert k_ch.max() <= 1.0 + 1e-6

    def test_bc_mask_borders(self, device, small_domain, single_placement, simple_layers):
        """Bordes del dominio deben tener bc_mask = 1."""
        input_grid, _ = make_input_channels(
            small_domain, [single_placement], [simple_layers],
            k_fn=1.5, Q_lins=[50.0], T_amb=300.0, N_g=8, device=device,
        )
        bc = input_grid[2]
        assert bc[0, :].min() == 1.0    # borde inferior
        assert bc[-1, :].min() == 1.0   # borde superior
        assert bc[:, 0].min() == 1.0    # borde izquierdo
        assert bc[:, -1].min() == 1.0   # borde derecho

    def test_callable_k_fn(self, device, small_domain, single_placement, simple_layers):
        """k_fn callable debe funcionar igual que escalar homogéneo."""
        def k_fn_homog(xy):
            return torch.full((xy.shape[0], 1), 1.5, device=xy.device)

        grid_callable, _ = make_input_channels(
            small_domain, [single_placement], [simple_layers],
            k_fn=k_fn_homog, Q_lins=[50.0], T_amb=300.0, N_g=8, device=device,
        )
        grid_scalar, _ = make_input_channels(
            small_domain, [single_placement], [simple_layers],
            k_fn=1.5, Q_lins=[50.0], T_amb=300.0, N_g=8, device=device,
        )
        assert torch.allclose(grid_callable[0], grid_scalar[0])

    def test_coord_grid_extent(self, device, small_domain, single_placement, simple_layers):
        """Coordenadas deben cubrir exactamente el dominio."""
        _, coord_grid = make_input_channels(
            small_domain, [single_placement], [simple_layers],
            k_fn=1.0, Q_lins=[50.0], T_amb=300.0, N_g=8, device=device,
        )
        assert abs(float(coord_grid[0].min()) - small_domain.xmin) < 1e-5
        assert abs(float(coord_grid[0].max()) - small_domain.xmax) < 1e-5
        assert abs(float(coord_grid[1].min()) - small_domain.ymin) < 1e-5
        assert abs(float(coord_grid[1].max()) - small_domain.ymax) < 1e-5


# ---------------------------------------------------------------------------
# FNOTrainConfig
# ---------------------------------------------------------------------------

class TestFNOTrainConfig:
    """Tests for the training config dataclass."""

    def test_default_w_pde_is_zero(self):
        """Por defecto debe ser data-driven puro (sin pérdida PDE)."""
        cfg = FNOTrainConfig()
        assert cfg.w_pde == 0.0

    def test_default_w_data_is_one(self):
        cfg = FNOTrainConfig()
        assert cfg.w_data == 1.0

    def test_custom_params(self):
        cfg = FNOTrainConfig(epochs=100, batch_size=8, lr=5e-4, w_pde=0.1)
        assert cfg.epochs == 100
        assert cfg.batch_size == 8
        assert cfg.lr == pytest.approx(5e-4)
        assert cfg.w_pde == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# train_fno (smoke test — pocas épocas)
# ---------------------------------------------------------------------------

class TestTrainFno:
    """Integration smoke tests for train_fno."""

    @pytest.fixture
    def tiny_dataset(self, device, small_domain, single_placement, simple_layers):
        """Dataset mínimo de 6 muestras con tensores aleatorios."""
        from torch.utils.data import TensorDataset
        N_g = 8
        inputs  = torch.randn(6, 3, N_g, N_g, device=device)
        targets = torch.randn(6, 1, N_g, N_g, device=device)
        return TensorDataset(inputs, targets)

    def test_returns_history_keys(self, device, tiny_dataset):
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=2, modes2=2).to(device)
        cfg = FNOTrainConfig(epochs=3, batch_size=2, print_every=999, device="cpu")
        history = train_fno(fno, tiny_dataset, cfg)
        for key in ("train_total", "train_data", "train_pde", "val_total"):
            assert key in history

    def test_history_length_equals_epochs(self, device, tiny_dataset):
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=2, modes2=2).to(device)
        cfg = FNOTrainConfig(epochs=5, batch_size=2, print_every=999, device="cpu")
        history = train_fno(fno, tiny_dataset, cfg)
        assert len(history["train_total"]) == 5

    def test_loss_decreases_trivial(self, device, tiny_dataset):
        """La pérdida total al final debe ser finita y no NaN."""
        torch.manual_seed(0)
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=2, modes2=2).to(device)
        cfg = FNOTrainConfig(epochs=10, batch_size=2, lr=1e-2,
                             print_every=999, device="cpu")
        history = train_fno(fno, tiny_dataset, cfg)
        assert all(not (v != v) for v in history["train_total"])  # no NaN

    def test_model_parameters_change(self, device, tiny_dataset):
        """Los pesos del FNO deben cambiar después del entrenamiento."""
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=2, modes2=2).to(device)
        params_before = {n: p.clone() for n, p in fno.named_parameters()
                         if p.is_floating_point()}
        cfg = FNOTrainConfig(epochs=3, batch_size=2, print_every=999, device="cpu")
        train_fno(fno, tiny_dataset, cfg)
        changed = any(
            not torch.equal(params_before[n], p)
            for n, p in fno.named_parameters()
            if p.is_floating_point() and n in params_before
        )
        assert changed


# ---------------------------------------------------------------------------
# eval_fno_field y fno_T_max
# ---------------------------------------------------------------------------

class TestEvalFno:
    """Tests for evaluation utilities."""

    def test_eval_fno_field_shapes(self, device, small_domain, single_placement, simple_layers):
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=4, modes2=4).to(device)
        N_g = 16
        X, Y, T = eval_fno_field(
            fno, small_domain, [single_placement], [simple_layers],
            k_fn=1.0, Q_lins=[50.0], T_amb=300.0, N_g=N_g, device=device,
        )
        assert X.shape == (N_g, N_g)
        assert Y.shape == (N_g, N_g)
        assert T.shape == (N_g, N_g)

    def test_eval_fno_field_T_includes_T_amb(self, device, small_domain,
                                              single_placement, simple_layers):
        """T_grid debe estar en escala absoluta (K), alrededor de T_amb."""
        T_amb = 283.15  # 10 °C
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=4, modes2=4).to(device)
        # Forzar salida = 0 (red con pesos tiny cerca de cero)
        with torch.no_grad():
            for p in fno.parameters():
                p.zero_()
        _, _, T = eval_fno_field(
            fno, small_domain, [single_placement], [simple_layers],
            k_fn=1.0, Q_lins=[50.0], T_amb=T_amb, N_g=8, device=device,
        )
        # Con pesos=0, delta_T ~ 0, luego T ≈ T_amb
        assert torch.allclose(T, torch.full_like(T, T_amb), atol=1e-4)

    def test_fno_T_max_returns_dict_keys(self, device, small_domain,
                                          single_placement, simple_layers):
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=4, modes2=4).to(device)
        result = fno_T_max(
            fno, small_domain, [single_placement], [simple_layers],
            k_fn=1.0, Q_lins=[50.0], T_amb=300.0, N_g=16, device=device,
        )
        for key in ("T_max_K", "T_max_C", "T_per_cable"):
            assert key in result

    def test_fno_T_max_celsius_offset(self, device, small_domain,
                                       single_placement, simple_layers):
        """T_max_C debe ser T_max_K - 273.15."""
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=4, modes2=4).to(device)
        result = fno_T_max(
            fno, small_domain, [single_placement], [simple_layers],
            k_fn=1.0, Q_lins=[50.0], T_amb=300.0, N_g=16, device=device,
        )
        assert abs(result["T_max_C"] - (result["T_max_K"] - 273.15)) < 1e-6

    def test_fno_T_per_cable_length(self, device, small_domain,
                                     single_placement, simple_layers):
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=4, modes2=4).to(device)
        result = fno_T_max(
            fno, small_domain, [single_placement], [simple_layers],
            k_fn=1.0, Q_lins=[50.0], T_amb=300.0, N_g=16, device=device,
        )
        assert len(result["T_per_cable"]) == 1

    def test_fno_T_max_multi_cable(self, device, small_domain, simple_layers):
        """Con N cables T_per_cable debe tener N entradas."""
        placements = [
            CablePlacement(1, -1.0, -1.5, 400, "cu", 800.0),
            CablePlacement(2,  1.0, -1.5, 400, "cu", 800.0),
        ]
        fno = CableFNO2d(d_model=8, n_layers=1, modes1=4, modes2=4).to(device)
        result = fno_T_max(
            fno, small_domain, placements, [simple_layers, simple_layers],
            k_fn=1.0, Q_lins=[50.0, 50.0], T_amb=300.0, N_g=16, device=device,
        )
        assert len(result["T_per_cable"]) == 2
        assert result["T_max_K"] == max(result["T_per_cable"])
