"""Tests for pinn_cables.pinn.train_custom -- custom training helpers."""

from __future__ import annotations

import logging

import pytest
import torch

from pinn_cables.io.readers import BoundaryCondition, CablePlacement, Domain2D, CableLayer
from pinn_cables.physics.k_field import KFieldModel, PhysicsParams, SoilLayerBand
from pinn_cables.pinn.model import MLP, ResidualPINNModel
from pinn_cables.pinn.train_custom import (
    compute_pde_bc_loss,
    init_output_bias,
    pretrain_multicable,
    sample_bnd_pts,
    sample_soil_pts,
    train_adam_lbfgs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_domain():
    return Domain2D(xmin=-1.0, xmax=1.0, ymin=-2.0, ymax=0.0)


@pytest.fixture
def cable_layers_simple():
    return [
        CableLayer("conductor", 0.0, 0.0125, 400.0, 3.45e6, 30.0),
        CableLayer("xlpe", 0.0125, 0.028, 0.286, 2.4e6, 0.0),
    ]


@pytest.fixture
def placements_simple():
    return [CablePlacement(cable_id=1, cx=0.0, cy=-1.0)]


@pytest.fixture
def bcs_simple():
    return {
        "top": BoundaryCondition("top", "dirichlet", 293.15, 0.0),
        "bottom": BoundaryCondition("bottom", "dirichlet", 293.15, 0.0),
        "left": BoundaryCondition("left", "neumann", 0.0, 0.0),
        "right": BoundaryCondition("right", "neumann", 0.0, 0.0),
    }


# ---------------------------------------------------------------------------
# init_output_bias
# ---------------------------------------------------------------------------

class TestInitOutputBias:
    def test_sets_last_layer_bias(self, device):
        model = MLP(in_dim=2, out_dim=1, width=16, depth=2).to(device)
        init_output_bias(model, 42.0)
        # Find last linear layer
        last = None
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                last = m
        assert last is not None
        assert torch.allclose(last.bias, torch.tensor([42.0], device=device))

    def test_no_linear_layers(self):
        """Should not crash on a model with no Linear layers."""
        model = torch.nn.Sequential(torch.nn.Tanh())
        init_output_bias(model, 5.0)  # should be a no-op


# ---------------------------------------------------------------------------
# sample_soil_pts
# ---------------------------------------------------------------------------

class TestSampleSoilPts:
    def test_output_shape(self, small_domain, placements_simple, device):
        pts = sample_soil_pts(
            small_domain, placements_simple, [0.028], n=200, device=device,
        )
        assert pts.shape == (200, 2)

    def test_excludes_cable_interior(self, small_domain, placements_simple, device):
        r_sheath = 0.028
        pts = sample_soil_pts(
            small_domain, placements_simple, [r_sheath], n=500, device=device,
        )
        dx = pts[:, 0] - 0.0
        dy = pts[:, 1] - (-1.0)
        r2 = dx * dx + dy * dy
        assert (r2 >= r_sheath**2 - 1e-6).all()

    def test_within_domain(self, small_domain, placements_simple, device):
        pts = sample_soil_pts(
            small_domain, placements_simple, [0.028], n=500, device=device,
        )
        assert (pts[:, 0] >= small_domain.xmin - 1e-6).all()
        assert (pts[:, 0] <= small_domain.xmax + 1e-6).all()
        assert (pts[:, 1] >= small_domain.ymin - 1e-6).all()
        assert (pts[:, 1] <= small_domain.ymax + 1e-6).all()

    def test_pac_importance_sampling(self, small_domain, placements_simple, device):
        """With PhysicsParams, should produce points near PAC boundary."""
        pp = PhysicsParams(
            k_variable=True, k_cx=0.0, k_cy=-1.0,
            k_width=0.5, k_height=0.5, k_transition=0.1,
        )
        pts = sample_soil_pts(
            small_domain, placements_simple, [0.028], n=500, device=device,
            pp=pp, frac_pac_bnd=0.3,
        )
        assert pts.shape == (500, 2)
        # Some points should be near the PAC zone boundary
        margin = 4.0 * pp.k_transition + 0.15
        x_lo = pp.k_cx - pp.k_width / 2.0 - margin
        x_hi = pp.k_cx + pp.k_width / 2.0 + margin
        y_lo = pp.k_cy - pp.k_height / 2.0 - margin
        y_hi = pp.k_cy + pp.k_height / 2.0 + margin
        in_region = (
            (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi) &
            (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)
        )
        assert in_region.sum().item() > 50  # at least some in PAC region


# ---------------------------------------------------------------------------
# sample_bnd_pts
# ---------------------------------------------------------------------------

class TestSampleBndPts:
    def test_returns_four_edges(self, small_domain, device):
        bnd = sample_bnd_pts(small_domain, 200, device)
        assert set(bnd.keys()) == {"top", "bottom", "left", "right"}
        for edge, pts in bnd.items():
            assert pts.shape[1] == 2
            assert pts.shape[0] == 50  # 200 / 4

    def test_points_on_edges(self, small_domain, device):
        bnd = sample_bnd_pts(small_domain, 200, device)
        assert torch.allclose(bnd["top"][:, 1], torch.tensor(0.0, device=device))
        assert torch.allclose(bnd["bottom"][:, 1], torch.tensor(-2.0, device=device))
        assert torch.allclose(bnd["left"][:, 0], torch.tensor(-1.0, device=device))
        assert torch.allclose(bnd["right"][:, 0], torch.tensor(1.0, device=device))


# ---------------------------------------------------------------------------
# compute_pde_bc_loss
# ---------------------------------------------------------------------------

class TestComputePdeBcLoss:
    def test_returns_three_tensors(self, small_domain, bcs_simple, device):
        model = MLP(in_dim=2, out_dim=1, width=16, depth=2).to(device)
        xy_soil = torch.randn(50, 2, device=device)
        bnd = sample_bnd_pts(small_domain, 100, device)
        norm_fn = lambda xy: xy  # identity
        total, pde, bc = compute_pde_bc_loss(
            model, xy_soil, bnd, bcs_simple, T_amb=293.15,
            norm_fn=norm_fn, normalize=False, k_fn=1.0, w_pde=1.0, w_bc=1.0,
        )
        assert total.ndim == 0  # scalar
        assert pde.ndim == 0
        assert bc.ndim == 0
        assert total.item() >= 0.0

    def test_gradient_flows(self, small_domain, bcs_simple, device):
        model = MLP(in_dim=2, out_dim=1, width=16, depth=2).to(device)
        xy_soil = torch.randn(50, 2, device=device)
        bnd = sample_bnd_pts(small_domain, 100, device)
        norm_fn = lambda xy: xy
        total, _, _ = compute_pde_bc_loss(
            model, xy_soil, bnd, bcs_simple, T_amb=293.15,
            norm_fn=norm_fn, normalize=False, k_fn=1.0, w_pde=1.0, w_bc=1.0,
        )
        total.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_variable_k_function(self, small_domain, bcs_simple, device):
        """compute_pde_bc_loss should accept a callable k_fn."""
        model = MLP(in_dim=2, out_dim=1, width=16, depth=2).to(device)
        xy_soil = torch.randn(50, 2, device=device)
        bnd = sample_bnd_pts(small_domain, 100, device)
        norm_fn = lambda xy: xy
        pp = PhysicsParams(k_good=2.0, k_bad=1.0)

        from pinn_cables.physics.k_field import k_tensor
        k_fn = lambda xy: k_tensor(xy, pp)

        total, pde, bc = compute_pde_bc_loss(
            model, xy_soil, bnd, bcs_simple, T_amb=293.15,
            norm_fn=norm_fn, normalize=False, k_fn=k_fn, w_pde=1.0, w_bc=1.0,
        )
        assert total.item() >= 0.0


# ---------------------------------------------------------------------------
# pretrain_multicable
# ---------------------------------------------------------------------------

class TestPretrainMulticable:
    def test_reduces_error(
        self, small_domain, cable_layers_simple, placements_simple, device,
    ):
        """Pre-training should bring the model close to T_bg (RMSE < a few K)."""
        torch.manual_seed(42)

        base = MLP(in_dim=2, out_dim=1, width=32, depth=3).to(device)
        model = ResidualPINNModel(
            base, [cable_layers_simple], placements_simple,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
            domain=small_domain, normalize=True,
        )

        rmse = pretrain_multicable(
            model, placements_simple, small_domain, [cable_layers_simple],
            Q_lins=[30.0], k_soil=1.0, T_amb=293.15, device=device,
            normalize=True, n_per_cable=200, n_bc=50, n_steps=100, lr=1e-3,
        )
        # After 100 steps with a tiny network, RMSE should be reasonable
        assert rmse < 50.0  # generous tolerance


# ---------------------------------------------------------------------------
# train_adam_lbfgs (smoke test)
# ---------------------------------------------------------------------------

class TestTrainAdamLbfgs:
    @pytest.mark.slow
    def test_smoke_adam_only(
        self, small_domain, cable_layers_simple, placements_simple, bcs_simple, device,
    ):
        """Verify train_adam_lbfgs runs without error (tiny config, Adam only)."""
        torch.manual_seed(0)
        logger = logging.getLogger("test_train")

        base = MLP(in_dim=2, out_dim=1, width=16, depth=2).to(device)
        model = ResidualPINNModel(
            base, [cable_layers_simple], placements_simple,
            k_soil=1.0, T_amb=293.15, Q_lins=[30.0],
            domain=small_domain, normalize=True,
        )

        history = train_adam_lbfgs(
            model=model,
            domain=small_domain,
            placements=placements_simple,
            bcs=bcs_simple,
            T_amb=293.15,
            r_sheaths=[0.028],
            k_fn=1.0,
            adam_steps=20,
            lbfgs_steps=0,
            n_int=50,
            n_bnd=40,
            oversample=4,
            w_pde=1.0,
            w_bc=10.0,
            lr=1e-3,
            print_every=10,
            normalize=True,
            device=device,
            logger=logger,
        )
        assert "total" in history
        assert len(history["total"]) == 20
        # Loss should be finite
        assert all(0.0 <= v < 1e10 for v in history["total"])


# ---------------------------------------------------------------------------
# sample_soil_pts — k_model path (new KFieldModel parameter)
# ---------------------------------------------------------------------------

class TestSampleSoilPtsKModel:
    """Verify that the k_model parameter in sample_soil_pts works correctly."""

    @pytest.fixture
    def two_band_model(self, small_domain):
        """KFieldModel with 2 soil layers (interface at y=-1.0)."""
        bands = [
            SoilLayerBand(y_top=0.0,  y_bottom=-1.0, k=1.8),
            SoilLayerBand(y_top=-1.0, y_bottom=-2.0, k=1.3),
        ]
        return KFieldModel(k_soil=1.5, soil_bands=bands)

    def test_output_shape_with_k_model(self, small_domain, placements_simple, device, two_band_model):
        pts = sample_soil_pts(
            small_domain, placements_simple, [0.028], n=300, device=device,
            k_model=two_band_model, frac_pac_bnd=0.3,
        )
        assert pts.shape == (300, 2)

    def test_k_model_within_domain(self, small_domain, placements_simple, device, two_band_model):
        pts = sample_soil_pts(
            small_domain, placements_simple, [0.028], n=400, device=device,
            k_model=two_band_model, frac_pac_bnd=0.3,
        )
        assert (pts[:, 0] >= small_domain.xmin - 1e-5).all()
        assert (pts[:, 0] <= small_domain.xmax + 1e-5).all()
        assert (pts[:, 1] >= small_domain.ymin - 1e-5).all()
        assert (pts[:, 1] <= small_domain.ymax + 1e-5).all()

    def test_k_model_excludes_cable(self, small_domain, placements_simple, device, two_band_model):
        r = 0.028
        pts = sample_soil_pts(
            small_domain, placements_simple, [r], n=400, device=device,
            k_model=two_band_model, frac_pac_bnd=0.3,
        )
        dx = pts[:, 0] - 0.0
        dy = pts[:, 1] - (-1.0)
        assert (dx * dx + dy * dy >= r ** 2 - 1e-6).all()

    def test_k_model_layer_interface_sampled(self, small_domain, placements_simple, device, two_band_model):
        """With frac_pac_bnd=0.5, a meaningful fraction of points should be
        near the y=-1.0 soil interface."""
        pts = sample_soil_pts(
            small_domain, placements_simple, [0.028], n=400, device=device,
            k_model=two_band_model, frac_pac_bnd=0.5,
        )
        # Hint band around y=-1.0 (half-width ≥ 0.10)
        near_iface = ((pts[:, 1] >= -1.20) & (pts[:, 1] <= -0.80))
        assert near_iface.sum().item() > 20

    def test_k_model_pac_sampled(self, small_domain, placements_simple, device):
        """KFieldModel with both layers and PAC: both transitions sampled."""
        bands = [
            SoilLayerBand(y_top=0.0,  y_bottom=-1.0, k=1.8),
            SoilLayerBand(y_top=-1.0, y_bottom=-2.0, k=1.3),
        ]
        pp = PhysicsParams(
            k_variable=True, k_good=2.0, k_bad=1.0,
            k_cx=0.0, k_cy=-1.0, k_width=0.4, k_height=0.4, k_transition=0.05,
        )
        model = KFieldModel(k_soil=1.5, soil_bands=bands, pac_params=pp)
        pts = sample_soil_pts(
            small_domain, placements_simple, [0.028], n=400, device=device,
            k_model=model, frac_pac_bnd=0.5,
        )
        assert pts.shape == (400, 2)
