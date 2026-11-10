"""Tests for pinn_cables.io.readers -- CSV loading and validation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pinn_cables.io.readers import (
    BoundaryCondition,
    CableLayer,
    Domain2D,
    Scenario,
    SolverParams,
    load_boundary_conditions,
    load_cable_layers,
    load_domain,
    load_placements,
    load_problem,
    load_scenarios,
    load_soil_properties,
    load_solver_params,
)


# -- helpers ---------------------------------------------------------------

def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


# -- load_cable_layers -----------------------------------------------------

class TestLoadCableLayers:
    def test_loads_correct_count(self, tmp_data_dir: Path) -> None:
        layers = load_cable_layers(tmp_data_dir / "cable_layers.csv")
        assert len(layers) == 2

    def test_layer_values(self, tmp_data_dir: Path) -> None:
        layers = load_cable_layers(tmp_data_dir / "cable_layers.csv")
        assert layers[0].name == "conductor"
        assert layers[0].r_inner == 0.0
        assert layers[0].r_outer == 0.0125
        assert layers[0].k == 400.0

    def test_invalid_radii_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        _write_csv(p, [
            ["layer_name", "r_inner", "r_outer", "k", "rho_c", "Q"],
            ["bad", "0.03", "0.01", "1.0", "1e6", "0.0"],
        ])
        with pytest.raises(ValueError, match="r_outer"):
            load_cable_layers(p)

    def test_invalid_k_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad_k.csv"
        _write_csv(p, [
            ["layer_name", "r_inner", "r_outer", "k", "rho_c", "Q"],
            ["bad", "0.0", "0.01", "-1.0", "1e6", "0.0"],
        ])
        with pytest.raises(ValueError, match="k must be > 0"):
            load_cable_layers(p)

    def test_invalid_rho_c_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad_rho.csv"
        _write_csv(p, [
            ["layer_name", "r_inner", "r_outer", "k", "rho_c", "Q"],
            ["bad", "0.0", "0.01", "1.0", "-1e6", "0.0"],
        ])
        with pytest.raises(ValueError, match="rho_c must be > 0"):
            load_cable_layers(p)


# -- load_domain -----------------------------------------------------------

class TestLoadDomain:
    def test_loads_values(self, tmp_data_dir: Path) -> None:
        dom = load_domain(tmp_data_dir / "domain.csv")
        assert dom.xmin == -1.0
        assert dom.ymax == 0.0

    def test_missing_param_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "dom.csv"
        _write_csv(p, [["param", "value"], ["xmin", "0"]])
        with pytest.raises(ValueError, match="Missing"):
            load_domain(p)

    def test_xmax_le_xmin_raises(self) -> None:
        with pytest.raises(ValueError, match="xmax must be > xmin"):
            Domain2D(xmin=1.0, xmax=0.0, ymin=0.0, ymax=1.0)

    def test_ymax_le_ymin_raises(self) -> None:
        with pytest.raises(ValueError, match="ymax must be > ymin"):
            Domain2D(xmin=0.0, xmax=1.0, ymin=1.0, ymax=0.0)


# -- load_placements ------------------------------------------------------

class TestLoadPlacements:
    def test_single_placement(self, tmp_data_dir: Path) -> None:
        pl = load_placements(tmp_data_dir / "cables_placement.csv")
        assert len(pl) == 1
        assert pl[0].cable_id == 1
        assert isinstance(pl[0].cable_id, int)
        assert pl[0].cx == 0.0
        assert pl[0].cy == -1.0


# -- load_boundary_conditions ---------------------------------------------

class TestLoadBoundaryConditions:
    def test_four_edges(self, tmp_data_dir: Path) -> None:
        bcs = load_boundary_conditions(tmp_data_dir / "boundary_conditions.csv")
        assert set(bcs.keys()) == {"top", "bottom", "left", "right"}
        assert bcs["top"].bc_type == "dirichlet"
        assert bcs["left"].bc_type == "neumann"

    def test_invalid_bc_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown BC type"):
            BoundaryCondition("top", "invalid_type", 0.0, 0.0)


# -- load_soil_properties -------------------------------------------------

class TestLoadSoilProperties:
    def test_rho_c(self, tmp_data_dir: Path) -> None:
        sp = load_soil_properties(tmp_data_dir / "soil_properties.csv")
        assert sp.rho_c == 2.0e6

    def test_defaults_when_fields_absent(self, tmp_data_dir: Path) -> None:
        # CSV only has rho_c; k, variable, amp should use defaults
        sp = load_soil_properties(tmp_data_dir / "soil_properties.csv")
        assert sp.k == 1.0
        assert sp.variable is False
        assert sp.amp == 0.0


# -- load_scenarios --------------------------------------------------------

class TestLoadScenarios:
    def test_count(self, tmp_data_dir: Path) -> None:
        sc = load_scenarios(tmp_data_dir / "scenarios.csv")
        assert len(sc) == 2
        assert sc[0].mode == "steady"
        assert sc[1].mode == "transient"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            Scenario("bad", "invalid_mode", 1.0, 1.0, 293.15, 0.0)


# -- load_problem ----------------------------------------------------------

class TestLoadProblem:
    def test_assembles_all(self, tmp_data_dir: Path) -> None:
        prob = load_problem(tmp_data_dir)
        assert len(prob.layers) == 2
        assert prob.domain.xmin == -1.0
        assert len(prob.placements) == 1
        assert "top" in prob.bcs
        assert prob.soil.k == 1.0
        assert len(prob.scenarios) == 2

    def test_solver_params_loaded_when_present(self, tmp_data_dir: Path) -> None:
        prob = load_problem(tmp_data_dir)
        assert prob.solver_params is not None
        assert prob.solver_params.model_width == 64
        assert prob.solver_params.adam_steps == 100

    def test_solver_params_none_when_missing(self, tmp_data_dir: Path) -> None:
        (tmp_data_dir / "solver_params.csv").unlink()
        prob = load_problem(tmp_data_dir)
        assert prob.solver_params is None


# -- SolverParams ----------------------------------------------------------

class TestSolverParams:
    def test_defaults(self) -> None:
        sp = SolverParams()
        assert sp.model_width == 128
        assert sp.model_depth == 6
        assert sp.adam_steps == 20_000
        assert sp.lbfgs_steps == 5_000
        assert sp.normalize_coords is True
        assert sp.w_pde == 1.0
        assert sp.device == "auto"

    def test_to_solver_cfg_structure(self) -> None:
        sp = SolverParams()
        cfg = sp.to_solver_cfg()
        assert "model" in cfg
        assert "training" in cfg
        assert "sampling" in cfg
        assert "normalization" in cfg
        assert "loss_weights" in cfg
        assert "time" in cfg
        assert "logging" in cfg

    def test_to_solver_cfg_model_values(self) -> None:
        sp = SolverParams(model_width=64, model_depth=3, model_activation="relu")
        cfg = sp.to_solver_cfg()
        assert cfg["model"]["width"] == 64
        assert cfg["model"]["depth"] == 3
        assert cfg["model"]["activation"] == "relu"

    def test_to_solver_cfg_training_values(self) -> None:
        sp = SolverParams(lr=5e-4, adam_steps=1000, lbfgs_steps=0)
        cfg = sp.to_solver_cfg()
        assert cfg["training"]["lr"] == 5e-4
        assert cfg["training"]["adam_steps"] == 1000
        assert cfg["training"]["lbfgs_steps"] == 0

    def test_to_solver_cfg_loss_weights(self) -> None:
        sp = SolverParams(w_pde=2.0, w_bc_dirichlet=5.0)
        cfg = sp.to_solver_cfg()
        assert cfg["loss_weights"]["pde"] == 2.0
        assert cfg["loss_weights"]["bc_dirichlet"] == 5.0

    def test_to_solver_cfg_sampling(self) -> None:
        sp = SolverParams(n_interior=500, n_boundary=50)
        cfg = sp.to_solver_cfg()
        assert cfg["sampling"]["n_interior"] == 500
        assert cfg["sampling"]["n_boundary"] == 50


class TestLoadSolverParams:
    def test_loads_from_csv(self, tmp_data_dir: Path) -> None:
        sp = load_solver_params(tmp_data_dir / "solver_params.csv")
        assert sp.model_width == 64
        assert sp.model_depth == 3
        assert sp.adam_steps == 100
        assert sp.lbfgs_steps == 0
        assert sp.normalize_coords is True
        assert sp.model_fourier_features is False
        assert isinstance(sp.lr, float)
        assert isinstance(sp.model_width, int)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_solver_params(tmp_path / "nonexistent.csv")

    def test_unknown_params_ignored(self, tmp_path: Path) -> None:
        p = tmp_path / "sp.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows([
                ["param", "value"],
                ["model_width", "32"],
                ["unknown_future_param", "hello"],
            ])
        sp = load_solver_params(p)
        assert sp.model_width == 32

    def test_partial_csv_uses_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "sp_partial.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows([["param", "value"], ["adam_steps", "500"]])
        sp = load_solver_params(p)
        assert sp.adam_steps == 500
        assert sp.model_width == 128    # default

    def test_bool_parsing_variants(self, tmp_path: Path) -> None:
        for val_str, expected in [("true", True), ("True", True), ("1", True),
                                   ("false", False), ("False", False), ("0", False)]:
            p = tmp_path / f"sp_{val_str}.csv"
            with open(p, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerows([["param", "value"],
                                   ["model_fourier_features", val_str]])
            sp = load_solver_params(p)
            assert sp.model_fourier_features is expected, f"failed for '{val_str}'"

    def test_scientific_notation_int(self, tmp_path: Path) -> None:
        p = tmp_path / "sp_sci.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows([["param", "value"], ["adam_steps", "2.0e3"]])
        sp = load_solver_params(p)
        assert sp.adam_steps == 2000
