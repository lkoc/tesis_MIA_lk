"""Tests for pinn_cables.io.readers — CSV loading and validation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pinn_cables.io.readers import (
    BoundaryCondition,
    CableLayer,
    Domain2D,
    load_boundary_conditions,
    load_cable_layers,
    load_domain,
    load_placements,
    load_problem,
    load_scenarios,
    load_soil_properties,
)


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


class TestLoadPlacements:
    def test_single_placement(self, tmp_data_dir: Path) -> None:
        pl = load_placements(tmp_data_dir / "cables_placement.csv")
        assert len(pl) == 1
        assert pl[0].cx == 0.0
        assert pl[0].cy == -1.0


class TestLoadBoundaryConditions:
    def test_four_edges(self, tmp_data_dir: Path) -> None:
        bcs = load_boundary_conditions(tmp_data_dir / "boundary_conditions.csv")
        assert set(bcs.keys()) == {"top", "bottom", "left", "right"}
        assert bcs["top"].bc_type == "dirichlet"
        assert bcs["left"].bc_type == "neumann"


class TestLoadSoilProperties:
    def test_values(self, tmp_data_dir: Path) -> None:
        sp = load_soil_properties(tmp_data_dir / "soil_properties.csv")
        assert sp.k == 1.0
        assert sp.variable is False
        assert sp.amp == 0.3


class TestLoadScenarios:
    def test_count(self, tmp_data_dir: Path) -> None:
        sc = load_scenarios(tmp_data_dir / "scenarios.csv")
        assert len(sc) == 2
        assert sc[0].mode == "steady"
        assert sc[1].mode == "transient"


class TestLoadProblem:
    def test_assembles_all(self, tmp_data_dir: Path) -> None:
        prob = load_problem(tmp_data_dir)
        assert len(prob.layers) == 2
        assert prob.domain.xmin == -1.0
        assert len(prob.placements) == 1
        assert "top" in prob.bcs
        assert prob.soil.k == 1.0
        assert len(prob.scenarios) == 2


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
