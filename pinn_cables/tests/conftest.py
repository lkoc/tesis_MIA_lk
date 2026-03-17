"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest
import torch

from pinn_cables.io.readers import (
    CableLayer,
    CablePlacement,
    Domain2D,
    SoilProperties,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def cable_layers() -> list[CableLayer]:
    return [
        CableLayer("conductor", 0.0, 0.0125, 400.0, 3.45e6, 30.0),
        CableLayer("xlpe", 0.0125, 0.0225, 0.286, 2.4e6, 0.0),
        CableLayer("screen", 0.0225, 0.024, 24.0, 3.45e6, 0.0),
        CableLayer("cover", 0.024, 0.028, 0.2, 1.5e6, 0.0),
    ]


@pytest.fixture
def domain() -> Domain2D:
    return Domain2D(xmin=-1.0, xmax=1.0, ymin=-2.0, ymax=0.0)


@pytest.fixture
def placement() -> CablePlacement:
    return CablePlacement(cable_id=1, cx=0.0, cy=-1.0)


@pytest.fixture
def soil() -> SoilProperties:
    return SoilProperties(k=1.0, rho_c=2.0e6, variable=False, amp=0.3)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with all required CSV files."""
    d = tmp_path / "data"
    d.mkdir()

    _write_csv(d / "cable_layers.csv", [
        ["layer_name", "r_inner", "r_outer", "k", "rho_c", "Q"],
        ["conductor", "0.0", "0.0125", "400.0", "3.45e6", "30.0"],
        ["xlpe", "0.0125", "0.0225", "0.286", "2.4e6", "0.0"],
    ])
    _write_csv(d / "domain.csv", [
        ["param", "value"],
        ["xmin", "-1.0"], ["xmax", "1.0"], ["ymin", "-2.0"], ["ymax", "0.0"],
    ])
    _write_csv(d / "cables_placement.csv", [
        ["cable_id", "cx", "cy"],
        ["1", "0.0", "-1.0"],
    ])
    _write_csv(d / "boundary_conditions.csv", [
        ["boundary", "type", "value", "h"],
        ["top", "dirichlet", "293.15", "0.0"],
        ["bottom", "dirichlet", "293.15", "0.0"],
        ["left", "neumann", "0.0", "0.0"],
        ["right", "neumann", "0.0", "0.0"],
    ])
    _write_csv(d / "soil_properties.csv", [
        ["param", "value"],
        ["k", "1.0"], ["rho_c", "2.0e6"], ["variable", "false"], ["amp", "0.3"],
    ])
    _write_csv(d / "scenarios.csv", [
        ["scenario_id", "mode", "Q_scale", "k_soil", "T_amb", "t_end"],
        ["test_steady", "steady", "1.0", "1.0", "293.15", "0"],
        ["test_transient", "transient", "1.0", "1.0", "293.15", "3600"],
    ])
    return d


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
