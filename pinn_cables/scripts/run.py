"""Unified entry-point for running PINN cable simulations.

Usage
-----
::

    python -m pinn_cables.scripts.run \\
        --data  pinn_cables/data \\
        --config pinn_cables/configs/solver.yaml \\
        --scenario base_steady

The ``--scenario`` argument selects a row from ``scenarios.csv``.  The
mode (``steady`` / ``transient``) is read from that row.
"""

from __future__ import annotations

import argparse
import sys

import torch

from pinn_cables.io.readers import Scenario, load_problem
from pinn_cables.pinn.model import build_model
from pinn_cables.pinn.train import SteadyStatePINNTrainer, TransientPINNTrainer
from pinn_cables.pinn.utils import get_device, load_config, set_seed, setup_logging
from pinn_cables.post.eval import evaluate_on_grid, evaluate_on_grid_transient
from pinn_cables.post.plots import (
    plot_cable_geometry,
    plot_loss_history,
    plot_temperature_field,
)


def _find_scenario(scenarios: list[Scenario], name: str) -> Scenario:
    for s in scenarios:
        if s.scenario_id == name:
            return s
    available = [s.scenario_id for s in scenarios]
    raise ValueError(
        f"Scenario '{name}' not found.  Available: {available}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a PINN cable heat-conduction simulation.",
    )
    parser.add_argument(
        "--data", required=True,
        help="Directory containing the problem CSV files.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the solver YAML configuration.",
    )
    parser.add_argument(
        "--scenario", required=True,
        help="Scenario ID (row in scenarios.csv).",
    )
    parser.add_argument(
        "--output", default="results/",
        help="Directory for output plots and checkpoints.",
    )
    args = parser.parse_args(argv)

    # --- Load problem and solver config ---
    problem = load_problem(args.data)
    solver_cfg = load_config(args.config)
    scenario = _find_scenario(problem.scenarios, args.scenario)

    # --- Setup ---
    set_seed(solver_cfg.get("seed", 42))
    device = get_device(solver_cfg.get("device", "auto"))
    logger = setup_logging(solver_cfg.get("logging", {}).get("log_dir", "runs/"))
    logger.info("Scenario: %s  mode=%s  device=%s", scenario.scenario_id, scenario.mode, device)

    in_dim = 2 if scenario.mode == "steady" else 3
    model = build_model(solver_cfg.get("model", {}), in_dim=in_dim, device=device)
    logger.info("Model: %s", model.__class__.__name__)

    # Use first placement (multi-cable support could iterate here)
    placement = problem.placements[0]

    # --- Train ---
    TrainerClass = (
        SteadyStatePINNTrainer if scenario.mode == "steady"
        else TransientPINNTrainer
    )
    trainer = TrainerClass(
        model=model,
        layers=problem.layers,
        placement=placement,
        domain=problem.domain,
        soil=problem.soil,
        bcs=problem.bcs,
        scenario=scenario,
        solver_cfg=solver_cfg,
        device=device,
        logger=logger,
    )
    history = trainer.train()

    # --- Post-process ---
    out = args.output
    plot_loss_history(history, save_path=f"{out}/{scenario.scenario_id}_loss.png")
    plot_cable_geometry(
        problem.layers, placement, problem.domain,
        save_path=f"{out}/{scenario.scenario_id}_geometry.png",
    )

    if scenario.mode == "steady":
        X, Y, T = evaluate_on_grid(
            model, problem.domain, device=device,
            normalize=solver_cfg.get("normalization", {}).get("normalize_coords", True),
        )
        plot_temperature_field(
            X, Y, T,
            title=f"T field — {scenario.scenario_id}",
            save_path=f"{out}/{scenario.scenario_id}_T.png",
        )
    else:
        for t_frac in [0.0, 0.25, 0.5, 1.0]:
            t_val = t_frac * scenario.t_end
            X, Y, T = evaluate_on_grid_transient(
                model, problem.domain, t_val, scenario.t_end,
                device=device,
                normalize=solver_cfg.get("normalization", {}).get("normalize_coords", True),
            )
            plot_temperature_field(
                X, Y, T,
                title=f"T at t={t_val:.0f}s — {scenario.scenario_id}",
                save_path=f"{out}/{scenario.scenario_id}_T_t{t_val:.0f}.png",
            )

    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
