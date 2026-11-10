"""Importance sampling keeps uniform coverage while refreshing high-error points."""
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_v3.problem.build import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.networks import build_model
from pinn_v3.solvers.pinn.objective import Objective
from pinn_v3.solvers.pinn.checkpoint import objective_state, restore_objective
from pinn_v3.solvers.pinn.trainer import fit


PROBLEM = Path(__file__).parents[1] / 'problems' / 'two_layer_slab' / 'problem.json'
CABLE_PROBLEM = Path(__file__).parents[1] / 'problems' / 'cable_disk' / 'problem.json'


def make_objective(criterion):
    problem = GenericProblem(load_problem(PROBLEM).doc)
    model = build_model(problem, 'M0', 12, 2, 'tanh').double()
    return problem, Objective(model, n_interior=12, n_layer=8, n_face=8, n_boundary=8,
        seed=11, focused=True, focus_criterion=criterion)


@pytest.mark.parametrize('criterion', ('residual', 'gradient', 'mixed'))
def test_focused_sampling_preserves_uniform_regions_and_surfaces(criterion):
    problem, objective = make_objective(criterion)
    initial_counts = {region.id: len(points) for region, points in (
        (region, objective.interior[region.id]) for region in problem.regions)}
    initial_faces = [len(entry[1]) for entry in objective.faces]
    initial_pieces = [len(points) for _, points, _ in objective.pieces]
    uniform_faces = [face.xy.copy() for face in problem.faces(8)]
    uniform_pieces = [piece.xy.copy() for piece in problem.pieces(8)]

    objective.resample(seed=29)

    assert objective.resampling_count == 1
    assert {region.id: len(objective.interior[region.id]) for region in problem.regions} == initial_counts
    assert [len(entry[1]) for entry in objective.faces] == initial_faces
    assert [len(points) for _, points, _ in objective.pieces] == initial_pieces
    for index, (surface, points, *_) in enumerate(objective.faces):
        uniform_count = len(surface.xy) - len(surface.xy) // 2
        expected = uniform_faces[index][:uniform_count]
        np.testing.assert_allclose(surface.xy[-uniform_count:], expected)
    for index, (surface, points, _) in enumerate(objective.pieces):
        uniform_count = len(surface.xy) - len(surface.xy) // 2
        expected = uniform_pieces[index][:uniform_count]
        np.testing.assert_allclose(surface.xy[-uniform_count:], expected)


def test_focused_objective_has_finite_loss_after_surface_remesh():
    _, objective = make_objective('mixed')
    before, _, _ = objective()
    objective.resample(seed=31)
    after, _, _ = objective()
    assert bool(before.isfinite())
    assert bool(after.isfinite())


def test_incremental_points_are_generated_only_when_growing():
    problem = GenericProblem(load_problem(PROBLEM).doc)
    model = build_model(problem, 'M1', 12, 2, 'tanh').double()
    objective = Objective(model, n_interior=20, n_layer=20, n_face=8, n_boundary=8,
        initial_fraction=0.1, seed=11)
    before = {region.id: objective.interior[region.id].detach().clone() for region in problem.regions}
    assert all(len(points) == 2 for points in before.values())
    objective.grow_interior(0.5, stage=1)
    for region in problem.regions:
        points = objective.interior[region.id]
        assert len(points) == 10
        torch.testing.assert_close(points[:2], before[region.id])
        assert np.all(problem.owner_index(points.detach().numpy()) == region.id)
    objective.grow_interior(1., stage=2)
    assert all(len(points) == 20 for points in objective.interior.values())


def test_progressive_point_fraction_is_nested_and_preserves_surface_integrals():
    problem, objective = make_objective('residual')
    full_interior = {key: value.detach().clone() for key, value in objective._full_interior.items()}
    full_faces = [(surface.xy.copy(), points.detach().clone(), normal.detach().clone())
        for surface, points, normal, *_ in objective.faces]
    full_pieces = [(surface.xy.copy(), points.detach().clone(), normal.detach().clone())
        for surface, points, normal in objective.pieces]

    objective.set_point_fraction(0.5)

    for region in problem.regions:
        active = objective.interior[region.id]
        maximum = full_interior[region.id]
        assert len(active) == int(np.ceil(len(maximum) * 0.5))
        focused_count = len(maximum) // 2
        active_focused = min(focused_count, (len(active) + 1) // 2)
        expected = torch.cat((maximum[:active_focused],
            maximum[focused_count:focused_count + len(active) - active_focused]))
        torch.testing.assert_close(active, expected)
        assert len(objective.sources[region.id]) == len(active)
    for full, active in ((full_faces, objective.faces), (full_pieces, objective.pieces)):
        for (full_xy, full_points, full_normal), (surface, points, normal, *_) in zip(full, active):
            np.testing.assert_array_equal(surface.xy, full_xy)
            torch.testing.assert_close(points, full_points)
            torch.testing.assert_close(normal, full_normal)

    state = objective_state(objective)
    _, restored = make_objective('residual')
    restore_objective(restored, state)
    assert restored.point_fraction == pytest.approx(0.5)
    restored.set_point_fraction(1.)
    for region in problem.regions:
        assert len(restored.interior[region.id]) == len(full_interior[region.id])


def test_fit_uses_all_progressive_point_stages_without_reducing_adam_steps():
    from torch import nn

    model = nn.Linear(1, 1)

    class RecordingObjective:
        def __init__(self):
            self.fractions = []

        def set_point_fraction(self, fraction):
            self.fractions.append(fraction)

        def __call__(self):
            total = model.weight.square().sum()
            zero = total.detach() * 0
            parts = {name: zero for name in ('pde', 'constitutive', 'interface_temperature',
                'interface_flux', 'dirichlet', 'neumann', 'robin', 'energy')}
            return total, parts, {}

        @staticmethod
        def update_attention():
            pass

    objective = RecordingObjective()
    history, best = fit(model, objective, adam_steps=100, lbfgs_steps=0, lr=1e-3, log_every=1000,
        point_schedule_stages=10, point_schedule_start=0.5)

    assert len(objective.fractions) == 10
    assert objective.fractions[0] == pytest.approx(0.5)
    assert objective.fractions[-1] == pytest.approx(1.0)
    assert history[-1]['step'] == 99
    assert best['phase_elapsed_s']['adam'] > 0
    assert best['phase_elapsed_s']['lbfgs'] == 0


def test_cycled_lbfgs_aborts_without_spinning_on_non_finite_loss():
    from torch import nn

    model = nn.Linear(1, 1)

    class NonFiniteObjective:
        def __init__(self):
            self.lbfgs_mode = False
            self.lbfgs_calls = 0

        def __call__(self):
            if self.lbfgs_mode:
                self.lbfgs_calls += 1
                total = model.weight.sum() * float('nan')
            else:
                total = model.weight.square().sum()
            zero = model.weight.sum().detach() * 0
            parts = {name: zero for name in ('pde', 'constitutive', 'interface_temperature',
                'interface_flux', 'dirichlet', 'neumann', 'robin', 'energy')}
            return total, parts, {}

        @staticmethod
        def update_attention():
            pass

    class ConvergenceProbe:
        reference = None
        triggered = False

        def check(self, row, _model, force=False):
            self.triggered = row['phase'] == 'adam'
            objective.lbfgs_mode = self.triggered

        @staticmethod
        def relax(_factor, _step):
            pass

        @staticmethod
        def summary():
            return {}

        @staticmethod
        def accepted(_model, _step):
            return False

    objective = NonFiniteObjective()
    monitor = ConvergenceProbe()

    history, best = fit(model, objective, adam_steps=2, lbfgs_steps=20, lr=1e-3, log_every=1,
        accuracy_stop=monitor, cycle_lbfgs_chunk=5,
        cycle_extension=dict(steps=2, times=1, factor=1.5))

    assert objective.lbfgs_calls == 1
    assert not any(row['phase'] == 'lbfgs' for row in history)
    assert 'non-finite' in best['cycle_lbfgs_failure']


def test_exact_energy_keeps_pde_and_energy_only():
    problem = GenericProblem(load_problem(CABLE_PROBLEM).doc)
    model = build_model(problem, 'M1_EXACT_BC', 12, 2, 'tanh').double()
    objective = Objective(model, n_interior=12, n_layer=8, n_face=8, n_boundary=8,
        seed=11, weights={'pde': 1.0, 'energy': 100.0}, exact_energy=True)
    assert objective.weights['pde'] == pytest.approx(1.0)
    assert objective.weights['energy'] == pytest.approx(100.0)
    assert objective.weights['interface_temperature'] == 0
    assert objective.weights['dirichlet'] == 0
    assert 'interface_flux' not in objective.weights

    def test_cable_interface_scale_uses_expected_power_per_length():
        problem = GenericProblem(load_problem(CABLE_PROBLEM).doc)
        face = problem.faces(64)[0]
        expected = max(abs(problem.region_power[face.left]), abs(problem.region_power[face.right])) / (face.weight * len(face.xy))
        assert problem.interface_flux_scale(face) == pytest.approx(expected)

    def test_sourced_region_uses_local_temperature_amplitude():
        problem = GenericProblem(load_problem(CABLE_PROBLEM).doc)
        conductor = next(region for region in problem.regions if problem.source_abs[region.id] > 0)
        assert problem.amplitude(conductor, 'region') == pytest.approx(
            problem.scales(conductor)['local_temperature'])
