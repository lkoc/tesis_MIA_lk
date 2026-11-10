import pytest
import torch

from pinn_v2.solvers.pinn.networks import build_model
from pinn_v2.solvers.pinn.objective import Objective
from pinn_v2.solvers.pinn.problem import load_problem

SMALL = dict(n_interior=48, n_layer=24, n_face=24, n_boundary=24)


def _model(architecture='M1_EXACT'):
    return build_model(load_problem('aras_flat'), architecture, width=8, depth=1).double()


def test_exact_energy_keeps_energy_weight_and_term():
    objective = Objective(_model(), weights=dict(energy=100.), exact_energy=True, **SMALL)
    assert objective.weights['energy'] == 100.
    assert all(value == 0. for name, value in objective.weights.items() if name not in ('pde', 'energy'))
    total, parts, _ = objective()
    assert float(parts['energy'].detach()) > 0.
    torch.testing.assert_close(total, objective.weights['pde'] * parts['pde'] + 100. * parts['energy'])
    total.backward()


def test_exact_without_exact_energy_stays_pde_only():
    objective = Objective(_model(), weights=dict(energy=100.), **SMALL)
    assert objective.weights['energy'] == 0.
    total, parts, _ = objective()
    torch.testing.assert_close(total, parts['pde'])
    assert float(parts['energy']) == 0.


def test_focused_resample_changes_cloud_and_keeps_counts():
    objective = Objective(_model(), focused=True, **SMALL)
    before = {key: value.detach().clone() for key, value in objective.interior.items()}
    objective.resample()
    for key, old in before.items():
        new = objective.interior[key]
        assert new.shape == old.shape
        assert not torch.equal(new.detach(), old)
    assert objective.resampling_count == 1


def test_trainer_rejects_invalid_exact_energy_and_focused_combinations(tmp_path):
    from pinn_v2.solvers.pinn import trainer
    with pytest.raises(ValueError):
        trainer.run('mms_interface', 'M1', tmp_path / 'a', adam_steps=1, exact_energy=True)
    with pytest.raises(ValueError):
        trainer.run('mms_interface', 'M1_EXACT', tmp_path / 'b', adam_steps=1, focused=True)
    with pytest.raises(ValueError):
        trainer.run('mms_interface', 'DeepXDE_EXACT', tmp_path / 'c', adam_steps=1, exact_energy=True)


def test_log_radial_points_stay_in_soil_near_cables():
    import numpy as np
    objective = Objective(_model(), focused=True, **SMALL)
    soil = objective.problem.regions[0]
    points = objective._log_radial(soil, 200, 1)
    assert len(points) > 0 and objective.problem.contains_points(soil, points).all()
    centers = np.asarray(objective.problem.case['cables'], dtype=float)
    distance = np.linalg.norm(points[:, None, :] - centers[None], axis=2).min(axis=1)
    assert distance.max() <= 4.0 + 1e-9


def test_load_problem_infers_kind_from_content():
    from pinn_v2.cases.catalog import cases
    from pinn_v2.solvers.pinn.problem import AnnulusProblem
    legacy = {key: value for key, value in cases()['annulus'].items() if key != 'kind'}
    assert isinstance(load_problem(legacy), AnnulusProblem)
    with pytest.raises(ValueError):
        load_problem(dict(id='x'))


def test_dirichlet_lifting_pins_soil_boundary_to_t_ref():
    from pinn_v2.solvers.pinn.networks import install_dirichlet_lifting
    problem = load_problem('aras_flat')
    model = install_dirichlet_lifting(build_model(problem, 'M1_EXACT', width=8, depth=1).double(), problem)
    x0, x1, y0, y1 = problem.bounds
    xy = torch.tensor([[x0, 0.5 * (y0 + y1)], [x1, y0], [0.5 * (x0 + x1), y1], [0.5 * (x0 + x1), y0]], dtype=torch.float64)
    temperature, _ = model.field(problem.regions[0], xy)
    torch.testing.assert_close(temperature, torch.full_like(temperature, problem.T_ref))