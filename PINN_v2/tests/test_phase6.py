import json

import numpy as np
import pytest

from pinn_v2.experiments import fem_ladder, pinn_vs_fem
from pinn_v2.solvers.pinn import trainer
from pinn_v2.solvers.pinn.problem import load_problem


def record(tmax):
    return dict(Tmax_C=tmax, ndofs=1, regions=[dict(T_mean_C=tmax - 0.1)])


def write_fem(folder, xy, labels, weights, values, converged):
    case = folder.joinpath('case_a', 'base')
    case.mkdir(parents=True)
    np.savez(case.joinpath('fem_l1.npz'), xy=xy, region=labels, weights=weights, T=values)
    folder.joinpath('summary.json').write_text(json.dumps(dict(cases=dict(case_a=dict(converged=converged, records=[dict(level=1)])))), encoding='utf-8')


def test_widened_bounds_double_width_and_depth_only():
    assert fem_ladder.widened([-1., 1., -2., 0.]) == [-2., 2., -4., 0.]


def test_convergence_flags_a_stalled_ladder_as_converged_and_a_drifting_one_not():
    good = fem_ladder.convergence([record(30.), record(30.4), record(30.41)], 20.)
    assert good['converged'] and good['last_change_pct_of_rise'] < fem_ladder.CONVERGENCE_PCT
    drifting = fem_ladder.convergence([record(30.), record(30.4), record(31.)], 20.)
    assert not drifting['converged']


def test_ladder_cases_are_known_and_kim_is_a_definition():
    assert fem_ladder.is_definition('kim_v5_summer')
    assert not fem_ladder.is_definition('xlpe_single')
    assert 3 not in fem_ladder.CASE_LEVELS['kim_v5_summer']


def test_fem_reference_refuses_mismatched_points(tmp_path):
    problem = load_problem('coaxial_full')
    xy, labels, weights = problem.evaluation()
    write_fem(tmp_path, xy + 1e-3, labels, weights, np.zeros(len(xy)), True)
    with pytest.raises(ValueError):
        pinn_vs_fem.fem_reference(tmp_path, 'case_a', problem)


def test_fem_reference_returns_temperature_for_matching_points(tmp_path):
    problem = load_problem('coaxial_full')
    xy, labels, weights = problem.evaluation()
    write_fem(tmp_path, xy, labels, weights, np.full(len(xy), 7.), False)
    values, converged, level = pinn_vs_fem.fem_reference(tmp_path, 'case_a', problem)
    assert level == 1 and not converged and np.allclose(values, 7.)


def test_reference_label_is_recorded_and_the_reference_stays_out_of_the_loss(tmp_path):
    problem = load_problem('mms_constant')
    xy, _, _ = problem.evaluation()
    result = trainer.run('mms_constant', 'M1', tmp_path.joinpath('run'), adam_steps=5, reference=problem.reference(xy), reference_label='external field, evaluation only')
    assert result['status'] == 'completed'
    assert result['evaluation']['reference_label'] == 'external field, evaluation only'
    assert result['reference_used_in_loss'] is False and result['fem_labels_used'] == 0
