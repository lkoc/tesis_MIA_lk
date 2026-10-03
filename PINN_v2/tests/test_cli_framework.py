"""Tests of the user-facing framework: same case file for every tool, CLI, run settings and search."""
import json

import numpy as np
import pytest

from pinn_v2.cases.catalog import cases
from pinn_v2.cases.definition import CaseDefinition
from pinn_v2.cli import main, read_pairs
from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.experiments.hparam_search import candidates, run_search, trial_kwargs
from pinn_v2.solvers.pinn.problem import load_problem


def test_case_file_gives_same_physics_as_id(tmp_path):
    definition = CaseDefinition.from_legacy(cases()['xlpe_single'], 'test', 'unit test')
    path = tmp_path.joinpath('case.json')
    path.write_text(json.dumps(definition.to_dict()), encoding='utf-8')
    assert load_problem(path).physics_sha256 == load_problem('xlpe_single').physics_sha256
    assert load_problem(str(path)).physics_sha256 == load_problem('xlpe_single').physics_sha256


def test_legacy_format_file_is_accepted(tmp_path):
    path = tmp_path.joinpath('legacy.json')
    path.write_text(json.dumps(cases()['xlpe_single']), encoding='utf-8')
    assert load_problem(path).physics_sha256 == load_problem('xlpe_single').physics_sha256


def test_v2_case_resolves_by_id():
    assert load_problem('kim_v5_summer').name == 'kim_v5_summer'


def test_cli_fem_dispatches_nested_annulus_definition(monkeypatch, tmp_path, capsys):
    from pinn_v2.solvers.fem import adapter

    calls = {}

    def run_verification(output, case=None, levels=(0, 1), definition=None):
        calls.update(output=output, case=case, levels=levels, definition=definition)
        return [dict(level=0, ndofs=42, l2_error_K=0.01, balance_pct=0.2)]

    monkeypatch.setattr(adapter, 'run_mms_fem', run_verification)
    monkeypatch.setattr(adapter, 'run_cable_fem', lambda *args, **kwargs: pytest.fail('annulus routed to cable FEM'))
    output = tmp_path / 'annulus-fem'
    assert main(['fem', '--case', 'annulus', '--levels', '0', '--output', str(output)]) == 0
    assert calls['definition'].name == 'annulus.json'
    assert calls['levels'] == [0]
    assert 'L2 error' in capsys.readouterr().out


def test_cli_help_cases_show_validate(capsys, tmp_path):
    assert main(['help', 'models']) == 0
    assert 'M1' in capsys.readouterr().out
    assert main(['cases']) == 0
    assert 'xlpe_single' in capsys.readouterr().out
    assert main(['show', 'xlpe_single']) == 0
    assert 'sha256' in capsys.readouterr().out
    bad = tmp_path.joinpath('bad.json')
    bad.write_text('{"id": "x", "kind": "cable"}', encoding='utf-8')
    assert main(['validate', str(bad)]) == 1
    assert main(['validate', 'xlpe_single']) == 0


def test_cli_pairs_reject_unknown_names():
    assert read_pairs(['pde=2'], ('pde', 'robin'), 'weights') == dict(pde=2.)
    with pytest.raises(SystemExit):
        read_pairs(['nothing=2'], ('pde',), 'weights')


def test_cli_train_with_config_and_flag_override(tmp_path, capsys):
    config = tmp_path.joinpath('run.json')
    config.write_text(json.dumps(dict(case='mms_constant', model='M1', adam_steps=5, width=8, depth=2)), encoding='utf-8')
    out = tmp_path.joinpath('run')
    assert main(['train', '--config', str(config), '--adam-steps', '20', '--lbfgs-steps', '0', '--output', str(out)]) == 0
    result = json.loads(out.joinpath('result.json').read_text(encoding='utf-8-sig'))
    assert result['status'] == 'completed' and result['config']['adam_steps'] == 20 and result['fem_labels_used'] == 0
    with pytest.raises(FileExistsError):
        main(['train', '--config', str(config), '--output', str(out)])


def test_reference_with_other_points_is_refused(tmp_path):
    problem = load_problem('mms_constant')
    xy, _, _ = problem.evaluation()
    good, bad = tmp_path.joinpath('good.npz'), tmp_path.joinpath('bad.npz')
    np.savez(good, xy=xy, T=np.zeros(len(xy)))
    np.savez(bad, xy=xy[:-1], T=np.zeros(len(xy) - 1))
    assert load_reference_npz(good, problem).shape == (len(xy),)
    with pytest.raises(ValueError):
        load_reference_npz(bad, problem)


def test_candidates_and_trial_kwargs():
    space = dict(width=[8, 16], lr=[1e-3, 1e-2], **{'weights.dirichlet': [10, 100]})
    assert len(candidates(space, 100, 'grid')) == 8
    random_a, random_b = candidates(space, 5, 'random', seed=3), candidates(space, 5, 'random', seed=3)
    assert random_a == random_b and len({json.dumps(c, sort_keys=True) for c in random_a}) == 5
    with pytest.raises(ValueError):
        candidates(dict(unknown=[1]), 2)
    model, kwargs = trial_kwargs({'model': 'M1+rba', 'weights.dirichlet': 50}, dict(lbfgs_steps=30))
    assert model == 'M1' and kwargs['attention'] == 'rba' and kwargs['lbfgs_steps'] == 0 and kwargs['weights'] == dict(dirichlet=50)


def test_search_ranks_by_held_out_loss_without_reference(tmp_path):
    space = dict(width=[8, 12], activation=['tanh'])
    summary = run_search('mms_constant', space, tmp_path.joinpath('search'), n_trials=2, base=dict(adam_steps=20, lbfgs_steps=0, depth=2))
    assert len(summary['rows']) == 2 and summary['best_trial'] in (0, 1)
    assert summary['selection_uses_reference'] is False and summary['fem_labels_used'] == 0
    assert all(r['held_out_loss'] is not None for r in summary['rows'])
    assert tmp_path.joinpath('search', 'report.md').exists()
    with pytest.raises(FileExistsError):
        run_search('mms_constant', space, tmp_path.joinpath('search'), n_trials=1)
