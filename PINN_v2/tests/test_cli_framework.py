"""Tests of the user-facing framework: same case file for every tool, CLI, run settings and search."""
import json
from pathlib import Path

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


def test_cli_exact_mode_records_pde_only_policy(tmp_path, capsys):
    reference = Path(__file__).resolve().parents[1] / 'results' / 'phase6' / 'p6_fem3' / 'aras_flat' / 'base' / 'fem_l2.npz'
    output = tmp_path / 'exact-run'
    assert main(['train', '--case', 'aras_flat', '--model', 'M1_EXACT', '--adam-steps', '1', '--lbfgs-steps', '0',
        '--width', '8', '--depth', '1', '--device', 'cpu', '--reference', str(reference), '--output', str(output),
        '--points', 'n_interior=8', 'n_layer=4', 'n_face=4', 'n_boundary=4',
        '--weights', 'pde=2', 'dirichlet=100', 'interface_flux=10', 'energy=5']) == 0
    result = json.loads(output.joinpath('result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'completed'
    assert result['config']['architecture'] == 'M1_EXACT'
    assert result['config']['loss_policy'] == 'pde_only_exact_transmission'
    assert result['config']['input_normalization'] == 'unit_box_v1'
    assert result['config']['weights']['pde'] == 2
    assert all(value == 0 for name, value in result['config']['weights'].items() if name != 'pde')


@pytest.mark.parametrize(('model', 'dtype'), [
    ('M1_EXACT', 'float64'), ('DeepXDE_EXACT', 'float64'), ('DeepXDE_EXACT', 'float32')])
def test_recorded_mms_recipe_forwards_reproducible_settings(tmp_path, monkeypatch, model, dtype):
    from pinn_v2.solvers.pinn import trainer
    package = Path(__file__).resolve().parents[1]
    recipe = package / 'configs' / 'train_mms_exact_repro_20261020.json'
    configuration = json.loads(recipe.read_text(encoding='utf-8'))
    captured = {}

    def fake_run(source, architecture, output, **settings):
        captured.update(source=source, architecture=architecture, output=output, settings=settings)
        return dict(status='completed', final_loss=0., elapsed_s=0., evaluation={})

    monkeypatch.chdir(package)
    monkeypatch.setattr(trainer, 'run', fake_run)
    assert main(['train', '--config', str(recipe), '--model', model, '--dtype', dtype,
        '--output', str(tmp_path / model)]) == 0
    assert captured['architecture'] == model
    settings = captured['settings']
    for key in ('adam_steps', 'lbfgs_steps', 'lr', 'width', 'depth', 'activation', 'points',
            'weights', 'sampling', 'resample_every', 'log_every', 'validation_every',
            'input_normalization', 'trace_parameterization', 'soil_envelope_gain'):
        assert settings[key] == configuration[key]
    assert settings['dtype'] == dtype and settings['device'] == 'cpu'
    assert settings['init_seed'] == 0 and settings['sampling_seed'] == 11
    assert settings['attention'] is None and settings['reference'] is None


def test_cli_infers_from_portable_pair_and_records_dtype(tmp_path, capsys):
    output = tmp_path / 'portable'
    assert main(['train', '--case', 'mms_constant', '--model', 'M1_EXACT', '--adam-steps', '1',
        '--width', '4', '--depth', '1', '--dtype', 'float32', '--device', 'cpu', '--output', str(output),
        '--points', 'n_interior=6', 'n_layer=6', 'n_face=6', 'n_boundary=6']) == 0
    assert json.loads((output / 'model.json').read_text())['config']['dtype'] == 'float32'
    assert main(['infer', '--checkpoint', str(output / 'model.pt'), '--xy', '.2', '.3', '--flux']) == 0
    assert 'region_ids' in capsys.readouterr().out


def test_cli_deepxde_exact_uses_pde_only_float32(tmp_path, capsys):
    pytest.importorskip('deepxde')
    output = tmp_path / 'deepxde-exact'
    assert main(['train', '--case', 'mms_interface', '--model', 'DeepXDE_EXACT', '--adam-steps', '1',
        '--lbfgs-steps', '0', '--width', '4', '--depth', '1', '--dtype', 'float32', '--device', 'cpu',
        '--points', 'n_interior=6', 'n_layer=6', 'n_face=6', 'n_boundary=6', '--output', str(output)]) == 0
    result = json.loads((output / 'result.json').read_text())
    assert result['constraint_count'] == 0
    assert result['config']['dtype'] == 'float32'
    assert result['config']['architecture'] == 'DeepXDE_EXACT'
    assert result['config']['loss_policy'] == 'pde_only_exact_transmission'


@pytest.mark.parametrize('case', ['mms_interface', 'annulus'])
def test_cli_exact_mode_trains_existing_mms_and_cigre_cases(case, tmp_path, capsys):
    output = tmp_path / case
    assert main(['train', '--case', case, '--model', 'M1_EXACT', '--adam-steps', '1', '--lbfgs-steps', '0',
        '--width', '6', '--depth', '1', '--device', 'cpu', '--output', str(output),
        '--points', 'n_interior=8', 'n_layer=4', 'n_face=4', 'n_boundary=4']) == 0
    result = json.loads(output.joinpath('result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'completed'
    assert result['config']['architecture'] == 'M1_EXACT'
    assert result['config']['loss_policy'] == 'pde_only_exact_transmission'
    assert result['evaluation_finite'] is True


def test_reference_with_other_points_is_refused(tmp_path):
    problem = load_problem('mms_constant')
    xy, _, _ = problem.evaluation()
    good, bad = tmp_path.joinpath('good.npz'), tmp_path.joinpath('bad.npz')
    np.savez(good, xy=xy, T=np.zeros(len(xy)))
    np.savez(bad, xy=xy[:-1], T=np.zeros(len(xy) - 1))
    assert load_reference_npz(good, problem).shape == (len(xy),)
    with pytest.raises(ValueError):
        load_reference_npz(bad, problem)


def test_reference_with_same_points_but_other_physics_is_refused(tmp_path):
    problem = load_problem('mms_interface')
    points, _, _ = problem.evaluation()
    path = tmp_path / 'reference.npz'
    np.savez(path, xy=points, T=np.zeros(len(points)))
    path.with_suffix('.json').write_text(json.dumps(dict(physics_sha256='different-material-or-source')))
    with pytest.raises(ValueError, match='Reference physics'):
        load_reference_npz(path, problem)


def test_candidates_and_trial_kwargs():
    space = dict(width=[8, 16], lr=[1e-3, 1e-2], **{'weights.dirichlet': [10, 100]})
    assert len(candidates(space, 100, 'grid')) == 8
    random_a, random_b = candidates(space, 5, 'random', seed=3), candidates(space, 5, 'random', seed=3)
    assert random_a == random_b and len({json.dumps(c, sort_keys=True) for c in random_a}) == 5
    with pytest.raises(ValueError):
        candidates(dict(unknown=[1]), 2)
    model, kwargs = trial_kwargs({'model': 'M1+rba', 'weights.dirichlet': 50}, dict(lbfgs_steps=30))
    assert model == 'M1' and kwargs['attention'] == 'rba' and kwargs['lbfgs_steps'] == 0 and kwargs['weights'] == dict(dirichlet=50)


def test_verification_campaign_excel_has_one_sheet_per_case_and_blank_fem_self_errors(tmp_path):
    load_workbook = pytest.importorskip('openpyxl').load_workbook
    from pinn_v2.experiments.verification_exact_campaign import export_reports, case_definitions, COLUMNS
    definitions = case_definitions()
    for case, definition in definitions.items():
        problem = load_problem(definition)
        points, labels, _ = problem.evaluation()
        folder = tmp_path/case/'fem'
        folder.mkdir(parents=True)
        np.savez(folder/f'{case}_l1.npz', xy=points, T=problem.reference(points, labels))
        (folder/f'{case}_l1.json').write_text(json.dumps(dict(physics_sha256=problem.physics_sha256,
            balance_pct=0., elapsed_s=1.)), encoding='utf-8')
    model_folder = tmp_path/'mms_interface'/'M0_EXACT'
    model_folder.mkdir()
    np.savez(model_folder/'collocation.npz', interior_0=np.zeros((2, 2)),
        interior_1=np.zeros((3, 2)), interface_0=np.zeros((2, 2)), boundary_0=np.zeros((4, 2)))
    export_reports(tmp_path, definitions, dict(adam_steps=400, lbfgs_steps=5, extra_lbfgs_steps=5))
    workbook = load_workbook(tmp_path/'comparison.xlsx', data_only=True)
    assert set(workbook.sheetnames) == set(definitions)
    for sheet in workbook:
        assert sheet.cell(8, 1).value == 'FEM_P2_level1'
        assert sheet.cell(8, COLUMNS.index('NRMSE_FEM_pct')+1).value is None
        assert sheet.cell(8, COLUMNS.index('Tmax_error_K')+1).value is None
        assert sheet.max_row == 14
        assert sheet.cell(8, COLUMNS.index('collocation_total_points')+1).value is None
    sheet = workbook['mms_interface']
    assert sheet.cell(9, COLUMNS.index('training_pde_points')+1).value == 5
    assert sheet.cell(9, COLUMNS.index('interface_points')+1).value == 2
    assert sheet.cell(9, COLUMNS.index('boundary_points')+1).value == 4
    assert sheet.cell(9, COLUMNS.index('collocation_total_points')+1).value == 11
    assert sheet.cell(9, COLUMNS.index('collocation_unique_points')+1).value == 1
    assert sheet.cell(10, COLUMNS.index('collocation_total_points')+1).value is None
    markdown = (tmp_path/'report.md').read_text(encoding='utf-8')
    assert '| M0_EXACT | incomplete | 5 | 2 | 4 | 11 | 1 |' in markdown


def test_annulus_extension_summary_adds_excel_tab_and_markdown(tmp_path):
    load_workbook=pytest.importorskip('openpyxl').load_workbook
    from pinn_v2.experiments.verification_exact_campaign import export_reports,case_definitions
    definitions={'annulus':case_definitions()['annulus']}
    problem=load_problem(definitions['annulus'])
    xy,labels,_=problem.evaluation()
    fem=tmp_path/'annulus'/'fem'; fem.mkdir(parents=True)
    np.savez(fem/'annulus_l1.npz',xy=xy,T=problem.reference(xy,labels))
    (fem/'annulus_l1.json').write_text(json.dumps(dict(physics_sha256=problem.physics_sha256,balance_pct=.2,elapsed_s=1)),encoding='utf-8')
    extension=tmp_path/'annulus_extension_1000_50'; extension.mkdir()
    (extension/'annulus_extension_summary_1000_50.json').write_text(json.dumps(dict(rows=[dict(
        model='M0_EXACT',initialization_policy='weights_only_warm_start',adam_steps=1000,
        lbfgs_outer_calls=80,extra_lbfgs_outer_calls=30,initial_nrmse_fem_pct=33.,
        final_nrmse_fem_pct=2.,inner_boundary_max_error_K=1.,global_balance_pct=4.,
        accuracy_targets_met=False,all_raw_gates=False,elapsed_s=123.,run_directory='run/M0')])) ,encoding='utf-8')
    export_reports(tmp_path,definitions,dict(adam_steps=400,lbfgs_steps=5))
    workbook=load_workbook(tmp_path/'comparison.xlsx',data_only=True)
    assert 'Annulus_extend' in workbook.sheetnames
    assert workbook['Annulus_extend'].cell(2,1).value=='M0_EXACT'
    assert workbook['Annulus_extend'].cell(2,2).value=='weights_only_warm_start'
    assert workbook['Annulus_extend'].auto_filter.ref.endswith('T2')
    report=(tmp_path/'report.md').read_text(encoding='utf-8')
    assert 'Annulus_extend' not in report
    assert 'Annulus: extension desde pesos previos' in report
    assert 'weights_only_warm_start' in report


@pytest.mark.parametrize('returncode', [0, 1])
def test_verification_campaign_validation_gate_blocks_failed_tests(tmp_path, monkeypatch, returncode):
    from types import SimpleNamespace
    from pinn_v2.experiments import verification_exact_campaign as campaign
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(returncode=returncode, stdout='captured tests', stderr=''))
    if returncode:
        with pytest.raises(RuntimeError, match='campaign was not started'):
            campaign.validate_before_campaign(tmp_path)
    else:
        campaign.validate_before_campaign(tmp_path)
    status = json.loads((tmp_path/'trainer_regression_status.json').read_text())
    assert status['passed'] == (returncode == 0)
    assert 'captured tests' in (tmp_path/'trainer_regression_output.txt').read_text()


def test_search_ranks_by_held_out_loss_without_reference(tmp_path):
    space = dict(width=[8, 12], activation=['tanh'])
    summary = run_search('mms_constant', space, tmp_path.joinpath('search'), n_trials=2, base=dict(adam_steps=20, lbfgs_steps=0, depth=2))
    assert len(summary['rows']) == 2 and summary['best_trial'] in (0, 1)
    assert summary['selection_uses_reference'] is False and summary['fem_labels_used'] == 0
    assert all(r['held_out_loss'] is not None for r in summary['rows'])
    assert tmp_path.joinpath('search', 'report.md').exists()
    with pytest.raises(FileExistsError):
        run_search('mms_constant', space, tmp_path.joinpath('search'), n_trials=1)
