import math

from pinn_v2.experiments import fem_hparam_optimization as optimization
from pinn_v2.experiments.fem_hparam_optimization import CASE_DEFINITIONS, DEFAULT_CASES, ROOT, _effective_config, _reference_path_record, choose_candidate, draw_candidates, quality_score


def test_default_sandbox_includes_all_mms_and_real_cable_cases():
    assert len(DEFAULT_CASES) == 11
    assert {case for case in DEFAULT_CASES if case.startswith('mms_')} == {
        'mms_constant', 'mms_high_contrast', 'mms_interface', 'mms_layered_y',
        'mms_robin', 'mms_smooth_2d', 'mms_variable',
    }
    assert {'aras_flat', 'aras_single', 'xlpe_discrete_layers'} <= set(DEFAULT_CASES)
    assert CASE_DEFINITIONS['xlpe_discrete_layers'][1] == 'cable'
    assert 'DeepXDE' in optimization.DEFAULT_ARCHITECTURES


def test_candidate_design_is_seeded_and_contains_ui_baseline():
    first = draw_candidates(5, 23, 'M1')
    second = draw_candidates(5, 23, 'M1')
    assert first == second
    assert first[0]['width'] == 24
    assert first[0]['adam_steps'] == 400
    assert first[0]['weights.dirichlet'] == 100.
    assert len({tuple(sorted(row.items())) for row in first}) == len(first)


def test_fb_pinn_baseline_uses_frontend_width():
    assert draw_candidates(1, 0, 'FBPINN')[0]['width'] == 24


def test_candidate_design_includes_every_loss_term_weight():
    candidate = draw_candidates(1, 0, 'M0')[0]
    assert {key.removeprefix('weights.') for key in candidate if key.startswith('weights.')} == {
        'pde', 'constitutive', 'energy', 'interface_temperature', 'interface_flux',
        'dirichlet', 'neumann', 'robin',
    }


def test_tpe_candidates_keep_pde_active_for_every_architecture():
    class FirstChoiceTrial:
        def suggest_categorical(self, name, choices):
            return choices[0]

        def suggest_float(self, name, low, high, log=False):
            return low

    for architecture in ('M0', 'M1', 'M2', 'CUSP', 'DCSNN', 'FBPINN', 'DeepXDE'):
        candidate = optimization._suggest_candidate(FirstChoiceTrial(), architecture)
        assert candidate['weights.pde'] > 0.
        if architecture == 'DeepXDE':
            assert candidate['weights.constitutive'] == 0.
        else:
            assert candidate['weights.constitutive'] in optimization.TPE_WEIGHT_VALUES


def test_effective_config_records_attention_disabling_lbfgs():
    requested = dict(attention='rba', lbfgs_steps=40)
    effective = _effective_config(requested)
    assert requested['lbfgs_steps'] == 40
    assert effective['lbfgs_steps'] == 0
    assert _effective_config(dict(attention=None, lbfgs_steps=40))['lbfgs_steps'] == 40


def test_transferred_candidates_are_normalized_for_each_architecture():
    transferred = optimization.M0_TRANSFERRED_PRIORS[0]
    fb_config = optimization._normalize_candidate(transferred, 'FBPINN')
    assert fb_config['width'] in (8, 16, 24)
    assert fb_config['depth'] in (2, 3, 4)
    assert fb_config['adam_steps'] == optimization.REFINE_STEPS
    assert fb_config['point_multiplier'] <= 2.

    deepxde_config = optimization._normalize_candidate(transferred, 'DeepXDE')
    assert deepxde_config['attention'] is None
    assert deepxde_config['weights.constitutive'] == 0.
    assert deepxde_config['deepxde_model'] == 'PFNN'


def test_reference_path_record_handles_project_and_external_paths(tmp_path):
    project_reference = ROOT / 'results' / 'reference.npz'
    assert _reference_path_record(project_reference) == str(project_reference.relative_to(ROOT))
    external_reference = tmp_path / 'reference.npz'
    assert _reference_path_record(external_reference) == str(external_reference.resolve())


def test_deepxde_report_documents_only_supported_search_options(tmp_path, monkeypatch):
    monkeypatch.setattr(optimization, '_write_optimization_plot', lambda *_: None)
    summary = dict(architecture='DeepXDE', screen_cases=['mms_interface'], cases=['mms_interface'],
        finalist_seeds=[11, 23], screen_seeds=[0], search_seed=1, n_trials=1,
        smoke_steps=optimization.SMOKE_STEPS, promoted_candidates=optimization.PROMOTED_CANDIDATES,
        refine_steps=optimization.REFINE_STEPS, prune_min_steps=optimization.PRUNE_MIN_STEPS,
        prune_patience=optimization.PRUNE_PATIENCE, checkpoint_interval=optimization.SMOKE_LOG_EVERY,
        promoted_rows=[],
        fem_level=2, target_accuracy_pct=1., base_points=optimization.BASE_POINTS,
        selected_trial=None, selected_config=None, selection_reason='not selected', selection_metric='worst case',
        finalists=0, rows=[], finalist_rows=[])

    optimization.write_report(tmp_path, summary)

    report = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert 'PFNN/FNN' in report
    assert 'constitutive = 0' in report
    assert 'RBA/SA' not in report


def test_quality_score_uses_worst_normalized_metric():
    evaluation = dict(nrmse_pct=2.5, error_rise_pct=1., region_rmse_rise_pct={'soil': 2.},
        balance={'global_derived_pct': 1., 'region_derived_pct': {'soil': .5}}, interfaces=[])
    assert quality_score(evaluation) == 2.5
    assert math.isinf(quality_score({}))
    assert quality_score(dict(nrmse_pct=0., error_rise_pct=0.)) == 0.
    assert quality_score(dict(nrmse_pct=0., error_rise_pct=0.), variability_error_pct=5.) == 1.
    assert quality_score(dict(nrmse_pct=0., error_rise_pct=0., hotspot_distance_m=.1)) == 2.
    assert quality_score(dict(nrmse_pct=0., error_rise_pct=0., hotspot_distance_m=.1), include_hotspot=False) == 0.


def test_candidate_selection_prefers_fastest_accepted_candidate():
    rows = [dict(trial=0, quality_score=0.9, elapsed_s=20.), dict(trial=1, quality_score=0.7, elapsed_s=30.)]
    selected, reason = choose_candidate(rows)
    assert selected['trial'] == 0
    assert 'fastest' in reason


def test_candidate_selection_uses_error_time_tradeoff_when_none_accepted():
    rows = [dict(trial=0, quality_score=2.0, elapsed_s=20.), dict(trial=1, quality_score=1.5, elapsed_s=40.)]
    selected, reason = choose_candidate(rows)
    assert selected['trial'] == 1
    assert 'no candidate met' in reason


def test_staged_run_writes_method_report_and_plot(tmp_path, monkeypatch):
    stages = []

    def fake_evaluate(architecture, candidate, cases, references_root, model_root, seeds, case_index, study_id,
        stage, **options):
        stages.append((stage, tuple(cases), tuple(seeds), options))
        case_rows = [dict(case=case, seed=seed, status='completed', elapsed_s=1., quality_score=.5,
            nrmse_pct=.5, error_rise_pct=.5, error_Tmax_abs_K=.01, hotspot_distance_m=0.,
            temperature_std_error_pct_rise=1., worst_region_rmse_rise_pct=.5,
            balance_global_pct=.5, acceptance='accepted', error=None) for case in cases for seed in seeds]
        return dict(trial=candidate['_trial'],
            config={key: value for key, value in candidate.items() if key != '_trial'},
            effective_config=optimization._effective_config(candidate), cases=case_rows,
            quality_score=.5, elapsed_s=float(len(case_rows)), complete=True)

    monkeypatch.setattr(optimization, '_evaluate_candidate', fake_evaluate)
    output = tmp_path / 'study'
    finalist_seeds = optimization.DEFAULT_FINALIST_SEEDS
    summaries = optimization.run_optimization(output, tmp_path / 'refs', architectures=('M0',),
        cases=('mms_interface', 'aras_flat'), n_trials=5, screen_cases=('mms_interface',),
        screen_seeds=(0,), finalist_seeds=finalist_seeds, finalists=2, promoted_candidates=4,
        study_id='test_study')

    summary = summaries['M0']
    assert [stage for stage, _, _, _ in stages] == ['screen'] * 5 + ['refine'] * 4 + ['finalist'] * 2
    assert stages[0][3] == dict(adam_steps=optimization.SMOKE_STEPS, full_audit=False, prune=True)
    assert all(options == dict(adam_steps=optimization.REFINE_STEPS, full_audit=False)
        for _, _, _, options in stages[5:9])
    assert all(cases == ('mms_interface', 'aras_flat') and seeds == finalist_seeds
        for _, cases, seeds, _ in stages[9:])
    assert all(options == dict(adam_steps=optimization.REFINE_STEPS, full_audit=True)
        for _, _, _, options in stages[9:])
    assert summary['promoted_candidates'] == 4
    assert summary['finalists'] == 2
    assert summary['default_recommendation'] == 'validated'
    assert (output / 'M0' / 'optimization.png').is_file()
    report = (output / 'M0' / 'report.md').read_text(encoding='utf-8')
    assert 'Optuna TPE' in report
    assert '400 Adam steps' in report
    assert 'five independent seeds' in report