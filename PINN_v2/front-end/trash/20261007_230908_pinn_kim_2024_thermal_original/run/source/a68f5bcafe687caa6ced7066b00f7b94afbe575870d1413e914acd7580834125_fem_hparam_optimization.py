"""Tune frontend PINN settings against FEM level-2 references on a fixed case suite.

FEM fields are evaluation-only. Each candidate uses common initialization and sampling
seeds across candidates, then is ranked by the worst normalized case error and runtime.
"""
import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from pinn_v2.evaluation.metrics import CRITERIA
from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.manifest.run_manifest import build_manifest, write_json_atomic, write_manifest
from pinn_v2.solvers.pinn import trainer
from pinn_v2.solvers.pinn.problem import load_problem

ROOT = Path(__file__).resolve().parents[2]
CASE_DEFINITIONS = {
    'annulus': ('references/verification/annulus.json', 'mms'),
    'mms_constant': ('references/verification/mms_constant.json', 'mms'),
    'mms_high_contrast': ('references/verification/mms_high_contrast.json', 'mms'),
    'mms_interface': ('references/verification/mms_interface.json', 'mms'),
    'mms_layered_y': ('references/verification/mms_layered_y.json', 'mms'),
    'mms_robin': ('references/verification/mms_robin.json', 'mms'),
    'mms_smooth_2d': ('references/verification/mms_smooth_2d.json', 'mms'),
    'mms_variable': ('references/verification/mms_variable.json', 'mms'),
    'aras_flat': ('references/cables/aras_flat.json', 'cable'),
    'aras_single': ('references/cables/aras_single.json', 'cable'),
    'xlpe_discrete_layers': ('references/cables/xlpe_discrete_layers.json', 'cable'),
}
DEFAULT_CASES = tuple(CASE_DEFINITIONS)
SCREEN_CASES = ('aras_flat', 'aras_single', 'xlpe_discrete_layers', 'mms_high_contrast')
DEFAULT_ARCHITECTURES = ('M0', 'M1', 'M2', 'CUSP', 'DCSNN', 'FBPINN', 'DeepXDE')
BASE_POINTS = dict(n_interior=256, n_layer=128, n_face=96, n_boundary=96)
SMOKE_STEPS = 400
REFINE_STEPS = 800
PROMOTED_CANDIDATES = 4
DEFAULT_FINALISTS = 2
DEFAULT_FINALIST_SEEDS = (11, 23, 37, 101, 202)
SMOKE_LOG_EVERY = 25
PRUNE_MIN_STEPS = 200
PRUNE_PATIENCE = 5
SMOKE_POINT_MULTIPLIER_MAX = 1.0
VARIABILITY_TOLERANCE_PCT = 5.0
HOTSPOT_TOLERANCE_M = 0.05
ACCURACY_TARGET_PCT = 1.0
BASE_CANDIDATE = dict(width=24, depth=3, lr=3e-3, activation='tanh', adam_steps=400,
    lbfgs_steps=0, attention=None, point_multiplier=1.0)
SEARCH_SPACE = dict(width=(16, 32, 48), depth=(2, 3, 4), lr=(1e-3, 2e-3, 3e-3),
    activation=('tanh', 'silu'), adam_steps=(400, 800), lbfgs_steps=(0, 40),
    attention=(None, 'rba', 'sa'), point_multiplier=(0.5, 1.0, 1.5),
    **{'weights.pde': (0.5, 1., 2.), 'weights.constitutive': (0.5, 1., 2.), 'weights.energy': (0.5, 1., 2.),
       'weights.interface_temperature': (30., 100., 300.), 'weights.interface_flux': (3., 10., 30.),
       'weights.dirichlet': (30., 100., 300.), 'weights.neumann': (3., 10., 30.), 'weights.robin': (3., 10., 30.)})

TPE_WEIGHT_VALUES = (0., 0.01, 0.1, 1., 10., 100., 1000.)
TPE_PDE_WEIGHT_VALUES = (0.01, 0.1, 1., 10., 100., 1000.)
TPE_BOUNDARY_WEIGHT_VALUES = (0.1, 1., 10., 100., 1000.)
TPE_BASELINE = dict(width=24, depth=3, lr=3e-3, activation='tanh', adam_steps=REFINE_STEPS,
    lbfgs_steps=0, attention=None, point_multiplier=1., **{
        'weights.pde': 1., 'weights.constitutive': 1., 'weights.energy': 1.,
        'weights.interface_temperature': 100., 'weights.interface_flux': 10.,
        'weights.dirichlet': 100., 'weights.neumann': 10., 'weights.robin': 10.,
    })
M0_TRANSFERRED_PRIORS = (
    dict(width=48, depth=6, lr=0.0032645262904700397, activation='silu', adam_steps=1600,
        lbfgs_steps=100, attention=None, point_multiplier=2., **{
            'weights.pde': 0.01, 'weights.constitutive': 1., 'weights.energy': 0.01,
            'weights.interface_temperature': 1000., 'weights.interface_flux': 100.,
            'weights.dirichlet': 0.1, 'weights.neumann': 10., 'weights.robin': 10.,
        }),
    dict(width=16, depth=5, lr=0.006960987623079458, activation='tanh', adam_steps=3200,
        lbfgs_steps=0, attention='sa', point_multiplier=2., **{
            'weights.pde': 10., 'weights.constitutive': 10., 'weights.energy': 1000.,
            'weights.interface_temperature': 1000., 'weights.interface_flux': 100.,
            'weights.dirichlet': 100., 'weights.neumann': 100., 'weights.robin': 1.,
        }),
)


def draw_candidates(n_trials, seed, architecture):
    """Return a deterministic random design that always includes the UI's quick baseline."""
    if n_trials < 1:
        raise ValueError('n_trials must be positive')
    space = dict(SEARCH_SPACE)
    if architecture == 'FBPINN':
        space['width'] = (8, 16, 24)
        space['depth'] = (2, 3)
    baseline = dict(BASE_CANDIDATE)
    if architecture == 'FBPINN':
        baseline['width'] = 24
    baseline.update({'weights.pde': 1., 'weights.constitutive': 1., 'weights.energy': 1.,
        'weights.interface_temperature': 100., 'weights.interface_flux': 10.,
        'weights.dirichlet': 100., 'weights.neumann': 10., 'weights.robin': 10.})
    keys = tuple(space)
    rng = random.Random(seed)
    values = []
    total = math.prod(len(space[key]) for key in keys)
    if n_trials > total:
        raise ValueError(f'n_trials exceeds the {total} unique search configurations')
    seen = {tuple(baseline[key] for key in keys)}
    values.append(baseline)
    while len(values) < n_trials:
        choice = tuple(rng.choice(space[key]) for key in keys)
        if choice in seen:
            continue
        seen.add(choice)
        values.append(dict(zip(keys, choice)))
    return values


def quality_score(evaluation, variability_error_pct=None, include_hotspot=True):
    """Worst normalized FEM accuracy/physics error; 1.0 meets the 1% accuracy target."""
    if not evaluation or evaluation.get('nrmse_pct') is None:
        return math.inf
    ratios = [evaluation['nrmse_pct'] / ACCURACY_TARGET_PCT,
        evaluation['error_rise_pct'] / ACCURACY_TARGET_PCT]
    if include_hotspot and evaluation.get('hotspot_distance_m') is not None:
        ratios.append(evaluation['hotspot_distance_m'] / HOTSPOT_TOLERANCE_M)
    if variability_error_pct is not None:
        ratios.append(variability_error_pct / VARIABILITY_TOLERANCE_PCT)
    region = evaluation.get('region_rmse_rise_pct', {})
    if region:
        ratios.append(max(region.values()) / ACCURACY_TARGET_PCT)
    balance = evaluation.get('balance', {})
    balance_values = [balance.get('global_derived_pct')]
    balance_values.extend((balance.get('region_derived_pct') or {}).values())
    balance_values = [value for value in balance_values if value is not None]
    if balance_values:
        ratios.append(max(balance_values) / CRITERIA['balance_pct'])
    interfaces = evaluation.get('interfaces', [])
    if interfaces:
        ratios.append(max(item['T_jump_max_K'] for item in interfaces) / CRITERIA['interface_T_jump_max_K'])
        ratios.append(max(item['flux_jump_rms_pct'] for item in interfaces) / CRITERIA['interface_flux_rms_pct'])
    return max(ratios)


def choose_candidate(rows):
    """Prefer the fastest candidate meeting all gates; otherwise trade error against log time."""
    completed = [row for row in rows if math.isfinite(row['quality_score']) and math.isfinite(row['elapsed_s'])]
    if not completed:
        return None, 'no candidate completed with finite FEM metrics'
    accepted = [row for row in completed if row['quality_score'] <= 1.0]
    if accepted:
        return min(accepted, key=lambda row: (row['elapsed_s'], row['quality_score'])), 'fastest candidate meeting every proposed accuracy gate'
    median_time = max(float(np.median([row['elapsed_s'] for row in completed])), 1e-9)
    return min(completed, key=lambda row: (row['quality_score'] * (1. + 0.05 * math.log1p(row['elapsed_s'] / median_time)), row['quality_score'], row['elapsed_s'])), 'best accuracy/runtime tradeoff; no candidate met every proposed gate'


def _case_source(case):
    relative, kind = CASE_DEFINITIONS[case]
    return ROOT.joinpath('pinn_v2', 'cases', 'v2', relative), kind


def _weights(candidate):
    result = {}
    for key, value in candidate.items():
        if key.startswith('weights.'):
            result[key.split('.', 1)[1]] = value
    return result


def _points(candidate):
    multiplier = candidate['point_multiplier']
    return {key: max(1, round(value * multiplier)) for key, value in BASE_POINTS.items()}


def _suggest_candidate(trial, architecture):
    """Suggest a broad mixed search point; TPE adapts proposals to prior scores."""
    widths = (8, 16, 24) if architecture == 'FBPINN' else (16, 24, 32, 48, 64, 96)
    depths = (2, 3, 4) if architecture == 'FBPINN' else (2, 3, 4, 5, 6)
    candidate = dict(
        width=trial.suggest_categorical('width', widths),
        depth=trial.suggest_categorical('depth', depths),
        lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        activation=trial.suggest_categorical('activation', ('tanh', 'silu')),
        adam_steps=trial.suggest_categorical('adam_steps', (REFINE_STEPS,)),
        attention=trial.suggest_categorical('attention', (None,)) if architecture == 'DeepXDE' else trial.suggest_categorical('attention', (None, 'rba', 'sa')),
        point_multiplier=trial.suggest_categorical('point_multiplier', (1., 1.5, 2.)),
    )
    if architecture == 'DeepXDE':
        candidate['deepxde_model'] = trial.suggest_categorical('deepxde_model', ('PFNN', 'FNN'))
    for key in ('pde', 'constitutive', 'energy'):
        values = TPE_PDE_WEIGHT_VALUES if key == 'pde' else TPE_WEIGHT_VALUES
        candidate[f'weights.{key}'] = 0.0 if architecture == 'DeepXDE' and key == 'constitutive' else trial.suggest_categorical(f'weights.{key}', values)
    candidate['weights.interface_temperature'] = trial.suggest_categorical(
        'weights.interface_temperature', TPE_BOUNDARY_WEIGHT_VALUES)
    for key in ('interface_flux', 'dirichlet', 'neumann', 'robin'):
        candidate[f'weights.{key}'] = trial.suggest_categorical(
            f'weights.{key}', TPE_BOUNDARY_WEIGHT_VALUES)
    candidate['lbfgs_steps'] = 0 if candidate['attention'] else trial.suggest_categorical(
        'lbfgs_steps', (0, 100, 300))
    return candidate


def _effective_config(candidate):
    config = {key: value for key, value in candidate.items() if not key.startswith('_')}
    if config['attention']:
        config['lbfgs_steps'] = 0
    return config


def _normalize_candidate(candidate, architecture):
    """Keep transferred priors within the active architecture's search space."""
    normalized = dict(candidate)
    normalized['adam_steps'] = REFINE_STEPS
    normalized['point_multiplier'] = min(normalized['point_multiplier'], 2.0)
    if architecture == 'FBPINN':
        normalized['width'] = min((8, 16, 24), key=lambda value: abs(value - normalized['width']))
        normalized['depth'] = min((2, 3, 4), key=lambda value: abs(value - normalized['depth']))
    if architecture == 'DeepXDE':
        normalized['attention'] = None
        normalized['weights.constitutive'] = 0.0
        normalized.setdefault('deepxde_model', 'PFNN')
    if normalized['attention']:
        normalized['lbfgs_steps'] = 0
    return normalized


def _reference_path_record(reference_path):
    resolved = Path(reference_path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _candidate_from_artifact(result, architecture, trial_number):
    config = result['config']
    points = config['points']
    multiplier = points['n_interior'] / BASE_POINTS['n_interior']
    candidate = dict(width=config['width'], depth=config['depth'], lr=config['lr'],
        activation=config['activation'], adam_steps=REFINE_STEPS,
        lbfgs_steps=config.get('lbfgs_steps', 0), attention=config.get('attention'),
        point_multiplier=multiplier, _trial=trial_number)
    candidate.update({f'weights.{key}': value for key, value in config.get('weights', {}).items()})
    if architecture == 'DeepXDE':
        candidate['deepxde_model'] = config.get('deepxde_model', 'PFNN')
    return _normalize_candidate(candidate, architecture)


def _load_case_artifact(output, architecture, trial_number, case, seed, stage_steps, study_id):
    result_path = Path(output).joinpath('result.json')
    manifest_path = Path(output).joinpath('manifest.json')
    field_path = Path(output).joinpath('evaluation.npz')
    if not result_path.is_file() or not manifest_path.is_file() or not field_path.is_file():
        raise RuntimeError(f'Cannot resume incomplete run without overwriting it: {output}')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    extra = manifest.get('extra', {})
    if (result.get('case') != case or result.get('config', {}).get('architecture') != architecture
            or extra.get('study') != study_id or extra.get('status') != 'completed'
            or manifest.get('seeds', {}).get('init') != seed):
        raise RuntimeError(f'Existing artifact does not match a completed run: {output}')
    evaluation = result.get('evaluation', {})
    field = np.load(field_path)
    weights = np.asarray(field['weights'], dtype=float)
    predicted = np.asarray(field['T'], dtype=float)
    truth = np.asarray(field['T_reference'], dtype=float)
    pred_mean = np.average(predicted, weights=weights)
    truth_mean = np.average(truth, weights=weights)
    pred_std = float(np.sqrt(np.average((predicted - pred_mean) ** 2, weights=weights)))
    truth_std = float(np.sqrt(np.average((truth - truth_mean) ** 2, weights=weights)))
    source, _ = _case_source(case)
    problem = load_problem(source)
    rise = max(float(np.max(truth)) - problem.T_level, 1e-12)
    variability_error_pct = 100. * abs(pred_std - truth_std) / rise
    return dict(case=case, seed=seed, status='completed', elapsed_s=result['elapsed_s'],
        quality_score=quality_score(evaluation, variability_error_pct, include_hotspot=case != 'annulus'),
        nrmse_pct=evaluation.get('nrmse_pct'), training_adam_steps=stage_steps,
        training_lbfgs_steps=result.get('config', {}).get('lbfgs_steps', 0),
        actual_adam_steps=stage_steps, early_stopped=False, stop_reason=None,
        checkpoint_count=0, best_checkpoint_score=None, points=result['config']['points'],
        error_rise_pct=evaluation.get('error_rise_pct'), error_Tmax_abs_K=evaluation.get('error_Tmax_abs_K'),
        hotspot_distance_m=evaluation.get('hotspot_distance_m'), temperature_std_pred_K=pred_std,
        temperature_std_ref_K=truth_std, temperature_std_error_pct_rise=variability_error_pct,
        worst_region_rmse_rise_pct=max(evaluation.get('region_rmse_rise_pct', {}).values(), default=None),
        balance_global_pct=evaluation.get('balance', {}).get('global_derived_pct'),
        acceptance=evaluation.get('acceptance', {}).get('state'),
        physics_sha256=manifest.get('extra', {}).get('physics_sha256'),
        fem_reference=None, error=(result.get('error') or {}).get('message'))


def _load_candidate_stage(architecture, model_root, stage, trial_number, cases, seeds, stage_steps, study_id):
    case_rows = []
    candidate = None
    for case in cases:
        for seed in seeds:
            output = model_root.joinpath(stage, f'trial_{trial_number:03d}', case, f'seed_{seed}')
            result_path = output.joinpath('result.json')
            result = json.loads(result_path.read_text(encoding='utf-8'))
            candidate = _candidate_from_artifact(result, architecture, trial_number)
            case_rows.append(_load_case_artifact(output, architecture, trial_number, case, seed, stage_steps, study_id))
    finite_scores = [row['quality_score'] for row in case_rows if math.isfinite(row['quality_score'])]
    return dict(trial=trial_number,
        config={key: value for key, value in candidate.items() if not key.startswith('_')},
        effective_config=_effective_config(candidate), cases=case_rows,
        quality_score=max(finite_scores) if len(finite_scores) == len(case_rows) else math.inf,
        elapsed_s=sum(row['elapsed_s'] for row in case_rows),
        complete=len(finite_scores) == len(case_rows))


def _evaluate_candidate(architecture, candidate, cases, references_root, model_root, seeds, case_index, study_id,
    stage, adam_steps=None, full_audit=True, prune=False, resume=False):
    case_rows = []
    effective = _effective_config(candidate)
    if architecture == 'DeepXDE':
        effective['deepxde_model'] = candidate.get('deepxde_model', 'PFNN')
        effective['attention'] = None
    for case in cases:
        source, kind = _case_source(case)
        problem = load_problem(source)
        reference_path = references_root.joinpath(case, f'{case}_l2.npz' if kind == 'mms' else 'fem_l2.npz')
        reference = load_reference_npz(reference_path, problem)
        for seed in seeds:
            output = model_root.joinpath(stage, f"trial_{candidate['_trial']:03d}", case, f'seed_{seed}')
            stage_steps = adam_steps or candidate['adam_steps']
            if resume and output.joinpath('result.json').is_file():
                case_rows.append(_load_case_artifact(output, architecture, candidate['_trial'], case, seed,
                    stage_steps, study_id))
                continue
            if resume and output.exists():
                if any(output.iterdir()):
                    raise RuntimeError(f'Cannot resume incomplete run without overwriting it: {output}')
                output.rmdir()
            point_multiplier = min(candidate['point_multiplier'], SMOKE_POINT_MULTIPLIER_MAX) if prune else candidate['point_multiplier']
            stage_points = {key: max(1, round(value * point_multiplier)) for key, value in BASE_POINTS.items()}
            stage_lbfgs_steps = 0 if prune else effective['lbfgs_steps']
            monitor = dict(best_score=math.inf, stale=0, checkpoints=0, stopped=False, reason=None)

            def checkpoint_score(row, model):
                evaluation, _ = trainer.evaluate_model(model, reference,
                    f'FEniCSx level 2 ({problem.physics_sha256})', include_audit=False)
                score = quality_score(evaluation, include_hotspot=case != 'annulus')
                monitor['checkpoints'] += 1
                best_score = monitor['best_score']
                improved = math.isfinite(score) and (not math.isfinite(best_score)
                    or score < best_score - max(0.01, abs(best_score) * 0.01))
                if improved:
                    monitor.update(best_score=score, stale=0)
                else:
                    monitor['stale'] += 1
                if row['step'] + 1 >= PRUNE_MIN_STEPS and not math.isfinite(score):
                    monitor.update(stopped=True, reason='non-finite FEM checkpoint score')
                    return True
                if row['step'] + 1 >= PRUNE_MIN_STEPS and monitor['stale'] >= PRUNE_PATIENCE:
                    monitor.update(stopped=True, reason=f'no material FEM-score improvement for {PRUNE_PATIENCE} checkpoints')
                    return True
                return False

            result = trainer.run(source, architecture, output, adam_steps=stage_steps,
                lbfgs_steps=stage_lbfgs_steps, lr=candidate['lr'],
                width=candidate['width'], depth=candidate['depth'], activation=candidate['activation'],
                weights=_weights(candidate), points=stage_points, init_seed=seed,
                sampling_seed=20261002 + case_index[case] + 1000 * seed,
                log_every=SMOKE_LOG_EVERY if prune else 100,
                reference=reference, reference_label=f'FEniCSx level 2 ({problem.physics_sha256})',
                study=study_id, attention=candidate['attention'],
                deepxde_model=effective.get('deepxde_model', 'PFNN'),
                stop_callback=checkpoint_score if prune else None, full_audit=full_audit)
            evaluation = result.get('evaluation', {})
            variability_error_pct = None
            pred_std = truth_std = None
            if result['status'] == 'completed':
                field = np.load(output.joinpath('evaluation.npz'))
                weights = np.asarray(field['weights'], dtype=float)
                predicted = np.asarray(field['T'], dtype=float)
                truth = np.asarray(field['T_reference'], dtype=float)
                pred_mean = np.average(predicted, weights=weights)
                truth_mean = np.average(truth, weights=weights)
                pred_std = float(np.sqrt(np.average((predicted - pred_mean) ** 2, weights=weights)))
                truth_std = float(np.sqrt(np.average((truth - truth_mean) ** 2, weights=weights)))
                rise = max(float(np.max(truth)) - problem.T_level, 1e-12)
                variability_error_pct = 100. * abs(pred_std - truth_std) / rise
            case_rows.append(dict(case=case, seed=seed, status=result['status'], elapsed_s=result['elapsed_s'],
                quality_score=quality_score(evaluation, variability_error_pct, include_hotspot=case != 'annulus'), nrmse_pct=evaluation.get('nrmse_pct'),
                training_adam_steps=stage_steps,
                training_lbfgs_steps=stage_lbfgs_steps,
                actual_adam_steps=max((row['step'] + 1 for row in result.get('history', []) if row.get('phase') == 'adam'), default=0),
                early_stopped=monitor['stopped'], stop_reason=monitor['reason'],
                checkpoint_count=monitor['checkpoints'], best_checkpoint_score=monitor['best_score'] if math.isfinite(monitor['best_score']) else None,
                points=stage_points,
                error_rise_pct=evaluation.get('error_rise_pct'), error_Tmax_abs_K=evaluation.get('error_Tmax_abs_K'),
                hotspot_distance_m=evaluation.get('hotspot_distance_m'), temperature_std_pred_K=pred_std,
                temperature_std_ref_K=truth_std, temperature_std_error_pct_rise=variability_error_pct,
                worst_region_rmse_rise_pct=max(evaluation.get('region_rmse_rise_pct', {}).values(), default=None),
                balance_global_pct=evaluation.get('balance', {}).get('global_derived_pct'),
                acceptance=evaluation.get('acceptance', {}).get('state'), physics_sha256=problem.physics_sha256,
                fem_reference=_reference_path_record(reference_path), error=(result.get('error') or {}).get('message')))
    finite_scores = [row['quality_score'] for row in case_rows if math.isfinite(row['quality_score'])]
    aggregate_score = max(finite_scores) if len(finite_scores) == len(case_rows) else math.inf
    return dict(trial=candidate['_trial'], config={key: value for key, value in candidate.items() if not key.startswith('_')},
        effective_config=effective,
        cases=case_rows, quality_score=aggregate_score, elapsed_s=sum(row['elapsed_s'] for row in case_rows),
        complete=len(finite_scores) == len(case_rows))


def run_optimization(output_root, references_root, architectures=DEFAULT_ARCHITECTURES, cases=DEFAULT_CASES,
    n_trials=12, seed=20261004, screen_cases=SCREEN_CASES, screen_seeds=(0,),
    finalist_seeds=DEFAULT_FINALIST_SEEDS, finalists=DEFAULT_FINALISTS,
    promoted_candidates=PROMOTED_CANDIDATES, study_id='hparam_optimization_20261004_staged', resume=False):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError('Install the adaptive-search extra with: pip install -e ".[optimization]"') from exc
    output_root, references_root = Path(output_root), Path(references_root)
    unknown_cases = set(cases) - set(CASE_DEFINITIONS)
    if unknown_cases:
        raise ValueError(f'Unknown cases: {sorted(unknown_cases)}')
    unknown_screen_cases = set(screen_cases) - set(cases)
    if unknown_screen_cases:
        raise ValueError(f'Screen cases must be in the full case suite: {sorted(unknown_screen_cases)}')
    if not screen_seeds or not finalist_seeds or finalists < 1 or promoted_candidates < 1 or n_trials < 1:
        raise ValueError('Each optimization stage needs at least one seed and positive candidate counts')
    output_root.mkdir(parents=True, exist_ok=True)
    case_index = {case: index for index, case in enumerate(cases)}
    results = {}
    for architecture in architectures:
        model_root = output_root.joinpath(architecture)
        summary_path = model_root.joinpath('summary.json')
        if summary_path.is_file():
            if not resume:
                raise FileExistsError(f'Architecture output already exists: {model_root}; use --resume to continue')
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            expected = dict(study_id=study_id, architecture=architecture, cases=list(cases),
                screen_cases=list(screen_cases), screen_seeds=list(screen_seeds),
                finalist_seeds=list(finalist_seeds), search_seed=seed, n_trials=n_trials)
            if any(summary.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Existing summary is incompatible with requested resume options: {summary_path}')
            results[architecture] = summary
            print(architecture, 'already complete; reusing summary', flush=True)
            continue

        resuming_architecture = model_root.exists()
        if resuming_architecture and not resume:
            raise FileExistsError(f'Architecture output already exists: {model_root}; use --resume to continue')
        model_root.mkdir(parents=True, exist_ok=True)
        rows = []
        promoted_rows = []
        if resuming_architecture:
            rows = [_load_candidate_stage(architecture, model_root, 'screen', trial_number,
                screen_cases, screen_seeds, SMOKE_STEPS, study_id) for trial_number in range(n_trials)]
            screen_by_trial = {row['trial']: row for row in rows}
            refine_root = model_root.joinpath('refine')
            refine_trial_dirs = sorted(refine_root.glob('trial_*'))
            if len(refine_trial_dirs) != promoted_candidates:
                raise RuntimeError(f'Resume needs all {promoted_candidates} refinement trials complete in {refine_root}')
            for trial_dir in refine_trial_dirs:
                trial_number = int(trial_dir.name.removeprefix('trial_'))
                promoted = _load_candidate_stage(architecture, model_root, 'refine', trial_number,
                    screen_cases, screen_seeds, REFINE_STEPS, study_id)
                promoted['screen_quality_score'] = screen_by_trial[trial_number]['quality_score']
                promoted_rows.append(promoted)
            print(architecture, 'recovered screen/refine artifacts', flush=True)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(8, max(4, n_trials // 2)),
                multivariate=True, group=True)
            study = optuna.create_study(direction='minimize', sampler=sampler,
                study_name=f'{study_id}_{architecture}')
            baseline = _normalize_candidate(TPE_BASELINE, architecture)
            priors = [baseline] + [_normalize_candidate(prior, architecture) for prior in M0_TRANSFERRED_PRIORS]
            for prior in priors[:n_trials]:
                if architecture == 'DeepXDE' and prior.get('deepxde_model') is None:
                    prior['deepxde_model'] = 'PFNN'
                if architecture != 'DeepXDE':
                    prior.pop('deepxde_model', None)
                enqueue = dict(prior)
                if architecture == 'DeepXDE':
                    enqueue.pop('weights.constitutive', None)
                if enqueue.get('attention'):
                    enqueue.pop('lbfgs_steps', None)
                study.enqueue_trial(enqueue)

        def objective(trial):
            config = _normalize_candidate(_suggest_candidate(trial, architecture), architecture)
            candidate = dict(config, _trial=trial.number)
            row = _evaluate_candidate(architecture, candidate, screen_cases, references_root, model_root,
                screen_seeds, case_index, study_id, 'screen', adam_steps=SMOKE_STEPS,
                full_audit=False, prune=True)
            rows.append(row)
            trial.set_user_attr('quality_score', row['quality_score'])
            trial.set_user_attr('complete', row['complete'])
            score = row['quality_score'] if math.isfinite(row['quality_score']) else 1e12
            print(architecture, 'trial', trial.number, f"quality={score:.4g}",
                f"elapsed_s={row['elapsed_s']:.1f}", flush=True)
            return score

        if not resuming_architecture:
            study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
            promoted_screen_rows = sorted((row for row in rows if row['complete']),
                key=lambda row: (row['quality_score'], row['elapsed_s']))[:promoted_candidates]
            for screen_row in promoted_screen_rows:
                candidate = dict(screen_row['config'], _trial=screen_row['trial'])
                promoted = _evaluate_candidate(architecture, candidate, screen_cases, references_root, model_root,
                    screen_seeds, case_index, study_id, 'refine', adam_steps=REFINE_STEPS, full_audit=False)
                promoted['screen_quality_score'] = screen_row['quality_score']
                promoted_rows.append(promoted)
                print(architecture, 'promoted', promoted['trial'],
                    f"quality={promoted['quality_score']:.4g}", flush=True)
        finalist_screen_rows = sorted((row for row in promoted_rows if row['complete']),
            key=lambda row: (row['quality_score'], row['elapsed_s']))[:finalists]
        finalist_rows = []
        for promoted in finalist_screen_rows:
            candidate = dict(promoted['config'], _trial=promoted['trial'])
            finalist_options = dict(adam_steps=REFINE_STEPS, full_audit=True)
            if resume:
                finalist_options['resume'] = True
            finalist = _evaluate_candidate(architecture, candidate, cases, references_root, model_root,
                finalist_seeds, case_index, study_id, 'finalist', **finalist_options)
            finalist['refine_quality_score'] = promoted['quality_score']
            finalist_rows.append(finalist)
            print(architecture, 'finalist', finalist['trial'],
                f"robust_quality={finalist['quality_score']:.4g}", flush=True)
        best, reason = choose_candidate(finalist_rows)
        default_recommendation = 'no finalist completed the full evaluation'
        if best is not None:
            default_recommendation = 'validated' if best['complete'] and best['quality_score'] <= 1.0 else 'provisional; 1% accuracy/physics gates not met'
        summary = dict(study_id=study_id, architecture=architecture, cases=list(cases),
            screen_cases=list(screen_cases), screen_seeds=list(screen_seeds),
            finalist_seeds=list(finalist_seeds), finalists=len(finalist_rows), promoted_candidates=len(promoted_rows),
            smoke_steps=SMOKE_STEPS, refine_steps=REFINE_STEPS,
            prune_min_steps=PRUNE_MIN_STEPS, prune_patience=PRUNE_PATIENCE,
            checkpoint_interval=SMOKE_LOG_EVERY, smoke_point_multiplier_max=SMOKE_POINT_MULTIPLIER_MAX,
            search_method='Optuna TPE Bayesian optimization', search_seed=seed, n_trials=n_trials,
            base_points=BASE_POINTS, fem_level=2, target_accuracy_pct=ACCURACY_TARGET_PCT,
            selection_metric='worst normalized NRMSE, Tmax-rise, region error and physics errors; fastest accepted, else accuracy/runtime tradeoff',
            criteria_status='proposed gates from pinn_v2.evaluation.metrics; not independently confirmed',
            selected_trial=None if best is None else best['trial'], selection_reason=reason,
            selected_requested_config=None if best is None else best['config'],
            selected_config=None if best is None else best['effective_config'], rows=rows,
            promoted_rows=promoted_rows, finalist_rows=finalist_rows,
            default_recommendation=default_recommendation)
        if resuming_architecture:
            summary['resumed_from_artifacts'] = True
            summary['screen_point_multiplier_note'] = (
                'Recovered screen rows record the effective 1x smoke sampling; original TPE point multipliers '
                'were not persisted for unpromoted trials.')
        write_json_atomic(model_root.joinpath('summary.json'), summary)
        write_manifest(model_root, build_manifest('fem_hparam_optimization',
            [sys.executable, '-m', 'pinn_v2.experiments.fem_hparam_optimization', '--study-id', study_id],
            seeds=dict(search=seed, screening=list(screen_seeds), finalists=list(finalist_seeds),
                sampling='20261002 + case_index + 1000*seed'),
            extra=dict(architecture=architecture, cases=list(cases), n_trials=n_trials,
                screen_cases=list(screen_cases), screening_seeds=list(screen_seeds),
                finalist_seeds=list(finalist_seeds), promoted_candidates=len(promoted_rows),
                finalists=len(finalist_rows), smoke_steps=SMOKE_STEPS, refine_steps=REFINE_STEPS,
                prune_min_steps=PRUNE_MIN_STEPS, prune_patience=PRUNE_PATIENCE,
                checkpoint_interval=SMOKE_LOG_EVERY, fem_level=2,
                fem_labels_used_for_training=False, references_root=str(references_root))))
        write_report(model_root, summary)
        results[architecture] = summary
    write_executive_summary(output_root, results, n_trials, seed, cases)
    return results


def write_report(output, summary):
    output = Path(output)
    _write_optimization_plot(output, summary)
    is_deepxde = summary['architecture'] == 'DeepXDE'
    search_range = ('Search ranges: width 16-96; depth 2-6; log-uniform learning rate 1e-4 to 1e-2; tanh/silu; 400-step smoke, then 800-step promoted/finalist runs; L-BFGS 0/100/300; internal network PFNN/FNN.'
        if is_deepxde else 'Search ranges: width 16-96 (FBPINN 8-24); depth 2-6 (FBPINN 2-4); log-uniform learning rate 1e-4 to 1e-2; tanh/silu; 400-step smoke, then 800-step promoted/finalist runs; optional L-BFGS 0/100/300 when attention is off; attention none/RBA/SA.')
    loss_range = ('Loss-weight levels: PDE and energy = 0, 0.01, 0.1, 1, 10, 100, 1000; constitutive = 0 (strong-form network has no auxiliary flux); interface-temperature, interface-flux, Dirichlet, Neumann, Robin = 0.1, 1, 10, 100, 1000.'
        if is_deepxde else 'Loss-weight levels: PDE, constitutive, energy = 0, 0.01, 0.1, 1, 10, 100, 1000; interface-temperature, interface-flux, Dirichlet, Neumann, Robin = 0.1, 1, 10, 100, 1000.')
    lines = [f"# PINN hyperparameter optimization: {summary['architecture']}", '',
        '## Methodology', '',
        f"The search uses three fidelities. Optuna TPE screens {summary['n_trials']} configurations on {len(summary['screen_cases'])} representative cases, capped at {summary['smoke_steps']} Adam steps. Checkpoints every {summary['checkpoint_interval']} steps prune non-finite candidates or those without material improvement for {summary['prune_patience']} checkpoints after step {summary['prune_min_steps']}. The top {summary['promoted_candidates']} candidates are promoted to {summary['refine_steps']} steps. The top {summary['finalists']} are evaluated on the complete suite with {len(summary['finalist_seeds'])} independent seeds and full physics audits. FEM/analytic reference fields are evaluation-only and never enter the training loss.", '',
        f"Coverage: {', '.join(summary['cases'])} ({len(summary['cases'])} cases; all seven MMS variants, annulus, both ARAS cases, and the discrete-layer XLPE cable). Reference resolution: level {summary['fem_level']} for all cases.",
        f"Architecture: {summary['architecture']}; screening trials: {summary['n_trials']}; Optuna seed: {summary['search_seed']}; screening cases/seeds: {summary['screen_cases']} / {summary['screen_seeds']}; finalist seeds: {summary['finalist_seeds']}.",
        f"Sampling: base counts {json.dumps(summary['base_points'], sort_keys=True)}; smoke caps the point multiplier at {SMOKE_POINT_MULTIPLIER_MAX:g}x, while promoted runs use 1x, 1.5x, or 2x. Target: NRMSE, Tmax-rise error, and worst-region RMSE each <= {summary['target_accuracy_pct']:.1f}% of the FEM temperature rise; physics gates also constrain balance, interfaces, and variability.",
        search_range,
        loss_range + ' The baseline and two normalized M0-derived priors are enqueued first.', '',
        f"Selection: {summary['selection_reason']}. {summary['selection_metric']}.", '',
        '## Trial Results', '',
        'The first plot shows worst-case NRMSE across the screening suite against the 1% target, and how trial outcomes vary with PDE weight and point density.', '',
        '![TPE search results](optimization.png)', '',
        '| trial | requested config | effective config | worst normalized error | total runtime (s) | complete |',
        '|---:|---|---|---:|---:|:---:|']
    for row in sorted(summary['rows'], key=lambda item: (not item['complete'], item['quality_score'], item['elapsed_s'])):
        score = 'inf' if not math.isfinite(row['quality_score']) else f"{row['quality_score']:.4g}"
        effective = row.get('effective_config', _effective_config(row['config']))
        lines.append(f"| {row['trial']} | `{json.dumps(row['config'], sort_keys=True)}` | `{json.dumps(effective, sort_keys=True)}` | "
            f"{score} | {row['elapsed_s']:.1f} | {row['complete']} |")
    early_stopped = sum(case.get('early_stopped', False) for row in summary['rows'] for case in row['cases'])
    smoke_runs = sum(len(row['cases']) for row in summary['rows'])
    lines.append(f'\nSmoke runs stopped early: {early_stopped}/{smoke_runs}.')
    lines += ['', '## Case Metrics', '', '| trial | case | seed | NRMSE % | Tmax error K | hotspot error m | std error / rise % | worst-region % | balance % | runtime (s) | verdict |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in summary['rows']:
        for case in row['cases']:
            fmt = lambda value: '' if value is None else f'{value:.4g}'
            lines.append(f"| {row['trial']} | {case['case']} | {case['seed']} | {fmt(case['nrmse_pct'])} | {fmt(case['error_Tmax_abs_K'])} | "
                f"{fmt(case['hotspot_distance_m'])} | {fmt(case['temperature_std_error_pct_rise'])} | {fmt(case['worst_region_rmse_rise_pct'])} | "
                f"{fmt(case['balance_global_pct'])} | {case['elapsed_s']:.1f} | {case['acceptance'] or case['status']} |")
    lines += ['', '## Promotion', '',
        f"{len(summary.get('promoted_rows', []))} candidates survived the smoke screen and were trained to {summary['refine_steps']} steps with audit disabled.", '',
        '| trial | smoke score | refined score | runtime (s) | complete |',
        '|---:|---:|---:|---:|:---:|']
    for row in summary.get('promoted_rows', []):
        lines.append(f"| {row['trial']} | {row['screen_quality_score']:.4g} | {row['quality_score']:.4g} | {row['elapsed_s']:.1f} | {row['complete']} |")
    lines += ['', '## All-suite Finalist Comparison', '',
        f"Top {summary['finalists']} promoted candidates were evaluated across all cases with five independent seeds and full physics audits. Selection uses the worst normalized case/seed score.", '',
        '| trial | refined score | robust score | worst NRMSE % | worst Tmax error % | mean NRMSE % | SD NRMSE % | complete |',
        '|---:|---:|---:|---:|---:|---:|---:|:---:|']
    for row in summary['finalist_rows']:
        nrmse_values = [case['nrmse_pct'] for case in row['cases'] if case['nrmse_pct'] is not None]
        worst_nrmse = max(nrmse_values, default=math.inf)
        mean_nrmse = float(np.mean(nrmse_values)) if nrmse_values else math.inf
        sd_nrmse = float(np.std(nrmse_values, ddof=1)) if len(nrmse_values) > 1 else 0.
        worst_tmax = max((case['error_rise_pct'] or 0. for case in row['cases']), default=math.inf)
        lines.append(f"| {row['trial']} | {row['refine_quality_score']:.4g} | {row['quality_score']:.4g} | "
            f"{worst_nrmse:.4g} | {worst_tmax:.4g} | {mean_nrmse:.4g} | {sd_nrmse:.4g} | {row['complete']} |")
    for row in summary['finalist_rows']:
        lines += ['', f"### Finalist trial {row['trial']}: full-suite seed statistics", '',
            '| case | seeds | NRMSE mean ± SD (%) | NRMSE max (%) | Tmax error max (%) | worst-region max (%) |',
            '|---|---:|---:|---:|---:|---:|']
        for case_name in summary['cases']:
            case_rows = [case for case in row['cases'] if case['case'] == case_name]
            values = [case['nrmse_pct'] for case in case_rows if case['nrmse_pct'] is not None]
            mean_value = float(np.mean(values)) if values else math.inf
            sd_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
            max_value = max(values, default=math.inf)
            max_tmax = max((case['error_rise_pct'] or 0. for case in case_rows), default=math.inf)
            max_region = max((case['worst_region_rmse_rise_pct'] or 0. for case in case_rows), default=math.inf)
            lines.append(f'| {case_name} | {len(values)} | {mean_value:.4g} ± {sd_value:.3g} | {max_value:.4g} | {max_tmax:.4g} | {max_region:.4g} |')
    if summary['selected_config'] is not None:
        requested = summary.get('selected_requested_config', summary['selected_config'])
        lines += ['', '## Selected settings', '', f"Trial {summary['selected_trial']} requested: `{json.dumps(requested, sort_keys=True)}`.",
            f"Effective during training: `{json.dumps(summary['selected_config'], sort_keys=True)}`.",
            f"Default recommendation status: **{summary['default_recommendation']}**.",
            'Promote this as a default only if the finalist evaluation meets the 1% accuracy target and physical gates; otherwise treat it as a provisional best-observed configuration.']
    output.joinpath('report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_optimization_plot(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    trials = [row for row in summary['rows'] if row['complete']]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    trial_numbers = [row['trial'] for row in trials]
    worst_nrmse = [max((case['nrmse_pct'] or 0.) for case in row['cases']) for row in trials]
    pde_weights = [row['config']['weights.pde'] for row in trials]
    point_multipliers = [row['config']['point_multiplier'] for row in trials]
    scatter = axes[0].scatter(trial_numbers, worst_nrmse, c=point_multipliers, cmap='viridis', s=50)
    axes[0].axhline(summary['target_accuracy_pct'], color='firebrick', linestyle='--', label='1% target')
    axes[0].set(xlabel='TPE trial', ylabel='Worst case NRMSE (%)', title='Accuracy by trial')
    axes[0].legend()
    figure.colorbar(scatter, ax=axes[0], label='Point multiplier')
    axes[1].scatter(pde_weights, worst_nrmse, c=trial_numbers, cmap='plasma', s=55)
    axes[1].set_xscale('symlog', linthresh=0.01)
    axes[1].axhline(summary['target_accuracy_pct'], color='firebrick', linestyle='--')
    axes[1].set(xlabel='PDE loss weight', ylabel='Worst case NRMSE (%)', title='PDE weight sensitivity')
    selected = next((row for row in summary['finalist_rows'] if row['trial'] == summary['selected_trial']), None)
    if selected:
        seed_scores = [[case['nrmse_pct'] for case in selected['cases']
            if case['case'] == case_name and case['nrmse_pct'] is not None] for case_name in summary['cases']]
        axes[2].boxplot(seed_scores)
        axes[2].set_xticks(range(1, len(summary['cases']) + 1), summary['cases'])
        axes[2].axhline(summary['target_accuracy_pct'], color='firebrick', linestyle='--')
        axes[2].tick_params(axis='x', rotation=60, labelsize=7)
    axes[2].set(ylabel='NRMSE across finalist seeds (%)', title='Finalist full-suite evaluation')
    figure.savefig(Path(output).joinpath('optimization.png'), dpi=160)
    plt.close(figure)


def write_executive_summary(output_root, results, n_trials, seed, cases):
    lines = [
        '# Resumen ejecutivo: optimización adaptativa de PINN', '',
        '## Metodología', '',
        f'- **Búsqueda:** TPE de Optuna con {n_trials} ensayos por arquitectura y semilla {seed}; se encolan primero el baseline y dos priors derivados de M0.',
        '- **Objetivo primario:** minimizar el peor error normalizado entre NRMSE de campo, error de aumento de Tmax '
        'y RMSE por región. La meta es ≤1% en cada métrica de precisión.',
        '- **Restricciones físicas adicionales:** variabilidad térmica ≤5% del aumento de temperatura; '
        'balance energético ≤2%; salto de temperatura de interfaz ≤0,1 K; salto RMS de flujo ≤1%. '
        'La ubicación de hotspot se informa, pero no ordena el caso anular simétrico.',
        '- **Selección:** candidato más rápido que cumple todas las metas; si ninguno cumple, menor score de precisión/física '
        'con runtime como desempate. Los umbrales son metas de investigación, no validación de ingeniería.',
        f'- **Fidelidad y semillas:** smoke de hasta {SMOKE_STEPS} pasos con pruning; {PROMOTED_CANDIDATES} promociones a {REFINE_STEPS} pasos; {DEFAULT_FINALISTS} finalistas en {len(cases)} casos con cinco seeds independientes y auditoría completa.',
        '- **Anti-fuga:** FEM se usa exclusivamente para evaluación; sus temperaturas no entran en la función de pérdida ni el entrenamiento.',
        '- **Muestreo:** base de 256 puntos interiores, 128 por capa y 96 por cara/frontera; smoke limita el multiplicador a 1× y TPE explora 1×, 1.5× y 2× después. '
        'La comparación FEM/PINN usa las mallas de evaluación fijas y específicas de cada caso.',
        '- **Entrenamiento:** Adam hasta 400 pasos en smoke y 800 en promoción/finalistas; tasa de aprendizaje log-uniforme entre 1e-4 y 1e-2; '
        'ancho 16–96 (FBPINN 8–24), profundidad 2–6 (FBPINN 2–4), activación tanh/SiLU y atención ninguna/RBA/SA. '
        'L-BFGS se prueba con 0/100/300 pasos y se desactiva cuando la atención adaptativa hace variable la pérdida.',
        '- **Pesos de pérdida:** PDE, constitutiva y energía ∈ {0, 0.01, 0.1, 1, 10, 100, 1000}; '
        'temperatura/flujo de interfaz y Dirichlet/Neumann/Robin ∈ {0.1, 1, 10, 100, 1000}.',
        f'- **Cobertura:** {', '.join(cases)}; referencias FEniCSx nivel 2. Se evalúan M0, M1, M2, CUSP, DCSNN, FBPINN y DeepXDE.',
        '', '## Resultados', '',
        '| Arquitectura | Ensayo elegido | Peor NRMSE (%) | Peor error Tmax (%) | NRMSE medio ± SD (%) | Score robusto | Tiempo finalista (s) | Meta ≤1% en todos los casos |',
        '|---|---:|---:|---:|---:|---:|---:|:---:|',
    ]
    for architecture, summary in results.items():
        finalist = next((row for row in summary['finalist_rows'] if row['trial'] == summary['selected_trial']), None)
        if finalist is None:
            lines.append(f'| {architecture} | - | - | - | - | - | - | - | No |')
            continue
        nrmse_values = [case['nrmse_pct'] for case in finalist['cases'] if case.get('nrmse_pct') is not None]
        worst_nrmse = max(nrmse_values, default=math.inf)
        mean_nrmse = float(np.mean(nrmse_values)) if nrmse_values else math.inf
        sd_nrmse = float(np.std(nrmse_values, ddof=1)) if len(nrmse_values) > 1 else 0.
        worst_tmax = max((case.get('error_rise_pct') or 0. for case in finalist['cases']), default=math.inf)
        target_met = all(case.get('nrmse_pct') is not None and case['nrmse_pct'] <= ACCURACY_TARGET_PCT
            and case.get('error_rise_pct') is not None and case['error_rise_pct'] <= ACCURACY_TARGET_PCT
            and (case.get('worst_region_rmse_rise_pct') is None
                or case['worst_region_rmse_rise_pct'] <= ACCURACY_TARGET_PCT)
            and (case['case'] == 'annulus' or case.get('hotspot_distance_m') is None
                or case['hotspot_distance_m'] <= HOTSPOT_TOLERANCE_M)
            for case in finalist['cases'])
        target_met = target_met and finalist['quality_score'] <= 1.0
        lines.append(f"| {architecture} | {summary['selected_trial']} | {worst_nrmse:.4g} | {worst_tmax:.4g} "
            f"| {mean_nrmse:.4g} ± {sd_nrmse:.3g} | {finalist['quality_score']:.4g} | {finalist['elapsed_s']:.1f} | {'Sí' if target_met else 'No'} |")
    lines += ['', 'Cada arquitectura incluye un `report.md`, `summary.json`, `optimization.png` y artefactos de ensayo con configuración, semillas, '
        'métricas, tiempos y manifiestos de procedencia. Los notebooks de la campaña muestran la tabla y las mismas visualizaciones.', '']
    Path(output_root).joinpath('RESUMEN_EJECUTIVO_OPTIMIZACION_PINNS.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-id', default='hparam_optimization_20261004_staged')
    parser.add_argument('--output', type=Path, default=ROOT.joinpath('results', 'hparam_optimization_20261004_staged'))
    parser.add_argument('--references', type=Path, default=ROOT.joinpath('results', 'hparam_optimization_20261003_adaptive', 'shared_references'))
    parser.add_argument('--architectures', nargs='*', default=list(DEFAULT_ARCHITECTURES))
    parser.add_argument('--cases', nargs='*', default=list(DEFAULT_CASES))
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--screen-cases', nargs='*', default=list(SCREEN_CASES))
    parser.add_argument('--screen-seeds', nargs='*', type=int, default=[0])
    parser.add_argument('--finalist-seeds', nargs='*', type=int, default=list(DEFAULT_FINALIST_SEEDS))
    parser.add_argument('--finalists', type=int, default=DEFAULT_FINALISTS)
    parser.add_argument('--promoted-candidates', type=int, default=PROMOTED_CANDIDATES)
    parser.add_argument('--resume', action='store_true', help='Reuse completed architecture/stage artifacts and continue missing finalist runs')
    args = parser.parse_args()
    run_optimization(args.output, args.references, tuple(args.architectures), tuple(args.cases), args.trials,
        args.seed, tuple(args.screen_cases), tuple(args.screen_seeds), tuple(args.finalist_seeds),
        args.finalists, args.promoted_candidates, args.study_id, args.resume)


if __name__ == '__main__':
    main()