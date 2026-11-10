"""Transfer the MMS exact-PINN recipe to selected cable cases, with compatibility preflight."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.manifest.run_manifest import atomic_replace, build_manifest, replace_json_atomic, write_json_atomic, write_manifest
from pinn_v2.solvers.fem import adapter
from pinn_v2.solvers.pinn import trainer
from pinn_v2.solvers.pinn.networks import build_model
from pinn_v2.solvers.pinn.problem import load_problem

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
CASE_ROOT = ROOT / 'cases' / 'v2'
RECIPE = PACKAGE_ROOT / 'configs' / 'train_mms_exact_repro_20261020.json'
CASES = ('aras_flat', 'kim_v5_summer', 'xlpe_backfill', 'xlpe_single', 'aras_single')
MODELS = ('M0_EXACT', 'CUSP_EXACT', 'DCSNN_EXACT', 'FBPINN_EXACT',
    'DeepXDE_PFNN_EXACT', 'DeepXDE_FNN_EXACT')


def case_definitions(case_names=CASES):
    definitions = {}
    for case in case_names:
        matches = list(CASE_ROOT.rglob(f'{case}.json'))
        if len(matches) != 1:
            raise ValueError(f'Expected one v2 definition for {case}; found {len(matches)}')
        definitions[case] = matches[0]
    return definitions


def campaign_settings():
    recipe = json.loads(RECIPE.read_text(encoding='utf-8'))
    settings = {key: value for key, value in recipe.items() if key not in ('definition', 'model')}
    settings.update(accuracy_stop=dict(error_Tmax_abs_K=.01, nrmse_pct=.1, balance_pct=.5, every=100),
        extra_lbfgs_steps=5, fem_level=1, fem_degree=2, fem_node_multiplier=1.0,
        recipe_origin=RECIPE.relative_to(PACKAGE_ROOT).as_posix())
    return settings


def preflight(definitions, settings, model_names=MODELS):
    report = {}
    for case, definition in definitions.items():
        problem = load_problem(definition)
        models = {}
        for architecture in model_names:
            try:
                build_model(problem, architecture, width=settings['width'], depth=settings['depth'],
                    activation=settings['activation'], input_normalization=settings['input_normalization'],
                    trace_parameterization=settings['trace_parameterization'],
                    soil_envelope_gain=settings['soil_envelope_gain'])
                models[architecture] = dict(status='supported', reason=None)
            except Exception as exc:
                models[architecture] = dict(status='incompatible', reason=f'{type(exc).__name__}: {exc}')
        supported = [name for name, row in models.items() if row['status'] == 'supported']
        report[case] = dict(definition=definition.relative_to(PACKAGE_ROOT).as_posix(),
            physics_sha256=problem.physics_sha256, regions=len(problem.regions),
            status='pending' if supported else 'blocked', compatible_models=supported, models=models,
            runs={})
    return report


def run_regression_gate(output):
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', DDE_BACKEND='pytorch')
    command = [sys.executable, '-m', 'pytest', 'tests', '-m', 'not slow and not fem',
        '-k', 'not test_provenance_record_matches_untouched_originals', '-q', '--tb=short',
        f'--junitxml={output / "trainer_regression_gate.xml"}']
    result = subprocess.run(command, cwd=PACKAGE_ROOT, env=environment, capture_output=True, text=True)
    (output / 'trainer_regression_output.txt').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    write_json_atomic(output / 'trainer_regression_status.json',
        dict(returncode=result.returncode, passed=result.returncode == 0, command=command))
    if result.returncode:
        raise RuntimeError('Trainer regression gate failed; no campaign training was started')


def export_reports(output, settings, cases):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    output = Path(output)
    summary = dict(campaign_id=output.name, settings=settings, cases=cases,
        training_status='complete' if all(item['status'] in ('completed', 'blocked') for item in cases.values()) else 'in_progress',
        requested_runs=sum(len(item['models']) for item in cases.values()),
        completed_runs=sum(row.get('status') == 'completed' for item in cases.values() for row in item['runs'].values()),
        blocked_runs=sum(model['status'] == 'incompatible' for item in cases.values() for model in item['models'].values()))
    replace_json_atomic(output / 'summary.json', summary)
    book = Workbook()
    config_sheet = book.active
    config_sheet.title = 'Settings'
    config_sheet.append(['Parameter', 'Value'])
    for key, value in settings.items():
        config_sheet.append([key, json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value])
    preflight_sheet = book.create_sheet('Preflight')
    preflight_sheet.append(['Case', 'Model', 'Compatibility', 'Reason'])
    columns = ['Model', 'Status', 'NRMSE FEM %', 'Tmax error K', 'Global balance %',
        'Inner boundary max error K', 'Elapsed s', 'L-BFGS calls', 'Extra calls', 'Targets met', 'Raw gates', 'Result path']
    for case, item in cases.items():
        sheet = book.create_sheet(case[:31])
        sheet.append(['Case', case])
        sheet.append(['Physics SHA256', item['physics_sha256']])
        sheet.append(['FEM reference', item.get('fem_reference', '')])
        sheet.append([])
        sheet.append(columns)
        for model, compatibility in item['models'].items():
            preflight_sheet.append([case, model, compatibility['status'], compatibility['reason'] or ''])
            run = item['runs'].get(model, {})
            evaluation = run.get('evaluation', {})
            accuracy = run.get('accuracy_stop', {})
            balance = evaluation.get('balance', {})
            gates = evaluation.get('acceptance', {}).get('raw_gates', {})
            sheet.append([model, run.get('status', compatibility['status']), evaluation.get('nrmse_pct'),
                evaluation.get('error_Tmax_abs_K'), balance.get('global_derived_pct'),
                evaluation.get('annulus_boundary', {}).get('inner', {}).get('max_error_K'),
                run.get('elapsed_s'), accuracy.get('lbfgs_outer_calls'), accuracy.get('extra_lbfgs_outer_calls'),
                accuracy.get('triggered'), all(gates.values()) if gates else None, run.get('result_path', '')])
    for sheet in book:
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions if sheet.max_row > 1 else None
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='424242')
        for index in range(1, sheet.max_column + 1):
            sheet.column_dimensions[get_column_letter(index)].width = min(max(16, len(str(sheet.cell(1, index).value or '')) + 3), 46)
    temporary = output / 'campaign.tmp.xlsx'
    book.save(temporary)
    atomic_replace(temporary, output / 'comparison.xlsx')

    lines = ['# Exact PINN cable transfer campaign', '',
        'MMS recipe: 400 Adam + 5 external L-BFGS calls, up to 5 extra calls if the strict joint gates are not met.',
        'PDE-only objective; FEM P2 level 1 is used only for evaluation and stopping, never in the loss.',
        'The campaign includes preflight status for all five requested cases. Unsupported cases are not trained or silently changed.', '',
        f"Runs complete: {summary['completed_runs']}; incompatible model/case pairs: {summary['blocked_runs']}; requested pairs: {summary['requested_runs']}.", '']
    for case, item in cases.items():
        lines.extend([f'## {case}', '', f"Status: **{item['status']}**; regions: {item['regions']}; physics SHA256: `{item['physics_sha256']}`.", ''])
        if item.get('compatibility_reason'):
            lines.extend([f"Compatibility note: {item['compatibility_reason']}", ''])
        reasons = sorted({model['reason'] for model in item['models'].values() if model['reason']})
        if reasons:
            lines.extend(['Preflight blockers: ' + '; '.join(reasons), ''])
        lines.extend(['| Model | Compatibility | Status | NRMSE FEM % | Tmax error K | Balance % | Time s |',
            '|---|---|---|---:|---:|---:|---:|'])
        for model, compatibility in item['models'].items():
            run = item['runs'].get(model, {})
            evaluation = run.get('evaluation', {})
            balance = evaluation.get('balance', {}).get('global_derived_pct')
            lines.append(f"| {model} | {compatibility['status']} | {run.get('status', '-') } | "
                f"{evaluation.get('nrmse_pct', '')} | {evaluation.get('error_Tmax_abs_K', '')} | "
                f"{balance if balance is not None else ''} | {run.get('elapsed_s', '')} |")
        lines.extend(['', ''])
    lines.extend(['Artifacts: `summary.json`, `comparison.xlsx`, and per-run checkpoints/results.', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def run_campaign(output, preflight_only=False, validate_first=False, case_names=CASES, model_names=MODELS):
    output = Path(output).resolve()
    if output.exists() and not (output / 'campaign_config.json').is_file() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is non-empty and has no campaign config: {output}')
    output.mkdir(parents=True, exist_ok=True)
    definitions = case_definitions(case_names)
    settings = campaign_settings()
    config = dict(requested_cases=list(case_names), models=list(model_names), settings=settings)
    config_path = output / 'campaign_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Campaign config differs; use a new output directory')
    if not config_path.exists():
        write_json_atomic(config_path, config)
    manifest_path = output / 'manifest.json'
    if not manifest_path.exists():
        manifest = build_manifest('phase7_exact_cable_transfer_campaign',
            [sys.executable, '-m', 'pinn_v2.experiments.exact_cable_transfer_campaign',
                '--output', str(output)],
            seeds=dict(init=settings['seed'], sampling=settings['sampling_seed']),
            extra=dict(requested_cases=list(case_names), models=list(model_names),
                fem_labels_used_for_training=False, reference_level=1))
        write_manifest(output, manifest)
    cases = preflight(definitions, settings, model_names)
    state_path = output / 'summary.json'
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding='utf-8'))
        for case, item in old['cases'].items():
            if case in cases:
                cases[case]['runs'] = item.get('runs', {})
                cases[case]['fem_reference'] = item.get('fem_reference')
                if item.get('status') == 'completed':
                    cases[case]['status'] = 'completed'
    export_reports(output, settings, cases)
    if preflight_only:
        return cases
    if validate_first and not (output / 'trainer_regression_status.json').is_file():
        run_regression_gate(output)
    import torch
    torch.set_num_threads(1)

    for case, item in cases.items():
        supported = item['compatible_models']
        if not supported:
            item['status'] = 'blocked'
            item['compatibility_reason'] = next(iter(item['models'].values()))['reason']
            export_reports(output, settings, cases)
            continue
        problem = load_problem(definitions[case])
        fem_dir = output / case / 'fem'
        fem_npz = fem_dir / 'fem_l1.npz'
        fem_json = fem_dir / 'fem_l1.json'
        if not fem_npz.is_file() or not fem_json.is_file():
            if fem_dir.exists() and any(fem_dir.iterdir()):
                raise FileExistsError(f'Incomplete FEM output preserved: {fem_dir}')
            adapter.run_cable_fem(fem_dir, (1,), definition=definitions[case], node_multiplier=1.0)
        fem_record = json.loads(fem_json.read_text(encoding='utf-8'))
        if fem_record.get('physics_sha256') != problem.physics_sha256:
            raise ValueError(f'FEM physics hash differs for {case}')
        reference = load_reference_npz(fem_npz, problem)
        item['fem_reference'] = fem_npz.relative_to(output).as_posix()
        item['fem_dofs'] = fem_record.get('ndofs')
        item['status'] = 'running'
        export_reports(output, settings, cases)

        for architecture in supported:
            if architecture in item['runs'] and item['runs'][architecture].get('status') == 'completed':
                continue
            folder = output / case / architecture
            result_path = folder / 'result.json'
            if result_path.is_file():
                record = json.loads(result_path.read_text(encoding='utf-8'))
            elif folder.exists():
                item['runs'][architecture] = dict(status='incomplete_preserved', result_path=str(result_path))
                export_reports(output, settings, cases)
                continue
            else:
                run_settings = dict(settings)
                run_settings.pop('fem_level')
                run_settings.pop('fem_degree')
                run_settings.pop('fem_node_multiplier')
                run_settings.pop('recipe_origin')
                record = trainer.run(definitions[case], architecture, folder,
                    init_seed=run_settings.pop('seed'), sampling_seed=run_settings.pop('sampling_seed'),
                    reference=reference, reference_label=f'FEM P2 level 1, {case} (evaluation only)',
                    full_audit=True, study=output.name,
                    command=['exact_cable_transfer_campaign', case, architecture, str(folder)], **run_settings)
            evaluation = record.get('evaluation', {})
            accuracy = record.get('accuracy_stop', {})
            item['runs'][architecture] = dict(status=record['status'],
                elapsed_s=record.get('elapsed_s'), final_loss=record.get('final_loss'),
                evaluation=evaluation, accuracy_stop=accuracy,
                error=(record.get('error') or {}).get('message'),
                result_path=result_path.relative_to(output).as_posix())
            export_reports(output, settings, cases)
        item['status'] = 'completed' if all(item['runs'].get(name, {}).get('status') == 'completed'
            for name in supported) else 'incomplete'
        export_reports(output, settings, cases)
    return export_reports(output, settings, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--validate-first', action='store_true')
    parser.add_argument('--cases', type=lambda text: tuple(text.split(',')), default=CASES)
    parser.add_argument('--models', type=lambda text: tuple(text.split(',')), default=MODELS)
    args = parser.parse_args()
    if not args.preflight_only and not adapter.fem_available():
        raise RuntimeError('Cable campaign requires the configured WSL2 FEniCSx environment')
    run_campaign(args.output, args.preflight_only, args.validate_first, args.cases, args.models)
    print(args.output.resolve())


if __name__ == '__main__':
    main()