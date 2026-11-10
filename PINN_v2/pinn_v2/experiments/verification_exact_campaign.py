"""FEM-first exact-PINN verification campaign with incremental Markdown and Excel reports."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from pinn_v2.evaluation.metrics import error_report
from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.manifest.run_manifest import replace_json_atomic, write_json_atomic
from pinn_v2.solvers.fem import adapter
from pinn_v2.solvers.pinn.problem import load_problem

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT/'cases'/'v2'/'references'/'verification'
PACKAGE_ROOT = ROOT.parent
MODELS = ('M0_EXACT', 'CUSP_EXACT', 'DCSNN_EXACT', 'FBPINN_EXACT',
    'DeepXDE_PFNN_EXACT', 'DeepXDE_FNN_EXACT')
COLUMNS = ('model', 'status', 'termination_reason', 'Tmax_C', 'Tmax_reference_C', 'Tmax_error_K', 'NRMSE_FEM_pct',
    'max_error_FEM_K', 'global_balance_pct', 'worst_region_balance_pct', 'elapsed_s', 'elapsed_min',
    'accuracy_targets_met', 'lbfgs_outer_calls', 'extra_lbfgs_calls', 'all_raw_gates',
    'annulus_inner_boundary_max_error_K', 'annulus_outer_boundary_max_error_K',
    'annulus_boundary_points', 'NRMSE_analytic_pct', 'max_error_analytic_K', 'training_pde_points', 'interface_points',
    'boundary_points', 'collocation_total_points', 'collocation_unique_points',
    'validation_pde_points', 'evaluation_points', 'audit_pde_points', 'audit_surface_points',
    'fem_dofs', 'error', 'artifact_directory')


def collocation_counts(folder):
    path = Path(folder)/'collocation.npz'
    if not path.is_file():
        return {}
    counts = dict(training_pde_points=0, interface_points=0, boundary_points=0)
    groups = dict(interior_='training_pde_points', interface_='interface_points', boundary_='boundary_points')
    arrays = []
    with np.load(path) as cloud:
        for name in cloud.files:
            group = next((value for prefix, value in groups.items() if name.startswith(prefix)), None)
            if group is None:
                raise ValueError(f'Unrecognized collocation group {name} in {path}')
            coordinates = cloud[name]
            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise ValueError(f'Invalid collocation coordinates in {path}: {name}')
            counts[group] += len(coordinates)
            arrays.append(coordinates)
    counts['collocation_total_points'] = sum(counts.values())
    counts['collocation_unique_points'] = len(np.unique(np.vstack(arrays), axis=0)) if arrays else 0
    return counts


def case_definitions():
    return {json.loads(path.read_text(encoding='utf-8'))['id']: path
        for path in sorted(DEFINITIONS.glob('*.json'))}


def evaluate_annulus_inner_epsilon(extension_root, epsilons=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2), n_angles=256):
    """Compare saved extension fields on circles ri+epsilon, strictly inside the material."""
    import torch
    from pinn_v2.solvers.pinn.checkpoint import infer, load_checkpoint
    from pinn_v2.evaluation.metrics import error_report
    extension_root = Path(extension_root)
    rows = []
    angle = (np.arange(n_angles) + .5) * 2*np.pi/n_angles
    case = json.loads((DEFINITIONS/'annulus.json').read_text(encoding='utf-8'))['physical']
    ri, ro = case['ri'], case['ro']
    for architecture in MODELS:
        folder = extension_root/architecture
        result_path = folder/'result.json'
        if not result_path.exists():
            continue
        model, _, _ = load_checkpoint(folder/'model.pt')
        problem = model.problem
        xy_area, labels, weights = problem.evaluation()
        area_prediction = infer(model, xy_area)['T']
        area_reference = problem.reference(xy_area, labels)
        area_report = error_report(area_prediction, area_reference, xy_area, labels, weights,
            [region.name for region in problem.regions], problem.T_level, problem.conductors(),
            temperature_scale=problem.T_scale)
        reference_parameter = next(model.parameters())
        for epsilon in epsilons:
            radius = ri + epsilon
            if not ri < radius < ro:
                raise ValueError(f'epsilon={epsilon} must keep ri+epsilon inside the annulus')
            xy = np.column_stack([radius*np.cos(angle), radius*np.sin(angle)])
            points = torch.as_tensor(xy, dtype=reference_parameter.dtype,
                device=reference_parameter.device).requires_grad_(True)
            with torch.enable_grad():
                prediction, _ = model.field(problem.regions[0], points)
            predicted = prediction.detach().cpu().numpy().ravel()
            expected = problem.reference(xy)
            ring_error = predicted-expected
            union_prediction = np.concatenate([area_prediction, predicted])
            union_reference = np.concatenate([area_reference, expected])
            rows.append(dict(model=architecture, epsilon_m=epsilon, radius_m=radius,
                epsilon_over_ri=epsilon/ri, points=n_angles,
                ring_Tmax_pred_C=float(predicted.max()), ring_Tmax_reference_C=float(expected.max()),
                ring_max_error_K=float(np.abs(ring_error).max()),
                ring_rmse_K=float(np.sqrt(np.mean(ring_error**2))),
                area_nrmse_fem_pct=area_report['nrmse_pct'],
                area_Tmax_pred_C=area_report['Tmax_pred_C'],
                combined_area_plus_ring_Tmax_C=float(union_prediction.max()),
                combined_max_error_K=float(np.abs(union_prediction-union_reference).max())))
    return dict(case='annulus', evaluation='area quadrature plus separate inner rings ri+epsilon',
        epsilon_m=list(epsilons), n_angles=n_angles, rows=rows,
        interpretation='The epsilon rings are diagnostic only; they are not added to the area-weighted NRMSE.')


def comparison_rows(output, definitions):
    result = {}
    for case, definition in definitions.items():
        problem = load_problem(definition)
        xy, labels, weights = problem.evaluation()
        names = [region.name for region in problem.regions]
        truth = problem.reference(xy, labels)
        fem_path = output/case/'fem'/f'{case}_l1.npz'
        metadata_path = fem_path.with_suffix('.json')
        rows = []
        if fem_path.exists() and metadata_path.exists():
            reference = load_reference_npz(fem_path, problem)
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            analytic = error_report(reference, truth, xy, labels, weights, names,
                problem.T_level, problem.conductors(), temperature_scale=problem.T_scale)
            rows.append(dict(model='FEM_P2_level1', status='reference', Tmax_C=float(reference.max()),
                global_balance_pct=metadata['balance_pct'], elapsed_s=metadata['elapsed_s'],
                elapsed_min=metadata['elapsed_s']/60, NRMSE_analytic_pct=analytic['nrmse_pct'],
                max_error_analytic_K=analytic['max_error_K'], evaluation_points=len(xy),
                fem_dofs=metadata.get('ndofs'), artifact_directory=str(fem_path.parent)))
        for architecture in MODELS:
            folder = output/case/architecture
            path = folder/'result.json'
            row = dict(model=architecture, status='incomplete' if folder.exists() else 'pending',
                artifact_directory=str(folder))
            row.update(collocation_counts(folder))
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                assert record['physics_sha256'] == problem.physics_sha256
                evaluation = record.get('evaluation', {})
                balance = evaluation.get('balance', {})
                stopping = record.get('accuracy_stop', {})
                row.update(status=record['status'], Tmax_C=evaluation.get('Tmax_pred_C'),
                    termination_reason=stopping.get('termination_reason'),
                    Tmax_reference_C=evaluation.get('Tmax_ref_C'), Tmax_error_K=evaluation.get('error_Tmax_abs_K'),
                    NRMSE_FEM_pct=evaluation.get('nrmse_pct'), max_error_FEM_K=evaluation.get('max_error_K'),
                    global_balance_pct=balance.get('global_derived_pct'),
                    worst_region_balance_pct=max(balance.get('region_derived_pct', {}).values(), default=None),
                    elapsed_s=record.get('elapsed_s'), elapsed_min=record.get('elapsed_s', 0)/60,
                    accuracy_targets_met=stopping.get('triggered'),
                    lbfgs_outer_calls=stopping.get('lbfgs_outer_calls'),
                    extra_lbfgs_calls=stopping.get('extra_lbfgs_outer_calls'),
                    annulus_inner_boundary_max_error_K=evaluation.get('annulus_boundary', {}).get('inner', {}).get('max_error_K'),
                    annulus_outer_boundary_max_error_K=evaluation.get('annulus_boundary', {}).get('outer', {}).get('max_error_K'),
                    annulus_boundary_points=evaluation.get('annulus_boundary_points'),
                        all_raw_gates=all(evaluation.get('acceptance', {}).get('raw_gates', {}).values())
                        if evaluation.get('acceptance', {}).get('raw_gates') else None,
                    error=(record.get('error') or {}).get('message'))
                if (folder/'evaluation.npz').exists():
                    with np.load(folder/'evaluation.npz') as arrays:
                        assert np.array_equal(arrays['xy'], xy)
                        row['evaluation_points'] = len(arrays['xy'])
                        analytic = error_report(arrays['T'], truth, xy, labels, weights, names,
                            problem.T_level, problem.conductors(), temperature_scale=problem.T_scale)
                    row.update(NRMSE_analytic_pct=analytic['nrmse_pct'], max_error_analytic_K=analytic['max_error_K'])
                if record.get('checkpoint_selection', {}).get('final_validation_loss') is not None:
                    points = record['config']['points']
                    row['validation_pde_points'] = sum(points['n_interior'] if region.kind == 'soil'
                        else points['n_layer'] for region in problem.regions)
                if evaluation.get('pde_rms'):
                    row['audit_pde_points'] = sum(2048 if region.kind == 'soil' else 1024
                        for region in problem.regions)
                    row['audit_surface_points'] = sum(len(face.xy) for face in problem.faces(512)) + sum(
                        len(piece.xy) for piece in problem.pieces(512))
            rows.append(row)
        result[case] = dict(physics_sha256=problem.physics_sha256, evaluation_points=len(xy), rows=rows)
    return result


def export_reports(output, definitions, settings):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    output = Path(output)
    cases = comparison_rows(output, definitions)
    stamp = datetime.now(timezone.utc).isoformat()
    effective_settings = {}
    for case, entry in cases.items():
        case_settings = dict(settings)
        if case == 'annulus':
            case_settings['sampling'] = 'annulus_log_v1'
        entry['effective_settings'] = case_settings
        effective_settings[case] = case_settings
    summary = dict(updated_utc=stamp, settings=settings, effective_settings=effective_settings, cases=cases)
    replace_json_atomic(output/'summary.json', summary)
    workbook = Workbook()
    workbook.remove(workbook.active)
    markdown = ['# Campana exacta MMS y annulus', '', f'Actualizado UTC: {stamp}', '',
        'FEM primero: nivel 1, P2, multiplicador 1.0. CPU PINN: un hilo, float64.',
        'Referencia principal y parada: FEM del mismo caso; errores analiticos separados.',
        'Loss: solo PDE. La referencia se usa para detener/evaluar, nunca en la loss.',
        'Parada AND estricta: error Tmax <0.01 K, NRMSE <0.1%, balance global <0.5%.',
        'Presupuesto: 400 Adam +5 L-BFGS; hasta5 adicionales con el MISMO optimizador si hace falta.',
        'Tiempos originales del solver; no benchmark controlado de aceleracion ni prueba multisemilla.',
        'FEM no se compara consigo mismo: sus errores vs FEM quedan vacios.', '',
        'Muestreo por fila: conteos reales de collocation.npz. Total=interiores+interfaces+frontera.',
        'PDE-only usa SOLO interiores; interfaces/frontera son diagnostico, no penalties.',
        'Total y unicos describen la nube fija, NO puntos multiplicados por epocas.',
        'Validacion, evaluacion y auditoria se informan aparte y NO se suman al total de colocacion.',
        'Validacion: conteo del sampler fijo configurado; auditoria: conteos del auditor estandar.',
        'FEM informa DoFs y puntos de evaluacion; no tiene colocacion PINN. Pendientes quedan vacios.', '',
        '## Parametros', '', '```json', json.dumps(settings, indent=2), '```', '']
    for case, entry in cases.items():
        problem = load_problem(definitions[case])
        points = settings.get('points', dict(n_interior=512, n_layer=256))
        entry['training_pde_points'] = sum(points['n_interior'] if region.kind == 'soil'
            else points['n_layer'] for region in problem.regions)
        sheet = workbook.create_sheet(case)
        sheet.append(['case', case])
        sheet.append(['physics_sha256', entry['physics_sha256']])
        sheet.append(['updated_utc', stamp])
        sheet.append(['parameters', json.dumps(entry['effective_settings'], sort_keys=True)])
        sheet.append(['reference', 'FEM P2 level1; analytic errors in separate columns'])
        sheet.append(['planned_training_pde_points', entry['training_pde_points'], 'evaluation_points', entry['evaluation_points']])
        sheet.append([])
        sheet.append(list(COLUMNS))
        for cell in sheet[7]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='424242')
        for row in entry['rows']:
            sheet.append([row.get(name) for name in COLUMNS])
        sheet.freeze_panes = 'C8'
        sheet.auto_filter.ref = f'A7:{get_column_letter(len(COLUMNS))}{sheet.max_row}'
        for index, name in enumerate(COLUMNS, 1):
            sheet.column_dimensions[get_column_letter(index)].width = min(max(len(name)+2, 18), 34)
        sheet.column_dimensions['A'].width = 28
        sheet['B4'].alignment = Alignment(wrap_text=True)
        sheet.row_dimensions[4].height = 60
        for row in sheet.iter_rows(min_row=8):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = '0.000000'
        markdown.extend([f'## {case}', '',
            f"Puntos PDE de entrenamiento: {entry['training_pde_points']}; evaluacion: {entry['evaluation_points']}.", '',
            f"Muestreo efectivo: `{entry['effective_settings']['sampling']}`.", '',
            '| Modelo | Estado | Tmax C | Error Tmax K | NRMSE FEM % | Balance global % | Error borde int K | Error borde ext K | Pts borde | Tiempo s | L-BFGS | Extra | Objetivos AND | Raw gates |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|'])
        for row in entry['rows']:
            keys = ('model','status','Tmax_C','Tmax_error_K','NRMSE_FEM_pct','global_balance_pct',
                'annulus_inner_boundary_max_error_K','annulus_outer_boundary_max_error_K',
                'annulus_boundary_points','elapsed_s','lbfgs_outer_calls','extra_lbfgs_calls',
                'accuracy_targets_met','all_raw_gates')
            values = [f'{row[key]:.6f}' if isinstance(row.get(key), float)
                else str(row.get(key, '')) if row.get(key) is not None else '' for key in keys]
            markdown.append('| '+' | '.join(values)+' |')
        markdown.extend(['',
            '| Modelo | Estado | Interiores PDE | Interfaz | Frontera | Total colocacion | Unicos | Validacion PDE | Evaluacion | Auditoria PDE | Auditoria superficies | FEM DoFs |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for row in entry['rows']:
            keys = ('model', 'status', 'training_pde_points', 'interface_points', 'boundary_points',
                'collocation_total_points', 'collocation_unique_points', 'validation_pde_points',
                'evaluation_points', 'audit_pde_points', 'audit_surface_points', 'fem_dofs')
            markdown.append('| '+' | '.join(str(row[key]) if row.get(key) is not None else ''
                for key in keys)+' |')
        finished = [row for row in entry['rows'] if row['model'] != 'FEM_P2_level1' and row['status'] == 'completed']
        reached = sum(row.get('accuracy_targets_met') is True for row in finished)
        markdown.extend(['', f'Concluidas: {len(finished)}/6. Objetivos estrictos alcanzados: {reached}/{len(finished)}.',
            'Un presupuesto agotado no equivale a objetivos logrados; los pendientes no tienen resultados.', ''])
    annulus_comparison_path = output/'annulus'/'comparison_log_v1_20261021.json'
    if 'annulus' in cases and annulus_comparison_path.is_file():
        comparison = json.loads(annulus_comparison_path.read_text(encoding='utf-8'))
        sheet = workbook.create_sheet('Before_after')
        headers = ('Model', 'Old inner-boundary max error K', 'New inner-boundary max error K',
            'Old area NRMSE FEM %', 'New area NRMSE FEM %', 'NRMSE reduction %',
            'New global balance %', 'New raw gates')
        sheet.append(list(headers))
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='424242')
        for row in comparison['models']:
            sheet.append([row['model'], row['old']['inner_boundary_max_error_K'],
                row['new']['inner_boundary_max_error_K'], row['old']['nrmse_fem_pct'],
                row['new']['nrmse_fem_pct'], row['relative_nrmse_reduction_pct'],
                row['new']['global_balance_pct'], row['new']['all_raw_gates']])
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = f'A1:H{sheet.max_row}'
        for column, width in zip('ABCDEFGH', (26, 34, 34, 25, 25, 24, 25, 18)):
            sheet.column_dimensions[column].width = width
        markdown.extend(['## annulus: antes/despues del lifting log-radial', '',
            'Misma fisica/FEM, seis arquitecturas, seed0/sampling11 y presupuesto. El NRMSE usa la nube de area de 6000 puntos; los errores de borde usan 256 puntos de la circunferencia interior.', '',
            '| Modelo | Error borde viejo K | Error borde nuevo K | NRMSE FEM viejo % | NRMSE FEM nuevo % | Reduccion NRMSE % | Balance nuevo % | Raw gates nuevos |',
            '|---|---:|---:|---:|---:|---:|---:|---|'])
        for row in comparison['models']:
            markdown.append(f"| {row['model']} | {row['old']['inner_boundary_max_error_K']:.3f} | "
                f"{row['new']['inner_boundary_max_error_K']:.3f} | {row['old']['nrmse_fem_pct']:.3f} | "
                f"{row['new']['nrmse_fem_pct']:.3f} | {row['relative_nrmse_reduction_pct']:.1f} | "
                f"{row['new']['global_balance_pct']:.3f} | {row['new']['all_raw_gates']} |")
        markdown.extend(['', 'Cambio limitado a annulus: lifting Hermite en log-radio, muestreo mixto area/log-radio y auditoria de bordes explicita. Hubo mejora marcada, pero las seis variantes siguen rechazadas; no aumentar pasos hasta resolver la traza interior/identificabilidad.', ''])
    extension_path = output/'annulus_extension_1000_50'/'annulus_extension_summary_1000_50.json'
    if extension_path.is_file():
        extension = json.loads(extension_path.read_text(encoding='utf-8'))
        sheet = workbook.create_sheet('Annulus_extend')
        headers = ('Model', 'Initialization', 'Adam steps', 'L-BFGS calls', 'Extra calls',
            'Training PDE points', 'Saved collocation total', 'Initial PDE loss', 'Final PDE loss',
            'Initial FEM NRMSE %', 'Final FEM NRMSE %', 'Initial Tmax error K', 'Final Tmax C',
            'Final Tmax error K', 'Final inner boundary max error K', 'Final global balance %',
            'Accuracy targets met', 'Raw gates', 'Elapsed s', 'Run directory')
        sheet.append(list(headers))
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='424242')
        for row in extension['rows']:
            sheet.append([row.get(key) for key in ('model','initialization_policy','adam_steps',
                'lbfgs_outer_calls','extra_lbfgs_outer_calls','training_pde_points','collocation_total_points',
                'initial_pde_loss','final_pde_loss','initial_nrmse_fem_pct','final_nrmse_fem_pct',
                'initial_Tmax_error_K','final_Tmax_C','final_Tmax_error_K','inner_boundary_max_error_K',
                'global_balance_pct','accuracy_targets_met','all_raw_gates','elapsed_s','run_directory')])
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = f'A1:T{sheet.max_row}'
        for index, width in enumerate((26,30,14,16,14,22,22,18,18,22,22,22,18,22,34,24,22,16,16,76),1):
            sheet.column_dimensions[get_column_letter(index)].width=width
        markdown.extend(['## Annulus: extension desde pesos previos', '',
            'Pesos del checkpoint log-v1; Adam reiniciado 1000 pasos; hasta50 llamadas L-BFGS base y otras50 si fallan los tres criterios. No se reutiliza el estado interno del optimizador.', '',
            '| Modelo | Inicializacion | Adam | L-BFGS | Extra | PDE pts | Total nube | Loss inicial | Loss final | NRMSE inicial FEM % | NRMSE final FEM % | Tmax error inicial K | Tmax final C | Tmax error final K | Error borde int K | Balance % | Targets | Raw gates | Tiempo s |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|'])
        for row in extension['rows']:
            markdown.append('| '+' | '.join(str(row.get(key,'')) if row.get(key) is not None else '' for key in
                ('model','initialization_policy','adam_steps','lbfgs_outer_calls','extra_lbfgs_outer_calls',
                 'training_pde_points','collocation_total_points','initial_pde_loss','final_pde_loss',
                 'initial_nrmse_fem_pct','final_nrmse_fem_pct','initial_Tmax_error_K','final_Tmax_C',
                 'final_Tmax_error_K','inner_boundary_max_error_K','global_balance_pct',
                 'accuracy_targets_met','all_raw_gates','elapsed_s'))+' |')
        markdown.append('')
    epsilon_path = output/'annulus_extension_1000_50'/'epsilon_evaluation_20261021.json'
    if epsilon_path.is_file():
        epsilon_report = json.loads(epsilon_path.read_text(encoding='utf-8'))
        sheet = workbook.create_sheet('Annulus_epsilon')
        headers = ('Model','Epsilon m','Epsilon/ri','Ring points','Ring Tmax pred C',
            'Ring Tmax exact C','Ring max error K','Ring RMSE K','Area NRMSE FEM %',
            'Area Tmax pred C','Area + ring Tmax C','Area + ring max error K')
        sheet.append(list(headers))
        for cell in sheet[1]:
            cell.font=Font(bold=True,color='FFFFFF'); cell.fill=PatternFill('solid',fgColor='424242')
        for row in epsilon_report['rows']:
            sheet.append([row.get(key) for key in ('model','epsilon_m','epsilon_over_ri','points',
                'ring_Tmax_pred_C','ring_Tmax_reference_C','ring_max_error_K','ring_rmse_K',
                'area_nrmse_fem_pct','area_Tmax_pred_C','combined_area_plus_ring_Tmax_C','combined_max_error_K')])
        sheet.freeze_panes='A2'; sheet.auto_filter.ref=f'A1:L{sheet.max_row}'
        for col,width in zip('ABCDEFGHIJKL',(26,16,16,14,22,22,20,18,22,22,24,26)):
            sheet.column_dimensions[col].width=width
        epsilon_by_model={name:[row for row in epsilon_report['rows'] if row['model']==name]
            for name in MODELS}
        markdown.extend(['## Annulus: evaluacion interior ri+epsilon', '',
            'Se excluye la frontera misma y se evalua a radios estrictamente interiores. NRMSE permanece ponderada sobre el area original; los anillos epsilon son diagnosticos separados.', '',
            '| Modelo | Error max a ri+1e-8 m K | Error max a ri+1e-4 m K | Error max a ri+1e-2 m K | NRMSE area FEM % | Tmax area+epsilon C | Max error area+epsilon K |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for name, entries in epsilon_by_model.items():
            by_epsilon={row['epsilon_m']:row for row in entries}
            if 1e-8 not in by_epsilon or 1e-4 not in by_epsilon or 1e-2 not in by_epsilon:
                continue
            last=by_epsilon[1e-8]; middle=by_epsilon[1e-4]; inner=by_epsilon[1e-2]
            markdown.append(f"| {name} | {last['ring_max_error_K']:.6f} | {middle['ring_max_error_K']:.6f} | "
                f"{inner['ring_max_error_K']:.6f} | {last['area_nrmse_fem_pct']:.6f} | "
                f"{last['combined_area_plus_ring_Tmax_C']:.6f} | {last['combined_max_error_K']:.6f} |")
        markdown.extend(['', epsilon_report['interpretation'], ''])
    temporary = output/'comparison.tmp.xlsx'
    workbook.save(temporary)
    os.replace(temporary, output/'comparison.xlsx')
    replace_json_atomic(output/'summary.json', summary)
    (output/'report.md').write_text('\n'.join(markdown)+'\n', encoding='utf-8')
    return summary


def run_campaign(output, cases=None, models=MODELS, report_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    definitions = case_definitions()
    selected = list(definitions) if cases is None else list(cases)
    if set(selected)-set(definitions) or set(models)-set(MODELS):
        raise ValueError('Unknown verification case or exact model')
    settings = json.loads((PACKAGE_ROOT/'configs'/'train_mms_exact_repro_20261020.json').read_text())
    settings.update(accuracy_stop=dict(error_Tmax_abs_K=.01,nrmse_pct=.1,balance_pct=.5,every=100),
        extra_lbfgs_steps=5)
    config_path = output/'campaign_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != settings:
            raise ValueError('Campaign configuration changed; use a new output directory')
    else:
        write_json_atomic(config_path, settings)
    selected_definitions = {case: definitions[case] for case in selected}
    if not report_only:
        for case, definition in selected_definitions.items():
            if not (output/case/'fem'/f'{case}_l1.json').exists():
                adapter.run_mms_fem(output/case/'fem', definition=definition, levels=(1,), node_multiplier=1.0)
    export_reports(output, selected_definitions, settings)
    if report_only:
        return
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', DDE_BACKEND='pytorch')
    for case in selected:
        for architecture in models:
            folder = output/case/architecture
            if (folder/'result.json').exists():
                continue
            if folder.exists():
                raise ValueError(f'Incomplete run preserved: {folder}; do not overwrite its checkpoint')
            run_config = dict(settings)
            if case == 'annulus':
                run_config['sampling'] = 'annulus_log_v1'
            case_config_path = output/'configs'/f'{case}.json'
            case_config_path.parent.mkdir(parents=True, exist_ok=True)
            if case_config_path.exists():
                if json.loads(case_config_path.read_text()) != run_config:
                    raise ValueError(f'Campaign configuration changed for {case}; use a new output directory')
            else:
                write_json_atomic(case_config_path, run_config)
            command = [sys.executable, '-m', 'pinn_v2', 'train', '--config', str(case_config_path),
                '--definition', str(definitions[case]), '--model', architecture,
                '--reference', str(output/case/'fem'/f'{case}_l1.npz'), '--output', str(folder)]
            print('START', case, architecture, 'sampling='+run_config['sampling'], flush=True)
            subprocess.run(command, cwd=PACKAGE_ROOT, env=environment, check=False)
            export_reports(output, selected_definitions, settings)
            print('FINISHED', case, architecture, flush=True)


def validate_before_campaign(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', DDE_BACKEND='pytorch')
    command = [sys.executable, '-m', 'pytest', 'tests', '-m', 'not slow and not fem',
        '-k', 'not test_provenance_record_matches_untouched_originals', '-q', '--tb=short',
        f'--junitxml={output/"trainer_regression_gate.xml"}']
    print('VALIDATION_START', flush=True)
    completed = subprocess.run(command, cwd=PACKAGE_ROOT, env=environment, capture_output=True, text=True)
    (output/'trainer_regression_output.txt').write_text(completed.stdout+'\n'+completed.stderr, encoding='utf-8')
    replace_json_atomic(output/'trainer_regression_status.json',
        dict(returncode=completed.returncode, passed=completed.returncode == 0, command=command))
    print(completed.stdout[-10000:], flush=True)
    if completed.returncode:
        raise RuntimeError('Trainer regression failed; PINN campaign was not started. See trainer_regression_output.txt')


def extend_annulus_from_weights(root, source_root, models=MODELS):
    """Continue annulus log-v1 runs from trained weights with a longer Adam/L-BFGS horizon."""
    from pinn_v2.evaluation.reference import load_reference_npz
    from pinn_v2.solvers.pinn import trainer
    from pinn_v2.solvers.pinn.checkpoint import load_checkpoint
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    definition = case_definitions()['annulus']
    reference_path = root/'annulus'/'fem'/'annulus_l1.npz'
    problem = load_problem(definition)
    reference = load_reference_npz(reference_path, problem)
    extension_root = root/'annulus_extension_1000_50'
    extension_root.mkdir(parents=True, exist_ok=True)
    config = json.loads((root/'configs'/'annulus.json').read_text(encoding='utf-8'))
    config.update(adam_steps=1000, lbfgs_steps=50, extra_lbfgs_steps=50,
        accuracy_stop=dict(error_Tmax_abs_K=.01,nrmse_pct=.1,balance_pct=.5,every=100,after_phase='lbfgs'),
        initialization_policy='weights_only_warm_start', optimizer_state_reused=False)
    config_path=extension_root/'extension_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Annulus extension configuration changed; use a new extension directory')
    else:
        write_json_atomic(config_path,config)
    summary_path=extension_root/'annulus_extension_summary_1000_50.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else dict(
        case='annulus', method='weights-only warm start; Adam horizon restarted; L-BFGS history is new',
        source_campaign=str(source_root), config=config, rows=[])
    rows={row['model']:row for row in summary['rows']}
    for architecture in models:
        if architecture in rows and rows[architecture].get('status') == 'completed':
            continue
        prior=source_root/'annulus'/architecture/'model.pt'
        if not prior.is_file():
            raise FileNotFoundError(f'Previous annulus weights missing: {prior}')
        _, specification, _=load_checkpoint(prior)
        if specification['physics_sha256'] != problem.physics_sha256:
            raise ValueError(f'Warm-start physics mismatch for {architecture}')
        folder=extension_root/architecture
        prior_result=json.loads((prior.parent/'result.json').read_text())
        initial_nrmse=prior_result['evaluation']['nrmse_pct']
        initial_pde_loss=prior_result.get('final_loss')
        initial_tmax_error=prior_result['evaluation'].get('error_Tmax_abs_K')
        result_path=folder/'result.json'
        if folder.exists() and not result_path.exists():
            retry=1
            while (extension_root/f'{architecture}_retry{retry}').exists():
                retry += 1
            print('PRESERVE_INCOMPLETE_EXTENSION',folder,flush=True)
            folder=extension_root/f'{architecture}_retry{retry}'
            result_path=folder/'result.json'
        if result_path.exists():
            result=json.loads(result_path.read_text(encoding='utf-8'))
        else:
            model, _, _=load_checkpoint(prior)
            settings=dict(adam_steps=1000,lbfgs_steps=50,extra_lbfgs_steps=50,
                lr=config['lr'],width=config['width'],depth=config['depth'],activation=config['activation'],
                weights=config['weights'],points=config['points'],init_seed=config['seed'],
                sampling_seed=config['sampling_seed'],log_every=config['log_every'],reference=reference,
                reference_label=str(reference_path)+' (same FEM level1)',device=config['device'],dtype=config['dtype'],
                input_normalization=config['input_normalization'],sampling='annulus_log_v1',
                validation_every=config['validation_every'],initial_state=model.state_dict(),warm_start_source=str(prior),
                accuracy_stop=config['accuracy_stop'],full_audit=True)
            print('EXTEND_START',architecture,'from',prior,flush=True)
            result=trainer.run(definition,architecture,folder,**settings)
        evaluation=result.get('evaluation',{})
        stopping=result.get('accuracy_stop',{})
        row=dict(model=architecture,status=result['status'],initialization_policy='weights_only_warm_start',
            warm_start_source=str(prior),initial_nrmse_fem_pct=initial_nrmse,
            initial_pde_loss=initial_pde_loss,initial_Tmax_error_K=initial_tmax_error,
            final_pde_loss=result.get('final_loss'),final_Tmax_C=evaluation.get('Tmax_pred_C'),
            final_Tmax_error_K=evaluation.get('error_Tmax_abs_K'),
            final_nrmse_fem_pct=evaluation.get('nrmse_pct'),
            inner_boundary_max_error_K=evaluation.get('annulus_boundary',{}).get('inner',{}).get('max_error_K'),
            global_balance_pct=evaluation.get('balance',{}).get('global_derived_pct'),
            accuracy_targets_met=stopping.get('triggered'),termination_reason=stopping.get('termination_reason'),
            lbfgs_outer_calls=stopping.get('lbfgs_outer_calls'),
            extra_lbfgs_outer_calls=stopping.get('extra_lbfgs_outer_calls'),
            all_raw_gates=all(evaluation.get('acceptance',{}).get('raw_gates',{}).values())
                if evaluation.get('acceptance',{}).get('raw_gates') else None,
            **collocation_counts(folder),
            elapsed_s=result.get('elapsed_s'),adam_steps=1000,base_lbfgs_steps=50,max_extra_lbfgs_steps=50,
            parameters=result.get('parameters'),run_directory=str(folder))
        rows[architecture]=row
        summary['rows']=[rows[name] for name in models if name in rows]
        summary['updated_utc']=datetime.now(timezone.utc).isoformat()
        if summary_path.exists():
            replace_json_atomic(summary_path,summary)
        else:
            write_json_atomic(summary_path,summary)
        export_reports(root,{'annulus':definition},json.loads((root/'campaign_config.json').read_text()))
        print('EXTEND_DONE',architecture,'nrmse',row['final_nrmse_fem_pct'],
            'accuracy_targets_met',row['accuracy_targets_met'],'time_s',row['elapsed_s'],flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=MODELS)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--validate-first', action='store_true')
    parser.add_argument('--extend-annulus-from', type=Path,
        help='Start a 1000-Adam/50+50-L-BFGS weights-only extension from a prior annulus campaign')
    args = parser.parse_args()
    if args.extend_annulus_from is not None:
        extend_annulus_from_weights(args.output,args.extend_annulus_from)
        return
    if args.validate_first and not args.report_only:
        validate_before_campaign(args.output)
    run_campaign(args.output, args.cases, args.models, args.report_only)


if __name__ == '__main__':
    main()