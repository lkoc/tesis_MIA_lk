"""Compare fresh 8000-Adam/200-L-BFGS CUSP_EXACT runs on the annulus."""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinn_v2.evaluation.metrics import error_report
from pinn_v2.evaluation.reference import load_reference_npz
from pinn_v2.manifest.run_manifest import replace_json_atomic, write_json_atomic
from pinn_v2.solvers.pinn.checkpoint import infer, load_checkpoint
from pinn_v2.solvers.pinn.problem import load_problem
from pinn_v2.solvers.pinn.trainer import run

ROOT = Path(__file__).resolve().parents[2]
DEFINITION = ROOT/'pinn_v2'/'cases'/'v2'/'references'/'verification'/'annulus.json'
FEM_NPZ = ROOT/'results'/'phase7'/'p7_annulus_fem_line_20261021'/'fem'/'annulus_l1.npz'
FEM_JSON = FEM_NPZ.with_suffix('.json')
SAMPLERS = ('mixed', 'annulus_log_v1')


def write_outputs(output, settings):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    output = Path(output)
    fem = np.load(FEM_NPZ)
    rows = []
    predictions = {}
    problem = load_problem(DEFINITION)
    points, labels, weights = problem.evaluation()
    truth = problem.reference(points, labels)
    left = fem['line_xy'][:len(fem['line_xy'])//2]
    right = fem['line_xy'][len(fem['line_xy'])//2:]
    for sampler in SAMPLERS:
        folder = output/sampler
        path = folder/'result.json'
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding='utf-8'))
        model, _, _ = load_checkpoint(folder/'model.pt')
        line_prediction = infer(model, fem['line_xy'])['T']
        predictions[sampler] = line_prediction
        with np.load(folder/'evaluation.npz') as evaluation:
            analytic = error_report(evaluation['T'].ravel(), truth, points, labels, weights,
                [region.name for region in problem.regions], problem.T_level, problem.conductors(),
                temperature_scale=problem.T_scale)
        line_error_fem = line_prediction-fem['line_T']
        line_error_exact = line_prediction-fem['line_T_exact']
        report = result['evaluation']
        rows.append(dict(sampler=sampler, status=result['status'], elapsed_s=result['elapsed_s'],
            elapsed_min=result['elapsed_s']/60, parameters=result['parameters'],
            area_Tmax_C=report['Tmax_pred_C'], Tmax_error_FEM_K=report['error_Tmax_abs_K'],
            NRMSE_FEM_pct=report['nrmse_pct'], NRMSE_analytic_pct=analytic['nrmse_pct'],
            global_balance_pct=report['balance']['global_derived_pct'],
            inner_boundary_max_error_K=report['annulus_boundary']['inner']['max_error_K'],
            outer_boundary_max_error_K=report['annulus_boundary']['outer']['max_error_K'],
            horizontal_line_RMSE_FEM_K=float(np.sqrt(np.mean(line_error_fem**2))),
            horizontal_line_max_error_FEM_K=float(np.abs(line_error_fem).max()),
            horizontal_line_RMSE_analytic_K=float(np.sqrt(np.mean(line_error_exact**2))),
            all_raw_gates=all(report['acceptance']['raw_gates'].values()),
            termination='fixed_budget_no_early_stop', run_directory=folder.as_posix()))

    summary = dict(config=settings, fem_reference=str(FEM_NPZ),
        fem_record=json.loads(FEM_JSON.read_text(encoding='utf-8')),
        status='complete' if len(rows) == len(SAMPLERS) else 'in_progress', results=rows)
    replace_json_atomic(output/'summary.json', summary)

    # Draw both annular branches with a visible break across the excluded hole.
    x = fem['line_xy'][:, 0]
    split = len(x)//2
    plot_x = np.r_[x[:split], np.nan, x[split:]]
    exact = np.r_[fem['line_T_exact'][:split], np.nan, fem['line_T_exact'][split:]]
    fem_temperature = np.r_[fem['line_T'][:split], np.nan, fem['line_T'][split:]]
    fig, (axis, error_axis) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    axis.plot(plot_x, exact, color='black', linewidth=2.6, label='Solución analítica exacta')
    axis.plot(plot_x, fem_temperature, color='#d1495b', linewidth=1.6, linestyle='--', label='FEM P2 nivel 1')
    colors = {'mixed': '#2878b5', 'annulus_log_v1': '#e07a1f'}
    for sampler, values in predictions.items():
        axis.plot(plot_x, np.r_[values[:split], np.nan, values[split:]], color=colors[sampler],
            linewidth=1.7, label=f'CUSP_EXACT {sampler}')
    for target in (axis, error_axis):
        target.axvspan(-.06, .06, color='#d9dde3', alpha=.8,
            label='Hueco excluido del dominio' if target is axis else None)
        target.axvline(-.06, color='#777', linewidth=.7, linestyle=':')
        target.axvline(.06, color='#777', linewidth=.7, linestyle=':')
        target.grid(alpha=.25)
    axis.set_ylabel('Temperatura (°C)')
    axis.set_title('CUSP_EXACT annulus: 8000 Adam + 200 llamadas externas L-BFGS')
    axis.legend(ncol=2, fontsize=8)
    error_axis.plot(fem['line_xy'][:, 0], fem['line_T']-fem['line_T_exact'],
        color='#555', linewidth=1.2, label='FEM − analítica')
    for sampler, values in predictions.items():
        error_axis.plot(fem['line_xy'][:, 0], values-fem['line_T'], color=colors[sampler],
            linewidth=1.0, label=f'CUSP {sampler} − FEM')
    error_axis.axhline(0, color='black', linewidth=.8)
    error_axis.set(xlabel='x (m), línea horizontal y=0', ylabel='Diferencia (K)')
    image = output/'cusp_8000_adam_200_lbfgs_y0.png'
    fig.savefig(image, dpi=180)
    plt.close(fig)

    csv_path = output/'cusp_8000_adam_200_lbfgs_y0.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x_m', 'region', 'T_analytic_C', 'T_FEM_C',
            'T_CUSP_mixed_C', 'T_CUSP_annulus_log_v1_C'])
        for index, coordinate in enumerate(plot_x):
            region = 'hole' if not np.isfinite(coordinate) else ('annulus_left' if coordinate < 0 else 'annulus_right')
            writer.writerow([coordinate, region, exact[index], fem_temperature[index],
                np.r_[predictions.get('mixed', np.full(len(x), np.nan))[:split], np.nan,
                    predictions.get('mixed', np.full(len(x), np.nan))[split:]][index],
                np.r_[predictions.get('annulus_log_v1', np.full(len(x), np.nan))[:split], np.nan,
                    predictions.get('annulus_log_v1', np.full(len(x), np.nan))[split:]][index]])
    summary.update(plot=str(image), csv=str(csv_path))
    replace_json_atomic(output/'summary.json', summary)

    workbook = Workbook()
    metrics = workbook.active
    metrics.title = 'Metrics'
    headers = ('Sampler', 'Status', 'Adam', 'L-BFGS calls', 'Parameters', 'NRMSE FEM %',
        'NRMSE analytic %', 'Tmax error FEM K', 'Inner boundary max error K',
        'Global balance %', 'Line RMSE FEM K', 'Line max error FEM K', 'Runtime s', 'Runtime min', 'Raw gates')
    metrics.append(list(headers))
    for row in rows:
        metrics.append([row['sampler'], row['status'], settings['adam_steps'], settings['lbfgs_external_calls'],
            row['parameters'], row['NRMSE_FEM_pct'], row['NRMSE_analytic_pct'], row['Tmax_error_FEM_K'],
            row['inner_boundary_max_error_K'], row['global_balance_pct'], row['horizontal_line_RMSE_FEM_K'],
            row['horizontal_line_max_error_FEM_K'], row['elapsed_s'], row['elapsed_min'], row['all_raw_gates']])
    line_sheet = workbook.create_sheet('Horizontal_y0')
    with csv_path.open(encoding='utf-8', newline='') as stream:
        for values in csv.reader(stream):
            line_sheet.append(values)
    for sheet in workbook:
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='424242')
        for column in range(1, sheet.max_column+1):
            sheet.column_dimensions[get_column_letter(column)].width = 24
    temporary = output/'comparison.tmp.xlsx'
    workbook.save(temporary)
    os.replace(temporary, output/'cusp_8000_adam_200_lbfgs_comparison.xlsx')
    write_report(output, summary)
    return summary


def write_report(output, summary):
    lines = ['# CUSP_EXACT annulus: 8000 Adam + 200 L-BFGS', '',
        'Dos corridas frescas desde inicialización; mismas física, FEM P2 nivel 1, semilla 0 y arquitectura. Solo cambia el sampler.',
        'Referencia FEM y analítica: evaluación en los mismos 6000 puntos de área; las métricas de la línea y=0 son diagnósticos adicionales.',
        'Criterios de parada desactivados: se completó el presupuesto fijo solicitado.', '',
        '## Parámetros', '', '```json', json.dumps(summary['config'], indent=2), '```', '',
        '| Sampler | NRMSE FEM % | NRMSE analítica % | Error Tmax FEM K | Error interior K | Balance % | RMSE línea FEM K | Gates | Tiempo s | Tiempo min |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for row in summary['results']:
        lines.append(f"| {row['sampler']} | {row['NRMSE_FEM_pct']:.6f} | {row['NRMSE_analytic_pct']:.6f} | "
            f"{row['Tmax_error_FEM_K']:.6f} | {row['inner_boundary_max_error_K']:.6f} | "
            f"{row['global_balance_pct']:.6f} | {row['horizontal_line_RMSE_FEM_K']:.6f} | "
            f"{row['all_raw_gates']} | {row['elapsed_s']:.2f} | {row['elapsed_min']:.2f} |")
    lines.extend(['', f"Estado: {summary['status']} ({len(summary['results'])}/2 corridas).",
        '', f"![Comparación CUSP_EXACT annulus]({Path(summary['plot']).name})", '',
        '- [Excel comparativo](cusp_8000_adam_200_lbfgs_comparison.xlsx).',
        '- [CSV de la línea y=0](cusp_8000_adam_200_lbfgs_y0.csv).', ''])
    (Path(output)/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run_comparison(output, adam_steps=8000, lbfgs_steps=200):
    import torch
    from pinn_v2.evaluation.reference import load_reference_npz

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    problem = load_problem(DEFINITION)
    fem_record = json.loads(FEM_JSON.read_text(encoding='utf-8'))
    if fem_record['physics_sha256'] != problem.physics_sha256 or fem_record['level'] != 1:
        raise ValueError('The annulus FEM reference hash/level does not match this case')
    reference = load_reference_npz(FEM_NPZ, problem)
    settings = dict(case='annulus', model='CUSP_EXACT', initialization='fresh_seed_0',
        sampling_seed=11, dtype='float64', device='cpu', torch_threads=1,
        width=32, depth=3, activation='tanh', lr=.003, adam_steps=adam_steps,
        lbfgs_external_calls=lbfgs_steps, lbfgs_max_iter_per_call=20,
        extra_lbfgs_calls=0, accuracy_stop=None, n_interior=512,
        input_normalization='unit_box_v1', FEM_level=1, FEM_degree=2,
        FEM_doFs=fem_record['ndofs'], FEM_seconds=fem_record['elapsed_s'],
        pde_loss_only=True, physics_sha256=problem.physics_sha256)
    config_path = output/'comparison_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError('Comparison config differs; use a fresh output directory')
    if not config_path.exists():
        write_json_atomic(config_path, settings)
    torch.set_num_threads(1)
    for sampler in SAMPLERS:
        folder = output/sampler
        if (folder/'result.json').exists():
            continue
        if folder.exists():
            raise ValueError(f'Incomplete run preserved; do not overwrite: {folder}')
        print('START', sampler, 'Adam', adam_steps, 'L-BFGS external', lbfgs_steps, flush=True)
        result = run(DEFINITION, 'CUSP_EXACT', folder, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
            lr=settings['lr'], width=settings['width'], depth=settings['depth'], activation=settings['activation'],
            weights=dict(pde=1.), points=dict(n_interior=512,n_layer=256,n_face=128,n_boundary=128),
            init_seed=0, sampling_seed=11, log_every=400, reference=reference,
            reference_label='FEM P2 level 1, same physics and evaluation points', device='cpu', dtype='float64',
            input_normalization='unit_box_v1', sampling=sampler, resample_every=0,
            validation_every=100, accuracy_stop=None, extra_lbfgs_steps=0, full_audit=True,
            command=['annulus_cusp_budget_comparison',sampler,str(adam_steps),str(lbfgs_steps)])
        assert result['status'] == 'completed', (sampler, result.get('error'))
        print('DONE', sampler, 'seconds', result['elapsed_s'],
            'NRMSE_FEM', result['evaluation']['nrmse_pct'], flush=True)
        write_outputs(output, settings)
    return write_outputs(output, settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adam-steps', type=int, default=8000)
    parser.add_argument('--lbfgs-steps', type=int, default=200)
    args = parser.parse_args()
    summary = run_comparison(args.output,args.adam_steps,args.lbfgs_steps)
    print('COMPLETE',len(summary['results']),'sampler arms',flush=True)


if __name__ == '__main__':
    main()