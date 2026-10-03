"""Phase 3 study: FEniCSx reference for V0-V2 and the Neumann/Robin cable checks (3 or more meshes).

Writes only to a new directory results/phase3/<study_id>; an existing directory is never reused.
Needs the WSL2 FEniCSx environment (see solvers/fem/adapter.py). Nothing here trains or fits anything.
"""
import argparse
import math
import sys
from pathlib import Path

from pinn_v2.cases.verification import coaxial_robin, coaxial_dirichlet, symmetric_pair
from pinn_v2.manifest.run_manifest import build_manifest, write_manifest, write_json_atomic
from pinn_v2.solvers.fem import adapter

ROOT = Path(__file__).resolve().parents[2]
MMS_CASES = ('mms_constant', 'mms_interface', 'mms_robin')


def observed_orders(errors, sizes):
    return [math.log(errors[i] / errors[i + 1]) / math.log(sizes[i] / sizes[i + 1]) for i in range(len(errors) - 1)]


def mms_block(output, case, levels):
    records = adapter.run_mms_fem(output.joinpath('mms', case), case, levels)
    h = [r['h'] for r in records]
    block = dict(case=case, records=records)
    for key in ('l2_error_K', 'max_nodal_error_K', 'rmse_eval_K'):
        block['order_' + key] = observed_orders([r[key] for r in records], h)
    return block


def cable_block(output, definition, levels):
    path = output.joinpath('definitions', f'{definition.id}.json')
    write_json_atomic(path, definition.to_dict())
    records = adapter.run_cable_fem(output.joinpath('cable', definition.id), levels, definition=path)
    return dict(case=definition.id, definition_sha256=definition.sha256, records=records)


def convergence_of_cable(block, key):
    records = [r for r in block['records'] if key in r]
    values = [r[key] for r in records]
    sizes = [r['ndofs'] ** -.5 for r in records]
    return dict(values=values, orders=observed_orders(values, sizes) if len(values) > 1 else [])


def symmetry_comparison(full, half):
    """Half domain with a q.n = 0 symmetry side against the full two-cable domain, per level."""
    rows = []
    for f, h in zip(full['records'], half['records']):
        full_conductors = [r for r in f['regions'] if 'T_mean_C' in r]
        half_conductor = [r for r in h['regions'] if 'T_mean_C' in r][0]
        rows.append(dict(level=f['level'],
            source_ratio_full_over_half=f['source_W_m'] / h['source_W_m'],
            top_flux_ratio_full_over_half=f['boundary_flux_W_m']['top'] / h['boundary_flux_W_m']['top'],
            symmetry_side_flux_half_W_m=h['boundary_flux_W_m']['left'],
            T_mean_mirror_difference_K=abs(full_conductors[0]['T_mean_C'] - full_conductors[1]['T_mean_C']),
            T_mean_half_minus_full_K=half_conductor['T_mean_C'] - full_conductors[1]['T_mean_C'],
            Tmax_half_minus_full_K=h['Tmax_C'] - f['Tmax_C']))
    return rows


def run_study(study_id, levels_cable=(0, 1, 2), levels_mms=(0, 1, 2, 3), root=ROOT):
    output = Path(root).joinpath('results', 'phase3', study_id)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new study id')
    output.mkdir(parents=True)
    summary = dict(study_id=study_id, levels_cable=list(levels_cable), levels_mms=list(levels_mms))
    summary['mms'] = {case: mms_block(output, case, levels_mms) for case in MMS_CASES}
    legacy = adapter.run_cable_fem(output.joinpath('cable', 'coaxial_full'), levels_cable, case='coaxial_full')
    summary['coaxial_full'] = dict(case='coaxial_full', records=legacy)
    summary['coaxial_bc_dirichlet'] = cable_block(output, coaxial_dirichlet(), levels_cable)
    summary['coaxial_robin'] = cable_block(output, coaxial_robin(), levels_cable)
    summary['symmetric_full'] = cable_block(output, symmetric_pair(False), levels_cable)
    summary['symmetric_half'] = cable_block(output, symmetric_pair(True), levels_cable)
    summary['symmetry'] = symmetry_comparison(summary['symmetric_full'], summary['symmetric_half'])
    for name in ('coaxial_full', 'coaxial_bc_dirichlet', 'coaxial_robin'):
        summary[name]['rmse_exact_convergence'] = convergence_of_cable(summary[name], 'rmse_exact_K')
    write_json_atomic(output.joinpath('summary.json'), summary)
    manifest = build_manifest('phase3_fem_reference', [sys.executable, '-m', 'pinn_v2.experiments.fem_reference', '--study-id', study_id],
        seeds=dict(evaluation_points=20260910), fem_environment=adapter.environment(),
        extra=dict(levels_cable=list(levels_cable), levels_mms=list(levels_mms), fem_labels_used_for_training=False))
    write_manifest(output, manifest)
    write_report(output, summary)
    return output, summary


def write_report(output, summary):
    lines = [f"# Phase 3 FEM reference, study {summary['study_id']}", '', '## Manufactured cases (unit square, P2)', '',
        '| case | level | dofs | L2 error K | max nodal K | balance % | L2 order |', '|---|---|---|---|---|---|---|']
    for case, block in summary['mms'].items():
        orders = [None] + block['order_l2_error_K']
        for r, o in zip(block['records'], orders):
            order = '' if o is None else f'{o:.2f}'
            lines.append(f"| {case} | {r['level']} | {r['ndofs']} | {r['l2_error_K']:.3e} | {r['max_nodal_error_K']:.3e} | {r['balance_pct']:.3e} | {order} |")
    lines += ['', '## Concentric cable (exact radial solution)', '', '| case | level | dofs | rmse K | Tmax error K | balance % |', '|---|---|---|---|---|---|']
    nan = float('nan')
    for name in ('coaxial_full', 'coaxial_bc_dirichlet', 'coaxial_robin'):
        for r in summary[name]['records']:
            lines.append(f"| {name} | {r['level']} | {r['ndofs']} | {r.get('rmse_exact_K', nan):.3e} | {r.get('error_Tmax_exact_K', nan):.3e} | {r['balance_pct']:.3e} |")
    lines += ['', '## Symmetry cross-check (half domain with q.n = 0 side against full domain)', '',
        '| level | source ratio | top flux ratio | symmetry-side flux W/m | mirror dT mean K | half-full dT mean K |', '|---|---|---|---|---|---|']
    for r in summary['symmetry']:
        lines.append(f"| {r['level']} | {r['source_ratio_full_over_half']:.6f} | {r['top_flux_ratio_full_over_half']:.6f} | {r['symmetry_side_flux_half_W_m']:.2e} | {r['T_mean_mirror_difference_K']:.2e} | {r['T_mean_half_minus_full_K']:.2e} |")
    path = output.joinpath('convergence.md')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-id', required=True)
    arguments = parser.parse_args()
    out, _ = run_study(arguments.study_id)
    print(out)


if __name__ == '__main__':
    main()