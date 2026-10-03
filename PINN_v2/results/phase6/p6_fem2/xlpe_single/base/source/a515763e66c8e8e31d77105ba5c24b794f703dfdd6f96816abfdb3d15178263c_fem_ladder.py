"""Phase 6 study: FEniCSx reference for V3-V5 with a mesh convergence study and a domain-size study.

A FEM result is called a reference only if the last two meshes agree within CONVERGENCE_PCT of the temperature rise
(proposed gate, not confirmed by the user). The domain study doubles the half-width and the depth with the same
mesh level; it documents how much the controlled reduced domain changes the answer. Nothing here trains a model.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

from pinn_v2.cases.catalog import cases as legacy_cases
from pinn_v2.cases.definition import CaseDefinition
from pinn_v2.manifest.run_manifest import build_manifest, write_manifest, write_json_atomic
from pinn_v2.solvers.fem import adapter

ROOT = Path(__file__).resolve().parents[2]
V2_DIR = Path(__file__).resolve().parents[1].joinpath('cases', 'v2')
LADDER = dict(V3=('xlpe_single',), V4=('xlpe_backfill', 'xlpe_discrete_layers', 'aras_flat'), V5=('kim_v5_summer',))
CONVERGENCE_PCT = 0.5
DOMAIN_PCT = 2.0
LEVELS = (0, 1, 2, 3)
# Level 3 of the 55-region Kim case has about 10 million dofs and was killed by the WSL memory limit (study p6_fem, exit code 9).
CASE_LEVELS = dict(kim_v5_summer=(0, 1, 2))
DOMAIN_LEVEL = 2


def is_definition(case):
    return V2_DIR.joinpath(f'{case}.json').exists()


def widened(bounds, factor=2.):
    x0, x1, y0, y1 = bounds
    return [factor * x0, factor * x1, factor * y0, y1]


def conductor_means(record):
    return [r['T_mean_C'] for r in record['regions'] if 'T_mean_C' in r]


def convergence(records, level_C):
    rise = records[-1]['Tmax_C'] - level_C
    steps = [abs(b['Tmax_C'] - a['Tmax_C']) for a, b in zip(records, records[1:])]
    last = steps[-1]
    return dict(Tmax_C=[r['Tmax_C'] for r in records], ndofs=[r['ndofs'] for r in records], successive_Tmax_change_K=steps,
        rise_K=rise, last_change_pct_of_rise=100 * last / rise, converged=100 * last / rise <= CONVERGENCE_PCT and (len(steps) < 2 or steps[-1] <= steps[-2]),
        conductor_mean_C=[conductor_means(r) for r in records])


def run_case(output, case, levels):
    folder = Path(output).joinpath(case)
    if is_definition(case):
        path = V2_DIR.joinpath(f'{case}.json')
        definition = CaseDefinition.load(path)
        records = adapter.run_cable_fem(folder.joinpath('base'), levels, definition=path)
        wide = json.loads(path.read_text(encoding='utf-8'))
        wide['id'] = case + '_wide'
        wide['physical']['id'] = case + '_wide'
        wide['physical']['bounds'] = widened(wide['physical']['bounds'])
        wide_path = folder.joinpath('wide_definition.json')
        write_json_atomic(wide_path, wide)
        wide_record = adapter.run_cable_fem(folder.joinpath('wide'), [DOMAIN_LEVEL], definition=wide_path)[0]
        level_C, sha = definition.physical['T0'], definition.sha256
    else:
        base = legacy_cases()[case]
        records = adapter.run_cable_fem(folder.joinpath('base'), levels, case=case)
        wide = copy.deepcopy(base)
        wide['id'] = case + '_wide'
        wide['bounds'] = widened(base['bounds'])
        wide_path = folder.joinpath('wide_case.json')
        write_json_atomic(wide_path, wide)
        wide_record = adapter.run_cable_fem(folder.joinpath('wide'), [DOMAIN_LEVEL], case_file=wide_path)[0]
        level_C, sha = base['T0'], records[0].get('physics_sha256')
    block = convergence(records, level_C)
    reference = next(r for r in records if r['level'] == DOMAIN_LEVEL)
    change = abs(wide_record['Tmax_C'] - reference['Tmax_C'])
    block.update(case=case, T_level_C=level_C, physics_sha256=sha, records=records,
        domain=dict(level=DOMAIN_LEVEL, base_bounds=records[0]['specification'].get('bounds'), Tmax_base_C=reference['Tmax_C'], Tmax_wide_C=wide_record['Tmax_C'],
            change_K=change, change_pct_of_rise=100 * change / (reference['Tmax_C'] - level_C), domain_insensitive=100 * change / (reference['Tmax_C'] - level_C) <= DOMAIN_PCT,
            wide_ndofs=wide_record['ndofs']))
    return block


def run_study(study_id, cases=None, levels=LEVELS, root=ROOT):
    output = Path(root).joinpath('results', 'phase6', study_id)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new study id')
    output.mkdir(parents=True)
    selected = [(v, c) for v, names in LADDER.items() for c in names if cases is None or c in cases]
    summary = dict(study_id=study_id, levels=list(levels), convergence_gate_pct=CONVERGENCE_PCT, domain_gate_pct=DOMAIN_PCT, gates_status='proposed, not confirmed by the user',
        cases={})
    for level, case in selected:
        try:
            block = run_case(output, case, CASE_LEVELS.get(case, levels))
            block['ladder'] = level
            print(case, 'converged' if block['converged'] else 'NOT converged', f"{block['last_change_pct_of_rise']:.3f}%", 'domain change', f"{block['domain']['change_pct_of_rise']:.2f}%", flush=True)
        except Exception as exc:
            block = dict(case=case, ladder=level, failed=True, converged=False, error=str(exc)[-1500:])
            print(case, 'FAILED', str(exc)[-200:], flush=True)
        summary['cases'][case] = block
        write_json_atomic(output.joinpath('summary.json'), summary)
    write_json_atomic(output.joinpath('summary.json'), summary)
    manifest = build_manifest('phase6_fem_ladder', [sys.executable, '-m', 'pinn_v2.experiments.fem_ladder', '--study-id', study_id], seeds=dict(), fem_environment=adapter.environment(),
        extra=dict(levels=list(levels), cases=[c for _, c in selected], fem_labels_used_for_training=False))
    write_manifest(output, manifest)
    write_report(output, summary)
    return output, summary


def write_report(output, summary):
    lines = [f"# Phase 6 FEM reference, study {summary['study_id']}", '', f"Gates ({summary['gates_status']}): mesh change <= {summary['convergence_gate_pct']}% of rise, domain change <= {summary['domain_gate_pct']}% of rise.", '',
        '| ladder | case | Tmax per level C | last change % of rise | mesh converged | Tmax wide domain C | domain change % of rise | domain insensitive |', '|---|---|---|---|---|---|---|---|']
    for case, b in summary['cases'].items():
        if b.get('failed'):
            lines.append(f"| {b['ladder']} | {case} | FAILED | | False | | | |")
            continue
        d = b['domain']
        lines.append(f"| {b['ladder']} | {case} | {', '.join(f'{t:.4f}' for t in b['Tmax_C'])} | {b['last_change_pct_of_rise']:.3f} | {b['converged']} | {d['Tmax_wide_C']:.4f} | {d['change_pct_of_rise']:.2f} | {d['domain_insensitive']} |")
    Path(output).joinpath('convergence.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-id', required=True)
    parser.add_argument('--cases', nargs='*')
    arguments = parser.parse_args()
    out, _ = run_study(arguments.study_id, arguments.cases)
    print(out)


if __name__ == '__main__':
    main()
