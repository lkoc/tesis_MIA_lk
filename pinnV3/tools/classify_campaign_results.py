"""Apply the thesis qualification thresholds to every existing V3 campaign summary."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUTPUT_JSON = ROOT / 'CAMPAIGN_CLASSIFICATION_2026-10-27.json'
OUTPUT_MD = ROOT / 'CAMPAIGN_CLASSIFICATION_2026-10-27.md'
CRITERIA = dict(error_Tmax_abs_K=0.05, nrmse_pct=0.25, balance_pct=0.5)


def metric_record(key, record):
    if not isinstance(record, dict):
        return None
    direct = {name: record.get(name) for name in CRITERIA}
    if all(value is not None for value in direct.values()):
        return direct
    evaluation = record.get('evaluation')
    if isinstance(evaluation, dict):
        balance = evaluation.get('balance') or {}
        values = dict(error_Tmax_abs_K=evaluation.get('error_Tmax_abs_K'),
                      nrmse_pct=evaluation.get('nrmse_pct'),
                      balance_pct=balance.get('global_derived_pct'))
        if all(value is not None for value in values.values()):
            return values
    stages = record.get('stages')
    if stages:
        for stage in reversed(stages):
            values = stage.get('metrics', {})
            if all(values.get(name) is not None for name in CRITERIA):
                return {name: values[name] for name in CRITERIA}
    return None


def flatten(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            path = f'{prefix}:{key}' if prefix else str(key)
            metrics = metric_record(path, child)
            if metrics is not None:
                yield path, metrics
            else:
                yield from flatten(child, path)


def main():
    rows = []
    for path in sorted(RESULTS.rglob('summary.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        for key, metrics in flatten(data):
            qualification = all(metrics[name] < CRITERIA[name] for name in CRITERIA)
            rows.append(dict(source=str(path.relative_to(ROOT)), key=key, metrics=metrics,
                             qualification='qualified' if qualification else 'requires_more_steps',
                             gates={name: metrics[name] < CRITERIA[name] for name in CRITERIA}))
    counts = Counter(row['qualification'] for row in rows)
    payload = dict(criteria=CRITERIA, total=len(rows), counts=dict(counts), rows=rows)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=float), encoding='utf-8')
    lines = ['# Campaign classification 2026-10-27', '', 'Applied to existing summaries without changing their raw results.', '',
             '| Criterion | Threshold |', '|---|---:|']
    lines.extend(f'| `{name}` | `< {value}` |' for name, value in CRITERIA.items())
    lines.extend(['', f"Qualified: **{counts['qualified']}**; requires more steps: **{counts['requires_more_steps']}**.", '',
                   '| Source | Run | Tmax error (K) | NRMSE (%) | Balance (%) | Qualification |',
                   '|---|---|---:|---:|---:|---|'])
    for row in rows:
        m = row['metrics']
        lines.append(f"| `{row['source']}` | `{row['key']}` | {m['error_Tmax_abs_K']:.6g} | {m['nrmse_pct']:.6g} | {m['balance_pct']:.6g} | {row['qualification']} |")
    OUTPUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(total=len(rows), counts=dict(counts)), indent=2), flush=True)


if __name__ == '__main__':
    main()
