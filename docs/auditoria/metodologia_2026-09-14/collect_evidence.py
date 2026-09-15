"""Read-only review of existing evidence; write only into this audit directory."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'Benchmarks'
sys.path.insert(0, str(ROOT))
from Benchmarks.cases import cases, fingerprint
from Benchmarks.validate_artifacts import reference_path, sources

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    catalog = cases()
    selection = json.loads((BASE / 'selection.json').read_text(encoding='utf-8'))
    rows = []
    for family in ['cases', 'ampacity_cases']:
        for name, directory in selection[family].items():
            results = []
            for path in sorted((BASE / directory).glob('pinn_seed*.json')):
                r = json.loads(path.read_text(encoding='utf-8'))
                meta = r['metadata']
                assert meta['case_sha256'] == fingerprint(catalog[name])
                ref = reference_path(r['evaluation'])
                assert digest(ref.with_suffix('.json')) == r['evaluation']['fem_json_sha256']
                sources(path.parent, meta['source_sha256'])
                sources(path.parent, r['evaluation']['source_sha256'])
                with np.load(path.with_suffix('.npz')) as d, np.load(ref) as fd:
                    assert np.allclose(d['xy'], fd['xy'], rtol=0, atol=1e-14)
                    rmse = float(np.sqrt(np.mean((d['T'] - fd['T'])**2)))
                    assert np.isclose(rmse, r['rmse_fem_K'], rtol=1e-10, atol=1e-10)
                    spread = float(np.max(np.ptp(d['surface_T'], axis=-1))) if d['surface_T'].size else None
                key = 'ampacity_criteria_pass' if family == 'ampacity_cases' else 'thermal_criteria_pass'
                boundaries = [b['relative_power_error_pct'] for b in r['boundary_diagnostics'] if 'relative_power_error_pct' in b]
                result = dict(seed=meta['seed'], accepted=bool(r[key]), rmse_K=rmse,
                    Tmax_C=r['Tmax_C'], error_Tmax_K=r['error_Tmax_K'], nrmse_pct=r['nrmse_fem_pct'],
                    balance_pct=r['balance_pct'], surface_temperature_range_K=spread,
                    max_cable_power_error_pct=max(boundaries, default=None),
                    interface_T_max_K=r.get('interface_T_max_K'), interface_flux_max_W_m2=r.get('interface_flux_max_W_m2'),
                    current_A=r.get('current_A'), error_current_pct=r.get('error_current_pct'),
                    n_interior=meta['n_interior'], source_record=str(path.relative_to(ROOT)),
                    record_sha256=digest(path), field_sha256=digest(path.with_suffix('.npz')))
                results.append(result)
            rows.append(dict(family=family, case=name, directory=directory,
                accepted=sum(r['accepted'] for r in results), total=len(results),
                median_rmse_K=float(np.median([r['rmse_K'] for r in results])),
                Tmax_seed_range_K=float(np.ptp([r['Tmax_C'] for r in results])), results=results))
    fem_rows=[]
    for name, c in catalog.items():
        for folder, stem in [('results','fem')] + ([('coupled_results','fem'),('coupled_results','fem_ampacity')] if c['kind']=='cable' else []):
            levels=[json.loads((BASE/folder/name/f'{stem}_l{i}.json').read_text()) for i in range(3)]
            for r in levels:
                assert r['case_sha256']==fingerprint(c)
            fem_rows.append(dict(case=name, family=folder+'/'+stem, Tmax_C=[r['Tmax_C'] for r in levels],
                fine_difference_K=abs(levels[2]['Tmax_C']-levels[1]['Tmax_C']), fine_balance_pct=levels[2]['balance_pct']))
    resolution=[]
    for name in ['mms_smooth_2d','xlpe_single']:
        for label in ['N384','W32','N1536']:
            records=[json.loads(p.read_text()) for p in (BASE/'comparisons'/('resolution_'+label)/name).glob('pinn_seed*.json')]
            resolution.append(dict(case=name, label=label, n=[r['metadata']['n_interior'] for r in records],
                median_rmse_K=float(np.median([r['rmse_fem_K'] for r in records]))))
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        methodology_sha256=digest(ROOT/'docs/METODOLOGIA_PROPUESTA_2026-09-14.md'),
        verified_selected_records=sum(r['total'] for r in rows), selection_sha256=digest(BASE/'selection.json'),
        scope='Recomputed selected stored-field RMSE and checked case/reference/source hashes; not a retraining or explicit-layer validation.',
        summary=rows, fem=fem_rows, resolution=resolution)
    (OUT/'evidence.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    for r in rows:
        print(f"{r['family']:15} {r['case']:22} {r['accepted']}/{r['total']} RMSE={r['median_rmse_K']:.6f} K seed_range={r['Tmax_seed_range_K']:.5f} K")
    print('Verified records:',output['verified_selected_records'])

if __name__=='__main__':
    main()
