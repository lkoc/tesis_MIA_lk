"""Rewrite the checkpoint inference metrics without the historical trainer-evaluation error."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'results' / 'exact_bc_long' / 'summary.json'
target = ROOT / 'results' / 'exact_bc_long' / 'ad_hoc_summary.json'
raw = json.loads(source.read_text(encoding='utf-8'))
clean = {}
for key, record in raw.items():
    clean[key] = {
        field: record[field]
        for field in ('steps', 'lbfgs_steps', 'resample_every', 'focus_criterion', 'points',
                      'nan_points', 'rmse_K', 'max_abs_error_K', 'fem_Tmax_C', 'pinn_Tmax_C',
                      'reference_rmse_K')
        if field in record
    }
    clean[key]['status'] = 'inference_only'
    clean[key]['training_status'] = 'checkpoint_completed'
    clean[key]['note'] = 'Metrics obtained by loading latest.pt after training; no retraining.'
target.write_text(json.dumps(clean, indent=2), encoding='utf-8')
print(target)
