"""1200 Adam + 50 L-BFGS model-selection campaign on MMS and cable FEM references."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.trainer import run

CASES = ('mms_sine', 'cable_disk')
MODELS = ('M0', 'M1', 'M2', 'DCSNN', 'M0R', 'CUSP', 'CUSPR', 'FBPINN')
ADAM_STEPS, LBFGS_STEPS, REFRESH = 1200, 50, 100
RESULTS = ROOT / 'results' / 'selection_campaign_1200_50_20261026'
WEIGHTS = dict(pde=1.0, energy=100.0)


def fem_reference(problem, path):
    fem = np.load(path)
    xy, _, _ = problem.evaluation()
    values = np.asarray(LinearNDInterpolator(fem['dof_xy'], fem['dof_T'], fill_value=np.nan)(xy), dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = NearestNDInterpolator(fem['dof_xy'], fem['dof_T'])(xy[missing])
    return values


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for case_id in CASES:
        problem_path = ROOT / 'problems' / case_id / 'problem.json'
        fem_path = ROOT / 'results' / 'fem_generic' / case_id / 'fem_l1.npz'
        for architecture in MODELS:
            key = f'{case_id}:{architecture}'
            output = RESULTS / case_id / architecture
            try:
                problem = GenericProblem(load_problem(problem_path).doc)
                reference = fem_reference(problem, fem_path)
                result = run(problem, architecture, output, adam_steps=ADAM_STEPS, lbfgs_steps=LBFGS_STEPS,
                    width=32, depth=3, lr=0.002, weights=WEIGHTS, sampling='mixed', focused=True,
                    focus_criterion='residual', focus_surfaces=False, resample_every=REFRESH,
                    reference=reference, reference_label=f'FEM interpolated for {case_id}', full_audit=True)
                evaluation = result.get('evaluation') or {}
                summary[key] = dict(status=result.get('status'), final_loss=result.get('final_loss'),
                    final_parts=result.get('final_parts'), Tmax_pred_C=evaluation.get('Tmax_pred_C'),
                    error_Tmax_abs_K=evaluation.get('error_Tmax_abs_K'), nrmse_pct=evaluation.get('nrmse_pct'),
                    balance_pct=(evaluation.get('balance') or {}).get('global_derived_pct'),
                    interfaces=evaluation.get('interfaces'), elapsed_s=result.get('elapsed_s'))
            except Exception:
                summary[key] = dict(status='failed', error=traceback.format_exc())
            (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=float), flush=True)


if __name__ == '__main__':
    main()
