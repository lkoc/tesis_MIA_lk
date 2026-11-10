"""Long V2-style convergence campaign for the nearest V3 PINN models."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.trainer import run

CASES = ('mms_sine', 'cable_disk')
ARCHITECTURES = ('M1', 'DCSNN')
ADAM_STEPS, LBFGS_STEPS, REFRESH = 8000, 20, 100
RESULTS = ROOT / 'results' / 'long_v2_focused_campaign_20251025b'
WEIGHTS = dict(pde=1.0, energy=100.0)
ACCURACY_STOP = dict(error_Tmax_abs_K=0.01, nrmse_pct=0.1, balance_pct=0.5, every=100, after_phase='adam')


def supplied_reference(problem, fem_path):
    """Interpolate FEM values onto the deterministic PINN evaluation points for stopping only."""
    fem = np.load(fem_path)
    xy, _, _ = problem.evaluation()
    linear = LinearNDInterpolator(fem['dof_xy'], fem['dof_T'], fill_value=np.nan)
    values = np.asarray(linear(xy), dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        nearest = NearestNDInterpolator(fem['dof_xy'], fem['dof_T'])
        values[missing] = nearest(xy[missing])
    return values


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for case_id in CASES:
        problem_path = ROOT / 'problems' / case_id / 'problem.json'
        fem_path = ROOT / 'results' / 'fem_generic' / case_id / 'fem_l1.npz'
        for architecture in ARCHITECTURES:
            key = f'{case_id}:{architecture}'
            run_dir = RESULTS / case_id / architecture
            try:
                problem = GenericProblem(load_problem(problem_path).doc)
                reference = supplied_reference(problem, fem_path)
                result = run(problem, architecture, run_dir, adam_steps=ADAM_STEPS, lbfgs_steps=LBFGS_STEPS,
                    width=32, depth=3, lr=0.002, weights=WEIGHTS, sampling='mixed', focused=True,
                    focus_criterion='residual', focus_surfaces=False, resample_every=REFRESH,
                    reference=reference, reference_label=f'FEM interpolated for {case_id}',
                    accuracy_stop=ACCURACY_STOP, extra_lbfgs_steps=5)
                summary[key] = dict(status=result.get('status'), final_loss=result.get('final_loss'),
                    final_parts=result.get('final_parts'), accuracy_stop=result.get('accuracy_stop'),
                    evaluation=result.get('evaluation'), elapsed_s=result.get('elapsed_s'))
            except Exception:
                summary[key] = dict(status='failed', error=traceback.format_exc())
            (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=float), flush=True)


if __name__ == '__main__':
    main()
