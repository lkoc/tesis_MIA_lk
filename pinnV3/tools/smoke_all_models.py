"""Short cross-architecture smoke on MMS and cable after the shared-feature adaptation."""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_v3.problem.repository import load_problem
from pinn_v3.solvers.pinn.generic_problem import GenericProblem
from pinn_v3.solvers.pinn.trainer import run

CASES = ('mms_sine', 'cable_disk')
MODELS = ('M0', 'M1', 'M2', 'DCSNN', 'M0R', 'CUSP', 'CUSPR', 'FBPINN')
RESULTS = ROOT / 'results' / 'all_models_smoke_20251026'


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for case_id in CASES:
        problem = GenericProblem(load_problem(ROOT / 'problems' / case_id / 'problem.json').doc)
        for architecture in MODELS:
            key = f'{case_id}:{architecture}'
            output = RESULTS / case_id / architecture
            try:
                result = run(problem, architecture, output, adam_steps=200, lbfgs_steps=0,
                    width=24, depth=2, lr=0.002, weights={'pde': 1.0, 'energy': 100.0},
                    sampling='mixed', focused=True, focus_criterion='residual',
                    focus_surfaces=False, resample_every=100, full_audit=False)
                summary[key] = dict(status=result.get('status'), final_loss=result.get('final_loss'),
                    final_parts=result.get('final_parts'), evaluation=result.get('evaluation'))
            except Exception:
                summary[key] = dict(status='failed', error=traceback.format_exc())
            (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=float), flush=True)


if __name__ == '__main__':
    main()
