"""Re-score saved evaluations: Tmax error at the reference hotspot site, with doubled acceptance criteria."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CRITERIA = dict(nrmse_pct=10., Tmax_rise_pct=10., region_rmse_rise_pct=10., balance_pct=4.,
    interface_T_jump_max_K=.2, interface_flux_rms_pct=2.)
TOLERANCE = dict(region_rmse_rise_pct=2.)
ZERO_RISE_ABSOLUTE_K = .2


def reference_site(arrays, evaluation):
    """Reference Tmax value, predicted T at that same point, and its coordinates."""
    if 'annulus_boundary' in evaluation:
        xs, preds, refs = [arrays['xy']], [arrays['T']], [arrays['T_reference']]
        for key in arrays:
            if key.startswith('annulus_') and key.endswith('_xy'):
                side = key[len('annulus_'):-len('_xy')]
                xs.append(arrays[key])
                preds.append(arrays[f'annulus_{side}_T'])
                refs.append(arrays[f'annulus_{side}_T_reference'])
        X, T, R = np.concatenate(xs), np.concatenate(preds), np.concatenate(refs)
        index = int(np.argmax(R))
        return float(R[index]), float(T[index]), X[index].tolist()
    xy, T, ref = arrays['xy'], arrays['T'], arrays['T_reference']
    hot = evaluation['hotspot_ref_xy']
    index = int(np.argmin(np.hypot(xy[:, 0] - hot[0], xy[:, 1] - hot[1])))
    return float(ref[index]), float(T[index]), xy[index].tolist()


def rescore(evaluation, arrays):
    tmax_ref, tmax_pred_at_site, site_xy = reference_site(arrays, evaluation)
    level = evaluation['level_C']
    error_K = abs(tmax_pred_at_site - tmax_ref)
    if 'annulus_boundary' in evaluation:
        normalization = abs(tmax_ref - level)
    else:
        normalization = evaluation['temperature_normalization_K']
    error_rise_pct = 100 * error_K / normalization if normalization else None
    worst_region = max(evaluation['region_rmse_rise_pct'].values())
    raw = {}
    raw['nrmse_pct'] = evaluation['nrmse_pct'] <= CRITERIA['nrmse_pct']
    raw['Tmax_rise_pct'] = error_rise_pct is not None and error_rise_pct <= CRITERIA['Tmax_rise_pct']
    raw['region_rmse_rise_pct'] = worst_region <= CRITERIA['region_rmse_rise_pct']
    if evaluation.get('temperature_normalization') == 'characteristic_scale':
        raw['zero_rise_absolute_error_K'] = evaluation['max_error_K'] <= ZERO_RISE_ABSOLUTE_K
    balance = evaluation['balance']
    raw['balance_pct'] = (balance['global_derived_pct'] <= CRITERIA['balance_pct'] and
                          max(balance['region_derived_pct'].values()) <= CRITERIA['balance_pct'])
    interfaces = evaluation.get('interfaces') or []
    if interfaces:
        raw['interface_T_jump_max_K'] = max(i['T_jump_max_K'] for i in interfaces) <= CRITERIA['interface_T_jump_max_K']
        raw['interface_flux_rms_pct'] = max(i['flux_jump_rms_pct'] for i in interfaces) <= CRITERIA['interface_flux_rms_pct']
    gates = dict(raw)
    if not raw['region_rmse_rise_pct']:
        gates['region_rmse_rise_pct'] = worst_region - CRITERIA['region_rmse_rise_pct'] < TOLERANCE['region_rmse_rise_pct']
    if all(raw.values()):
        state = 'accepted'
    elif all(gates.values()):
        state = 'accepted_with_tolerance'
    else:
        state = 'rejected'
    return dict(Tmax_ref_C=tmax_ref, Tmax_pred_at_ref_site_C=tmax_pred_at_site, ref_site_xy=site_xy,
        error_Tmax_at_site_K=error_K, error_rise_site_pct=error_rise_pct,
        T_jump_max_K=max((i['T_jump_max_K'] for i in interfaces), default=None),
        flux_jump_rms_pct=max((i['flux_jump_rms_pct'] for i in interfaces), default=None),
        worst_region_rmse_rise_pct=worst_region, balance_pct=balance['global_derived_pct'],
        state=state, accepted=state.startswith('accepted'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    summary = json.loads((campaign / 'summary.json').read_text(encoding='utf-8'))
    rows = {}
    for label, run in summary['runs'].items():
        result_path = campaign / label / 'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding='utf-8'))
        evaluation = result['evaluation']
        with np.load(campaign / label / 'evaluation.npz') as data:
            arrays = {key: data[key] for key in data.files}
        row = dict(case=run['case'], seed=run['seed'], wall_s=run['wall_s'],
            termination=(result.get('accuracy_stop') or {}).get('termination_reason'),
            nrmse_pct=evaluation['nrmse_pct'], error_Tmax_old_K=evaluation['error_Tmax_abs_K'],
            accepted_old=(evaluation.get('acceptance') or {}).get('accepted'),
            Tmax_pred_old_C=evaluation.get('Tmax_pred_C'), Tmax_ref_old_C=evaluation.get('Tmax_ref_C'))
        row.update(rescore(evaluation, arrays))
        rows[label] = row
    out = campaign / 'tmax_site_reanalysis.json'
    out.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(rows, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
