"""Solver-independent metrics (numpy only): the same functions score PINN, FEM and later variants.

Temperature rise is measured from problem.T_level, never from absolute Celsius:
  rise_ref = max(T_ref over the hot set) - T_level.
The hot set is the conductor region(s) for cables and the whole domain for manufactured cases.
"""
import numpy as np

CRITERIA = dict(nrmse_pct=5., Tmax_rise_pct=5., region_rmse_rise_pct=5., balance_pct=2.,
    interface_T_jump_max_K=.1, interface_flux_rms_pct=1.)
PERCENTILES = (50, 90, 95, 99)


def hot_mask(labels, conductor_ids):
    return np.isin(labels, list(conductor_ids)) if conductor_ids else np.ones(len(labels), dtype=bool)


def temperature_summary(T, xy, labels, weights, names, conductor_ids, hot):
    """T_max, hotspot coordinates and the area-weighted mean temperature of each conductor."""
    index = np.flatnonzero(hot)[int(np.argmax(T[hot]))]
    summary = dict(Tmax_C=float(T[index]), hotspot_xy=[float(xy[index, 0]), float(xy[index, 1])])
    summary['conductor_mean_C'] = {names[i]: float(np.average(T[labels == i], weights=weights[labels == i])) for i in conductor_ids}
    return summary


def error_report(T, reference, xy, labels, weights, names, level, conductor_ids):
    hot = hot_mask(labels, conductor_ids)
    error = T - reference
    pred = temperature_summary(T, xy, labels, weights, names, conductor_ids, hot)
    ref = temperature_summary(reference, xy, labels, weights, names, conductor_ids, hot)
    rise_ref = ref['Tmax_C'] - level
    rise_pred = pred['Tmax_C'] - level
    rmse = float(np.sqrt(np.average(error ** 2, weights=weights)))
    absolute = np.abs(error)
    report = dict(level_C=float(level), rise_ref_K=float(rise_ref), Tmax_pred_C=pred['Tmax_C'], Tmax_ref_C=ref['Tmax_C'],
        error_Tmax_abs_K=float(abs(pred['Tmax_C'] - ref['Tmax_C'])), error_rise_pct=float(100 * abs(rise_pred - rise_ref) / rise_ref),
        hotspot_pred_xy=pred['hotspot_xy'], hotspot_ref_xy=ref['hotspot_xy'],
        hotspot_distance_m=float(np.hypot(*(np.array(pred['hotspot_xy']) - np.array(ref['hotspot_xy'])))),
        rmse_K=rmse, nrmse_pct=float(100 * rmse / rise_ref), max_error_K=float(absolute.max()),
        error_percentiles_K={str(p): float(np.percentile(absolute, p)) for p in PERCENTILES},
        conductor_mean_pred_C=pred['conductor_mean_C'], conductor_mean_ref_C=ref['conductor_mean_C'],
        conductor_mean_error_K={n: pred['conductor_mean_C'][n] - ref['conductor_mean_C'][n] for n in pred['conductor_mean_C']},
        region_rmse_K={}, region_rmse_rise_pct={}, region_max_error_K={})
    for i, name in enumerate(names):
        select = labels == i
        if select.any():
            value = float(np.sqrt(np.mean(error[select] ** 2)))
            report['region_rmse_K'][name] = value
            report['region_rmse_rise_pct'][name] = float(100 * value / rise_ref)
            report['region_max_error_K'][name] = float(absolute[select].max())
    return report


def acceptance(report, criteria=CRITERIA):
    """Proposed gates (to be confirmed by the user before Phase 6); returns per-gate booleans and a verdict."""
    gates = {}
    if 'nrmse_pct' in report:
        gates['nrmse_pct'] = report['nrmse_pct'] <= criteria['nrmse_pct']
        gates['Tmax_rise_pct'] = report['error_rise_pct'] <= criteria['Tmax_rise_pct']
        gates['region_rmse_rise_pct'] = max(report['region_rmse_rise_pct'].values()) <= criteria['region_rmse_rise_pct']
    gates['balance_pct'] = report['balance']['global_derived_pct'] <= criteria['balance_pct'] and max(report['balance']['region_derived_pct'].values()) <= criteria['balance_pct']
    if report['interfaces']:
        gates['interface_T_jump_max_K'] = max(i['T_jump_max_K'] for i in report['interfaces']) <= criteria['interface_T_jump_max_K']
        gates['interface_flux_rms_pct'] = max(i['flux_jump_rms_pct'] for i in report['interfaces']) <= criteria['interface_flux_rms_pct']
    return dict(gates=gates, accepted=bool(all(gates.values())) if 'nrmse_pct' in report else None,
        state='no_reference' if 'nrmse_pct' not in report else ('accepted' if all(gates.values()) else 'rejected'))
