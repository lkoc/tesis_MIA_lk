"""Solver-independent metrics (numpy only): the same functions score PINN, FEM and later variants.

Temperature rise is measured from problem.T_level, never from absolute Celsius:
  rise_ref = max(T_ref over the hot set) - T_level.
The hot set is the conductor region(s) for cables and the whole domain for manufactured cases.
"""
import numpy as np

CRITERIA = dict(nrmse_pct=10., Tmax_rise_pct=10.)
PERCENTILES = (50, 90, 95, 99)


def hot_mask(labels, conductor_ids):
    return np.isin(labels, list(conductor_ids)) if conductor_ids else np.ones(len(labels), dtype=bool)


def temperature_summary(T, xy, labels, weights, names, conductor_ids, hot):
    """T_max, hotspot coordinates and the area-weighted mean temperature of each conductor."""
    index = np.flatnonzero(hot)[int(np.argmax(T[hot]))]
    summary = dict(Tmax_C=float(T[index]), hotspot_xy=[float(xy[index, 0]), float(xy[index, 1])])
    conductor_means = {}
    for region_id in conductor_ids:
        selected = labels == region_id
        if not selected.any():
            continue
        region_weights = weights[selected]
        average = np.average(T[selected], weights=region_weights) if region_weights.sum() > 0 else np.mean(T[selected])
        conductor_means[names[region_id]] = float(average)
    summary['conductor_mean_C'] = conductor_means
    return summary


def error_report(T, reference, xy, labels, weights, names, level, conductor_ids, temperature_scale=None):
    hot = hot_mask(labels, conductor_ids)
    error = T - reference
    pred = temperature_summary(T, xy, labels, weights, names, conductor_ids, hot)
    ref = temperature_summary(reference, xy, labels, weights, names, conductor_ids, hot)
    rise_ref = ref['Tmax_C'] - level
    rise_pred = pred['Tmax_C'] - level
    zero_rise = abs(rise_ref) <= 1e-12 * max(abs(ref['Tmax_C']), 1.)
    normalization = temperature_scale if zero_rise else abs(rise_ref)
    if normalization is None or normalization <= 0:
        raise ValueError('A positive characteristic temperature scale is required for zero-rise references')
    rmse = float(np.sqrt(np.average(error ** 2, weights=weights)))
    absolute = np.abs(error)
    site = np.flatnonzero(hot)[int(np.argmax(reference[hot]))]
    site_error = float(abs(T[site] - reference[site]))
    report = dict(level_C=float(level), rise_ref_K=float(rise_ref), Tmax_pred_C=pred['Tmax_C'], Tmax_ref_C=ref['Tmax_C'],
        error_Tmax_abs_K=float(abs(pred['Tmax_C'] - ref['Tmax_C'])), error_rise_pct=float(100 * abs(rise_pred - rise_ref) / normalization),
        error_Tmax_site_K=site_error, error_rise_site_pct=float(100 * site_error / normalization),
        temperature_normalization='characteristic_scale' if zero_rise else 'reference_rise',
        temperature_normalization_K=float(normalization),
        hotspot_pred_xy=pred['hotspot_xy'], hotspot_ref_xy=ref['hotspot_xy'],
        hotspot_distance_m=float(np.hypot(*(np.array(pred['hotspot_xy']) - np.array(ref['hotspot_xy'])))),
        rmse_K=rmse, nrmse_pct=float(100 * rmse / normalization), max_error_K=float(absolute.max()),
        error_percentiles_K={str(p): float(np.percentile(absolute, p)) for p in PERCENTILES},
        conductor_mean_pred_C=pred['conductor_mean_C'], conductor_mean_ref_C=ref['conductor_mean_C'],
        conductor_mean_error_K={n: pred['conductor_mean_C'][n] - ref['conductor_mean_C'][n] for n in pred['conductor_mean_C']},
        region_rmse_K={}, region_rmse_rise_pct={}, region_max_error_K={})
    for i, name in enumerate(names):
        select = labels == i
        if select.any():
            value = float(np.sqrt(np.mean(error[select] ** 2)))
            report['region_rmse_K'][name] = value
            report['region_rmse_rise_pct'][name] = float(100 * value / normalization)
            report['region_max_error_K'][name] = float(absolute[select].max())
    return report


def acceptance(report, criteria=CRITERIA):
    """Accept on NRMSE and reference-site Tmax error only; flux and balance are reported."""
    raw_gates = {}
    if 'nrmse_pct' in report:
        raw_gates['nrmse_pct'] = report['nrmse_pct'] <= criteria['nrmse_pct']
        raw_gates['Tmax_rise_pct'] = report['error_rise_site_pct'] <= criteria['Tmax_rise_pct']
    if 'nrmse_pct' not in report:
        state, accepted = 'no_reference', None
    elif all(raw_gates.values()):
        state, accepted = 'accepted', True
    else:
        state, accepted = 'rejected', False
    return dict(gates=dict(raw_gates), raw_gates=dict(raw_gates), criteria=dict(criteria), accepted=accepted, state=state)
