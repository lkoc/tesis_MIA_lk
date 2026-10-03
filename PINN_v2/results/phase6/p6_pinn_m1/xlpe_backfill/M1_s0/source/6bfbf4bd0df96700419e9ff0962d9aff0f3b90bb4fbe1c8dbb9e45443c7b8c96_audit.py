"""Independent audit of a trained PINN: dense points drawn with a seed that training never uses.

Reported per run: interface [T] and [q.n], PDE residual per region, constitutive residual (M2),
global and per-region energy balance, and the error against the reference (exact or supplied).
The balance is closed twice: with the derived flux -k grad T and, on Robin sides, with the
constitutive law h (T - T_inf). Percentages are relative to power_scale (the integral of |Q|).
"""
import numpy as np
import torch

from pinn_v2.evaluation.metrics import error_report, acceptance
from pinn_v2.physics.residuals import heat_residual, mixed_residual
from pinn_v2.solvers.pinn.objective import flux_fields

AUDIT_SEED = 909091


def _tensor(model, array):
    reference = next(model.parameters())
    return torch.as_tensor(np.asarray(array), dtype=reference.dtype, device=reference.device).clone().detach().requires_grad_(True)


def audit(model, n_interior=2048, n_layer=1024, n_face=512, n_boundary=512, seed=AUDIT_SEED):
    problem = model.problem
    names = {r.id: r.name for r in problem.regions}
    pde, constitutive = {}, {}
    for region in problem.regions:
        xy = _tensor(model, problem.sample_interior(region, n_interior if region.kind == 'soil' else n_layer, seed + 7919 * region.id, 'uniform'))
        T, predicted = model.field(region, xy)
        k = problem.conductivity(region, xy)
        source = problem.source(region, xy).detach()
        scales = problem.scales(region)
        if model.mixed:
            residual, law = mixed_residual(T, predicted, xy, k, source)
            constitutive[names[region.id]] = float(torch.sqrt(torch.mean((law / scales['flux']) ** 2)).detach())
        else:
            residual = heat_residual(T, xy, k, source)
        pde[names[region.id]] = float(torch.sqrt(torch.mean((residual / scales['divergence']) ** 2)).detach())
    derived_balance = {r.id: -problem.region_power[r.id] for r in problem.regions}
    law_balance = dict(derived_balance)
    interfaces = []
    for face in problem.faces(n_face):
        xy, normal = _tensor(model, face.xy), _tensor(model, face.normal)
        left, right = problem.regions[face.left], problem.regions[face.right]
        Tl, ql, dl, _ = flux_fields(model, left, xy)
        Tr, qr, dr, _ = flux_fields(model, right, xy)
        qln, qrn = (ql * normal).sum(1, keepdim=True), (qr * normal).sum(1, keepdim=True)
        dln, drn = (dl * normal).sum(1, keepdim=True), (dr * normal).sum(1, keepdim=True)
        scale = max(problem.scales(left)['flux'], problem.scales(right)['flux'])
        jump_T, jump_q, jump_d = (Tl - Tr).detach().numpy(), (qln - qrn).detach().numpy(), (dln - drn).detach().numpy()
        interfaces.append(dict(name=face.name, T_jump_rms_K=float(np.sqrt(np.mean(jump_T ** 2))), T_jump_max_K=float(np.abs(jump_T).max()),
            flux_jump_rms_pct=float(100 * np.sqrt(np.mean(jump_q ** 2)) / scale), flux_jump_max_pct=float(100 * np.abs(jump_q).max() / scale),
            derived_flux_jump_rms_pct=float(100 * np.sqrt(np.mean(jump_d ** 2)) / scale), flux_scale_W_m2=float(scale)))
        derived_balance[left.id] += face.weight * float(dln.sum().detach())
        derived_balance[right.id] -= face.weight * float(drn.sum().detach())
        law_balance[left.id] += face.weight * float(qln.sum().detach())
        law_balance[right.id] -= face.weight * float(qrn.sum().detach())
    boundary = []
    for piece in problem.pieces(n_boundary):
        xy, normal = _tensor(model, piece.xy), _tensor(model, piece.normal)
        region = problem.regions[piece.region]
        T, q, derived, _ = flux_fields(model, region, xy)
        qn, dn = (q * normal).sum(1, keepdim=True), (derived * normal).sum(1, keepdim=True)
        bc = piece.bc
        scale = problem.scales(region)['flux']
        if bc['type'] == 'dirichlet':
            error, flux_out = (T - bc['value'](xy)).detach().numpy() , dn
        elif bc['type'] == 'neumann':
            error, flux_out = ((dn - bc['value'](xy)) / scale).detach().numpy(), dn
        else:
            robin = bc['h'] * (T - bc['T_inf'](xy))
            error, flux_out = ((dn - robin) / scale).detach().numpy(), robin
        boundary.append(dict(side=piece.side, region=names[region.id], type=bc['type'], rms=float(np.sqrt(np.mean(error ** 2))),
            unit='K' if bc['type'] == 'dirichlet' else 'flux_scale'))
        derived_balance[region.id] += piece.weight * float(dn.sum().detach())
        law_balance[region.id] += piece.weight * float(flux_out.sum().detach())
    scale = problem.power_scale
    balance = dict(region_derived_pct={names[i]: float(100 * abs(v) / scale) for i, v in derived_balance.items()},
        region_constitutive_pct={names[i]: float(100 * abs(v) / scale) for i, v in law_balance.items()},
        global_derived_pct=float(100 * abs(sum(derived_balance.values())) / scale),
        global_constitutive_pct=float(100 * abs(sum(law_balance.values())) / scale), power_scale_W_m=float(scale))
    return dict(pde_rms=pde, constitutive_rms=constitutive, interfaces=interfaces, boundary=boundary, balance=balance)


def evaluate_model(model, reference=None, reference_label='supplied'):
    """Error report on the problem's evaluation points plus the audit. reference: optional array on those points."""
    problem = model.problem
    xy, labels, weights = problem.evaluation()
    with torch.no_grad():
        reference_tensor = next(model.parameters())
        T = model.temperature(torch.as_tensor(xy, dtype=reference_tensor.dtype), torch.as_tensor(labels)).numpy().ravel()
    truth = problem.reference(xy, labels) if reference is None else np.asarray(reference, dtype=float).ravel()
    names = [r.name for r in problem.regions]
    report = dict(audit(model))
    arrays = dict(xy=xy, labels=labels, weights=weights, T=T)
    if truth is not None:
        report.update(error_report(T, truth, xy, labels, weights, names, problem.T_level, problem.conductors()))
        arrays['T_reference'] = truth
    else:
        hot = np.isin(labels, problem.conductors())
        report.update(Tmax_pred_C=float(T[hot].max()) if hot.any() else float(T.max()))
    report['acceptance'] = acceptance(report)
    report['reference_label'] = problem.reference_label if reference is None else reference_label
    return report, arrays
