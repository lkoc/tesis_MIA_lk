"""RBF-FD field copied into the PINN output layer by least squares, with no gradient steps and no WSL call."""
import time

import numpy as np

from pinn_v3.solvers.pinn.fem_initializer import collocation_result, local_quadratic_fit, readout_transfer
from pinn_v3.solvers.pinn.rbf_fd import solve_regions


def rbf_readout_initialization(model, problem, objective, settings, seed):
    """Solve the RBF-FD field, fit values and gradients at the PINN collocation points, and copy them into the model."""
    collocation = {region.id: objective.interior[region.id].detach().cpu().numpy() for region in problem.regions}
    rng = np.random.default_rng(seed)
    anchors = {region_id: points[np.sort(rng.choice(len(points), max(1, int(np.ceil(len(points) * settings['rbf_anchor_fraction']))), replace=False))]
               for region_id, points in collocation.items()}
    started = time.perf_counter()
    field = solve_regions(problem.definition, interior_per_region=settings['rbf_interior_per_region'],
                          surface_count=settings['surface_count'], stencil_size=settings['stencil_size'], seed=seed)
    solve_s = time.perf_counter() - started
    fitted = local_quadratic_fit(problem.definition, field['xy'], field['temperature'], field['labels'], anchors)
    result = collocation_result(anchors, {region_id: values for region_id, (values, _) in fitted.items()})
    gradients = {region_id: gradient for region_id, (_, gradient) in fitted.items()}
    report = readout_transfer(model, result, gradients=gradients if settings['rbf_transfer'] == 'readout_gls' else None)
    report.update(source='rbf_fd', transfer=settings['rbf_transfer'], rbf_nodes=int(len(field['xy'])),
                  rbf_solve_s=solve_s, rbf_scaled_residual=float(field['scaled_residual_inf']),
                  anchor_fraction=settings['rbf_anchor_fraction'], anchor_count=int(len(result['xy'])))
    return report
