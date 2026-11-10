import copy
import json
import numpy as np
import pytest
import torch

from pinn_v2.geometry.domain import coaxial_case
from pinn_v2.cases.verification import coaxial_robin, symmetric_pair
from pinn_v2.cases.definition import CaseDefinition
from pinn_v2.evaluation.metrics import error_report, acceptance, CRITERIA
from pinn_v2.solvers.pinn.networks import PINNModel, ARCHITECTURES, build_model
from pinn_v2.solvers.pinn.objective import Objective
from pinn_v2.solvers.pinn.problem import load_problem, MMSProblem, AnnulusProblem, CableProblem, V2_DIR
from pinn_v2.solvers.pinn import trainer

SMALL = dict(n_interior=48, n_layer=24, n_face=24, n_boundary=24)


@pytest.mark.parametrize('values, expected', [
    ((.009, .099, .499), True),
    ((.01, .099, .499), False),
    ((.009, .1, .499), False),
    ((.009, .099, .5), False),
    ((.009, .099, float('nan')), False),
    ((None, .099, .499), False),
])
def test_accuracy_stop_requires_all_three_strict_finite_thresholds(monkeypatch, values, expected):
    report = dict(error_Tmax_abs_K=values[0], nrmse_pct=values[1],
        balance=dict(global_derived_pct=values[2]))
    monkeypatch.setattr(trainer, 'evaluate_model', lambda *args, **kwargs: (report, {}))
    monitor = trainer.AccuracyStop({})
    assert monitor.check(dict(step=0, phase='adam'), _model('mms_interface')) == expected
    assert monitor.summary()['triggered'] == expected
    assert (monitor.snapshot is not None) == expected


def test_accuracy_stop_can_be_configured_to_start_after_base_lbfgs(monkeypatch):
    report = dict(error_Tmax_abs_K=.001,nrmse_pct=.01,balance=dict(global_derived_pct=.1))
    monkeypatch.setattr(trainer,'evaluate_model',lambda *args, **kwargs:(report,{}))
    monitor=trainer.AccuracyStop(dict(after_phase='lbfgs'))
    model=_model('annulus','M1')
    assert not monitor.check(dict(step=1000,phase='adam'),model,force=True)
    assert monitor.check(dict(step=1001,phase='lbfgs'),model,force=True)
    assert len(monitor.checks)==1 and monitor.checks[0]['step']==1001


def test_accuracy_stop_can_check_adam_and_lbfgs(monkeypatch):
    report = dict(error_Tmax_abs_K=.02, nrmse_pct=.2, balance=dict(global_derived_pct=.6))
    monkeypatch.setattr(trainer, 'evaluate_model', lambda *args, **kwargs: (report, {}))
    monitor = trainer.AccuracyStop(dict(after_phase='any'))
    model = _model('annulus', 'M1')

    assert not monitor.check(dict(step=100, phase='adam'), model)
    report.update(error_Tmax_abs_K=.001, nrmse_pct=.01, balance=dict(global_derived_pct=.1))
    assert monitor.check(dict(step=101, phase='lbfgs'), model, force=True)
    assert [check['phase'] for check in monitor.checks] == ['adam', 'lbfgs']


@pytest.mark.parametrize('trigger_after, expected_calls', [(None, 10), (5, 5), (7, 7)])
def test_accuracy_stop_extra_lbfgs_calls_keep_optimizer_history(monkeypatch, trigger_after, expected_calls):
    model = torch.nn.Linear(1, 1, bias=False).double()
    constructors, calls = [], []

    class Objective:
        def __call__(self):
            loss = model.weight.square().sum()
            return loss, dict(pde=loss), {}

        def update_attention(self):
            pass

    class Monitor:
        triggered = False

        def check(self, row, current_model, force=False):
            self.triggered = trigger_after is not None and len(calls) >= trigger_after

        def summary(self):
            return dict(triggered=self.triggered, checks=[])

    class LBFGS:
        def __init__(self, parameters, **kwargs):
            constructors.append(self)
            self.state = dict(calls=0)

        def step(self, closure):
            self.state['calls'] += 1
            calls.append(self.state['calls'])
            with torch.no_grad():
                model.weight.fill_(self.state['calls'])

        def state_dict(self):
            return dict(self.state)

    monkeypatch.setattr(torch.optim, 'LBFGS', LBFGS)
    history, best = trainer.fit(model, Objective(), adam_steps=0, lbfgs_steps=5,
        lr=.001, log_every=100, accuracy_stop=Monitor(), extra_lbfgs_steps=5)
    assert len(constructors) == 1
    assert calls == list(range(1, expected_calls + 1))
    assert len(history) == expected_calls
    assert best['training']['optimizer']['calls'] == expected_calls
    assert model.weight.item() == expected_calls


def test_accuracy_stop_native_run_records_extra_calls(tmp_path):
    result = trainer.run('mms_interface', 'M0_EXACT', tmp_path/'run', adam_steps=1,
        lbfgs_steps=1, width=4, depth=1, points=dict(n_interior=4, n_layer=4, n_face=4, n_boundary=4),
        device='cpu', full_audit=False, accuracy_stop=dict(error_Tmax_abs_K=1e-20,
            nrmse_pct=1e-20, balance_pct=1e-20), extra_lbfgs_steps=1)
    assert result['status'] == 'completed', result.get('error')
    assert result['accuracy_stop']['lbfgs_outer_calls'] == 2
    assert result['accuracy_stop']['extra_lbfgs_outer_calls'] == 1
    assert result['accuracy_stop']['termination_reason'] == 'budget_exhausted'
    assert result['reference_used_in_loss'] is False


def test_exact_annulus_extension_can_start_from_previous_weights(tmp_path):
    problem=load_problem('annulus')
    torch.manual_seed(5)
    initial=build_model(problem,'M0_EXACT',width=4,depth=1,input_normalization='unit_box_v1').double()
    result=trainer.run('annulus','M0_EXACT',tmp_path/'warm',adam_steps=1,lbfgs_steps=1,
        width=4,depth=1,points=dict(n_interior=8,n_layer=8,n_face=8,n_boundary=8),
        device='cpu',sampling='annulus_log_v1',initial_state=initial.state_dict(),
        warm_start_source='previous-annulus-result/model.pt',accuracy_stop=dict(
            error_Tmax_abs_K=1e-20,nrmse_pct=1e-20,balance_pct=1e-20,after_phase='lbfgs'),
        extra_lbfgs_steps=1,full_audit=False)
    assert result['status']=='completed',result.get('error')
    assert result['config']['initialization_policy']=='weights_only_warm_start'
    assert result['config']['warm_start_source']=='previous-annulus-result/model.pt'
    assert result['config']['adam_steps']==1 and result['config']['lbfgs_steps']==1
    assert result['accuracy_stop']['lbfgs_outer_calls']==2
    assert result['accuracy_stop']['extra_lbfgs_outer_calls']==1


def test_accuracy_stop_resume_preserves_embedded_reference_and_extra_budget(tmp_path):
    from pinn_v2.solvers.pinn.checkpoint import load_checkpoint
    problem = load_problem('mms_interface')
    points, labels, _ = problem.evaluation()
    reference = problem.reference(points, labels) + .0001
    first = trainer.run('mms_interface', 'M0_EXACT', tmp_path/'first', adam_steps=2,
        lbfgs_steps=1, width=4, depth=1, points=dict(n_interior=4, n_layer=4, n_face=4, n_boundary=4),
        log_every=1, device='cpu', full_audit=False, reference=reference, reference_label='test FEM',
        accuracy_stop=dict(error_Tmax_abs_K=1e-20, nrmse_pct=1e-20, balance_pct=1e-20),
        extra_lbfgs_steps=1, stop_callback=lambda row, model: True)
    assert first['status'] == 'completed'
    _, _, payload = load_checkpoint(tmp_path/'first'/'model.pt')
    np.testing.assert_array_equal(payload['training']['accuracy_reference'].numpy(), reference)
    resumed = trainer.resume_training(tmp_path/'first'/'model.pt', tmp_path/'resumed', full_audit=False)
    assert resumed['status'] == 'completed', resumed.get('error')
    assert resumed['accuracy_stop']['reference_label'] == 'test FEM'
    assert resumed['accuracy_stop']['lbfgs_outer_calls'] == 2
    assert resumed['accuracy_stop']['extra_lbfgs_outer_calls'] == 1


def _model(case, architecture='M1', seed=0):
    torch.manual_seed(seed)
    return PINNModel(load_problem(case), architecture, width=16, depth=2).double()


def test_imperfect_contact_is_not_solved_silently():
    base = coaxial_robin().to_dict()
    name = coaxial_robin().domain().interfaces[-1].name
    base['contact'] = [dict(interface=name, R=dict(value=2e-3, unit='m2 K/W'), status='supuesto', source='pinn_v2_verification')]
    base['data_status']['contact'] = dict(status='supuesto', source='pinn_v2_verification', note='test value, not a measurement')
    definition = CaseDefinition.from_dict(base)
    with pytest.raises(NotImplementedError):
        load_problem(definition)


def test_dc_temperature_mode_is_rejected():
    with pytest.raises(NotImplementedError):
        CableProblem(coaxial_case(), mode='dc_temperature')


@pytest.mark.parametrize('name', ['coaxial_full', 'mms_interface', 'mms_robin'])
def test_region_power_is_the_integral_of_q(name):
    problem = load_problem(name)
    assert sum(problem.region_power.values()) == pytest.approx(problem.power_total)
    if name == 'coaxial_full':
        assert problem.power_total == pytest.approx(coaxial_case()['power'], rel=1e-6)
        assert problem.region_power[0] == 0.
    assert problem.power_scale > 0


def test_analytic_annulus_builds_a_finite_pinn_objective():
    problem = load_problem('annulus')
    assert isinstance(problem, AnnulusProblem)
    assert problem.reference(np.array([[0.06, 0.], [20., 0.]])) == pytest.approx([49.5857, 20.], abs=1e-3)
    pieces = {piece.side: piece for piece in problem.pieces(32)}
    assert pieces['inner'].bc['type'] == 'neumann'
    assert pieces['inner'].bc['value'](torch.tensor([[0.06, 0.]], dtype=torch.float64)).item() == pytest.approx(-106.1033)
    model = PINNModel(problem, 'M1', width=8, depth=1).double()
    total, _, _ = Objective(model, **SMALL)()
    assert torch.isfinite(total)
    total.backward()


def test_annulus_log_lifting_reproduces_exact_radial_solution_with_exact_traces():
    problem = load_problem('annulus')
    model = build_model(problem, 'M1_EXACT', width=4, depth=1).double()
    with torch.no_grad():
        for parameter in model.networks.parameters():
            parameter.zero_()
        for offset in model.offsets:
            offset.zero_()
    ri, ro, power, conductivity, outer = (problem.case[key]
        for key in ('ri', 'ro', 'power', 'k', 'T0'))
    inner_temperature = outer + power/(2*np.pi*conductivity)*np.log(ro/ri)
    inner_flux = -power/(2*np.pi*ri)
    outer_flux = power/(2*np.pi*ro)

    def exact_boundary_trace(side, points, region):
        if side == 'inner':
            return (points[:, :1]*0+inner_temperature, points[:, :1]*0+inner_flux)
        return (points[:, :1]*0+outer, points[:, :1]*0+outer_flux)

    model._boundary_trace = exact_boundary_trace
    radii = torch.as_tensor(np.geomspace(ri, ro, 257), dtype=torch.float64)
    points = torch.stack((radii, torch.zeros_like(radii)), dim=1)
    prediction = model.field(problem.regions[0], points)[0].detach().numpy().ravel()
    expected = problem.reference(points.numpy())
    assert np.max(np.abs(prediction-expected)) < 1e-10


def test_annulus_log_sampler_covers_inner_radii_without_sampling_the_hole():
    problem = load_problem('annulus')
    region = problem.regions[0]
    points = problem.sample_interior(region, 512, 11, 'annulus_log_v1')
    radii = np.linalg.norm(points, axis=1)
    assert points.shape == (512, 2)
    assert radii.min() > problem.case['ri']
    assert radii.max() < problem.case['ro']
    assert np.sum(radii < 1.0) >= 80
    assert np.array_equal(points, problem.sample_interior(region, 512, 11, 'annulus_log_v1'))


def test_annulus_evaluation_audits_exact_boundaries_without_area_bias():
    from pinn_v2.evaluation.audit import evaluate_model
    problem = load_problem('annulus')
    points, _, weights = problem.evaluation()
    assert len(points) == 6000
    assert weights.sum() == pytest.approx(1.)
    model = build_model(problem, 'M1_EXACT', width=4, depth=1).double()
    report, arrays = evaluate_model(model, include_audit=False)
    assert set(report['annulus_boundary']) == {'inner', 'outer'}
    assert report['annulus_boundary_points'] == 512
    assert report['Tmax_reference_C'] == pytest.approx(
        problem.case['T0']+problem.case['power']/(2*np.pi*problem.case['k'])*
        np.log(problem.case['ro']/problem.case['ri']))
    assert len(arrays['xy']) == 6000
    assert len(arrays['annulus_inner_xy']) == 256
    assert len(arrays['annulus_outer_xy']) == 256


def test_annulus_log_sampler_is_rejected_for_other_geometry(tmp_path):
    with pytest.raises(ValueError, match='only by the concentric annulus'):
        trainer.run('mms_interface', 'M0_EXACT', tmp_path/'unsupported', adam_steps=1,
            sampling='annulus_log_v1')


def test_interface_probe_gives_one_sided_conductivity():
    problem = load_problem('mms_interface')
    xy = torch.tensor([[.5, .3], [.5, .7]], dtype=torch.float64)
    low, high = problem.regions
    assert problem.conductivity(low, xy).ravel().tolist() == pytest.approx([.5, .5])
    assert problem.conductivity(high, xy).ravel().tolist() == pytest.approx([2., 2.])


def test_boundary_pieces_cover_every_side_once():
    problem = load_problem('mms_interface')
    length = {}
    for piece in problem.pieces(40):
        length[piece.side] = length.get(piece.side, 0.) + piece.weight * len(piece.xy)
    assert length == pytest.approx(dict(bottom=1., top=1., left=1., right=1.))


def test_robin_side_is_robin_not_dirichlet():
    types = {(p.side, p.bc['type']) for p in load_problem('mms_robin').pieces(8)}
    assert ('top', 'robin') in types and ('top', 'dirichlet') not in types


def test_cable_dirichlet_profiles_build_as_dirichlet_pinn_pieces():
    problem = load_problem('kim_2024_thermal_original')
    pieces = [piece for piece in problem.pieces(8) if piece.side in ('left', 'right')]

    assert pieces
    assert all(piece.bc['type'] == 'dirichlet' for piece in pieces)
    for piece in pieces:
        values = piece.bc['value'](torch.tensor(piece.xy, dtype=torch.float64))
        assert torch.isfinite(values).all()

    model = PINNModel(problem, 'M1', width=8, depth=1).double()
    total, _, _ = Objective(model, **SMALL)()
    assert torch.isfinite(total)
    total.backward()
    types = {p.bc['type'] for p in load_problem(coaxial_robin()).pieces(8)}
    assert types == {'robin'}


@pytest.mark.parametrize('name', ['mms_constant', 'mms_high_contrast', 'mms_interface', 'mms_layered_y', 'mms_robin', 'mms_smooth_2d', 'mms_variable', 'coaxial_full'])
@pytest.mark.parametrize('architecture', sorted(ARCHITECTURES))
def test_objective_is_finite_and_differentiable(name, architecture):
    model = _model(name, architecture)
    total, parts, region_pde = Objective(model, **SMALL)()
    assert torch.isfinite(total)
    total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert set(region_pde) == {r.name for r in model.problem.regions}
    assert (parts['constitutive'] > 0) == (architecture == 'M2')


@pytest.mark.parametrize('sampling', ['interface', 'paired'])
def test_objective_interface_resampling_is_deterministic_and_preserves_counts(sampling):
    model = _model('aras_flat', 'M2')
    first = Objective(model, seed=23, sampling=sampling, **SMALL)
    second = Objective(model, seed=23, sampling=sampling, **SMALL)
    first.resample(seed=701)
    second.resample(seed=701)

    assert first.resampling_count == 1
    assert {key: len(points) for key, points in first.interior.items()} == {key: len(points) for key, points in second.interior.items()}
    assert all(torch.equal(first.interior[key], second.interior[key]) for key in first.interior)
    assert all(first.sources[key].shape[0] == len(first.interior[key]) for key in first.interior)


@pytest.mark.parametrize('policy', ['mixed', 'interface'])
@pytest.mark.parametrize('cables', [[[-.34, -1.2], [0., -1.2], [.34, -1.2]],
    [[-.17, -1.35], [.17, -1.35], [0., -1.05]]])
def test_near_sampling_covers_every_cable_from_json(policy, cables):
    case = dict(load_problem('aras_flat').case, id='arbitrary_layout', cables=cables)
    problem = load_problem(case)
    points = problem.sample_interior(problem.regions[0], 512, 11, policy)
    distances = np.linalg.norm(points[:, None, :] - np.asarray(cables)[None, :, :], axis=2)
    near = np.min(distances, axis=1) < case['radius'] + .06
    owners = np.argmin(distances[near], axis=1)
    counts = np.bincount(owners, minlength=len(cables))
    assert counts.min() >= 8
    assert len(points) == 512
    np.testing.assert_array_equal(points, problem.sample_interior(problem.regions[0], 512, 11, policy))


def test_model_shapes_and_parameter_partition():
    for architecture, networks in (('M0', 1), ('M1', 2), ('M2', 2)):
        model = _model('mms_interface', architecture)
        assert len(model.networks) == networks
        xy = torch.rand(5, 2, dtype=torch.float64)
        T, q = model.field(model.problem.regions[0], xy)
        assert T.shape == (5, 1) and (q is None) == (architecture != 'M2')


def test_exact_interface_mode_enforces_transmission_and_optimizes_only_pde():
    problem = load_problem('aras_flat')
    model = build_model(problem, 'M1_EXACT', width=8, depth=1).double()
    with torch.no_grad():
        model.trace_coefficients[:, 0, 0] = .02
        model.trace_coefficients[:, 1, 0] = .1
        model.trace_coefficients[:, 0, 1] = .01
        model.trace_coefficients[:, 1, 2] = .05

    for face in problem.faces(12):
        xy = torch.tensor(face.xy, dtype=torch.float64, requires_grad=True)
        normal = torch.tensor(face.normal, dtype=torch.float64)
        fluxes = []
        temperatures = []
        for region_id in (face.left, face.right):
            region = problem.regions[region_id]
            temperature, _ = model.field(region, xy)
            gradient = torch.autograd.grad(temperature.sum(), xy, create_graph=True)[0][:, :2]
            conductivity = problem.conductivity(region, xy)
            fluxes.append(-(conductivity * gradient * normal).sum(dim=1))
            temperatures.append(temperature)
        torch.testing.assert_close(temperatures[0], temperatures[1], rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(fluxes[0], fluxes[1], rtol=1e-8, atol=1e-8)
        assert torch.max(torch.abs(fluxes[0])) > 1e-4

    for xy, _, _ in problem.domain.outer_boundary(12):
        points = torch.tensor(xy, dtype=torch.float64)
        boundary_temperature, _ = model.field(problem.regions[0], points)
        torch.testing.assert_close(boundary_temperature, torch.full_like(boundary_temperature, 20.))

    total, parts, _ = Objective(model, **SMALL)()
    assert torch.isfinite(total)
    torch.testing.assert_close(total, parts['pde'])
    assert all(float(value.detach()) == 0. for name, value in parts.items() if name != 'pde')
    total.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize('kwargs', [dict(trace_parameterization='conditioned_v1'),
    dict(trace_parameterization='conservative_v1'), dict(soil_envelope_gain=16.)])
def test_exact_interface_mode_rejects_removed_variants(kwargs):
    with pytest.raises(ValueError, match='one construction'):
        build_model(load_problem('aras_flat'), 'M1_EXACT', width=4, depth=1, **kwargs)


def test_exact_interface_mode_rejects_unsupported_problems():
    with pytest.raises(ValueError, match='rectangular domain'):
        build_model(load_problem(coaxial_robin()), 'M1_EXACT', width=8, depth=1)


@pytest.mark.parametrize('case_file', ['kim_v5_summer.json', 'references/cables/xlpe_backfill.json'])
def test_exact_interface_mode_builds_heterogeneous_cable_cases(case_file):
    problem = load_problem(V2_DIR / case_file)
    model = build_model(problem, 'M1_EXACT', width=4, depth=1)

    assert model.exact_constraints
    assert model.architecture == 'M1_EXACT'


@pytest.mark.parametrize('name', ['kim_v5_summer', 'xlpe_backfill'])
def test_exact_cable_mode_enforces_general_boundaries_and_material_interfaces(name):
    problem = load_problem(name)
    model = build_model(problem, 'M1_EXACT', width=4, depth=1).double()

    for piece in problem.pieces(8):
        region = problem.regions[piece.region]
        xy = torch.tensor(piece.xy, dtype=torch.float64, requires_grad=True)
        normal = torch.tensor(piece.normal, dtype=torch.float64)
        temperature, _ = model.field(region, xy)
        bc = piece.bc
        if bc['type'] == 'dirichlet':
            residual = temperature - bc['value'](xy)
        else:
            gradient = torch.autograd.grad(temperature.sum(), xy, create_graph=True)[0][:, :2]
            normal_flux = -(problem.conductivity(region, xy) * gradient * normal).sum(1, keepdim=True)
            residual = (normal_flux - bc['value'](xy) if bc['type'] == 'neumann'
                else normal_flux - bc['h'] * (temperature - bc['T_inf'](xy)))
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=1e-7, atol=1e-7)

    for face in problem.faces(8):
        xy = torch.tensor(face.xy, dtype=torch.float64, requires_grad=True)
        normal = torch.tensor(face.normal, dtype=torch.float64)
        temperatures, fluxes = [], []
        for region_id in (face.left, face.right):
            region = problem.regions[region_id]
            temperature, _ = model.field(region, xy)
            gradient = torch.autograd.grad(temperature.sum(), xy, create_graph=True)[0][:, :2]
            flux = -(problem.conductivity(region, xy) * gradient * normal).sum(1, keepdim=True)
            temperatures.append(temperature)
            fluxes.append(flux)
        torch.testing.assert_close(temperatures[0], temperatures[1], rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(fluxes[0], fluxes[1], rtol=1e-7, atol=1e-7)

    total, parts, _ = Objective(model, n_interior=8, n_layer=8, n_face=8, n_boundary=8)()
    torch.testing.assert_close(total, parts['pde'])
    assert all(float(value.detach()) == 0. for term, value in parts.items() if term != 'pde')


@pytest.mark.parametrize(('case_id', 'cables'), [
    ('exact_flat_three', [[-.06, 0.], [0., 0.], [.06, 0.]]),
    ('exact_trefoil_three', [[-.06, -.035], [.06, -.035], [0., .07]]),
    ('exact_arbitrary_five', [[-.075, -.04], [0., -.04], [.075, -.04], [-.0375, .04], [.0375, .04]]),
])
def test_exact_interface_mode_supports_flat_trefoil_and_arbitrary_cable_counts(case_id, cables):
    case = dict(coaxial_case(), id=case_id, cables=cables)
    case.pop('outer_radius', None)
    problem = load_problem(case)
    model = build_model(problem, 'M1_EXACT', width=4, depth=1)

    assert len(model.outer_interfaces) == len(cables)
    assert len(model.interfaces) == len(cables) * len(case['layers'])
    assert len(model.networks) == 1 + len(cables) * len(case['layers'])
    assert model.collar_width > 0


def test_exact_cable_soil_features_are_case_normalized():
    problem = load_problem('aras_flat')
    model = build_model(problem, 'M1_EXACT', width=8, depth=1).double()
    soil = problem.regions[0]
    points = torch.as_tensor(problem.sample_interior(soil, 512, 11, 'mixed'), dtype=torch.float64)
    features = model._coordinates(soil, points)

    assert features.shape[1] == 2 + len(problem.case['cables'])
    assert torch.all(features >= -1.) and torch.all(features <= 1.)
    torch.testing.assert_close(features[:, :2].min(), torch.tensor(-1., dtype=torch.float64), atol=.02, rtol=0)


@pytest.mark.parametrize('name', ['aras_flat', 'mms_interface', 'annulus'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_unit_box_features_and_exact_pde_are_finite(name, dtype):
    problem = load_problem(name)
    model = build_model(problem, 'M1_EXACT', width=4, depth=1, input_normalization='unit_box_v1').to(dtype=dtype)
    for region in problem.regions:
        points = torch.as_tensor(problem.sample_interior(region, 24, 11), dtype=dtype).requires_grad_(True)
        features = model.input_coordinates(region, points)
        assert torch.all(features >= -1e-6) and torch.all(features <= 1 + 1e-6)
        assert torch.isfinite(torch.autograd.grad(features.sum(), points, create_graph=True)[0]).all()
    objective = Objective(model, attention='rba', **SMALL)
    total, _, _ = objective()
    total.backward()
    objective.update_attention()
    assert total.dtype == dtype and torch.isfinite(total)
    assert all(state.dtype == dtype for state in objective.attention.state.values())
@pytest.mark.parametrize('name', [
    'mms_constant', 'mms_variable', 'mms_smooth_2d', 'mms_high_contrast', 'mms_interface', 'mms_robin', 'annulus'
])
def test_exact_mode_satisfies_supported_mms_and_cigre_boundary_conditions(name):
    problem = load_problem(name)
    model = build_model(problem, 'M1_EXACT', width=6, depth=1).double()
    with torch.no_grad():
        model.trace_coefficients[:, 0, 0] = .01
        model.trace_coefficients[:, 1, 0] = .1

    for piece in problem.pieces(12):
        region = problem.regions[piece.region]
        xy = torch.tensor(piece.xy, dtype=torch.float64, requires_grad=True)
        normal = torch.tensor(piece.normal, dtype=torch.float64)
        temperature, _ = model.field(region, xy)
        bc = piece.bc
        if bc['type'] == 'dirichlet':
            residual = temperature - bc['value'](xy)
        else:
            gradient = torch.autograd.grad(temperature.sum(), xy, create_graph=True)[0][:, :2]
            normal_flux = -(problem.conductivity(region, xy) * gradient * normal).sum(1, keepdim=True)
            residual = (normal_flux - bc['value'](xy) if bc['type'] == 'neumann'
                else normal_flux - bc['h'] * (temperature - bc['T_inf'](xy)))
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=1e-8, atol=1e-8)

    total, parts, _ = Objective(model, **SMALL)()
    torch.testing.assert_close(total, parts['pde'])
    assert all(float(value.detach()) == 0. for term, value in parts.items() if term != 'pde')
    total.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_exact_mms_interface_has_shared_nonzero_flux_trace():
    problem = load_problem('mms_interface')
    model = build_model(problem, 'M1_EXACT', width=6, depth=1).double()
    with torch.no_grad():
        model.trace_coefficients[:, 1, 0] = .2
    for face in problem.faces(12):
        xy = torch.tensor(face.xy, dtype=torch.float64, requires_grad=True)
        normal = torch.tensor(face.normal, dtype=torch.float64)
        temperatures, fluxes, normal_gradients = [], [], []
        for region_id in (face.left, face.right):
            region = problem.regions[region_id]
            temperature, _ = model.field(region, xy)
            gradient = torch.autograd.grad(temperature.sum(), xy, create_graph=True)[0][:, :2]
            flux = -(problem.conductivity(region, xy) * gradient * normal).sum(1, keepdim=True)
            temperatures.append(temperature)
            fluxes.append(flux)
            normal_gradients.append((gradient * normal).sum(1, keepdim=True))
        torch.testing.assert_close(temperatures[0], temperatures[1], rtol=1e-8, atol=1e-8)
        torch.testing.assert_close(fluxes[0], fluxes[1], rtol=1e-8, atol=1e-8)
        assert torch.max(torch.abs(fluxes[0])) > 1e-6
        assert torch.max(torch.abs(normal_gradients[0] - normal_gradients[1])) > 1e-4


def test_manufactured_interface_fixture_has_consistent_source_and_nonzero_flux():
    from pinn_v2.cases.catalog import exact
    from pinn_v2.physics.residuals import heat_residual
    problem = load_problem('mms_interface')
    conductivities = []
    for region in problem.regions:
        points = torch.tensor(problem.sample_interior(region, 32, 91),
            dtype=torch.float64, requires_grad=True)
        temperature = exact(problem.case, points[:, :1], points[:, 1:2], torch)
        conductivity = problem.conductivity(region, points)
        source = problem.source(region, points)
        residual = heat_residual(temperature, points, conductivity, source)
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-10, rtol=0)
        assert source.abs().max() > 1.
        conductivities.append(float(conductivity[0, 0].detach()))
    assert conductivities == [.5, 2.]
    fluxes, temperatures = [], []
    for region, position in zip(problem.regions, [.5 - 1e-7, .5 + 1e-7]):
        points = torch.tensor([[position, .5]], dtype=torch.float64, requires_grad=True)
        temperature = exact(problem.case, points[:, :1], points[:, 1:2], torch)
        gradient = torch.autograd.grad(temperature.sum(), points)[0]
        fluxes.append(-problem.conductivity(region, points) * gradient[:, :1])
        temperatures.append(temperature)
    torch.testing.assert_close(fluxes[0], fluxes[1], atol=1e-10, rtol=0)
    assert abs(float(fluxes[0].detach())) > 1.
    torch.testing.assert_close(temperatures[0], temperatures[1], atol=3e-6, rtol=0)


def test_aras_single_fixture_has_one_cable_and_its_own_source_power():
    problem = load_problem('aras_single')
    assert len(problem.case['cables']) == 1
    assert len(problem.regions) == 1 + len(problem.case['layers'])
    assert problem.power_total == pytest.approx(problem.case['power'], rel=1e-10)
    assert problem.power_total == pytest.approx(41.4592999)
    model = build_model(problem, 'M1_EXACT', width=4, depth=1,
        input_normalization='unit_box_v1').double()
    total, parts, _ = Objective(model, **SMALL)()
    assert torch.isfinite(total)
    torch.testing.assert_close(total, parts['pde'])
    assert all(float(value.detach()) == 0. for name, value in parts.items() if name != 'pde')


@pytest.mark.parametrize('name', ['mms_interface', 'mms_robin', 'aras_single', 'aras_flat', 'annulus'])
def test_paired_pde_rows_keep_bulk_counts_material_and_determinism(name):
    problem = load_problem(name)
    for region in problem.regions:
        points = problem.sample_interior(region, 64, 23, 'paired')
        bulk = problem.sample_interior(region, 32, 23, 'uniform')
        assert points.shape == (64, 2)
        assert problem.contains_points(region, points).all()
        np.testing.assert_array_equal(points[:32], bulk)
        np.testing.assert_array_equal(points, problem.sample_interior(region, 64, 23, 'paired'))


def test_objective_reads_no_reference(monkeypatch):
    problem = load_problem('mms_constant')

    def forbidden(*a, **k):
        raise AssertionError('reference accessed by the objective')
    monkeypatch.setattr(MMSProblem, 'reference', forbidden)
    Objective(PINNModel(problem, 'M1', 8, 1).double(), **SMALL)()


def test_zero_weights_remove_terms():
    model = _model('mms_robin')
    total, parts, _ = Objective(model, weights=dict(pde=0, dirichlet=0, robin=0, energy=0, interface_temperature=0, interface_flux=0, neumann=0, constitutive=0), **SMALL)()
    assert float(total.detach()) == 0.
    with pytest.raises(ValueError):
        Objective(model, weights=dict(unknown=1.), **SMALL)


@pytest.mark.parametrize('expression', ['20 + 0*x', '20 + 2*x'])
def test_source_free_manufactured_solution_has_finite_audit(expression):
    from pinn_v2.evaluation.audit import evaluate_model
    case = dict(load_problem('mms_constant').case, id='renamed_source_free', kind='mms',
        conductivity=dict(type='expression', expression='1 + 0*x'),
        exact_expression=expression, source_expression='0*x')
    problem = load_problem(case)
    model = build_model(problem, 'M1_EXACT', width=4, depth=1, input_normalization='unit_box_v1').double()
    report, _ = evaluate_model(model)
    assert problem.power_scale > 0
    assert np.isfinite(report['nrmse_pct'])
    assert np.isfinite(report['balance']['global_derived_pct'])
    assert report['balance']['normalization'] == 'characteristic_scale'


def test_source_free_cable_scales_and_exact_objective_are_positive():
    case = dict(coaxial_case(), id='zero_current_cable', current=0., power=0.)
    case.pop('outer_radius', None)
    problem = load_problem(case)
    assert problem.power_total == 0 and problem.power_scale > 0
    model = build_model(problem, 'M1_EXACT', width=4, depth=1).double()
    for region in problem.regions:
        assert all(value > 0 for value in problem.scales(region).values())
    total, _, _ = Objective(model, **SMALL)()
    assert torch.isfinite(total)


def test_sampling_seed_changes_points_and_init_seed_changes_weights():
    a, b, c = _model('mms_constant', seed=1), _model('mms_constant', seed=2), _model('mms_constant', seed=1)
    pa = torch.cat([p.flatten() for p in a.parameters()])
    assert not torch.equal(pa, torch.cat([p.flatten() for p in b.parameters()]))
    assert torch.equal(pa, torch.cat([p.flatten() for p in c.parameters()]))
    o1, o2, o3 = (Objective(a, seed=s, **SMALL) for s in (5, 6, 5))
    assert not torch.equal(o1.interior[0], o2.interior[0])
    assert torch.equal(o1.interior[0], o3.interior[0])


def test_metrics_on_synthetic_fields():
    xy = np.random.default_rng(0).random((200, 2))
    labels = np.zeros(200, dtype=int)
    labels[100:] = 1
    weights = np.full(200, .005)
    truth = 20 + 10 * xy[:, 0]
    report = error_report(truth + .5, truth, xy, labels, weights, ['a', 'b'], 20., [])
    assert report['rmse_K'] == pytest.approx(.5)
    assert report['nrmse_pct'] == pytest.approx(100 * .5 / report['rise_ref_K'])
    assert report['error_Tmax_abs_K'] == pytest.approx(.5)
    assert report['hotspot_distance_m'] == 0.
    assert set(report['region_rmse_K']) == {'a', 'b'}
    assert report['error_percentiles_K']['99'] == pytest.approx(.5)


def test_acceptance_rejects_and_records_gates():
    report = dict(nrmse_pct=1., error_rise_pct=1., region_rmse_rise_pct=dict(a=1.),
        balance=dict(global_derived_pct=1., region_derived_pct=dict(a=1.)), interfaces=[dict(T_jump_max_K=.05, flux_jump_rms_pct=.5)])
    assert acceptance(report)['state'] == 'accepted'
    report['interfaces'][0]['T_jump_max_K'] = CRITERIA['interface_T_jump_max_K'] * 2
    verdict = acceptance(report)
    assert verdict['state'] == 'rejected' and verdict['gates']['interface_T_jump_max_K'] is False


def test_acceptance_allows_sub_one_percentage_point_regional_excess():
    report = dict(nrmse_pct=1., error_rise_pct=1., region_rmse_rise_pct=dict(a=5.32),
        balance=dict(global_derived_pct=1., region_derived_pct=dict(a=1.)),
        interfaces=[dict(T_jump_max_K=.05, flux_jump_rms_pct=.5)])

    verdict = acceptance(report)

    assert verdict['state'] == 'accepted_with_tolerance'
    assert verdict['accepted'] is True
    assert verdict['raw_gates']['region_rmse_rise_pct'] is False
    assert verdict['gates']['region_rmse_rise_pct'] is True
    assert verdict['criteria']['region_rmse_rise_pct'] == 5.
    assert verdict['tolerance']['region_rmse_rise_pct'] == 1.


def test_acceptance_rejects_regional_excess_at_or_above_one_percentage_point():
    report = dict(nrmse_pct=1., error_rise_pct=1., region_rmse_rise_pct=dict(a=6.),
        balance=dict(global_derived_pct=1., region_derived_pct=dict(a=1.)),
        interfaces=[dict(T_jump_max_K=.05, flux_jump_rms_pct=.5)])

    assert acceptance(report)['state'] == 'rejected'


def test_trainer_writes_record_and_never_overwrites(tmp_path):
    out = tmp_path / 'run'
    result = trainer.run('mms_constant', 'M1', out, adam_steps=30, width=8, depth=2, points=SMALL, log_every=10)
    assert result['status'] == 'completed' and result['fem_labels_used'] == 0 and result['reference_used_in_loss'] is False
    for name in ('result.json', 'manifest.json', 'evaluation.npz', 'collocation.npz', 'model.pt',
                 'latest.pt', 'best_train_loss.pt', 'best_validation.pt', 'checkpoint_meta.json'):
        assert out.joinpath(name).exists()
    manifest = json.loads(out.joinpath('manifest.json').read_text())
    assert manifest['seeds'] == dict(init=0, sampling=11, validation=1_000_014)
    assert manifest['extra']['device'] == result['device']
    checkpoint_meta = json.loads(out.joinpath('checkpoint_meta.json').read_text())
    assert checkpoint_meta['checkpoint_file'] == 'model.pt'
    assert checkpoint_meta['status'] == 'completed'
    assert checkpoint_meta['progress']['phase'] == 'finished'
    assert checkpoint_meta['progress']['step_zero_based'] is True
    assert checkpoint_meta['progress']['step'] == result['history'][-1]['step']
    assert checkpoint_meta['progress']['optimizer_steps_completed'] == result['history'][-1]['step'] + 1
    assert checkpoint_meta['progress']['best_step'] == result['best_step']
    assert checkpoint_meta['physics_sha256'] == result['physics_sha256']
    assert checkpoint_meta['progress']['checkpoint_files'] == dict(latest='latest.pt',
        best_train_loss='best_train_loss.pt', best_validation='best_validation.pt')
    assert result['checkpoint_selection']['best_validation']['loss'] < float('inf')
    state = torch.load(out.joinpath('model.pt'), map_location='cpu', weights_only=True)['model']
    assert all(tensor.device.type == 'cpu' for tensor in state.values())
    with np.load(out.joinpath('collocation.npz')) as collocation:
        assert collocation.files
    with pytest.raises(FileExistsError):
        trainer.run('mms_constant', 'M1', out, adam_steps=1, points=SMALL)


def test_trainer_fails_and_preserves_nonfinite_evaluation(tmp_path, monkeypatch):
    report = dict(nrmse_pct=float('nan'), acceptance=dict(state='rejected'))
    arrays = dict(T=np.array([np.nan]), xy=np.array([[0., 0.]]))
    monkeypatch.setattr(trainer, 'evaluate_model', lambda *args, **kwargs: (report, arrays))

    out = tmp_path / 'nonfinite_evaluation'
    result = trainer.run('mms_constant', 'M1', out, adam_steps=1, width=8, depth=1,
        points=SMALL, log_every=1)

    assert result['status'] == 'failed'
    assert result['evaluation_finite'] is False
    assert result['error']['type'] == 'FloatingPointError'
    assert out.joinpath('failure.json').exists()
    assert out.joinpath('evaluation.npz').exists()
    assert out.joinpath('model.pt').exists()
    checkpoint_meta = json.loads(out.joinpath('checkpoint_meta.json').read_text())
    assert checkpoint_meta['status'] == 'failed'
    assert checkpoint_meta['progress']['phase'] in ('initialized', 'adam')


def test_trainer_rejects_invalid_validation_cadence_before_creating_output(tmp_path):
    output = tmp_path / 'invalid_validation'

    with pytest.raises(ValueError, match='validation_every'):
        trainer.run('mms_constant', 'M1', output, validation_every=0)

    assert not output.exists()


def test_trainer_is_deterministic(tmp_path):
    kwargs = dict(adam_steps=20, width=8, depth=2, points=SMALL, log_every=10)
    a = trainer.run('mms_interface', 'M1', tmp_path / 'a', **kwargs)
    b = trainer.run('mms_interface', 'M1', tmp_path / 'b', **kwargs)
    assert a['final_loss'] == b['final_loss']
    c = trainer.run('mms_interface', 'M1', tmp_path / 'c', init_seed=3, **kwargs)
    assert c['final_loss'] != a['final_loss']


def test_adam_best_checkpoint_matches_the_loss_evaluation():
    parameter = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
    model = torch.nn.Module()
    model.register_parameter('parameter', parameter)

    class QuadraticObjective:
        def __call__(self):
            loss = model.parameter.square()
            return loss, {'quadratic': loss}, {}

        def update_attention(self):
            pass

    _, best = trainer.fit(model, QuadraticObjective(), adam_steps=1, lbfgs_steps=0, lr=.1, log_every=1)
    assert best['total'] == pytest.approx(.81)
    assert best['state']['parameter'].item() == pytest.approx(.9)
    assert model.parameter.item() == pytest.approx(.9)


def test_adam_stop_callback_ends_at_requested_checkpoint():
    parameter = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
    model = torch.nn.Module()
    model.register_parameter('parameter', parameter)

    class QuadraticObjective:
        def __call__(self):
            loss = model.parameter.square()
            return loss, {'quadratic': loss}, {}

        def update_attention(self):
            pass

    checkpoints = []
    history, _ = trainer.fit(model, QuadraticObjective(), adam_steps=20, lbfgs_steps=0,
        lr=.1, log_every=2, stop_callback=lambda row, _: checkpoints.append(row['step']) or row['step'] >= 4)

    assert checkpoints == [0, 2, 4]
    assert len(history) == 3


def test_adam_resamples_objective_at_requested_cadence():
    parameter = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
    model = torch.nn.Module()
    model.register_parameter('parameter', parameter)

    class QuadraticObjective:
        def __init__(self):
            self.resampled_at = []

        def __call__(self):
            loss = model.parameter.square()
            return loss, {'quadratic': loss}, {}

        def update_attention(self):
            pass

        def resample(self):
            self.resampled_at.append(len(self.resampled_at))

    objective = QuadraticObjective()
    trainer.fit(model, objective, adam_steps=5, lbfgs_steps=0, lr=.1, log_every=5, resample_every=2)

    assert objective.resampled_at == [0, 1]


def test_adam_best_loss_resets_when_collocation_cloud_changes():
    parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    model = torch.nn.Module()
    model.register_parameter('parameter', parameter)

    class ChangingObjective:
        target = 0.

        def __call__(self):
            loss = (model.parameter - self.target).square()
            return loss, {'quadratic': loss}, {}

        def update_attention(self):
            pass

        def resample(self):
            self.target = 1.

    _, best = trainer.fit(model, ChangingObjective(), adam_steps=4, lbfgs_steps=0,
        lr=.1, log_every=4, resample_every=2)

    assert best['step'] == 3
    assert best['state']['parameter'].item() > 0.


def test_failure_is_preserved(tmp_path):
    result = trainer.run('mms_constant', 'M1', tmp_path / 'bad', adam_steps=5, lr=float('nan'), width=8, depth=1, points=SMALL)
    assert result['status'] == 'failed'
    assert tmp_path.joinpath('bad', 'failure.json').exists() and tmp_path.joinpath('bad', 'manifest.json').exists()


@pytest.mark.slow
def test_m1_learns_manufactured_case(tmp_path):
    result = trainer.run('mms_constant', 'M1', tmp_path / 'learn', adam_steps=600, lr=3e-3, log_every=200)
    assert result['evaluation']['nrmse_pct'] < 5.
    assert result['evaluation']['acceptance']['state'] == 'accepted'
