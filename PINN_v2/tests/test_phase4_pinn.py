import copy
import json
import numpy as np
import pytest
import torch

from pinn_v2.geometry.domain import coaxial_case
from pinn_v2.cases.verification import coaxial_robin, symmetric_pair
from pinn_v2.cases.definition import CaseDefinition
from pinn_v2.evaluation.metrics import error_report, acceptance, CRITERIA
from pinn_v2.solvers.pinn.networks import PINNModel, ARCHITECTURES
from pinn_v2.solvers.pinn.objective import Objective
from pinn_v2.solvers.pinn.problem import load_problem, MMSProblem, AnnulusProblem, CableProblem
from pinn_v2.solvers.pinn import trainer

SMALL = dict(n_interior=48, n_layer=24, n_face=24, n_boundary=24)


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


def test_objective_interface_resampling_is_deterministic_and_preserves_counts():
    model = _model('aras_flat', 'M2')
    first = Objective(model, seed=23, sampling='interface', **SMALL)
    second = Objective(model, seed=23, sampling='interface', **SMALL)
    first.resample(seed=701)
    second.resample(seed=701)

    assert first.resampling_count == 1
    assert {key: len(points) for key, points in first.interior.items()} == {key: len(points) for key, points in second.interior.items()}
    assert all(torch.equal(first.interior[key], second.interior[key]) for key in first.interior)
    assert all(first.sources[key].shape[0] == len(first.interior[key]) for key in first.interior)


def test_model_shapes_and_parameter_partition():
    for architecture, networks in (('M0', 1), ('M1', 2), ('M2', 2)):
        model = _model('mms_interface', architecture)
        assert len(model.networks) == networks
        xy = torch.rand(5, 2, dtype=torch.float64)
        T, q = model.field(model.problem.regions[0], xy)
        assert T.shape == (5, 1) and (q is None) == (architecture != 'M2')


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
    state = torch.load(out.joinpath('model.pt'), map_location='cpu', weights_only=True)
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
    assert best['total'] == pytest.approx(1.)
    assert best['state']['parameter'].item() == pytest.approx(1.)
    assert model.parameter.item() == pytest.approx(1.)


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
