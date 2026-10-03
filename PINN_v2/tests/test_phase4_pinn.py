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


def test_trainer_writes_record_and_never_overwrites(tmp_path):
    out = tmp_path / 'run'
    result = trainer.run('mms_constant', 'M1', out, adam_steps=30, width=8, depth=2, points=SMALL, log_every=10)
    assert result['status'] == 'completed' and result['fem_labels_used'] == 0 and result['reference_used_in_loss'] is False
    for name in ('result.json', 'manifest.json', 'evaluation.npz', 'collocation.npz', 'model.pt'):
        assert out.joinpath(name).exists()
    manifest = json.loads(out.joinpath('manifest.json').read_text())
    assert manifest['seeds'] == dict(init=0, sampling=11)
    with pytest.raises(FileExistsError):
        trainer.run('mms_constant', 'M1', out, adam_steps=1, points=SMALL)


def test_trainer_is_deterministic(tmp_path):
    kwargs = dict(adam_steps=20, width=8, depth=2, points=SMALL, log_every=10)
    a = trainer.run('mms_interface', 'M1', tmp_path / 'a', **kwargs)
    b = trainer.run('mms_interface', 'M1', tmp_path / 'b', **kwargs)
    assert a['final_loss'] == b['final_loss']
    c = trainer.run('mms_interface', 'M1', tmp_path / 'c', init_seed=3, **kwargs)
    assert c['final_loss'] != a['final_loss']


def test_failure_is_preserved(tmp_path):
    result = trainer.run('mms_constant', 'M1', tmp_path / 'bad', adam_steps=5, lr=float('nan'), width=8, depth=1, points=SMALL)
    assert result['status'] == 'failed'
    assert tmp_path.joinpath('bad', 'failure.json').exists() and tmp_path.joinpath('bad', 'manifest.json').exists()


@pytest.mark.slow
def test_m1_learns_manufactured_case(tmp_path):
    result = trainer.run('mms_constant', 'M1', tmp_path / 'learn', adam_steps=600, lr=3e-3, log_every=200)
    assert result['evaluation']['nrmse_pct'] < 5.
    assert result['evaluation']['acceptance']['state'] == 'accepted'
