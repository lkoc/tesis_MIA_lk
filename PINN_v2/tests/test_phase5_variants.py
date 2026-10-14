import numpy as np
import pytest
import torch

from pinn_v2.solvers.pinn.attention import Attention
from pinn_v2.solvers.pinn.networks import build_model, PINNModel
from pinn_v2.solvers.pinn.objective import Objective, flux_fields
from pinn_v2.solvers.pinn.problem import load_problem
from pinn_v2.solvers.pinn.variants import VARIANTS
from pinn_v2.solvers.pinn import trainer

SMALL = dict(n_interior=48, n_layer=24, n_face=24, n_boundary=24)
CASES = ['annulus', 'mms_constant', 'mms_high_contrast', 'mms_interface', 'mms_layered_y',
    'mms_robin', 'mms_smooth_2d', 'mms_variable', 'aras_flat', 'aras_single',
    'xlpe_discrete_layers', 'coaxial_full']
ARCHITECTURES = ['M0', 'M1', 'M2', *sorted(VARIANTS)]


def _model(case, architecture, seed=0):
    torch.manual_seed(seed)
    return build_model(load_problem(case), architecture, width=12, depth=2).double()


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_variant_objective_is_finite_and_differentiable(case, architecture):
    model = _model(case, architecture)
    total, parts, _ = Objective(model, **SMALL)()
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    expected_kind = 'faithful_to_idea_adapted' if architecture in VARIANTS else 'baseline'
    assert model.kind == expected_kind


def test_unknown_architecture_and_unimplemented_ipinn_are_rejected():
    with pytest.raises(ValueError):
        build_model(load_problem('mms_interface'), 'IPINN')
    with pytest.raises(ValueError):
        build_model(load_problem('mms_interface'), 'AdaI-PINN')


def _interface_slopes(model):
    problem = model.problem
    face = problem.faces(8)[0]
    xy = torch.as_tensor(face.xy, dtype=torch.float64).requires_grad_(True)
    normal = torch.as_tensor(face.normal, dtype=torch.float64)
    values = []
    for index in (face.left, face.right):
        T, _, derived, k = flux_fields(model, problem.regions[index], xy)
        values.append(((derived / k) * normal).sum(1).detach())
    return values


def test_cusp_features_give_a_gradient_kink_that_a_global_net_cannot():
    left, right = _interface_slopes(_model('mms_interface', 'CUSP'))
    assert torch.max(torch.abs(left - right)) > 1e-3
    left, right = _interface_slopes(_model('mms_interface', 'M0'))
    assert torch.max(torch.abs(left - right)) < 1e-12


def test_cusp_temperature_is_continuous_across_the_interface():
    model = _model('mms_interface', 'CUSP')
    problem = model.problem
    face = problem.faces(8)[0]
    xy = torch.as_tensor(face.xy, dtype=torch.float64)
    Tl = model.field(problem.regions[face.left], xy)[0]
    Tr = model.field(problem.regions[face.right], xy)[0]
    assert torch.max(torch.abs(Tl - Tr)) < 1e-12


def test_cusp_signs_follow_the_region_side():
    model = _model('coaxial_full', 'CUSP')
    problem = model.problem
    soil = next(r for r in problem.regions if r.kind == 'soil')
    conductor = next(r for r in problem.regions if r.kind == 'conductor')
    assert set(model.signs[soil.id].tolist()) == {1.}
    assert set(model.signs[conductor.id].tolist()) == {-1.}


@pytest.mark.parametrize('architecture', ['M0', 'M1', 'M2', 'CUSP', 'DCSNN', 'FBPINN'])
def test_annulus_geometry_features_work_for_every_architecture(architecture):
    problem = load_problem('annulus')
    model = _model('annulus', architecture)
    region = problem.regions[0]
    xy = torch.tensor([[.06, 0.], [20., 0.], [0., 20.]], dtype=torch.float64)
    coordinates = problem.coordinates(region, xy, model.partition)
    assert coordinates.shape == (3, 3)
    assert torch.allclose(coordinates[:, 0], torch.tensor([-1., 1., 1.], dtype=torch.float64))
    temperature = model.field(region, xy)[0]
    temperature.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_dcsnn_is_one_shared_network_with_the_requested_depth():
    model = _model('mms_interface', 'DCSNN')
    assert len(model.networks) == 1
    hidden = [m for m in model.networks[0] if isinstance(m, torch.nn.Linear)]
    assert len(hidden) == 3
    problem = model.problem
    xy = torch.as_tensor(problem.faces(4)[0].xy, dtype=torch.float64)
    a = model.field(problem.regions[0], xy)[0]
    b = model.field(problem.regions[1], xy)[0]
    assert torch.max(torch.abs(a - b)) > 1e-6


def test_fbpinn_cuts_align_with_the_planar_interface_and_blend_constants():
    model = _model('mms_interface', 'FBPINN')
    assert model.edges[0] == [0., .5, 1.]
    assert len(model.networks) == 4
    for net, level in zip(model.networks, (1., 2., 3., 4.)):
        for p in net.parameters():
            torch.nn.init.zeros_(p)
        net[-1].bias.data.fill_(level)
    problem = model.problem
    xy = torch.tensor([[.1, .1], [.9, .1], [.1, .9], [.9, .9]], dtype=torch.float64)
    values = model.field(problem.regions[0], xy)[0].flatten()
    expected = problem.T_ref + problem.T_scale * torch.tensor([1., 3., 2., 4.], dtype=torch.float64)
    assert torch.allclose(values, expected, atol=1e-6 * problem.T_scale)


def test_fbpinn_adds_local_cuts_around_cable_geometry():
    model = _model('coaxial_full', 'FBPINN')
    center = model.problem.case['cables'][0]
    radius = model.problem.case['radius']
    for axis in (0, 1):
        assert center[axis] - radius in model.edges[axis]
        assert center[axis] in model.edges[axis]
        assert center[axis] + radius in model.edges[axis]
    assert len(model.networks) > 4
    xmid = torch.tensor([[(center[0] - radius + center[0]) / 2]], dtype=torch.float64)
    local_weight = model._window(xmid, center[0] - radius, center[0], 0,
        model.problem.bounds[0], model.problem.bounds[1])
    assert float(local_weight) > 0.8


def test_rba_and_sa_update_rules():
    r2 = torch.tensor([1., 4., 0.25, 0.01], dtype=torch.float64)
    rba = Attention('rba', dict(a=4))
    for _ in range(5):
        rba.update('a', r2)
    weight = rba.weight('a').flatten()
    assert torch.all(weight >= 1.) and int(torch.argmax(weight)) == 1
    sa = Attention('sa', dict(a=4))
    for _ in range(5):
        sa.update('a', r2)
    weight = sa.weight('a').flatten()
    assert float(weight.mean()) == pytest.approx(1.) and int(torch.argmax(weight)) == 1
    with pytest.raises(ValueError):
        Attention('unknown', dict(a=1))


@pytest.mark.parametrize('kind', ['rba', 'sa'])
def test_attention_keeps_the_loss_value_and_changes_the_gradient(kind):
    model = _model('mms_interface', 'M1')
    plain = Objective(model, **SMALL)
    adaptive = Objective(model, attention=kind, **SMALL)
    assert all(state.device == next(model.parameters()).device for state in adaptive.attention.state.values())
    assert float(plain()[0].detach()) == pytest.approx(float(adaptive()[0].detach()), rel=1e-12)
    for _ in range(20):
        adaptive()
        adaptive.update_attention()

    def gradient(objective):
        model.zero_grad()
        objective()[0].backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])
    assert float(plain()[0].detach()) == pytest.approx(float(adaptive()[0].detach()), rel=1e-12)
    assert torch.linalg.vector_norm(gradient(plain) - gradient(adaptive)) > 1e-9


def test_attention_with_lbfgs_is_refused_and_run_is_recorded(tmp_path):
    refused = trainer.run('mms_constant', 'M1', tmp_path.joinpath('a'), adam_steps=1, lbfgs_steps=1, attention='rba', width=8, depth=1)
    assert refused['status'] == 'failed' and refused['error']['type'] == 'ValueError'
    assert tmp_path.joinpath('a', 'failure.json').exists()
    result = trainer.run('mms_constant', 'CUSP', tmp_path.joinpath('b'), adam_steps=3, width=8, depth=1, attention='sa', points=SMALL)
    assert result['status'] == 'completed' and result['variant_kind'] == 'faithful_to_idea_adapted'
    assert result['fem_labels_used'] == 0 and 'attention_weights' in result
