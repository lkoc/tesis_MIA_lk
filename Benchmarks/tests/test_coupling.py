import numpy as np
import pytest
import torch
from Benchmarks.cases import cases
from Benchmarks.electrothermal import solve_powers,ampacity
from Benchmarks.pinn import Network,predict

def test_single_conductor_closed_form_and_ampacity():
    c=cases()['xlpe_single'];a=np.array([[1.2]])
    p,trace=solve_powers(c,a)
    expected=c['current']**2*c['R20']/(1-c['alpha']*c['current']**2*c['R20']*a[0,0])
    assert p[0]==pytest.approx(expected,rel=1e-12)
    current,power,_=ampacity(c,a)
    analytic=np.sqrt((90-c['T0'])/(a[0,0]*c['R20']*(1+c['alpha']*70)))
    assert current==pytest.approx(analytic,rel=2e-6)
    assert abs(c['T0']+(a@power)[0]-90)<1e-4
    assert trace['iterations']>2

def test_individual_temperature_changes_individual_losses():
    c=cases()['aras_flat'];a=np.array([[.8,.2,.1],[.2,.9,.2],[.1,.2,.8]])
    p,_=solve_powers(c,a)
    tc=c['T0']+a@p
    np.testing.assert_allclose(p,c['current']**2*c['R20']*(1+c['alpha']*(tc-20)),rtol=1e-12)
    assert p[1]>p[0]
    assert p[0]==pytest.approx(p[2])

def test_unstable_linear_electrothermal_feedback_is_rejected():
    c=cases()['xlpe_single']
    with pytest.raises(RuntimeError,match='stable'):
        solve_powers(c,np.array([[100.]]))

def test_pinn_temperature_depends_on_each_power_unknown():
    torch.set_default_dtype(torch.float64)
    c=cases()['aras_flat'];m=Network(c,16,3,'multipole',True,False)
    z=torch.tensor([[.1,-.7],[.8,-2.]],requires_grad=True)
    t=predict(m,z)
    g=torch.autograd.grad(t.sum(),m.power_correction)[0]
    assert torch.isfinite(g).all() and torch.all(abs(g)>0)
    np.testing.assert_allclose(m.powers().detach().numpy(),np.full(3,c['power']))
