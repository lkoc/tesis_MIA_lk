import pytest
from pinn_cables.physics.iec60287 import iterate_R_T

def test_resistance_iteration_tracks_each_conductor_and_total_loss(monkeypatch):
    calls=[]
    def estimate(*args,**kwargs):
        calls.append(kwargs['Q_lins'])
        return dict(cables=[dict(T_cond=303.15),dict(T_cond=313.15)],T_cond_ref=313.15)
    monkeypatch.setattr('pinn_cables.physics.kennelly.iec60287_estimate',estimate)
    result,q=iterate_R_T([[],[]],[None,None],1.,293.15,[10.,10.],[.01,.01],[.004,.004],293.15,Q_d=.2)
    assert result['converged']
    assert q==pytest.approx([1.24,1.28])
    assert len(calls)==2

def test_nonconverged_resistance_iteration_is_not_returned_as_a_result(monkeypatch):
    def estimate(*args,**kwargs):return dict(cables=[dict(T_cond=300.)],T_cond_ref=300.)
    monkeypatch.setattr('pinn_cables.physics.kennelly.iec60287_estimate',estimate)
    with pytest.raises(RuntimeError,match='failed to converge'):
        iterate_R_T([[]],[None],1.,293.15,[10.],[.01],[.004],293.15,n_iter=1)
