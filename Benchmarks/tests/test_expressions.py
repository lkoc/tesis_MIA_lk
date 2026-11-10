import numpy as np
import pytest
import torch
from Benchmarks.expressions import evaluate,conductivity,derivative
from Benchmarks.cases import cases,exact,field_k
from pinn_cables.pinn.pde import gradients

@pytest.mark.parametrize('expr',["__import__('os').system('bad')",'x.__class__','x[0]','unknown(x)','[x for x in y]'])
def test_expression_language_rejects_code(expr):
    with pytest.raises((ValueError,SyntaxError)):evaluate(expr,np.array([1.]),np.array([1.]),np)

def test_discrete_layers_preserve_jump_and_smoothing_is_explicit():
    spec=dict(type='layers',axis='y',interfaces=[0.],values=[.5,2.],smoothing=0.)
    x=np.zeros(2);y=np.array([-1e-8,1e-8])
    assert np.allclose(conductivity(spec,x,y,np),[.5,2.])
    spec['smoothing']=.1
    assert np.allclose(conductivity(spec,x,y,np),[1.25,1.25],atol=1e-6)

def test_manufactured_horizontal_interface_traces():
    c=cases()['mms_layered_y'];eps=1e-8
    a=torch.tensor([[.2,.5-eps],[.7,.5-eps]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[.2,.5+eps],[.7,.5+eps]],dtype=torch.float64,requires_grad=True)
    ta=exact(c,a[:,:1],a[:,1:2],torch);tb=exact(c,b[:,:1],b[:,1:2],torch)
    qa=field_k(c,a[:,:1],a[:,1:2],torch)*gradients(ta,a)[:,1:2]
    qb=field_k(c,b[:,:1],b[:,1:2],torch)*gradients(tb,b)[:,1:2]
    assert torch.max(abs(ta-tb)).item()<1e-6
    assert torch.max(abs(qa-qb)).item()<1e-10

def test_constant_power_derivative_remains_finite_at_zero():
    assert np.isfinite(evaluate(derivative('x**0','x'),np.array([0.]),np.array([0.]),np)).all()
