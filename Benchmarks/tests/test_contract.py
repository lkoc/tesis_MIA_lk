import copy
import numpy as np
import pytest
import torch
from Benchmarks.cases import cases,validate_case,field_k,exact,source,fingerprint
from pinn_cables.pinn.pde import laplace_variable_k

def test_saved_fem_matches_current_specification():
    import json
    from pathlib import Path
    for name,c in cases().items():
        p=Path('Benchmarks/results')/name/'fem_l2.json'
        if p.exists(): assert json.loads(p.read_text())['case_sha256']==fingerprint(c)

@pytest.mark.parametrize('name',['mms_constant','mms_variable','mms_interface','mms_robin','mms_smooth_2d','mms_high_contrast','mms_layered_y'])
def test_manufactured_source_is_independent_identity(name):
    c=cases()[name]
    xy=torch.tensor([[.13,.22],[.41,.77],[.63,.36],[.89,.58]],dtype=torch.float64,requires_grad=True)
    x,y=xy[:,:1],xy[:,1:2]
    r=laplace_variable_k(exact(c,x,y,torch),xy,field_k(c,x,y,torch))+source(c,x,y,torch)
    assert torch.max(abs(r)).item()<1e-10

def test_invalid_geometry_and_power_are_rejected():
    c=copy.deepcopy(cases()['xlpe_single']);c['cables']*=2
    with pytest.raises(ValueError,match='Overlapping'):validate_case(c)
    c=copy.deepcopy(cases()['xlpe_single']);c['power']*=2
    with pytest.raises(ValueError,match='power'):validate_case(c)

def test_scalar_tensor_conductivity_agree():
    for c in cases().values():
        xy=np.array([[.1,-.2],[.2,-.8],[-.3,-1.4],[.4,-2.]])
        a=field_k(c,xy[:,0],xy[:,1]);z=torch.tensor(xy)
        assert np.allclose(a,field_k(c,z[:,0],z[:,1],torch).numpy())
