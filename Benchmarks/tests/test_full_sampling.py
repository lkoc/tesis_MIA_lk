import numpy as np
import pytest
import torch
from Benchmarks.full_domain import FullDomain
from Benchmarks.cases import cases
from Benchmarks.full_pinn import FullPINN,PhysicsLoss
from Benchmarks.full_sampling import sample,adapt,POLICIES

@pytest.mark.parametrize('policy',POLICIES)
def test_sampling_preserves_each_material_and_global_coverage(policy):
    d=FullDomain(cases()['xlpe_discrete_layers'])
    for r in d.regions:
        xy=sample(d,r,128,17+r.id,policy)
        assert len(xy)==128 and d.contains(r,xy).all()
        assert np.array_equal(xy,sample(d,r,128,17+r.id,policy))
        if r.kind=='soil':assert np.ptp(xy[:,0])>3

@pytest.mark.parametrize('variant',['subdomain','mixed'])
def test_adaptation_keeps_budget_and_allows_training_without_fem_labels(variant):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    d=FullDomain(cases()['xlpe_discrete_layers']);model=FullPINN(d,variant,8,2)
    objective=PhysicsLoss(model,32,24,16,sampling='residual')
    counts={i:len(x) for i,x in objective.interior.items()}
    records=adapt(objective,881)
    assert len(records)==len(d.regions)
    for r in d.regions:
        assert len(objective.interior[r.id])==counts[r.id]
        assert d.contains(r,objective.interior[r.id].detach().numpy()).all()
    loss,_,_=objective();loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
