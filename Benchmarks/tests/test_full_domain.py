"""Physical identities and common interior enforcement, not saved loss snapshots."""
import copy
import math
import numpy as np
import pytest
import torch
from Benchmarks.cases import cases
from Benchmarks.full_domain import FullDomain,coaxial_case,radial_exact,dc_conductivity
from Benchmarks.full_physics import heat_residual,mixed_residual,polar_heat_residual
from Benchmarks.full_pinn import FullPINN,PhysicsLoss,VARIANTS
from Benchmarks.skin_effect import skin_factor,prescribed_source,loss_specification


def tensor(values):return torch.tensor(values,dtype=torch.float64,requires_grad=True)


def test_all_layers_partition_and_have_physical_sources():
    domain=FullDomain(cases()['xlpe_single'])
    assert len(domain.regions)==5 and len(domain.interfaces)==4
    assert sum(domain.area(r) for r in domain.regions)==pytest.approx(32.)
    for region in domain.regions:
        points=domain.sample(region,300,region.id)
        assert domain.contains(region,points).all()
        source=domain.source(region,points)
        expected=domain.case['power'] if region.kind=='conductor' else 0.
        assert float(source.mean())*domain.area(region)==pytest.approx(expected)
    assert all(i.radius>0 for i in domain.interfaces)


@pytest.mark.parametrize('variant',VARIANTS)
def test_every_variant_enforces_pde_in_every_layer_and_backpropagates(variant):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.manual_seed(3)
    domain=FullDomain(coaxial_case());model=FullPINN(domain,variant,width=8,depth=2)
    objective=PhysicsLoss(model,n=20,n_layer=16,n_interface=12)
    for _ in range(2):
        model.zero_grad(set_to_none=True);loss,parts,regions=objective()
        assert set(regions)=={r.name for r in domain.regions}
        assert all(torch.isfinite(v) and v.item()>0 for v in regions.values())
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_polar_operator_retains_angular_diffusion_and_storage():
    coordinates=tensor([[.3,.2,.7],[.4,.9,.1],[.8,1.2,.4]])
    r=coordinates[:,:1];phi=coordinates[:,1:2];t=coordinates[:,2:3]
    T=r**4*torch.cos(2*phi)+t*3.
    k=2.;capacity=5.;Q=15.-24*r*r*torch.cos(2*phi)
    assert polar_heat_residual(T,coordinates,k,Q,capacity).abs().max()<1e-10
    with pytest.raises(ValueError,match='rho'):
        polar_heat_residual(T,coordinates,k,Q)


def test_cartesian_transient_and_mixed_equations_agree():
    xy=tensor([[0.,0.,.3],[.3,-.2,.7],[-.4,.7,1.]])
    T=(xy[:,:2]**2).sum(1,keepdim=True)+3*xy[:,2:3]
    k=2.;Q=7.;capacity=5.;q=-4*xy[:,:2]
    assert heat_residual(T,xy,k,Q,capacity).abs().max()<1e-12
    balance,law=mixed_residual(T,q,xy,k,Q,capacity)
    assert balance.abs().max()<1e-12 and law.abs().max()<1e-12


def test_logpolar_coordinates_are_periodic_and_center_is_regular():
    model=FullPINN(FullDomain(coaxial_case()),'subdomain',8,2).double()
    core=model.domain.conductors[0];center=tensor([[0.,0.]])
    T,_=model.field(core,center)
    assert torch.isfinite(heat_residual(T,center,core.k,model.domain.source(core,center))).all()
    ring=model.domain.regions[core.id+1]
    points=tensor([[ring.ri*1.1,1e-12],[ring.ri*1.1,-1e-12]])
    temp,_=model.field(ring,points)
    assert abs(temp[0]-temp[1])<1e-7


def test_skin_source_integrates_to_ac_loss_and_dc_limit():
    c=coaxial_case();a=c['layers'][0][1]
    c['electrical']=dict(resistance_basis='dc',frequency_Hz=60.,skin_model='solid_round',profile='skin')
    z,w=np.polynomial.legendre.leggauss(100);r=a*(z+1)/2
    q=prescribed_source(c,np.c_[r,r*0],[0,0]).ravel()
    power=float(np.sum(w*q*r)*a/2*2*math.pi)
    assert power==pytest.approx(c['power']*skin_factor(c['R20'],a,60.),rel=1e-10)
    assert skin_factor(c['R20'],a,0.)==1.
    assert skin_factor(c['R20'],a,.001)==pytest.approx(1.,abs=1e-10)
    assert q[-1]>q[0]


def test_ac_resistance_cannot_be_corrected_twice():
    c=coaxial_case();c['electrical']=dict(resistance_basis='ac',frequency_Hz=60.,skin_model='solid_round')
    with pytest.raises(ValueError,match='second time'):loss_specification(c)
    c['electrical']['skin_model']='none'
    assert loss_specification(c)['frequency_Hz']==60.


def test_transient_requires_documented_capacity_and_new_reference_identity():
    d=FullDomain(coaxial_case())
    with pytest.raises(ValueError,match='Missing documented'):d.capacity(d.conductors[0])
    changed=copy.deepcopy(d.case);changed['heat_capacity_J_m3K']={r.name:1e6 for r in d.regions}
    assert FullDomain(changed).fingerprint!=d.fingerprint
    assert FullDomain(changed).capacity(d.conductors[0])==1e6


def test_local_dc_source_conserves_prescribed_current_without_mean_temperature():
    d=FullDomain(coaxial_case(),'dc_temperature');r=d.conductors[0]
    temperatures=tensor([[20.],[60.],[90.],[30.]])
    sigma=dc_conductivity(d,r,temperatures);field=d.case['current']/(sigma.mean()*d.area(r))
    assert float((sigma*field).mean().detach())*d.area(r)==pytest.approx(d.case['current'])
    assert sigma.max()>sigma.min()


def test_multiscale_sampling_covers_far_soil_and_thin_layers():
    d=FullDomain(cases()['xlpe_single']);soil=d.regions[0]
    points=d.sample(soil,512,11,near=True)
    distances=np.linalg.norm(points-np.asarray(d.case['cables'][0]),axis=1)
    assert (distances>1.).sum()>100
    assert (distances<.31).sum()>=256
    assert np.array_equal(points,d.sample(soil,512,11,near=True))


def test_dc_input_cannot_silently_omit_ac_skin_correction():
    c=coaxial_case();c['electrical']=dict(frequency_Hz=60.)
    with pytest.raises(ValueError,match='requires a documented skin correction'):loss_specification(c)
    with pytest.raises(ValueError,match='finite'):skin_factor(float('nan'),.01,60.)
