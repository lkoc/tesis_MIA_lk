"""Physical identity and soil-average controls for the nonlinear current limit."""
import copy
import numpy as np
import pytest
from Benchmarks.full_ampacity import AmpacityDomain,ampacity_cases
from Benchmarks.cases import field_k
from Benchmarks.full_domain import dc_conductivity


def test_unknown_current_identity_is_stable_but_limit_and_physics_are_not():
    case=ampacity_cases()['xlpe_single'];domain=AmpacityDomain(case)
    identity=domain.fingerprint
    domain.case['current']=480.
    assert domain.fingerprint==identity
    assert domain.case['current']!=case['current']
    assert AmpacityDomain(case,89.).fingerprint!=identity
    changed=copy.deepcopy(case);changed['R20']*=1.1;changed['power']*=1.1
    assert AmpacityDomain(changed).fingerprint!=identity
    with pytest.raises(ValueError):AmpacityDomain(case,case['T0'])


def test_soil_mean_agrees_with_independent_cell_midpoint_integration():
    catalog=ampacity_cases();heterogeneous=catalog['xlpe_dry_near']
    domain=AmpacityDomain(heterogeneous);x0,x1,y0,y1=heterogeneous['bounds']
    nx,ny=1800,900
    x=x0+(np.arange(nx)+.5)*(x1-x0)/nx
    y=y0+(np.arange(ny)+.5)*(y1-y0)/ny
    xx,yy=np.meshgrid(x,y);points=np.c_[xx.ravel(),yy.ravel()]
    points=points[domain.contains(domain.regions[0],points)]
    mean=field_k(heterogeneous,points[:,0],points[:,1],np).mean()
    control=catalog['xlpe_dry_near_hom_arithmetic']
    assert control['k']==pytest.approx(mean,abs=3e-5)
    for key in ['cables','layers','current','R20','bounds']:
        assert control[key]==heterogeneous[key]


def test_local_joule_source_recovers_uniform_temperature_dc_power():
    domain=AmpacityDomain(ampacity_cases()['xlpe_single']);r=domain.conductors[0]
    temperature=np.full((64,1),90.);current=480.
    sigma=dc_conductivity(domain,r,temperature)
    conductance=domain.area(r)*sigma.mean();field=current/conductance
    power=domain.area(r)*np.mean(sigma*field**2)
    expected=current**2*domain.case['R20']*(1+domain.case['alpha']*70.)
    assert power==pytest.approx(expected,rel=1e-12)
