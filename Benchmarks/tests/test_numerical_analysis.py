import numpy as np
from Benchmarks.numerical_analysis import thermal_state,tangent,matrix_ampacity

def test_scalar_thermal_tangent_matches_closed_form_derivative():
    c=dict(current=10.,R20=.01,alpha=.004,T0=20.)
    a=np.array([[2.]]);direction=np.array([[.3]])
    dt,di=tangent(c,a,10.,direction)
    p0=1.;den=1-p0*.004*2
    np.testing.assert_allclose(dt,[p0*.3/den**2],rtol=1e-12)
    np.testing.assert_allclose(di,[2*.01*10*2/den**2],rtol=1e-12)

def test_scalar_ampacity_matches_independent_temperature_limit_formula():
    c=dict(current=10.,R20=.01,alpha=.004,T0=20.)
    current=matrix_ampacity(c,np.array([[2.]]))
    np.testing.assert_allclose(current,np.sqrt(70/(.01*1.28*2)),rtol=1e-10)

def test_multiconductor_tangent_matches_directional_perturbation():
    c=dict(current=10.,R20=.01,alpha=.004,T0=23.)
    a=np.array([[2.,.3],[.3,1.7]]);direction=np.array([[.2,-.01],[-.01,.1]])
    dt,_=tangent(c,a,10.,direction);eps=1e-3
    finite=(thermal_state(c,a+eps*direction,10.)[0]-thermal_state(c,a-eps*direction,10.)[0])/(2*eps)
    np.testing.assert_allclose(dt,finite,rtol=1e-7,atol=1e-9)
