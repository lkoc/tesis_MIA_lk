"""Skin-only estimate for a homogeneous, nonmagnetic, solid circular conductor.

Exact cylindrical electromagnetic solution, RMS phasors. This is an electrical
source model, not a thermal equivalent. Stranding/proximity need separate data.
"""
import math
import numpy as np
from scipy.special import jv

MU0=4*math.pi*1e-7


def skin_factor(resistance_dc_ohm_m,radius_m,frequency_Hz,mu_r=1.):
    if not all(np.isfinite(v) for v in (resistance_dc_ohm_m,radius_m,frequency_Hz,mu_r)):
        raise ValueError('Electrical inputs must be finite')
    if min(resistance_dc_ohm_m,radius_m,mu_r)<=0 or frequency_Hz<0:
        raise ValueError('Positive DC resistance, radius, permeability and nonnegative frequency required')
    if frequency_Hz==0:return 1.
    rho=resistance_dc_ohm_m*math.pi*radius_m**2
    delta=math.sqrt(rho/(math.pi*frequency_Hz*MU0*mu_r))
    z=(1-1j)*radius_m/delta
    if abs(z)<1e-3:return 1.+abs(z)**4/192.
    return float(np.real(z*jv(0,z)/(2*jv(1,z))))


def loss_specification(case):
    spec=dict(case.get('electrical',{}))
    spec.setdefault('resistance_basis','dc')
    spec.setdefault('resistance_ohm_m',case['R20'])
    spec.setdefault('reference_temperature_C',20.)
    spec.setdefault('source_temperature_C',spec['reference_temperature_C'])
    spec.setdefault('frequency_Hz',0.)
    spec.setdefault('skin_model','none')
    spec.setdefault('profile','uniform')
    spec.setdefault('mu_r',1.)
    spec.setdefault('provenance','Legacy R20 explicitly treated as DC; confirm original datasheet before AC operation')
    allowed={'resistance_basis','resistance_ohm_m','reference_temperature_C','source_temperature_C',
        'frequency_Hz','skin_model','profile','mu_r','provenance','includes_proximity'}
    if set(spec)-allowed:raise ValueError('Unknown electrical fields')
    if spec['resistance_basis'] not in ('dc','ac'):raise ValueError('Resistance basis must be dc or ac')
    if spec['skin_model'] not in ('none','solid_round'):raise ValueError('Unknown skin model')
    if spec['profile'] not in ('uniform','skin'):raise ValueError('Unknown heat-source profile')
    if spec['resistance_ohm_m']<=0 or spec['frequency_Hz']<0 or spec['mu_r']<=0:
        raise ValueError('Invalid electrical inputs')
    if spec['resistance_basis']=='ac':
        if spec['frequency_Hz']<=0:raise ValueError('AC resistance requires its measurement frequency')
        if spec['skin_model']!='none':raise ValueError('Do not correct an AC resistance a second time for skin effect')
        if spec['profile']=='skin':raise ValueError('An AC resistance alone does not determine the radial current profile; provide an independent field model')
    if spec['profile']=='skin' and (spec['skin_model']!='solid_round' or spec['frequency_Hz']<=0):
        raise ValueError('Skin heat profile requires the solid-round AC source model')
    if spec['resistance_basis']=='dc' and spec['frequency_Hz']>0 and spec['skin_model']=='none':
        raise ValueError('DC resistance at AC frequency requires a documented skin correction')
    if spec['resistance_basis']=='ac' and spec['source_temperature_C']!=spec['reference_temperature_C']:
        raise ValueError('AC R(T) cannot be inferred by multiplying measured Rac by the DC temperature factor; supply Rac at source temperature')
    return spec


def prescribed_power(case):
    spec=loss_specification(case);resistance=spec['resistance_ohm_m']
    if spec['resistance_basis']=='dc':
        resistance*=1+case['alpha']*(spec['source_temperature_C']-spec['reference_temperature_C'])
    factor=skin_factor(resistance,case['layers'][0][1],spec['frequency_Hz'],spec['mu_r']) if spec['skin_model']=='solid_round' else 1.
    return case['current']**2*resistance*factor


def prescribed_source(case,xy,center):
    spec=loss_specification(case);a=case['layers'][0][1];area=math.pi*a*a
    resistance=spec['resistance_ohm_m']
    if spec['resistance_basis']=='dc':
        resistance*=1+case['alpha']*(spec['source_temperature_C']-spec['reference_temperature_C'])
    if resistance<=0:raise ValueError('Nonpositive electrical resistivity')
    ratio=skin_factor(resistance,a,spec['frequency_Hz'],spec['mu_r']) if spec['skin_model']=='solid_round' else 1.
    power=case['current']**2*resistance*ratio
    if spec['profile']=='uniform':return np.full((len(xy),1),power/area)
    rho=resistance*area;delta=math.sqrt(rho/(math.pi*spec['frequency_Hz']*MU0*spec['mu_r']))
    gamma=(1-1j)/delta
    radius=np.minimum(np.linalg.norm(np.asarray(xy)[:,:2]-center,axis=1),a)
    current_density=case['current']*gamma*jv(0,gamma*radius)/(2*math.pi*a*jv(1,gamma*a))
    return (rho*abs(current_density)**2).reshape(-1,1)


def estimate(case,frequency_Hz=60.,temperature_C=20.):
    a=case['layers'][0][1];k=case['layers'][0][2]
    rdc=case['R20']*(1+case['alpha']*(temperature_C-20.))
    ratio=skin_factor(rdc,a,frequency_Hz)
    rho=rdc*math.pi*a*a
    return dict(case=case['id'],temperature_C=temperature_C,radius_mm=1000*a,
        frequency_Hz=frequency_Hz,skin_depth_mm=1000*math.sqrt(rho/(math.pi*frequency_Hz*MU0)),
        Rac_Rdc=ratio,loss_increase_pct=100*(ratio-1),
        current_change_at_fixed_loss_pct=100*(1/math.sqrt(ratio)-1),
        source_redistribution_radial_bound_K=case['current']**2*rdc*ratio/(4*math.pi*k),
        assumptions='R20 assumed DC; solid homogeneous round isolated conductor, mu_r=1; no proximity. Bound compares uniform versus outward-redistributed heat at the SAME total power under radial symmetry; not a full 2D error bound.')
