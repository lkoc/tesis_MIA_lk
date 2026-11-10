"""Common per-conductor DC R(T) law and converged thermal-response coupling."""
import numpy as np

LOSS_MODEL=dict(type='dc_linear_temperature',reference_temperature_C=20.,temperature_tolerance_K=1e-5,temperature_limit_C=90.,max_iterations=500)

def solve_powers(case,response,current=None):
    n=len(case['cables']);I=case['current'] if current is None else current
    p20=I**2*case['R20'];alpha=case['alpha'];a=np.asarray(response)
    coupling=p20*alpha*a
    radius=float(max(abs(np.linalg.eigvals(coupling))))
    if radius>=1:raise RuntimeError('No stable steady state in linear R(T) model')
    power=np.full(n,p20*(1+alpha*(case['T0']-20.)));history=[]
    previous=np.full(n,case['T0'])
    for i in range(LOSS_MODEL['max_iterations']):
        temperature=case['T0']+a@power
        updated=p20*(1+alpha*(temperature-20.))
        residual=float(max(abs(temperature-previous)))
        history.append(dict(iteration=i,conductor_C=temperature.tolist(),powers_W_m=power.tolist(),temperature_change_K=residual))
        if residual<LOSS_MODEL['temperature_tolerance_K']:break
        previous=temperature;power=updated
    else:raise RuntimeError('R(T) fixed point did not converge')
    exact=np.linalg.solve(np.eye(n)-coupling,np.full(n,p20*(1+alpha*(case['T0']-20.))))
    if not np.allclose(power,exact,rtol=1e-6,atol=1e-6):raise RuntimeError('Iteration differs from independent matrix solution')
    return exact,dict(spectral_radius=radius,iterations=len(history),history=history,iteration_closed_form_difference_W_m=float(max(abs(power-exact))))

def ampacity(case,response,limit=90.):
    lo=0.;hi=case['current'];history=[]
    def temp(I):
        try:p,_=solve_powers(case,response,I);return float(max(case['T0']+response@p))
        except RuntimeError:return float('inf')
    while temp(hi)<limit:hi*=2
    for i in range(80):
        current=(lo+hi)/2;t=temp(current)
        history.append(dict(iteration=i,current_A=current,Tmax_C=t if np.isfinite(t) else None,stable=bool(np.isfinite(t))))
        if abs(t-limit)<1e-4:break
        if t>limit:hi=current
        else:lo=current
    else:raise RuntimeError('Ampacity bisection did not converge')
    power,details=solve_powers(case,response,current)
    return current,power,dict(bisection=history,**details)
