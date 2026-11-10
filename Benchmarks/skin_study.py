"""Reproducible 60 Hz screening, not a construction-specific cable rating."""
from pathlib import Path
import copy,json,math,sys
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.cases import cases
from Benchmarks.skin_effect import estimate,prescribed_source,prescribed_power


def study():
    rows=[]
    for name in ['xlpe_single','aras_single','kim_pac']:
        case=cases()[name]
        for temperature in [20.,90.]:
            result=estimate(case,60.,temperature);c=copy.deepcopy(case)
            c['electrical']=dict(resistance_basis='dc',frequency_Hz=60.,skin_model='solid_round',profile='skin',source_temperature_C=temperature)
            a=c['layers'][0][1];k=c['layers'][0][2]
            # Integral solution of radial Poisson in the conductor with T(a)=0;
            # independent diagnostic only, never a thermal PINN loss term.
            def integrand(r):
                return float(prescribed_source(c,np.array([[r,0.]]),[0.,0.])[0,0])*r*math.log(a/r)/k if r else 0.
            skin_rise,error=quad(integrand,0,a,epsabs=1e-12)
            uniform_rise=prescribed_power(c)/(4*math.pi*k)
            result.update(radial_core_rise_skin_K=skin_rise,radial_core_rise_uniform_same_power_K=uniform_rise,
                Tmax_change_skin_minus_uniform_same_power_K=skin_rise-uniform_rise,quadrature_error_K=error,
                catalog_source=case['source'],catalog_current_A=case['current'],catalog_R20_ohm_m=case['R20'])
            rows.append(result)
    return dict(scope='Conditional electrical screening: R20 assumed DC, solid homogeneous isolated round conductor. Not an actual cable AC rating.',
        sources=['https://arxiv.org/abs/1303.5452','https://www.comsol.com/support/learning-center/article/81171'],rows=rows)


if __name__=='__main__':
    output=ROOT/'docs/auditoria/metodologia_2026-09-14/skin_effect.json'
    output.write_text(json.dumps(study(),indent=2),encoding='utf-8');print(output)
