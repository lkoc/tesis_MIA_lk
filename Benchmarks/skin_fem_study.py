"""Full 2D source-profile comparison at equal AC power on the Aras cable case."""
from pathlib import Path
import copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.cases import cases
from Benchmarks.full_domain import FullDomain
from Benchmarks.full_fem import solve


def main():
    output=ROOT/'Benchmarks/explicit_results/skin_aras_60Hz/equal_power';records={}
    for profile in ['uniform','skin']:
        c=copy.deepcopy(cases()['aras_single'])
        c['electrical']=dict(resistance_basis='dc',frequency_Hz=60.,skin_model='solid_round',profile=profile,
            provenance='Sensitivity only: catalog R20 assumed DC and conductor treated as homogeneous solid round; original construction not certified')
        domain=FullDomain(c);folder=output/profile;folder.mkdir(parents=True,exist_ok=True)
        (folder/'case.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
        records[profile]=[solve(domain,level,folder,normalize_source=True) for level in [0,1,2,3]]
    fields={profile:np.load(output/profile/'fem_l3.npz') for profile in records}
    difference=fields['skin']['T']-fields['uniform']['T']
    report=dict(scope='Fixed 60 Hz electrical source at 20 C, same nominal AC power, full 2D conduction; conditional solid-round skin model only',
        source_difference_pct=100*(records['skin'][-1]['source_W_m']/records['uniform'][-1]['source_W_m']-1),
        Tmax_change_K=records['skin'][-1]['Tmax_C']-records['uniform'][-1]['Tmax_C'],
        max_sample_field_difference_K=float(abs(difference).max()),
        rmse_field_difference_K=float(np.sqrt(np.average(difference**2,weights=fields['skin']['weights']))),
        records=records)
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
