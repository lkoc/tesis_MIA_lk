"""Compare the unchanged default loss/gradient against the archived A solver."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_domain import FullDomain
from Benchmarks.cases import cases
from Benchmarks.full_pinn import FullPINN,PhysicsLoss


def main():
    folder=ROOT/'Benchmarks/explicit_study/A/subdomain_w32_d3/xlpe_single/seed11'
    old_path=next((folder/'source').glob('*_full_pinn.py'))
    spec=importlib.util.spec_from_file_location('archived_pinn_weight_audit',old_path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);domain=FullDomain(cases()['xlpe_discrete_layers'])
    torch.manual_seed(11);a=old.FullPINN(domain,'subdomain',8,2)
    torch.manual_seed(11);b=FullPINN(domain,'subdomain',8,2)
    la,_,ra=old.PhysicsLoss(a,32,24,16,101)();lb,_,_=PhysicsLoss(b,32,24,16,101)()
    la.backward();lb.backward()
    gradients=all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()) if p.requires_grad)
    _,_,rc=PhysicsLoss(b,32,24,16,101,temperature_weight=10000)()
    result=dict(default_loss_bitwise_identical=bool(torch.equal(la,lb)),default_gradients_bitwise_identical=gradients,PDE_parts_unchanged_by_weight=all(torch.equal(ra[k],rc[k]) for k in ra),scope='PhysicsLoss weight extension; shared current domain and operators in both comparisons')
    output=ROOT/'docs/auditoria/revision_integral_2026-09-15';archive=output/'weight_source';archive.mkdir(exist_ok=True)
    for key,path in [('old',old_path),('new',ROOT/'Benchmarks/full_pinn.py')]:
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();result[key+'_source_sha256']=digest;(archive/(digest+'_full_pinn.py')).write_bytes(data)
    (output/'weight_regression.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(result[k] for k in ['default_loss_bitwise_identical','default_gradients_bitwise_identical','PDE_parts_unchanged_by_weight'])
    print(json.dumps(result))


if __name__=='__main__':main()
