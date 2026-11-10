"""Explicit heat-equation PINNs: identical material physics in every variant.

Cartesian conductor, angular/logradial annuli, Cartesian soil. Differentiation
is with respect to physical x,y, so all coordinate Jacobians remain in the PDE.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,platform,sys,time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Benchmarks.cases import cases
from Benchmarks.full_domain import FullDomain,coaxial_case,dc_conductivity,radial_exact,PHYSICS_VERSION
from Benchmarks.full_physics import heat_residual,mixed_residual
from pinn_cables.pinn.pde import gradients

VARIANTS=('global','direct','subdomain','enriched','local','conservative','multipole','mixed','fourier')


class FullPINN(nn.Module):
    def __init__(self,domain,variant='mixed',width=32,depth=3):
        super().__init__()
        if variant not in VARIANTS:raise ValueError('Unknown explicit-domain architecture')
        self.domain=domain;self.variant=variant;self.width=width;self.depth=depth
        self.global_network=variant in ('global','direct')
        self.mixed=variant=='mixed'
        self.networks=nn.ModuleList();self.offsets=nn.ParameterList()
        self.harmonics=nn.ParameterList()
        representatives=domain.regions[:1] if self.global_network else domain.regions
        for region in representatives:
            nin=self.input_size(region)
            if variant=='fourier':nin*=3
            layers=[]
            for _ in range(depth):
                linear=nn.Linear(nin,width);nn.init.xavier_normal_(linear.weight);nn.init.zeros_(linear.bias)
                layers.extend([linear,nn.Tanh()]);nin=width
            last=nn.Linear(width,3 if self.mixed else 1)
            nn.init.xavier_normal_(last.weight);nn.init.zeros_(last.bias);layers.append(last)
            self.networks.append(nn.Sequential(*layers));self.offsets.append(nn.Parameter(torch.zeros(())))
            self.harmonics.append(nn.Parameter(torch.zeros(6),requires_grad=variant=='multipole'))

    def input_size(self,region):
        if self.global_network or self.variant=='local':return 2
        if region.kind=='soil':return 2+len(self.domain.case['cables'])
        return 2 if region.kind=='conductor' else 3

    def coordinates(self,region,xy):
        c=self.domain.case
        if self.global_network or region.kind=='soil':
            x0,x1,y0,y1=c['bounds']
            result=torch.cat([2*(xy[:,:1]-x0)/(x1-x0)-1,2*(xy[:,1:2]-y0)/(y1-y0)-1],dim=1)
            if not self.global_network and self.variant!='local':
                extras=[]
                for center in c['cables']:
                    delta=xy[:,:2]-xy.new_tensor(center)
                    extras.append(torch.log(torch.linalg.vector_norm(delta,dim=1,keepdim=True)/c['radius']))
                result=torch.cat([result,*extras],dim=1)
        else:
            delta=xy[:,:2]-xy.new_tensor(c['cables'][region.cable])
            if region.kind=='conductor' or self.variant=='local':result=delta/region.ro
            else:
                radius=torch.linalg.vector_norm(delta,dim=1,keepdim=True)
                eta=torch.log(radius/region.ri)/math.log(region.ro/region.ri)
                result=torch.cat([2*eta-1,delta/radius],dim=1)
        if self.variant=='fourier':result=torch.cat([result,torch.sin(math.pi*result),torch.cos(math.pi*result)],dim=1)
        return result

    def envelope(self,xy):
        c=self.domain.case
        if c.get('outer_radius'):return 1-(xy[:,:2]**2).sum(dim=1,keepdim=True)/c['outer_radius']**2
        x0,x1,y0,y1=c['bounds'];u=2*(xy[:,:1]-x0)/(x1-x0)-1;v=2*(xy[:,1:2]-y0)/(y1-y0)-1
        return (1-u*u)*(1-v*v)

    def field(self,region,xy):
        index=0 if self.global_network else region.id
        raw=self.networks[index](self.coordinates(region,xy));scales=self.domain.scales(region)
        amplitude=self.domain.case['scale'] if self.global_network or region.kind=='soil' else scales['temperature']
        temperature=self.domain.case['scale']*self.offsets[index]+amplitude*raw[:,:1]
        if self.variant=='multipole':
            c=self.domain.case
            if region.kind=='soil':
                # Regular on soil (conductor disks are excluded). No thermal reconstruction.
                z=xy[:,:2]-xy.new_tensor(c['cables'][0]);rr=(z*z).sum(1,keepdim=True)
                real=c['radius']*z[:,:1]/rr;imag=c['radius']*z[:,1:2]/rr
            else:
                z=(xy[:,:2]-xy.new_tensor(c['cables'][region.cable]))/region.ro
                real=z[:,:1];imag=z[:,1:2] # positive powers, finite at r=0
            ar,ai=real,imag;columns=[]
            for _ in range(3):columns.extend([ar,ai]);ar,ai=ar*real-ai*imag,ar*imag+ai*real
            temperature=temperature+amplitude*(torch.cat(columns,dim=1)@self.harmonics[index].reshape(-1,1))
        if self.global_network or region.kind=='soil':temperature=temperature*self.envelope(xy)
        temperature=temperature+self.domain.boundary_temperature(xy)
        flux=raw[:,1:3]*scales['flux'] if self.mixed else None
        return temperature,flux

    def temperature(self,xy,labels):
        result=torch.empty((len(xy),1),dtype=xy.dtype,device=xy.device)
        for r in self.domain.regions:
            mask=labels==r.id
            if mask.any():result[mask]=self.field(r,xy[mask])[0]
        return result


class PhysicsLoss:
    """Every variant traverses every material. Source and transmission are shared."""
    def __init__(self,model,n=512,n_layer=256,n_interface=128,seed=11):
        self.model=model;self.domain=model.domain
        reference=next(model.parameters())
        self.tensor=lambda a: torch.as_tensor(a,dtype=reference.dtype,device=reference.device).clone().detach().requires_grad_(True)
        self.interior={r.id:self.tensor(self.domain.sample(r,n if r.kind=='soil' else n_layer,seed+7919*r.id,near=r.kind=='soil')) for r in self.domain.regions}
        self.quadrature={r.id:self.tensor(self.domain.sample(r,max(256,n_layer),800000+seed+r.id)) for r in self.domain.conductors}
        self.transmission=[]
        for interface in self.domain.interfaces:
            xy,normal,length=self.domain.interface_points(interface,n_interface)
            self.transmission.append((interface,self.tensor(xy),self.tensor(normal),length))
        self.outer=[]
        for xy,normal,length in self.domain.outer_boundary(n_interface):
            for r in self.domain.regions[:self.domain.nsoil]:
                mask=(xy[:,r.axis]>=r.lower)&(xy[:,r.axis]<=r.upper)
                if np.any(mask):self.outer.append((r,self.tensor(xy[mask]),self.tensor(normal[mask]),length*np.mean(mask)))
        self.static={r.id:self.domain.source(r,self.interior[r.id]).detach() for r in self.domain.regions}

    def electrical_fields(self):
        if self.domain.mode=='fixed':return {},{r.id:float(self.domain.source(r,self.quadrature[r.id]).mean().detach())*self.domain.area(r) for r in self.domain.conductors}
        fields={};powers={}
        for r in self.domain.conductors:
            temperature=self.model.field(r,self.quadrature[r.id])[0]
            denominator=1+self.domain.case['alpha']*(temperature-20.)
            if torch.any(denominator.detach()<=0):raise RuntimeError('Nonpositive local electrical resistivity')
            sigma=dc_conductivity(self.domain,r,temperature)
            conductance=self.domain.area(r)*sigma.mean()
            fields[r.id]=self.domain.case['current']/conductance
            powers[r.id]=fields[r.id]**2*conductance
        return fields,powers

    def __call__(self):
        reference=next(self.model.parameters());zero=reference.new_zeros(())
        fields,powers=self.electrical_fields();pde=zero;constitutive=zero;temperature_jump=zero;flux_jump=zero
        region_pde={};balance={r.id:zero for r in self.domain.regions}
        for r in self.domain.regions:
            xy=self.interior[r.id];T,q=self.model.field(r,xy)
            k=self.domain.conductivity(r,xy,torch);fixed=self.static[r.id]
            source=dc_conductivity(self.domain,r,T)*fields[r.id]**2 if r.id in fields else fixed
            scales=self.domain.scales(r)
            if self.model.mixed:
                residual,law=mixed_residual(T,q,xy,k,source)
                constitutive=constitutive+torch.mean((law/scales['flux'])**2)
            else:residual=heat_residual(T,xy,k,source)
            value=torch.mean((residual/scales['divergence'])**2);pde=pde+value;region_pde[r.name]=value
            if r.id in powers:balance[r.id]=balance[r.id]-powers[r.id]
        for interface,xy,normal,length in self.transmission:
            left=self.domain.regions[interface.left];right=self.domain.regions[interface.right]
            Tl,ql=self.model.field(left,xy);Tr,qr=self.model.field(right,xy)
            kl=self.domain.conductivity(left,xy,torch);kr=self.domain.conductivity(right,xy,torch)
            # Obtain exact one-sided soil coefficients at discrete flat interfaces.
            if not interface.radius:
                minus=xy.clone();plus=xy.clone();minus[:,interface.axis]-=1e-7;plus[:,interface.axis]+=1e-7
                kl=self.domain.conductivity(left,minus,torch);kr=self.domain.conductivity(right,plus,torch)
            if ql is None:ql=-kl*gradients(Tl,xy)[:,:2]
            if qr is None:qr=-kr*gradients(Tr,xy)[:,:2]
            qln=(ql*normal).sum(1,keepdim=True);qrn=(qr*normal).sum(1,keepdim=True)
            temperature_jump=temperature_jump+torch.mean(((Tl-Tr)/self.domain.case['scale'])**2)
            qs=max(self.domain.scales(left)['flux'],self.domain.scales(right)['flux'])
            flux_jump=flux_jump+torch.mean(((qln-qrn)/qs)**2)
            balance[left.id]=balance[left.id]+length*qln.mean()
            balance[right.id]=balance[right.id]-length*qrn.mean()
        for r,xy,normal,length in self.outer:
            T,q=self.model.field(r,xy)
            if q is None:q=-self.domain.conductivity(r,xy,torch)*gradients(T,xy)[:,:2]
            balance[r.id]=balance[r.id]+length*(q*normal).sum(1).mean()
        energy=sum((value/self.domain.case['power'])**2 for value in balance.values())
        parts=dict(pde=pde,constitutive=constitutive,interface_temperature=temperature_jump,interface_flux=flux_jump,energy=energy)
        # Same physics terms in every variant; mixed adds the constitutive equation.
        total=pde+constitutive+100*temperature_jump+10*flux_jump+10*energy
        return total,parts,region_pde


def evaluate(model,output,seed,reference_path=None):
    d=model.domain;ref=next(model.parameters())
    xy,labels,weights=d.evaluation_points()
    tensor=lambda a: torch.tensor(a,dtype=ref.dtype,device=ref.device,requires_grad=True)
    with torch.no_grad():T=model.temperature(tensor(xy),torch.as_tensor(labels,device=ref.device)).cpu().numpy().ravel()
    report=dict(physics_version=PHYSICS_VERSION,physics_sha256=d.fingerprint,seed=seed,
        variant=model.variant,mode=d.mode,regions=[],interfaces=[],accepted=False,
        acceptance_state='pending_independent_reference',conductor_max_C=[])
    for r in d.regions:
        select=labels==r.id
        report['regions'].append(dict(name=r.name,count=int(select.sum()),Tmin_C=float(T[select].min()),Tmax_C=float(T[select].max()),area_m2=d.area(r)))
        if r.kind=='conductor':report['conductor_max_C'].append(float(T[select].max()))
    report['Tmax_C']=max(report['conductor_max_C'])
    # Independent quadrature; compare predicted and temperature-derived flux in mixed models.
    balance_q={r.id:0. for r in d.regions};balance_T=dict(balance_q)
    audit_loss=PhysicsLoss(model,n=64,n_layer=1024,n_interface=256,seed=909090)
    _,powers=audit_loss.electrical_fields()
    ptotal=sum(float(p.detach()) if torch.is_tensor(p) else p for p in powers.values())
    for r in d.conductors:
        p=float(powers[r.id].detach()) if torch.is_tensor(powers[r.id]) else powers[r.id]
        balance_q[r.id]-=p;balance_T[r.id]-=p
    for inter,points,normal,length in audit_loss.transmission:
        traces=[]
        for regionid,side in [(inter.left,-1),(inter.right,1)]:
            r=d.regions[regionid];temp,q=model.field(r,points)
            probe=points.clone()
            if not inter.radius:probe[:,inter.axis]+=side*1e-7
            derived=-d.conductivity(r,probe,torch)*gradients(temp,points)[:,:2]
            q=derived if q is None else q
            traces.append((temp.detach(),(q*normal).sum(1).detach(),(derived*normal).sum(1).detach()))
        jump=(traces[0][0]-traces[1][0]).cpu().numpy().ravel()
        qjump=(traces[0][1]-traces[1][1]).cpu().numpy()
        tjump=(traces[0][2]-traces[1][2]).cpu().numpy()
        qs=max(d.scales(d.regions[inter.left])['flux'],d.scales(d.regions[inter.right])['flux'])
        report['interfaces'].append(dict(name=inter.name,T_jump_max_K=float(abs(jump).max()),
            T_jump_rms_K=float(np.sqrt(np.mean(jump**2))),flux_jump_rms_pct=float(100*np.sqrt(np.mean(qjump**2))/qs),
            derived_flux_jump_rms_pct=float(100*np.sqrt(np.mean(tjump**2))/qs)))
        for rid,sign,j in [(inter.left,1,0),(inter.right,-1,1)]:
            balance_q[rid]+=sign*length*float(traces[j][1].mean());balance_T[rid]+=sign*length*float(traces[j][2].mean())
    for r,points,normal,length in audit_loss.outer:
        temp,q=model.field(r,points);derived=-d.conductivity(r,points,torch)*gradients(temp,points)[:,:2]
        q=derived if q is None else q
        balance_q[r.id]+=length*float((q*normal).sum(1).mean().detach())
        balance_T[r.id]+=length*float((derived*normal).sum(1).mean().detach())
    report['region_energy_error_pct']={r.name:100*abs(balance_q[r.id])/d.case['power'] for r in d.regions}
    report['region_derived_energy_error_pct']={r.name:100*abs(balance_T[r.id])/d.case['power'] for r in d.regions}
    report['balance_pct']=100*abs(sum(balance_q.values()))/ptotal
    report['derived_balance_pct']=100*abs(sum(balance_T.values()))/ptotal
    report['total_power_W_m']=ptotal
    for region in report['regions']:
        region['energy_error_pct']=report['region_energy_error_pct'][region['name']]
    expected=None;reference_max=None
    if reference_path:
        path=Path(reference_path);meta=json.loads(path.with_suffix('.json').read_text())
        if meta['physics_sha256']!=d.fingerprint:raise ValueError('Reference physics differs (reduced references are forbidden)')
        with np.load(path) as data:
            if not np.array_equal(labels,data['region']) or not np.allclose(xy,data['xy'],rtol=0,atol=1e-14):raise ValueError('Reference quadrature differs')
            expected=data['T'].copy()
        reference_max=meta['Tmax_C'];report['reference']=str(path)
        report['reference_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    elif d.case.get('outer_radius') and not d.case.get('outer_gradient_K_m',0) and d.mode=='fixed' and not d.case.get('electrical'):
        expected=radial_exact(d,np.linalg.norm(xy,axis=1));reference_max=float(radial_exact(d,np.array([0.]))[0])
        report['reference']='Analytical Poisson/Laplace solution; evaluation only'
    if expected is not None:
        error=T-expected;delta=abs(reference_max-d.case['T0'])
        report.update(rmse_K=float(np.sqrt(np.average(error**2,weights=weights))),
            error_Tmax_K=float(abs(report['Tmax_C']-reference_max)),max_error_K=float(abs(error).max()))
        report['nrmse_pct']=100*report['rmse_K']/delta;report['error_Tmax_pct']=100*report['error_Tmax_K']/delta
        report['region_rmse_K']={r.name:float(np.sqrt(np.mean(error[labels==r.id]**2))) for r in d.regions}
        report['accepted']=bool(report['nrmse_pct']<=5 and report['error_Tmax_pct']<=5 and
            max(report['region_rmse_K'].values())<=.05*delta and report['balance_pct']<=2 and report['derived_balance_pct']<=2 and
            max(report['region_energy_error_pct'].values())<=2 and max(report['region_derived_energy_error_pct'].values())<=2 and
            max(i['T_jump_max_K'] for i in report['interfaces'])<=.1 and
            max(i['flux_jump_rms_pct'] for i in report['interfaces'])<=1 and
            max(i['derived_flux_jump_rms_pct'] for i in report['interfaces'])<=1)
        report['acceptance_state']='accepted' if report['accepted'] else 'rejected'
    report['acceptance_criteria']=dict(nrmse_pct=5,Tmax_rise_pct=5,max_region_rmse_rise_pct=5,
        global_and_region_energy_pct=2,interface_T_max_K=.1,interface_flux_rms_pct=1,
        mixed_flux='both predicted and -k*grad(T) must pass')
    report['completed_utc']=datetime.now(timezone.utc).isoformat()
    np.savez_compressed(output/f'pinn_seed{seed}.npz',xy=xy,region=labels,weights=weights,T=T)
    (output/f'pinn_seed{seed}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def train(domain,config,output):
    seed=config.get('seed',11);torch.manual_seed(seed)
    torch.set_num_threads(config.get('threads',1));torch.set_default_dtype(torch.float64)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/f'pinn_seed{seed}.json').exists():raise FileExistsError('Use a new run directory; existing evidence will not be overwritten')
    model=FullPINN(domain,config.get('variant','mixed'),config.get('width',32),config.get('depth',3))
    objective=PhysicsLoss(model,config.get('n',512),config.get('n_layer',256),config.get('n_interface',128),seed)
    config=dict(config,physics=domain.specification(),physics_sha256=domain.fingerprint)
    (output/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    source_files=[Path(__file__),Path(__file__).with_name('full_domain.py'),Path(__file__).with_name('full_physics.py'),Path(__file__).with_name('skin_effect.py'),Path(__file__).with_name('cases.py'),Path(__file__).with_name('expressions.py'),ROOT/'pinn_cables/pinn/pde.py']
    archive=output/'source';archive.mkdir(exist_ok=True);hashes={}
    for path in source_files:
        content=path.read_bytes();digest=hashlib.sha256(content).hexdigest();hashes[str(path.relative_to(ROOT))]=digest
        (archive/(digest+'_'+path.name)).write_bytes(content)
    clouds={r.name:objective.interior[r.id].detach().cpu().numpy() for r in domain.regions}
    np.savez_compressed(output/f'collocation_seed{seed}.npz',**clouds)
    history=[];start=time.perf_counter();optimizer=torch.optim.Adam(model.parameters(),lr=config.get('lr',.001))
    for step in range(config.get('adam',1500)):
        optimizer.zero_grad(set_to_none=True)
        loss,parts,regions=objective()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite full-domain loss')
        loss.backward();optimizer.step()
        if step%200==0:
            row=dict(phase='Adam',step=step,total=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()})
            history.append(row);print(json.dumps(row),flush=True)
    adam_state=optimizer.state_dict();calls=0
    optimizer=torch.optim.LBFGS(model.parameters(),max_iter=config.get('lbfgs',1500),max_eval=2*config.get('lbfgs',1500),
        line_search_fn='strong_wolfe',history_size=50,tolerance_grad=1e-10,tolerance_change=1e-12)
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True);loss,parts,regions=objective()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite full-domain loss')
        loss.backward();calls+=1
        if calls%250==0:
            row=dict(phase='LBFGS',step=calls,total=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()})
            history.append(row);print(json.dumps(row),flush=True)
        return loss
    optimizer.step(closure)
    loss,parts,regions=objective()
    torch.save(dict(state_dict=model.state_dict(),adam_state=adam_state,optimizer_state=optimizer.state_dict(),
        phase='completed',rng_state=torch.get_rng_state(),configuration=config),output/f'pinn_seed{seed}.pt')
    meta=dict(configuration=config,source_sha256=hashes,torch=torch.__version__,python=sys.version,platform=platform.platform(),
        device='cpu',dtype='float64',parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.perf_counter()-start,
        lbfgs_evaluations=calls,history=history,final_parts={k:float(v.detach()) for k,v in parts.items()},
        final_region_pde={k:float(v.detach()) for k,v in regions.items()},fem_labels_used=0)
    (output/f'training_seed{seed}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    report=evaluate(model,output,seed,config.get('reference'))
    print(json.dumps(dict(case=domain.case['id'],variant=model.variant,Tmax_C=report['Tmax_C'],accepted=report['accepted'],seconds=meta['training_seconds'])),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',default='coaxial_full');ap.add_argument('--case-file',type=Path)
    ap.add_argument('--variant',choices=VARIANTS,default='mixed');ap.add_argument('--mode',choices=['fixed','dc_temperature'],default='fixed')
    ap.add_argument('--seed',type=int,default=11);ap.add_argument('--width',type=int,default=32);ap.add_argument('--depth',type=int,default=3)
    ap.add_argument('--n',type=int,default=512);ap.add_argument('--n-layer',type=int,default=256);ap.add_argument('--n-interface',type=int,default=128)
    ap.add_argument('--adam',type=int,default=1500);ap.add_argument('--lbfgs',type=int,default=1500);ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--reference');ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();c=json.loads(a.case_file.read_text()) if a.case_file else (coaxial_case() if a.case=='coaxial_full' else cases()[a.case])
    config={k:v for k,v in vars(a).items() if k not in ('case_file','output')}
    train(FullDomain(c,a.mode),config,a.output)


if __name__=='__main__':main()
