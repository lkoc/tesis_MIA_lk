"""Explicit DC current-limit problem; interior heat equation in FEM and PINN."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,math,platform,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_domain import FullDomain,dc_conductivity
from Benchmarks.cases import cases,field_k


def archive_sources(output):
    archive=Path(output)/'source';archive.mkdir(parents=True,exist_ok=True);source={}
    paths=[Path(__file__).with_name(name) for name in ['full_ampacity.py','full_pinn.py','full_domain.py','full_physics.py','full_sampling.py','skin_effect.py','cases.py','expressions.py']]+[ROOT/'pinn_cables/pinn/pde.py']
    for path in paths:
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();source[path.name]=digest;(archive/(digest+'_'+path.name)).write_bytes(data)
    return source


class AmpacityDomain(FullDomain):
    def __init__(self,case,limit=90.):
        self.problem_case=copy.deepcopy(case);self.limit=limit
        if limit<=case['T0']:raise ValueError('Temperature limit must exceed ambient')
        super().__init__(case,'dc_temperature')
    def specification(self):
        spec=super().specification();spec['case']=self.problem_case
        spec.update(problem='explicit DC ampacity: current unknown',temperature_limit_C=self.limit,
            current_in_case='reference current for initialization and numerical scaling, not prescribed current')
        return spec

def ampacity_cases():
    catalog=cases();c=copy.deepcopy(catalog['xlpe_dry_near']);x0,x1,y0,y1=c['bounds'];area=(x1-x0)*(y1-y0)
    cx,cy,width,height,kp,eps=c['patch']
    lc=lambda v:np.logaddexp(v,-v)-math.log(2)
    def integral(a,b,center,extent):
        return eps/2*(lc((b-center+extent/2)/eps)-lc((a-center+extent/2)/eps)-lc((b-center-extent/2)/eps)+lc((a-center-extent/2)/eps))
    total=c['k']*area+(kp-c['k'])*integral(x0,x1,cx,width)*integral(y0,y1,cy,height)
    z,w=np.polynomial.legendre.leggauss(32);theta=np.arange(128)*2*np.pi/128
    for center in c['cables']:
        radius=c['radius']*np.sqrt((z+1)/2)
        xy=np.stack([center[0]+radius[:,None]*np.cos(theta),center[1]+radius[:,None]*np.sin(theta)],axis=-1)
        kval=field_k(c,xy[:,:,0],xy[:,:,1],np)
        total-=math.pi*c['radius']**2*np.sum(w*kval.mean(axis=1))/2;area-=math.pi*c['radius']**2
    c.update(id='xlpe_dry_near_hom_arithmetic',k=float(total/area),patch=None,pair='xlpe_dry_near',adaptation='Arithmetic spatial mean over soil only; exact rectangle window integral and Gauss disk exclusion')
    return {name:catalog[name] for name in ['xlpe_single','xlpe_dry_near']}|{c['id']:c}

def fem_root(case,level,output,limit=90.):
    from Benchmarks.full_fem import solve
    output=Path(output);output.mkdir(parents=True,exist_ok=True);problem=AmpacityDomain(case,limit);trace=[]
    final=output/f'fem_ampacity_l{level}.json'
    if final.exists():return json.loads(final.read_text())
    root_sources=archive_sources(output)
    def temperature(current):
        c=copy.deepcopy(case);c['current']=float(current);c['power']=current**2*c['R20']
        folder=output/f'l{level}_trials'/f'trial{len(trace):03d}';path=folder/f'fem_l{level}.json'
        record=json.loads(path.read_text()) if path.exists() else solve(FullDomain(c,'dc_temperature'),level,folder,save_fields=False)
        if record['specification']['case']['current']!=float(current):raise ValueError('Existing root trace belongs to another trial')
        trace.append(dict(current_A=current,Tmax_C=record['Tmax_C'],balance_pct=record['balance_pct']))
        return record['Tmax_C']-limit
    low=case['current']*.5;high=case['current'];fl=temperature(low);fh=temperature(high)
    for _ in range(30):
        if fl<=0<=fh:break
        if fl>0:high=low;fh=fl;low*=.8;fl=temperature(low)
        else:low=high;fl=fh;high*=1.15;fh=temperature(high)
    else:raise RuntimeError('Unable to bracket an explicit current-limit solution')
    for _ in range(40):
        current=(low+high)/2;value=temperature(current)
        if abs(value)<=.01 and (high-low)/current<=.0005:break
        if value>0:high=current
        else:low=current
    else:raise RuntimeError('Ampacity root search failed')
    c=copy.deepcopy(case);c['current']=current;c['power']=current**2*c['R20']
    folder=output/f'l{level}_solution';record=solve(FullDomain(c,'dc_temperature'),level,folder)
    record.update(physics_sha256=problem.fingerprint,specification=problem.specification(),solution_current_A=current,
        solved_state=c,root_trace=trace,root_bracket_A=[low,high],root_temperature_tolerance_K=.01,root_relative_bracket=.0005,
        method='explicit_multimaterial_FEM_with_DC_current_root',root_source_sha256=root_sources)
    for extension in ['.npz']:(output/f'fem_ampacity_l{level}{extension}').write_bytes((folder/f'fem_l{level}{extension}').read_bytes())
    final.write_text(json.dumps(record,indent=2),encoding='utf-8');return record

def train_ampacity(case,config,reference,output,limit=90.):
    import torch
    from torch import nn
    from Benchmarks.full_pinn import FullPINN,PhysicsLoss,evaluate
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);seed=config['seed'];torch.manual_seed(seed)
    d=AmpacityDomain(case,limit);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/f'pinn_seed{seed}.json').exists():raise FileExistsError('Existing ampacity evidence')
    source=archive_sources(output)
    class Model(FullPINN):
        def __init__(self):
            super().__init__(d,config['variant'],config['width'],config['depth']);self.log_current=nn.Parameter(torch.zeros(()))
        def current(self):return self.domain.problem_case['current']*torch.exp(self.log_current)
    class Loss(PhysicsLoss):
        def electrical_fields(self):
            fields={};powers={}
            for r in d.conductors:
                T=self.model.field(r,self.quadrature[r.id])[0];conductance=d.area(r)*dc_conductivity(d,r,T).mean()
                fields[r.id]=self.model.current()/conductance;powers[r.id]=fields[r.id]**2*conductance
            return fields,powers
        def __call__(self):
            value,parts,regions=super().__call__()
            temperatures=[]
            for r in d.conductors:
                points=torch.cat([self.quadrature[r.id],self.tensor([d.case['cables'][r.cable]])])
                temperatures.append(self.model.field(r,points)[0])
            term=((torch.cat(temperatures).max()-limit)/d.case['scale'])**2
            return value+100*term,dict(parts,temperature_limit=term),regions
    model=Model();objective=Loss(model,config['n'],config['n_layer'],config['n_interface'],config['sampling_seed'],config['sampling'],config.get('temperature_weight',100))
    (output/'configuration.json').write_text(json.dumps(dict(config,physics=d.specification(),reference=str(reference)),indent=2),encoding='utf-8')
    np.savez_compressed(output/f'collocation_seed{seed}_initial.npz',**{r.name:objective.interior[r.id].detach().numpy() for r in d.regions})
    start=time.perf_counter();history=[];optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    from Benchmarks.full_sampling import adapt
    for step in range(config['adam']):
        if config['sampling']=='residual' and step==config['adam']//2:
            diagnostics=adapt(objective,config['sampling_seed']+101*step)
            (output/'adaptation.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
        optimizer.zero_grad(set_to_none=True);loss,parts,_=objective()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite ampacity loss')
        loss.backward();optimizer.step()
        if step%300==0:history.append(dict(step=step,loss=float(loss.detach()),current_A=float(model.current().detach())));print(json.dumps(history[-1]),flush=True)
    adam_state=optimizer.state_dict()
    optimizer=torch.optim.LBFGS(model.parameters(),max_iter=config['lbfgs'],max_eval=2*config['lbfgs'],history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
    calls=0
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True);loss,_,_=objective()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite ampacity loss')
        loss.backward();calls+=1
        if calls%500==0:print(json.dumps(dict(phase='LBFGS',evaluations=calls,loss=float(loss.detach()),current_A=float(model.current().detach()))),flush=True)
        return loss
    optimizer.step(closure);current=float(model.current().detach())
    _,parts,regions=objective();training_seconds=time.perf_counter()-start
    # Set the solved current for the independent audit; keep P0 only as a
    # numerical scale so changing the state does not rescale the learned field.
    d.case['current']=current
    report=evaluate(model,output,seed,reference);ref=json.loads(Path(reference).with_suffix('.json').read_text())
    report.update(solution_current_A=current,current_error_pct=100*abs(current/ref['solution_current_A']-1),temperature_limit_error_K=abs(report['Tmax_C']-limit))
    report['accepted']=bool(report['accepted'] and report['current_error_pct']<=2 and report['temperature_limit_error_K']<=.1)
    report['acceptance_state']='accepted' if report['accepted'] else 'rejected'
    report['acceptance_criteria'].update(current_error_pct=2,temperature_limit_error_K=.1)
    (output/f'pinn_seed{seed}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    torch.save(dict(state_dict=model.state_dict(),adam_state=adam_state,optimizer_state=optimizer.state_dict(),rng_state=torch.get_rng_state(),configuration=config,problem=d.specification(),current_A=current),output/f'pinn_seed{seed}.pt')
    np.savez_compressed(output/f'collocation_seed{seed}.npz',**{r.name:objective.interior[r.id].detach().numpy() for r in d.regions})
    (output/f'training_seed{seed}.json').write_text(json.dumps(dict(configuration=config,source_sha256=source,history=history,training_seconds=training_seconds,postprocessing_seconds=time.perf_counter()-start-training_seconds,lbfgs_evaluations=calls,final_parts={k:float(v.detach()) for k,v in parts.items()},final_region_pde={k:float(v.detach()) for k,v in regions.items()},parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),fem_labels_used=0,python=sys.version,torch=torch.__version__,platform=platform.platform(),device='cpu',dtype='float64',threads=1),indent=2),encoding='utf-8')
    print(json.dumps(dict(current_A=current,accepted=report['accepted'])),flush=True);return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fem',action='store_true');parser.add_argument('--case',default='xlpe_single');parser.add_argument('--case-file',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path);parser.add_argument('--reference');parser.add_argument('--seed',type=int,default=71);parser.add_argument('--limit',type=float,default=90.);parser.add_argument('--levels',type=int,nargs='+',default=[0,1,2])
    args=parser.parse_args();case=json.loads(args.case_file.read_text(encoding='utf-8')) if args.case_file else ampacity_cases()[args.case]
    if args.fem:
        levels=sorted(set(args.levels))
        if len(levels)<3:parser.error('Current-limit refinement requires at least three distinct levels')
        records=[fem_root(case,level,args.output,args.limit) for level in levels]
        change=100*abs(records[-1]['solution_current_A']/records[-2]['solution_current_A']-1)
        gate=dict(passed=change<=.5 and records[-1]['balance_pct']<=.2,current_mesh_change_pct=change,levels=[dict(level=r['level'],current_A=r['solution_current_A'],Tmax_C=r['Tmax_C'],balance_pct=r['balance_pct']) for r in records])
        (args.output/'gate.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
        if not gate['passed']:raise RuntimeError('Explicit ampacity mesh gate failed')
    else:
        if args.config is None or args.reference is None:parser.error('PINN current-limit training requires --config and --reference')
        selected=json.loads(args.config.read_text());config=dict(selected.get('configuration',selected),seed=args.seed,sampling_seed={71:1001,83:1003,97:1007}.get(args.seed,args.seed))
        train_ampacity(case,config,args.reference,args.output,args.limit)
