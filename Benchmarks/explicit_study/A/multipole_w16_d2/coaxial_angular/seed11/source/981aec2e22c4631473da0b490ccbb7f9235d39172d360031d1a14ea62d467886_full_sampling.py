"""Stratified sampling policies; adaptive selection never receives FEM values."""
import numpy as np
import torch
from Benchmarks.full_physics import heat_residual,mixed_residual
from Benchmarks.full_domain import dc_conductivity

POLICIES=('uniform','mixed','interface','residual')

def sample(domain,region,n,seed,policy):
    if policy not in POLICIES:raise ValueError('Unknown sampling policy')
    if policy=='uniform':return domain.sample(region,n,seed)
    if policy in ('mixed','residual'):return domain.sample(region,n,seed,near=region.kind=='soil')
    # A uniform half prevents narrow interface bands from erasing bulk coverage.
    count=n//2;bulk=domain.sample(region,n-count,seed)
    if not count:return bulk
    rng=np.random.default_rng(seed+104729);c=domain.case;cloud=[];size=0
    for _ in range(1000):
        m=max(256,n);angle=rng.uniform(0,2*np.pi,m)
        if region.kind!='soil':
            width=region.ro-region.ri
            radius=region.ro-width*rng.uniform(1e-6,.1,m)
            if region.ri:
                mask=rng.random(m)<.5;radius[mask]=region.ri+width*rng.uniform(1e-6,.1,mask.sum())
            candidates=np.c_[radius*np.cos(angle),radius*np.sin(angle)]+c['cables'][region.cable]
        else:
            candidates=domain.sample(region,m,seed+50000+size)
            center=np.asarray(c['cables'][int(rng.integers(len(c['cables'])))])
            radius=c['radius']+rng.uniform(1e-6,max(c['radius'],.01),m)
            candidates[:m//2]=center+np.c_[radius*np.cos(angle),radius*np.sin(angle)][:m//2]
            interfaces=[i for i in domain.interfaces if not i.radius and region.id in (i.left,i.right)]
            if interfaces:
                i=interfaces[int(rng.integers(len(interfaces)))];sign=-1 if region.id==i.left else 1
                candidates[m//2:,i.axis]=i.position+sign*rng.uniform(1e-5,.05,m-m//2)
        accepted=candidates[domain.contains(region,candidates)];cloud.append(accepted);size+=len(accepted)
        if size>=count:return np.vstack([bulk,np.vstack(cloud)[:count]])
    raise RuntimeError('Unable to fill interface stratum')

def adapt(objective,seed,multiplier=4):
    fields,_=objective.electrical_fields();diagnostics=[]
    for region in objective.domain.regions:
        n=len(objective.interior[region.id]);candidates=sample(objective.domain,region,n*multiplier,seed+7919*region.id,'interface')
        xy=objective.tensor(candidates);T,q=objective.model.field(region,xy)
        k=objective.domain.conductivity(region,xy,torch)
        Q=dc_conductivity(objective.domain,region,T)*fields[region.id]**2 if region.id in fields else objective.domain.source(region,xy)
        scales=objective.domain.scales(region)
        if q is None:score=(heat_residual(T,xy,k,Q)/scales['divergence']).abs().detach().cpu().numpy().ravel()
        else:
            residual,law=mixed_residual(T,q,xy,k,Q)
            score=((residual/scales['divergence']).abs()+torch.linalg.vector_norm(law/scales['flux'],dim=1,keepdim=True)).detach().cpu().numpy().ravel()
        if not np.isfinite(score).all():raise RuntimeError('Nonfinite adaptation indicator')
        probability=score+max(float(score.mean()),1e-12)*.1;probability/=probability.sum()
        rng=np.random.default_rng(seed+region.id);chosen=rng.choice(len(candidates),n//2,replace=False,p=probability)
        bulk=sample(objective.domain,region,n-len(chosen),seed+region.id,'uniform')
        points=np.vstack([bulk,candidates[chosen]])
        objective.interior[region.id]=objective.tensor(points)
        objective.static[region.id]=objective.domain.source(region,objective.interior[region.id]).detach()
        diagnostics.append(dict(region=region.name,candidates=len(candidates),retained=n,score_median=float(np.median(score)),score_max=float(score.max())))
    return diagnostics
