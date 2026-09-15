"""PINN de colocación sin etiquetas FEM: MLP, descomposición y fondo analítico.

Los datos FEM se leen únicamente después del entrenamiento para evaluar errores.
"""
from pathlib import Path
import argparse
import json
import math
import platform
import sys
import time
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pinn_cables.pinn.pde import gradients, laplace_variable_k
from Benchmarks.cases import cases, field_k, source, exact, evaluation_points, surface_points, radial_resistance, fingerprint, in_domain
SOURCE_ROOT=Path(__file__).resolve().parents[1]
SOURCE_SNAPSHOT={str(p.relative_to(SOURCE_ROOT)):p.read_bytes() for p in [Path(__file__),Path(__file__).with_name('cases.py'),SOURCE_ROOT/'pinn_cables/pinn/pde.py']}

class Network(nn.Module):
    def __init__(self,c,width=32,depth=3,variant='enriched'):
        super().__init__();self.c=c;self.width=width;self.depth=depth;self.variant=variant
        count=2 if c['kind']=='mms_interface' and variant!='global' else 1
        self.nets=nn.ModuleList()
        if variant=='multipole' and c['kind']=='cable':
            self.multipoles=nn.Parameter(torch.zeros(len(c['cables']),3,2))
        for _ in range(count):
            layers=[];nin=1 if c['kind']=='annulus' else 2
            for _ in range(depth):
                lin=nn.Linear(nin,width);nn.init.xavier_normal_(lin.weight);nn.init.zeros_(lin.bias)
                layers.extend([lin,nn.Tanh()]);nin=width
            lin=nn.Linear(width,1);nn.init.zeros_(lin.bias);layers.append(lin)
            self.nets.append(nn.Sequential(*layers))
    def forward(self,xy):
        c=self.c
        if c['kind']=='annulus':
            z=torch.log(torch.linalg.vector_norm(xy,dim=1,keepdim=True)/c['ri'])/math.log(c['ro']/c['ri'])
        elif c['kind']=='cable':
            # Physical local scale, with tanh resolving the cable vicinity.
            z=torch.cat([xy[:,:1]/2.,(xy[:,1:2]+1.4)/2.],dim=1)
        else: z=2*xy-1
        if len(self.nets)==2:
            return torch.where(xy[:,:1]<=.5,self.nets[0](z),self.nets[1](z))
        result=self.nets[0](z)
        if hasattr(self,'multipoles'):
            for j,(cx,cy) in enumerate(c['cables']):
                dx=xy[:,:1]-cx;dy=xy[:,1:2]-cy;r2=dx*dx+dy*dy
                ar=c['radius']*dx/r2;ai=c['radius']*dy/r2
                real=ar;imag=ai
                for order in range(3):
                    result=result+self.multipoles[j,order,0]*real+self.multipoles[j,order,1]*imag
                    real,imag=real*ar-imag*ai,real*ai+imag*ar
        return result

def background(c,xy,variant='enriched'):
    out=xy[:,:1]*0+c['T0']
    if c['kind']=='cable' and variant!='direct':
        # Homogeneous half-space image solution used only as an enrichment.
        for cx,cy in c['cables']:
            r2=(xy[:,:1]-cx)**2+(xy[:,1:2]-cy)**2
            ri2=(xy[:,:1]-cx)**2+(xy[:,1:2]+cy)**2
            kb=float(field_k(c,np.array(cx),np.array(cy))) if variant in ('local','conservative','multipole') else c['k']
            out=out+c['power']/(4*math.pi*kb)*torch.log(ri2/r2)
    return out

def predict(model,xy): return background(model.c,xy,model.variant)+model.c['scale']*model(xy)

def tensor(a,grad=False): return torch.tensor(np.asarray(a),requires_grad=grad)

def training_points(c,seed,n):
    rng=np.random.default_rng(seed)
    if c['kind']=='annulus':
        r=c['ri']*(c['ro']/c['ri'])**rng.uniform(0,1,n)
        return np.c_[r,np.zeros(n)]
    x0,x1,y0,y1=c['bounds']
    xy=rng.uniform([x0,y0],[x1,y1],(n*2,2));xy=xy[in_domain(c,xy)][:n]
    if c['kind']=='cable':
        near=[]
        for cx,cy in c['cables']:
            a=rng.uniform(0,2*math.pi,max(128,n//len(c['cables'])))
            r=c['radius']*np.exp(rng.uniform(.005,math.log(1.2/c['radius']),len(a)))
            near.append(np.c_[r*np.cos(a)+cx,r*np.sin(a)+cy])
        near=np.vstack(near);xy=np.vstack([xy,near[in_domain(c,near)]])
        if c.get('patch') or c.get('bands'):
            near=rng.uniform([-1.4,-2.4],[1.4,-.15],(n,2));xy=np.vstack([xy,near[in_domain(c,near)]])
    return xy

def boundary_points(c,n=80):
    if c['kind']=='annulus':
        return [(np.array([[c['ro'],0.]]),np.array([[1.,0.]]),'outer'),(np.array([[c['ri'],0.]]),np.array([[-1.,0.]]),'inner')]
    x0,x1,y0,y1=c['bounds'];s=(np.arange(n)+.5)/n
    arr=[(np.c_[x0+(x1-x0)*s,0*s+y0],np.tile([0.,-1.],(n,1)),'bottom'),(np.c_[x0+(x1-x0)*s,0*s+y1],np.tile([0.,1.],(n,1)),'top'),(np.c_[0*s+x0,y0+(y1-y0)*s],np.tile([-1.,0.],(n,1)),'left'),(np.c_[0*s+x1,y0+(y1-y0)*s],np.tile([1.,0.],(n,1)),'right')]
    for j,pts in enumerate(surface_points(c,n,offset=0)):
        normal=-(pts-c['cables'][j])/c['radius'];arr.append((pts,normal,f'cable{j}'))
    return arr

def train(c,seed,args,out):
    torch.manual_seed(seed);start=time.perf_counter()
    model=Network(c,args.width,args.depth,args.variant)
    xy=tensor(training_points(c,seed,args.n),True)
    k=field_k(c,xy[:,:1],xy[:,1:2],torch)
    gk=gradients(k,xy).detach() if k.requires_grad else torch.zeros_like(xy)
    bg=background(c,xy,args.variant)
    rbg=laplace_variable_k(bg,xy,k).detach() if c['kind']=='cable' and args.variant!='direct' else torch.zeros_like(k)
    q=source(c,xy[:,:1],xy[:,1:2],torch).detach()
    kd=k.detach();scale=c['scale']
    cached=[]
    for pts,normals,edge in boundary_points(c):
        z=tensor(pts,True);normal=tensor(normals);bgz=background(c,z,args.variant)
        gz=gradients(bgz,z).detach()
        kz=field_k(c,z[:,:1],z[:,1:2],torch).detach()
        target=exact(c,z[:,:1],z[:,1:2],torch).detach() if c['kind'].startswith('mms') else z[:,:1].detach()*0+c['T0']
        cached.append((z,normal,edge,bgz.detach(),gz,kz,target))
    if c['kind']=='mms_interface':
        yi=(np.arange(128)+.5)/128
        inter=tensor(np.c_[np.full(128,.5),yi],True)
    def loss():
        u=model(xy)
        if c['kind']=='annulus':
            # In s=ln(r/ri)/ln(ro/ri), radial Laplace is exactly u_ss=0.
            s=torch.linspace(0,1,args.n).reshape(-1,1).requires_grad_(True)
            us=model.nets[0](s);du=gradients(us,s);ddu=gradients(du,s)
            lp=torch.mean(ddu**2)
        else:
            gu=gradients(u,xy)
            lu=gradients(gu[:,:1],xy)[:,:1]+gradients(gu[:,1:2],xy)[:,1:2]
            residual=(kd*lu+(gk*gu).sum(1,keepdim=True))*scale+rbg+q
            lp=torch.mean((residual/(kd*scale))**2)
        lb=u.new_zeros(());lf=u.new_zeros(());net_flux=u.new_zeros(())
        for z,normal,edge,bz,gz,kz,target in cached:
            uz=model(z);temp=bz+scale*uz
            if args.variant in ('conservative','multipole') and c['kind']=='cable':
                qout=-kz*((gz+scale*gradients(uz,z))*normal).sum(1,keepdim=True)
                x0,x1,y0,y1=c['bounds']
                length=2*math.pi*c['radius'] if edge.startswith('cable') else (x1-x0 if edge in ('top','bottom') else y1-y0)
                net_flux=net_flux+length*qout.mean()
            if edge.startswith('cable') or edge=='inner':
                gu=gradients(uz,z);qn=-kz*((gz+scale*gu)*normal).sum(1,keepdim=True)
                expected=-c['power']/(2*math.pi*c.get('radius',c.get('ri')))
                lf=lf+torch.mean(((qn-expected)/abs(expected))**2)
            elif c['kind']=='mms_robin' and edge=='top':
                gu=gradients(uz,z);tinf=target+3*z[:,:1]*(1-z[:,:1])
                r=-kz*(gu*normal).sum(1,keepdim=True)*scale-10*(temp-tinf)
                lb=lb+torch.mean((r/(10*scale))**2)
            else: lb=lb+torch.mean(((temp-target)/scale)**2)
        li=u.new_zeros(())
        if c['kind']=='mms_interface':
            z=2*inter-1;ul=model.nets[0](z);ur=model.nets[-1](z)
            gl=gradients(ul,inter)[:,:1];gr=gradients(ur,inter)[:,:1]
            li=torch.mean((ul-ur)**2)+torch.mean((.5*gl-2*gr)**2)
        le=(net_flux/(c.get('power',1)*len(c.get('cables',[0]))))**2
        wp=args.pde_weight if args.pde_weight is not None else (25 if args.variant in ('conservative','multipole') else 1)
        return wp*lp+args.bc_weight*lb+args.flux_weight*lf+10*li+args.energy_weight*le,dict(pde=lp,bc=lb,flux=lf,interface=li,energy=le)
    history=[];opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    for step in range(args.adam):
        opt.zero_grad(set_to_none=True);xy.grad=None
        total,parts=loss()
        if not torch.isfinite(total): raise RuntimeError('Nonfinite PINN loss')
        total.backward();opt.step()
        if step%200==0:
            history.append(dict(phase='Adam',step=step,loss=float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()}))
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=args.lbfgs,max_eval=args.lbfgs*2,history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
    calls=0
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True);xy.grad=None;total,parts=loss();total.backward();calls+=1
        if calls%100==0: history.append(dict(phase='LBFGS',step=calls,loss=float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()}))
        return total
    opt.step(closure)
    total,parts=loss()
    meta=dict(case=c,case_sha256=fingerprint(c),seed=seed,torch=torch.__version__,python=sys.version,platform=platform.platform(),device='cpu',dtype='float64',threads=torch.get_num_threads(),width=args.width,depth=args.depth,parameters=sum(p.numel() for p in model.parameters()),adam=args.adam,lbfgs_max_iter=args.lbfgs,lbfgs_closure_calls=calls,n_interior=len(xy),training_seconds=time.perf_counter()-start,final_loss=float(total.detach()),training_parts={k:float(v.detach()) for k,v in parts.items()},fem_labels_used=0)
    meta['variant']=args.variant
    meta['configuration']={key:value for key,value in vars(args).items() if key not in ('config','output')}
    import hashlib
    meta['source_sha256']={p:hashlib.sha256(data).hexdigest() for p,data in SOURCE_SNAPSHOT.items()}
    out.mkdir(parents=True,exist_ok=True)
    archive=out/'source'
    for name,data in SOURCE_SNAPSHOT.items():
        path=archive/(hashlib.sha256(data).hexdigest()+'_'+Path(name).name)
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists(): path.write_bytes(data)
    torch.save({'state_dict':model.state_dict(),'metadata':meta},out/f'pinn_seed{seed}.pt')
    (out/f'training_seed{seed}.json').write_text(json.dumps({'metadata':meta,'history':history},indent=2),encoding='utf-8')
    evaluate_model(model,meta,out)
    return model

def evaluate_model(model,meta,out):
    c=model.c;xy=evaluation_points(c)
    with torch.no_grad(): vals=predict(model,tensor(xy)).numpy().ravel()
    surface=surface_points(c,offset=max(1e-5,c.get('radius',.06)*.002))
    with torch.no_grad(): sv=[predict(model,tensor(s)).numpy().ravel() for s in surface]
    tc=[float(t.mean()+c['power']*radial_resistance(c)) for t in sv] if c['kind']=='cable' else []
    tm=max(tc) if tc else (float(np.max(sv)) if sv else float(np.max(vals)))
    if c['kind'].startswith('mms'):
        gx,gy=np.meshgrid(np.linspace(0,1,201),np.linspace(0,1,201))
        with torch.no_grad(): tm=float(predict(model,tensor(np.c_[gx.ravel(),gy.ravel()])).max())
    flux=0.;throughput=0.;bc_error=[]
    for points,normals,edge in boundary_points(c,512):
        z=tensor(points,True);T=predict(model,z);k=field_k(c,z[:,:1],z[:,1:2],torch)
        qn=(-k*(gradients(T,z)*tensor(normals)).sum(1,keepdim=True)).detach().numpy().ravel()
        if c['kind']=='annulus': length=2*math.pi*(c['ri'] if edge=='inner' else c['ro'])
        elif edge.startswith('cable'): length=2*math.pi*c['radius']
        else:
            x0,x1,y0,y1=c['bounds'];length=(x1-x0) if edge in ('top','bottom') else y1-y0
        flux+=float(qn.mean()*length);throughput+=float(abs(qn).mean()*length)
        if c['kind'].startswith('mms'):
            target=exact(c,points[:,0],points[:,1]);bc_error.extend((T.detach().numpy().ravel()-target).tolist())
    if c['kind'].startswith('mms'):
        # Deterministic tensor-product Gauss quadrature, split at the interface.
        ga,gw=np.polynomial.legendre.leggauss(64);generation=0.
        for left,right in [(0,.5),(.5,1)]:
            xx,yy=np.meshgrid(left+(ga+1)*(right-left)/2,(ga+1)/2)
            generation+=float(np.sum(source(c,xx,yy)*np.outer(gw,gw))*(right-left)/4)
    else: generation=0.
    rpts=tensor(xy,True);temp=predict(model,rpts);k=field_k(c,rpts[:,:1],rpts[:,1:2],torch)
    residual=laplace_variable_k(temp,rpts,k)+source(c,rpts[:,:1],rpts[:,1:2],torch)
    pde_rms=float(torch.sqrt(torch.mean(residual.detach()**2)))
    report=dict(metadata=meta,Tmax_C=tm,conductor_C=tc,net_flux_W_m=flux,source_W_m=generation,throughput_W_m=throughput,balance_pct=abs(flux-generation)/max(abs(generation),throughput/2,1e-12)*100,pde_rmse_W_m3=pde_rms)
    if c['kind']=='mms_interface':
        yy=np.linspace(.0001,.9999,512);z=tensor(np.c_[np.full(512,.5),yy],True)
        l=model.nets[0](2*z-1)*c['scale'];r=model.nets[-1](2*z-1)*c['scale']
        jump=.5*gradients(l,z)[:,:1]-2*gradients(r,z)[:,:1]
        report.update(interface_T_max_K=float(torch.max(abs(l-r)).detach()),interface_flux_max_W_m2=float(torch.max(abs(jump)).detach()))
    if c['kind']!='cable':
        err=vals-exact(c,xy[:,0],xy[:,1]);report.update(rmse_exact_K=float(np.sqrt(np.mean(err**2))),max_error_exact_K=float(np.max(abs(err))))
    f=out/'fem_l2.npz'
    if not f.exists(): f=Path(__file__).resolve().parent/'results'/c['id']/'fem_l2.npz'
    if f.exists():
        femdata=np.load(f)
        if not np.allclose(femdata['xy'],xy,rtol=0,atol=1e-14): raise ValueError('Different evaluation points')
        fm=json.loads(f.with_suffix('.json').read_text());assert fm['case_sha256']==fingerprint(c)
        err=vals-femdata['T'];delta=max(abs(fm['Tmax_C']-c['T0']),1e-12)
        report.update(rmse_fem_K=float(np.sqrt(np.mean(err**2))),nrmse_fem_pct=float(100*np.sqrt(np.mean(err**2))/delta),max_error_fem_K=float(np.max(abs(err))),Tmax_fem_C=fm['Tmax_C'],error_Tmax_K=tm-fm['Tmax_C'],error_Tmax_rise_pct=100*abs(tm-fm['Tmax_C'])/delta)
        report['thermal_criteria_pass']=report['nrmse_fem_pct']<=5 and report['error_Tmax_rise_pct']<=5 and report['balance_pct']<=2
    seed=meta['seed'];np.savez_compressed(out/f'pinn_seed{seed}.npz',xy=xy,T=vals,surface_xy=np.array(surface),surface_T=np.array(sv))
    (out/f'pinn_seed{seed}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'case':c['id'],'seed':seed,'Tmax_C':tm,'balance_pct':report['balance_pct'],'rmse':report.get('rmse_fem_K'), 'seconds':meta['training_seconds']},ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases',nargs='+',default=['all'])
    ap.add_argument('--config',type=Path)
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,23,37])
    ap.add_argument('--adam',type=int,default=1200)
    ap.add_argument('--lbfgs',type=int,default=800)
    ap.add_argument('--width',type=int,default=32)
    ap.add_argument('--depth',type=int,default=3)
    ap.add_argument('--n',type=int,default=768)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--pde-weight',type=float)
    ap.add_argument('--bc-weight',type=float,default=10.)
    ap.add_argument('--flux-weight',type=float,default=10.)
    ap.add_argument('--energy-weight',type=float,default=10.)
    ap.add_argument('--output',default='Benchmarks/results')
    ap.add_argument('--evaluate-only',action='store_true')
    ap.add_argument('--variant',choices=['enriched','direct','local','global','conservative','multipole'],default='enriched')
    args=ap.parse_args()
    if args.config:
        config=json.loads(args.config.read_text(encoding='utf-8'))
        for key,value in config.items():
            if key not in vars(args): raise ValueError(f'Unknown training option: {key}')
            setattr(args,key,value)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads);catalog=cases();names=list(catalog) if args.cases==['all'] else args.cases
    for name in names:
        for seed in args.seeds:
            out=Path(args.output)/name
            if args.evaluate_only:
                saved=torch.load(out/f'pinn_seed{seed}.pt',weights_only=False);m=saved['metadata'];model=Network(catalog[name],m['width'],m['depth'],m.get('variant','enriched'));model.load_state_dict(saved['state_dict']);evaluate_model(model,m,out)
            else:
                print(f'PINN {name} seed {seed}',flush=True);train(catalog[name],seed,args,out)

if __name__=='__main__': main()
