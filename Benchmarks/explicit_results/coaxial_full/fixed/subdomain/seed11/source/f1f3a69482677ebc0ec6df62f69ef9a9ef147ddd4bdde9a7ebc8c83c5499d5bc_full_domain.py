"""Explicit cable cross section shared by all active PINNs and the FEM reference.

SI units. No thermal resistance or reconstructed conductor temperature is used.
The old case catalog is an input source; the resulting physics has a new hash.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import numpy as np
from Benchmarks.cases import field_k, interfaces, validate_case
from Benchmarks.skin_effect import loss_specification,prescribed_source

PHYSICS_VERSION = 'explicit-heat-2d-v1'


@dataclass(frozen=True)
class Region:
    id: int
    name: str
    kind: str
    cable: int = -1
    layer: int = -1
    ri: float = 0.
    ro: float = 0.
    k: float = 0.
    axis: int = 1
    lower: float = -float('inf')
    upper: float = float('inf')


@dataclass(frozen=True)
class Interface:
    name: str
    left: int
    right: int
    radius: float = 0.
    cable: int = -1
    axis: int = 1
    position: float = 0.


def coaxial_case():
    """Synthetic verification case; exact solution is external to the loss."""
    return dict(id='coaxial_full', kind='cable', bounds=[-.15,.15,-.15,.15],
        T0=20., scale=30., cables=[[0.,0.]], radius=.015,
        layers=[[0.,.0055,400.],[.0055,.012,.286],[.012,.013,380.],[.013,.015,.45]],
        current=270., R20=.000193, alpha=.00393, power=270.**2*.000193,
        k=1., patch=None, bands=[], pair=None, source='Synthetic concentric verification',
        outer_radius=.15, physics_version=PHYSICS_VERSION)


class FullDomain:
    def __init__(self, case, mode='fixed'):
        if mode not in ('fixed','dc_temperature'):
            raise ValueError('Source mode must be fixed or dc_temperature')
        c=copy.deepcopy(case)
        extra={'outer_radius','physics_version','heat_capacity_J_m3K','outer_gradient_K_m','electrical'}
        validate_case({k:v for k,v in c.items() if k not in extra})
        if c['kind']!='cable':raise ValueError('Explicit cable solver requires a cable case')
        if c.get('outer_radius'):
            if len(c['cables'])!=1 or c['cables'][0]!=[0.,0.] or c['outer_radius']<=c['radius']:
                raise ValueError('Circular verification requires one centered cable')
            if interfaces(c):raise ValueError('Circular outer boundary with soil strata is not supported')
        self.case=c
        self.mode=mode
        self.electrical=loss_specification(c)
        if mode=='dc_temperature' and (self.electrical['resistance_basis']!='dc' or self.electrical['frequency_Hz']!=0):
            raise ValueError('Local DC coupling requires DC inputs; AC is a prescribed heat-source model in this version')
        self.regions=[]
        self.interfaces=[]
        cuts=interfaces(c)
        axes={0 if i['axis']=='x' else 1 for i in cuts}
        if len(axes)>1:raise ValueError('Soil strata must share one axis')
        axis=next(iter(axes),1)
        bounds=c['bounds']
        limits=[bounds[0 if axis==0 else 2]]+[i['position'] for i in cuts]+[bounds[1 if axis==0 else 3]]
        for j,(lo,hi) in enumerate(zip(limits,limits[1:])):
            self.regions.append(Region(j,f'soil_{j}','soil',axis=axis,lower=lo,upper=hi))
        self.nsoil=len(self.regions)
        for j,info in enumerate(cuts):
            self.interfaces.append(Interface(f'soil_interface_{j}',j,j+1,axis=axis,position=info['position']))
        for cable,center in enumerate(c['cables']):
            first=len(self.regions)
            for layer,(ri,ro,k) in enumerate(c['layers']):
                index=len(self.regions)
                self.regions.append(Region(index,f'cable_{cable}_layer_{layer}',
                    'conductor' if layer==0 else 'layer',cable,layer,ri,ro,k))
            soil=int(np.searchsorted(limits[1:-1],center[axis],side='right'))
            for layer,(_,ro,_) in enumerate(c['layers']):
                left=first+layer
                right=left+1 if layer<len(c['layers'])-1 else soil
                self.interfaces.append(Interface(f'cable_{cable}_interface_{layer}',left,right,ro,cable))
        self.conductors=[r for r in self.regions if r.kind=='conductor']
        self.capacities=c.get('heat_capacity_J_m3K',{})
        if set(self.capacities)-{r.name for r in self.regions}:
            raise ValueError('Unknown heat-capacity region')
        for value in self.capacities.values():
            if value is not None and (not np.isfinite(value) or value<=0):
                raise ValueError('Volumetric heat capacity must be positive')

    def specification(self):
        return dict(physics_version=PHYSICS_VERSION, case=self.case, source_mode=self.mode,
            equation='rho_cp*dT/dt + div(q) = Q; q = -k*grad(T)',
            interface='continuous T and normal heat flux',
            interior='explicit conductor and every material layer',
            electrical=self.electrical, electrical_coupling='prescribed source' if self.mode=='fixed' else 'sigma(T)*Ez^2; Ez=I/integral(sigma dA)',
            assumptions=['2D long cable','isotropic conduction','perfect thermal contact','DC or prescribed heat sources'],
            excluded=(['skin effect'] if self.electrical['skin_model']=='none' and self.electrical['resistance_basis']=='dc' else [])+['proximity effect','moisture transport','3D electromagnetic field'])

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.specification(),sort_keys=True,allow_nan=False).encode()).hexdigest()

    def capacity(self,region):
        value=self.capacities.get(region.name)
        if value is None:raise ValueError(f'Missing documented heat capacity for {region.name}')
        return float(value)

    def contains(self,region,xy):
        c=self.case;xy=np.asarray(xy)
        if region.kind!='soil':
            r2=np.sum((xy[:,:2]-c['cables'][region.cable])**2,axis=1)
            return (r2>=region.ri**2)&(r2<=region.ro**2)
        x0,x1,y0,y1=c['bounds']
        mask=(xy[:,0]>=x0)&(xy[:,0]<=x1)&(xy[:,1]>=y0)&(xy[:,1]<=y1)
        mask&=(xy[:,region.axis]>=region.lower)&(xy[:,region.axis]<=region.upper)
        if c.get('outer_radius'):mask&=np.sum(xy[:,:2]**2,axis=1)<=c['outer_radius']**2
        for center in c['cables']:mask&=np.sum((xy[:,:2]-center)**2,axis=1)>c['radius']**2
        return mask

    def area(self,region):
        if region.kind!='soil':return math.pi*(region.ro**2-region.ri**2)
        c=self.case
        if c.get('outer_radius'):return math.pi*(c['outer_radius']**2-c['radius']**2)
        x0,x1,y0,y1=c['bounds']
        length=(y1-y0) if region.axis==0 else (x1-x0)
        area=length*(region.upper-region.lower)
        for center in c['cables']:
            if region.lower<center[region.axis]<region.upper:area-=math.pi*c['radius']**2
        return area

    def sample(self,region,n,seed,near=False):
        if n<1:raise ValueError('Every material requires interior points')
        rng=np.random.default_rng(seed);c=self.case
        if region.kind!='soil':
            angle=rng.uniform(0,2*math.pi,n)
            radius=np.sqrt(rng.uniform(region.ri**2,region.ro**2,n))
            return np.c_[radius*np.cos(angle),radius*np.sin(angle)]+c['cables'][region.cable]
        x0,x1,y0,y1=c['bounds'];cloud=[];size=0
        for _ in range(1000):
            candidate=rng.uniform([x0,y0],[x1,y1],(max(256,2*(n-size)),2))
            if near and size<n//2:
                center=c['cables'][int(rng.integers(len(c['cables'])))];angle=rng.uniform(0,2*math.pi,len(candidate))
                radius=c['radius']*np.exp(rng.uniform(1e-5,math.log(max(.3,4*c['radius'])/c['radius']),len(candidate)))
                candidate=np.c_[radius*np.cos(angle),radius*np.sin(angle)]+center
            candidate=candidate[self.contains(region,candidate)]
            cloud.append(candidate);size+=len(candidate)
            if size>=n:return np.vstack(cloud)[:n]
        raise RuntimeError(f'Unable to sample region {region.name}')

    def interface_points(self,interface,n):
        if interface.radius:
            angle=(np.arange(n)+.5)*2*math.pi/n
            normal=np.c_[np.cos(angle),np.sin(angle)]
            return normal*interface.radius+self.case['cables'][interface.cable],normal,2*math.pi*interface.radius
        c=self.case;x0,x1,y0,y1=c['bounds'];s=(np.arange(n)+.5)/n
        if interface.axis==0:xy=np.c_[np.full(n,interface.position),y0+(y1-y0)*s];normal=np.tile([1.,0.],(n,1));length=y1-y0
        else:xy=np.c_[x0+(x1-x0)*s,np.full(n,interface.position)];normal=np.tile([0.,1.],(n,1));length=x1-x0
        return xy,normal,length

    def outer_boundary(self,n):
        c=self.case
        if c.get('outer_radius'):
            angle=(np.arange(n)+.5)*2*math.pi/n;normal=np.c_[np.cos(angle),np.sin(angle)]
            return [(normal*c['outer_radius'],normal,2*math.pi*c['outer_radius'])]
        x0,x1,y0,y1=c['bounds'];s=(np.arange(n)+.5)/n
        return [(np.c_[x0+(x1-x0)*s,s*0+y0],np.tile([0.,-1.],(n,1)),x1-x0),
                (np.c_[x0+(x1-x0)*s,s*0+y1],np.tile([0.,1.],(n,1)),x1-x0),
                (np.c_[s*0+x0,y0+(y1-y0)*s],np.tile([-1.,0.],(n,1)),y1-y0),
                (np.c_[s*0+x1,y0+(y1-y0)*s],np.tile([1.,0.],(n,1)),y1-y0)]

    def boundary_temperature(self,xy):
        return self.case['T0']+self.case.get('outer_gradient_K_m',0.)*xy[:,0:1]

    def conductivity(self,region,xy,backend=np):
        return field_k(self.case,xy[:,0:1],xy[:,1:2],backend) if region.kind=='soil' else xy[:,0:1]*0+region.k

    def source(self,region,xy):
        if region.kind!='conductor':return xy[:,0:1]*0
        points=xy.detach().cpu().numpy() if hasattr(xy,'detach') else np.asarray(xy)
        value=prescribed_source(self.case,points,self.case['cables'][region.cable])
        return xy[:,0:1]*0+xy.new_tensor(value) if hasattr(xy,'new_tensor') else value

    def scales(self,region):
        c=self.case;p=c['power']
        if region.kind=='soil':
            length=max(c.get('outer_radius',0.),.1) if c.get('outer_radius') else 1.
            k=c['k'];qscale=p/(2*math.pi*max(c['radius'],length*.1))
        else:
            length=region.ro-region.ri;k=region.k
            qscale=p/(2*math.pi*max((region.ri+region.ro)/2,c['layers'][0][1]))
        return dict(length=length,flux=qscale,temperature=qscale*length/k,
            divergence=qscale/length)

    def evaluation_points(self,n=512):
        xy=[];labels=[];weights=[]
        for r in self.regions:
            points=self.sample(r,n,203001+r.id)
            if r.kind=='conductor':points[0]=self.case['cables'][r.cable]
            xy.append(points);labels.extend([r.id]*n);weights.extend([self.area(r)/n]*n)
        return np.vstack(xy),np.array(labels),np.array(weights)


def dc_conductivity(domain,region,temperature):
    """Local electrical conductivity, S/m; R20 is only an electrical datum."""
    c=domain.case;spec=domain.electrical
    return 1/(spec['resistance_ohm_m']*domain.area(region)*(1+c['alpha']*(temperature-spec['reference_temperature_C'])))


def radial_exact(domain,radius):
    """External analytical verification only: integrated Poisson/Laplace solution."""
    c=domain.case
    if not c.get('outer_radius') or domain.mode!='fixed' or c.get('outer_gradient_K_m',0) or c.get('electrical'):
        raise ValueError('Analytical check only valid for concentric fixed-source verification')
    radius=np.asarray(radius);p=c['power'];a=c['layers'][0][1]
    result=np.zeros_like(radius,dtype=float)+c['T0']
    shells=c['layers'][1:]+[[c['radius'],c['outer_radius'],c['k']]]
    for ri,ro,k in shells:
        result+=p/(2*math.pi*k)*np.log(ro/np.clip(radius,ri,ro))
    result+=p/(4*math.pi*c['layers'][0][2])*np.maximum(1-(radius/a)**2,0)
    return result
