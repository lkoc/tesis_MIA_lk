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
from pinn_v2.cases.boundary import RECT_SIDES, SIDE_TAGS, parse_boundary_conditions, parse_contact
from pinn_v2.cases.catalog import field_k, interfaces, validate_case
from pinn_v2.physics.skin_effect import loss_specification,prescribed_power,prescribed_source

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
        extra={'outer_radius','physics_version','heat_capacity_J_m3K','outer_gradient_K_m','electrical','boundary_conditions','contact','schema_version','source_power_W_m'}
        validate_case({k:v for k,v in c.items() if k not in extra})
        if c['kind']!='cable':raise ValueError('Explicit cable solver requires a cable case')
        if 'source_power_W_m' in c and (not np.isfinite(c['source_power_W_m']) or c['source_power_W_m']<=0):
            raise ValueError('Thermal source power must be positive and finite')
        if mode=='dc_temperature' and 'source_power_W_m' in c:
            raise ValueError('A fixed thermal source cannot be used in dc_temperature mode')
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
        circular=bool(c.get('outer_radius'))
        if 'boundary_conditions' in c:
            if c.get('outer_gradient_K_m'):raise ValueError('outer_gradient_K_m belongs to the legacy Dirichlet boundary; use boundary_conditions only')
            self.bcs=parse_boundary_conditions(c['boundary_conditions'],circular)
        else:self.bcs=None
        self.contact=parse_contact(c.get('contact'),{i.name for i in self.interfaces})

    def specification(self):
        c=self.case
        case=copy.deepcopy(c)
        conditions=self.side_conditions()
        default_dirichlet=all(b['type']=='dirichlet' and math.isclose(
            c['T0'] if b.get('legacy') else b['T_C'], c['T0'], rel_tol=0, abs_tol=1e-12)
            for b in conditions.values())
        if default_dirichlet:
            case.pop('boundary_conditions',None)
        spec=dict(physics_version=PHYSICS_VERSION, case=case, source_mode=self.mode,
            equation='rho_cp*dT/dt + div(q) = Q; q = -k*grad(T)',
            interface='continuous T and normal heat flux' if not self.contact else 'continuous normal heat flux; temperature jump [T]=R*q.n on the documented interfaces only',
            interior='explicit conductor and every material layer',
            electrical=self.electrical, electrical_coupling='prescribed source' if self.mode=='fixed' else 'sigma(T)*Ez^2; Ez=I/integral(sigma dA)',
            assumptions=['2D long cable','isotropic conduction','perfect thermal contact' if not self.contact else 'perfect thermal contact except documented interfaces','DC or prescribed heat sources'],
            excluded=(['skin effect'] if self.electrical['skin_model']=='none' and self.electrical['resistance_basis']=='dc' else [])+([] if self.electrical.get('includes_proximity',False) else ['proximity effect'])+['moisture transport','3D electromagnetic field'])
        if 'source_power_W_m' in self.case:
            spec['thermal_source']=dict(profile='uniform volumetric source in the copper core',power_per_cable_W_m=prescribed_power(self.case))
            spec['assumptions'].append('fixed per-cable thermal source from the case data')
        return spec

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
        if near and region.kind=='soil':
            # Keep an actual half-domain, half-near mixture even when the first
            # rejection batch contains more accepted points than requested.
            far=self.sample(region,n-n//2,seed,near=False)
            if n//2==0:return far
            rng=np.random.default_rng(seed+104729);cloud=[];size=0;c=self.case
            for _ in range(1000):
                center=c['cables'][int(rng.integers(len(c['cables'])))];count=max(256,n)
                angle=rng.uniform(0,2*math.pi,count)
                distance=max(region.lower-center[region.axis],center[region.axis]-region.upper,0.)
                reach=max(.3,4*c['radius'],1.5*distance+.1)
                radius=c['radius']*np.exp(rng.uniform(1e-5,math.log(reach/c['radius']),count))
                candidate=np.c_[radius*np.cos(angle),radius*np.sin(angle)]+center
                candidate=candidate[self.contains(region,candidate)];cloud.append(candidate);size+=len(candidate)
                if size>=n//2:return np.vstack([far,np.vstack(cloud)[:n//2]])
            raise RuntimeError(f'Unable to sample near cables in {region.name}')
        rng=np.random.default_rng(seed);c=self.case
        if region.kind!='soil':
            angle=rng.uniform(0,2*math.pi,n)
            radius=np.sqrt(rng.uniform(region.ri**2,region.ro**2,n))
            return np.c_[radius*np.cos(angle),radius*np.sin(angle)]+c['cables'][region.cable]
        x0,x1,y0,y1=c['bounds'];cloud=[];size=0
        for _ in range(1000):
            candidate=rng.uniform([x0,y0],[x1,y1],(max(256,2*(n-size)),2))
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
        c=self.case;p=prescribed_power(c)
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

    @property
    def has_imperfect_contact(self):
        return bool(self.contact)

    def side_conditions(self):
        """Per-side boundary data; without boundary_conditions the legacy Dirichlet T0 applies."""
        if self.bcs is not None:return self.bcs
        sides=('outer',) if self.case.get('outer_radius') else RECT_SIDES
        return {s:dict(type='dirichlet',T_C=None,legacy=True) for s in sides}

    @property
    def all_dirichlet(self):
        return all(b['type'] in ('dirichlet','dirichlet_profile') for b in self.side_conditions().values())

    def dirichlet_value(self,side,xy):
        bc=self.side_conditions()[side]
        if bc['type']=='dirichlet_profile':
            depth=self.case['bounds'][3]-xy[:,1:2]
            if hasattr(xy,'new_tensor'):
                import torch
                knots=xy.new_tensor(bc['depth_m']);values=xy.new_tensor(bc['temperature_C'])
                index=torch.bucketize(depth.contiguous(),knots[1:-1])
                lo,hi=knots[index],knots[index+1]
                vlo,vhi=values[index],values[index+1]
                fraction=((depth-lo)/(hi-lo)).clamp(0,1)
                return vlo+fraction*(vhi-vlo)
            return np.interp(depth.ravel(),bc['depth_m'],bc['temperature_C']).reshape(-1,1)
        if bc['type']!='dirichlet':raise ValueError(f'{side} is not a Dirichlet side')
        return self.boundary_temperature(xy) if bc.get('legacy') else xy[:,0:1]*0+bc['T_C']

    def boundary_sides(self,n):
        """Outer boundary as named sides: name, tag, points, outward normals, length, condition."""
        names=('outer',) if self.case.get('outer_radius') else RECT_SIDES
        conditions=self.side_conditions()
        return [dict(name=name,tag=SIDE_TAGS[name],xy=xy,normal=normal,length=length,bc=conditions[name])
                for name,(xy,normal,length) in zip(names,self.outer_boundary(n))]

    def source_integral(self,region,n=96):
        """Integral of Q over a conductor, W/m, by Gauss quadrature in the radius."""
        if region.kind!='conductor':return 0.
        z,w=np.polynomial.legendre.leggauss(n);radius=region.ro*(z+1)/2
        center=np.asarray(self.case['cables'][region.cable])
        xy=np.c_[center[0]+radius,np.full(n,center[1])]
        values=np.asarray(self.source(region,xy)).ravel()
        return float(np.sum(w*region.ro/2*2*math.pi*radius*values))


def dc_conductivity(domain,region,temperature):
    """Local electrical conductivity, S/m; R20 is only an electrical datum."""
    c=domain.case;spec=domain.electrical
    return 1/(spec['resistance_ohm_m']*domain.area(region)*(1+c['alpha']*(temperature-spec['reference_temperature_C'])))


def outer_temperature(domain):
    """Exact outer-surface temperature of a concentric case from its Dirichlet or Robin condition."""
    c=domain.case;bc=domain.side_conditions()['outer']
    if bc['type']=='dirichlet':return c['T0'] if bc.get('legacy') else bc['T_C']
    if bc['type']=='robin':return bc['T_inf_C']+c['power']/(2*math.pi*c['outer_radius']*bc['h_W_m2K'])
    raise ValueError('A Neumann-only concentric boundary leaves the temperature level undetermined')


def radial_exact(domain,radius):
    """External analytical verification only: integrated Poisson/Laplace solution."""
    c=domain.case
    if not c.get('outer_radius') or domain.mode!='fixed' or c.get('outer_gradient_K_m',0) or c.get('electrical'):
        raise ValueError('Analytical check only valid for concentric fixed-source verification')
    radius=np.asarray(radius);p=c['power'];a=c['layers'][0][1]
    result=np.zeros_like(radius,dtype=float)+outer_temperature(domain)
    shells=c['layers'][1:]+[[c['radius'],c['outer_radius'],c['k']]]
    for ri,ro,k in shells:
        result+=p/(2*math.pi*k)*np.log(ro/np.clip(radius,ri,ro))
    result+=p/(4*math.pi*c['layers'][0][2])*np.maximum(1-(radius/a)**2,0)
    return result
