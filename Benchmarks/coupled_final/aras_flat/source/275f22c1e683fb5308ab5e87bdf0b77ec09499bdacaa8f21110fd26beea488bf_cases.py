"""Especificación física compartida: unidades SI y temperaturas en grados Celsius.

Los casos de cables son problemas controlados de suelo con fronteras circulares
de flujo uniforme; las capas internas se recuperan mediante resistencia radial.
No son reproducciones exactas de los artículos ni de sus pérdidas IEC completas.
"""
from pathlib import Path
import copy
import csv
import hashlib
import json
import math
import numpy as np
from Benchmarks.expressions import conductivity, evaluate as evaluate_expression, manufactured_source

ROOT = Path(__file__).resolve().parents[1]

def interfaces(c):
    spec=c.get('conductivity')
    if isinstance(spec,dict) and spec.get('type')=='layers' and spec.get('smoothing',0)==0:
        return [dict(axis=spec['axis'],position=z) for z in spec['interfaces']]
    if c['kind']=='mms_interface':return [c.get('interface',dict(axis='x',position=.5))]
    return []

def _historical_case_builder():
    result = {}
    for name in ['mms_constant', 'mms_variable', 'mms_interface', 'mms_robin']:
        result[name] = dict(id=name, kind=name, bounds=[0.,1.,0.,1.], T0=20., scale=30., source='Plan/envío_rev1/04_Datos_objeto_estudio; MMS-2D-01/02; Robin de elaboración propia')
    result['annulus'] = dict(id='annulus', kind='annulus', ri=.06, ro=20., k=1.25, power=40., T0=20., scale=40/(2*math.pi*1.25)*math.log(20/.06), source='CIGRE TB 963, pp. 52–55; potencia exacta 40 W/m')
    for name,folder in [('xlpe_single','xlpe_single_cable'),('aras_single','aras_2005_154kv'),('aras_flat','aras_2005_154kv_flat'),('kim_sand','kim_2024_154kv_bedding')]:
        base=ROOT/'examples'/folder/'data'
        with (base/'cables_placement.csv').open(encoding='utf-8-sig') as f: pl=list(csv.DictReader(f))
        is_kim=name.startswith('kim')
        if is_kim:
            layers=[(0,.0212,400),(.0212,.0232,.2857),(.0232,.0402,.2857),(.0402,.0415,.2857),(.0415,.0425,.167),(.0425,.045,237),(.045,.05,.2857),(.05,.10,2.15),(.10,.11,.2857)]
        elif name.startswith('aras'):
            layers=[(0,.01885,400),(.01885,.04085,.2857),(.04085,.04935,384.6),(.04935,.05335,.45)]
        else:
            layers=[(0,.0055,400),(.0055,.012,.286),(.012,.013,380),(.013,.015,.45)]
        R20=.000193 if name=='xlpe_single' else .0000151
        I=float(pl[0]['current_A'])
        # Fixed, transparent DC losses at 20 C; no use of paper temperatures.
        Q=I*I*R20
        result[name]=dict(id=name,kind='cable',bounds=[-4.,4.,-4.,0.],T0=20.,scale=50.,
            cables=[[float(p['cx']),float(p['cy'])] for p in pl],radius=layers[-1][1],layers=layers,
            power=Q,current=I,R20=R20,alpha=.00393,k=1.365 if is_kim else 1.,
            patch=None,bands=[],pair=None,source=str(base.relative_to(ROOT)).replace('\\','/'),
            adaptation='Dominio controlado 8 x 4 m; Dirichlet 20 C; pérdidas DC a 20 C; flujo circular uniforme. Kim usa nueve capas de la ficha corregida de 2025.')
    base=result['xlpe_single']
    for name,k,cx,cy,w,h in [('xlpe_dry_near',.5,0,-.7,.5,.5),('xlpe_dry_far',.5,1,-.7,.5,.5),('xlpe_dry_large',.5,0,-.7,1.,1.),('xlpe_backfill',2.,0,-.7,.5,.5)]:
        c=copy.deepcopy(base);c.update(id=name,pair='xlpe_single',patch=[cx,cy,w,h,k,.08]);result[name]=c
    c=copy.deepcopy(result['kim_sand']);c.update(id='kim_pac',pair='kim_sand',patch=[0.,-1.4,1.3,.9,2.094,.1]);result[c['id']]=c
    c=copy.deepcopy(result['kim_pac']);c.update(id='kim_layered',pair=None,bands=[[-.56,1.804,1.351],[-1.76,1.351,1.517]]);result[c['id']]=c
    return result


def validate_case(c):
    """Validate the common physical contract before either solver starts."""
    allowed={'id','kind','bounds','T0','scale','source','ri','ro','k','power','cables','radius','layers','current','R20','alpha','patch','bands','pair','adaptation','band_smoothing','conductivity','interface','exact_expression','source_expression'}
    if set(c)-allowed:raise ValueError(f'Unknown case fields: {sorted(set(c)-allowed)}')
    if not isinstance(c.get('id'),str) or not c['id']: raise ValueError('Missing case id')
    if c['kind'] not in ('cable','annulus','mms','mms_constant','mms_variable','mms_smooth_2d','mms_high_contrast','mms_interface','mms_robin'): raise ValueError('Unsupported case kind')
    if c['kind']=='mms' and not all(k in c for k in ('conductivity','exact_expression','source_expression')):raise ValueError('Generic MMS requires physical expressions')
    if c['kind'].startswith('mms'):
        if c['bounds']!=[0.,1.,0.,1.]:raise ValueError('Manufactured mesh currently requires the unit square')
        for info in interfaces(c):
            if not 0<info['position']<1 or not math.isclose(info['position']*16,round(info['position']*16),abs_tol=1e-10):raise ValueError('Manufactured interface must align with the coarsest 16-cell mesh')
    if c['scale']<=0 or not np.isfinite(c['scale']): raise ValueError('scale must be positive')
    if 'conductivity' in c:
        spec=c['conductivity']
        if isinstance(spec,dict) and set(spec)-{'type','expression','axis','interfaces','values','smoothing'}:raise ValueError('Unknown conductivity fields')
        expr=spec if isinstance(spec,str) else (spec.get('expression','') if isinstance(spec,dict) else '')
        if 'where' in expr:raise ValueError('Specify conductivity jumps with type=layers, not a hidden where expression')
        if isinstance(spec,dict) and spec.get('type')=='layers':
            if spec.get('axis') not in ('x','y') or len(spec['values'])!=len(spec['interfaces'])+1 or any(a>=b for a,b in zip(spec['interfaces'],spec['interfaces'][1:])) or min(spec['values'])<=0 or spec.get('smoothing',0)<0:raise ValueError('Invalid layer specification')
        if c.get('patch') or c.get('bands'):raise ValueError('Use either conductivity or legacy patch/bands, not both')
        if c['kind'].startswith('mms') and not all(k in c for k in ['exact_expression','source_expression']):raise ValueError('Manufactured data need exact and source expressions')
        x0,x1,y0,y1=c['bounds'];gx,gy=np.meshgrid(np.linspace(x0,x1,41),np.linspace(y0,y1,41))
        kval=conductivity(spec,gx,gy,np)
        if not np.all(np.isfinite(kval)) or np.min(kval)<=0:raise ValueError('Nonpositive/nonfinite conductivity on validation grid')
    if c['kind']=='annulus':
        if not 0<c['ri']<c['ro'] or c['k']<=0: raise ValueError('Invalid annulus')
    else:
        x0,x1,y0,y1=c['bounds']
        if x0>=x1 or y0>=y1: raise ValueError('Invalid domain')
    if c['kind']=='cable':
        if c['k']<=0 or c['power']<=0 or c['radius']<=0: raise ValueError('Invalid physical coefficients')
        if not math.isclose(c['power'],c['current']**2*c['R20']): raise ValueError('DC power differs from I^2 R20')
        last=0.
        for ri,ro,k in c['layers']:
            if not math.isclose(ri,last,abs_tol=1e-12) or ro<=ri or k<=0: raise ValueError('Invalid material layers')
            last=ro
        if not math.isclose(last,c['radius']): raise ValueError('Outer radius differs from layers')
        for i,(x,y) in enumerate(c['cables']):
            r=c['radius']
            if not (x0+r<x<x1-r and y0+r<y<y1-r): raise ValueError('Cable outside domain')
            for xx,yy in c['cables'][:i]:
                if math.hypot(x-xx,y-yy)<=2*r: raise ValueError('Overlapping cables')
            for info in interfaces(c):
                coord=x if info['axis']=='x' else y
                if abs(coord-info['position'])<=r:raise ValueError('A discrete soil interface must not intersect a cable in this geometry version')
        if c.get('patch'):
            _,_,w,h,k,e=c['patch']
            if min(w,h,k,e)<=0: raise ValueError('Invalid thermal patch')
    return c


def cases(directory=None):
    """One editable JSON file per case; no numerical options in physical data."""
    result={}
    for path in sorted(Path(directory or Path(__file__).with_name('cases')).glob('*.json')):
        c=validate_case(json.loads(path.read_text(encoding='utf-8')))
        if c['id'] in result: raise ValueError('Duplicate case id')
        result[c['id']]=c
    if not result: raise ValueError('Empty case catalog')
    return result

def fingerprint(case):
    return hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()

def field_k(c,x,y,backend=np):
    """Same expression evaluated by numpy, torch or a UFL adapter."""
    if 'conductivity' in c:return conductivity(c['conductivity'],x,y,backend)
    kind=c['kind']
    if kind=='mms_variable': return 1+x
    if kind=='mms_smooth_2d': return 1+.4*backend.sin(2*math.pi*x)*backend.cos(2*math.pi*y)
    if kind=='mms_high_contrast': return backend.exp(math.log(10)*(x+y)/2)
    if kind=='mms_interface': return backend.where(x<=.5,.5+0*x,2.+0*x)
    k=c.get('k',1.)+0*x
    if c.get('bands'):
        k=c['bands'][-1][2]+0*x
        for height,above,below in reversed(c['bands']):
            k=k+(above-below)*.5*(1+backend.tanh((y-height)/c.get('band_smoothing',.1)))
    if c.get('patch'):
        cx,cy,w,h,kp,eps=c['patch']
        wx=.5*(backend.tanh((x-cx+w/2)/eps)-backend.tanh((x-cx-w/2)/eps))
        wy=.5*(backend.tanh((y-cy+h/2)/eps)-backend.tanh((y-cy-h/2)/eps))
        k=k+(kp-k)*wx*wy
    return k

def exact(c,x,y,backend=np):
    if 'exact_expression' in c:return evaluate_expression(c['exact_expression'],x,y,backend)
    kind=c['kind']
    if kind=='annulus': return c['T0']+c['power']/(2*math.pi*c['k'])*backend.log(c['ro']/backend.sqrt(x*x+y*y))
    if kind=='mms_robin': return 20+30*x*(1-x)*(1+y)
    if kind=='mms_interface':
        phi=backend.where(x<=.5,10*x/.5,10*(.5/.5+(x-.5)/2.))
        return 20+phi*backend.sin(math.pi*y)
    return 20+30*backend.sin(math.pi*x)*backend.sin(math.pi*y)

def source(c,x,y,backend=np):
    if c.get('source_expression')=='manufactured':return manufactured_source(c['exact_expression'],c['conductivity'],x,y,backend)
    if 'source_expression' in c:return evaluate_expression(c['source_expression'],x,y,backend)
    kind=c['kind']
    if kind in ('cable','annulus'): return 0*x
    if kind=='mms_robin': return 60*(1+y)+0*x
    theta=exact(c,x,y,backend)-20
    if kind=='mms_interface': return field_k(c,x,y,backend)*math.pi**2*theta
    q=2*math.pi**2*field_k(c,x,y,backend)*theta
    if kind=='mms_variable': q=q-30*math.pi*backend.cos(math.pi*x)*backend.sin(math.pi*y)
    if kind in ('mms_smooth_2d','mms_high_contrast'):
        if kind=='mms_smooth_2d':
            kx=.8*math.pi*backend.cos(2*math.pi*x)*backend.cos(2*math.pi*y)
            ky=-.8*math.pi*backend.sin(2*math.pi*x)*backend.sin(2*math.pi*y)
        else:kx=ky=math.log(10)/2*field_k(c,x,y,backend)
        q=q-30*math.pi*(kx*backend.cos(math.pi*x)*backend.sin(math.pi*y)+ky*backend.sin(math.pi*x)*backend.cos(math.pi*y))
    return q

def radial_resistance(c):
    layers=c['layers']
    return 1/(4*math.pi*layers[0][2])+sum(math.log(ro/ri)/(2*math.pi*k) for ri,ro,k in layers[1:])

def in_domain(c,xy,margin=0.):
    if c['kind']=='annulus':
        r=np.linalg.norm(xy,axis=1);return (r>c['ri']+margin)&(r<c['ro']-margin)
    x0,x1,y0,y1=c['bounds'];mask=(xy[:,0]>=x0)&(xy[:,0]<=x1)&(xy[:,1]>=y0)&(xy[:,1]<=y1)
    for cx,cy in c.get('cables',[]): mask &= np.sum((xy-[cx,cy])**2,axis=1)>(c['radius']+margin)**2
    return mask

def evaluation_points(c):
    """Fixed evaluation quadrature independent of all training seeds."""
    rng=np.random.default_rng(20260910)
    if c['kind']=='annulus':
        # Area-uniform audit and a separate log-radial profile at the hot boundary.
        # Exclude the 0.5% rim affected by polygonal/curved mesh containment;
        # the inner hot boundary is evaluated independently below.
        r=np.sqrt(rng.uniform((c['ri']*1.005)**2,(c['ro']*.995)**2,6000));a=rng.uniform(0,2*math.pi,len(r))
        xy=np.c_[r*np.cos(a),r*np.sin(a)]
    else:
        x0,x1,y0,y1=c['bounds'];xy=rng.uniform([x0,y0],[x1,y1],(6500,2));xy=xy[in_domain(c,xy)][:6000]
    return xy

def surface_points(c,n=128,offset=1e-5):
    a=(np.arange(n)+.5)*2*math.pi/n
    if c['kind']=='annulus': return [np.c_[np.cos(a),np.sin(a)]*(c['ri']+offset)]
    return [np.c_[np.cos(a),np.sin(a)]*(c['radius']+offset)+[cx,cy] for cx,cy in c.get('cables',[])]
