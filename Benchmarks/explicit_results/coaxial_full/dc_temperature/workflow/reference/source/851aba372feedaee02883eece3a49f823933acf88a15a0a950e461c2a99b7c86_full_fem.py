"""Conforming P2 FEM reference with explicit conductor and all annular layers.

Run under the documented FEniCSx environment. No circular Neumann source and no
radial reconstruction: heat is generated volumetrically in tagged conductors.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,sys,time
import numpy as np
import dolfinx,gmsh,ufl
from dolfinx import fem,geometry
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Benchmarks.full_domain import FullDomain,coaxial_case,dc_conductivity,radial_exact,PHYSICS_VERSION
from Benchmarks.cases import cases,field_k
from Benchmarks.fem import UFLBackend


def create_mesh(domain,level):
    if MPI.COMM_WORLD.size!=1:raise ValueError('Reference currently requires one MPI rank; do not silently assemble partial balances')
    c=domain.case;gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0);gmsh.model.add(c['id']+'_explicit')
        occ=gmsh.model.occ;x0,x1,y0,y1=c['bounds']
        outer=occ.addDisk(0,0,0,c['outer_radius'],c['outer_radius']) if c.get('outer_radius') else occ.addRectangle(x0,y0,0,x1-x0,y1-y0)
        tools=[];disk_indices={}
        for cable,(cx,cy) in enumerate(c['cables']):
            for layer,(_,ro,_) in enumerate(c['layers']):
                disk_indices[(cable,layer)]=len(tools)
                tools.append((2,occ.addDisk(cx,cy,0,ro,ro)))
        for interface in domain.interfaces:
            if interface.radius:continue
            if interface.axis==0:p1=occ.addPoint(interface.position,y0,0);p2=occ.addPoint(interface.position,y1,0)
            else:p1=occ.addPoint(x0,interface.position,0);p2=occ.addPoint(x1,interface.position,0)
            tools.append((1,occ.addLine(p1,p2)))
        entities,mapping=occ.fragment([(2,outer)],tools)
        occ.synchronize()
        all_faces={tag for dim,tag in mapping[0] if dim==2};assigned=set();region_faces={}
        for r in domain.regions[domain.nsoil:]:
            outer_faces={tag for dim,tag in mapping[1+disk_indices[(r.cable,r.layer)]] if dim==2}
            inner_faces={tag for dim,tag in mapping[1+disk_indices[(r.cable,r.layer-1)]] if dim==2} if r.layer else set()
            faces=outer_faces-inner_faces
            if not faces:raise RuntimeError(f'Missing material geometry: {r.name}')
            region_faces[r.id]=faces;assigned|=faces
        soil_faces=all_faces-assigned
        for r in domain.regions[:domain.nsoil]:
            faces={tag for tag in soil_faces if r.lower<=occ.getCenterOfMass(2,tag)[r.axis]<=r.upper}
            region_faces[r.id]=faces
        if set.union(*region_faces.values())!=all_faces or sum(len(v) for v in region_faces.values())!=len(all_faces):
            raise RuntimeError('Material partition does not cover the domain exactly once')
        fields=[];interface_curves=set()
        far=min(x1-x0,y1-y0)/10/2**level
        for r in domain.regions:
            faces=region_faces[r.id]
            if not faces:raise RuntimeError('Empty material region')
            gmsh.model.addPhysicalGroup(2,sorted(faces),r.id+1);gmsh.model.setPhysicalName(2,r.id+1,r.name)
            if r.kind!='soil':
                curves={tag for dim,tag in gmsh.model.getBoundary([(2,t) for t in faces],combined=True,oriented=False) if dim==1}
                interface_curves|=curves
                distance=gmsh.model.mesh.field.add('Distance')
                gmsh.model.mesh.field.setNumbers(distance,'CurvesList',sorted(curves));gmsh.model.mesh.field.setNumber(distance,'Sampling',256)
                threshold=gmsh.model.mesh.field.add('Threshold')
                local=(r.ro-r.ri)/max(2,2**(level+1))
                for key,value in [('InField',distance),('SizeMin',local),('SizeMax',far),('DistMin',(r.ro-r.ri)*.6),('DistMax',max(.1,4*r.ro))]:
                    gmsh.model.mesh.field.setNumber(threshold,key,value)
                fields.append(threshold)
        boundary=gmsh.model.getBoundary([(2,t) for t in all_faces],combined=True,oriented=False)
        outer_curves=[tag for dim,tag in boundary if dim==1]
        gmsh.model.addPhysicalGroup(1,outer_curves,1000)
        for j,inter in enumerate(domain.interfaces):
            left=set(tag for dim,tag in gmsh.model.getBoundary([(2,t) for t in region_faces[inter.left]],combined=True,oriented=False) if dim==1)
            right=set(tag for dim,tag in gmsh.model.getBoundary([(2,t) for t in region_faces[inter.right]],combined=True,oriented=False) if dim==1)
            common=left&right
            if not common:raise RuntimeError(f'Interface topology missing: {inter.name}')
            gmsh.model.addPhysicalGroup(1,sorted(common),2000+j)
        if fields:
            minimum=gmsh.model.mesh.field.add('Min');gmsh.model.mesh.field.setNumbers(minimum,'FieldsList',fields);gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0);gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0);gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.model.mesh.generate(2);gmsh.model.mesh.setOrder(2)
        data=gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,gdim=2)
        return data.mesh,data.cell_tags,data.facet_tags
    finally:gmsh.finalize()


def evaluate_material(solution,xy,labels,cell_tags):
    """Choose the correct material trace on curved cells before interpolation.

    A geometric collision alone can select the other side of a curved interface.
    Inverse isoparametric coordinates determine which candidate contains a point.
    """
    mesh=solution.function_space.mesh;points=np.c_[xy,np.zeros(len(xy))]
    tree=geometry.bb_tree(mesh,2,padding=1e-8)
    candidates=geometry.compute_collisions_points(tree,points)
    tag=np.zeros(mesh.topology.index_map(2).size_local,dtype=np.int32)
    tag[cell_tags.indices]=cell_tags.values
    chosen=[]
    for i,point in enumerate(points):
        options=[cell for cell in candidates.links(i) if tag[cell]==labels[i]+1]
        if not options:raise RuntimeError(f'No {labels[i]} material cell for evaluation point {point}')
        scores=[]
        for cell in options:
            nodes=mesh.geometry.x[mesh.geometry.dofmap[cell]]
            xi=mesh.geometry.cmap.pull_back(point.reshape(1,3),nodes)[0]
            scores.append(min(xi[0],xi[1],1-xi.sum()))
        best=int(np.argmax(scores))
        if scores[best]<-1e-3:raise RuntimeError('Evaluation outside the declared material mesh')
        chosen.append(options[best])
    return solution.eval(points,np.asarray(chosen,dtype=np.int32)).ravel()


def solve(domain,level,output):
    start=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stem=f'fem_l{level}'
    if (output/(stem+'.json')).exists():raise FileExistsError('Use a new directory for a new reference run')
    mesh,cells,facets=create_mesh(domain,level);c=domain.case
    V=fem.functionspace(mesh,('Lagrange',2));u=ufl.TrialFunction(V);v=ufl.TestFunction(V);x=ufl.SpatialCoordinate(mesh)
    dx=ufl.Measure('dx',domain=mesh,subdomain_data=cells,metadata={'quadrature_degree':8})
    ds=ufl.Measure('ds',domain=mesh,subdomain_data=facets,metadata={'quadrature_degree':8})
    # Each material has a separate integration domain; no averaging of layer k.
    kvals={r.id:(field_k(c,x[0],x[1],UFLBackend) if r.kind=='soil' else r.k) for r in domain.regions}
    a=sum(kvals[r.id]*ufl.inner(ufl.grad(u),ufl.grad(v))*dx(r.id+1) for r in domain.regions)
    old=fem.Function(V);old.x.array[:]=c['T0'];boundary=fem.Function(V)
    boundary.interpolate(lambda points:c['T0']+c.get('outer_gradient_K_m',0.)*points[0])
    bd=fem.locate_dofs_topological(V,1,facets.find(1000));bc=fem.dirichletbc(boundary,bd)
    electric={r.id:fem.Constant(mesh,0.) for r in domain.conductors}
    sources={}
    for r in domain.conductors:
        if domain.mode=='dc_temperature':
            spec=domain.electrical
            rho20=spec['resistance_ohm_m']*domain.area(r)
            sources[r.id]=electric[r.id]**2/(rho20*(1+c['alpha']*(old-spec['reference_temperature_C'])))
        else:
            # Interpolate only a prescribed source, never a temperature solution.
            source=fem.Function(V)
            source.interpolate(lambda points,r=r:domain.source(r,points[:2].T).ravel())
            sources[r.id]=source
    L=sum(sources[r.id]*v*dx(r.id+1) for r in domain.conductors)
    problem=LinearProblem(a,L,bcs=[bc],petsc_options_prefix=f'explicit_{c["id"]}_{level}_',
        petsc_options={'ksp_type':'preonly','pc_type':'lu','pc_factor_mat_solver_type':'mumps','ksp_error_if_not_converged':True})
    history=[]
    for iteration in range(100):
        if domain.mode=='dc_temperature':
            for r in domain.conductors:
                spec=domain.electrical
                sigma=1/(spec['resistance_ohm_m']*domain.area(r)*(1+c['alpha']*(old-spec['reference_temperature_C'])))
                conductance=fem.assemble_scalar(fem.form(sigma*dx(r.id+1)))
                if conductance<=0:raise RuntimeError('Invalid electrical conductance')
                electric[r.id].value=c['current']/conductance
        solution=problem.solve();solution.x.scatter_forward()
        delta=float(np.max(abs(solution.x.array-old.x.array)));history.append(dict(iteration=iteration,temperature_change_K=delta))
        if domain.mode=='fixed' or delta<1e-7:break
        old.x.array[:]=solution.x.array
    else:raise RuntimeError('Full-domain electrothermal iteration did not converge')
    xy,labels,weights=domain.evaluation_points();values=evaluate_material(solution,xy,labels,cells)
    conductor_max=[];regions=[];power=0.
    for r in domain.regions:
        area=fem.assemble_scalar(fem.form(1*dx(r.id+1)))
        entry=dict(name=r.name,area_m2=area,exact_area_m2=domain.area(r),area_error_pct=100*abs(area/domain.area(r)-1))
        if r.kind=='conductor':
            p=fem.assemble_scalar(fem.form(sources[r.id]*dx(r.id+1)));power+=p
            # Interior FEM dofs plus independent dense disk samples include the origin.
            dofxy=V.tabulate_dof_coordinates()[:,:2];mask=domain.contains(r,dofxy)
            maximum=max(float(solution.x.array[mask].max()),float(values[labels==r.id].max()))
            conductor_max.append(maximum);entry.update(Tmax_C=maximum,power_W_m=p)
        regions.append(entry)
    normal=ufl.FacetNormal(mesh)
    flux=fem.assemble_scalar(fem.form(-field_k(c,x[0],x[1],UFLBackend)*ufl.dot(ufl.grad(solution),normal)*ds(1000)))
    meta=dict(physics_version=PHYSICS_VERSION,physics_sha256=domain.fingerprint,specification=domain.specification(),
        method='explicit_multimaterial_FEniCSx',level=level,ndofs=V.dofmap.index_map.size_global,degree=2,
        regions=regions,Tmax_C=max(conductor_max),conductor_max_C=conductor_max,
        source_W_m=power,outer_flux_W_m=flux,balance_pct=100*abs(flux-power)/power,
        electrothermal_history=history,elapsed_s=time.perf_counter()-start,
        dolfinx=dolfinx.__version__,gmsh=gmsh.__version__,completed_utc=datetime.now(timezone.utc).isoformat())
    if c.get('outer_radius') and not c.get('outer_gradient_K_m',0) and domain.mode=='fixed' and not c.get('electrical'):
        exact=radial_exact(domain,np.linalg.norm(xy,axis=1));meta['rmse_exact_K']=float(np.sqrt(np.average((values-exact)**2,weights=weights)))
        meta['error_Tmax_exact_K']=abs(meta['Tmax_C']-float(radial_exact(domain,np.array([0.]))[0]))
    np.savez_compressed(output/(stem+'.npz'),xy=xy,region=labels,weights=weights,T=values,
        dof_xy=V.tabulate_dof_coordinates()[:,:2],dof_T=solution.x.array.copy())
    with XDMFFile(mesh.comm,str(output/(stem+'.xdmf')),'w') as file:
        file.write_mesh(mesh);file.write_meshtags(cells,mesh.geometry);file.write_function(solution)
    files=[Path(__file__),Path(__file__).with_name('full_domain.py'),Path(__file__).with_name('skin_effect.py'),Path(__file__).with_name('cases.py'),Path(__file__).with_name('expressions.py'),Path(__file__).with_name('fem.py')]
    archive=output/'source';archive.mkdir(exist_ok=True);meta['source_sha256']={}
    for path in files:
        content=path.read_bytes();digest=hashlib.sha256(content).hexdigest();meta['source_sha256'][str(path.relative_to(ROOT))]=digest
        (archive/(digest+'_'+path.name)).write_bytes(content)
    (output/(stem+'.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['level','ndofs','Tmax_C','balance_pct','elapsed_s']}),flush=True)
    return meta


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--case',default='coaxial_full');ap.add_argument('--case-file',type=Path)
    ap.add_argument('--mode',choices=['fixed','dc_temperature'],default='fixed');ap.add_argument('--levels',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();case=json.loads(a.case_file.read_text()) if a.case_file else (coaxial_case() if a.case=='coaxial_full' else cases()[a.case])
    domain=FullDomain(case,a.mode)
    for level in a.levels:solve(domain,level,a.output)


if __name__=='__main__':main()
