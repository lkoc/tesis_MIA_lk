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

from pinn_v2.geometry.domain import FullDomain,coaxial_case,dc_conductivity,radial_exact,PHYSICS_VERSION
from pinn_v2.cases.boundary import RECT_SIDES,SIDE_TAGS
from pinn_v2.cases.definition import CaseDefinition,load_all
from pinn_v2.cases.catalog import cases,field_k
from pinn_v2.physics.skin_effect import prescribed_power
from pinn_v2.solvers.fem.backend import UFLBackend
from pinn_v2.manifest.provenance import archive_sources


NO_CONTACT='Imperfect thermal contact ([T]=R q.n) is not implemented in the FEM reference; it is never treated as perfect contact'


def check_corner_consistency(domain,conditions):
    """Two adjacent Dirichlet sides with different values would make the corner temperature ambiguous."""
    if domain.bcs is None:return
    for first,second in (('bottom','left'),('bottom','right'),('top','left'),('top','right')):
        a,b=conditions.get(first),conditions.get(second)
        if a and b and a['type']==b['type']=='dirichlet' and not math.isclose(a['T_C'],b['T_C']):
            raise ValueError(f'Dirichlet values differ at the {first}/{second} corner')


def create_mesh(domain,level):
    if MPI.COMM_WORLD.size!=1:raise ValueError('Reference currently requires one MPI rank; do not silently assemble partial balances')
    if domain.has_imperfect_contact:raise NotImplementedError(NO_CONTACT)
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
        if domain.bcs is None or c.get('outer_radius'):gmsh.model.addPhysicalGroup(1,outer_curves,1000)
        else:
            sides={s:[] for s in RECT_SIDES};tol=1e-9*max(x1-x0,y1-y0)
            for tag in outer_curves:
                mx,my,_=occ.getCenterOfMass(1,tag)
                found=[s for s,coordinate,value in (('bottom',my,y0),('top',my,y1),('left',mx,x0),('right',mx,x1)) if abs(coordinate-value)<tol]
                if len(found)!=1:raise RuntimeError('Outer curve does not lie on exactly one rectangular side')
                sides[found[0]].append(tag)
            for side,curves in sides.items():
                if not curves:raise RuntimeError(f'No boundary curves on side {side}')
                gmsh.model.addPhysicalGroup(1,sorted(curves),SIDE_TAGS[side])
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
    chosen=[];fallback_tree=None

    def interior_candidates(point,cells,label):
        scores=[]
        for cell in cells:
            if tag[cell]!=label+1:continue
            nodes=mesh.geometry.x[mesh.geometry.dofmap[cell]]
            try:xi=mesh.geometry.cmap.pull_back(point.reshape(1,3),nodes)[0]
            except RuntimeError:continue
            score=min(xi[0],xi[1],1-xi.sum())
            if score>=-1e-3:scores.append((score,int(cell)))
        return scores

    for i,point in enumerate(points):
        options=set(candidates.links(i))
        scores=interior_candidates(point,options,labels[i])
        if not scores:
            # Curved P2 edges can extend beyond the linear cell's bounding box.
            # Broaden only the search, never the material or inverse-map test.
            if fallback_tree is None:fallback_tree=geometry.bb_tree(mesh,2,padding=1e-4)
            nearby=geometry.compute_collisions_points(fallback_tree,point.reshape(1,3))
            options.update(nearby.links(0))
            scores=interior_candidates(point,options,labels[i])
        if not scores:raise RuntimeError(f'No mappable {labels[i]} material cell for evaluation point {point}')
        chosen.append(max(scores,key=lambda item:item[0])[1])
    return solution.eval(points,np.asarray(chosen,dtype=np.int32)).ravel()


def solve(domain,level,output,normalize_source=False,save_fields=True):
    if domain.has_imperfect_contact:raise NotImplementedError(NO_CONTACT)
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
    old=fem.Function(V);old.x.array[:]=c['T0']
    conditions=domain.side_conditions();check_corner_consistency(domain,conditions)
    tag_of=lambda side:1000 if domain.bcs is None else SIDE_TAGS[side]
    dirichlet=[s for s,b in conditions.items() if b['type']=='dirichlet']
    if domain.bcs is None:dirichlet=dirichlet[:1]
    bcs=[];L_boundary=0
    for side in dirichlet:
        values=fem.Function(V);values.interpolate(lambda points,side=side:domain.dirichlet_value(side,points[:2].T).ravel())
        bcs.append(fem.dirichletbc(values,fem.locate_dofs_topological(V,1,facets.find(tag_of(side)))))
    for side,condition in conditions.items():
        tag=tag_of(side)
        if condition['type']=='robin':
            a=a+condition['h_W_m2K']*u*v*ds(tag);L_boundary=L_boundary+condition['h_W_m2K']*condition['T_inf_C']*v*ds(tag)
        elif condition['type']=='neumann':
            L_boundary=L_boundary-condition['q_n_W_m2']*v*ds(tag)
    electric={r.id:fem.Constant(mesh,0.) for r in domain.conductors}
    sources={};source_factors={}
    for r in domain.conductors:
        if domain.mode=='dc_temperature':
            spec=domain.electrical
            rho20=spec['resistance_ohm_m']*domain.area(r)
            sources[r.id]=electric[r.id]**2/(rho20*(1+c['alpha']*(old-spec['reference_temperature_C'])))
        else:
            # Interpolate only a prescribed source, never a temperature solution.
            source=fem.Function(V)
            source.interpolate(lambda points,r=r:domain.source(r,points[:2].T).ravel())
            if normalize_source:
                integrated=fem.assemble_scalar(fem.form(source*dx(r.id+1)))
                factor=prescribed_power(c)/integrated
                source.x.array[:]*=factor;source_factors[r.name]=factor
            sources[r.id]=source
    L=sum(sources[r.id]*v*dx(r.id+1) for r in domain.conductors)+L_boundary
    problem=LinearProblem(a,L,bcs=bcs,petsc_options_prefix=f'explicit_{c["id"]}_{level}_',
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
            candidate_xy=np.vstack([dofxy[mask],xy[labels==r.id]]);candidate_T=np.r_[solution.x.array[mask],values[labels==r.id]]
            best=int(np.argmax(candidate_T));maximum=float(candidate_T[best])
            conductor_max.append(maximum)
            entry.update(Tmax_C=maximum,hotspot_xy_m=candidate_xy[best].tolist(),
                T_mean_C=float(fem.assemble_scalar(fem.form(solution*dx(r.id+1))))/area,power_W_m=p)
        regions.append(entry)
    normal=ufl.FacetNormal(mesh);kk=field_k(c,x[0],x[1],UFLBackend)
    flux_tags={'boundary':1000} if domain.bcs is None else {s:SIDE_TAGS[s] for s in conditions}
    side_flux={name:float(fem.assemble_scalar(fem.form(-kk*ufl.dot(ufl.grad(solution),normal)*ds(tag)))) for name,tag in flux_tags.items()}
    flux=sum(side_flux.values());boundary_check={}
    psi=fem.Function(V);psi.x.array[:]=0.
    for side in dirichlet:psi.x.array[fem.locate_dofs_topological(V,1,facets.find(tag_of(side)))]=1.
    reaction=-float(fem.assemble_scalar(fem.form(ufl.replace(a,{u:solution,v:psi})-ufl.replace(L,{v:psi}))))
    constitutive=sum(float(fem.assemble_scalar(fem.form(b_['h_W_m2K']*(solution-b_['T_inf_C'])*ds(tag_of(s))))) for s,b_ in conditions.items() if b_['type']=='robin')
    prescribed=sum(float(fem.assemble_scalar(fem.form(b_['q_n_W_m2']*1*ds(tag_of(s))))) for s,b_ in conditions.items() if b_['type']=='neumann')
    consistent_flux=reaction+constitutive+prescribed
    if domain.bcs is not None:
        for side,condition in conditions.items():
            length=float(fem.assemble_scalar(fem.form(1*ds(SIDE_TAGS[side]))))
            if condition['type']=='robin':
                outflow=float(fem.assemble_scalar(fem.form(condition['h_W_m2K']*(solution-condition['T_inf_C'])*ds(SIDE_TAGS[side]))))
                boundary_check[side]=dict(type='robin',length_m=length,constitutive_outflow_W_m=outflow,traction_outflow_W_m=side_flux[side])
            elif condition['type']=='neumann':
                boundary_check[side]=dict(type='neumann',length_m=length,prescribed_outflow_W_m=condition['q_n_W_m2']*length,traction_outflow_W_m=side_flux[side])
    meta=dict(physics_version=PHYSICS_VERSION,physics_sha256=domain.fingerprint,specification=domain.specification(),
        method='explicit_multimaterial_FEniCSx',level=level,ndofs=V.dofmap.index_map.size_global,degree=2,
        regions=regions,Tmax_C=max(conductor_max),conductor_max_C=conductor_max,
        source_W_m=power,outer_flux_W_m=flux,boundary_flux_W_m=side_flux,boundary_conditions=conditions,boundary_check=boundary_check,
        balance_pct=100*abs(flux-power)/power,
        dirichlet_reaction_W_m=reaction,consistent_outflow_W_m=consistent_flux,balance_consistent_pct=100*abs(consistent_flux-power)/power,
        source_normalization_factors=source_factors,
        electrothermal_history=history,elapsed_s=time.perf_counter()-start,
        dolfinx=dolfinx.__version__,gmsh=gmsh.__version__,completed_utc=datetime.now(timezone.utc).isoformat())
    if c.get('outer_radius') and not c.get('outer_gradient_K_m',0) and domain.mode=='fixed' and not c.get('electrical'):
        exact=radial_exact(domain,np.linalg.norm(xy,axis=1));meta['rmse_exact_K']=float(np.sqrt(np.average((values-exact)**2,weights=weights)))
        meta['error_Tmax_exact_K']=abs(meta['Tmax_C']-float(radial_exact(domain,np.array([0.]))[0]))
    if save_fields:
        np.savez_compressed(output/(stem+'.npz'),xy=xy,region=labels,weights=weights,T=values,
            dof_xy=V.tabulate_dof_coordinates()[:,:2],dof_T=solution.x.array.copy())
        with XDMFFile(mesh.comm,str(output/(stem+'.xdmf')),'w') as file:
            file.write_mesh(mesh);file.write_meshtags(cells,mesh.geometry);file.write_function(solution)
    meta['source_sha256']=archive_sources(output)
    (output/(stem+'.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['level','ndofs','Tmax_C','balance_pct','elapsed_s']}),flush=True)
    return meta


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--case',default='coaxial_full');ap.add_argument('--case-file',type=Path)
    ap.add_argument('--definition',type=Path,help='CaseDefinition v2 JSON file')
    ap.add_argument('--mode',choices=['fixed','dc_temperature'],default='fixed');ap.add_argument('--levels',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.definition:domain=CaseDefinition.load(a.definition).domain(a.mode)
    elif a.case_file:domain=FullDomain(json.loads(a.case_file.read_text()),a.mode)
    elif a.case=='coaxial_full':domain=FullDomain(coaxial_case(),a.mode)
    else:domain=load_all()[a.case].domain(a.mode)
    for level in a.levels:solve(domain,level,a.output)


if __name__=='__main__':main()
