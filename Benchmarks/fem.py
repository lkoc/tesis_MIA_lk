"""Referencia independiente FEniCSx 0.10 / Gmsh. Ejecutar en Ubuntu/WSL."""
from pathlib import Path
import argparse
import json
import math
import platform
import sys
import time
import numpy as np
import dolfinx
from dolfinx import fem, geometry, mesh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import gmsh
import ufl

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases, field_k, source, exact, evaluation_points, surface_points, radial_resistance, fingerprint, interfaces

class UFLBackend:
    sin=staticmethod(ufl.sin); cos=staticmethod(ufl.cos); tanh=staticmethod(ufl.tanh)
    log=staticmethod(ufl.ln); sqrt=staticmethod(ufl.sqrt); exp=staticmethod(ufl.exp)
    @staticmethod
    def where(condition,a,b): return ufl.conditional(condition,a,b)

def create_mesh(c,level):
    if c['kind'].startswith('mms'):
        # x=.5 is aligned for the discontinuous two-material case.
        return mesh.create_unit_square(MPI.COMM_WORLD,16*2**level,16*2**level),None
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add(c['id'])
        occ=gmsh.model.occ
        if c['kind']=='annulus':
            outer=occ.addDisk(0,0,0,c['ro'],c['ro'])
            holes=[occ.addDisk(0,0,0,c['ri'],c['ri'])]
            centers=[[0,0]];r=c['ri'];far=3.
        else:
            x0,x1,y0,y1=c['bounds'];outer=occ.addRectangle(x0,y0,0,x1-x0,y1-y0)
            centers=c['cables'];r=c['radius'];far=.65
            holes=[occ.addDisk(cx,cy,0,r,r) for cx,cy in centers]
        surf,_=occ.cut([(2,outer)],[(2,t) for t in holes])
        cuts=[]
        for info in interfaces(c):
            pos=info['position']
            if info['axis']=='y':a=occ.addPoint(x0,pos,0);b=occ.addPoint(x1,pos,0)
            else:a=occ.addPoint(pos,y0,0);b=occ.addPoint(pos,y1,0)
            cuts.append((1,occ.addLine(a,b)))
        if cuts:
            fragments,_=occ.fragment(surf,cuts);surf=[s for s in fragments if s[0]==2]
        occ.synchronize()
        gmsh.model.addPhysicalGroup(2,[s[1] for s in surf],1)
        outer_curves=[];inner_curves=[]
        for dim,t in gmsh.model.getBoundary(surf,combined=True,oriented=False):
            box=gmsh.model.getBoundingBox(dim,t)
            mid=np.array([(box[0]+box[3])/2,(box[1]+box[4])/2])
            chosen=next((j for j,cc in enumerate(centers) if np.linalg.norm(mid-cc)<r*.1 and box[3]-box[0]<r*2.1),None)
            if chosen is None: outer_curves.append(t)
            else:
                gmsh.model.addPhysicalGroup(1,[t],10+chosen);inner_curves.append(t)
        gmsh.model.addPhysicalGroup(1,outer_curves,2)
        f=gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(f,'CurvesList',inner_curves)
        gmsh.model.mesh.field.setNumber(f,'Sampling',160)
        th=gmsh.model.mesh.field.add('Threshold')
        for key,val in [('InField',f),('SizeMin',r*.65/2**level),('SizeMax',far/2**level),('DistMin',r*.4),('DistMax',3. if c['kind']=='annulus' else 1.)]:
            gmsh.model.mesh.field.setNumber(th,key,val)
        gmsh.model.mesh.field.setAsBackgroundMesh(th)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.model.mesh.generate(2);gmsh.model.mesh.setOrder(2)
        data=gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,gdim=2)
        return data.mesh,data.facet_tags
    finally:
        gmsh.finalize()

def evaluate(uh,xy):
    pts=np.c_[xy,np.zeros(len(xy))]
    tree=geometry.bb_tree(uh.function_space.mesh,2)
    candidates=geometry.compute_collisions_points(tree,pts)
    coll=geometry.compute_colliding_cells(uh.function_space.mesh,candidates,pts)
    cells=np.array([coll.links(i)[0] if len(coll.links(i)) else -1 for i in range(len(xy))],dtype=np.int32)
    if np.any(cells<0): raise ValueError(f'{sum(cells<0)} evaluation points outside FEM mesh')
    return uh.eval(pts,cells).reshape(-1)

def solve(c,level,out):
    start=time.perf_counter();msh,tags=create_mesh(c,level)
    V=fem.functionspace(msh,('Lagrange',2))
    u=ufl.TrialFunction(V);v=ufl.TestFunction(V);x=ufl.SpatialCoordinate(msh)
    k=field_k(c,x[0],x[1],UFLBackend);q=source(c,x[0],x[1],UFLBackend)
    dx=ufl.Measure('dx',domain=msh,metadata={'quadrature_degree':8})
    if tags is None:
        boundary=mesh.locate_entities_boundary(msh,1,lambda x: np.full(x.shape[1],True))
        values=np.ones(len(boundary),dtype=np.int32)
        mid=mesh.compute_midpoints(msh,1,boundary)
        values[np.isclose(mid[:,1],1.)]=3
        order=np.argsort(boundary);tags=mesh.meshtags(msh,1,boundary[order],values[order])
    ds=ufl.Measure('ds',domain=msh,subdomain_data=tags,metadata={'quadrature_degree':8})
    a=ufl.inner(k*ufl.grad(u),ufl.grad(v))*dx
    L=q*v*dx
    ub=fem.Function(V)
    if c['kind'].startswith('mms'):
        ub.interpolate(lambda z: exact(c,z[0],z[1]))
        if c['kind']=='mms_robin':
            facets=tags.find(1)
            tinf=exact(c,x[0],x[1],UFLBackend)+3*x[0]*(1-x[0])
            a+=10*u*v*ds(3);L+=10*tinf*v*ds(3)
        else: facets=np.r_[tags.find(1),tags.find(3)]
    else:
        ub.x.array[:]=c['T0'];facets=tags.find(2)
        n_holes=1 if c['kind']=='annulus' else len(c['cables'])
        r=c.get('ri',c.get('radius'))
        for j in range(n_holes): L+=c['power']/(2*math.pi*r)*v*ds(10+j)
    dofs=fem.locate_dofs_topological(V,1,facets)
    bc=fem.dirichletbc(ub,dofs)
    problem=LinearProblem(a,L,bcs=[bc],petsc_options_prefix=f'b_{c["id"]}_{level}_',petsc_options={'ksp_type':'preonly','pc_type':'lu','pc_factor_mat_solver_type':'mumps','ksp_error_if_not_converged':True})
    uh=problem.solve();uh.x.scatter_forward()
    if problem.solver.getConvergedReason()<=0: raise RuntimeError('PETSc did not converge')
    xy=evaluation_points(c);values=evaluate(uh,xy)
    surfaces=surface_points(c,offset=max(1e-5,c.get('radius',.06)*.002))
    tv=[evaluate(uh,s) for s in surfaces]
    tmax=float(max(np.max(t) for t in tv)) if tv else float(np.max(uh.x.array))
    tc=[]
    if c['kind']=='cable':
        tc=[float(np.mean(t)+c['power']*radial_resistance(c)) for t in tv]
        tmax=max(tc)
    n=ufl.FacetNormal(msh)
    flux=fem.assemble_scalar(fem.form(-k*ufl.dot(ufl.grad(uh),n)*ds))
    generation=fem.assemble_scalar(fem.form(q*dx))
    throughput=fem.assemble_scalar(fem.form(abs(k*ufl.dot(ufl.grad(uh),n))*ds))
    balance=abs(flux-generation)/max(abs(generation),throughput/2,1e-12)*100
    meta=dict(case=c,case_sha256=fingerprint(c),method='FEniCSx',dolfinx=dolfinx.__version__,gmsh=gmsh.__version__,python=sys.version,platform=platform.platform(),level=level,degree=2,ndofs=V.dofmap.index_map.size_global,Tmax_C=tmax,conductor_C=tc,net_flux_W_m=flux,source_W_m=generation,throughput_W_m=throughput,balance_pct=balance,elapsed_s=time.perf_counter()-start,petsc_reason=int(problem.solver.getConvergedReason()))
    if c['kind']!='cable':
        error=values-exact(c,xy[:,0],xy[:,1]);meta.update(rmse_exact_K=float(np.sqrt(np.mean(error**2))),max_error_exact_K=float(np.max(abs(error))))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'fem_l{level}.npz',xy=xy,T=values,surface_xy=np.array(surfaces),surface_T=np.array(tv),dof_xy=V.tabulate_dof_coordinates()[:,:2],dof_T=uh.x.array)
    (out/f'fem_l{level}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['level','ndofs','Tmax_C','balance_pct','elapsed_s']},ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+',default=['all']);ap.add_argument('--levels',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--output',default='Benchmarks/results');args=ap.parse_args()
    catalog=cases();names=list(catalog) if args.cases==['all'] else args.cases
    for name in names:
        for level in args.levels:
            print(f'FEM {name} level {level}',flush=True)
            solve(catalog[name],level,Path(args.output)/name)

if __name__=='__main__': main()
