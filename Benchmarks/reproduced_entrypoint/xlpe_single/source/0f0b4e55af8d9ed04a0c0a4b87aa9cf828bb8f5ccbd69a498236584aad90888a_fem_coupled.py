"""FEniCSx thermal response matrix with per-conductor R(T), nominal and ampacity."""
from pathlib import Path
import argparse,json,sys,time,hashlib,platform
from datetime import datetime,timezone
import numpy as np
import dolfinx,ufl,gmsh
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.fem import create_mesh,evaluate,UFLBackend,archive_previous
from Benchmarks.cases import cases,field_k,surface_points,evaluation_points,radial_resistance,fingerprint
from Benchmarks.electrothermal import solve_powers,ampacity,LOSS_MODEL

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT={str(p.relative_to(ROOT)):p.read_bytes() for p in [Path(__file__),Path(__file__).with_name('fem.py'),Path(__file__).with_name('cases.py'),Path(__file__).with_name('expressions.py'),Path(__file__).with_name('electrothermal.py'),Path(__file__).with_name('electrical_model.json')]}

def solve(c,level,out):
    start=time.perf_counter();msh,tags=create_mesh(c,level)
    V=fem.functionspace(msh,('Lagrange',2));u=ufl.TrialFunction(V);v=ufl.TestFunction(V);x=ufl.SpatialCoordinate(msh)
    dx=ufl.Measure('dx',domain=msh,metadata={'quadrature_degree':8});ds=ufl.Measure('ds',domain=msh,subdomain_data=tags,metadata={'quadrature_degree':8})
    k=field_k(c,x[0],x[1],UFLBackend);a=ufl.inner(k*ufl.grad(u),ufl.grad(v))*dx
    n=len(c['cables']);coeff=[fem.Constant(msh,0.) for _ in range(n)]
    L=sum(coeff[j]/(2*np.pi*c['radius'])*v*ds(10+j) for j in range(n))
    ub=fem.Function(V);ub.x.array[:]=c['T0'];bd=fem.locate_dofs_topological(V,1,tags.find(2))
    problem=LinearProblem(a,L,bcs=[fem.dirichletbc(ub,bd)],petsc_options_prefix=f'rt_{c["id"]}_{level}_',petsc_options={'ksp_type':'preonly','pc_type':'lu','pc_factor_mat_solver_type':'mumps','ksp_error_if_not_converged':True})
    points=evaluation_points(c);surfaces=surface_points(c,offset=max(1e-5,c['radius']*.002));radial=radial_resistance(c)
    response=np.zeros((n,n));field_response=[]
    def run(powers):
        for q,value in zip(coeff,powers):q.value=float(value)
        uh=problem.solve();uh.x.scatter_forward()
        if problem.solver.getConvergedReason()<=0:raise RuntimeError('FEM solve failed')
        return uh
    for j in range(n):
        power=np.zeros(n);power[j]=1.;uh=run(power)
        response[:,j]=[evaluate(uh,p).mean()-c['T0']+(radial if i==j else 0) for i,p in enumerate(surfaces)]
        field_response.append(evaluate(uh,points)-c['T0'])
    out.mkdir(parents=True,exist_ok=True)
    for mode in ['nominal','ampacity']:
        if mode=='nominal':current=c['current'];powers,trace=solve_powers(c,response)
        else:current,powers,trace=ampacity(c,response)
        uh=run(powers);vals=evaluate(uh,points);sv=np.array([evaluate(uh,p) for p in surfaces]);tc=sv.mean(axis=1)+powers*radial
        normal=ufl.FacetNormal(msh);flux=fem.assemble_scalar(fem.form(-k*ufl.dot(ufl.grad(uh),normal)*ds));throughput=fem.assemble_scalar(fem.form(abs(k*ufl.dot(ufl.grad(uh),normal))*ds))
        target=current**2*c['R20']*(1+c['alpha']*(tc-20.))
        meta=dict(case=c,case_sha256=fingerprint(c),method='FEniCSx',coupled=True,ampacity=mode=='ampacity',loss_model=LOSS_MODEL,current_A=current,powers_W_m=powers.tolist(),resistance_ohm_m=(powers/current**2).tolist(),electrical_residual_pct=float(max(abs(powers-target)/target)*100),conductor_C=tc.tolist(),Tmax_C=float(max(tc)),balance_pct=abs(flux)/max(throughput/2,1e-12)*100,net_flux_W_m=flux,source_W_m=0.,throughput_W_m=throughput,level=level,degree=2,ndofs=V.dofmap.index_map.size_global,response_K_m_W=response.tolist(),trace=trace,linear_solves_for_both_modes=n+2,elapsed_s=time.perf_counter()-start,dolfinx=dolfinx.__version__,gmsh=gmsh.__version__,python=sys.version,platform=platform.platform(),completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256={p:hashlib.sha256(b).hexdigest() for p,b in SNAPSHOT.items()})
        meta['boundary_diagnostics']=[]
        for j,power in enumerate(powers):
            measured=-fem.assemble_scalar(fem.form(-k*ufl.dot(ufl.grad(uh),normal)*ds(10+j)))
            meta['boundary_diagnostics'].append(dict(boundary=f'cable{j}',prescribed_heat_into_soil_W_m=float(power),computed_heat_into_soil_W_m=measured,relative_power_error_pct=100*abs(measured/power-1)))
        stem=f'fem_{"ampacity_" if mode=="ampacity" else ""}l{level}'
        archive_previous(out,stem)
        (out/(stem+'.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
        np.savez_compressed(out/(stem+'.npz'),xy=points,T=vals,surface_xy=np.array(surfaces),surface_T=sv,dof_xy=V.tabulate_dof_coordinates()[:,:2],dof_T=uh.x.array,response=response,field_response=np.array(field_response))
        print(json.dumps(dict(case=c['id'],level=level,mode=mode,current=current,Tmax=float(max(tc)),balance=meta['balance_pct'],electrical=meta['electrical_residual_pct'])),flush=True)
    for name,data in SNAPSHOT.items():
        path=out/'source'/(hashlib.sha256(data).hexdigest()+'_'+Path(name).name);path.parent.mkdir(exist_ok=True)
        if not path.exists():path.write_bytes(data)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+',default=['all']);ap.add_argument('--levels',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--output',type=Path,default=ROOT/'Benchmarks/coupled_results');args=ap.parse_args()
    for name,c in cases().items():
        if c['kind']=='cable' and (args.cases==['all'] or name in args.cases):
            for level in args.levels:solve(c,level,args.output/name)
