"""Regresiones de errores físicos detectados durante la auditoría de tesis."""
import torch
from pinn_cables.io.readers import BoundaryCondition
from pinn_cables.pinn.pde import laplace_variable_k, neumann_residual
from pinn_cables.pinn.train_custom import compute_pde_bc_loss

def test_affine_field_has_zero_laplacian():
    xy=torch.rand(11,2,requires_grad=True)
    T=20+2*xy[:,:1]+3*xy[:,1:2]
    assert torch.max(abs(laplace_variable_k(T,xy,2.)))==0

def test_neumann_prescribes_heat_flux_not_temperature_gradient():
    xy=torch.rand(11,2,requires_grad=True)
    T=20+3*xy[:,1:2]
    residual=neumann_residual(T,xy,torch.tensor([[0.,1.]]),-6.,2.)
    assert torch.max(abs(residual))==0

def test_custom_robin_accepts_nonzero_surface_temperature_difference():
    # T=30-y, k=2, h=2; at y=1, flux=2 and T-T_inf=1.
    class Exact(torch.nn.Module):
        def forward(self,xy): return 30-xy[:,1:2]+0*xy[:,:1]**2
    points=torch.stack([torch.linspace(0,1,17),torch.ones(17)],dim=1)
    bc={'top':BoundaryCondition('top','robin',28.,2.)}
    total,_,boundary=compute_pde_bc_loss(Exact(),torch.rand(31,2),{'top':points},bc,20.,lambda x:x,False,2.,1.,1.)
    assert boundary<1e-10
    assert total<1e-10

def test_custom_neumann_is_not_ignored():
    class Exact(torch.nn.Module):
        def forward(self,xy): return 30-xy[:,1:2]+0*xy[:,:1]**2
    points=torch.stack([torch.linspace(0,1,17),torch.ones(17)],dim=1)
    bc={'top':BoundaryCondition('top','neumann',0.,0.)}
    _,_,boundary=compute_pde_bc_loss(Exact(),torch.rand(31,2),{'top':points},bc,20.,lambda x:x,False,2.,1.,1.)
    assert torch.isclose(boundary,torch.tensor(4.))
