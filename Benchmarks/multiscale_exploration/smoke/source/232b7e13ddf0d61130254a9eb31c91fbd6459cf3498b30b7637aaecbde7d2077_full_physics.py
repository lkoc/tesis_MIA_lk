"""The same local heat equation for every active architecture and every material."""
import torch
from pinn_cables.pinn.pde import gradients,laplace_variable_k


def heat_residual(temperature,coordinates,k,source,capacity=None):
    storage=0.
    if coordinates.shape[1]==3:
        if capacity is None or capacity<=0:raise ValueError('Transient heat equation requires positive rho*cp')
        storage=capacity*gradients(temperature,coordinates)[:,2:3]
    return storage-laplace_variable_k(temperature,coordinates,k)-source


def mixed_residual(temperature,flux,coordinates,k,source,capacity=None):
    grad=gradients(temperature,coordinates)
    constitutive=flux+k*grad[:,:2]
    divergence=gradients(flux[:,0:1],coordinates)[:,0:1]+gradients(flux[:,1:2],coordinates)[:,1:2]
    storage=0.
    if coordinates.shape[1]==3:
        if capacity is None or capacity<=0:raise ValueError('Transient heat equation requires positive rho*cp')
        storage=capacity*grad[:,2:3]
    return storage+divergence-source,constitutive


def polar_heat_residual(temperature,r_phi_t,k,source,capacity=None):
    """Full r,phi equation on annuli; origin must be handled in Cartesian form."""
    radius=r_phi_t[:,0:1]
    if torch.any(radius<=0):raise ValueError('Polar operator excludes r=0; use Cartesian conductor coordinates')
    derivative=gradients(temperature,r_phi_t)
    radial=gradients(radius*k*derivative[:,0:1],r_phi_t)[:,0:1]/radius
    angular=gradients(k*derivative[:,1:2],r_phi_t)[:,1:2]/radius**2
    storage=0.
    if r_phi_t.shape[1]==3:
        if capacity is None or capacity<=0:raise ValueError('Transient heat equation requires positive rho*cp')
        storage=capacity*derivative[:,2:3]
    return storage-radial-angular-source
