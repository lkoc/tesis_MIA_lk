"""Diagnostic: verify autograd gradient at cable surface after fix."""
import torch, math, sys

sys.path.insert(0, 'examples/aras_2005_154kv')
if 'run_example' in sys.modules:
    del sys.modules['run_example']
from run_example import _multilayer_T
from pinn_cables.pinn.pde import gradients
from pinn_cables.io.readers import load_cable_layers, load_placements

data = 'examples/aras_2005_154kv/data'
layers = load_cable_layers(f'{data}/cable_layers.csv')
placement = load_placements(f'{data}/cables_placement.csv')[0]
Q_lin, k_soil, T_amb = 66.40, 1.2, 293.15
cx, cy = placement.cx, placement.cy
r_sheath = layers[-1].r_outer

n = 100
angles = torch.linspace(0, 2*math.pi, n+1)[:-1]
x = cx + (r_sheath + 0.001) * torch.cos(angles)
y = cy + (r_sheath + 0.001) * torch.sin(angles)
pts = torch.stack([x, y], dim=1).requires_grad_(True)

T_bg = _multilayer_T(pts, layers, placement, k_soil, T_amb, Q_lin)
print(f'T_bg.grad_fn = {T_bg.grad_fn}')
gT = gradients(T_bg, pts)
dx = pts[:, 0:1] - cx
dy = pts[:, 1:2] - cy
rc = torch.sqrt(dx**2 + dy**2).clamp(min=1e-12)
nr = torch.cat([dx/rc, dy/rc], dim=1)
dTdr = (gT * nr).sum(dim=1, keepdim=True)

expected = -Q_lin / (2*math.pi*r_sheath)
flux = k_soil * dTdr
print(f'T_bg mean       = {T_bg.mean().item():.2f} K')
print(f'dT_bg/dr mean   = {dTdr.mean().item():.2f} K/m')
print(f'k*dT/dr mean    = {flux.mean().item():.2f} W/m2')
print(f'Expected flux   = {expected:.2f} W/m2')
print(f'Ratio           = {flux.mean().item()/expected:.4f}')
mse_norm = ((flux - expected)/abs(expected)).pow(2).mean().item()
print(f'MSE normalized  = {mse_norm:.6f}')
