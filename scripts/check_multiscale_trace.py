"""Audit interface identities and the full transformed 2D operator before fitting."""
from pathlib import Path
import sys, json, math, hashlib
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Benchmarks.full_domain import FullDomain
from Benchmarks.multiscale_trace import TracePINN
from Benchmarks.full_physics import heat_residual
from pinn_cables.pinn.pde import gradients

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(20260915)
case = json.loads((ROOT/'Benchmarks/explicit_study/cases/xlpe_single.json').read_text())
domain = FullDomain(case)
model = TracePINN(domain, width=8, depth=2)
with torch.no_grad():
    model.trace_T.normal_(0, .02)
    model.trace_q.normal_(0, .05)
rows = []
for interface in domain.interfaces:
    points, normals, _ = domain.interface_points(interface, 257)
    xy = torch.tensor(points, requires_grad=True)
    normal = torch.tensor(normals)
    values = []
    for rid in [interface.left, interface.right]:
        region = domain.regions[rid]
        t = model.field(region, xy)[0]
        q = -domain.conductivity(region, xy, torch)*gradients(t, xy)[:, :2]
        values.append((t, (q*normal).sum(1)))
    tj = float((values[0][0]-values[1][0]).abs().max().detach())
    qj = float((values[0][1]-values[1][1]).abs().max().detach())
    rows.append(dict(interface=interface.name, temperature_jump_K=tj, flux_jump_W_m2=qj))
    assert tj < 1e-9 and qj < 1e-6, rows[-1]
center = torch.tensor(case['cables'], requires_grad=True)
core = domain.conductors[0]
t = model.field(core, center)[0]
residual = heat_residual(t, center, center[:, :1]*0+core.k, center[:, :1]*0)
assert torch.isfinite(residual).all()
outer_errors = []
for points, _, _ in domain.outer_boundary(31):
    xy = torch.tensor(points, requires_grad=True)
    outer_errors.append(float((model.field(domain.regions[0], xy)[0]-domain.boundary_temperature(xy)).abs().max().detach()))
assert max(outer_errors) < 1e-10
# Independent Cartesian versus log-polar differential operator, with variable
# conductivity and nonzero angular dependence. No trained field or FEM labels.
rng = np.random.default_rng(101)
a, b = .012, .013
rad = rng.uniform(a, b, 100)
angle = rng.uniform(0, 2*np.pi, 100)
xy = torch.tensor(np.c_[rad*np.cos(angle), rad*np.sin(angle)], requires_grad=True)
r = (xy.square().sum(1, keepdim=True)).sqrt()
theta = torch.atan2(xy[:, 1:2], xy[:, :1])
ell = math.log(b/a)
s = torch.log(r/a)/ell
T = torch.exp(.2*s)*torch.cos(3*theta)+s*s
k = 1+.1*s+.05*torch.sin(theta)
cart = -heat_residual(T, xy, k, xy[:, :1]*0)
uv = torch.tensor(np.c_[s.detach().numpy().ravel(), angle], requires_grad=True)
ss, tt = uv[:, :1], uv[:, 1:2]
temp = torch.exp(.2*ss)*torch.cos(3*tt)+ss*ss
kk = 1+.1*ss+.05*torch.sin(tt)
gt = gradients(temp, uv)
normal_term = gradients(kk*gt[:, :1], uv)[:, :1]/(ell*ell)
angular_term = gradients(kk*gt[:, 1:2], uv)[:, 1:2]
polar = (normal_term+angular_term)/(a*torch.exp(ell*ss)).square()
relative = float((cart-polar).abs().max().detach()/cart.abs().max().detach())
assert relative < 1e-11, relative
result = dict(passed=True, interfaces=rows, center_residual_finite=True,
              outer_temperature_error_K=max(outer_errors),
              variable_k_full_2d_operator_relative_error=relative,
              operator_angular_term_max=float(angular_term.abs().max().detach()),
              notes='Algebraic/numerical construction checks; no trained-solution accuracy claim',
              source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),ROOT/'Benchmarks/multiscale_trace.py']})
out = ROOT/'Benchmarks/multiscale_exploration/construction_audit.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
