"""Exploratory circular shared-trace PINN; separate from the frozen A--D study.

Full 2D heat PDE is supplied by full_pinn.PhysicsLoss without modifications.
The ansatz enforces T and radial heat flux across every circular material face.
Only one cable and one soil material region are supported in this prototype.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Benchmarks import full_pinn as engine
from Benchmarks.full_domain import FullDomain
from pinn_cables.pinn.pde import gradients

OriginalPINN = engine.FullPINN


class TracePINN(OriginalPINN):
    def __init__(self, domain, variant='trace', width=32, depth=3):
        if domain.nsoil != 1 or len(domain.case['cables']) != 1:
            raise ValueError('Exploratory prototype requires one cable and no soil strata')
        super().__init__(domain, 'subdomain', width, depth)
        self.variant = 'trace'
        self.order = 6
        count = len(domain.case['layers'])
        self.trace_T = nn.Parameter(torch.zeros(count, 1 + 2*self.order))
        self.trace_q = nn.Parameter(torch.zeros(count, 1 + 2*self.order))
        for offset in self.offsets[1:]:
            offset.requires_grad_(False)
        c = domain.case
        xc, yc = c['cables'][0]
        x0, x1, y0, y1 = c['bounds']
        self.collar = min(.15, xc-x0, x1-xc, yc-y0, y1-yc,
                          c.get('outer_radius', float('inf')))
        if self.collar <= c['radius']:
            raise ValueError('No room for a collar contained in the soil')

    def coefficients(self, layer):
        c = self.domain.case
        t = c['scale'] * self.trace_T[layer]
        q = c['power']/(2*math.pi*c['layers'][layer][1]) * self.trace_q[layer]
        return t, q

    def basis(self, unit):
        real, imag = unit[:, :1], unit[:, 1:2]
        ar, ai = torch.ones_like(real), torch.zeros_like(real)
        columns = [ar]
        for _ in range(self.order):
            ar, ai = ar*real-ai*imag, ar*imag+ai*real
            columns.extend([ar, ai])
        return torch.cat(columns, dim=1)

    def trace(self, layer, unit):
        basis = self.basis(unit)
        t, q = self.coefficients(layer)
        return self.domain.case['T0'] + basis @ t[:, None], basis @ q[:, None]

    def global_soil(self, region, xy):
        c = self.domain.case
        raw = self.networks[region.id](self.coordinates(region, xy))
        return self.domain.boundary_temperature(xy) + self.envelope(xy)*c['scale']*(self.offsets[region.id]+raw)

    def field(self, region, xy):
        # Independent evaluator may call temperature under no_grad. The collar
        # contains a derivative of G, so it must locally enable differentiation.
        with torch.enable_grad():
            if not xy.requires_grad:
                xy = xy.clone().requires_grad_(True)
            c = self.domain.case
            delta = xy[:, :2] - xy.new_tensor(c['cables'][0])
            if region.kind == 'soil':
                radius = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
                unit = delta/radius
                R = c['radius']
                projected = xy.new_tensor(c['cables'][0]) + R*unit
                G = self.global_soil(region, xy)
                G0 = self.global_soil(region, projected)
                Gr = (gradients(G0, projected)[:, :2]*unit).sum(1, keepdim=True)
                tau, q = self.trace(len(c['layers'])-1, unit)
                k = self.domain.conductivity(region, projected, torch)
                h = self.collar-R
                s = ((radius-R)/h).clamp(max=1.)
                w0 = 1-10*s**3+15*s**4-6*s**5
                w1 = s-6*s**3+8*s**4-3*s**5
                return G+w0*(tau-G0)+h*w1*(-q/k-Gr), None
            raw = self.networks[region.id](self.coordinates(region, xy))
            amplitude = self.domain.scales(region)['temperature']
            if region.kind == 'conductor':
                z = delta/region.ro
                rho2 = (z*z).sum(1, keepdim=True)
                t, q = self.coefficients(0)
                orders = t.new_tensor([0]+[n for n in range(1, self.order+1) for _ in range(2)])
                b = (-region.ro*q/region.k-orders*t)/2
                a = t-b
                T = c['T0']+(self.basis(z)*(a[None, :]+rho2*b[None, :])).sum(1, keepdim=True)
                return T+amplitude*(1-rho2)**2*raw, None
            radius = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
            unit = delta/radius
            ell = math.log(region.ro/region.ri)
            s = torch.log(radius/region.ri)/ell
            ta, qa = self.trace(region.layer-1, unit)
            tb, qb = self.trace(region.layer, unit)
            da = -ell*region.ri*qa/region.k
            db = -ell*region.ro*qb/region.k
            T = (2*s**3-3*s**2+1)*ta+(-2*s**3+3*s**2)*tb
            T = T+(s**3-2*s**2+s)*da+(s**3-s**2)*db
            return T+16*amplitude*s**2*(1-s)**2*raw, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source = ROOT/'Benchmarks/explicit_study/C/lr_0.0005/xlpe_single'/f'seed{args.seed}/configuration.json'
    config = json.loads(source.read_text(encoding='utf-8'))
    domain = FullDomain(config['physics']['case'], config['physics']['source_mode'])
    config.update(variant='trace', temperature_weight=100., trace_order=6,
                  experimental=True, baseline_configuration=str(source))
    if args.smoke:
        config.update(adam=2, lbfgs=2, n=32, n_layer=16, n_interface=16)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output/'source'
    archive.mkdir(exist_ok=True)
    hashes = {}
    for path in [Path(__file__), ROOT/'docs/INVESTIGACION_PINN_MULTIESCALA_2026-09-15.md']:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        (archive/(digest+'_'+path.name)).write_bytes(content)
        hashes[str(path.relative_to(ROOT))] = digest
    (args.output/'experimental_source_sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    engine.FullPINN = TracePINN  # process-local factory; frozen production files stay intact
    engine.train(domain, config, args.output)


if __name__ == '__main__':
    main()
