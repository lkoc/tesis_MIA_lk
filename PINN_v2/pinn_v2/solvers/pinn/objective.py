"""Physics objective shared by M0, M1 and M2.

Terms (all dimensionless, each normalised by a region scale):
  pde                    -div(k grad T) - Q in every region (first-order form for M2)
  constitutive           q + k grad T = 0 for M2 at interior, face and boundary points
  interface_temperature  [T] = 0 on every interface (perfect contact)
  interface_flux         [q.n] = 0 on every interface
  dirichlet / neumann / robin   soft boundary losses; Robin is q.n = h (T - T_inf), never a Dirichlet value
  energy                 per-region balance: outflow through faces and sides minus the integral of Q
Nothing here reads a reference solution or a FEM field.
"""
import numpy as np
import torch

from pinn_v2.physics.operators import gradients
from pinn_v2.physics.residuals import heat_residual, mixed_residual
from pinn_v2.solvers.pinn.attention import Attention

DEFAULT_WEIGHTS = dict(pde=1., constitutive=1., interface_temperature=100., interface_flux=10.,
    dirichlet=100., neumann=10., robin=10., energy=1.)
TERMS = tuple(DEFAULT_WEIGHTS)


def flux_fields(model, region, xy):
    """T, primary flux (predicted for M2, -k grad T otherwise), derived flux -k grad T, and k."""
    problem = model.problem
    T, predicted = model.field(region, xy)
    k = problem.conductivity(region, xy)
    derived = -k * gradients(T, xy)[:, :2]
    return T, (derived if predicted is None else predicted), derived, k


class Objective:
    def __init__(self, model, n_interior=512, n_layer=256, n_face=128, n_boundary=128, seed=11, sampling='mixed', weights=None, attention=None,
        exact_energy=False, focused=False):
        self.model = model
        self.problem = problem = model.problem
        self.seed = int(seed)
        self.sampling = sampling
        self.counts = {r.id: n_interior if r.kind == 'soil' else n_layer for r in problem.regions}
        self.weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
        unknown = set(self.weights) - set(TERMS)
        if unknown or any(not np.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError(f'Invalid objective weights: {sorted(unknown)}')
        self.exact_constraints = bool(getattr(model, 'exact_constraints', False))
        self.exact_energy = bool(exact_energy) and self.exact_constraints
        self.focused = bool(focused)
        if self.exact_constraints:
            if self.weights['pde'] <= 0:
                raise ValueError('M1_EXACT requires a positive PDE weight')
            kept = ('pde', 'energy') if self.exact_energy else ('pde',)
            self.weights = {name: self.weights[name] if name in kept else 0. for name in TERMS}
        reference = next(model.parameters())
        self.tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=reference.dtype, device=reference.device).clone().detach().requires_grad_(True)
        self.interior = {r.id: self.tensor(problem.sample_interior(r, self.counts[r.id], seed + 7919 * r.id, sampling))
                         for r in problem.regions}
        self.sources = {r.id: problem.source(r, self.interior[r.id]).detach() for r in problem.regions}
        self.faces = [(f, self.tensor(f.xy), self.tensor(f.normal)) for f in problem.faces(n_face)]
        self.pieces = [(p, self.tensor(p.xy), self.tensor(p.normal)) for p in problem.pieces(n_boundary)]
        self.attention = None if attention is None else Attention(attention,
            {r.name: len(self.interior[r.id]) for r in problem.regions}, device=reference.device, dtype=reference.dtype)
        self.residuals = {}
        self.resampling_count = 0

    def resample(self, seed=None):
        """Refresh fixed-size interior clouds without changing boundary/interface quadrature.
        focused: keep the half of a 16x candidate pool with the largest |PDE residual|, fill the rest with fresh points."""
        seed = self.seed + (self.resampling_count + 1) * 1_000_003 if seed is None else int(seed)
        if self.focused:
            self.interior = {region.id: self.tensor(self._focused_points(region, seed + 7919 * region.id))
                for region in self.problem.regions}
        else:
            self.interior = {region.id: self.tensor(self.problem.sample_interior(
                region, self.counts[region.id], seed + 7919 * region.id, self.sampling))
                for region in self.problem.regions}
        self.sources = {region.id: self.problem.source(region, self.interior[region.id]).detach()
            for region in self.problem.regions}
        if self.attention is not None:
            self.attention = Attention(self.attention.kind,
                {region.name: self.counts[region.id] for region in self.problem.regions},
                device=next(self.model.parameters()).device,
                dtype=next(self.model.parameters()).dtype, rate=self.attention.rate, decay=self.attention.decay)
        self.residuals = {}
        self.resampling_count += 1
        return seed

    def _log_radial(self, region, n, seed, r_max=4.0):
        """n log-radial points around each cable centre (r in [radius, r_max]), kept only if inside the region."""
        rng = np.random.default_rng(seed)
        centers, r0 = np.asarray(self.problem.case['cables'], dtype=float), float(self.problem.case['radius'])
        radius = r0 * np.exp(rng.uniform(0., np.log(r_max / r0), n))
        angle = rng.uniform(0., 2 * np.pi, n)
        xy = centers[rng.integers(0, len(centers), n)] + np.c_[radius * np.cos(angle), radius * np.sin(angle)]
        return xy[self.problem.contains_points(region, xy)]

    def _focused_points(self, region, seed):
        n = self.counts[region.id]
        pool = np.asarray(self.problem.sample_interior(region, 16 * n, seed, self.sampling))
        radial = region.kind == 'soil' and 'cables' in self.problem.case and 'radius' in self.problem.case
        if radial:
            pool = np.vstack([pool[:8 * n], self._log_radial(region, 8 * n, seed + 2)])
        reference = next(self.model.parameters())
        xy = torch.as_tensor(pool, dtype=reference.dtype, device=reference.device).requires_grad_(True)
        T, _ = self.model.field(region, xy)
        residual = heat_residual(T, xy, self.problem.conductivity(region, xy), self.problem.source(region, xy).detach())
        score = (residual / self.problem.scales(region)['divergence']).detach().abs().ravel().cpu().numpy()
        worst = pool[np.argsort(-score)[:n // 2]]
        fresh = np.asarray(self.problem.sample_interior(region, n - len(worst), seed + 1, self.sampling))
        return np.vstack([worst, fresh])

    def update_attention(self):
        if self.attention is not None:
            for name, r2 in self.residuals.items():
                self.attention.update(name, r2)

    def __call__(self):
        model, problem = self.model, self.problem
        zero = next(model.parameters()).new_zeros(())
        parts = {name: zero for name in TERMS}
        region_pde = {}
        weighted_pde = zero
        balance = {r.id: -problem.region_power[r.id] for r in problem.regions}
        constitutive = zero
        for region in problem.regions:
            xy = self.interior[region.id]
            T, predicted = model.field(region, xy)
            k = problem.conductivity(region, xy)
            scales = problem.scales(region)
            if model.mixed:
                residual, law = mixed_residual(T, predicted, xy, k, self.sources[region.id])
                constitutive = constitutive + torch.mean((law / scales['flux']) ** 2)
            else:
                residual = heat_residual(T, xy, k, self.sources[region.id])
            r2 = (residual / scales['divergence']) ** 2
            region_pde[region.name] = torch.mean(r2)
            parts['pde'] = parts['pde'] + region_pde[region.name]
            if self.attention is not None:
                self.residuals[region.name] = r2.detach()
                weighted_pde = weighted_pde + torch.mean(self.attention.weight(region.name) * r2)
        skip_surfaces = self.exact_constraints and not self.exact_energy
        for face, xy, normal in (() if skip_surfaces else self.faces):
            left, right = problem.regions[face.left], problem.regions[face.right]
            Tl, ql, dl, _ = flux_fields(model, left, xy)
            Tr, qr, dr, _ = flux_fields(model, right, xy)
            if model.mixed:
                constitutive = constitutive + torch.mean(((ql - dl) / problem.scales(left)['flux']) ** 2)
                constitutive = constitutive + torch.mean(((qr - dr) / problem.scales(right)['flux']) ** 2)
            qln, qrn = (ql * normal).sum(1, keepdim=True), (qr * normal).sum(1, keepdim=True)
            scale = max(problem.scales(left)['flux'], problem.scales(right)['flux'])
            parts['interface_temperature'] = parts['interface_temperature'] + torch.mean(((Tl - Tr) / problem.T_scale) ** 2)
            parts['interface_flux'] = parts['interface_flux'] + torch.mean(((qln - qrn) / scale) ** 2)
            balance[left.id] = balance[left.id] + face.weight * qln.sum()
            balance[right.id] = balance[right.id] - face.weight * qrn.sum()
        for piece, xy, normal in (() if skip_surfaces else self.pieces):
            region = problem.regions[piece.region]
            T, q, derived, _ = flux_fields(model, region, xy)
            scale = problem.scales(region)['flux']
            if model.mixed:
                constitutive = constitutive + torch.mean(((q - derived) / scale) ** 2)
            qn = (q * normal).sum(1, keepdim=True)
            bc = piece.bc
            if bc['type'] == 'dirichlet':
                parts['dirichlet'] = parts['dirichlet'] + torch.mean(((T - bc['value'](xy)) / problem.T_scale) ** 2)
            elif bc['type'] == 'neumann':
                parts['neumann'] = parts['neumann'] + torch.mean(((qn - bc['value'](xy)) / scale) ** 2)
            else:
                parts['robin'] = parts['robin'] + torch.mean(((qn - bc['h'] * (T - bc['T_inf'](xy))) / scale) ** 2)
            balance[region.id] = balance[region.id] + piece.weight * qn.sum()
        parts['constitutive'] = constitutive
        if self.exact_constraints and not self.exact_energy:
            parts['energy'] = zero
            total = self.weights['pde'] * parts['pde']
        elif self.exact_constraints:
            parts['energy'] = sum((value / problem.power_scale) ** 2 for value in balance.values())
            total = self.weights['pde'] * parts['pde'] + self.weights['energy'] * parts['energy']
        else:
            parts['energy'] = sum((value / problem.power_scale) ** 2 for value in balance.values())
            total = sum(self.weights[name] * value for name, value in parts.items())
        if self.attention is not None:
            # Value stays the unweighted loss (comparable across steps); the gradient is that of the attention-weighted PDE term.
            total = total + self.weights['pde'] * ((weighted_pde - weighted_pde.detach()) - (parts['pde'] - parts['pde'].detach()))
        self.optimized_pde = (weighted_pde if self.attention is not None else parts['pde']).detach()
        return total, parts, region_pde
