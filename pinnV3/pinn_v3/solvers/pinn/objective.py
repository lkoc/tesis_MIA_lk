"""Physics objective shared by M0, M1 and M2.

Terms (all dimensionless, each normalised by a region scale):
  pde                    -div(k grad T) - Q in every region (first-order form for M2)
  constitutive           q + k grad T = 0 for M2 at interior, face and boundary points
  interface_temperature  [T] = 0 on every interface, sampled on two rings/rows (one inside each region)
  dirichlet / neumann / robin   soft boundary losses on two rows (boundary and one inside offset);
                         Robin is q.n = h (T - T_inf), never a Dirichlet value
  energy                 opt-in per-region balance (weight 0 by default); it integrates interface flux
                         and is therefore not part of the default loss
Interface flux [q.n] is not penalised; it is only used to score focused points.
Nothing here reads a reference solution or a FEM field.
"""
import numpy as np
import torch
from dataclasses import replace

from pinn_v3.physics.operators import gradients
from pinn_v3.physics.residuals import heat_residual, mixed_residual
from pinn_v3.solvers.pinn.attention import Attention

DEFAULT_WEIGHTS = dict(pde=1., constitutive=1., interface_temperature=100.,
    dirichlet=100., neumann=10., robin=10., energy=0.)
TERMS = tuple(DEFAULT_WEIGHTS)
FOCUS_CRITERIA = ('residual', 'gradient', 'mixed')


def flux_fields(model, region, xy):
    """T, primary flux (predicted for M2, -k grad T otherwise), derived flux -k grad T, and k."""
    problem = model.problem
    T, predicted = model.field(region, xy)
    k = problem.conductivity(region, xy)
    derived = -k * gradients(T, xy)[:, :2]
    return T, (derived if predicted is None else predicted), derived, k


class Objective:
    def __init__(self, model, n_interior=512, n_layer=256, n_face=128, n_boundary=128, seed=11, sampling='mixed', weights=None, attention=None,
        exact_energy=False, focused=False, focus_criterion='residual', focus_surfaces=True, initial_fraction=None):
        self.model = model
        self.problem = problem = model.problem
        self.seed = int(seed)
        self.sampling = sampling
        self.focus_criterion = focus_criterion
        self.focus_surfaces = bool(focus_surfaces)
        if focus_criterion not in FOCUS_CRITERIA:
            raise ValueError(f'focus_criterion must be one of {FOCUS_CRITERIA}')
        self.counts = {r.id: n_interior if r.kind == 'soil' else n_layer for r in problem.regions}
        if initial_fraction is not None and (not np.isfinite(initial_fraction) or not 0 < initial_fraction <= 1):
            raise ValueError('initial_fraction must be finite and in (0, 1]')
        if initial_fraction is not None and attention is not None:
            raise ValueError('incremental point generation does not support adaptive attention')
        self.face_count = int(n_face)
        self.boundary_count = int(n_boundary)
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
            kept = ('pde', 'energy') if exact_energy else ('pde',)
            self.weights = {name: self.weights[name] if name in kept else 0. for name in TERMS}
        reference = next(model.parameters())
        self.tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=reference.dtype, device=reference.device).clone().detach().requires_grad_(True)
        initial_counts = {r.id: self.counts[r.id] if initial_fraction is None else
            max(1, int(np.ceil(self.counts[r.id] * initial_fraction))) for r in problem.regions}
        self.interior = {r.id: self.tensor(self._focused_points(r, seed + 7919 * r.id, initial_counts[r.id]) if self.focused else
            problem.sample_interior(r, initial_counts[r.id], seed + 7919 * r.id, sampling)) for r in problem.regions}
        self.sources = {r.id: problem.source(r, self.interior[r.id]).detach() for r in problem.regions}
        self.faces = self._surface_clouds('face', n_face, seed + 104729) if self.focused and self.focus_surfaces else [
            self._surface_entry('face', f, f.xy, f.normal, self._right_xy(f)) for f in problem.faces(n_face)]
        self.pieces = self._surface_clouds('piece', n_boundary, seed + 209759) if self.focused and self.focus_surfaces else [
            (p, self.tensor(p.xy), self.tensor(p.normal)) for p in problem.pieces(n_boundary)]
        self.attention = None if attention is None else Attention(attention,
            {r.name: len(self.interior[r.id]) for r in problem.regions}, device=reference.device, dtype=reference.dtype)
        self.residuals = {}
        self.resampling_count = 0
        self._full_interior = self.interior
        self._full_sources = self.sources
        self.point_fraction = 1.0

    @staticmethod
    def _active_indices(point_count, fraction, focused, device):
        if point_count == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        active_count = max(1, min(point_count, int(np.ceil(point_count * fraction))))
        if not focused:
            return torch.arange(active_count, device=device)
        focused_count = point_count // 2
        active_focused = min(focused_count, (active_count + 1) // 2)
        active_uniform = active_count - active_focused
        indices = [torch.arange(active_focused, device=device)]
        if active_uniform:
            indices.append(torch.arange(focused_count, focused_count + active_uniform, device=device))
        return torch.cat(indices)

    def set_point_fraction(self, fraction):
        """Activate a nested subset of interior PDE points; keep physical surfaces complete."""
        fraction = float(fraction)
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('point fraction must be finite and in (0, 1]')
        self.point_fraction = fraction
        self.interior, self.sources = {}, {}
        for region in self.problem.regions:
            full_xy = self._full_interior[region.id]
            indices = self._active_indices(len(full_xy), fraction, self.focused, full_xy.device)
            self.interior[region.id] = full_xy.index_select(0, indices).detach().clone().requires_grad_(True)
            self.sources[region.id] = self._full_sources[region.id].index_select(0, indices).detach().clone()

    def grow_interior(self, fraction, stage):
        """Append fresh region-local points up to a requested fraction of the budget."""
        if self.attention is not None:
            raise ValueError('incremental point generation does not support adaptive attention')
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('point fraction must be finite and in (0, 1]')
        for region in self.problem.regions:
            target = max(1, int(np.ceil(self.counts[region.id] * fraction)))
            missing = target - len(self._full_interior[region.id])
            if missing <= 0:
                continue
            seed = self.seed + 7919 * region.id + 1_000_003 * (int(stage) + 1)
            new_points = self._focused_points(region, seed, missing) if self.focused else self.problem.sample_interior(
                region, missing, seed, self.sampling)
            points = self.tensor(np.vstack((self._full_interior[region.id].detach().cpu().numpy(), new_points)))
            self._full_interior[region.id] = points
            self._full_sources[region.id] = self.problem.source(region, points).detach()
        self.point_fraction = 1.
        self.interior = dict(self._full_interior)
        self.sources = dict(self._full_sources)

    def resample(self, seed=None):
        """Refresh clouds while retaining uniform coverage in every region and surface group."""
        fraction = self.point_fraction
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
        if self.focused and self.focus_surfaces:
            self.faces = self._surface_clouds('face', self.face_count, seed + 104729)
            self.pieces = self._surface_clouds('piece', self.boundary_count, seed + 209759)
        if self.attention is not None:
            self.attention = Attention(self.attention.kind,
                {region.name: self.counts[region.id] for region in self.problem.regions},
                device=next(self.model.parameters()).device,
                dtype=next(self.model.parameters()).dtype, rate=self.attention.rate, decay=self.attention.decay)
        self.residuals = {}
        self._full_interior = self.interior
        self._full_sources = self.sources
        self.set_point_fraction(fraction)
        self.resampling_count += 1
        return seed

    @staticmethod
    def _right_xy(face):
        """Ring inside the right region; a face without xy_right evaluates both sides at xy."""
        return face.xy if getattr(face, 'xy_right', None) is None else face.xy_right

    def _surface_entry(self, kind, surface, xy, normal, xy_right=None):
        """Tensorised surface record: (surface, xy, normal) for pieces and (face, xy, normal, xy_right) for faces."""
        entry = (surface, self.tensor(xy), self.tensor(normal))
        if kind != 'face':
            return entry
        return entry + (self.tensor(self._right_xy(surface) if xy_right is None else xy_right),)

    def _surface_clouds(self, kind, count, seed):
        """Keep half uniform surface points and focus the other half by a boundary score.

        Faces are matched by interface name so that both rings of a pair stay together even when a
        candidate set drops an interface; pieces are matched by position as before.
        """
        reference = next(self.model.parameters())
        if kind == 'face':
            uniform = self.problem.faces(count)
            candidates = {face.name: face for face in self.problem.faces(max(2, 16 * count))}
        else:
            uniform = self.problem.pieces(count)
            candidates = dict(enumerate(self.problem.pieces(max(2, 16 * count))))
        result = []
        for index, base in enumerate(uniform):
            group = candidates.get(base.name if kind == 'face' else index)
            if group is None or len(base.xy) < 2:
                result.append(self._surface_entry(kind, base, base.xy, base.normal))
                continue
            xy = torch.as_tensor(group.xy, dtype=reference.dtype, device=reference.device).requires_grad_(True)
            normal = torch.as_tensor(group.normal, dtype=reference.dtype, device=reference.device)
            xy_right = None
            if kind == 'face':
                xy_right = torch.as_tensor(self._right_xy(group), dtype=reference.dtype,
                    device=reference.device).requires_grad_(True)
            score = self._surface_score(kind, base, xy, normal, xy_right).detach().cpu().numpy()
            focused_count = min(len(base.xy) // 2, len(group.xy))
            selected = np.argsort(-score)[:focused_count]
            uniform_count = len(base.xy) - focused_count
            selected_xy = np.vstack([group.xy[selected], base.xy[:uniform_count]])
            selected_normal = np.vstack([group.normal[selected], base.normal[:uniform_count]])
            if kind == 'face':
                selected_right = np.vstack([self._right_xy(group)[selected], self._right_xy(base)[:uniform_count]])
                surface = replace(base, xy=selected_xy, normal=selected_normal, xy_right=selected_right)
                result.append(self._surface_entry(kind, surface, selected_xy, selected_normal, selected_right))
            else:
                surface = replace(base, xy=selected_xy, normal=selected_normal)
                result.append(self._surface_entry(kind, surface, selected_xy, selected_normal))
        return result

    def _surface_score(self, kind, surface, xy, normal, xy_right=None):
        """Score continuity or boundary error, optionally combined with temperature gradient."""
        if kind == 'face':
            left, right = self.problem.regions[surface.left], self.problem.regions[surface.right]
            xy_right = xy if xy_right is None else xy_right
            tl, ql, _, _ = flux_fields(self.model, left, xy)
            tr, qr, _, _ = flux_fields(self.model, right, xy_right)
            temperature_error = (tl - tr).abs() / max(self.problem.T_scale, 1e-12)
            flux_scale = max(self.problem.scales(left)['flux'], self.problem.scales(right)['flux'])
            flux_error = ((ql * normal).sum(1, keepdim=True) - (qr * normal).sum(1, keepdim=True)).abs() / max(flux_scale, 1e-12)
            residual = (temperature_error + flux_error).flatten()
            gradients_score = 0.5 * (torch.linalg.vector_norm(gradients(tl, xy)[:, :2], dim=1) /
                max(self.problem.scales(left)['temperature'] / max(np.sqrt(self.problem.area(left)), 1e-12), 1e-12) +
                torch.linalg.vector_norm(gradients(tr, xy_right)[:, :2], dim=1) /
                max(self.problem.scales(right)['temperature'] / max(np.sqrt(self.problem.area(right)), 1e-12), 1e-12))
        else:
            region = self.problem.regions[surface.region]
            temperature, flux, _, _ = flux_fields(self.model, region, xy)
            qn = (flux * normal).sum(1, keepdim=True)
            bc = surface.bc
            scale = max(self.problem.scales(region)['flux'], 1e-12)
            if bc['type'] == 'dirichlet':
                residual = ((temperature - bc['value'](xy)).abs() / max(self.problem.T_scale, 1e-12)).flatten()
            elif bc['type'] == 'neumann':
                residual = ((qn - bc['value'](xy)).abs() / scale).flatten()
            else:
                residual = ((qn - bc['h'] * (temperature - bc['T_inf'](xy))).abs() / scale).flatten()
            gradient_scale = max(self.problem.scales(region)['temperature'] / max(np.sqrt(self.problem.area(region)), 1e-12), 1e-12)
            gradients_score = torch.linalg.vector_norm(gradients(temperature, xy)[:, :2], dim=1) / gradient_scale
        if self.focus_criterion == 'residual':
            return residual
        if self.focus_criterion == 'gradient':
            return gradients_score
        residual = residual / residual.mean().clamp_min(1e-12)
        gradients_score = gradients_score / gradients_score.mean().clamp_min(1e-12)
        return 0.5 * (residual + gradients_score)

    def _importance_score(self, region, xy):
        """Score a candidate cloud by PDE residual, temperature gradient, or their mixture."""
        T, _ = self.model.field(region, xy)
        scales = self.problem.scales(region)
        residual_score = (heat_residual(T, xy, self.problem.conductivity(region, xy),
            self.problem.source(region, xy).detach()) / scales['divergence']).detach().abs().flatten()
        if self.focus_criterion == 'residual':
            return residual_score
        gradient = gradients(T, xy)[:, :2]
        gradient_scale = scales['temperature'] / max(np.sqrt(self.problem.area(region)), 1e-12)
        gradient_score = (torch.linalg.vector_norm(gradient, dim=1) / gradient_scale).detach().abs()
        if self.focus_criterion == 'gradient':
            return gradient_score
        residual_score = residual_score / (residual_score.mean().clamp_min(1e-12))
        gradient_score = gradient_score / (gradient_score.mean().clamp_min(1e-12))
        return 0.5 * (residual_score + gradient_score)

    def _focused_points(self, region, seed, count=None):
        """Keep half high-importance points and half fresh uniform points for this region."""
        n = self.counts[region.id] if count is None else int(count)
        pool = np.asarray(self.problem.focused_interior_pool(region, 16 * n, seed, self.sampling))
        reference = next(self.model.parameters())
        xy = torch.as_tensor(pool, dtype=reference.dtype, device=reference.device).requires_grad_(True)
        score = self._importance_score(region, xy).cpu().numpy()
        important = pool[np.argsort(-score)[:n // 2]]
        fresh = np.asarray(self.problem.sample_interior(region, n - len(important), seed + 1, self.sampling))
        return np.vstack([important, fresh])

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
        skip_boundary_pieces = self.exact_constraints and not self.exact_energy
        for face, xy, normal, xy_right in self.faces:
            left, right = problem.regions[face.left], problem.regions[face.right]
            Tl, ql, dl, _ = flux_fields(model, left, xy)
            Tr, qr, dr, _ = flux_fields(model, right, xy_right)
            if model.mixed:
                constitutive = constitutive + torch.mean(((ql - dl) / problem.scales(left)['flux']) ** 2)
                constitutive = constitutive + torch.mean(((qr - dr) / problem.scales(right)['flux']) ** 2)
            qln, qrn = (ql * normal).sum(1, keepdim=True), (qr * normal).sum(1, keepdim=True)
            parts['interface_temperature'] = parts['interface_temperature'] + torch.mean(((Tl - Tr) / problem.T_scale) ** 2)
            balance[left.id] = balance[left.id] + face.weight * qln.sum()
            balance[right.id] = balance[right.id] - face.weight * qrn.sum()
        for piece, xy, normal in (() if skip_boundary_pieces else self.pieces):
            region = problem.regions[piece.region]
            T, q, derived, _ = flux_fields(model, region, xy)
            scale = problem.scales(region)['flux']
            if model.mixed:
                constitutive = constitutive + torch.mean(((q - derived) / scale) ** 2)
            qn = (q * normal).sum(1, keepdim=True)
            bc = piece.bc
            if getattr(model, 'exact_boundary', False):
                pass
            elif bc['type'] == 'dirichlet':
                parts['dirichlet'] = parts['dirichlet'] + torch.mean(((T - bc['value'](xy)) / problem.T_scale) ** 2)
            elif bc['type'] == 'neumann':
                parts['neumann'] = parts['neumann'] + torch.mean(((qn - bc['value'](xy)) / scale) ** 2)
            else:
                parts['robin'] = parts['robin'] + torch.mean(((qn - bc['h'] * (T - bc['T_inf'](xy))) / scale) ** 2)
            balance[region.id] = balance[region.id] + piece.weight * qn.sum()
        parts['constitutive'] = constitutive
        if self.exact_constraints and not self.exact_energy:
            parts['energy'] = zero
            total = sum(self.weights[name] * parts[name] for name in ('pde', 'constitutive', 'interface_temperature'))
        elif self.exact_constraints:
            parts['energy'] = sum((value / problem.power_scale) ** 2 for value in balance.values())
            total = sum(self.weights[name] * parts[name] for name in (
                'pde', 'energy', 'constitutive', 'interface_temperature'))
        else:
            parts['energy'] = sum((value / problem.power_scale) ** 2 for value in balance.values())
            total = sum(self.weights[name] * value for name, value in parts.items())
        if self.attention is not None:
            # Value stays the unweighted loss (comparable across steps); the gradient is that of the attention-weighted PDE term.
            total = total + self.weights['pde'] * ((weighted_pde - weighted_pde.detach()) - (parts['pde'] - parts['pde'].detach()))
        self.optimized_pde = (weighted_pde if self.attention is not None else parts['pde']).detach()
        return total, parts, region_pde
