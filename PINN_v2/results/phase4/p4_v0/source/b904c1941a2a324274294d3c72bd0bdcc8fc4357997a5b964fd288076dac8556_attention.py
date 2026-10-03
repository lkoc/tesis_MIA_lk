"""Per-point weights on the interior PDE residual (adapted, not the original algorithms).

rba  residual-based attention, idea of Anagnostopoulos et al. 2024 (arXiv:2307.00379, por verificar):
     lam <- decay * lam + rate * |r| / max|r|, weight = 1 + lam.
sa   self-adaptive weights, idea of McClenny and Braga-Neto (arXiv:2009.04544, por verificar):
     weight <- weight + rate * r^2 / mean(r^2), applied normalised to unit mean so the PDE term keeps its overall scale.
The update is closed form from the detached residual; the weights never see a reference solution.
"""
import torch

KINDS = ('rba', 'sa')
DEFAULT_RATE = dict(rba=0.01, sa=0.01)
DECAY = 0.999


class Attention:
    def __init__(self, kind, sizes, dtype=torch.float64, rate=None, decay=DECAY):
        if kind not in KINDS:
            raise ValueError(f'Unknown attention {kind!r}; expected one of {KINDS}')
        self.kind, self.decay = kind, decay
        self.rate = DEFAULT_RATE[kind] if rate is None else rate
        start = 0. if kind == 'rba' else 1.
        self.state = {key: torch.full((n, 1), start, dtype=dtype) for key, n in sizes.items()}

    def weight(self, key):
        state = self.state[key]
        return 1. + state if self.kind == 'rba' else state / state.mean()

    def update(self, key, squared_residual):
        r2 = squared_residual.detach().reshape(-1, 1)
        if self.kind == 'rba':
            r = torch.sqrt(r2)
            self.state[key] = self.decay * self.state[key] + self.rate * r / r.max().clamp_min(1e-300)
        else:
            self.state[key] = self.state[key] + self.rate * r2 / r2.mean().clamp_min(1e-300)

    def summary(self):
        return {key: dict(min=float(self.weight(key).min()), max=float(self.weight(key).max()), mean=float(self.weight(key).mean())) for key in self.state}
