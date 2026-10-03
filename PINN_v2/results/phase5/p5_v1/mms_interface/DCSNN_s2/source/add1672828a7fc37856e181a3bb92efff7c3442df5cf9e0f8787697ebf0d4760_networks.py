"""M0, M1 and M2 behind one class.

partition  global: one network for the whole domain   | region: one network per material or subdomain
formulation strong: output T, flux from -k grad T       | mixed: output (T, qx, qy), flux is a separate field

  M0 = (global, strong)   control PINN, strong form
  M1 = (region, strong)   per-material PINN, interface conditions as losses
  M2 = (region, mixed)    first-order system T, q with the constitutive law as a loss

No boundary condition is built into the network (no hard envelope): Dirichlet, Neumann and Robin
are all soft losses, so the three kinds are treated alike. The output is T_ref + amplitude * raw.
"""
import torch
from torch import nn

ARCHITECTURES = dict(M0=('global', 'strong'), M1=('region', 'strong'), M2=('region', 'mixed'))


def mlp(inputs, width, depth, outputs, activation='tanh'):
    act = dict(tanh=nn.Tanh, silu=nn.SiLU)[activation]
    layers, size = [], inputs
    for _ in range(depth):
        linear = nn.Linear(size, width)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend([linear, act()])
        size = width
    last = nn.Linear(size, outputs)
    nn.init.xavier_normal_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class PINNModel(nn.Module):
    kind = 'baseline'

    def __init__(self, problem, architecture='M1', width=32, depth=3, activation='tanh'):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f'Unknown architecture {architecture!r}; expected one of {sorted(ARCHITECTURES)}')
        self._setup(problem, architecture, width, depth, *ARCHITECTURES[architecture])
        representatives = problem.regions[:1] if self.partition == 'global' else problem.regions
        self.networks = nn.ModuleList()
        self.offsets = nn.ParameterList()
        for region in representatives:
            self.networks.append(mlp(problem.input_size(region, self.partition), width, depth, 3 if self.mixed else 1, activation))
            self.offsets.append(nn.Parameter(torch.zeros(())))

    def _setup(self, problem, architecture, width, depth, partition, formulation):
        self.problem = problem
        self.architecture = architecture
        self.partition, self.formulation = partition, formulation
        self.mixed = formulation == 'mixed'
        self.width, self.depth = width, depth
    def field(self, region, xy):
        """Return (T, q) for a region; q is None in the strong formulation."""
        index = 0 if self.partition == 'global' else region.id
        raw = self.networks[index](self.problem.coordinates(region, xy, self.partition))
        scales = self.problem.scales(region)
        amplitude = self.problem.amplitude(region, self.partition)
        temperature = self.problem.T_ref + self.problem.T_scale * self.offsets[index] + amplitude * raw[:, :1]
        flux = raw[:, 1:3] * scales['flux'] if self.mixed else None
        return temperature, flux

    def temperature(self, xy, labels):
        result = torch.empty((len(xy), 1), dtype=xy.dtype, device=xy.device)
        for region in self.problem.regions:
            mask = labels == region.id
            if mask.any():
                result[mask] = self.field(region, xy[mask])[0]
        return result

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(problem, architecture, width=32, depth=3, activation='tanh'):
    """M0-M2 or a Phase 5 variant by name."""
    from pinn_v2.solvers.pinn.variants import VARIANTS
    if architecture in VARIANTS:
        return VARIANTS[architecture](problem, architecture, width, depth, activation)
    return PINNModel(problem, architecture, width, depth, activation)
