"""Generación de datos paramétricos para entrenamiento del FNO de cables.

El FNO necesita pares (entrada_grid, T_grid) para entrenamiento supervisado.
Este módulo genera esos pares variando los parámetros del problema:

    - k_soil   ∈ [k_min, k_max]   conductividad del suelo [W/(m K)]
    - k_pac    ∈ [k_min, k_max]   conductividad zona PAC
    - I_A      ∈ [I_min, I_max]   corriente de operación [A]

Para cada combinación se calcula la temperatura usando la solución analítica
de Kennelly (rápida) como aproximación de ground-truth, luego se rasteriza
en una malla uniforme N_g × N_g.

La solución Kennelly es exacta para k homogéneo. Para el caso PAC heterogéneo
la usamos como inicialización de baja fidelidad; el FNO aprende la corrección
residual que converge a la solución de la PDE completa cuando se añade la
pérdida de física (PINO).

Estructura del tensor de entrada (C_in = 3 canales):
    Canal 0: k(x,y)        — conductividad normalizada [-1, 1]
    Canal 1: Q_src(x,y)    — fuente de calor normalizada
    Canal 2: BC_mask(x,y)  — máscara de condición de borde [-1=nothing, +1=Dirichlet]

Estructura del tensor de salida (C_out = 1):
    Canal 0: T(x,y) - T_amb  — temperatura relativa normalizada [K]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset

from pinn_cables.physics.kennelly import multilayer_T_multi


# ---------------------------------------------------------------------------
# Construcción de canales de entrada
# ---------------------------------------------------------------------------

def make_input_channels(
    domain,
    placements: list,
    layers_list: list,
    k_fn: Callable[[torch.Tensor], torch.Tensor] | float,
    Q_lins: list[float],
    T_amb: float,
    N_g: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterizar k(x,y), Q_src(x,y), BC_mask(x,y) en malla uniforme.

    Args:
        domain:      Dominio 2D (xmin, xmax, ymin, ymax).
        placements:  Posiciones de los cables.
        layers_list: Capas de cables (para radio de vaina).
        k_fn:        Función k(x,y) o escalar.
        Q_lins:      Calor lineal por cable [W/m].
        T_amb:       Temperatura ambiente [K].
        N_g:         Resolución de la malla (N_g × N_g).
        device:      Dispositivo torch.

    Returns:
        (input_grid, coord_grid):
          - input_grid : ``(3, N_g, N_g)`` — [k_norm, Q_norm, BC_mask]
          - coord_grid : ``(2, N_g, N_g)`` — coordenadas físicas [m]
    """
    xs = torch.linspace(domain.xmin, domain.xmax, N_g, device=device)
    ys = torch.linspace(domain.ymin, domain.ymax, N_g, device=device)
    Yg, Xg = torch.meshgrid(ys, xs, indexing="ij")  # (N_g, N_g)
    xy_flat = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # (N_g², 2)

    # Canal 0: k(x,y) normalizado a [0,1]
    if callable(k_fn):
        k_vals = k_fn(xy_flat).reshape(N_g, N_g)
    else:
        k_vals = torch.full((N_g, N_g), float(k_fn), device=device)
    k_min_v = k_vals.min()
    k_max_v = k_vals.max()
    k_range = (k_max_v - k_min_v).clamp(min=1e-6)
    k_norm = (k_vals - k_min_v) / k_range  # [0, 1]

    # Canal 1: Q_src — gaussiana difusa en cada cable (normalizada)
    Q_map = torch.zeros(N_g, N_g, device=device)
    r_sheaths = [layers_list[i][-1].r_outer for i in range(len(placements))]
    for pl, Q_lin, r_sh in zip(placements, Q_lins, r_sheaths):
        # Radio de suavizado = 2 × radio de vaina
        sigma = 2.0 * r_sh
        dx = Xg - pl.cx
        dy = Yg - pl.cy
        gauss = Q_lin * torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        Q_map = Q_map + gauss
    Q_norm = Q_map / (Q_map.max().clamp(min=1e-6))

    # Canal 2: BC_mask — +1 en bordes Dirichlet, 0 en interior
    bc_mask = torch.zeros(N_g, N_g, device=device)
    bc_mask[0, :]  = 1.0   # y = ymin  (bottom)
    bc_mask[-1, :] = 1.0   # y = ymax  (top)
    bc_mask[:, 0]  = 1.0   # x = xmin  (left)
    bc_mask[:, -1] = 1.0   # x = xmax  (right)

    input_grid = torch.stack([k_norm, Q_norm, bc_mask], dim=0)  # (3, N_g, N_g)
    coord_grid = torch.stack([Xg, Yg], dim=0)                   # (2, N_g, N_g)
    return input_grid, coord_grid


# ---------------------------------------------------------------------------
# Dataset paramétrico
# ---------------------------------------------------------------------------

@dataclass
class SampleConfig:
    """Parámetros de un único sample paramétrico."""
    k_soil: float
    k_pac:  float | None
    I_A:    float


class CableParametricDataset(Dataset):
    """Dataset de pares (input_grid, T_grid) para entrenamiento del FNO.

    Genera ``n_samples`` variaciones del problema variando k_soil, k_pac e I_A.
    La temperatura de ground-truth se calcula con la solución analítica de
    Kennelly (rápida, exacta para k homogéneo).

    Para k heterogéneo (zona PAC), Kennelly usa el k efectivo del centroide
    del duct-bank como aproximación — el FNO aprende la corrección real.

    Args:
        domain:           Dominio 2D.
        placements:       Posiciones de cables.
        layers_template:  Capas de cable (sin Q modificado).
        bcs:              Condiciones de borde del problema.
        T_amb:            Temperatura ambiente [K].
        n_samples:        Número de muestras a generar.
        N_g:              Resolución de la malla.
        k_soil_range:     (k_min, k_max) para k_soil [W/(m K)].
        k_pac_range:      (k_min, k_max) para k_pac; None = sin zona PAC.
        I_range:          (I_min, I_max) para corriente [A].
        Q_d:              Pérdidas dieléctricas [W/m].
        device:           Dispositivo.
        seed:             Semilla aleatoria.
        iec_Q_fn:         Función que devuelve Q_cond dada (section_mm2, material, I_A, T_op).
    """

    def __init__(
        self,
        domain,
        placements: list,
        layers_template: list,
        T_amb: float,
        n_samples: int,
        N_g: int = 64,
        k_soil_range: tuple[float, float] = (1.0, 2.5),
        k_pac_range: tuple[float, float] | None = (1.5, 2.5),
        I_range: tuple[float, float] = (600.0, 1100.0),
        Q_d: float = 0.0,
        device: torch.device | None = None,
        seed: int = 42,
        iec_Q_fn: Callable | None = None,
        pp=None,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.placements = placements
        self.layers_template = layers_template
        self.T_amb = T_amb
        self.n_samples = n_samples
        self.N_g = N_g
        self.k_soil_range = k_soil_range
        self.k_pac_range = k_pac_range
        self.I_range = I_range
        self.Q_d = Q_d
        self.device = device or torch.device("cpu")
        self.iec_Q_fn = iec_Q_fn
        self.pp = pp

        # Pre-generar configuraciones
        rng = torch.Generator().manual_seed(seed)
        self._configs = self._generate_configs(n_samples, rng)

        # Pre-calcular y cachear todos los samples
        self._inputs: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._build_dataset()

    def _generate_configs(self, n: int, rng: torch.Generator) -> list[SampleConfig]:
        configs = []
        k_lo, k_hi = self.k_soil_range
        I_lo, I_hi = self.I_range
        for _ in range(n):
            k_s = float(torch.rand(1, generator=rng).item() * (k_hi - k_lo) + k_lo)
            k_p = None
            if self.k_pac_range is not None:
                pk_lo, pk_hi = self.k_pac_range
                k_p = float(torch.rand(1, generator=rng).item() * (pk_hi - pk_lo) + pk_lo)
                # k_pac >= k_soil (relleno PAC siempre mejor que el suelo)
                k_p = max(k_p, k_s)
            I_v = float(torch.rand(1, generator=rng).item() * (I_hi - I_lo) + I_lo)
            configs.append(SampleConfig(k_soil=k_s, k_pac=k_p, I_A=I_v))
        return configs

    def _build_dataset(self) -> None:
        """Calcula T(x,y) analítico para cada config y rasteriza."""
        import dataclasses
        from pinn_cables.io.readers import override_conductor_Q, CableLayer
        from pinn_cables.physics.k_field import PhysicsParams

        xs = torch.linspace(self.domain.xmin, self.domain.xmax, self.N_g, device=self.device)
        ys = torch.linspace(self.domain.ymin, self.domain.ymax, self.N_g, device=self.device)
        Yg, Xg = torch.meshgrid(ys, xs, indexing="ij")
        xy_flat = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

        for cfg in self._configs:
            # Calcular Q lineal para esta corriente
            if self.iec_Q_fn is not None:
                pl = self.placements[0]
                Q_cond = self.iec_Q_fn(pl.section_mm2, pl.conductor_material, cfg.I_A)
                Q_total = Q_cond + self.Q_d
            else:
                # Aproximación simple: Q ∝ I²
                pl = self.placements[0]
                Q_ref = self.layers_template[0].Q * (math.pi * self.layers_template[0].r_outer**2)
                I_ref = pl.current_A if pl.current_A > 0 else 1000.0
                Q_total = Q_ref * (cfg.I_A / I_ref) ** 2

            layers = override_conductor_Q(self.layers_template, Q_total)
            n_cables = len(self.placements)
            layers_list = [layers] * n_cables
            Q_lins = [Q_total] * n_cables

            # k(x,y): homogéneo o con zona PAC (sigmoid)
            if cfg.k_pac is not None and self.pp is not None:
                pp_var = PhysicsParams(
                    T_ref_R_K=self.pp.T_ref_R_K,
                    alpha_R=self.pp.alpha_R,
                    n_R_iter=self.pp.n_R_iter,
                    k_variable=True,
                    k_good=cfg.k_pac,
                    k_bad=cfg.k_soil,
                    k_cx=self.pp.k_cx,
                    k_cy=self.pp.k_cy,
                    k_width=self.pp.k_width,
                    k_height=self.pp.k_height,
                    k_transition=self.pp.k_transition,
                )
                from pinn_cables.physics.k_field import make_k_functions
                k_fn_pde, _, _ = make_k_functions(pp_var, cfg.k_soil, placements=self.placements)

                def k_fn_sample(xy: torch.Tensor) -> torch.Tensor:
                    return k_fn_pde(xy)
            else:
                k_soil_val = cfg.k_soil

                def k_fn_sample(xy: torch.Tensor) -> torch.Tensor:
                    return torch.full((xy.shape[0], 1), k_soil_val,
                                     device=xy.device, dtype=xy.dtype)

            # Ground-truth analítico (Kennelly = exacto para k uniforme)
            with torch.no_grad():
                T_flat = multilayer_T_multi(
                    xy_flat, layers_list, self.placements,
                    cfg.k_soil, self.T_amb, Q_lins, Q_d=self.Q_d,
                )  # (N_g², 1)
            T_grid = T_flat.reshape(self.N_g, self.N_g)  # (N_g, N_g)

            # Canales de entrada
            from pinn_cables.fno.dataset import make_input_channels
            input_grid, _ = make_input_channels(
                self.domain, self.placements, layers_list,
                k_fn_sample, Q_lins, self.T_amb, self.N_g, self.device,
            )

            self._inputs.append(input_grid)
            # Target: T - T_amb normalizado (se re-escala en la pérdida)
            self._targets.append((T_grid - self.T_amb).unsqueeze(0))  # (1, N_g, N_g)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inputs[idx], self._targets[idx]
