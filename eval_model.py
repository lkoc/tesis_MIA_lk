"""Quick evaluation of the trained Aras 2005 model."""
import sys, math, torch
sys.path.insert(0, ".")

from examples.aras_2005_154kv.run_example import (
    ResidualPINNModel, _multilayer_T, _compute_iec60287_Q,
)
from pinn_cables.io import readers
from pinn_cables.pinn.model import build_model

DATA = "examples/aras_2005_154kv/data"
dom = readers.load_domain(f"{DATA}/domain.csv")
cables = readers.load_placements(f"{DATA}/cables_placement.csv")
layers_raw = readers.load_cable_layers(f"{DATA}/cable_layers.csv")
scenarios = readers.load_scenarios(f"{DATA}/scenarios.csv")
sc = scenarios[0]

placement = cables[0]
k_soil = sc.k_soil
T_amb = sc.T_amb

# Build layers with corrected Q
from pinn_cables.io.readers import CableLayer
cable_layers = list(layers_raw)  # copy

iec = _compute_iec60287_Q(placement, cable_layers, T_amb)
q_lin = iec["Q_total"]
Q_vol = q_lin / (math.pi * cable_layers[0].r_outer ** 2)
cable_layers[0] = CableLayer(
    name=cable_layers[0].name,
    r_inner=cable_layers[0].r_inner,
    r_outer=cable_layers[0].r_outer,
    k=cable_layers[0].k,
    rho_c=cable_layers[0].rho_c,
    Q=Q_vol,
)

# Build model
sp = readers.load_solver_params(f"{DATA}/solver_params.csv")
base = build_model({
    "input_dim": 2, "output_dim": 1,
    "width": sp.model_width, "depth": sp.model_depth,
    "activation": sp.model_activation,
    "fourier_features": sp.model_fourier_features,
    "fourier_scale": sp.model_fourier_scale,
    "fourier_mapping_size": sp.model_fourier_mapping_size,
})

model = ResidualPINNModel(
    base=base, layers=cable_layers, placement=placement,
    k_soil=k_soil, T_amb=T_amb, Q_lin=q_lin, domain=dom,
    normalize=True,
)

# Load weights
ckpt = torch.load("examples/aras_2005_154kv/results/model_final.pt",
                   map_location="cpu", weights_only=False)
model.load_state_dict(ckpt)
model.eval()

# Evaluate at conductor center
pt = torch.tensor([[placement.cx, placement.cy]])
mins = torch.tensor([[dom.xmin, dom.ymin]])
maxs = torch.tensor([[dom.xmax, dom.ymax]])
pt_in = 2.0 * (pt - mins) / (maxs - mins) - 1.0

with torch.no_grad():
    T_cond = model(pt_in).item()

T_ref = T_amb + iec["dT_total"]
PAPER_T = 363.0  # 90 degC

print(f"T_cond PINN  = {T_cond:.2f} K = {T_cond - 273.15:.1f} degC")
print(f"T_cond IEC   = {T_ref:.2f} K = {T_ref - 273.15:.1f} degC")
print(f"T_cond Paper = {PAPER_T:.2f} K = 90.0 degC")
print(f"Error PINN vs IEC   = {T_cond - T_ref:+.2f} K")
print(f"Error PINN vs Paper = {T_cond - PAPER_T:+.2f} K")
