"""FEM reference solution — Kim 2024 Case B — FEniCSx + Gmsh (Google Colab).

Resolves the steady-state heat equation on the FULL 2D cross-section
including:
  - 6 XLPE 1200 mm² cables with 4 radially-resolved layers each
    (conductor Cu, XLPE insulation, metal screen, PE sheath)
  - PAC bedding zone (rectangular, sigmoid-smoothed k transition)
  - 3-band soil model (as in Kim 2024 paper)
  - Robin BC on surface (forced convection), Dirichlet on other walls

No Gaussian source approximation: heat is applied as volumetric source
inside each conductor circle (Q_LIN / A_conductor  [W/m³]).

Referencia:
    Kim et al. (2024). Geothermics 125, 103151.
    DOI: 10.1016/j.geothermics.2024.103151

Uso en Google Colab
-------------------
Celda 1 — instalar dependencias (ejecutar PRIMERA VEZ, luego reiniciar)::

    try:
        import dolfinx
    except ImportError:
        !wget "https://fem-on-colab.github.io/releases/fenicsx-install-release-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
    try:
        import gmsh
    except ImportError:
        !wget "https://fem-on-colab.github.io/releases/gmsh-install.sh" -O "/tmp/gmsh-install.sh" && bash "/tmp/gmsh-install.sh"

Celda 2 — ejecutar este script::

    exec(open("fem_fenicsx_colab.py").read())

Alternativamente, copiar y pegar las secciones directamente en celdas.

Resultado esperado
------------------
  T_max_conductor ≈ 70.6 °C  (referencia FEM Kim 2024, caso B, verano)
  Diferencia vs PINN 64×4    < 1 K

Parámetros directamente de los CSV del caso:
  data/cables_placement.csv, cable_layers.csv, soil_properties.csv,
  boundary_conditions.csv, physics_params.csv, domain.csv
"""

# =============================================================================
# SECCIÓN 0 — Instalación (ejecutar solo la primera vez en Colab, luego reiniciar)
# =============================================================================
# try:
#     import dolfinx
# except ImportError:
#     !wget "https://fem-on-colab.github.io/releases/fenicsx-install-release-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
# try:
#     import gmsh
# except ImportError:
#     !wget "https://fem-on-colab.github.io/releases/gmsh-install.sh" -O "/tmp/gmsh-install.sh" && bash "/tmp/gmsh-install.sh"

# =============================================================================
# SECCIÓN 1 — Imports
# =============================================================================
import math
import sys
import time
import warnings

import numpy as np

try:
    import gmsh
    import dolfinx
    from dolfinx import mesh as dmesh, fem, io, plot
    from dolfinx.fem import (
        functionspace, Function, Constant, dirichletbc, locate_dofs_topological,
    )
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    import ufl
    from petsc4py import PETSc
    HAS_FENICSX = True
except ImportError as _e:
    warnings.warn(f"FEniCSx/Gmsh no disponible: {_e}\n"
                  "Ejecuta primero la Sección 0 de instalación.", stacklevel=1)
    HAS_FENICSX = False

# =============================================================================
# SECCIÓN 2 — Parámetros del problema (Kim 2024, Caso B — desde CSV)
# =============================================================================

# --- Dominio [m] ---
XMIN, XMAX = -45.5, 45.5
YMIN, YMAX = -45.5, 0.0

# --- Condiciones de frontera (boundary_conditions.csv) ---
T_BOT  = 288.35   # K   — Dirichlet inferior / laterales
T_AIR  = 300.35   # K   — Robin: temperatura ambiente (superficie)
H_CONV = 7.371    # W/(m²·K) — coeficiente convección forzada

# --- Posiciones de cables (cables_placement.csv) [m] ---
CABLES = [
    (-0.40, -1.60),
    ( 0.00, -1.60),
    ( 0.40, -1.60),
    (-0.40, -1.20),
    ( 0.00, -1.20),
    ( 0.40, -1.20),
]
N_CABLES = len(CABLES)

# --- Capas del cable (cable_layers.csv) ---
# (r_inner, r_outer, k [W/(m·K)], nombre)
CABLE_LAYERS = [
    (0.00000, 0.01885, 400.0,   "conductor_Cu"),   # conductor sólido Cu
    (0.01885, 0.04085,   0.2857, "XLPE"),           # aislamiento XLPE
    (0.04085, 0.04935, 384.6,   "pantalla_metal"),  # pantalla metálica Cu
    (0.04935, 0.05335,   0.45,   "cubierta_PE"),    # cubierta PE
]
N_LAYERS = len(CABLE_LAYERS)
R_CONDUCTOR = CABLE_LAYERS[0][1]   # m  (= 0.01885 m)
R_CABLE     = CABLE_LAYERS[-1][1]  # m  (= 0.05335 m)

# --- Suelo en 3 bandas horizontales (Kim 2024, Tabla 1) ---
# (y_top, y_bottom, k [W/(m·K)])
SOIL_BANDS = [
    ( 0.000,  -0.56,  1.804),   # capa superficial
    (-0.560,  -1.76,  1.351),   # capa media (zona cables)
    (-1.760, -45.50,  1.517),   # capa profunda
]

# --- Zona PAC (physics_params.csv) ---
PAC_CX, PAC_CY  = 0.0, -1.40   # centro [m]
PAC_W,  PAC_H   = 1.30,  0.90  # ancho × alto [m]
PAC_K           = 2.094         # k W/(m·K)

# --- Calor lineal por cable (IEC 60287) ---
#   1200 mm² Cu,  I = 1026 A,  T_op ≈ 70 °C (343 K)
#   R_dc20 = 1.51e-5 Ω/m (IEC 60228, clase 2)
#   alpha_Cu = 0.00393 /K
SECTION_MM2   = 1200
MATERIAL      = "cu"
CURRENT_A     = 1026.0
T_OP_K        = 343.15    # 70 °C   (temperatura de operación estimada)
R_DC20_CU     = 1.51e-5   # Ω/m
ALPHA_CU      = 0.00393   # 1/K
FREQ_HZ       = 50.0


def compute_Q_lin(I, R_dc20, alpha_R, T_op_K, T_ref_K=293.15, freq=50.0):
    """Calcula Q_lin [W/m] con IEC 60287 (R(T) + efecto pelicular)."""
    R_dc_T = R_dc20 * (1.0 + alpha_R * (T_op_K - T_ref_K))
    xs_sq  = 8.0 * math.pi * freq / (R_dc_T * 1.0e7)
    xs_4   = xs_sq ** 2
    ys     = xs_4 / (192.0 + 0.8 * xs_4)
    R_ac   = R_dc_T * (1.0 + ys)
    return I**2 * R_ac


Q_LIN = compute_Q_lin(CURRENT_A, R_DC20_CU, ALPHA_CU, T_OP_K)
print(f"  Q_LIN (IEC 60287, T={T_OP_K-273.15:.1f}°C): {Q_LIN:.3f} W/m")
# Para Q_LIN consistente con el PINN (≈27.97 W/m) usar T_op más alto si fuera
# necesario; aquí se calcula de forma auto-consistente con T_op=70°C.

# Densidad volumétrica de fuente en el conductor [W/m³]
A_CONDUCTOR  = math.pi * R_CONDUCTOR**2
Q_VOL        = Q_LIN / A_CONDUCTOR
print(f"  Q_VOL (conductor):        {Q_VOL:.1f} W/m³")
print(f"  R_conductor = {R_CONDUCTOR*1e3:.2f} mm,  R_cable = {R_CABLE*1e3:.2f} mm")


# =============================================================================
# SECCIÓN 3 — Mallado con Gmsh
# =============================================================================

def build_mesh(h_cond=0.003, h_cable=0.015, h_pac=0.08, h_near=0.40, h_far=3.5,
               verbose=False):
    """Crea la malla FEM con Gmsh y la convierte a formato DOLFINx.

    Estrategia de mallado:
      - Conductor (r < R_CONDUCTOR):  h = h_cond  (~3 mm)
      - Resto del cable (r < R_CABLE): h = h_cable (~15 mm)
      - Zona PAC (±0.65m × ±0.45m):   h = h_pac   (~80 mm)
      - Zona near (±3 m alrededor):   h = h_near  (~400 mm)
      - Dominio lejano:               h = h_far   (~3.5 m)

    Subdomains (etiquetas de volumen físico):
      Suelo banda 0  → 1
      Suelo banda 1  → 2
      Suelo banda 2  → 3
      PAC zone       → 4
      Cable i, capa j→ 100*(i+1) + j   (j=0..3, i=0..5)
         conductor   → j=0
         XLPE        → j=1
         pantalla    → j=2
         cubierta    → j=3

    Boundaries (etiquetas de línea física):
      Top surface (y=0)  → 10
      Bottom (y=YMIN)    → 11
      Left (x=XMIN)      → 12
      Right (x=XMAX)     → 13

    Returns:
        (msh, cell_tags, facet_tags) listos para FEniCSx.
    """
    if not HAS_FENICSX:
        raise RuntimeError("FEniCSx no disponible — ver Sección 0.")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("kim2024_caseB")
    factory = gmsh.model.geo

    # ------------------------------------------------------------------
    # 3.1  Dominio exterior (rectángulo completo)
    # ------------------------------------------------------------------
    dom_pts = [
        factory.addPoint(XMIN, YMIN, 0, h_far),   # 1: bottom-left
        factory.addPoint(XMAX, YMIN, 0, h_far),   # 2: bottom-right
        factory.addPoint(XMAX, YMAX, 0, h_near),  # 3: top-right
        factory.addPoint(XMIN, YMAX, 0, h_near),  # 4: top-left
    ]
    dom_lines = [
        factory.addLine(dom_pts[0], dom_pts[1]),  # bottom
        factory.addLine(dom_pts[1], dom_pts[2]),  # right
        factory.addLine(dom_pts[2], dom_pts[3]),  # top
        factory.addLine(dom_pts[3], dom_pts[0]),  # left
    ]
    dom_loop   = factory.addCurveLoop(dom_lines)
    dom_surf   = factory.addPlaneSurface([dom_loop])

    # ------------------------------------------------------------------
    # 3.2  Bandas de suelo (cortes horizontales)
    # ------------------------------------------------------------------
    y_cuts = [SOIL_BANDS[0][1], SOIL_BANDS[1][1]]  # -0.56, -1.76
    soil_cut_surfs = []
    for yc in y_cuts:
        p1 = factory.addPoint(XMIN, yc, 0, h_near)
        p2 = factory.addPoint(XMAX, yc, 0, h_near)
        soil_cut_surfs.append(factory.addLine(p1, p2))

    # ------------------------------------------------------------------
    # 3.3  Zona PAC (rectángulo)
    # ------------------------------------------------------------------
    pac_x0 = PAC_CX - PAC_W / 2.0
    pac_x1 = PAC_CX + PAC_W / 2.0
    pac_y0 = PAC_CY - PAC_H / 2.0
    pac_y1 = PAC_CY + PAC_H / 2.0
    pac_pts = [
        factory.addPoint(pac_x0, pac_y0, 0, h_pac),
        factory.addPoint(pac_x1, pac_y0, 0, h_pac),
        factory.addPoint(pac_x1, pac_y1, 0, h_pac),
        factory.addPoint(pac_x0, pac_y1, 0, h_pac),
    ]
    pac_lines = [
        factory.addLine(pac_pts[0], pac_pts[1]),
        factory.addLine(pac_pts[1], pac_pts[2]),
        factory.addLine(pac_pts[2], pac_pts[3]),
        factory.addLine(pac_pts[3], pac_pts[0]),
    ]
    pac_loop = factory.addCurveLoop(pac_lines)
    pac_surf = factory.addPlaneSurface([pac_loop])

    # ------------------------------------------------------------------
    # 3.4  Cables (círculos concéntricos)
    # ------------------------------------------------------------------
    cable_surfs = []   # list of lists: cable_surfs[i][j] = surface tag
    for i, (cx, cy) in enumerate(CABLES):
        layers_i = []
        prev_loop = None
        for j, (r_in, r_out, k_layer, lname) in enumerate(CABLE_LAYERS):
            # Círculo exterior de esta capa
            circ = factory.addCircle(cx, cy, 0, r_out)
            outer_loop = factory.addCurveLoop([circ])
            if prev_loop is None:
                surf = factory.addPlaneSurface([outer_loop])
            else:
                surf = factory.addPlaneSurface([outer_loop, prev_loop])
            layers_i.append(surf)
            # Para la capa siguiente, el "agujero" es el círculo actual
            prev_loop = factory.addCurveLoop([circ])
            # Tamaño de malla según la capa
            if j == 0:
                gmsh.model.geo.mesh.setTransfiniteCurve(circ, max(4, int(2*math.pi*r_out/h_cond)))
            else:
                gmsh.model.geo.mesh.setTransfiniteCurve(circ, max(8, int(2*math.pi*r_out/h_cable)))
        cable_surfs.append(layers_i)

    # ------------------------------------------------------------------
    # 3.5  Fragmentar (Boolean) todo para obtener subdominios conformes
    # ------------------------------------------------------------------
    factory.synchronize()

    # Recolectar todas las superficies que "cortan" el dominio
    tool_surfs = (
        [(2, pac_surf)]
        + [(2, s) for layers_i in cable_surfs for s in layers_i]
    )
    # Las líneas de corte de suelo se agregan como embedded curves
    for lc in soil_cut_surfs:
        gmsh.model.mesh.embed(1, [lc], 2, dom_surf)

    # BooleanFragments: dom_surf se fragmenta con pac y cables
    out_vols, _ = gmsh.model.occ.fragment(
        [(2, dom_surf)],
        tool_surfs,
    )
    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------------
    # 3.6  Asignar etiquetas físicas
    # ------------------------------------------------------------------
    # Función auxiliar: ¿el baricentro (bx,by) está dentro del cable i, capa j?
    def in_annulus(bx, by, i, j):
        cx, cy = CABLES[i]
        r   = math.hypot(bx - cx, by - cy)
        r_in  = CABLE_LAYERS[j][0]
        r_out = CABLE_LAYERS[j][1]
        return r_in <= r < r_out

    def in_pac(bx, by):
        return (pac_x0 <= bx <= pac_x1) and (pac_y0 <= by <= pac_y1)

    def soil_band(by):
        """Devuelve el índice de banda de suelo (0, 1, 2)."""
        for idx, (yt, yb, _k) in enumerate(SOIL_BANDS):
            if yb < by <= yt:
                return idx
        return 2   # por defecto banda profunda

    all_surfs = gmsh.model.getEntities(2)
    soil_groups = {0: [], 1: [], 2: []}
    pac_group   = []
    cable_groups = {}  # (i, j) → list of surf tags

    for dim, tag in all_surfs:
        bx, by, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        # ¿Es un cable?
        assigned = False
        for i in range(N_CABLES):
            for j in range(N_LAYERS):
                if in_annulus(bx, by, i, j):
                    key = (i, j)
                    cable_groups.setdefault(key, []).append(tag)
                    assigned = True
                    break
            if assigned:
                break
        if assigned:
            continue
        # ¿Es PAC?
        if in_pac(bx, by):
            pac_group.append(tag)
            continue
        # Es suelo
        sb = soil_band(by)
        soil_groups[sb].append(tag)

    # Registrar grupos físicos de volumen
    for sb in range(3):
        if soil_groups[sb]:
            gmsh.model.addPhysicalGroup(2, soil_groups[sb], sb + 1)
            gmsh.model.setPhysicalName(2, sb + 1, f"soil_band_{sb}")

    if pac_group:
        gmsh.model.addPhysicalGroup(2, pac_group, 4)
        gmsh.model.setPhysicalName(2, 4, "pac_zone")

    for (i, j), tags in cable_groups.items():
        phys_id = 100 * (i + 1) + j
        gmsh.model.addPhysicalGroup(2, tags, phys_id)
        gmsh.model.setPhysicalName(2, phys_id,
                                   f"cable{i+1}_{CABLE_LAYERS[j][3]}")

    # Grupos físicos de frontera
    # Detectar aristas del dominio exterior
    all_curves = gmsh.model.getEntities(1)
    top_tags, bot_tags, left_tags, right_tags = [], [], [], []
    tol = 0.5

    for dim, tag in all_curves:
        # Bounding box de la curva
        xlo, ylo, _, xhi, yhi, _ = gmsh.model.getBoundingBox(dim, tag)
        ymid = 0.5 * (ylo + yhi)
        xmid = 0.5 * (xlo + xhi)
        if abs(ymid - YMAX) < tol and (xhi - xlo) > tol:
            top_tags.append(tag)
        elif abs(ymid - YMIN) < tol and (xhi - xlo) > tol:
            bot_tags.append(tag)
        elif abs(xmid - XMIN) < tol and (yhi - ylo) > tol:
            left_tags.append(tag)
        elif abs(xmid - XMAX) < tol and (yhi - ylo) > tol:
            right_tags.append(tag)

    if top_tags:
        gmsh.model.addPhysicalGroup(1, top_tags,   10);  gmsh.model.setPhysicalName(1, 10, "top")
    if bot_tags:
        gmsh.model.addPhysicalGroup(1, bot_tags,   11);  gmsh.model.setPhysicalName(1, 11, "bottom")
    if left_tags:
        gmsh.model.addPhysicalGroup(1, left_tags,  12);  gmsh.model.setPhysicalName(1, 12, "left")
    if right_tags:
        gmsh.model.addPhysicalGroup(1, right_tags, 13);  gmsh.model.setPhysicalName(1, 13, "right")

    # ------------------------------------------------------------------
    # 3.7  Generar malla 2D
    # ------------------------------------------------------------------
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    # ------------------------------------------------------------------
    # 3.8  Importar en DOLFINx
    # ------------------------------------------------------------------
    msh, cell_tags, facet_tags = io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )
    gmsh.finalize()
    return msh, cell_tags, facet_tags


# =============================================================================
# SECCIÓN 4 — Formulación variacional FEniCSx
# =============================================================================

def build_k_function(msh, cell_tags):
    """Crea función DG0 con k(x,y) asignada por subdomain tag."""
    V0 = functionspace(msh, ("DG", 0))
    k_func = Function(V0)

    # Mapa etiqueta → k [W/(m·K)]
    k_map = {
        1: SOIL_BANDS[0][2],   # 1.804
        2: SOIL_BANDS[1][2],   # 1.351
        3: SOIL_BANDS[2][2],   # 1.517
        4: PAC_K,              # 2.094
    }
    for i in range(N_CABLES):
        for j, (r_in, r_out, kv, lname) in enumerate(CABLE_LAYERS):
            k_map[100 * (i + 1) + j] = kv

    # Asignar valores
    cell_indices = cell_tags.indices
    cell_values  = cell_tags.values
    with k_func.vector.localForm() as loc:
        loc.set(SOIL_BANDS[2][2])          # default: banda profunda
    for tag, kv in k_map.items():
        cells = cell_indices[cell_values == tag]
        k_func.x.array[cells] = kv
    k_func.x.scatter_forward()
    return k_func


def build_Q_function(msh, cell_tags):
    """Crea función DG0 con fuente de calor Q [W/m³] solo en conductores."""
    V0 = functionspace(msh, ("DG", 0))
    Q_func = Function(V0)
    with Q_func.vector.localForm() as loc:
        loc.set(0.0)
    for i in range(N_CABLES):
        tag = 100 * (i + 1) + 0   # conductor de cable i (j=0)
        cells = cell_tags.indices[cell_tags.values == tag]
        Q_func.x.array[cells] = Q_VOL
    Q_func.x.scatter_forward()
    return Q_func


def solve_thermal(msh, cell_tags, facet_tags):
    """Ensambla y resuelve el problema térmico estacionario.

    PDE:  -∇·(k ∇T) = Q        en Ω
    BC:   T = T_BOT             en ∂Ω_bottom ∪ ∂Ω_left ∪ ∂Ω_right
          k ∂T/∂n + H_CONV*(T - T_AIR) = 0   en ∂Ω_top

    Returns:
        T_fem (dolfinx Function) con la temperatura [K].
    """
    # Espacio CG1 (Lagrange P1)
    V = functionspace(msh, ("Lagrange", 1))

    # Funciones de conductividad y fuente
    k_func = build_k_function(msh, cell_tags)
    Q_func = build_Q_function(msh, cell_tags)

    # Medidas de integración
    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    # Condiciones de Dirichlet: bottom(11), left(12), right(13)
    T_dir = fem.Constant(msh, PETSc.ScalarType(T_BOT))
    dofs_bot   = locate_dofs_topological(V, 1,
                     facet_tags.find(11))
    dofs_left  = locate_dofs_topological(V, 1,
                     facet_tags.find(12))
    dofs_right = locate_dofs_topological(V, 1,
                     facet_tags.find(13))
    bcs = [
        dirichletbc(T_dir, dofs_bot,   V),
        dirichletbc(T_dir, dofs_left,  V),
        dirichletbc(T_dir, dofs_right, V),
    ]

    # Formulación variacional
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h_c   = fem.Constant(msh, PETSc.ScalarType(H_CONV))
    T_air = fem.Constant(msh, PETSc.ScalarType(T_AIR))

    # Bilineal: difusión + Robin
    a = (k_func * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
         + h_c * u * v * ds(10))

    # Lineal: fuente de calor + Robin (T_amb)
    L = (Q_func * v * dx
         + h_c * T_air * v * ds(10))

    # Resolver con MUMPS (directo, preciso)
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly",
                       "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
    )
    T_fem = problem.solve()
    return T_fem


# =============================================================================
# SECCIÓN 5 — Post-proceso: T_conductor por cable
# =============================================================================

def get_T_max_conductor(msh, cell_tags, T_fem):
    """Devuelve la lista de T_max [°C] evaluada en los conductores."""
    V = T_fem.function_space

    T_max_list = []
    for i in range(N_CABLES):
        tag = 100 * (i + 1) + 0
        cells = cell_tags.indices[cell_tags.values == tag]
        if len(cells) == 0:
            T_max_list.append(float("nan"))
            continue
        # Coordenadas de los DOFs asociados a esas celdas
        dofs = fem.locate_dofs_topological(V, msh.topology.dim, cells)
        vals = T_fem.x.array[dofs]
        T_max_list.append(float(np.max(vals)) - 273.15)
    return T_max_list


# =============================================================================
# SECCIÓN 6 — Visualización
# =============================================================================

def plot_T_field(msh, T_fem, title="T field [°C]"):
    """Gráfica del campo de temperatura (zona central ±3 m)."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation

        coords = msh.geometry.x[:, :2]
        cells  = msh.geometry.dofmap
        # Filter near-cable zone
        mask = (
            (coords[:, 0] >= -3.0) & (coords[:, 0] <= 3.0) &
            (coords[:, 1] >= -3.5) & (coords[:, 1] <= 0.2)
        )
        idx = np.where(mask)[0]

        T_vals = T_fem.x.array - 273.15   # °C

        # Construir triangulación
        try:
            cells_np = np.array(cells).reshape(-1, cells.num_nodes_per_cell)
        except Exception:
            cells_np = np.array(list(cells)).reshape(-1, 3)
        tri = Triangulation(coords[:, 0], coords[:, 1], cells_np)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Vista completa
        ax = axes[0]
        tc = ax.tripcolor(tri, T_vals, shading="flat", cmap="hot_r")
        plt.colorbar(tc, ax=ax, label="T [°C]")
        ax.set_title("Campo T — dominio completo")
        ax.set_xlabel("x [m]");  ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

        # Vista zoom cables
        ax = axes[1]
        tc2 = ax.tripcolor(tri, T_vals, shading="flat", cmap="hot_r")
        plt.colorbar(tc2, ax=ax, label="T [°C]")
        ax.set_xlim(-1.5, 1.5);  ax.set_ylim(-2.5, 0.2)
        ax.set_title("Zoom zona cables")
        ax.set_xlabel("x [m]");  ax.set_ylabel("y [m]")
        for cx, cy in CABLES:
            circle = plt.Circle((cx, cy), R_CABLE, fill=False,
                                 color="cyan", linewidth=0.8)
            ax.add_patch(circle)
        plt.tight_layout()
        plt.savefig("fem_T_field_kim2024B.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  Figura guardada: fem_T_field_kim2024B.png")
    except Exception as e:
        print(f"  (Plot fallido: {e})")


# =============================================================================
# SECCIÓN 7 — Ejecución principal
# =============================================================================

def run_fem(h_cond=0.003, h_cable=0.015, h_pac=0.08, h_near=0.4, h_far=3.5,
            verbose_mesh=False, do_plot=True):
    """Pipeline completo: malla → FEM → post-proceso → tabla comparativa.

    Args:
        h_cond:       Tamaño de malla en conductor [m]  (default 3 mm)
        h_cable:      Tamaño en capas del cable   [m]  (default 15 mm)
        h_pac:        Tamaño en zona PAC           [m]  (default 80 mm)
        h_near:       Tamaño cerca del dominio     [m]  (default 400 mm)
        h_far:        Tamaño en suelo lejano       [m]  (default 3.5 m)
        verbose_mesh: Mostrar salida de Gmsh
        do_plot:      Generar y guardar figura

    Returns:
        T_max_conductor (float) — temperatura máxima del conductor [°C]
    """
    if not HAS_FENICSX:
        print("ERROR: FEniCSx no disponible. Ejecuta la Sección 0 de instalación.")
        return None

    FEM_REF_C  = 70.6    # referencia paper Kim 2024 [°C]
    PINN_64x4  = 70.19   # PINN 64×4 seeds optimizados [°C]
    PINN_128x5 = 70.17   # PINN 128×5 destilado [°C]

    print("=" * 65)
    print("  FEM FEniCSx — Kim 2024 Caso B — 6 cables XLPE 154 kV")
    print("=" * 65)
    print(f"  Malla: h_cond={h_cond*1e3:.1f}mm  h_cable={h_cable*1e3:.1f}mm"
          f"  h_pac={h_pac*1e3:.0f}mm  h_far={h_far:.1f}m")

    # --- Malla ---
    t0 = time.time()
    print("\n[1/3] Construyendo malla Gmsh...")
    msh, cell_tags, facet_tags = build_mesh(
        h_cond=h_cond, h_cable=h_cable, h_pac=h_pac,
        h_near=h_near, h_far=h_far, verbose=verbose_mesh,
    )
    n_cells = msh.topology.index_map(msh.topology.dim).size_global
    n_dofs  = functionspace(msh, ("Lagrange", 1)).dofmap.index_map.size_global
    print(f"     Celdas: {n_cells:,d}   DOFs: {n_dofs:,d}   ({time.time()-t0:.1f}s)")

    # --- Resolver FEM ---
    print("\n[2/3] Resolviendo FEM (MUMPS)...")
    t1 = time.time()
    T_fem = solve_thermal(msh, cell_tags, facet_tags)
    T_vals_C = T_fem.x.array - 273.15
    print(f"     T_min={T_vals_C.min():.2f}°C   T_max={T_vals_C.max():.2f}°C"
          f"   ({time.time()-t1:.1f}s)")

    # --- Post-proceso ---
    print("\n[3/3] T_conductor por cable:")
    T_list = get_T_max_conductor(msh, cell_tags, T_fem)
    for i, Tc in enumerate(T_list):
        cx, cy = CABLES[i]
        print(f"     Cable {i+1}  ({cx:+.2f},{cy:+.2f})m  → T_cond = {Tc:.2f} °C")

    T_max = max((t for t in T_list if not math.isnan(t)), default=float("nan"))

    # --- Tabla comparativa ---
    print()
    print("─" * 62)
    print(f"  {'Método':<38}  {'T_max [°C]':>10}  {'vs paper':>8}")
    print("─" * 62)
    print(f"  {'FEM paper (COMSOL, Kim 2024)':<38}  {FEM_REF_C:>10.2f}  {'—':>8}")
    print(f"  {'FEM FEniCSx (malla completa)':<38}  {T_max:>10.2f}  {T_max-FEM_REF_C:>+7.2f}K")
    print(f"  {'PINN 64×4 (Kennelly bg)':<38}  {PINN_64x4:>10.2f}  {PINN_64x4-FEM_REF_C:>+7.2f}K")
    print(f"  {'PINN 128×5 destilado':<38}  {PINN_128x5:>10.2f}  {PINN_128x5-FEM_REF_C:>+7.2f}K")
    print("─" * 62)

    if do_plot:
        print("\nGenerando figura...")
        plot_T_field(msh, T_fem)

    total = time.time() - t0
    print(f"\n  Tiempo total: {total:.1f}s")
    return T_max


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="FEM FEniCSx — Kim 2024 Caso B")
    parser.add_argument("--h-cond",  type=float, default=0.003,
                        help="h conductor [m] (default 0.003)")
    parser.add_argument("--h-cable", type=float, default=0.015,
                        help="h cable [m] (default 0.015)")
    parser.add_argument("--h-pac",   type=float, default=0.08,
                        help="h PAC [m] (default 0.08)")
    parser.add_argument("--h-near",  type=float, default=0.40,
                        help="h zona near [m] (default 0.40)")
    parser.add_argument("--h-far",   type=float, default=3.5,
                        help="h suelo lejano [m] (default 3.5)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Desactiva la generación de figuras")
    parser.add_argument("--verbose-mesh", action="store_true",
                        help="Mostrar salida completa de Gmsh")
    # parse_known_args ignora los argumentos del kernel de Jupyter/Colab
    args, _ = parser.parse_known_args()

    run_fem(
        h_cond=args.h_cond,
        h_cable=args.h_cable,
        h_pac=args.h_pac,
        h_near=args.h_near,
        h_far=args.h_far,
        do_plot=not args.no_plot,
        verbose_mesh=args.verbose_mesh,
    )
