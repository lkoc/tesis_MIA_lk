#!/usr/bin/env bash
# =============================================================
# setup_wsl_fenicsx.sh
# Instala FEniCSx 0.9 + gmsh + OpenCL Intel en Ubuntu 24.04 WSL2
#
# Uso (dentro de WSL o desde PowerShell):
#   wsl -d Ubuntu -- bash /mnt/c/.../setup_wsl_fenicsx.sh
#   — o —
#   En una terminal WSL:  bash setup_wsl_fenicsx.sh
#
# Qué hace:
#   1. Instala dependencias del sistema (libGL, OpenCL runtime Intel,
#      clinfo, MPI, libXrender, etc.)
#   2. Instala Miniforge3 en $HOME/miniforge3 (si no existe)
#   3. Crea el entorno conda "fenicsx" con:
#         python 3.12, fenics-dolfinx 0.9, gmsh, meshio,
#         mpi4py, petsc4py, ipykernel, matplotlib, scipy
#   4. Registra el kernel Jupyter "FEniCSx WSL (Ubuntu 24.04)"
#   5. Instala Intel GPU compute-runtime para OpenCL
#      (disponible como dispositivo OpenCL; PETSc usa MPI/CPU)
#
# GPU / OpenCL:
#   El Intel Arc/Xe iGPU (Core Ultra 7 165U) es accesible vía /dev/dxg
#   y OpenCL (intel-opencl-icd / intel-level-zero-gpu).
#   FEniCSx/PETSc NO tiene backend OpenCL; la aceleración GPU sólo
#   existe para CUDA (NVIDIA) o HIP (AMD ROCm).
#   Con este hardware, el camino óptimo es MPI multi-proceso:
#     mpirun -n $(nproc) python fem_fenicsx_colab.py
# =============================================================

set -euo pipefail

MINIFORGE_DIR="$HOME/miniforge3"
ENV_NAME="fenicsx"
KERNEL_NAME="fenicsx-wsl"
KERNEL_DISPLAY="FEniCSx WSL (Ubuntu 24.04)"

# ── 0. Verificar que estamos en Ubuntu/WSL ──────────────────
if ! grep -qi "ubuntu" /etc/os-release 2>/dev/null; then
    echo "ADVERTENCIA: Este script está diseñado para Ubuntu." >&2
fi
if [ ! -e /dev/dxg ]; then
    echo "ADVERTENCIA: /dev/dxg no encontrado. La GPU podría no estar disponible." >&2
fi

# ── 1. Dependencias del sistema ─────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  1/5  Instalando dependencias del sistema            ║"
echo "╚══════════════════════════════════════════════════════╝"
sudo apt-get update -qq
sudo apt-get install -y -q \
    libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 \
    libgl1 libgomp1 \
    openmpi-bin libopenmpi-dev \
    ocl-icd-opencl-dev clinfo \
    curl wget git build-essential

# ── 2. Intel compute-runtime (OpenCL para Arc/Xe iGPU) ──────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  2/5  Instalando Intel OpenCL compute-runtime        ║"
echo "╚══════════════════════════════════════════════════════╝"
# Intel no publica paquetes oficiales .deb para noble (24.04) aún;
# usamos el repositorio de Intel Neo / level-zero
if ! dpkg -l | grep -q intel-opencl-icd 2>/dev/null; then
    # Añadir repo Intel para compute-runtime
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
        | sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg || true
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu noble unified" \
        | sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list > /dev/null || true
    sudo apt-get update -qq || true
    # Intentar instalar; si falla, continuar sin GPU OpenCL
    sudo apt-get install -y -q \
        intel-opencl-icd intel-level-zero-gpu level-zero 2>/dev/null \
        || echo "  ⚠ Intel OpenCL ICD no disponible para noble; omitido."
else
    echo "  ✓ intel-opencl-icd ya instalado."
fi

# ── 3. Miniforge (conda) ────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  3/5  Instalando / verificando Miniforge3            ║"
echo "╚══════════════════════════════════════════════════════╝"
if [ ! -d "$MINIFORGE_DIR" ]; then
    echo "  Descargando Miniforge3..."
    curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -o /tmp/Miniforge3.sh
    bash /tmp/Miniforge3.sh -b -p "$MINIFORGE_DIR"
    rm -f /tmp/Miniforge3.sh
    echo "  ✓ Miniforge instalado en $MINIFORGE_DIR"
else
    echo "  ✓ Miniforge ya existe en $MINIFORGE_DIR"
fi

# Activar conda en esta shell
# shellcheck source=/dev/null
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

# Inicializar conda para bash (solo si aún no está en .bashrc)
grep -q "miniforge3/etc/profile.d/conda.sh" "$HOME/.bashrc" 2>/dev/null \
    || conda init bash >/dev/null 2>&1

# ── 4. Entorno conda fenicsx ────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  4/5  Creando entorno conda '$ENV_NAME'              ║"
echo "╚══════════════════════════════════════════════════════╝"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  ✓ Entorno '$ENV_NAME' ya existe. Actualizando paquetes clave..."
    conda install -n "$ENV_NAME" -c conda-forge --quiet -y \
        "fenics-dolfinx>=0.9" gmsh meshio mpi4py petsc4py \
        ipykernel matplotlib scipy numpy
else
    echo "  Creando entorno (puede tardar 5-10 min)..."
    conda create -n "$ENV_NAME" -c conda-forge -y \
        python=3.12 \
        "fenics-dolfinx>=0.9" \
        gmsh \
        meshio \
        mpi4py \
        petsc4py \
        ipykernel \
        matplotlib \
        scipy \
        numpy \
        jupyterlab
fi

# ── 5. Registrar kernel Jupyter ─────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  5/5  Registrando kernel Jupyter                     ║"
echo "╚══════════════════════════════════════════════════════╝"
conda run -n "$ENV_NAME" python -m ipykernel install --user \
    --name "$KERNEL_NAME" \
    --display-name "$KERNEL_DISPLAY"
echo "  ✓ Kernel registrado: $KERNEL_DISPLAY"

# ── Verificación final ──────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Verificación de la instalación                      ║"
echo "╚══════════════════════════════════════════════════════╝"
conda run -n "$ENV_NAME" python - <<'EOF'
import dolfinx, gmsh, mpi4py, petsc4py
print(f"  ✓ dolfinx  {dolfinx.__version__}")
print(f"  ✓ gmsh     {gmsh.__version__}")
print(f"  ✓ mpi4py   {mpi4py.__version__}")
print(f"  ✓ petsc4py {petsc4py.__version__}")
EOF

echo ""
echo "── OpenCL (informativo) ──"
if command -v clinfo &>/dev/null; then
    clinfo -l 2>/dev/null || echo "  (sin plataformas OpenCL detectadas)"
else
    echo "  clinfo no disponible"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  INSTALACIÓN COMPLETA"
echo ""
echo "  Para usar el notebook desde VS Code:"
echo "  ┌─────────────────────────────────────────────────────"
echo "  │ Opción A — VS Code Remote WSL (recomendado):"
echo "  │   1. Ctrl+Shift+P → 'Remote-WSL: Reopen in WSL'"
echo "  │   2. Abrir: examples/kim_2024_154kv_optim_C/"
echo "  │            fem_fenicsx_colab.ipynb"
echo "  │   3. Kernel: '$KERNEL_DISPLAY'"
echo "  │"
echo "  │ Opción B — JupyterLab en WSL:"
echo "  │   wsl -d Ubuntu -- bash -c \\"
echo "  │     'source ~/miniforge3/etc/profile.d/conda.sh && \\"
echo "  │      conda activate fenicsx && \\"
echo "  │      jupyter lab --no-browser --port=8888'"
echo "  │   Abrir en navegador: http://localhost:8888"
echo "  │"
echo "  │ Para paralelizar con MPI:"
echo "  │   conda run -n fenicsx mpirun -n 4 python run_fem_C.py"
echo "  └─────────────────────────────────────────────────────"
echo ""
