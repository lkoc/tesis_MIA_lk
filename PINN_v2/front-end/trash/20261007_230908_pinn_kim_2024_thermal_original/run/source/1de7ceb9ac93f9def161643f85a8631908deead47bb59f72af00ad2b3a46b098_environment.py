"""Print the FEniCSx environment versions as JSON; run inside the FEniCSx environment."""
import json
import platform
import sys


def versions():
    import dolfinx
    import gmsh
    import numpy
    import petsc4py
    import mpi4py
    return dict(python=sys.version.split()[0], platform=platform.platform(), dolfinx=dolfinx.__version__, gmsh=gmsh.__version__,
        petsc4py=petsc4py.__version__, mpi4py=mpi4py.__version__, numpy=numpy.__version__)


if __name__ == '__main__':
    print(json.dumps(versions()))