import sys, importlib, json, platform
result = {'python':sys.version, 'executable':sys.executable, 'platform':platform.platform()}
for name in ['dolfinx','gmsh','mpi4py','petsc4py','torch','numpy','scipy','ufl']:
    try:
        module=importlib.import_module(name)
        result[name]=getattr(module,'__version__','installed')
    except ImportError as e:
        result[name]=str(e)
print(json.dumps(result,indent=2))
