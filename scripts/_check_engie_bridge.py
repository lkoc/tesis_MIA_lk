"""Verificacion estatica de engie_bridge.py.

Comprueba que no queden llamadas a funciones inexistentes tras las ediciones
incrementales del script (un fallo asi no lo detecta py_compile, porque solo
se manifiesta en tiempo de ejecucion).
"""
import ast
import builtins
import sys

PATH = r"c:\usr\ths_mia_fiis\tesis_MIA_lk\scripts\engie_bridge.py"


def main():
    with open(PATH, "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)

    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    # Los imports done dentro de funciones (p.ej. sync_playwright) tambien son
    # nombres validos en tiempo de ejecucion.
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.asname or a.name.split(".")[0]
                            for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)

    known = defined | imported | set(dir(builtins))

    missing = sorted(name for name in called if name not in known)
    print("DEFINED:", ", ".join(sorted(defined)))
    print("IMPORTED:", ", ".join(sorted(imported)))
    print("UNRESOLVED:", missing or "ninguna")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
