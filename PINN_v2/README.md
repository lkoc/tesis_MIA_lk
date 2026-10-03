# PINN_v2

Framework modular y reproducible para comparar familias de PINN (M0, M1, M2, CUSP, DCSNN, FBPINN, RBA/SA)
contra una referencia FEM (FEniCSx) en conducción de calor 2D estacionaria de cables enterrados.

Reglas físicas: -div(k grad T) = Q explícita en cada capa, contacto perfecto en las interfaces,
Robin sin convertir a Dirichlet, Q = p'/A con verificación de la integral. Los datos FEM y del
paper nunca se usan como etiquetas de entrenamiento. I-PINN, AdaI-PINN y VPINN no están implementados.

## Uso rápido

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pinn_v2 help
python -m pinn_v2 cases
python -m pinn_v2 train --case xlpe_single --model M1 --adam-steps 2000 --output results/runs/mi_primera
```

## Interfaz web

```powershell
python front-end/server.py --open     # http://127.0.0.1:8765
```

Permite editar y crear casos, ejecutar FEM o PINN con progreso en vivo, comparar soluciones con mapas
de error y consultar la documentación. Ver `docs/frontend.md`.

## Documentación

Páginas locales en `docs/*.md`; la versión HTML sin conexión se genera con
`python docs/build_docs.py` y queda en `docs/_build/html/index.html`.

## Estructura

- `pinn_v2/`: paquete (casos, modelos, entrenamiento, evaluación, CLI, FEM).
- `configs/`: casos y configuraciones; `tests/`: pruebas.
- `results/`: resultados por fase (no se sobrescriben).
- `front-end/`: interfaz web.
- `REFACTOR_PLAN.md` e `IMPLEMENTATION_LOG.md`: plan y bitácora.

## FEM

Requiere WSL2 con un entorno que tenga `dolfinx` (ver `docs/fem.md`). Todo lo demás funciona sin internet.

## Pruebas

```powershell
python -m pytest -q
```