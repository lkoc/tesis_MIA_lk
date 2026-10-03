# Inicio rápido

## Requisitos

- Python 3.12 con `numpy` y `scipy`; PyTorch para entrenar (la configuración usada es CPU, `float64`).
- Opcional, solo para la referencia FEM: WSL2 con un entorno que tenga `dolfinx` (ver [Referencia FEM](fem.md)).
- No necesita conexión a internet para ejecutar nada de lo descrito aquí.

## Instalación

Desde la carpeta `PINN_v2`:

```bash
pip install -e ".[dev]"
```

Si prefieres no instalar, basta con definir la ruta del paquete en la sesión de PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

## Primeros comandos

```bash
python -m pinn_v2 help            # ayuda integrada
python -m pinn_v2 cases           # casos disponibles
python -m pinn_v2 show xlpe_single
```

## Primer entrenamiento

```bash
python -m pinn_v2 train --case xlpe_single --model M1 --adam-steps 2000 --output results/runs/mi_primera
```

Al terminar, la carpeta de salida contiene `result.json`, `evaluation.npz`, `collocation.npz`,
`model.pt` y `manifest.json`. La consola resume el estado, la pérdida final y el veredicto de aceptación.

> El veredicto `no_reference` aparece cuando no hay solución de referencia. Para comparar con FEM,
> ver [Referencia FEM](fem.md).

## Cambiar de modelo

Solo cambia `--model`:

```bash
python -m pinn_v2 train --case xlpe_single --model M0 --output results/runs/m0
python -m pinn_v2 train --case xlpe_single --model M2 --output results/runs/m2
python -m pinn_v2 train --case xlpe_single --model M1+rba --output results/runs/m1_rba
```

## Usar un archivo de configuración

```bash
python -m pinn_v2 train --config configs/train_m1.json --output results/runs/desde_json
```

Cualquier opción dada en la línea de comandos tiene prioridad sobre el archivo.

## Pruebas

```bash
pytest tests -q                  # pruebas rápidas
pytest tests -q -m fem           # incluye FEM (requiere WSL2)
```

## Construir esta documentación

```bash
python docs/build_docs.py
```

El resultado queda en `docs/_build/html/index.html` y se abre con doble clic.