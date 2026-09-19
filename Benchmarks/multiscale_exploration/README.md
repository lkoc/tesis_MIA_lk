# Exploración multiescala independiente de la tesis

La metodología, bibliografía, protocolo y decisiones se registran en
[INVESTIGACION_PINN_MULTIESCALA_2026-09-15.md](../../docs/INVESTIGACION_PINN_MULTIESCALA_2026-09-15.md).

No modifica el protocolo A–D ni los resultados históricos. `prior_status_audit.json`
identifica qué quedó terminado y qué falta. `construction_audit.json` comprueba
interfaces y operador, sin afirmar precisión de una solución entrenada.

## Reproducción

Desde la raíz del proyecto, Python y PyTorch en CPU/doble precisión:

```powershell
python -X utf8 scripts/check_multiscale_trace.py
python -X utf8 Benchmarks/multiscale_trace.py --seed 11 --output RUTA_NUEVA_PINN
python -X utf8 Benchmarks/multiscale_linear.py --seed 11 --output RUTA_NUEVA_SVD
python -X utf8 Benchmarks/multiscale_linear.py --seed 23 --factor 2 --output RUTA_NUEVA_REFINADA
python -X utf8 Benchmarks/multiscale_linear.py --seed 11 --width 16 --depth 2 --output RUTA_NUEVA_PEQUENA
python -X utf8 Benchmarks/multiscale_linear.py --seed 11 --case xlpe_dry_near --patch-expert --output RUTA_NUEVA_PARCHE
python -X utf8 scripts/report_multiscale_exploration.py
```

Se necesitan los JSON de configuración de C y el FEM de `xlpe_single` en
`explicit_study/references/fixed/xlpe_single/fem_l2.npz`, con su JSON físico.
Los programas se niegan a sobrescribir resultados terminados; usar rutas nuevas.
El reporte recoge las carpetas canónicas de esta exploración y escribe solamente
en `report/`; no transfiere tablas ni afirmaciones a la tesis.

## Evidencia conservada

- `pilot/`: dos entrenamientos completos Adam/L-BFGS con trazas compartidas.
- `linear/`: dos soluciones de características neuronales fijas 32×3.
- `points_half/`, `points_double/`: refinamiento conjunto de puntos y cuadraturas.
- `small/`: arquitectura 16×2 con seis armónicos de traza.
- `transfer/`: prueba sin reajuste en coaxial angular y suelo seco cercano; conserva fallos.
- `patch/`: iteración posterior que añade un experto local para la heterogeneidad suave.
- `reproduction/`: repetición de control de la semilla 11 con sus fuentes archivadas.
- `smoke/`: verificación corta de ejecución, excluida de las comparaciones.
- `report/RESULTADOS.md`, `results.csv`, `summary.json`: resultados y verificaciones.

Cada ejecución archiva código con SHA256, configuración física/SI, semilla de red
y de muestreo, colocaciones NPZ, estado PyTorch y métricas JSON. SVD conserva además
`linear_system.npz` con A, b, coeficientes, escalas de columnas y valores singulares.
El reporte verifica archivos fuente, identidad física, referencia FEM, coordenadas
de evaluación y RMSE. La coincidencia de versiones se verifica con los archivos
archivados: volver a ejecutar el código activo después de cambiarlo es una nueva
versión del ensayo, no una reproducción exacta del anterior.

La PDE del conductor y de todas las capas sigue explícita. La SVD sólo se admite
con generación y conductividad prescritas. Su conjunto de pesos ocultos permanece
fijo y debe contarse al informar el tamaño total del modelo.
