# Campaña válida para tesis: V3 PINN/FEM

Fecha de inicio: 2026-10-27
Estado: campaña final iniciada desde cero.
Directorio de resultados: `results/adaptive_all_cases_20261027_final/`

## Alcance

Esta es la única campaña que debe usarse para el reporte final y la inclusión en la tesis. Los directorios anteriores son históricos, de depuración o de pruebas de arquitectura y no deben mezclarse con esta estadística.

Se ejecutan los modelos retenidos:

- `M1`
- `CUSPR`
- `M0R`
- `M2`

Con tres semillas: `0`, `1`, `2`.

Se recorren todos los `problems/*/problem.json` portados a V3. Cada caso debe tener FEM de referencia en `results/fem_generic/<case>/fem_l1.{json,npz}` antes de iniciar el PINN. Los casos sin referencia analítica o FEM se registran como `physics_only` y no se presentan como convergencia térmica contra referencia.

## Receta de entrenamiento

- Adam en bloques de 500 pasos, hasta 1500 pasos.
- Criterio relajado para detener Adam: el resultado debe estar por debajo de dos veces los umbrales oficiales.
- L-BFGS en bloques de 50 pasos, hasta cuatro extensiones.
- Criterio oficial de calificación de la tesis:
  - error absoluto de Tmax < 0.05 K;
  - NRMSE < 0.25 %;
  - balance < 0.5 %.
- Muestreo: mitad uniforme por región y mitad residual-focused; log-radial para regiones con discos fuente; superficies fijas.
- Remuestreo interior cada 100 pasos.
- Pesos: PDE `1`, energía `100`, continuidad interna activa.
- La referencia se usa solo para evaluación/parada; nunca entra en la pérdida.

## Hallazgos previos

- M1 es el modelo de referencia para cable: con coordenadas globales, log-radiales, coordenadas locales, amplitud regional y muestreo V2 obtuvo en `cable_disk` RMSE `0.19 K` en el smoke corregido.
- DCSNN converge bien en MMS, pero falla en cable con la representación one-hot original; queda fuera de los modelos retenidos finales salvo que esta campaña muestre una mejora reproducible.
- M2 reproduce temperaturas cercanas en cable, pero históricamente falla el balance energético por su término constitutivo/flujo.
- CUSPR es competitivo en MMS y cable preliminar, aunque más lento que M1.
- M0, CUSP y FBPINN se descartaron para la selección final de cable por errores Tmax mayores que 1 K en corridas comparables.
- La parametrización exacta corregida usa distancias normalizadas y correcciones térmicas en kelvin.
- Las interfaces internas se mantienen activas en la pérdida incluso cuando la frontera exterior se impone por construcción.
- El muestreo V2 focused no focaliza superficies: conserva las nubes de frontera/interfaz uniformes y fijas.
- El anillo V3 fue corregido para usar dominio anular real, frontera inner Neumann y outer Dirichlet. Su FEM V3 reproduce la solución analítica con error nodal RMS aproximado de `0.0012 K` y balance `0.073 %`.
- Las capas concéntricas de cables se representan como `disk + annulus` disjuntos; `priority` es explícita y común a PINN/FEM.

## Reglas de interpretación

- Un caso sin FEM o referencia analítica no puede declararse convergente por NRMSE.
- Un balance global pequeño no basta: también se revisan Tmax, NRMSE, continuidad de interfaz y dispersión entre semillas.
- Los checkpoints anteriores no se reutilizan porque pertenecen a parametrizaciones, geometrías o pesos de continuidad distintos.
- Los tiempos de cada bloque, semilla, modelo y FEM se conservan en el resumen de la campaña.
- Estos umbrales se aplican retrospectivamente a los resultados V3 ya generados. La clasificación consolidada está en `CAMPAIGN_CLASSIFICATION_2026-10-27.{json,md}`.

## Archivos de evidencia

- Resumen final: `results/adaptive_all_cases_20261027_final/summary.json`
- FEM por caso: `results/fem_generic/<case>/fem_l1.json` y `fem_l1.npz`
- Definición: `problems/<case>/problem.json`
- Reporte de cada caso: `problems/<case>/report.md`
- Auditoría de conversión: `CONVERSION_AUDIT_2026-10-26.json`
- Reglas geométricas: `GEOMETRY_RULES.md`
