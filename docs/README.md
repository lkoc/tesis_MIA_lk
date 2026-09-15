<!-- explicit-heat-2d-v1 -->
La metodología vigente es [METODOLOGIA_PROPUESTA_2026-09-14.md](METODOLOGIA_PROPUESTA_2026-09-14.md), desarrollada en [FULL_DOMAIN.md](../Benchmarks/FULL_DOMAIN.md). Véase [la revisión de física explícita](REVISION_FISICA_EXPLICITA.md) para el estado de verificación y los resultados sobre efecto piel.
<!-- /explicit-heat-2d-v1 -->

# Documentación de la tesis

- [Auditoría](AUDITORIA_TESIS.md): guías UNI, Plan y entrega Drive, bibliografía, código, decisiones y límites.
- [Plan de ejecución](PLAN_AUDITORIA_TESIS.md): secuencia y estado de las actividades.
- [Formulaciones PINN](teoria_mejoras_pinn.md): variantes implementadas y fundamento de la selección.
- [`auditoria/`](auditoria/): fuentes verificadas, registros de pruebas, compilación, revisión visual y manifiestos.
- [`auditoria/estado_inicial/`](auditoria/estado_inicial/): copias de los archivos anteriores a la auditoría.
- [`historico/comunicaciones/`](historico/comunicaciones/): informes y mensajes previos, conservados como antecedentes.

La evidencia numérica vigente se encuentra en [Benchmarks/INFORME_INTERNO.md](../Benchmarks/INFORME_INTERNO.md),
[ANALISIS_NUMERICO.md](../Benchmarks/ANALISIS_NUMERICO.md) y
[MUESTREO_ADAPTATIVO.md](../Benchmarks/MUESTREO_ADAPTATIVO.md). Las cifras de
informes históricos no sustituyen esa batería cuando cambian datos físicos,
ley de pérdidas, referencia FEM o presupuesto de entrenamiento.
