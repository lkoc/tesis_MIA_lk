<!-- explicit-heat-2d-v1 -->
La metodología vigente es [METODOLOGIA_PROPUESTA_2026-09-14.md](METODOLOGIA_PROPUESTA_2026-09-14.md), desarrollada en [FULL_DOMAIN.md](../Benchmarks/FULL_DOMAIN.md). Véase [la revisión de física explícita](REVISION_FISICA_EXPLICITA.md) para el estado de verificación y los resultados sobre efecto piel.
<!-- /explicit-heat-2d-v1 -->

# Documentación de la tesis

Estado vigente y cifras: [ESTADO.md](../ESTADO.md). Decisiones y pendientes: [PENDIENTES.md](../Tesis_LaTeX_Borrador_UNI/PENDIENTES.md). Auditoría general: [AUDITORIA_GENERAL_2026-09-29.md](../auditoria%20general/AUDITORIA_GENERAL_2026-09-29.md).

## Vigentes (formulación explícita)

- [Metodología propuesta](METODOLOGIA_PROPUESTA_2026-09-14.md) y [revisión de física explícita](REVISION_FISICA_EXPLICITA.md).
- Protocolos: [experimental A–D](PROTOCOLO_EXPERIMENTAL_EXPLICITO.md), [calibración C2](PROTOCOLO_CALIBRACION_INTERFACES.md), [ampacidad explícita](PROTOCOLO_AMPACIDAD_EXPLICITA.md) y el borrador de la [ronda E](PROTOCOLO_RONDA_E.md) (no ejecutado).
- [`auditoria/`](auditoria/): fuentes verificadas, registros de pruebas, compilación, revisión visual y manifiestos.
- [`auditoria/estado_inicial/`](auditoria/estado_inicial/): copias de los archivos anteriores a la auditoría.

## Históricos (formulación reducida, hasta el 14-sep-2026)

Se conservan como antecedentes; no sustituyen la evidencia de la campaña explícita.

- [Auditoría](AUDITORIA_TESIS.md) y [plan de ejecución](PLAN_AUDITORIA_TESIS.md).
- [Formulaciones PINN](teoria_mejoras_pinn.md): variantes implementadas y fundamento de la selección.
- [`historico/comunicaciones/`](historico/comunicaciones/): informes y mensajes previos.

La evidencia numérica vigente está en `Benchmarks/explicit_study/` y en el capítulo III de la tesis. Los informes
[INFORME_INTERNO.md](../Benchmarks/INFORME_INTERNO.md), [ANALISIS_NUMERICO.md](../Benchmarks/ANALISIS_NUMERICO.md) y
[MUESTREO_ADAPTATIVO.md](../Benchmarks/MUESTREO_ADAPTATIVO.md) corresponden a la formulación reducida: sus cifras no sustituyen
la campaña vigente cuando cambian datos físicos, ley de pérdidas, referencia FEM o presupuesto de entrenamiento.
