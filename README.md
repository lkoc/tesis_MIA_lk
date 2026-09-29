# Tesis PINN para cálculo térmico de cables enterrados

La formulación activa resuelve la ecuación de calor en conductor, todas las capas y suelo 2D, con interfaces explícitas. La metodología y el flujo vigente están en [FULL_DOMAIN.md](Benchmarks/FULL_DOMAIN.md). La verificación nueva, el efecto piel y sus límites se documentan en [la revisión de física explícita](docs/REVISION_FISICA_EXPLICITA.md). Los PDF, tablas y campañas anteriores conservan la formulación reducida histórica; no constituyen resultados finales de la nueva formulación.

## Documentos y evidencia activa

Las cifras vigentes (páginas, entrenamientos, aceptación y pruebas) están en [ESTADO.md](ESTADO.md); los pendientes y las decisiones, en [PENDIENTES.md](Tesis_LaTeX_Borrador_UNI/PENDIENTES.md). La auditoría general del 29-sep-2026 está en [AUDITORIA_GENERAL_2026-09-29.md](auditoria%20general/AUDITORIA_GENERAL_2026-09-29.md).

- [Tesis PDF](Tesis_LaTeX_Borrador_UNI/tesis.pdf) y [fuente LaTeX](Tesis_LaTeX_Borrador_UNI/tesis.tex).
- [Campaña explícita A–D](Benchmarks/explicit_study/report/): 101 entrenamientos y confirmación con semillas nuevas; sus protocolos están en [docs/](docs/) (`PROTOCOLO_EXPERIMENTAL_EXPLICITO.md`, `PROTOCOLO_CALIBRACION_INTERFACES.md`, `PROTOCOLO_AMPACIDAD_EXPLICITA.md` y el borrador [`PROTOCOLO_RONDA_E.md`](docs/PROTOCOLO_RONDA_E.md)).
- [Formato común](Benchmarks/FORMAT.md), [catálogo JSON](Benchmarks/cases/) y [configuraciones](Benchmarks/configurations/).
- [Entornos y reproducción](Benchmarks/environment/README.md) y [tablas completas](Benchmarks/summary/).

### Documentos históricos (formulación reducida anterior al 14-sep-2026)

No sustituyen los resultados de la campaña explícita:

- [Informe interno de todos los casos](Benchmarks/INFORME_INTERNO.md), [análisis numérico](Benchmarks/ANALISIS_NUMERICO.md) y [muestreo adaptativo](Benchmarks/MUESTREO_ADAPTATIVO.md).
- [Auditoría de tesis y código](docs/AUDITORIA_TESIS.md) y [plan de trabajo](docs/PLAN_AUDITORIA_TESIS.md).
- [Cuadernos Jupyter](Benchmarks/notebooks/) y [fundamento de las variantes implementadas](docs/teoria_mejoras_pinn.md).

## Organización

| Carpeta | Función |
|---|---|
| `Benchmarks` | Campaña explícita vigente (`explicit_study`, etapas A–D) y, como historia, la batería común de 19 casos de la formulación reducida. |
| `Tesis_LaTeX_Borrador_UNI` | Documento académico, referencias y figuras derivadas de la evidencia. |
| `pinn_cables` | Biblioteca original revisada; pruebas de física y utilidades. La tesis usa la ruta auditada de `Benchmarks`. |
| `Plan` | Plan de tesis en LaTeX (`plan_tesis_cables_pinn.tex` y `capitulos/`), guías, literatura, datos de fuentes y `old/`; las entregas `envío_*` son históricas y no se modifican. |
| `examples` | Ejemplos previos y salidas históricas; sus comparaciones no sustituyen la batería común. |
| `docs` | Auditoría, criterios, registro de cambios y antecedentes. |
| `benchmark_papers`, `zotero_offline` | Fuentes bibliográficas y adjuntos; las erratas de metadatos están documentadas. |

`results/` y `comparisons/` dentro de Benchmarks identifican verificaciones
a fuente fija y campañas comparativas. `coupled_results/` contiene FEM con
R(T); `coupled_final/` y `ampacity_final/` contienen las campañas PINN nominal
y de corriente límite. El selector explícito `selection.json` identifica la
configuración principal de cada caso, conservando las demás ejecuciones.
`verification_final/` conserva las repeticiones analíticas con fuentes
archivadas; `resolution_*` identifica el estudio controlado de resolución PINN.

## Revisión mínima

```powershell
python -m pytest Benchmarks/tests -q
python scripts/verify_suite.py
python Benchmarks/notebooks.py --cases mms_constant xlpe_single
```

Para recalcular FEM, PINN o toda la campaña, seguir las instrucciones de
Benchmarks. La ejecución guardada de los cuadernos carga artefactos producidos
por los scripts; sus celdas opcionales permiten volver a calcularlos.

La evidencia vigente corresponde a la formulación explícita 2D estacionaria con fuentes prescritas por cable. No demuestra validación
de campo, generalización a cualquier geometría, pérdidas AC completas, ampacidad explícita ni
una ventaja de tiempo frente a FEM. La confirmación con semillas nuevas aceptó 7 de 21 ejecuciones. Dedicatoria, agradecimientos y constancias
académicas requieren los datos auténticos de los tesistas. La licencia está en [LICENSE.md](LICENSE.md).
