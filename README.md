# Tesis PINN para cálculo térmico de cables enterrados

La formulación activa resuelve la ecuación de calor en conductor, todas las capas y suelo 2D, con interfaces explícitas. La metodología y el flujo vigente están en [FULL_DOMAIN.md](Benchmarks/FULL_DOMAIN.md). La verificación nueva, el efecto piel y sus límites se documentan en [la revisión de física explícita](docs/REVISION_FISICA_EXPLICITA.md). Los PDF, tablas y campañas anteriores conservan la formulación reducida histórica; no constituyen resultados finales de la nueva formulación.

## Documentos y evidencia activa

- [Tesis PDF](Tesis_LaTeX_Borrador_UNI/tesis.pdf) y [fuente LaTeX](Tesis_LaTeX_Borrador_UNI/tesis.tex).
- [Informe interno de todos los casos](Benchmarks/INFORME_INTERNO.md): datos, resultados, ventajas, limitaciones y gráficos.
- [Análisis numérico](Benchmarks/ANALISIS_NUMERICO.md): sensibilidad al tamaño de red, colocaciones, presupuesto, orden observado y propagación del error.
- [Muestreo adaptativo](Benchmarks/MUESTREO_ADAPTATIVO.md): comparación de puntos fijos, renovación aleatoria, residuo y gradiente térmico, con gráficos y selección auditable.
- [Auditoría de tesis y código](docs/AUDITORIA_TESIS.md) y [plan de trabajo](docs/PLAN_AUDITORIA_TESIS.md).
- [Formato común](Benchmarks/FORMAT.md), [catálogo JSON](Benchmarks/cases/) y [configuraciones](Benchmarks/configurations/).
- [Cuadernos Jupyter](Benchmarks/notebooks/), con citas y referencias; [instrucciones de ejecución](Benchmarks/README.md).
- [Entornos y reproducción](Benchmarks/environment/README.md) y [tablas completas](Benchmarks/summary/).
- [Fundamento de las variantes implementadas](docs/teoria_mejoras_pinn.md).

## Organización

| Carpeta | Función |
|---|---|
| `Benchmarks` | Batería común y evidencia numérica actual: 19 casos, ambos métodos y comparación de configuraciones. |
| `Tesis_LaTeX_Borrador_UNI` | Documento académico, referencias y figuras derivadas de la evidencia. |
| `pinn_cables` | Biblioteca original revisada; pruebas de física y utilidades. La tesis usa la ruta auditada de `Benchmarks`. |
| `Plan` | Plan, guías y entregas históricas; se conserva su identidad documental. |
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

La evidencia corresponde a un modelo DC reducido. No demuestra validación
de campo, generalización a cualquier geometría, pérdidas AC completas ni
una ventaja de tiempo frente a FEM. Dedicatoria, agradecimientos y constancias
académicas requieren los datos auténticos de los tesistas.
