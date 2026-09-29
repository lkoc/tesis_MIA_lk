# Estado de cierre y pendientes auténticos

Actualizado el 29-sep-2026 a partir de la auditoría general
([AUDITORIA_GENERAL_2026-09-29.md](../auditoria%20general/AUDITORIA_GENERAL_2026-09-29.md)).
La auditoría técnica anterior está en [AUDITORIA_TESIS.md](../docs/AUDITORIA_TESIS.md) y se conserva como histórica.

## Estado de la evidencia

- Campaña explícita A–D: 101 entrenamientos (36 A, 24 B, 12 C, 8 C2 nuevos, 21 D). La confirmación con semillas nuevas aceptó 7 de 21 ejecuciones.
- Dominio de aceptación hoy: `xlpe_single` y `xlpe_dry_far` aceptados (3/3); `xlpe_dry_near` restringido (1/3); estratos, relleno, Aras plano y Kim en arena no aceptados.
- La búsqueda de corriente límite con la formulación explícita **no se ha ejecutado**; OE4 se declara no alcanzado en el texto de la tesis.
- Las cifras vigentes (páginas, entrenamientos, aceptación y pruebas) están en [ESTADO.md](../ESTADO.md), que es la fuente única. Las cifras anteriores al 29-sep-2026 (290 entrenamientos, 346 pruebas, 74 páginas) corresponden a la línea reducida y ya no aplican.

## Decisiones registradas (D1–D6)

| Decisión | Resolución | Dónde consta |
|---|---|---|
| D1 dominio de aceptación (R-12) | Se fija por escrito el dominio de arriba. La ronda E para estratos, relleno y zona seca cercana queda **preinscrita y no ejecutada**; Aras y Kim no se reabren sin cambio justificado de arquitectura o presupuesto. | Tesis, Sección «Dominio de aceptación»; [docs/PROTOCOLO_RONDA_E.md](../docs/PROTOCOLO_RONDA_E.md) (falta la firma del asesor). |
| D2 estructura del Plan (R-03 a R-05) | Se sigue la Guía N.° 01: Capítulos I a V, Referencias y Anexos aparte. | `Plan/plan_tesis_cables_pinn.tex` y `Plan/capitulos/`. |
| D3 título | Se mantiene el título más reciente. **No se ha enviado a la Unidad de Posgrado.** | Carátulas y metadatos de tesis y Plan. |
| D4 espaciado de referencias (R-30) | **Rechazada:** las referencias conservan interlineado simple. | Sin cambio. |
| D5 hipótesis (R-29) | Se conservan HE1–HE4 y la sección se llama «Criterios de aceptación y proposiciones de diseño». | Tesis, Capítulo I. |
| D6 viabilidad (R-28) | Se añade costo computacional, matriz de riesgos y sostenibilidad; sin afirmar ahorro económico ni aceleración. | Tesis, Sección «Impacto y producto reproducible». |
| Interlineado (R-31) | Se adopta `\setstretch{1.4}` (19,0 pt, equivalente a 1,5 líneas de Word con Arial 11; `\onehalfspacing` daba 16,4 pt). La tesis pasa de 82 a 88 páginas. | `tesis.tex`. |

## Pendiente por la restricción «sin cálculos nuevos de PINN ni FEM»

- Ronda E: E1 (hasta 10 entrenamientos nuevos) y E2 (9 entrenamientos con semillas 131, 149 y 167). Requiere la firma del protocolo.
- Extensión de ampacidad explícita: 3 casos × 3 semillas (protocolo en [docs/PROTOCOLO_AMPACIDAD_EXPLICITA.md](../docs/PROTOCOLO_AMPACIDAD_EXPLICITA.md)).
- Contraste numérico con valores publicados (Aras 2005, Kim 2025) con la formulación explícita: exige nuevas referencias FEM con sus pérdidas AC, propiedades y contornos.
- Regenerar `Benchmarks/explicit_study/report/` y el `MANIFEST.json` con SHA-256 de la campaña.

## Pendiente de código y configuración (no se tocó)

R-16 (unificar `pytest`), R-17 (integración continua), R-19 (rutas absolutas), R-20 (peso del repositorio),
R-21 (rangos de dependencias), R-35 (`.gitignore`), R-36 (ordenar la raíz), R-37 (unificar `.bib`),
R-38 (generadores de la matriz) y R-40 (informe de similitud).

## Advertencia sobre el generador del informe

`Benchmarks/full_study_report.py` (invocado por `scripts/build_final_thesis.py`) **reescribe** `tablas/explicit_results.tex`,
`tablas/explicit_all_runs.tex`, `tablas/explicit_abstract_*.tex` y `capitulos/04_conclusiones_y_recomendaciones.tex`.
Esos archivos contienen ediciones manuales del 29-sep-2026: las seis leyendas con «Fuente» de `explicit_results.tex`,
la redacción de las conclusiones y `\thispagestyle{empty}` en el capítulo IV. Antes de volver a ejecutarlo hay que
actualizar el generador (leyendas, tablas de restricción dominante de D, C y C2, texto de las conclusiones) o restaurar
los archivos desde el control de versiones.

## Datos que deben aportar los tesistas

- Presupuesto del Plan: los montos marcados «Por confirmar» en la Tabla del presupuesto.
- Cronograma del Plan: las fechas de octubre de 2026 a junio de 2027 son una programación por validar con el asesor.
- Una dirección estable para el repositorio digital; el anexo del Plan conserva por ahora la de Google Drive.
- Firma del protocolo de la ronda E.

## Datos personales y administrativos

- Dedicatoria y agradecimientos: requieren el texto de los tesistas.
- Copias de documentos administrativos: deben incorporarse los documentos auténticos.
- Antes del depósito, verificar que nombres, grado del asesor, mención y datos
  de portada coincidan con los registros oficiales de la universidad.

Las marcas `PENDIENTE` se conservan únicamente en los espacios personales y
administrativos. La portada utiliza Poppins Bold distribuida con su licencia,
y el cuerpo Arial. Las decisiones sobre la estructura profesionalizante de
cuatro capítulos se fundamentan en las páginas específicas de la guía UNI,
con sus discrepancias internas identificadas en la auditoría.

## Límites de la evidencia

Los fallos PINN se informan, sin atribuirles resultados aceptados. La campaña
vigente resuelve la ecuación de calor 2D estacionaria por material, con
transmisión de temperatura y flujo en las interfaces y fuentes prescritas por cable.
La validación de campo, las pérdidas AC completas, la humedad y el régimen
transitorio requieren nuevos datos y desarrollos; no se presentan como
actividades ejecutadas ni como espacios que puedan completarse por redacción.
Las campañas de la formulación reducida anterior se conservan como historia y no
se suman a los resultados vigentes.

Las referencias y configuraciones seleccionadas constan en los informes
internos. Los archivos históricos con código original incompleto están
identificados individualmente. Los resultados principales conservan sus
fuentes completas y sus huellas para auditoría posterior.
