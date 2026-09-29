---
name: auditoria-tesis-reproducible
description: 'Auditoría reproducible de un plan de tesis o tesis con componente computacional (LaTeX, bibliografía, código PINN/FEM, repositorio). Use when the user asks for auditoría, diagnóstico, revisión exhaustiva, conformidad con guías institucionales (Plan, tesis, rúbrica 8.2), coherencia entre documentos y resultados, matriz de conformidad, registro de hallazgos, recomendaciones priorizadas por olas o estado de implementación antes de una entrega, defensa o depósito.'
---

# Auditoría reproducible de un proyecto de tesis

Método para producir un diagnóstico como `auditoria general/AUDITORIA_GENERAL_2026-09-29.md`: cada cifra sale de un script o de una lectura directa de la fuente, cada hallazgo tiene evidencia y cada recomendación tiene un criterio de aceptación verificable. Otra persona debe poder repetirlo y obtener las mismas cifras.

Para que VS Code cargue esta skill automáticamente, copie o enlace esta carpeta en `.github/skills/`.

## Cuándo usarla

- Revisión general antes de entregar un Plan, una tesis o un depósito.
- Comprobar conformidad con las guías institucionales y con la rúbrica de evaluación.
- Detectar contradicciones entre Plan, tesis, PDF, código, resultados y README.
- Priorizar el trabajo pendiente y registrar qué se hizo tras aprobar las recomendaciones.

No la use para redactar capítulos, entrenar modelos ni validar la física del código línea por línea.

## Entregable

Una carpeta `auditoria general/` con:

| Ruta | Contenido |
|---|---|
| `AUDITORIA_GENERAL_<fecha>.md` | Documento único con los 13 apartados de la sección «Estructura del documento». |
| `evidencia/scripts/` | Scripts que producen cada cifra citada. |
| `evidencia/salidas/` | Salida de cada script, guardada en UTF-8. |
| `evidencia/ocr_guias/` | Texto de las guías escaneadas y hojas de contacto para cotejo visual. |

## Principios no negociables

1. **Solo lectura hasta que se apruebe.** La auditoría no modifica el Plan, la tesis, el código ni la bibliografía. No relanza campañas de cómputo ni recompila PDF en su ubicación (alteraría `.aux` y `.pdf`); compila en una copia temporal.
2. **Solo cifras verificadas.** Use únicamente números recalculados por un script o leídos de la fuente. Los resúmenes de exploraciones automáticas y de informes previos se contrastan; los errores encontrados se registran en un anexo.
3. **Evidencia con ubicación.** Cada hallazgo cita archivo y línea, salida de script o página impresa de la guía.
4. **Hallazgo distinto de recomendación.** Primero se describe lo que se observa; después se propone la acción.
5. **Las decisiones son del usuario.** Cuando las guías se contradicen o hay opciones legítimas, registre la decisión con opciones y recomendación; no la tome.
6. **Honestidad sobre los límites.** Declare qué no se revisó (código línea por línea, validez de campo, aprobación del asesor).
7. **Un solo criterio de conteo.** Defina cada cifra una vez (entrenamientos, pruebas, páginas) y úsela igual en todos los documentos.

## Habilidades necesarias

| Habilidad | Para qué | Señal de dominio |
|---|---|---|
| Lectura normativa | Convertir guías y rúbricas en requisitos verificables (N-nn) y detectar contradicciones (G-n). | Cada requisito tiene página impresa, estado por documento y prueba. |
| OCR y cotejo visual | Guías escaneadas sin texto. | PDF renderizado a PNG, OCR `es-MX`, medidas críticas confirmadas sobre el original. |
| LaTeX y biblatex | Leer la estructura, las etiquetas, las llamadas y los registros de compilación. | Compila en copia temporal con `lualatex` y `biber` y da 0 referencias indefinidas. |
| Análisis de PDF | Medir geometría, fuentes, páginas, numeración y sangría. | Scripts con PyMuPDF; separación entre líneas medida en puntos. |
| Bibliografía | Cruzar `\cite` con `.bib`, edad de fuentes, campos faltantes y duplicados. | 0 citas sin entrada; lista de entradas sin citar y de URL privadas. |
| Lectura de campañas PINN/FEM | Entender protocolo, umbrales, semillas, referencias y resultados por corrida. | Reproduce el conteo de aceptaciones y la restricción dominante de cada rechazo. |
| Ciencia del diseño (DSR) | Juzgar si el artefacto, la evaluación y la contribución son coherentes. | Objetivo, evidencia, conclusión y recomendación se corresponden. |
| Redacción académica | Medir estilo con indicadores y separar fallas de fondo y de forma. | Palabras por oración, primera persona, adjetivos valorativos, párrafos de una oración. |
| Git y repositorio | Estado inicial, higiene, tamaño, licencia, secretos, rutas absolutas. | `git status` y commit base registrados antes de empezar. |
| Excel (`openpyxl`) | Leer y rehacer matrices y rúbricas. | Detecta acentos perdidos («?») y reconstruye sin dañar el formato. |
| Priorización | Ordenar por severidad, dependencia y esfuerzo. | Olas de trabajo y orden mínimo para desbloquear al asesor. |

## Entorno y herramientas

- Python 3.12 con PyMuPDF (`fitz`) y `openpyxl`; `git`; MiKTeX o TeX Live con `lualatex`, `biber` y `latexmk`.
- OCR nativo de Windows (`ocr.ps1`, idioma `es-MX`) si las guías son escaneos.
- Scripts existentes en `auditoria general/evidencia/scripts/`. Cada uno fija la raíz con `os.chdir` al inicio; ajuste esa constante en otra máquina.
- Compilar: `powershell -NoProfile -ExecutionPolicy Bypass -File "auditoria general/evidencia/scripts/compilar_temp.ps1" -Que tesis|plan` (copia en `%TEMP%\ths_build`).
- Guardar salidas en UTF-8: use `python -X utf8` y `PYTHONIOENCODING=utf-8`; en PowerShell 5.1 la redirección `>` corrompe UTF-8, así que use `cmd /c "python ... > archivo"`.

## Procedimiento

### Fase 0. Preparar

1. Fije la fecha de corte y registre `git status --short` y el commit base. Anote los cambios previos que no son suyos.
2. Cree `auditoria general/evidencia/{scripts,salidas,ocr_guias}`.
3. Inventarie qué se revisa: guías, Plan, tesis, código, bibliografía, repositorio.

### Fase 1. Normativa

1. Si el PDF no tiene texto, renderice a PNG (`renderall.py`) y aplique OCR (`ocr.ps1`). La página del PDF es la impresa más 1; cite siempre la impresa.
2. Extraiga cada requisito a una matriz `N-nn` con fuente, estado en el Plan y estado en la tesis (C conforme, P parcial, NC no conforme, PD pendiente por trámite, NA no aplica).
3. Registre contradicciones entre guías como `G-n` con el tratamiento propuesto y márquelas como decisión del usuario.
4. Confirme sobre el original las medidas y plazos críticos: el OCR pierde acentos.

### Fase 2. Plan de tesis

- Compare el índice del PDF (`toc_pdf.py`) con la estructura institucional capítulo por capítulo.
- Lea la rúbrica de evaluación vigente (`xlsxdump.py`) y repítala sobre la versión actual: una rúbrica antigua no prueba el estado de hoy.
- Compare versiones y fechas de los PDF, títulos y entregas (`versiones_plan.py`).
- Revise carátula, datos generales, cronograma, presupuesto, financiamiento, antecedentes, metodología y anexos.

### Fase 3. Documento de tesis

- Recorra sección por sección con estado C/P/NC y hallazgo `TS-nn`.
- Ejecute `style.py`, `conteos_latex.py`, `etiquetas_sin_referencia.py`, `pdfcheck.py`, `pagecheck.py`, `refcheck.py` y `logcheck.py`.
- Construya la trazabilidad objetivo → hipótesis → evidencia → conclusión con el estado real: cumplido, parcial o no alcanzado.
- Compare fechas de las fuentes `.tex` con la del PDF.

### Fase 4. Modelamiento (PINN/FEM)

1. Inventario: archivos de código, líneas, pruebas recolectadas (`pytest --collect-only`), manifiestos.
2. Reconstruya el resultado de cada etapa a partir de los JSON o CSV (`explicit_sum.py`, `dgate2.py`), no del texto de la tesis.
3. Para cada rechazo calcule la restricción dominante por corrida (`restriccion_dominante.py`) y el costo (`costo_computacional.py`).
4. Distinga líneas de evidencia (p. ej. formulación reducida y explícita) y no sume sus cifras.
5. Compare el informe consolidado con `status.json` y con los manifiestos.
6. Evalúe la validez científica: verificación no es validación, generalización, costo, incertidumbre de entradas, número de semillas.

### Fase 5. Bibliografía

Ejecute `bibcheck.py` y `bibyears.py`. Verifique citas sin entrada, entradas sin citar, campos faltantes, `urldate`, URL a nubes privadas, duplicados y diferencias entre los `.bib` de Plan y tesis.

### Fase 6. Coherencia transversal

Compare cifras, títulos, fechas, enlaces y nombres entre `README`, `PENDIENTES`, Plan, tesis, PDF, Excel y carpeta de entregas. Un hallazgo `CT-nn` por cada contradicción.

### Fase 7. Repositorio

Licencia, `.gitignore`, raíz, tamaño, rutas absolutas (`rutas_usuario.py`), secretos por patrón (`repocheck.py`), CI, terceros redistribuidos, hashes de entregas.

### Fase 8. Recomendaciones

1. Convierta cada hallazgo en una recomendación con acción concreta, archivo, esfuerzo y criterio de aceptación.
2. Agrupe en olas y marque el orden mínimo para desbloquear al asesor.
3. Deje una columna de decisión (A aprobar, R rechazar, M modificar) para el usuario.

### Fase 9. Control de calidad del documento

Ejecute `validar_documento.py`: enlaces relativos rotos, filas con distinto número de columnas, IDs duplicados o faltantes, caracteres de reemplazo. Corrija hasta que salga limpio.

## Criterios de verificación

Los umbrales de formato provienen de las guías UNI; sustitúyalos por los de la institución si cambia.

### Formato del documento

| Criterio | Regla | Cómo se mide |
|---|---|---|
| Papel y márgenes | A4; 3,0 cm izquierdo y superior, 2,5 cm derecho e inferior (±0,1 cm) | `pdfcheck.py` |
| Tipografía | Arial 11 en el cuerpo; cursiva solo para extranjerismos | Fuentes del PDF |
| Carátula | Poppins 22 negrita y Arial 16, 14 y 12; título de hasta 20 palabras sin artículos ni preposiciones | Inspección visual y fuentes |
| Interlineado | 1,5 líneas de Word con Arial 11 ≈ 19 pt; simple en notas, listas, bibliografía, índices y anexos | Separación entre líneas en el PDF. En LaTeX `\onehalfspacing` da 16,4 pt y `\setstretch{1.4}` da 19,0 pt |
| Paginación | Romana hasta el resumen, arábiga desde la introducción, sin número visible en la introducción ni en el primer folio de cada capítulo, número arriba a la derecha | `pagecheck.py` |
| Extensión | Máximo 130 páginas con anexos | Páginas del PDF |
| Estructura | Índice idéntico al de la guía; Referencias y Anexos fuera de la numeración de capítulos | `toc_pdf.py` |

### LaTeX y referencias cruzadas

| Criterio | Regla |
|---|---|
| Compilación | 0 citas indefinidas, 0 referencias indefinidas, 0 anclas duplicadas; `Underfull hbox` no cuenta como hallazgo por sí solo |
| Llamadas | 0 etiquetas de tabla, figura, fórmula o anexo sin `\ref` en el texto |
| Leyendas | Toda tabla, figura y fórmula con «Fuente» |
| Siglas | Definidas en su primera mención, incluido el resumen |
| Etiquetas fijas | Sin números escritos a mano («ANEXO 5.1»): usar `\ref` |
| Marcadores | `\pendiente` solo en las páginas personales y administrativas hasta el depósito |

### Bibliografía

| Criterio | Regla |
|---|---|
| Integridad | 0 claves citadas sin entrada; lista de entradas sin citar |
| Actualidad | Mayoría de los últimos 5 años; las excepciones (clásicos, normas, fundamentos metodológicos) se declaran y se identifican en una tabla cronológica |
| Metadatos | DOI o URL en artículos; `urldate` en entradas en línea; sin URL a nubes privadas |
| Consistencia | Un solo `.bib` fuente o diferencias documentadas entre Plan y tesis |
| Orden de antecedentes | Del más antiguo al más reciente, con tabla de evolución cronológica |

### Redacción (indicadores de referencia)

No son normas; sirven para separar fallas de fondo y de forma.

| Indicador | Referencia observada |
|---|---|
| Palabras por oración | Alrededor de 17 a 20 |
| Oraciones de más de 45 palabras | Menos del 2 % |
| Primera persona | 0 |
| Adjetivos valorativos | Pocos y justificados |
| Párrafos de una sola oración | Menos del 15 % |
| Resumen y abstract | Pasado; introducción, objetivo, diseño, resultados numéricos y conclusión |
| Conclusiones | Una por objetivo con su estado, y una recomendación por conclusión |

### Modelamiento y evidencia computacional

| Criterio | Regla |
|---|---|
| Preinscripción | Umbrales, casos, presupuestos y reglas de selección fijados antes de observar la campaña que los usa |
| Confirmación independiente | Semillas nuevas y casos no usados para ajustar; se conservan y se informan todos los resultados, incluidos los rechazos |
| Referencia independiente | Al menos tres niveles de malla con criterio de estabilidad y balance; identidad física comprobada por hash |
| Conteo único | Regla de conteo de entrenamientos definida una vez y aplicada en todos los documentos |
| Diagnóstico de rechazos | Razón entre el valor observado y la tolerancia por criterio y por corrida, con el criterio dominante |
| Dominio de aceptación | Lista explícita de problemas aceptados, restringidos y no aceptados; sin extrapolar a otros casos |
| Matriz de validez | Qué se verifica, contra qué referencia, qué evidencia hay y qué no se establece |
| Semillas | Con dos o tres semillas no se estiman probabilidades de éxito; se informan mediana, rango y cantidad aceptada |
| Costo | Costo por caso, hardware y concurrencia; sin afirmar aceleración sin medición comparable |
| Trazabilidad | Manifiestos con huellas que cubran toda la campaña; informe consolidado igual a `status.json` |
| Pruebas | Total recolectado y ejecutado registrado; la ruta por defecto de `pytest` cubre todos los conjuntos |
| Reproducibilidad | 0 rutas absolutas de usuario en scripts; versiones de dependencias acotadas |

### Coherencia y repositorio

| Criterio | Regla |
|---|---|
| Cifras | Una fuente única de cifras (`ESTADO.md`); los demás documentos remiten a ella |
| Versiones | El PDF es posterior a sus fuentes; el título coincide en carátula, metadatos, rúbrica y trámites |
| Enlaces | 0 enlaces relativos rotos en `docs/` y README |
| Documentos históricos | Rotulados como históricos cuando describen una formulación anterior |
| Licencia | Definida por separado para código y texto, con inventario de materiales de terceros |
| Higiene | `.gitignore` sin BOM; raíz ordenada; sin secretos por patrón; tamaño compatible con el alojamiento |

## Severidad y hallazgos

| Nivel | Criterio |
|---|---|
| Crítico | Impide aprobar o defender, o invalida una afirmación principal |
| Mayor | Incumple una guía o crea riesgo de observación formal o de irreproducibilidad |
| Menor | Desviación corregible sin afectar el fondo |
| Sugerencia | Mejora opcional |

Prefijos de ID: `PL` Plan, `TS` tesis, `MD` modelamiento, `BI` bibliografía, `CT` coherencia transversal, `RP` repositorio, `N-nn` requisito normativo, `G-n` inconsistencia entre guías, `D-n` decisión del usuario, `R-nn` recomendación.

Formato de hallazgo:

```markdown
| ID | Sev. | Hallazgo | Evidencia | Recomendación |
| TS-02 | Crítico | Resumen en presente y sin resultados | `00_resumen_abstract.tex` L1–12; `pdfcheck.txt` | R-25 |
```

Formato de recomendación:

```markdown
| ID | Sev. | Recomendación (acción, archivo) | Hallazgos | Esfuerzo | Criterio de aceptación | Decisión |
| R-25 | Crítico | Reescribir Resumen y Abstract en pasado con cifras de D | TS-02 | B | Estructura de la Guía 02 con cifras | A |
```

Esfuerzo: B bajo, M medio, A alto. El criterio de aceptación debe poder comprobarse con un script o una lectura concreta (por ejemplo, «0 etiquetas sin llamada»).

## Olas de implementación

| Ola | Contenido | Depende de |
|---|---|---|
| 0 | Decisiones con el asesor y la Unidad | — |
| 1 | Plan de tesis (sin cómputo nuevo) | Ola 0 |
| 2 | Modelamiento: cerrar y explicar la campaña | Decisión sobre el dominio de aceptación |
| 3 | Tesis | Ola 2 |
| 4 | Repositorio y bibliografía | Ninguna; puede ir en paralelo |

Indique siempre el orden mínimo para desbloquear al asesor.

## Estructura del documento

1. Resumen ejecutivo: dictamen por componente, diez hallazgos principales, lo que conviene conservar, decisiones del usuario, calendario.
2. Alcance, método y límites: qué se revisó, método, estado del repositorio al iniciar, escala de severidad, límites.
3. Marco normativo: fuentes, matriz de conformidad, inconsistencias entre guías.
4. Plan de tesis: hallazgos por sección, rúbrica, versiones, bibliografía.
5. Tesis: hallazgos por sección, indicadores de redacción, trazabilidad de objetivos, registro de hallazgos.
6. Modelamiento: inventario, líneas de evidencia, resultados por etapa, hallazgos abiertos, validez científica.
7. Bibliografía.
8. Coherencia transversal.
9. Repositorio, reproducibilidad y seguridad.
10. Registro de recomendaciones por ola, con columna de decisión.
11. Ruta de implementación.
12. Anexos: tabla de evidencia reproducible, correcciones a los informes de apoyo, códigos de hallazgos, estado de la carpeta.
13. Estado de implementación (se añade tras la aprobación).

## Implementar tras la aprobación

1. Registre las decisiones y las restricciones del usuario (por ejemplo, «sin cálculos nuevos»). Lo que las incumple queda como pendiente de cálculo, de código, de archivos o de datos.
2. Aplique los cambios de texto por olas. Para reestructurar archivos grandes use un script con verificaciones de conteo y entornos balanceados, y guárdelo en `evidencia/scripts/`.
3. Tras cada bloque, compile en copia temporal y compruebe 0 referencias indefinidas, páginas, índice y etiquetas sin llamada.
4. Antes de regenerar informes automáticos, compruebe qué archivos reescribe el generador: pueden contener ediciones manuales.
5. Registre en el §13 el estado por recomendación (Hecho, Parcial, Pendiente con su causa, Rechazada), la verificación y los archivos tocados.
6. Actualice la fuente única de cifras y `PENDIENTES.md`.

## Trampas y falsos positivos

- Las exploraciones automáticas y los informes previos contienen errores de conteo (por ejemplo, ~130 entradas `.bib` frente a 88 reales). No los cite sin recalcular.
- Buscar «PENDIENTE» en el PDF coincide con «INDEPENDIENTES»: confirme el marcador real en el `.tex`.
- La numeración del PDF difiere de la impresa; cite la impresa.
- Un Excel con «?» en lugar de acentos perdió la codificación; reconstruya los textos, no los copie.
- Las rúbricas y actas antiguas pueden citar un título o una estructura anteriores.
- Un `Underfull hbox` no explica una conclusión incompleta: son hallazgos independientes.
- Dos líneas de evidencia distintas no se suman. Etiquete la histórica y no la mezcle con la vigente.
- Los generadores de tablas o capítulos pueden sobrescribir texto escrito a mano. Avise antes de ejecutarlos.
- Un resultado desfavorable se informa; no se sustituye por su mejor semilla ni se incorpora en silencio al ajuste.
- El acuerdo PINN–FEM verifica la implementación; no valida el modelo físico frente a instalaciones reales.

## Lista de verificación final

- [ ] Toda cifra citada tiene un script o una lectura directa en `evidencia/`.
- [ ] Cada hallazgo tiene ID, severidad, evidencia y recomendación.
- [ ] Cada recomendación tiene esfuerzo, criterio de aceptación verificable y columna de decisión.
- [ ] Las decisiones del usuario están separadas de las recomendaciones.
- [ ] Los límites de la auditoría están declarados.
- [ ] `validar_documento.py` no reporta enlaces rotos, columnas irregulares ni IDs duplicados.
- [ ] No se modificó ningún archivo del proyecto durante la auditoría (salvo `auditoria general/`).
- [ ] Si se implementó: compilación en copia temporal limpia, §13 actualizado y cifras alineadas con la fuente única.

## Ejemplo de referencia

- Documento: `auditoria general/AUDITORIA_GENERAL_2026-09-29.md`.
- Scripts y salidas: `auditoria general/evidencia/scripts/` y `auditoria general/evidencia/salidas/`.
- Estado de implementación: §13 del mismo documento; cifras vigentes en `ESTADO.md`.
