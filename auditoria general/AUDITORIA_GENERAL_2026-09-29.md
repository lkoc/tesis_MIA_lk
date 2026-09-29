# Auditoría general del proyecto de tesis

**Proyecto:** Diseño y evaluación de una red neuronal informada por física para estimar la temperatura y la ampacidad de cables eléctricos enterrados en entornos heterogéneos.
**Programa:** Maestría en Inteligencia Artificial, FIIS-UNI (grupo 91). **Tesistas:** Luis Enrique Koc Góngora y Herbert Antonio Meléndez García. **Asesor:** Samuel Oporto.
**Fecha de corte:** 29 de septiembre de 2026. **Commit base:** `f6466b2` ("25.09.26") más cambios locales sin confirmar (véase §2.3).
**Naturaleza:** auditoría de solo lectura al momento de su emisión. Los cambios aprobados se aplicaron después y se registran en el §13.
**Estado del documento:** recomendaciones aprobadas por los tesistas (R-30 rechazada; R-02 modificada; R-12 aprobada). Implementación de los cambios de texto registrada en el §13 (29-sep-2026).

---

## 1. Resumen ejecutivo

### 1.1 Dictamen por componente

| Componente                          | Conformidad                                                             | Motivo principal                                                                                                                                                                                                                                  |
| ----------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Plan de tesis (Guía N.° 01)       | **No conforme** con la estructura institucional                   | Falta el Capítulo V (cronograma, presupuesto, financiamiento); Metodología precede al Marco teórico; Datos generales sin asesor. La base técnica, la bibliografía y la redacción son sólidas. El PDF vigente no refleja el`.tex` actual. |
| Documento de tesis (Guía N.° 02)  | **Parcial**: estructura y redacción conformes; cierre incompleto | Resumen sin resultados y en presente; conclusiones solo cubren OE1; resultados de las etapas C, C2 y D sin integrar; paginación y citas de tablas/anexos no conformes.                                                                           |
| Modelamiento PINN/FEM               | **Metodología rigurosa y trazable; aceptación parcial**         | La campaña explícita terminó con 7 de 21 ejecuciones de producción aceptadas (etapa D). Aras, Kim, relleno y estratos fueron rechazados. La ampacidad explícita solo tiene ensayos de humo.                                                  |
| Coherencia documental y repositorio | **Desalineado**                                                   | Conviven dos líneas de evidencia (reducida y explícita); documentos, PDF y manifiestos describen estados distintos. Licencia sin definir; rutas absolutas; repositorio de 2,3 GiB.                                                              |

### 1.2 Los diez hallazgos que más condicionan el avance

| #  | ID    | Sev.     | Hallazgo                                                                                                                                                                                                                                                                                                             |
| -- | ----- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | CT-01 | Crítico | Las tablas de la tesis quedaron en el estado del 15-sep 02:04 ("etapas en ejecución") aunque la campaña terminó el 15-sep 05:38 (`status.json`: `completed: true`). Faltan C, C2 y D en tablas, resultados, resumen y conclusiones.                                                                           |
| 2  | MD-01 | Crítico | Etapa D: 7/21 aceptadas.`xlpe_single` 3/3 y `xlpe_dry_far` 3/3; `xlpe_dry_near` 1/3; `aras_flat`, `kim_sand`, `xlpe_backfill` y `xlpe_discrete_layers` 0/3. OE2, OE3 y OE4 no están cumplidos en la formulación explícita fuera de casos simples.                                                   |
| 3  | PL-01 | Crítico | El Plan no tiene Capítulo V (0 apariciones de cronograma, presupuesto, financiamiento). Guía 01 pp. 34–35 y rúbrica 8.2, ítems 13–15.                                                                                                                                                                          |
| 4  | PL-02 | Crítico | El Plan ordena Metodología (Cap. 3) antes de Marco teórico (Cap. 4); la Guía 01 exige lo inverso. La autoevaluación del 03-jul ya lo marcó "Crítica"; sigue sin corregir.                                                                                                                                      |
| 5  | TS-01 | Crítico | Las conclusiones solo tratan OE1; no hay conclusiones para OE2 y OE3; OE4 figura "en ejecución". La Guía 02 (p. 19) pide una recomendación por conclusión.                                                                                                                                                       |
| 6  | TS-02 | Crítico | Resumen y Abstract en presente, sin resultados y con la frase "todavía está en ejecución". La Guía 02 (pp. 5 y 14) exige pasado, resultados y conclusiones.                                                                                                                                                      |
| 7  | CT-03 | Mayor    | Cifras contradictorias entre documentos: 74 páginas (`PENDIENTES.md`) frente a 61 (PDF); 19 casos/290 entrenamientos (línea reducida) frente a 101 entrenamientos A–D (línea explícita); 346 y 362 pruebas informadas frente a 307 + 59 = 366 recolectadas hoy; enlace roto a `REVISION_INTEGRAL_TESIS.md`. |
| 8  | CT-04 | Mayor    | PDF obsoletos:`plan_tesis_cables_pinn.pdf` (18-ago) frente a `.tex` (14-sep); `tesis.pdf` (15-sep 01:25) frente a fuentes modificadas hasta el 18-sep.                                                                                                                                                         |
| 9  | CT-05 | Mayor    | El título del PDF del Plan de junio (`Plan_Tesis_Grupo91_…_rev0.pdf`, 46 pp.) difiere del de los paquetes `envío_rev0` (23-jul) y `envío_rev1`. La Guía 02 (Anexo XII) prevé un trámite de "Modificación del nombre del Plan de Tesis".                                                                |
| 10 | TS-11 | Mayor    | Guía 02 (pp. 12 y 22) exige demostrar viabilidad y sostenibilidad con técnicas cuantitativas (VAN, TIR, beneficio-costo, riesgo). La tesis excluye CAPEX/OPEX y limita el impacto a lo metodológico.                                                                                                              |

### 1.3 Lo que está bien y conviene conservar

- **Redacción académica sobresaliente.** Tesis: 16,9 palabras por oración; 0,7 % de oraciones sobre 45 palabras; ninguna en primera persona; solo 5 adjetivos valorativos en 10 278 palabras (`evidencia/salidas/style.txt`).
- **Formato base conforme.** A4, márgenes 3/3/2,5/2,5 cm dentro de ±0,1 cm, Arial 11, Poppins en la portada de la tesis, numeración arriba a la derecha, APA con `biblatex`. Ambos PDF muestran 0 citas y 0 referencias indefinidas en los `.log`; el Plan tiene 69 páginas y la tesis 61 (límite 130).
- **Protocolo DSR honesto.** Umbrales fijados antes de la confirmación, semillas nuevas en D (71, 83, 97), 101 entrenamientos A–D contabilizados, rechazos conservados y cifras generadas desde registros.
- **Referencia FEM refinada.** Tres niveles de malla con criterios de estabilidad; identidad física por SHA-256.
- **Matriz de consistencia alineada con el Plan.** El Excel registra su fuente (`Plan\plan_tesis_cables_pinn.tex`) y coincide con OE1–OE4/HE1–HE4.

### 1.4 Decisiones que solo usted puede tomar

| #  | Decisión                                                                                                    | Opciones                                                                                                            | Recomendación                                                                                                              |
| -- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| D1 | ¿Cómo tratar los casos rechazados en D?                                                                    | (a) nueva ronda de desarrollo E con confirmación nueva; (b) restringir el dominio de validez y reportar; (c) ambas | (c): documentar el dominio aceptado hoy y abrir E solo para`xlpe_discrete_layers`, `xlpe_backfill` y `xlpe_dry_near`. |
| D2 | ¿Se corrige el Plan (Cap. V, orden) o se conserva el entregado y se documenta la desviación con el asesor? | Corregir; conservar y justificar                                                                                    | Corregir: la rúbrica 8.2 lo trata como no aprobable.                                                                       |
| D3 | Título oficial y trámite de modificación                                                                  | Mantener el actual; volver al del PDF de junio                                                                      | Mantner el título más recietne, el documento aún no a sido enviado a la  Unidad de Posgrado .                          |
| D4 | Espaciado de referencias (Guía 01 permite simple; Guía 02 pide doble con sangría francesa)                | Simple; doble                                                                                                       | Doble para la tesis (Guía 02 es la norma de la tesis).                                                                     |
| D5 | Sección de hipótesis en una tesis profesionalizante                                                        | Conservar como "criterios de aceptación y proposiciones de diseño"; eliminar                                      | Conservar renombrada y justificar en Cap. I.                                                                                |
| D6 | Viabilidad y sostenibilidad (Guía 02, pp. 12 y 22)                                                          | Añadir análisis mínimo de costo computacional y riesgo; solicitar exención escrita del asesor                   | Añadir el análisis mínimo y pedir visto bueno del asesor.                                                                |
| D7 | Licencia del repositorio y publicación de PDF de terceros                                                   | MIT/Apache para código y CC BY para texto; repositorio privado                                                     | Definirla antes de cualquier publicación.                                                                                  |

### 1.5 Calendario

El Plan declara "doce meses, desde julio de 2026 hasta junio de 2027" ([Plan](../Plan/plan_tesis_cables_pinn.tex#L247-L286)). A la fecha de corte han transcurrido unos 3 de esos 12 meses (25 %). Ya se ejecutó la campaña A–D; quedan pendientes ampacidad explícita, escenarios comparativos, conclusiones, capítulo de administración del Plan y trámites de depósito.

---

## 2. Alcance, método y límites

### 2.1 Qué se revisó

| Ámbito                  | Material                                                                                                                                                                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Normativa                | `GUÍA N°01 Plan.pdf` (60 pp.), `GUIA No.2 Desarrollo.pdf` (48 pp.), `evaluacion_formato_8_2_plan_tesis.xlsx`, guías locales de estructura y redacción, materiales DSR (Peffers, Hevner/Gregor, Delport) |
| Plan                     | [plan_tesis_cables_pinn.tex](../Plan/plan_tesis_cables_pinn.tex), `.toc`, `.log`, PDF vigente (69 pp.), entregas de junio (46 pp.), `envío_rev0` (68 pp.) y `envío_rev1` (69 pp.), bibliografía         |
| Tesis                    | [tesis.tex](../Tesis_LaTeX_Borrador_UNI/tesis.tex), 10 archivos de `capitulos/`, 9 de `tablas/`, PDF (61 pp.), `.log`, `PENDIENTES.md`, `README.md`                                                      |
| Modelamiento             | `pinn_cables/`, `Benchmarks/` (incluye `explicit_study/`), `tests/`, `pyproject.toml`, `docs/*.md`, `scripts/`                                                                                      |
| Bibliografía y gestión | `.bib` de Plan y tesis, `references_selection.bib`, scripts Zotero, matriz de consistencia                                                                                                                    |
| Repositorio              | Raíz,`.gitignore`, `LICENSE.md`, entregas Drive, tamaño del repositorio git                                                                                                                                 |

### 2.2 Método y trazabilidad

1. **Guías oficiales.** Los PDF son escaneos (0 caracteres de texto). Se renderizaron a PNG y se les aplicó OCR nativo de Windows (`es-MX`). El texto está en [ocr_guia01_plan.txt](evidencia/ocr_guias/ocr_guia01_plan.txt) y [ocr_guia02_tesis.txt](evidencia/ocr_guias/ocr_guia02_tesis.txt). Las páginas se citan con **numeración impresa** (la del PDF es +1) y los puntos críticos se cotejaron visualmente sobre una hoja de contacto de la Guía 01.
2. **Verificación automática.** Scripts reproducibles en [evidencia/scripts](evidencia/scripts/) con su salida en [evidencia/salidas](evidencia/salidas/): cruce `\cite`/`.bib`, edad de referencias, geometría y fuentes del PDF, numeración de páginas, advertencias `.log`, métricas de redacción, etiquetas y referencias cruzadas, resumen de la campaña explícita.
3. **Contraste de informes de apoyo.** Cinco exploraciones automáticas previas contuvieron errores (p. ej. "~130 entradas .bib" y "cigre2025 sin ISBN"; véase el Anexo B). **Solo se usan aquí cifras recalculadas o leídas directamente de la fuente.**
4. **Falsos positivos descartados.** La búsqueda de "PENDIENTE" en PDF coincide con "INDEPENDIENTES"; los marcadores reales están únicamente en las 3 páginas personales de la tesis (pp. PDF 3, 4 y 5).

### 2.3 Estado del repositorio al iniciar

`git status` reportó cambios locales previos: `.github/copilot-instructions.md`, `.gitignore` y la matriz `xlsx` modificados; sin seguimiento: `.clineignore`, `.clinerules/`, `.github/agents/`, `Plan/envío_rev0/`, `Plan/envío_rev1/01_Referencias_Zotero.zip`, `docs/ENGIE_AI_BRIDGE.md`, un `.log` de "problema de investigación" y scripts de `engie_bridge`. Esta auditoría añade únicamente la carpeta `auditoria general/`.

### 2.4 Escala de severidad

| Nivel                | Criterio                                                                         |
| -------------------- | -------------------------------------------------------------------------------- |
| **Crítico**   | Impide aprobar o defender, o invalida una afirmación principal.                 |
| **Mayor**      | Incumple una guía o crea riesgo de observación formal o de irreproducibilidad. |
| **Menor**      | Desviación corregible sin afectar el fondo.                                     |
| **Sugerencia** | Mejora opcional.                                                                 |

### 2.5 Límites

- No se relanzó ninguna campaña PINN ni FEM ni se recompilaron los PDF en su ubicación (se habría alterado `.aux` y `.pdf`). Las cifras de páginas y advertencias proceden de los PDF y `.log` existentes.
- No se revisó línea por línea la corrección física del código; se auditaron contratos, resultados, pruebas, trazabilidad y coherencia con los documentos.
- El OCR puede contener errores de acentos; las medidas y plazos críticos deben confirmarse en el PDF original antes de citarse ante la Escuela.
- Esta auditoría no sustituye la evaluación del asesor ni la aprobación de la Universidad.

---

## 3. Marco normativo verificado

### 3.1 Fuentes normativas y su lectura

| Fuente           | Contenido usado                                                                                                                                                       | Páginas impresas                            |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Guía 01 (Plan)  | Estructura de especialización, descripción de partes, carátula, cronograma/presupuesto/financiamiento, estándares de presentación, rúbrica 8.2                  | 13–14, 34–38, 43–44, 54–56               |
| Guía 02 (Tesis) | Estructura de tesis de especialización, resumen, conclusiones y recomendaciones, referencias, lista de cotejo, requisitos de depósito, declaración de autenticidad | 5–6, 12–14, 17–19, 21–23, 29, 32–35, 46 |
| Guías locales   | `Plan/guias/guia_estructura_y_contenido_plan_tesis.md`, `Plan/guias/guia_redaccion_plan_tesis.md`                                                                             | —                                           |
| DSR              | Peffers et al. (2007), Gregor y Hevner (2013), Delport et al. (2024)                                                                                                  | —                                           |

### 3.2 Matriz de conformidad

Estados: **C** conforme, **P** parcial, **NC** no conforme, **PD** pendiente por naturaleza del trámite, **NA** no aplica.

| ID   | Requisito                                                                                                                                                                                                                                        | Fuente                                     | Plan                                                                                | Tesis                                                        |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| N-01 | Carátula: institución, logo, título en mayúsculas (≤ 20 palabras sin contar artículos y preposiciones), propósito, autores, asesor, lugar y año                                                                                          | G01 pp. 35–37; G02 p. 23                  | P                                                                                   | C                                                            |
| N-02 | Tipografía de carátula: Poppins 22 negrita; Arial 16, 14 y 12                                                                                                                                                                                  | G01 p. 37; G02 p. 23                       | NC (Universidad y Escuela en Arial 18, "PLAN DE TESIS" 16, título 15; sin Poppins) | C                                                            |
| N-03 | Datos generales: título, autor(es),**asesor(es)**, área involucrada, lugar, duración en meses                                                                                                                                           | G01 pp. 13–14 (§5.1)                     | P (sin asesor)                                                                      | P (sin asesor, área ni duración)                           |
| N-04 | Estructura del Plan: I Problema (1.1 Diagnóstico; 1.2 Identificación con antecedentes, formulación, justificación y alcances), II Objetivos,**III Marco teórico, IV Metodología (4.1–4.8), V Administración**, Referencias, Anexos | G01 pp. 13–14                             | NC                                                                                  | NA                                                           |
| N-05 | Estructura de la tesis profesionalizante: Intro; Cap. I (1.1–1.5); II (2.1–2.3); III (3.1–3.5); IV Conclusiones y recomendaciones                                                                                                             | G02 pp. 12–13                             | NA                                                                                  | C (con 3.5 renombrado y hipótesis añadidas)                |
| N-06 | Cronograma (Gantt), presupuesto (rubros a–d) y financiamiento (tipo)                                                                                                                                                                            | G01 pp. 34–35; rúbrica 13–15            | NC                                                                                  | NA                                                           |
| N-07 | Especialización: sin contrastación de hipótesis                                                                                                                                                                                               | G01 p. 38; G02 pp. 12 y 21                 | P (HE1–HE4 como proposiciones de diseño)                                          | P (ídem)                                                    |
| N-08 | Metodología: tipo, nivel, métodos, población y muestra, diseño, técnicas e instrumentos de recolección y de análisis, etapas                                                                                                              | G01 §4.1–4.8; G02 lista de cotejo        | P                                                                                   | P                                                            |
| N-09 | Antecedentes en orden cronológico y de los últimos cinco años                                                                                                                                                                                 | G01 rúbrica 4 y p. 18; G02 p. 17 y cotejo | P (63 % ≥ 2021; temático)                                                         | P (50 % ≥ 2021; temático)                                  |
| N-10 | Ubicación del proyecto (justificación de localización y contexto)                                                                                                                                                                             | G01 rúbrica 11                            | P                                                                                   | NA                                                           |
| N-11 | Resumen y abstract: en pasado; introducción, objetivos, diseño, resultados y conclusiones; dos idiomas                                                                                                                                         | G02 pp. 5 y 14                             | NA                                                                                  | NC                                                           |
| N-12 | Introducción discursiva                                                                                                                                                                                                                         | G02 p. 14; rúbrica 3                      | C                                                                                   | C                                                            |
| N-13 | Conclusiones (mínimo 3) y una recomendación por cada conclusión                                                                                                                                                                               | G02 p. 19                                  | NA                                                                                  | NC                                                           |
| N-14 | Viabilidad y sostenibilidad demostradas con técnicas cuantitativas y estadística descriptiva (VAN, TIR, beneficio-costo, riesgo)                                                                                                               | G01 p. 38; G02 pp. 12 y 22                 | NA                                                                                  | P                                                            |
| N-15 | Referencias en APA vigente                                                                                                                                                                                                                       | G01 rúbrica 16; G02 p. 5                  | C                                                                                   | C                                                            |
| N-16 | Referencias: espacio simple permitido (G01 p. 43)**frente a** doble espacio con sangría francesa (G02 p. 19)                                                                                                                              | G01/G02                                    | C                                                                                   | P (simple con sangría francesa; conflicto entre guías)     |
| N-17 | Anexos incluidos y mencionados en el texto                                                                                                                                                                                                       | G01 rúbrica 18                            | C                                                                                   | P (5 anexos sin llamada en el texto)                         |
| N-18 | A4; márgenes 3,0/3,0/2,5/2,5 cm                                                                                                                                                                                                                 | G01 p. 43                                  | C                                                                                   | C                                                            |
| N-19 | Interlineado 1,5 (simple en notas, listas, bibliografía, índices y anexos)                                                                                                                                                                     | G01 p. 43                                  | C (nominal)                                                                         | C (nominal; ver TS-09)                                       |
| N-20 | Arial 11; cursiva solo para extranjerismos                                                                                                                                                                                                       | G01 p. 43                                  | C                                                                                   | C                                                            |
| N-21 | Paginación: romana desde portada hasta el resumen; arábiga desde la introducción;**introducción e inicio de capítulo sin número visible**; número arriba a la derecha; sin la palabra "página"                                     | G01 pp. 43–44                             | C                                                                                   | NC (introducción en romano; inicios de capítulo numerados) |
| N-22 | Máximo 130 páginas con anexos                                                                                                                                                                                                                  | G01 p. 44                                  | C (69)                                                                              | C (61)                                                       |
| N-23 | Listas de tablas, ilustraciones y cuadros                                                                                                                                                                                                        | G02 pp. 12 y 14                            | C                                                                                   | C                                                            |
| N-24 | Dedicatoria, agradecimientos, copia de documentos y**declaración de autenticidad y no plagio**                                                                                                                                            | G02 pp. 12, 14 y 46                        | NA                                                                                  | PD                                                           |
| N-25 | Depósito: 5 juegos empastados en color Guinda; título en el lomo                                                                                                                                                                               | G02 pp. 6 y 29                             | NA                                                                                  | PD                                                           |

### 3.3 Inconsistencias internas de las guías

| ID  | Inconsistencia                                                                                                                                                                                        | Tratamiento propuesto                                                                                      |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| G-1 | Guía 02 p. 8: el diagrama muestra 5 capítulos (I Problema, II Objetivos, III Marco, IV Propuesta, V Conclusiones); pp. 12–13 muestran 4 capítulos con objetivos y metodología dentro del Cap. I. | La tesis sigue pp. 12–13 (esquema detallado y específico). Confirmarlo con la Unidad y dejar constancia. |
| G-2 | Antecedentes: 2.1 del Marco teórico (p. 13) frente a 1.3 del Cap. I (p. 21).                                                                                                                         | Mantener en Marco teórico y resumir el estado en 1.1 Diagnóstico.                                        |
| G-3 | Referencias: espacio simple (G01) frente a doble con sangría francesa (G02).                                                                                                                         | Decisión D4.                                                                                              |
| G-4 | Especialización sin hipótesis frente a lista de cotejo (G02, Anexo V) que incluye hipótesis y variables.                                                                                           | Decisión D5.                                                                                              |
| G-5 | "Estadística descriptiva/VAN/TIR" para tesis computacionales.                                                                                                                                        | Decisión D6.                                                                                              |
| G-6 | "Tres conclusiones = tres recomendaciones" frente a cuatro objetivos específicos.                                                                                                                    | Cuatro conclusiones (OE1–OE4) con cuatro recomendaciones satisface "mínimo tres".                        |

---

## 4. Plan de tesis: auditoría por sección

Archivo principal: [plan_tesis_cables_pinn.tex](../Plan/plan_tesis_cables_pinn.tex) (última modificación 14-sep 23:54). PDF vigente: [plan_tesis_cables_pinn.pdf](../Plan/old/compilacion_2026-08-18/plan_tesis_cables_pinn.pdf), 69 páginas, compilado el 18-ago 15:14 (archivado en `Plan/old/`).

### 4.1 Hallazgos por sección

| Sección (línea)                   | Estado       | Hallazgos                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Carátula (L199–245)               | P            | **PL-06.** Universidad y Escuela en Arial 18 (sin Poppins), "PLAN DE TESIS" en 16 y título en 15; la muestra pide Poppins 22 y Arial 16, 14 y 12. Título de 24 palabras (14 sin artículos, preposiciones ni conjunciones): dentro del límite de 20 solo si se aplica esa excepción.                                                                  |
| Datos generales (L247–286)         | P            | **PL-05.** Sin fila de asesor; sí incluye lugar, duración y artefacto. Duración "julio 2026–junio 2027" sin cronograma que la respalde.                                                                                                                                                                                                               |
| Resumen (L288–311)                 | NA           | Comentado. La Guía 01 no lo exige para el Plan.                                                                                                                                                                                                                                                                                                                |
| Índice y listas (L302–311)        | C            | TOC a 4 niveles; listas de figuras, tablas, fórmulas y procedimientos.                                                                                                                                                                                                                                                                                         |
| Cap. 1 Planteamiento (L313–748)    | P            | **PL-04.** Secciones 1.1 Introducción, 1.2 Problemática, 1.3 Planteamiento, 1.4 Alcances y limitaciones. La Guía pide 1.1 Diagnóstico y 1.2 Identificación con antecedentes, formulación, justificación y alcances. La sección "Identificación y descripción abstracta" está comentada (L627). Contenido sólido y citado (49 claves citadas). |
| Cap. 2 Objetivos (L750–942)        | C            | 1 objetivo general, 4 específicos, 4 hipótesis, variables independientes/dependientes discretas y diseño de casos. Coherentes con la matriz.**PL-10:** hipótesis en una especialización (Guía 01 p. 38).                                                                                                                                            |
| Cap. 3 Metodología (L944–1244)    | P            | **PL-03.** DSR con 10 subsecciones; faltan los rótulos de la guía: tipo, nivel, métodos, población y muestra, diseño, técnicas de recolección y de análisis. "Contexto y unidad de análisis" sustituye a población/muestra. Etapas de intervención presentes.                                                                                  |
| Cap. 4 Marco teórico (L1246–1753) | P            | **PL-02.** Ubicado después de la metodología. Antecedentes temáticos, no cronológicos (**PL-07**). Definición de términos extensa. "Referencias" figura como sección 4.5.                                                                                                                                                                    |
| Cap. V Administración              | **NC** | **PL-01.** No existe.                                                                                                                                                                                                                                                                                                                                     |
| Referencias                         | P            | **PL-02.** Debe ser un bloque propio antes de Anexos, no una sección del Cap. 4.                                                                                                                                                                                                                                                                         |
| Anexos (L1755–)                    | C            | Matriz de consistencia, operacionalización, diseño de casos, revisión documental y repositorio. Numerados como "Capítulo 5".                                                                                                                                                                                                                                |
| Formato                             | C            | A4, márgenes, Arial, numeración: el primer folio de cada capítulo no se imprime (conforme a Guía 01 p. 43).                                                                                                                                                                                                                                                 |
| Compilación                        | C            | `.log`: 0 citas o referencias indefinidas; 6 `Underfull hbox`.                                                                                                                                                                                                                                                                                              |

### 4.2 Estado de la autoevaluación con la rúbrica 8.2 (03-jul-2026)

[evaluacion_formato_8_2_plan_tesis.xlsx](../Plan/evaluacion_formato_8_2_plan_tesis.xlsx) registró 30 criterios: 23 cumplen y 7 no. Situación hoy:

| Criterio no cumplido el 03-jul           | Hoy                 | Verificación                                                |
| ---------------------------------------- | ------------------- | ------------------------------------------------------------ |
| Índice conforme a la guía              | Persiste            | TOC: Metodología cap. 3, Marco cap. 4                       |
| Antecedentes ordenados cronológicamente | Persiste (probable) | Subsecciones temáticas 4.1.1–4.1.4                         |
| Ubicación del proyecto                  | Persiste (parcial)  | Solo "Contexto de aplicación y unidad de análisis" (3.1.5) |
| Esquema de contenido conforme            | Persiste            | Ídem índice; sin Cap. V                                    |
| Cronograma                               | Persiste            | 0 apariciones                                                |
| Presupuesto                              | Persiste            | 0 apariciones                                                |
| Financiamiento                           | Persiste            | 0 apariciones                                                |

Además, esa evaluación cita el título anterior y "cinco objetivos específicos"; el Plan actual tiene cuatro. **La rúbrica debe repetirse sobre la versión vigente.** El Excel también almacena los acentos como "?", lo que impide su lectura formal.

### 4.3 Versiones del Plan

| Versión                                                                   | Fecha       | Páginas       | Título (portada y metadatos PDF)                                                                             |
| -------------------------------------------------------------------------- | ----------- | -------------- | ------------------------------------------------------------------------------------------------------------- |
| `Plan/Plan_Tesis_Grupo91_…_rev0.pdf`                                    | 25-jun-2026 | 46             | "…estimar la temperatura y la ampacidad**en cables enterrados con heterogeneidad térmica**"           |
| `Plan/envío_rev0/02_Informe…pdf`                                       | 23-jul-2026 | 68             | "…estimar la temperatura y la ampacidad**de cables eléctricos enterrados en entornos heterogéneos**" |
| `Plan/envío_rev1/02_Informe…pdf` y `Plan/plan_tesis_cables_pinn.pdf` | 18-ago-2026 | 69             | Ídem 23-jul                                                                                                  |
| `plan_tesis_cables_pinn.tex`                                             | 14-sep-2026 | (sin compilar) | Ídem; añade el "Precisamiento de la formulación térmica y eléctrica"                                     |

Consecuencias: **CT-04** (PDF sin reflejar el `.tex`) y **CT-05** (el título del PDF de junio difiere del de los paquetes de entrega; además hay dos archivos llamados "rev0", de 46 y 68 páginas, lo que dificulta saber cuál se registró ante la Unidad).

### 4.4 Bibliografía del Plan

| Indicador                              | Valor                                                          |
| -------------------------------------- | -------------------------------------------------------------- |
| Entradas en`.bib`                    | 72                                                             |
| Claves citadas / entradas sin citar    | 49 / 23                                                        |
| Citas sin entrada                      | 0                                                              |
| Publicadas desde 2021                  | 63 % de las citadas; 14 anteriores a 2016 (normas y clásicos) |
| URL de Google Drive                    | 2 (`carslaw1959conduction`, `patankar1980numerical`)       |
| Sin`doi` ni `url` (no libros)      | 1 (`cigre2025`, con ISBN)                                    |
| Leyendas sin la palabra "Fuente"       | 20 de 36 (mayoría fórmulas)                                  |
| Figuras/tablas sin llamada en el texto | 2 figuras y 3 tablas                                           |

### 4.5 Registro de hallazgos del Plan

| ID    | Sev.     | Hallazgo                                                                                                 | Recomendación      |
| ----- | -------- | -------------------------------------------------------------------------------------------------------- | ------------------- |
| PL-01 | Crítico | Falta el Capítulo V (cronograma, presupuesto, financiamiento)                                           | R-02                |
| PL-02 | Crítico | Metodología precede al Marco teórico; "Referencias" dentro del Cap. 4; anexos numerados como capítulo | R-03                |
| PL-03 | Mayor    | Metodología sin los rótulos 4.1–4.8 de la Guía                                                       | R-05                |
| PL-04 | Mayor    | Cap. I no sigue 1.1 Diagnóstico y 1.2 Identificación                                                   | R-04                |
| PL-05 | Mayor    | Datos generales sin asesor                                                                               | R-06                |
| PL-06 | Menor    | Carátula sin Poppins y con tamaños distintos a la muestra                                              | R-09                |
| PL-07 | Mayor    | Antecedentes temáticos, sin orden cronológico; falta la ubicación del proyecto                        | R-07                |
| PL-08 | Menor    | 20 de 36 leyendas sin "Fuente"; 2 figuras y 3 tablas sin llamada                                         | R-10                |
| PL-09 | Info     | Formato y compilación conformes (A4, márgenes, Arial, 69 pp., 0 referencias indefinidas)               | —                  |
| PL-10 | Menor    | Hipótesis en una tesis de especialización                                                              | R-29 (decisión D5) |

---

## 5. Documento de tesis: auditoría por sección

Archivo principal: [tesis.tex](../Tesis_LaTeX_Borrador_UNI/tesis.tex). PDF: [tesis.pdf](../Tesis_LaTeX_Borrador_UNI/tesis.pdf), 61 páginas, compilado el 15-sep 01:25. Fuentes posteriores al PDF: `03_propuesta_y_desarrollo.tex` (01:58), `04_conclusiones…` (02:04), `05_anexos.tex` (02:06), tablas (02:04) y `02_marco_teorico.tex` (18-sep 22:04).

### 5.1 Hallazgos por sección

| Sección                                                                                                                      | Estado       | Hallazgos                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Portada (L139–175)                                                                                                           | C            | Poppins 22, Arial 16/14/12, "MAESTRO CON MENCIÓN EN". Título de 24 palabras (14 significativas).                                                                                                                                                                                                                                                                                                         |
| Datos generales (L182–198)                                                                                                   | P            | **TS-04.** Sin asesor, área ni duración.                                                                                                                                                                                                                                                                                                                                                           |
| Dedicatoria, agradecimientos, copia de documentos                                                                             | PD           | **TS-08.** Marcador `\pendiente` en rojo a 18 pt (PDF pp. 3, 4, 5). Legítimo hasta el depósito. La declaración de autenticidad (G02 Anexo XIII) figura solo como texto de recordatorio.                                                                                                                                                                                                         |
| Índice y listas                                                                                                              | C            | Contenidos, figuras, tablas y fórmulas.                                                                                                                                                                                                                                                                                                                                                                   |
| Resumen / Abstract ([00_resumen_abstract.tex](../Tesis_LaTeX_Borrador_UNI/capitulos/00_resumen_abstract.tex))                  | **NC** | **TS-02.** Unas 170 palabras por idioma. Presente ("evalúa", "resuelve"). Sin cifras: inserta `explicit_abstract_es.tex` con el texto "todavía está en ejecución".                                                                                                                                                                                                                             |
| Introducción ([00_introduccion.tex](../Tesis_LaTeX_Borrador_UNI/capitulos/00_introduccion.tex))                               | C            | Discursiva, con problema, objeto, artefacto y organización.**TS-03:** paginada en romano.                                                                                                                                                                                                                                                                                                           |
| Cap. I ([01_planteamiento_del_problema.tex](../Tesis_LaTeX_Borrador_UNI/capitulos/01_planteamiento_del_problema.tex))          | C            | 1.1 Diagnóstico, 1.2 Identificación, 1.3 Formulación, 1.4 Objetivos, 1.5 Hipótesis y criterios, 1.6 Metodología.**TS-06:** hipótesis en L108–119; **TS-11:** excluye CAPEX/OPEX (L72).                                                                                                                                                                                                  |
| Cap. II ([02_marco_teorico.tex](../Tesis_LaTeX_Borrador_UNI/capitulos/02_marco_teorico.tex))                                   | P            | Antecedentes temáticos y no cronológicos (**TS-07**); FEM se define solo en el glosario (L332).                                                                                                                                                                                                                                                                                                    |
| Cap. III ([03_propuesta_y_desarrollo.tex](../Tesis_LaTeX_Borrador_UNI/capitulos/03_propuesta_y_desarrollo.tex))                | P            | 3.1–3.5 conformes (3.5 se titula "Impacto y producto reproducible"). Los resultados (`tablas/explicit_results.tex`) incluyen referencias FEM, etapas A y B, insensibilidad a las colocaciones (0 de 8 comparaciones) y efecto piel. Las etapas C, C2 y la confirmación D figuran como "todavía no finalizada" (L118–124 y L185) y la extensión de ampacidad como "en ejecución" (**CT-01**). |
| Cap. IV ([04_conclusiones_y_recomendaciones.tex](../Tesis_LaTeX_Borrador_UNI/capitulos/04_conclusiones_y_recomendaciones.tex)) | **NC** | **TS-01.** L5–15: OE1; L9 "Las etapas numéricas siguen en ejecución"; L11 tabla "en ejecución"; sin OE2 ni OE3. Recomendaciones sin correspondencia uno a uno.                                                                                                                                                                                                                                   |
| Referencias (L224–226)                                                                                                       | P            | **TS-05.** Espacio simple con sangría francesa de 1,27 cm (medido en PDF p. 55).                                                                                                                                                                                                                                                                                                                    |
| Anexos ([05_anexos.tex](../Tesis_LaTeX_Borrador_UNI/capitulos/05_anexos.tex))                                                  | P            | Matrices de consistencia y operacionalización, requisito–evidencia, casos, reproducción.**TS-10:** los cinco anexos tienen etiqueta pero ninguna llamada `\ref`.                                                                                                                                                                                                                                |

### 5.2 Indicadores de redacción y estructura

| Indicador                            | Tesis                                               | Plan                                                     |
| ------------------------------------ | --------------------------------------------------- | -------------------------------------------------------- |
| Palabras por oración (promedio)     | 16,9                                                | 19,8                                                     |
| Oraciones con más de 45 palabras    | 4 (0,7 %)                                           | 10 (1,7 %)                                               |
| Oraciones con más de 90 palabras    | 0                                                   | 0                                                        |
| Primera persona                      | 0                                                   | 0                                                        |
| Adjetivos valorativos                | 5 (relevante ×3, clave, sólido)                   | 19                                                       |
| "en esta tesis" / "presente estudio" | 1 / 1                                               | 1 / —                                                   |
| Párrafos de una sola oración       | 38 de 215 (17,7 %)                                  | 39 de 236 (16,5 %)                                       |
| Figuras / tablas / fórmulas         | 7 / 20 / 13                                         | 15 / 9 / 9                                               |
| Etiquetas sin llamada en el texto    | 44 de 61 (18 tablas, 5 anexos, 4 fórmulas)         | 17 de 58 (2 figuras, 3 tablas)                           |
| Leyendas sin "Fuente"                | 16 de 40                                            | 20 de 36                                                 |
| Cursivas extranjeras                 | `rating`, `feed-forward` (conforme a G01 p. 43) | `rating`, `benchmark`, `cross-linked polyethylene` |

Interpretación: la prosa cumple las guías locales con holgura. Los puntos a corregir son formales (llamadas a tablas y anexos, siglas en primera mención, párrafos de una oración) y no de estilo.

### 5.3 Trazabilidad objetivo → resultado → conclusión (estado real)

| Objetivo                                            | Hipótesis | Evidencia disponible                                                                                                           | Conclusión en la tesis | Estado                                                                             |
| --------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------- | ---------------------------------------------------------------------------------- |
| OE1 Especificación trazable                        | HE1        | Contratos JSON, fichas, matriz requisito–evidencia                                                                            | Sí (párrafo 1)        | Cumplido                                                                           |
| OE2 Artefacto verificado (criterios ≤ 5 %, ≤ 2 %) | HE2        | A:`subdomain_w32_d3` 3/4 aceptadas, "no calificada"; B: `mixed_2x` 2/4; C2: `weight_1000` 2/4; **D: 7/21**         | No                      | **Parcial** en dominio simple; no aceptado en Aras, Kim, relleno ni estratos |
| OE3 Escenarios heterogéneos                        | HE3        | Escenarios heterogéneos de D:`xlpe_dry_far` 3/3, `xlpe_dry_near` 1/3, `xlpe_backfill` 0/3, `xlpe_discrete_layers` 0/3 | No                      | **Parcial**: 2 de 4 escenarios tienen alguna semilla aceptada                |
| OE4 Comparación de ampacidad                       | HE4        | Solo ensayos de humo en`explicit_study/ampacity` (no hay las 9 ejecuciones previstas)                                        | "En ejecución"         | **No ejecutado** en la formulación explícita                               |

### 5.4 Registro de hallazgos de la tesis

| ID    | Sev.     | Hallazgo                                                                                       | Recomendación |
| ----- | -------- | ---------------------------------------------------------------------------------------------- | -------------- |
| TS-01 | Crítico | Conclusiones solo de OE1; recomendaciones sin correspondencia uno a uno                        | R-24           |
| TS-02 | Crítico | Resumen y Abstract en presente y sin resultados                                                | R-25           |
| TS-03 | Mayor    | Introducción en romano y números impresos en los inicios de capítulo                        | R-26           |
| TS-04 | Mayor    | Datos generales sin asesor, área ni duración                                                 | R-06           |
| TS-05 | Menor    | Referencias a espacio simple (conflicto entre guías)                                          | R-30           |
| TS-06 | Menor    | Sección de hipótesis en una tesis de especialización                                        | R-29           |
| TS-07 | Mayor    | Antecedentes no cronológicos; 50 % de las citas desde 2021                                    | R-07           |
| TS-08 | Menor    | Preliminares personales y declaración de autenticidad pendientes                              | R-33           |
| TS-09 | Menor    | Interlineado nominal de 1,5 medido en 16,4 pt                                                  | R-31           |
| TS-10 | Mayor    | 18 tablas, 5 anexos y 4 fórmulas sin llamada; 16 leyendas sin "Fuente"                        | R-27           |
| TS-11 | Mayor    | Sin análisis de viabilidad y sostenibilidad exigido por la Guía 02; se excluyen CAPEX y OPEX | R-28           |
| TS-12 | Menor    | Siglas sin definir en la primera mención y 38 párrafos de una sola oración                  | R-32           |

---

## 6. Modelamiento PINN/FEM

### 6.1 Inventario verificado

| Elemento                                          | Valor                                                                                                                 |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `pinn_cables/`                                  | 50 archivos`.py`, 8 797 líneas                                                                                     |
| `Benchmarks/` (código activo, nivel 0)         | 48`.py`, 4 559 líneas                                                                                              |
| Copias de fuentes archivadas dentro de resultados | 1 907`.py` en `explicit_study/`; 355 en `comparisons/`; 192 en `multiscale_exploration/`; 161 en `results/` |
| `scripts/`                                      | 34`.py`, 5 210 líneas                                                                                              |
| Raíz                                             | 38`.py` (26 con prefijo `_`) y 18 archivos de otro tipo                                                           |
| Pruebas recolectadas (hoy)                        | `pinn_cables/tests`: 307 (+3 lentas excluidas); `Benchmarks/tests`: 59                                            |
| Repositorio git                                   | 8 765 archivos versionados; pack de 2,30 GiB                                                                          |

### 6.2 Dos líneas de evidencia

| Aspecto                     | Línea reducida (10-sep)                                           | Línea explícita (desde 14-sep)                                                                              |
| --------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Modelo                      | Suelo 2D con capas del cable reducidas a una resistencia radial    | Ecuación de calor en conductor, capas y suelo con interfaces                                                 |
| Volumen                     | 19 casos, 290 entrenamientos, 57 FEM de verificación              | 101 entrenamientos A–D + 9 previstos de ampacidad; 8 casos con FEM refinado                                  |
| Documentos que la describen | `PENDIENTES.md`, README de la tesis, `docs/AUDITORIA_TESIS.md` | `README.md` raíz, `docs/README.md`, `Benchmarks/FULL_DOMAIN.md`, `docs/REVISION_FISICA_EXPLICITA.md` |
| Manifiesto                  | `Benchmarks/MANIFEST.json` (10-sep; sin `explicit_study`)      | Manifiestos por etapa (`explicit_study/*/manifest.json`)                                                    |
| Estado                      | Histórica ("legacy-reduced")                                      | Activa; sin informe consolidado de C, C2 y D                                                                  |

### 6.3 Resultados de la campaña explícita

**Etapas de desarrollo (semillas 11 y 23):**

| Etapa | Objetivo                              | Selección           | Aceptadas de 4 | ¿Calificada? |
| ----- | ------------------------------------- | -------------------- | -------------- | ------------- |
| A     | Arquitectura y tamaño (9 candidatas) | `subdomain_w32_d3` | 3              | No            |
| B     | Colocaciones (6 políticas)           | `mixed_2x`         | 2              | No            |
| C     | Tasa de aprendizaje                   | `lr_0.0005`        | 2              | No            |
| C2    | Peso de continuidad                   | `weight_1000`      | 2              | No            |

**Etapa D (semillas nuevas 71, 83 y 97; receta congelada):**

| Caso                     | Aceptadas | RMSE (K)   | Error de Tmax (K) | NRMSE (%) | Balance (%) | Causa dominante                                                                                                                                                          |
| ------------------------ | --------- | ---------- | ----------------- | --------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `xlpe_single`          | 3/3       | 0,24–0,26 | 0,14–0,38        | 1,4–1,5  | 0,50–0,87  | —                                                                                                                                                                       |
| `xlpe_dry_far`         | 3/3       | 0,26–0,34 | 0,33–0,44        | 1,6–2,0  | 0,79–0,96  | —                                                                                                                                                                       |
| `xlpe_dry_near`        | 1/3       | 0,50–0,75 | 0,86–2,47        | 2,2–3,3  | 1,1–1,7    | Error de Tmax de 2,4 K en semillas 71 y 97                                                                                                                               |
| `xlpe_discrete_layers` | 0/3       | 0,16–0,47 | 0,23–0,80        | 0,9–2,7  | 0,8–1,0    | Interfaces: salto de flujo RMS de 1,88 % (semilla 71, interfaz 0) frente al límite de 1 %; violación máxima de 1,6–1,9 en las tres semillas, con temperatura precisa |
| `xlpe_backfill`        | 0/3       | 0,73–1,19 | 0,77–1,35        | 5,4–8,8  | 1,9–2,9    | NRMSE > 5 % y balance regional > 2 %                                                                                                                                     |
| `aras_flat`            | 0/3       | 1,70–2,63 | 7,5–8,3          | 5,6–8,8  | 0,07–0,24  | Error de Tmax de 7–8 K                                                                                                                                                  |
| `kim_sand`             | 0/3       | 1,63–2,50 | 6,4–8,6          | 5,5–8,4  | 1,4–2,7    | Error de Tmax de 6–9 K; balance hasta 2,68 %                                                                                                                            |

Tiempos por corrida: 500–675 s en casos XLPE, ~1 280 s en `aras_flat` y ~5 500 s en `kim_sand` (125 038 parámetros). La receta que resuelve casos simples no transfiere a las geometrías de literatura (Aras y Kim), que son precisamente las que dan valor práctico al artefacto.

### 6.4 Hallazgos abiertos de la auditoría del 14-sep (A1–A5) frente al estado actual

| ID | Hallazgo original                          | Estado hoy                                                                                                                               | Evidencia                                                                                                                 |
| -- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| A1 | Capas del cable fuera del dominio          | **Atendido** por la formulación explícita (verificado en coaxial y XLPE simple)                                                  | `FULL_DOMAIN.md`; D con capas explícitas                                                                               |
| A2 | Aceptación sin interfaces                 | **Atendido**: la puerta incluye salto de T ≤ 0,1 K y salto de flujo ≤ 1 %                                                        | `03_propuesta_y_desarrollo.tex` L114–118; `explicit_study/D/production/xlpe_discrete_layers/seed71/pinn_seed71.json` |
| A3 | Selección sin confirmación independiente | **Atendido** procedimentalmente (D con semillas nuevas); resultado desfavorable                                                    | `explicit_study/D/summary.json`                                                                                         |
| A4 | Transiciones`tanh` y equivalencia        | **Abierto** para los casos de parche; falta comparación con interfaz exacta                                                       | `explicit_cases.tex` (tabla de parches)                                                                                 |
| A5 | Curva de resolución multicapa             | **Abierto/parcial**: B compara 1×, 2× y 4×, pero la comprobación de insensibilidad se cumple en **0 de 8** comparaciones | `tablas/explicit_results.tex` L181                                                                                      |

### 6.5 Hallazgos de modelamiento y calidad de ingeniería

| ID    | Sev.     | Hallazgo                                                                                                                                                                                                                                                                                                                                | Evidencia                                                                                                |
| ----- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| MD-01 | Crítico | Fracaso de D en los cuatro casos de mayor interés y ausencia de ampacidad explícita                                                                                                                                                                                                                                                   | §6.3;`explicit_study/ampacity/` solo tiene `implementation_smoke`, `references`, `replay_smoke` |
| MD-02 | Mayor    | El informe consolidado está obsoleto:`report/summary.json` cubre 60 corridas (solo A y B) y muestra `production 0/0`, mientras `status.json` declara 7/21                                                                                                                                                                        | `explicit_study/report/`; `explicit_study/status.json`                                               |
| MD-03 | Mayor    | No hay tabla de restricción dominante por corrida en D (existe para A y B)                                                                                                                                                                                                                                                             | `tablas/explicit_results.tex` L52 y L99                                                                |
| MD-04 | Mayor    | `pyproject.toml` fija `testpaths` solo en `pinn_cables/tests`; `pytest` desde la raíz omite las 59 pruebas de `Benchmarks/tests`. Los totales 346 (10-sep), 362 (14-sep), 307+59 (hoy) no se reconcilian                                                                                                                     | `pyproject.toml`; `evidencia/salidas`                                                                |
| MD-05 | Mayor    | Sin CI (`.github/workflows` no existe)                                                                                                                                                                                                                                                                                                | Verificado                                                                                               |
| MD-06 | Mayor    | `MANIFEST.json` no incluye `explicit_study` (0 coincidencias)                                                                                                                                                                                                                                                                       | Verificado                                                                                               |
| MD-07 | Menor    | 926 archivos con rutas absolutas (`C:\usr\…`, `/home/lkoc/…`), sobre todo `job.json` y `manifest.json` de `explicit_study`; además 19 archivos de código (`.py` y `.js`) fuera de `explicit_study` con `C:\Users\QU1267\…` fijo (scripts Zotero, `scripts/build_envio_rev0.py`, `temas_cursos_Tesis_1/*.py`) | `evidencia/salidas/rutas_usuario.txt`; `evidencia/scripts/repocheck.py`                              |
| MD-08 | Menor    | 1 907 copias de fuentes`.py` en `explicit_study` inflan el repositorio (pack de 2,30 GiB)                                                                                                                                                                                                                                           | Conteo por carpeta                                                                                       |
| MD-09 | Menor    | `pyproject.toml` sin límites superiores (`torch>=2.1`); las versiones fijas están en `Benchmarks/environment/requirements-review.txt` y `conda-explicit-linux-64.txt`                                                                                                                                                         | Verificado                                                                                               |
| MD-10 | Menor    | `docs/REVISION_FISICA_EXPLICITA.md` enlaza a `REVISION_INTEGRAL_TESIS.md`, que no existe                                                                                                                                                                                                                                            | `file_search` sin resultados                                                                           |

### 6.6 Validez científica y riesgos

| Riesgo                       | Descripción                                                                                                | Mitigación propuesta                                                                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Verificación ≠ validación | No hay datos de campo; el acuerdo PINN–FEM no prueba el modelo físico. La tesis lo declara correctamente. | Añadir una "matriz de validez" (qué se verificó, con qué referencia, qué no) y contraste con valores publicados (Aras 2005, Kim 2025) donde existan. |
| Generalización              | Reentrenamiento por caso; la receta no transfiere a Aras ni Kim.                                            | Reportar como límite del dominio aceptado (`xlpe_single`, `xlpe_dry_far`).                                                                           |
| Costo computacional          | ~5 500 s por corrida en`kim_sand`. Sin ventaja de tiempo frente a FEM (la tesis lo reconoce).             | Documentar costo por caso y no afirmar aceleración.                                                                                                      |
| Incertidumbre de entradas    | k, R20, humedad, contactos sin propagación de incertidumbre.                                               | Sensibilidad de un solo factor sobre 1–2 casos, como límite declarado.                                                                                  |
| Modelo eléctrico            | DC con R(T); AC completa, proximidad y transitorio no ejecutados.                                           | Ya declarado; mantener en resumen y conclusiones.                                                                                                         |
| Pocas semillas               | 2 en desarrollo y 3 en confirmación.                                                                       | Reportar rangos; evitar generalizar tasas de aceptación.                                                                                                 |

---

## 7. Bibliografía y gestión de fuentes

| Indicador             | Plan                                                                    | Tesis                                                                   |
| --------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Archivos`.bib`      | `referencias_plan_tesis.bib`                                          | `referencias.bib` + `Benchmarks/references_selection.bib`           |
| Entradas              | 72                                                                      | 73 + 15 = 88                                                            |
| Claves citadas        | 49                                                                      | 50                                                                      |
| Citas sin entrada     | 0                                                                       | 0                                                                       |
| Entradas sin citar    | 23                                                                      | 38                                                                      |
| Citadas desde 2021    | 31 (63 %)                                                               | 25 (50 %)                                                               |
| Citadas 2016–2020    | 4                                                                       | 11                                                                      |
| Citadas antes de 2016 | 14                                                                      | 14                                                                      |
| Tipos citados         | 21 artículos, 10 informes técnicos, 6 en línea, 7 libros, 4 reportes | 24 artículos, 11 en línea, 5 informes técnicos, 5 libros, 4 reportes |

Hallazgos:

- **BI-01 (Menor).** Los `.bib` de Plan y tesis comparten 72 entradas con una diferencia de campos (`weiss2021regulacion`). La tesis añade `raissi2017`. Conviene un único `.bib` compartido.
- **BI-02 (Menor).** 23 y 38 entradas sin citar; `biblatex` no las imprime, pero ensucian el mantenimiento.
- **BI-03 (Menor).** Dos URL a Google Drive en `carslaw1959conduction` y `patankar1980numerical`.
- **BI-04 (Menor).** Seis entradas en línea de la tesis sin `urldate` (`baratta2023dolfinx`, `deryck2024analysis`, `mishra2020error`, `moseley2021fbpinn`, `raissi2017`, `wang2020fourier`).
- **BI-05 (Sugerencia).** Guía 01 exige referencias de máximo 5 años "salvo textos clásicos". El 50 % de la tesis cumple; declarar en Marco teórico qué referencias son clásicas o normas.
- **BI-06 (Menor).** `scripts/generar_matriz_consistencia.py` y `scripts/corregir_verificacion_matriz.py` codifican un problema general anterior ("¿Cuál es el efecto de la heterogeneidad térmica…?"). Si se ejecutan, pueden sobrescribir la matriz vigente.

---

## 8. Coherencia transversal y trazabilidad

| ID    | Sev.     | Elemento                                | Estado                                                                                                                                                                                  | Evidencia                                                                                       |
| ----- | -------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| CT-01 | Crítico | Estado de la campaña frente a la tesis | Tesis: "en ejecución"; campaña: completada el 15-sep 05:38                                                                                                                            | `tablas/explicit_abstract_es.tex` L1; `explicit_results.tex` L185; `04_conclusiones…` L9 |
| CT-02 | Mayor    | Objetivos frente a resultados           | OE2–OE4 no respaldados (§5.3)                                                                                                                                                         | §5.3                                                                                           |
| CT-03 | Mayor    | Cifras entre documentos                 | 74 pp. frente a 61; 19 casos/290 entrenamientos frente a 101; 346/362/366 pruebas; "PDF revisado 74 páginas"                                                                           | `PENDIENTES.md`; `tesis.pdf`; `pdfcheck.txt`                                              |
| CT-04 | Mayor    | PDF frente a fuentes                    | Plan: 18-ago frente a 14-sep; tesis: 15-sep 01:25 frente a 18-sep 22:04                                                                                                                 | Marcas de tiempo                                                                                |
| CT-05 | Mayor    | Título                                 | El PDF de junio (46 pp.) usa un título anterior;`envío_rev0`, `envío_rev1` y el `.tex` usan el actual. El Excel de la rúbrica cita el anterior. Dos archivos se llaman "rev0" | Metadatos y portadas PDF (`versiones_plan.txt`); `xlsxdump.txt`                             |
| CT-06 | Menor    | Enlaces documentales                    | `REVISION_INTEGRAL_TESIS.md` inexistente                                                                                                                                              | `docs/REVISION_FISICA_EXPLICITA.md`                                                           |
| CT-07 | Menor    | Entregas                                | `Plan/envío_rev1` contiene el PDF del 18-ago; el `.tex` posterior no está entregado                                                                                               | Listado de carpetas                                                                             |
| CT-08 | Menor    | `Datos generales`                     | Ni el Plan ni la tesis incluyen asesor (Guía 01 §5.1)                                                                                                                                 | `tesis.tex` L182–198; Plan L247–286                                                         |
| CT-09 | Info     | Consistencia Plan ↔ tesis              | Objetivos, problemas específicos, hipótesis, umbrales (5 %, 2 %) y criterios idénticos                                                                                               | Lectura comparada                                                                               |

---

## 9. Repositorio, reproducibilidad y seguridad

| ID    | Sev.       | Hallazgo                                                                                                                                                                                                                                                                                         | Evidencia                                                  |
| ----- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| RP-01 | Mayor      | `LICENSE.md` contiene solo `tbd`.                                                                                                                                                                                                                                                            | Verificado                                                 |
| RP-02 | Menor      | `.gitignore` tiene BOM y 4 reglas con doble codificación (p. ej. `InformaciÃ³n/`, `problema de investigaciÃ³n/*.log`, `Plan/envÃ­o_rev0/`, `/Plan/envÃ­o_rev*/01_Referencias_Zotero.zip`). No coinciden con las carpetas reales y `git status` las muestra sin seguimiento. | `evidencia/salidas/otras_verificaciones.txt`             |
| RP-03 | Menor      | Raíz con 56 archivos: 38`.py` (26 con `_`), sondas (`__cmd_probe.txt`, `__ps_probe*.txt`), `kim_python_procs.txt`, `validate_output.txt`, `zotero_network - Copy.html`.                                                                                                           | Verificado                                                 |
| RP-04 | Menor      | 19 archivos de código con`C:\Users\QU1267\…` fijo, además de `estado_arte_zotero_additions.json` (66 apariciones).                                                                                                                                                                        | `evidencia/salidas/rutas_usuario.txt`                    |
| RP-05 | Info       | No se encontraron secretos por patrón (claves, tokens, contraseñas);`.engie/` está ignorado.                                                                                                                                                                                                | `repocheck.py`                                           |
| RP-06 | Menor      | Repositorio de 2,30 GiB en pack. Las entregas contienen un ZIP de 254,5 MB (`01_Referencias_Zotero.zip`) que la regla corrupta de `.gitignore` no excluye. Puede chocar con límites de alojamiento.                                                                                         | `git count-objects -vH`; listado de `Plan/envío_rev*` |
| RP-07 | Menor      | Sin CI, sin linter, sin verificación automática de compilación LaTeX.                                                                                                                                                                                                                         | Verificado                                                 |
| RP-08 | Info       | Las entregas Drive y`envío_rev*` conservan hashes (`01_SHA256SUMS.txt`).                                                                                                                                                                                                                    | Listado                                                    |
| RP-09 | Sugerencia | Guías y PDF de terceros dentro del repositorio: revisar derechos de redistribución antes de publicar.                                                                                                                                                                                          | —                                                         |

---

## 10. Registro de recomendaciones

Cada recomendación queda **pendiente de su decisión**. Marque una casilla por fila (A = Aprobar, R = Rechazar, M = Modificar). Esfuerzo relativo: B bajo, M medio, A alto. El orden dentro de cada ola respeta las dependencias del §11.

Resumen: 40 recomendaciones; 7 críticas, 16 mayores, 15 menores y 2 sugerencias.

### Ola 0 — Decisiones y verificación con el asesor

| ID   | Sev.     | Recomendación                                                                                                                                                                                           | Hallazgos                        | Esfuerzo | Criterio de aceptación                    | Decisión |
| ---- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | -------- | ------------------------------------------ | --------- |
| R-01 | Crítico | Reunir al asesor y a la Unidad de Posgrado para fijar por escrito D2 (estructura del Plan), D3 (título), D4 (referencias), D5 (hipótesis) y D6 (viabilidad). Registrar el acuerdo en`PENDIENTES.md`. | PL-01/02, CT-05, TS-11, G-1..G-5 | B        | Acta breve con decisión por punto y fecha | A         |

### Ola 1 — Plan de tesis (no depende de nuevos experimentos)

| ID   | Sev.     | Recomendación                                                                                                                                                                                                                                                                                                                                                                                  | Hallazgos                | Esfuerzo | Criterio de aceptación                                                          | Decisión |
| ---- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | -------- | -------------------------------------------------------------------------------- | --------- |
| R-02 | Crítico | Añadir**Capítulo V. Administración del Plan de Tesis**: 5.1 cronograma en Gantt (de oct-2026 a jun-2027, marcando hitos ya cumplidos), 5.2 presupuesto con rubros a–d de la Guía 01 (equipos, materiales e insumos, servicios, viáticos y publicación) y 5.3 financiamiento (autofinanciamineto)  con su tipo. Archivo: `Plan/plan_tesis_cables_pinn.tex`, antes de los anexos. | PL-01                    | M        | Rúbrica 8.2 ítems 13–15 en "Sí"; compila sin referencias indefinidas         | A         |
| R-03 | Crítico | Reordenar capítulos:**III Marco teórico, IV Metodología**. Extraer "Referencias" del Cap. 4 como bloque propio y separar "Anexos" de la numeración de capítulos. Ajustar `\ref`.                                                                                                                                                                                                   | PL-02                    | M        | Índice idéntico a Guía 01 §5.1; 0 advertencias de referencia                 | A         |
| R-04 | Mayor    | Alinear el Cap. I con la Guía: 1.1 Diagnóstico; 1.2 Identificación y descripción (1.2.1 Antecedentes bibliográficos en síntesis cronológica, 1.2.2 Formulación, 1.2.3 Justificación y alcances). Reactivar la sección comentada de L627.                                                                                                                                              | PL-04                    | M        | Títulos coinciden con Guía 01; rúbrica ítems 4–5                            | A         |
| R-05 | Mayor    | En Metodología añadir los rótulos 4.1–4.8: tipo, nivel, métodos de trabajo, población y muestra (unidad de análisis y muestra intencional de casos), tipo de diseño, técnicas e instrumentos de recolección, técnicas de análisis y etapas, reutilizando el contenido existente.                                                                                                    | PL-03                    | M        | Ocho rótulos presentes con enlace a las subsecciones actuales                   | A         |
| R-06 | Mayor    | Incluir en Datos generales del Plan y de la tesis la fila de asesor y las fechas de inicio y fin; en la tesis, además, área involucrada y duración.                                                                                                                                                                                                                                          | PL-05, TS-04, CT-08      | B        | Seis ítems de Guía 01 §5.1 presentes                                          | A         |
| R-07 | Mayor    | Reordenar cada bloque de antecedentes de más antiguo a más reciente y añadir una tabla de evolución cronológica; añadir subsección "Ubicación del proyecto". Declarar qué referencias son clásicas o normas.                                                                                                                                                                          | PL-07, TS-07, N-09, N-10 | M        | Rúbrica ítems 4 y 11 en "Sí"                                                  | A         |
| R-08 | Mayor    | Recompilar el Plan a partir del`.tex` vigente en una copia limpia, versionarlo como rev2 y archivar los PDF previos.                                                                                                                                                                                                                                                                          | CT-04                    | B        | PDF con fecha posterior al`.tex`; hashes registrados                           | A         |
| R-09 | Menor    | Ajustar la carátula del Plan a la muestra de la Guía 01 (Poppins 22; Arial 16, 14, 12).                                                                                                                                                                                                                                                                                                       | PL-06                    | B        | Fuentes de la portada iguales a la muestra                                       | A         |
| R-10 | Menor    | Higiene del Plan: podar o citar las 23 entradas no usadas del`.bib`, sustituir las 2 URL de Drive, unificar `weiss2021regulacion` con la tesis; añadir llamada en el texto a las 2 figuras y 3 tablas sin cita y "Fuente" a las leyendas que la omiten.                                                                                                                                    | BI-01..03, PL-08         | B        | 0 entradas huérfanas; 0 URL de Drive; 0 etiquetas de figura o tabla sin llamada | A         |
| R-11 | Mayor    | Repetir la rúbrica 8.2 sobre la versión vigente y corregir la codificación de acentos del Excel.                                                                                                                                                                                                                                                                                             | §4.2                    | B        | Excel con acentos legibles y resultado actualizado                               | A         |

### Ola 2 — Modelamiento: cerrar y explicar la campaña

| ID   | Sev.     | Recomendación                                                                                                                                                                                                                                                                                                                                                                            | Hallazgos    | Esfuerzo | Criterio de aceptación                                                           | Decisión |
| ---- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------- | --------------------------------------------------------------------------------- | --------- |
| R-12 | Crítico | Ejecutar D1: (a) fijar por escrito el**dominio aceptado hoy** (`xlpe_single`, `xlpe_dry_far`; `xlpe_dry_near` restringido); (b) decidir si se abre la ronda E para `xlpe_discrete_layers`, `xlpe_backfill` y `xlpe_dry_near` con criterios y semillas nuevos preinscritos; (c) no reabrir Aras ni Kim sin un cambio de arquitectura o de presupuesto que se justifique. | MD-01        | A        | Protocolo E firmado antes de ejecutar o texto de restricción de dominio aprobado | E         |
| R-13 | Mayor    | Añadir al informe la**restricción dominante por corrida** para D (y C, C2), incluida la causa de rechazo de `xlpe_discrete_layers` (salto de flujo).                                                                                                                                                                                                                            | MD-03        | B        | Tabla equivalente a`tab:explicit-A-diagnosis` para D                            | A         |
| R-14 | Mayor    | Ejecutar la extensión de**ampacidad explícita** prevista (3 casos × 3 semillas) o declarar OE4 como no alcanzado en la formulación explícita, con el motivo.                                                                                                                                                                                                                   | MD-01, §5.3 | A        | 9 corridas archivadas o texto de OE4 aprobado                                     | A         |
| R-15 | Mayor    | Regenerar`explicit_study/report/` con A–D y producción; conciliar `status.json`; extender `MANIFEST.json` a `explicit_study` con SHA-256.                                                                                                                                                                                                                                       | MD-02, MD-06 | M        | `summary.json` con `production_total = 21`; manifiesto con la campaña        | A         |
| R-16 | Mayor    | Unificar pruebas: incluir`Benchmarks/tests` en la ruta por defecto de `pytest`; ejecutar la suite completa y registrar el total en un archivo generado; reconciliar 346, 362 y 366.                                                                                                                                                                                                   | MD-04        | B        | `pytest` en la raíz ejecuta ambos conjuntos; total documentado                 | A         |
| R-17 | Mayor    | Añadir integración continua mínima (pruebas rápidas y verificación de que el`.tex` compila).                                                                                                                                                                                                                                                                                       | MD-05, RP-07 | M        | Flujo de GitHub Actions verde en la rama principal                                | A         |
| R-18 | Mayor    | Redactar una "matriz de validez" (qué se verifica, contra qué referencia y qué no) y añadir un contraste con valores publicados donde existan datos (Aras 2005, Kim 2025).                                                                                                                                                                                                            | §6.6        | M        | Tabla en Cap. III y citas a cada fuente                                           | A         |
| R-19 | Menor    | Sustituir rutas absolutas por relativas o por variables de entorno en manifiestos y scripts; registrar el intérprete FEM en un archivo de entorno.                                                                                                                                                                                                                                       | MD-07, RP-04 | M        | 0 rutas`C:\Users\…` en scripts; `job.json` reproducible en otra máquina     | A         |
| R-20 | Menor    | Reducir el peso del repositorio: sustituir las copias de fuentes por una tienda direccionada por hash y referencias en cada manifiesto.                                                                                                                                                                                                                                                   | MD-08, RP-06 | A        | Pack menor con verificación de hashes intacta                                    | A         |
| R-21 | Menor    | Fijar rangos de dependencias en`pyproject.toml` y enlazar `requirements-review.txt` y el entorno WSL desde el README.                                                                                                                                                                                                                                                                 | MD-09        | B        | Versiones acotadas y documentadas                                                 | A         |
| R-22 | Menor    | Marcar como históricos (banner) los documentos de la línea reducida y corregir el enlace roto.                                                                                                                                                                                                                                                                                          | CT-06, MD-10 | B        | 0 enlaces rotos en`docs/`                                                       | A         |

### Ola 3 — Tesis (depende de la Ola 2)

| ID   | Sev.     | Recomendación                                                                                                                                                                                                                                                    | Hallazgos         | Esfuerzo | Criterio de aceptación                                                           | Decisión              |
| ---- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------- | --------------------------------------------------------------------------------- | ---------------------- |
| R-23 | Crítico | Regenerar las tablas con`scripts/build_final_thesis.py` y eliminar todo texto "en ejecución"; integrar C, C2, D y, si procede, la ampacidad.                                                                                                                   | CT-01             | M        | 0 apariciones de "en ejecución" en`tablas/` y `capitulos/`; tabla D presente | A                      |
| R-24 | Crítico | Reescribir**Conclusiones y recomendaciones** con cuatro conclusiones (OE1–OE4), cada una con su evidencia cuantitativa y estado (cumplido, parcial, no alcanzado), y una recomendación por conclusión.                                                   | TS-01             | M        | Cuatro pares conclusión–recomendación trazables a tablas                       | A                      |
| R-25 | Crítico | Reescribir**Resumen y Abstract** en pasado con introducción, objetivo, diseño, resultados numéricos y conclusión.                                                                                                                                       | TS-02             | B        | Estructura de Guía 02 p. 14 con cifras de D                                      | A                      |
| R-26 | Mayor    | Corregir la paginación: introducción con numeración arábiga y sin número visible; sin número en el primer folio de cada capítulo (la técnica del Plan ya lo hace).                                                                                        | TS-03             | B        | PDF con números ausentes en inicios de capítulo                                 | A                      |
| R-27 | Mayor    | Añadir llamada`\ref` con oración de anuncio para las 18 tablas, los 5 anexos y las 4 fórmulas sin cita; estandarizar "Fuente:" en las 16 leyendas sin ella.                                                                                                  | TS-10, N-17       | M        | 0 etiquetas de tabla o anexo sin llamada                                          | A                      |
| R-28 | Mayor    | Ampliar el apartado de impacto con un análisis mínimo de viabilidad y sostenibilidad: costo computacional por caso (registrado), riesgos y beneficio-costo cualitativo; sin afirmar ahorro económico ni aceleración. Si el asesor lo exime, dejar constancia. | TS-11             | M        | Sección con tabla de costo y matriz de riesgo, o exención escrita               | A                      |
| R-29 | Menor    | Renombrar "Hipótesis y criterios de contraste" a "Criterios de aceptación y proposiciones de diseño" y justificar en una frase la conservación (D5).                                                                                                          | TS-06             | B        | Título y justificación aprobados                                                | A                      |
| R-30 | Menor    | Decidir el espaciado de referencias (D4) y, si se elige doble espacio, ajustar el bloque`printbibliography`.                                                                                                                                                    | TS-05             | B        | Referencias con el espaciado acordado y sangría francesa                         | R (manetner espaciado) |
| R-31 | Menor    | Verificar el interlineado: la separación medida entre líneas del cuerpo es 16,4 pt; un "1,5 líneas" de Word con Arial 11 ronda 19 pt. Probar`\setstretch{1.4}` y comparar el número de páginas.                                                            | TS-09             | B        | Decisión registrada con muestra visual                                           | A                      |
| R-32 | Menor    | Definir siglas en su primera mención (FEM, XLPE, IEC, CIGRÉ, PDE, NRMSE, SEIN, RMSE) y revisar los 38 párrafos de una sola oración; sustituir "en esta tesis" y "presente estudio".                                                                           | TS-12             | B        | 0 siglas sin definir en primera mención                                          | A                      |
| R-33 | Menor    | Para el depósito: insertar dedicatoria y agradecimientos auténticos, documentos administrativos y la**declaración de autenticidad y no plagio**; preparar los 5 juegos empastados en Guinda y el lomo.                                                   | TS-08, N-24, N-25 | B        | 0 marcas`\pendiente` en el PDF final                                            | A                      |

### Ola 4 — Repositorio y bibliografía

| ID   | Sev.       | Recomendación                                                                                                                                                   | Hallazgos    | Esfuerzo | Criterio de aceptación                                       | Decisión |
| ---- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------- | ------------------------------------------------------------- | --------- |
| R-34 | Mayor      | Definir la licencia (código y texto por separado) y revisar los PDF de terceros antes de publicar.                                                              | RP-01, RP-09 | B        | `LICENSE.md` con texto definido; inventario de terceros     | A         |
| R-35 | Menor      | Guardar`.gitignore` sin BOM y con las cuatro reglas corregidas.                                                                                                | RP-02        | B        | `git status` sin ruido de esas carpetas                     | A         |
| R-36 | Menor      | Ordenar la raíz: mover scripts Zotero a`scripts/zotero/`, mover o eliminar sondas y el duplicado `zotero_network - Copy.html` (solo tras su confirmación). | RP-03        | B        | Raíz con solo README, licencia,`pyproject.toml` y carpetas | A         |
| R-37 | Menor      | Unificar el`.bib` del Plan y de la tesis; incorporar `references_selection.bib`; completar `urldate` en 6 entradas.                                        | BI-01, BI-04 | B        | Un solo`.bib` fuente                                        | A         |
| R-38 | Menor      | Actualizar los generadores de la matriz para leer del Plan vigente o marcarlos como obsoletos.                                                                   | BI-06        | B        | Matriz regenerada sin diferencias con el Plan                 | A         |
| R-39 | Sugerencia | Añadir un`ESTADO.md` generado que sea la única fuente de cifras (páginas, corridas, pruebas, aceptación) y enlazarlo desde `PENDIENTES.md` y los README. | CT-03        | M        | Cifras de documentos coinciden con`ESTADO.md`               | A         |
| R-40 | Sugerencia | Si la Universidad lo ofrece, pasar un reporte de similitud antes del depósito (las guías solo exigen la declaración de autenticidad).                         | N-24         | B        | Reporte archivado                                             | A         |

---

## 11. Ruta de implementación por olas

| Ola | Contenido                 | Depende de                    | Habilita                 |
| --- | ------------------------- | ----------------------------- | ------------------------ |
| 0   | R-01                      | —                            | Todas                    |
| 1   | R-02..R-11 (Plan)         | R-01                          | Aprobación del Plan     |
| 2   | R-12..R-22 (modelamiento) | R-01 (D1)                     | Ola 3                    |
| 3   | R-23..R-33 (tesis)        | Ola 2, R-06                   | Revisión del asesor     |
| 4   | R-34..R-40 (repositorio)  | Ninguna; puede ir en paralelo | Publicación y depósito |

Orden mínimo para desbloquear al asesor: **R-01, R-02, R-03, R-12, R-23, R-24, R-25**.

---

## 12. Anexos

### A. Evidencia reproducible

| Comprobación                                  | Script                                                                                       | Salida                                                                                                  |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Claves`\cite` frente a `.bib`              | [bibcheck.py](evidencia/scripts/bibcheck.py)                                                  | [bibcheck.txt](evidencia/salidas/bibcheck.txt)                                                           |
| Antigüedad de referencias                     | [bibyears.py](evidencia/scripts/bibyears.py)                                                  | [bibyears.txt](evidencia/salidas/bibyears.txt)                                                           |
| Geometría, fuentes, páginas y resumen en PDF | [pdfcheck.py](evidencia/scripts/pdfcheck.py)                                                  | [pdfcheck.txt](evidencia/salidas/pdfcheck.txt)                                                           |
| Numeración por capítulo                      | [pagecheck.py](evidencia/scripts/pagecheck.py)                                                | [pagecheck.txt](evidencia/salidas/pagecheck.txt)                                                         |
| Interlineado y sangría en referencias         | [refcheck.py](evidencia/scripts/refcheck.py)                                                  | [refcheck.txt](evidencia/salidas/refcheck.txt)                                                           |
| Advertencias de`.log`                        | [logcheck.py](evidencia/scripts/logcheck.py)                                                  | [logcheck.txt](evidencia/salidas/logcheck.txt)                                                           |
| Métricas de redacción y siglas               | [style.py](evidencia/scripts/style.py)                                                        | [style.txt](evidencia/salidas/style.txt)                                                                 |
| Conteos y leyendas sin "Fuente"                | [conteos_latex.py](evidencia/scripts/conteos_latex.py)                                        | [conteos_latex.txt](evidencia/salidas/conteos_latex.txt)                                                 |
| Etiquetas sin llamada                          | [etiquetas_sin_referencia.py](evidencia/scripts/etiquetas_sin_referencia.py)                  | [etiquetas_sin_referencia.txt](evidencia/salidas/etiquetas_sin_referencia.txt)                           |
| Campaña explícita                            | [explicit_sum.py](evidencia/scripts/explicit_sum.py), [dgate2.py](evidencia/scripts/dgate2.py) | [explicit_sum.txt](evidencia/salidas/explicit_sum.txt), [dgate2.txt](evidencia/salidas/dgate2.txt)        |
| Excel de matriz y rúbrica                     | [xlsxdump.py](evidencia/scripts/xlsxdump.py)                                                  | [xlsxdump.txt](evidencia/salidas/xlsxdump.txt)                                                           |
| Higiene del repositorio                        | [repocheck.py](evidencia/scripts/repocheck.py)                                                | Resultados resumidos en[otras_verificaciones.txt](evidencia/salidas/otras_verificaciones.txt)            |
| Versiones del Plan                             | [versiones_plan.py](evidencia/scripts/versiones_plan.py)                                      | [versiones_plan.txt](evidencia/salidas/versiones_plan.txt)                                               |
| Rutas fijas del usuario                        | [rutas_usuario.py](evidencia/scripts/rutas_usuario.py)                                        | [rutas_usuario.txt](evidencia/salidas/rutas_usuario.txt)                                                 |
| Control de calidad de este documento           | [validar_documento.py](evidencia/scripts/validar_documento.py)                                | [validar_documento.txt](evidencia/salidas/validar_documento.txt)                                         |
| OCR de las guías                              | [ocr.ps1](evidencia/scripts/ocr.ps1), [renderall.py](evidencia/scripts/renderall.py)           | [Guía 01](evidencia/ocr_guias/ocr_guia01_plan.txt), [Guía 02](evidencia/ocr_guias/ocr_guia02_tesis.txt) |

Los scripts fijan la raíz del proyecto con `os.chdir` (constante al inicio de cada archivo); ajustarla al reproducir en otra máquina. Requieren Python 3.12 con PyMuPDF; `bibyears.py` importa `bibcheck.py` desde la carpeta de trabajo temporal usada durante la auditoría. Las salidas se guardaron en UTF-8.

### B. Correcciones a los informes de apoyo

| Afirmación de un informe de apoyo                             | Resultado de la verificación                                                                                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| "El`.bib` de la tesis tiene ~130 entradas"                   | 73 + 15 = 88                                                                                                                                     |
| "`cigre2025` no tiene ISBN"                                  | Tiene ISBN; no tiene`doi` ni `url`                                                                                                           |
| "El resumen debería tener 150–250 palabras (Guía 02)"       | Ninguna guía fija un mínimo; ~170 palabras por idioma                                                                                          |
| "`\pendiente` en 5 páginas de la tesis"                     | Solo 3 (dedicatoria, agradecimientos, copia de documentos); el resto era "INDEPENDIENTES"                                                        |
| "El Plan tiene ~60 páginas / la tesis 74"                     | 69 y 61                                                                                                                                          |
| "Turnitin exigido por la guía"                                | Las guías solo exigen la declaración de autenticidad                                                                                           |
| "95 % de objetivos soportados; 9/11 ampacidades aceptadas"     | Corresponde a la línea reducida; en la explícita, D acepta 7/21 y no hay ampacidad                                                             |
| "Cap. III y Cap. IV sin conclusiones por un`Underfull hbox`" | Son hallazgos independientes: las conclusiones incompletas son reales (TS-01); el`Underfull hbox` es irrelevante (1 en la tesis, 6 en el Plan) |

### C. Código de los hallazgos

| Prefijo   | Dominio                                          |
| --------- | ------------------------------------------------ |
| CT        | Coherencia transversal                           |
| PL        | Plan de tesis                                    |
| TS        | Documento de tesis                               |
| MD        | Modelamiento PINN/FEM                            |
| BI        | Bibliografía                                    |
| RP        | Repositorio y seguridad                          |
| N-nn, G-n | Requisito normativo, inconsistencia entre guías |

### D. Estado de la carpeta

`auditoria general/` contiene este documento y la subcarpeta `evidencia/` (scripts, salidas y OCR). Al emitirse la auditoría no se modificó ningún otro archivo del proyecto; los cambios posteriores se listan en el §13.3.

---

## 13. Estado de implementación (29-sep-2026)

Restricción fijada por los tesistas: **no hacer cálculos nuevos de PINN ni FEM**; solo cambios en el texto de los documentos. Por eso los trabajos que exigen entrenamientos, FEM nuevo, código, configuración o mover archivos quedan pendientes. Estados: **Hecho**; **Parcial**; **Pendiente (cálculo)**, **Pendiente (código)**, **Pendiente (archivos)** o **Pendiente (datos)**; **Rechazada**.

### 13.1 Estado por recomendación

| ID | Estado | Qué se hizo y qué falta |
| --- | --- | --- |
| R-01 | Parcial | Decisiones D1–D6 registradas en `Tesis_LaTeX_Borrador_UNI/PENDIENTES.md`. Falta la reunión y el acta con el asesor; el título (D3) sigue sin enviarse a la Unidad de Posgrado. |
| R-02 | Hecho | Capítulo V del Plan: 5.1 Gantt de oct-2026 a jun-2027 con bloque cumplido jul–sep, 5.2 presupuesto con rubros a–d y 5.3 autofinanciamiento. Los montos figuran «Por confirmar» y el cronograma debe validarse con el asesor. |
| R-03 | Hecho | Plan: I Problema, II Objetivos, III Marco teórico, IV Metodología, V Administración; Referencias como bloque propio y Anexos fuera de la numeración («Anexo 1» a «Anexo 5»; tablas A.n). |
| R-04 | Hecho | 1.1 Diagnóstico; 1.2 Identificación y descripción del problema con 1.2.1 Antecedentes, 1.2.2 Formulación y 1.2.3 Justificación y alcances. |
| R-05 | Hecho | Rótulos 4.1 a 4.8 con los títulos de la Guía; se redactaron el nivel de investigación, la población y muestra y el tipo de diseño; el contenido existente se reubicó sin cambios. |
| R-06 | Hecho | Datos generales del Plan (asesor, fechas de inicio y culminación) y de la tesis (asesor, área, fechas y duración). |
| R-07 | Hecho | Plan (19 fuentes) y tesis (21): tabla de evolución cronológica, orden cronológico dentro de cada tema, declaración de clásicos, normas y fundamentos, y subsección «Ubicación del proyecto». |
| R-08 | Cumplida | Plan recompilado en copia temporal y guardado como `Plan/Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev2.pdf` con hashes en `Plan/rev2_SHA256.txt`. Los PDF previos (rev0 y la compilación del 18-ago) están archivados en `Plan/old/`. |
| R-09 | Hecho | Carátula del Plan con Poppins 22 y Arial 16, 14 y 12, igual que la tesis. |
| R-10 | Parcial | 23 entradas no citadas movidas a `referencias_plan_tesis_no_usadas.bib`; 2 URL de Drive quitadas de los libros (Plan y tesis); 5 llamadas a figuras y tablas y «Fuente» en 20 leyendas. Falta sustituir la dirección de Google Drive del anexo del repositorio digital por una estable (Pendiente: datos). |
| R-11 | Hecho | Excel 8.2 rehecho sobre el Plan vigente: 30 de 30 criterios, acentos legibles, 5 observaciones abiertas (1 mayor: montos del presupuesto). |
| R-12 | Hecho (texto); Pendiente (cálculo) | Dominio aceptado fijado en la tesis; ronda E preinscrita en `docs/PROTOCOLO_RONDA_E.md`, sin firma y sin ejecutar. |
| R-13 | Parcial | Tablas de restricción dominante de D, C y C2 en el capítulo III; el informe automático aún no las genera (Pendiente: código). |
| R-14 | Parcial | OE4 declarado «no alcanzado» con su motivo en la tesis; las 9 corridas siguen pendientes (Pendiente: cálculo). |
| R-15 | Parcial | `explicit_study/report/` regenerado con A–D y producción; falta extender `MANIFEST.json` con SHA-256 (Pendiente: código). |
| R-16, R-17 | Pendiente (código) | Unificar `pytest` y añadir integración continua. |
| R-18 | Parcial | Matriz de validez en el capítulo III. El contraste con valores publicados (Aras 2005, Kim 2025) exige nuevas referencias FEM con sus pérdidas, propiedades y contornos (Pendiente: cálculo). |
| R-19, R-20, R-21 | Pendiente (código) | Rutas absolutas, peso del repositorio y rangos de dependencias. |
| R-22 | Hecho | Banners históricos en `docs/AUDITORIA_TESIS.md` y `docs/PLAN_AUDITORIA_TESIS.md`, secciones de históricos en ambos README y enlace roto corregido. Los informes generados de `Benchmarks/` no llevan banner porque su generador los reescribe. |
| R-23 | Parcial | Tablas regeneradas y 0 apariciones de «en ejecución» en `tablas/` y `capitulos/`; tabla D presente. Las ediciones manuales de `04` y de las leyendas deben protegerse del generador (véase PENDIENTES.md). |
| R-24, R-25 | Hecho | Conclusiones OE1–OE4 con recomendaciones; Resumen y Abstract en pasado con cifras de D. |
| R-26 | Hecho | Numeración arábiga desde la Introducción y sin número visible en el primer folio de cada capítulo. |
| R-27 | Hecho | 0 etiquetas de tabla, figura, fórmula y anexo sin llamada en tesis y Plan; «Fuente» en todas las leyendas. |
| R-28 | Hecho | Costo computacional por caso, matriz de riesgos, beneficio–costo cualitativo y sostenibilidad; sin afirmar ahorro ni aceleración. Falta el visto bueno del asesor. |
| R-29 | Hecho | Sección renombrada a «Criterios de aceptación y proposiciones de diseño» con su justificación. |
| R-30 | Rechazada | Se mantiene el espaciado simple de las referencias. |
| R-31 | Hecho | Se adoptó `\setstretch{1.4}` (19,0 pt frente a 16,4 pt); la tesis pasa de 82 a 88 páginas. |
| R-32 | Parcial | Siglas definidas en su primera mención; «esta tesis» y «presente estudio» sustituidos. No se revisaron los 38 párrafos de una sola oración. |
| R-33 | Pendiente (datos) | Dedicatoria, agradecimientos, documentos administrativos y declaración de autenticidad. |
| R-34 | Hecho | `LICENSE.md`: MIT para el código y CC BY 4.0 para el texto, con inventario de terceros por carpeta. Falta el inventario archivo por archivo y la revisión con la Universidad. |
| R-35, R-36, R-37, R-38 | Pendiente (código/archivos) | `.gitignore`, orden de la raíz, `.bib` único y generadores de la matriz. |
| R-39 | Parcial | `ESTADO.md` creado a mano como fuente única de cifras y enlazado desde PENDIENTES.md y ambos README; falta el generador (Pendiente: código). |
| R-40 | Pendiente (datos) | Informe de similitud. |

### 13.2 Verificación

- Tesis compilada en copia temporal con `lualatex` y `biber`: 88 páginas, 0 citas o referencias indefinidas.
- Plan compilado del mismo modo: 76 páginas, 0 citas o referencias indefinidas; índice igual al de la Guía N.° 01 (`evidencia/scripts/toc_pdf.py`).
- Sin cálculos nuevos de PINN ni FEM. Se usaron los registros existentes (`restriccion_dominante.py`, `costo_computacional.py`).
- Reorganización posterior de `Plan/` (29-sep-2026): el `.tex` monolítico se dividió en `plan_tesis_cables_pinn.tex` (principal) y `capitulos/` (00 a 06) con un archivo por anexo en `capitulos/anexos/`, como en la tesis. `evidencia/scripts/plan_modularizar.py` verificó que al expandir los `\input` y `\include` se recupera el monolito línea a línea (sin comentarios ni líneas vacías), y la compilación en copia temporal dio las mismas 76 páginas, con texto, índice e imagen idénticos al PDF rev2. Las referencias de línea a `plan_tesis_cables_pinn.tex` de este informe corresponden al monolito, conservado en `Plan/old/plan_tesis_cables_pinn_monolitico_2026-09-29.tex`.

### 13.3 Archivos creados o modificados

- **Plan:** `plan_tesis_cables_pinn.tex` (ahora principal), `capitulos/` y `capitulos/anexos/` (nuevos), `referencias_plan_tesis.bib`, `referencias_plan_tesis_no_usadas.bib` (nuevo), `evaluacion_formato_8_2_plan_tesis.xlsx`, `Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev2.pdf` (nuevo) y `rev2_SHA256.txt` (nuevo). Carpetas ordenadas: `guias/`, `literatura/`, `datos_fuente/` y `old/`; `envío_rev0`, `envío_rev1` y `presentación` no se tocaron.
- **Tesis:** `tesis.tex`, `referencias.bib`, `PENDIENTES.md`, `capitulos/00_introduccion.tex`, `00_resumen_abstract.tex`, `01` a `05`, y `tablas/explicit_*.tex`.
- **Documentos:** `README.md`, `LICENSE.md`, `ESTADO.md` (nuevo), `docs/README.md`, `docs/AUDITORIA_TESIS.md`, `docs/PLAN_AUDITORIA_TESIS.md`, `docs/REVISION_FISICA_EXPLICITA.md`, `docs/ENGIE_AI_BRIDGE.md` (dos enlaces relativos corregidos) y `docs/PROTOCOLO_RONDA_E.md` (nuevo).
- **Evidencia regenerada por el generador del informe (sin nuevos entrenamientos):** `Benchmarks/explicit_study/report/` e `imagenes/explicit/`.
- **Scripts nuevos en `evidencia/scripts/`:** `compilar_temp.ps1`, `plan_reestructurar.py`, `plan_modularizar.py`, `plan_fuentes.py` (lista y expande las fuentes del Plan dividido; lo usan `bibcheck.py`, `bibyears.py`, `conteos_latex.py`, `etiquetas_sin_referencia.py`, `style.py`, `autoria_plan.py`, `bib_podar_plan.py` y `verificar_implementacion.py`), `bib_podar_plan.py`, `rubrica_actualizar.py`, `paginas_capitulos.py`, `toc_pdf.py`, `anios_referencias.py`, `restriccion_dominante.py`, `costo_computacional.py`, `xlsx_rubrica_dump.py` y `verificar_implementacion.py` (comprueba R-01 a R-40 y deja su salida en `evidencia/salidas/verificacion_implementacion.txt`).
