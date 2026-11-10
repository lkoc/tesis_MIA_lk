# Metodologia de busqueda de papers

Fecha de ejecucion: 2026-06-05  
Tema de investigacion: efecto de la heterogeneidad termica del entorno sobre la temperatura del conductor en cables electricos subterraneos en centrales solares y eolicas: modelado numerico 2D.

## Objetivo

Completar y ordenar la base documental de Zotero bajo la coleccion `Tesis_MIA_2026` para que cada carpeta tematica tenga, como minimo:

- 2 papers de estado del arte.
- 2 papers de aporte metodologico/teorico.
- 2 papers de aplicacion.

Se distinguio entre:

- `estado del arte estricto`: review, systematic review, mapping, assessment u overview que reporta bases consultadas, estrategia de busqueda y/o string de busqueda.
- `estado del arte ampliado`: review, assessment, overview, comparison o historical perspective relevante, pero sin string literal completamente reproducible.
- `aporte`: paper que introduce un metodo, modelo, formulacion, algoritmo o marco teorico relevante.
- `aplicacion`: paper que aplica un metodo conocido a un caso concreto, numerico, experimental, normativo o de ingenieria.

## Fuentes consultadas

1. Auditoria local de `C:\Users\QU1267\Zotero\zotero.sqlite`.
2. Colecciones de Zotero bajo `Tesis_MIA_2026`.
3. PDFs locales en `Estado del arte` y `zotero_offline/pdfs`.
4. `Estado del arte/README.md` y `ANALISIS_BIBLIOGRAFICO_ZOTERO.md`.
5. Busqueda web por DOI/titulo en fuentes editoriales y bases abiertas.
6. Metadatos Crossref y disponibilidad OA via OpenAlex.
7. Verificacion textual de PDFs con `pdfminer` para detectar terminos como `search string`, `search query`, `Scopus`, `Web of Science`, `PRISMA` y `methodology`.

## Strings base usados

String general de partida, corregido para comillas y operadores booleanos:

```text
("underground power cable*" OR "buried power cable*" OR "underground cable system*")
AND (thermal OR ampacity OR "thermal rating" OR temperature)
AND (review OR assessment OR systematic OR survey OR mapping OR overview)
```

## Strings por categoria

### 01 tema-pinn - PINNs

Estado del arte:

```text
("physics-informed neural network*" OR PINN* OR "scientific machine learning")
AND ("partial differential equation*" OR PDE OR "heat conduction" OR "heat transfer" OR thermal)
AND (review OR systematic OR bibliometric OR survey OR mapping OR overview)
```

Aporte:

```text
("physics-informed neural network*" OR PINN* OR XPINN* OR "domain decomposition")
AND ("heat conduction" OR "inverse heat transfer" OR "thermal conductivity" OR PDE)
AND (method OR framework OR "forward problem" OR "inverse problem")
```

Aplicacion:

```text
("physics-informed" OR PINN* OR "physics-enhanced" OR "FEM-BPNN")
AND ("underground power cable*" OR "buried cable*" OR "temperature field" OR "heat conduction")
AND (application OR prediction OR reconstruction OR monitoring OR "case study")
```

### 02 tema-transferencia-calor

Estado del arte:

```text
("heat transfer" OR "heat conduction" OR "thermal assessment")
AND ("power cable*" OR "underground cable*" OR soil OR backfill)
AND (review OR overview OR assessment OR systematic)
```

Aporte:

```text
("heat conduction" OR "thermal model")
AND ("power cable*" OR "multi-layered soil" OR "buried cable*")
AND (model OR formulation OR "theoretical model" OR method)
```

Aplicacion:

```text
("heat transfer" OR "temperature field" OR "thermal analysis")
AND ("underground power cable*" OR "buried cable*")
AND ("numerical study" OR simulation OR experiment OR "case study")
```

### 03 tema-suelo-backfill-humedad

Estado del arte:

```text
("soil thermal conductivity" OR "soil thermal resistivity" OR "thermal backfill" OR "unsaturated soils")
AND (review OR "critical review" OR systematic OR assessment OR "future directions")
AND (measurement OR model* OR moisture OR backfill)
```

Aporte:

```text
("soil thermal conductivity" OR "soil thermal resistivity" OR "thermal backfill")
AND ("underground cable*" OR "buried infrastructure" OR "power cable*")
AND (model OR method OR optimization OR measurement)
```

Aplicacion:

```text
("thermal backfill" OR "soil moisture" OR "soil thermal resistivity")
AND ("underground power cable*" OR "buried infrastructure")
AND (experiment OR "case study" OR "field measurement" OR simulation)
```

### 04 tema-fem-numerico

Estado del arte:

```text
("finite element method" OR FEM OR "spectral finite element" OR "adaptive finite element")
AND ("heat transfer" OR "heat conduction" OR "computational mechanics" OR "power cable rating")
AND (review OR "comprehensive review" OR survey OR overview)
```

Aporte:

```text
(FEM OR "finite element" OR "finite difference" OR "numerical simulation")
AND ("heat conduction" OR "thermal analysis" OR "heterogeneous media" OR "multi-layered soil")
AND (method OR formulation OR algorithm OR "inverse problem")
```

Aplicacion:

```text
(FEM OR "finite element" OR "numerical study" OR simulation)
AND ("underground power cable*" OR "buried cable*" OR "cable rating")
AND (application OR "case study" OR "temperature field" OR ampacity)
```

### 05 tema-cables-ampacidad

Estado del arte:

```text
("power cable*" OR "underground power cable*" OR "insulated cable*")
AND (ampacity OR "current rating" OR "thermal rating")
AND (review OR assessment OR overview OR "historical perspective" OR survey)
```

Aporte:

```text
("power cable*" OR "underground cable*")
AND (ampacity OR "current rating" OR "temperature rise" OR "load capability")
AND (method OR calculation OR model OR formulation)
```

Aplicacion:

```text
("underground power cable*" OR "buried power cable*")
AND (ampacity OR "current rating" OR "thermal rating" OR temperature)
AND (simulation OR experiment OR optimization OR "case study" OR "power flow")
```

### 06 tema-dtr-cargas-ciclicas

Estado del arte:

```text
("dynamic thermal rating" OR DTR OR "cyclic loading" OR "emergency rating")
AND ("underground cable*" OR "power cable*" OR "transmission line*")
AND (review OR systematic OR mapping OR assessment OR overview)
```

Aporte:

```text
("dynamic thermal analysis" OR "dynamic thermal rating" OR "cyclic loading")
AND ("underground cable*" OR "power cable*")
AND (method OR model OR "van Wormer coefficient" OR "thermal resistivity")
```

Aplicacion:

```text
("dynamic thermal rating" OR "cyclic loading" OR "transient temperature")
AND ("underground cable*" OR "XLPE cable*" OR "power cable*")
AND (application OR "case study" OR experiment OR "network planning")
```

### 07 tema-normas-estandares

Estado del arte:

```text
(IEC OR IEEE OR CIGRE OR standard*)
AND (ampacity OR "current rating" OR "thermal rating")
AND ("power cable*" OR "underground cable*")
AND (comparison OR assessment OR review OR overview)
```

Aporte:

```text
(IEC OR IEEE OR CIGRE OR ASTM)
AND ("power cable*" OR soil OR backfill)
AND (guide OR standard OR "calculation method" OR "measurement method")
```

Aplicacion:

```text
(IEC OR IEEE OR CIGRE)
AND ("cable ampacity" OR "current rating")
AND (comparison OR example OR verification OR application)
```

## Resultados de auditoria Zotero

Conteo antes de la actualizacion:

| Carpeta | Total | Estado arte | Aporte | Aplicacion | PDF |
|---|---:|---:|---:|---:|---:|
| 01 tema-pinn - PINNs | 21 | 4 | 15 | 1 | 13 |
| 02 tema-transferencia-calor | 13 | 0 | 13 | 0 | 13 |
| 03 tema-suelo-backfill-humedad | 10 | 0 | 5 | 5 | 7 |
| 04 tema-fem-numerico | 8 | 0 | 7 | 1 | 8 |
| 05 tema-cables-ampacidad | 7 | 0 | 4 | 3 | 6 |
| 06 tema-dtr-cargas-ciclicas | 5 | 1 | 1 | 2 | 4 |
| 07 tema-normas-estandares | 16 | 1 | 3 | 2 | 14 |

Actualizacion aplicada a Zotero:

- Respaldo creado: `C:\Users\QU1267\Zotero\zotero.sqlite.codex-paper-search-backup-20260605-161015`.
- Items nuevos creados: 15.
- Items existentes reclasificados o agregados a carpetas faltantes: 10.
- Membresias de coleccion agregadas: 45.
- Tags agregados: 158.
- PDFs adjuntos a Zotero: 3.

Conteo despues de la actualizacion, contando papers unicos por DOI/titulo:

| Carpeta | Estado arte | Aporte | Aplicacion |
|---|---:|---:|---:|
| 01 tema-pinn - PINNs | 5 | 16 | 3 |
| 02 tema-transferencia-calor | 6 | 16 | 4 |
| 03 tema-suelo-backfill-humedad | 4 | 5 | 5 |
| 04 tema-fem-numerico | 4 | 9 | 4 |
| 05 tema-cables-ampacidad | 8 | 6 | 8 |
| 06 tema-dtr-cargas-ciclicas | 3 | 3 | 2 |
| 07 tema-normas-estandares | 3 | 2 | 3 |

Todas las carpetas alcanzan el minimo solicitado de 2 papers por tipo.

## Papers incorporados o reclasificados

### Estado del arte estricto o con busqueda declarada

- Lawal et al. (2022), `Physics-informed neural network (PINN) evolution and beyond: a systematic literature review and bibliometric analysis`, DOI: https://doi.org/10.3390/bdcc6040140. Ya estaba en Zotero. Verificado: PRISMA, Scopus/Web of Science, keyword search.
- Cuomo et al. (2022), `Scientific Machine Learning Through Physics-Informed Neural Networks: Where we are and What's Next`, DOI: https://doi.org/10.1007/s10915-022-01939-z. Agregado a Zotero con PDF OA. Verificado: reporta busqueda en Scopus y string.
- Fariz et al. (2026), `A Systematic Mapping of Dynamic Thermal Rating for Underground Cables`, DOI: https://doi.org/10.1109/ACCESS.2025.3650354. Ya estaba en Zotero y fue agregado tambien a cables/ampacidad. Verificado: PRISMA, Scopus/Web of Science, search string.
- `A Review of Soil Thermal Conductivity Measurement Techniques, Challenges, and Future Directions` (2026), DOI: https://doi.org/10.1007/s11831-026-10509-7. Agregado a Zotero con PDF OA. Verificado: metodologia, Scopus/Web of Science y terminos clave.

### Estado del arte ampliado

- `A Review: Applications of the Spectral Finite Element Method`, DOI: https://doi.org/10.1007/s11831-023-09911-2. Agregado a Zotero con PDF OA.
- `Fractional Spectral and Fractional Finite Element Methods: A Comprehensive Review and Future Prospects`, DOI: https://doi.org/10.1007/s11831-024-10083-w. Agregado a Zotero, sin PDF OA disponible.
- `Adaptive finite element methods in computational mechanics`, DOI: https://doi.org/10.1016/0045-7825(92)90020-K. Agregado a Zotero, sin PDF OA disponible.
- `Critical Review of Thermal Conductivity Models for Unsaturated Soils`, DOI: https://doi.org/10.1007/s10706-015-9843-2. Agregado a Zotero. OpenAlex reporta OA verde, pero el enlace PDF devolvio HTTP 403 durante la descarga automatizada.
- `Review of soil thermal conductivity and predictive models`, DOI: https://doi.org/10.1016/j.ijthermalsci.2017.03.013. Agregado a Zotero, sin PDF OA disponible.
- `Power Cable Rating Calculations - A Historical Perspective [History]`, DOI: https://doi.org/10.1109/MIAS.2015.2417094. Agregado a Zotero, sin PDF OA disponible.
- `Review of high current rating insulated cable solutions`, DOI: https://doi.org/10.1016/j.epsr.2015.12.005. Agregado a Zotero, sin PDF OA disponible.
- `Dynamic thermal rating of transmission lines: A review`, DOI: https://doi.org/10.1016/j.rser.2018.04.001. Agregado a Zotero, sin PDF OA disponible.
- `Comparison between IEEE and CIGRE ampacity standards`, DOI: https://doi.org/10.1109/61.796253. Agregado a Zotero, sin PDF OA disponible.

### Aportes y aplicaciones agregadas/reclasificadas

- `Thermal analysis of power cables in multi-layered soil. I. Theoretical model`, DOI: https://doi.org/10.1109/61.252604. Agregado como aporte metodologico en transferencia de calor, FEM y cables/ampacidad.
- `Thermal analysis of power cables in multi-layered soil. II. Practical considerations`, DOI: https://doi.org/10.1109/61.252605. Agregado como aplicacion numerica en transferencia de calor, FEM y cables/ampacidad.
- `Cable Ampacity Calculations: A Comparison of Methods`, DOI: https://doi.org/10.1109/TIA.2015.2475244. Agregado como aplicacion/comparacion normativa.
- `Cyclic Loading of Underground Cables Including the Variations of Backfill Soil Thermal Resistivity and Specific Heat With Temperature Variation`, DOI: https://doi.org/10.1109/TPWRD.2018.2849017. Agregado como aporte metodologico en DTR y suelo/backfill.
- `Dynamic thermal analysis for underground cables under continuously fluctuant load considering time-varying Van Wormer coefficient`, DOI: https://doi.org/10.1016/j.epsr.2021.107395. Ya existia; se agrego a DTR y transferencia de calor.
- `Adaptive FEM-BPNN model for predicting underground cable temperature considering varied soil composition`, DOI: https://doi.org/10.1016/j.jestch.2024.101658. Ya existia; se marco tambien como aplicacion numerica y se agrego a FEM/transferencia.
- `Numerical study of heat transfer in underground power cable system`, DOI: https://doi.org/10.1016/j.egypro.2019.01.636. Ya existia; se agrego a transferencia de calor y FEM como aplicacion.
- `Effect of ambient air and ground temperatures on heat transfer in underground power cable system buried in newly developed cable bedding material`, DOI: https://doi.org/10.1016/j.geothermics.2024.103151. Ya existia; se agrego a transferencia/cables como aplicacion.

## PDFs

PDFs OA adjuntados a Zotero en esta actualizacion:

- `Scientific Machine Learning Through Physics-Informed Neural Networks`, Springer OA.
- `A Review: Applications of the Spectral Finite Element Method`, Springer OA.
- `A Review of Soil Thermal Conductivity Measurement Techniques, Challenges, and Future Directions`, Springer OA.

PDF descargado al proyecto pero no incorporado por bloqueo de fuente:

- `Critical Review of Thermal Conductivity Models for Unsaturated Soils`: OpenAlex reporto OA verde, pero eScholarship devolvio HTTP 403 al intentar descargar el PDF.

Papers sin PDF adjunto:

- La mayoria de items IEEE/Elsevier agregados quedaron con metadatos y DOI, porque OpenAlex/Crossref no reportaron PDF OA directo o el publisher es de acceso cerrado.

## Observaciones metodologicas

- En PINNs y DTR si se encontraron trabajos estrictos con PRISMA/bases/string.
- En suelo/backfill se encontro al menos un review reciente con metodologia, bases y terminos de busqueda; se completo con critical reviews y reviews de modelos predictivos.
- En FEM numerico, cables/ampacidad y normas, los resultados especificos con string literal fueron escasos. Se completaron los minimos con reviews, assessments, comparisons e historical perspectives de alto ajuste tematico.
- No se eliminaron duplicados ni tags previos de Zotero para no revertir trabajo existente. Solo se agregaron items, carpetas y tags necesarios.
