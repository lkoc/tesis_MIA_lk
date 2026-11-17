# Bitacora de revision y cambios de `plan_v2`

**Actualizado:** 2026-11-17

**Estado general:** correccion de las observaciones del asesor del 13-nov-2026 aplicada en las fuentes. Las secciones historicas conservan el estado de sus fechas y no sustituyen este registro.
**Copia de trabajo:** `plan_v2/`.  
**Fuente original protegida:** `Plan/`; no editarla durante esta revision.

## Proposito y reglas de reanudacion

Este archivo conserva el estado de trabajo para retomar la revision despues de una interrupcion o perdida de contexto. Es un indice operativo, no sustituye los documentos fuente ni la evidencia.

## Cotejo final con el informe del asesor: 17 de noviembre de 2026

- [x] Se contrasto el informe de evaluacion del 13-nov-2026 con las fuentes vigentes, la matriz y el PDF. Dictamen documental: nueve observaciones atendidas y la numero 9 parcial, porque el presupuesto tiene importes y riesgos pero sigue pendiente de cotizaciones/tarifas verificadas. No se declara absolucion formal ni ejecucion experimental.
- Correccion puntual en `capitulos/05_administracion.tex`: aclarar que los documentos aprobados enumerados para H1 son evidencia exigida para su cierre, no un logro ya acreditado cuando el hito sigue en revision. Comprobacion prevista: control de esa fila y compilacion final.
- Verificacion el 17-nov-2026 a las 18:04 en Lima: LuaLaTeX/Biber generaron 85 paginas sin errores ni referencias/citas indefinidas y sin advertencias de Biber. Bibliografia: 50 claves unicas, 21 entradas tipo articulo; la referencia de lineas aereas no esta activa. La matriz externa rev2 paso `--check` en sus tres hojas sin ser reescrita. La fila H1 corregida se verifico en el PDF y visualmente, sin recortes ni solapamientos.
- Evidencia y pendientes: `comentarios_plan/Verificacion_final_observaciones_20261117.md`. Se distingue la verificacion interna de citas de una auditoria externa de DOI/indexacion, y el protocolo propuesto de resultados experimentales acreditados.
- PDF final de trabajo: SHA-256 `baa97227721ad14407759c90149bf9da85d6df62a1a695993ac352da9c530a89`. La version evaluada en SlopTotal se preserva en `old/plan_previo_cierre_asesor_20261117_9123017d.pdf` con su SHA-256 `9123017df6815b867c10145947f2d828b84fd7575cfda478c0c6e970f37644fc`. No se repitio SlopTotal para esta aclaracion de H1.

## Evaluacion SlopTotal del ultimo afinado IV/V: 17 de noviembre de 2026

- [x] Se evaluaron exclusivamente IV y V del PDF SHA-256 `9123017df6815b867c10145947f2d828b84fd7575cfda478c0c6e970f37644fc`, sin modificar el texto durante la ejecucion. Dos envios, dos identificadores nuevos respecto de la version editorial, 23 motores por capitulo y coincidencia pantalla/JSON.
- Capturas a las 22:44:42 y 22:45:57 UTC (17:44:42 y 17:45:57 en Lima): IV 33,9/100, `Low risk`, informe `a9e25cc2a4a0`; V 41,0/100, `Low risk`, informe `d12f5b6fb613`.
- Comparacion inmediata: IV 30,4 a 33,9 (+3,5 puntos); V 47,0 a 41,0 (-6,0). Se conserva el resultado mixto, sin atribuir causalidad a cambios concretos ni presentar las puntuaciones como prueba de autoria o calidad.
- Evidencia: `auditoria_IA/auditoria_resultados/afinamiento_IV_V_20261117/`. Se conservaron las ejecuciones anteriores, se verificaron las huellas y se detuvo el recolector al terminar. I--III, preliminares, referencias y anexos no se enviaron.

## Contraste y afinado puntual de IV y V: 17 de noviembre de 2026

- [x] El usuario solicita comparar los cambios de IV/V y tomar recursos expositivos de I--III. Se compararon las exportaciones completas anteriores y editoriales; no se atribuye causalmente el agregado del detector a una correccion concreta.
- En IV, TMR paso de 72,9 a 94,8, mientras Desklib paso de 35,9 a 3,7. En V, la heuristica estructural bajo de 45,3 a 41,0 y Burstiness de 61,2 a 53,6, aunque el agregado aumento. Estos desacuerdos no respaldan una explicacion simple basada en uniformidad o muletillas.
- Antes de editar se registra la version actual de `capitulos/04_metodologia.tex` y `capitulos/05_administracion.tex` y se protegen formulas, cifras, referencias, citas, figuras y tablas. En V se afinan apertura y prosa presupuestaria para referir acciones y recursos concretos; no se cambian importes ni hitos. Se conservaran los PDF evaluados y los resultados de ambas versiones.
- En IV se afinan la apertura, la clasificacion del estudio y la introduccion de DSR: explicacion de decisiones y comprobaciones en lugar de frases paralelas que solo vuelven a definir la clasificacion. Se conserva el protocolo; el funcionamiento del artefacto queda planteado como objeto de evaluacion, no como exito supuesto.
- Verificacion cerrada el 17-nov-2026 a las 17:34 en Lima: ocho filas del contraste de motores comprobadas contra los JSON; IV/V conservan formulas, cifras, referencias, citas y los bloques completos de figuras/tablas. I--III no se modificaron en esta ronda.
- Compilacion limpia en `%TEMP%/plan_tesis_afinamiento_IV_V_20261117/`: 85 paginas, sin errores ni citas/referencias indefinidas, sin advertencias de Biber. Se verificaron los pasajes nuevos y se inspeccionaron visualmente las paginas de apertura de IV y de presupuesto, sin recortes ni solapamientos. Se mantienen los dos desbordamientos bibliograficos previos inferiores a 0,2 pt.
- PDF de trabajo actualizado, SHA-256 `9123017df6815b867c10145947f2d828b84fd7575cfda478c0c6e970f37644fc`. La version editorial evaluada, SHA-256 `ce82f135ae0e05d1e5700c5a37355bbd26275b4537c7af390f9dc2b721d8a824`, se preserva en `old/plan_previo_afinamiento_IV_V_20261117_ce82f135.pdf`. Al cerrar el afinado aun no se habia enviado a SlopTotal; la evaluacion posterior consta en la seccion anterior.
- Analisis y fundamentos de los cambios: `auditoria_IA/auditoria_resultados/version_editorial_20261117/analisis_IV_V.md`. No se promete una puntuacion menor ni una percepcion garantizada de autoria humana.

## Reevaluacion de SlopTotal de la version editorial: 17 de noviembre de 2026

- [x] Se evaluaron exclusivamente I--V del PDF SHA-256 `ce82f135ae0e05d1e5700c5a37355bbd26275b4537c7af390f9dc2b721d8a824`, mediante cinco envios secuenciales sin alterar el documento. Se verificaron 23 motores por capitulo, coincidencia pantalla/JSON, identificadores y huellas de entrada.
- Capturas entre 21:51:52 y 22:00:59 UTC (16:51:52--17:00:59 en Lima): I 29,5; II 11,8; III 22,7; IV 30,4; V 47,0. Etiquetas: I--III `Clean - likely human-written`; IV `Low risk`; V `Suspicious`.
- Comparacion con los resultados completos anteriores: -7,3; -5,5; +6,4; +8,3; +19,7 puntos, respectivamente. No se interpreta la variacion como demostracion de autoria ni se atribuye a una causa editorial especifica. No se hicieron reenvios para buscar un valor mas bajo.
- Evidencia y comparativo: `auditoria_IA/auditoria_resultados/version_editorial_20261117/`. Se conservan las ejecuciones anteriores y las exportaciones originales. Recolector detenido al terminar; preliminares, referencias y anexos no enviados.

## Revision editorial puntual de los capitulos I a V: 17 de noviembre de 2026

- [x] Revision de texto autorizada expresamente despues de completar la auditoria de SlopTotal; el usuario incluye tambien II y III. Se siguieron las guias Markdown de redaccion y estructura. Los resultados del detector orientaron la lectura, no se usaron como objetivo de puntuacion.
- Primer ajuste, `capitulos/02_objetivos.tex`: dividir HE2 y separar el bloque de umbrales para mejorar legibilidad, sin cambiar criterios ni indicadores. Se registran antes de editar los cinco capitulos y sus formulas, cifras, referencias, citas, figuras y tablas para comprobar su conservacion.
- Siguiente ajuste, `capitulos/03_marco_teorico.tex`: aclarar el vocabulario de la PDE, reducir repeticiones y hacer concordar la explicacion de las entradas de la red y de la descomposicion por material con el protocolo ya fijado. Se atiende tambien la naturalidad solicitada: continuidad, conectores justificados y ritmo de lectura, sin prometer autoria humana ni buscar eludir detectores.
- Ajuste de `capitulos/01_planteamiento_del_problema.tex`: acotar la afirmacion sobre FEM, distinguir el contraste descriptivo de demanda/longitud de red de un diagnostico de congestion y completar la referencia discursiva al Anexo de matriz. No se modifican las cifras de contexto ni PG/PE.
- Ajuste de `capitulos/04_metodologia.tex`: separar justificaciones del nivel de investigacion, reducir transiciones redundantes, ordenar la explicacion de las metricas en parrafos y cerrar el capitulo. Se conservan completos el protocolo, las tolerancias, las semillas y las reglas de decision.
- Ajuste de `capitulos/05_administracion.tex`: separar el corte del cronograma de la programacion futura y precisar en prosa la naturaleza estimada del presupuesto y su financiamiento. Las tablas, los importes y los hitos permanecen sin cambios.
- La revision se limita a fuentes de `plan_v2/` y a su PDF de trabajo. Se preservan matrices, referencias bibliograficas, entregas historicas y las entradas/resultados de la auditoria; estos ultimos describen el PDF anterior y no se atribuiran a la version editada.
- Verificacion cerrada el 17-nov-2026 a las 16:32 en Lima: los cinco capitulos conservan, frente al texto anterior a esta revision, las secuencias de formulas, cifras, referencias y citas, asi como los bloques completos de figuras y tablas. Se corrigio una ambiguedad del comprobador de matematicas en linea usando siempre la copia previa a la edicion como referencia; no se alteraron formulas para satisfacer el control.
- LuaLaTeX/Biber compilaron 85 paginas en `%TEMP%/plan_tesis_estilo_20261117/`, sin errores ni citas/referencias indefinidas y sin advertencias de Biber. Persisten los dos desbordamientos bibliograficos previos, ambos inferiores a 0,2 pt. Se verifico la presencia de pasajes nuevos en el PDF y se inspeccionaron visualmente HE2, el cierre de III y la pagina de hitos/presupuesto, sin recortes ni solapamientos.
- PDF de trabajo actualizado: `plan_tesis_cables_pinn.pdf`, SHA-256 `ce82f135ae0e05d1e5700c5a37355bbd26275b4537c7af390f9dc2b721d8a824`. El PDF anterior, usado en SlopTotal, se preserva en `old/plan_previo_revision_editorial_20261117_3f9d80b9.pdf`, con su SHA-256 original `3f9d80b9baee06a76f1d9cffdf4d3abe0315440ec515d1acfdc7eee4f97d6c62`.
- La revision de naturalidad es editorial: se atendieron conectores sin funcion, repeticiones prescindibles, transiciones y extension de oraciones segun el contenido. No acredita autoria humana, no garantiza la percepcion de cualquier lector y no introduce variacion artificial. Al cierre editorial esta version aun no se habia vuelto a evaluar; la evaluacion posterior consta en la seccion anterior. No se modificaron sus resultados historicos.

## Aclaracion de la revision del capitulo I: 17 de noviembre de 2026

- El usuario aclara que solicita un nuevo analisis en SlopTotal, no una revision editorial. Se detiene la correccion academica iniciada por interpretacion equivocada de la solicitud.
- Se restaura exclusivamente el parrafo de la introduccion sobre FEM que se acababa de modificar. Las correcciones anteriores del plan y las evidencias del primer analisis se conservan. Se comprobara la coincidencia del texto extraido y de sus huellas antes del nuevo envio.
- El nuevo analisis se registrara por separado; no se sustituira el resultado anterior ni se enviaran otros capitulos.
- [x] Verificado: se restauro el unico parrafo modificado por error y se comprobaron las huellas del PDF, la entrada y el texto normalizado. Se realizo un unico nuevo envio del capitulo I; el portal devolvio el informe existente `3de8ee861873`, no un ticket nuevo.
- Captura del 17-nov-2026 a las 19:12:19 UTC (14:12:19 en Lima): pantalla y JSON coinciden en 36,8/100, `Low risk`, con 23 motores puntuados. Se archivan DOM y exportacion original en `auditoria_IA/auditoria_resultados/reanalisis_capitulo_1_20261117/`. Es una segunda consulta de un informe reutilizado, no un recalculo independiente acreditado; las capturas anteriores se conservan. Recolector detenido al terminar; no se recompilo ni se altero el PDF.

## Revision de observaciones del asesor: 17 de noviembre de 2026

Solicitud vigente: corregir el plan con el informe del asesor y el resumen de aplicabilidad de `comentarios_plan/`. Se conservan las entregas historicas y no se modifica codigo de entrenamiento, la matriz XLSX externa ni resultados de campanas.

| Observacion | Aplicacion en la fuente |
|---|---|
| 1. Umbrales | Capitulos 2 y 4 y anexo 1: F2 E_T <0,1 K y NRMSE <1 %; F3 E_T <0,5 K y NRMSE <2,5 %; sin puerta alternativa del 5 %. Error frente a referencia separado del efecto pareado. |
| 2. Perdidas y ampacidad | Capitulo 4: resistencia DC o AC efectiva excluyentes, fuente uniforme, exclusiones, acoplamiento, biseccion, tolerancias e intervalos. Capitulo 1: limites concordantes. |
| 3. Entrenamiento | Capitulos 3 y 4: una red por subdominio, tabla de configuracion, escalas, pesos, piloto, semillas nuevas, parada fisica y w_d=0. Protocolo propuesto, no ejecucion acreditada. |
| 4. Homogenizacion | Capitulos 2 y 4 y matriz: media ponderada por area solo del entorno; datos conservados en el par; hipotesis refutable. |
| 5. FEM | Capitulo 4: elementos, solucionador, refinamiento, tolerancias, verificacion exacta y estimacion de error de referencia separada del error PINN. |
| 6. Relevancia | Capitulos 2 y 4 y matriz: cinco semillas, envolventes conservadoras, delta_T=1 K, delta_I=2 %, fallos e inconclusos. |
| 7. Interpretacion externa | Capitulos 1 y 4: sin campo ni certificacion operacional, limites de geometria, contornos y perdidas. |
| 8. Antecedentes | Capitulo 1: tabla critica metodo/geometria/limite/decision; retirado el uso de la referencia de lineas aereas para sustentar cables enterrados. No se agregan fuentes nuevas. |
| 9. Viabilidad | Capitulo 5: estimacion de planificacion de S/ 3528 (2940 +20 %), supuestos visibles, no cotizaciones ni tarifas verificadas; riesgos y contingencias. |
| 10. Cronograma | Capitulo 5: corte 17-nov-2026, campanas historicas exploratorias, H1-H6 con evidencia y reprogramacion; no se declaran aceptaciones sin prueba. |

Verificacion: compilacion limpia con LuaLaTeX/Biber de 85 paginas, sin citas ni referencias indefinidas y sin advertencias de Biber. Salida final aislada: `%TEMP%/plan_tesis_revision_20261117_final/`; la primera salida temporal tuvo un auxiliar truncado y se descarto. Inspeccion visual de tablas de entrenamiento, casos, criterios, presupuesto y matriz: sin recortes ni solapamientos. Se corrige tambien la duplicacion de la palabra ANEXO en referencias cruzadas y se da mayor separacion a columnas de la matriz. Dos desbordamientos de URL bibliograficas inferiores a 0,2 pt ya estaban presentes. El presupuesto exige contrastar importes antes del gasto; no se obtuvieron cotizaciones en esta revision. La matriz integrada en LaTeX queda actualizada; la matriz XLSX externa no se sincroniza en esta tarea. No se certifica una auditoria externa de DOI ni la validacion de campo del prototipo.

Al reanudar:

1. Leer primero este log y comprobar las rutas y estados contra el workspace actual. No asumir que una tarea marcada o descrita aqui sigue igual.
2. Releer completos los documentos rectores antes de editar: `Plan/guias/guia_redaccion_plan_tesis.md`, `Plan/guias/guia_estructura_y_contenido_plan_tesis.md`, `Plan/guias/GUÍA N°01 Plan.pdf`, `matriz_consistencia/GUÍA N°01 Plan.pdf` y `matriz_consistencia/rubrica_plan_tesis.pdf`. Comparar las copias de las guias dentro de `plan_v2/guias/` si existen.
3. Releer la matriz vigente `matriz_consistencia/Matriz_de_Consistencia_Tesis_Cables_Heterogeneidad_Termica_rev1.xlsx` y su PDF, asi como `plan_v2/capitulos/anexos/anexo_1_matriz_consistencia.tex`. Confirmar version, hojas y celdas; no trasladar cifras de memoria ni de resúmenes sin verificarlas.
4. Revisar el archivo principal `plan_v2/plan_tesis_cables_pinn.tex`, los capitulos 00--06, sus anexos, `plan_v2/referencias_plan_tesis.bib` y las fuentes/documentos que sostienen afirmaciones relevantes. Mantener coherencia entre cuerpo, matriz, figuras, tablas, bibliografia y anexos.
5. Para evidencia de modelos, revisar las configuraciones, salidas JSON/logs, checkpoints y referencias de las campañas actuales de `pinnV3/`. La memoria `pinnV3_plan.md` tiene entradas de octubre que ya estan desactualizadas; contrastarla con el codigo y los resultados actuales.
6. Antes de cada edicion, registrar aqui el archivo, el objetivo y la discrepancia que se va a resolver. Despues, registrar fecha/hora local, cambio concreto, evidencia de verificacion y pendientes. Usar estados `[ ] pendiente`, `[~] en curso`, `[x] hecho y verificado`; no marcar `[x]` solo porque el texto fue editado.
7. Conservar `Plan/` y los artefactos historicos de `plan_v2/envío_rev0/`, `plan_v2/envío_rev1/`, los PDF de revision y `rev2_SHA256.txt`. No sobrescribir ni regenerar entregables historicos durante la edicion de fuentes.

## Fuentes de verdad

- **Normas de escritura y estructura:** las dos guias Markdown en `Plan/guias/` y sus copias en `plan_v2/guias/`.
- **Formato institucional:** `GUÍA N°01 Plan.pdf` en `Plan/guias/` y en `matriz_consistencia/`.
- **Evaluacion del parcial:** `matriz_consistencia/rubrica_plan_tesis.pdf` (3 paginas sin capa de texto; texto completo verificado el 2026-11-07 con OCR de Windows; ver la seccion de hallazgos).
- **Contrato de consistencia:** matriz Excel rev1 y su PDF en `matriz_consistencia/`; comprobar si hay una revision posterior antes de cada ronda sustantiva.
- **Representacion incluida en la tesis:** `plan_v2/capitulos/anexos/anexo_1_matriz_consistencia.tex`, que debe quedar sincronizada con la matriz fuente.
- **Plan a revisar:** `plan_v2/plan_tesis_cables_pinn.tex`, `plan_v2/capitulos/00_datos_generales.tex` y `01_planteamiento_del_problema.tex` a `06_anexos.tex`.
- **Fuentes cientificas:** `plan_v2/referencias_plan_tesis.bib`, `plan_v2/literatura/` y las fuentes primarias citadas. Verificar los datos bibliograficos y los pasajes antes de ampliar antecedentes.
- **Evidencia computacional:** codigo, manifests, informes, logs y resultados de `pinnV3/`; usar `PINN_v2/` solo si hace falta una comparacion historica claramente identificada.

## Estado comprobado al crear este log

- `plan_v2/` existe y contiene capitulos, anexos, guias, datos fuente, literatura, bibliografia y artefactos de revision/envio. Incluye PDF de rev2 y `rev2_SHA256.txt`; preservarlos como historicos.
- La busqueda de archivos confirmo que `plan_v2/log_cambios.md` no existia antes de esta creacion. En esta continuacion no se editaron capitulos ni anexos; este log es el unico archivo nuevo.
- El plan tiene la arquitectura de cinco capitulos, referencias y anexos. La conformidad exacta con las guias UNI debe revisarse visualmente y por seccion; la mera presencia de la estructura no basta.
- El capitulo 2 conserva umbrales generales HE2 de error de Tmax y NRMSE <= 5 %, desequilibrio energetico <= 2 %, y una regla de variabilidad entre semillas que depende de un efecto relevante por fijar. Tambien dice que reproducibilidad se evaluara transversalmente, aunque ningun OE la trata como producto especifico.
- El anexo de matriz incluido en `plan_v2` contiene criterios por etapa que no coinciden con todos los del capitulo 2: F1 reporta cobertura de requisitos criticos >90 %; F2 reporta `|Delta Tmax| < 0,1 K` y NRMSE <1 %; F3 reporta `|Delta Tmax| < 0,5 K` y NRMSE <2,5 %; F4 incluye ampacidad, cambio porcentual, margen termico y juicio experto de calidad de optimizacion en escala 1--10. Estos valores se observaron en el anexo; deben cotejarse celda por celda con el Excel/PDF rev1 antes de armonizar.
- El capitulo 4 presenta formulas para error relativo de Tmax y cambio porcentual de ampacidad, pero no operacionaliza todas las metricas mencionadas (por ejemplo NRMSE y balance energetico), ni desarrolla el juicio de expertos.
- El usuario aclaro que el juicio de expertos debe evaluar, tan objetivamente como sea posible, las soluciones y propuestas de calculo termico para cables enterrados, comparandolas con soluciones convencionales. El protocolo debe elaborarse antes de aplicarlo y definir criterios, evaluadores, procedimiento, puntuacion y regla de decision. La evaluacion experta complementa la verificacion cuantitativa; no la sustituye. No hay puntuaciones expertas realizadas que se puedan reportar.
- El capitulo 4 contiene un parrafo titulado `Precisamiento de la formulacion termica y electrica`, incluye efecto piel/corriente AC y fuente electromagnetica; debe revisarse frente al alcance y a las limitaciones que describen perdidas Joule `I^2 R`. No resolver esa tension silenciosamente.
- El presupuesto del capitulo 5 deja varios montos por confirmar y no presenta total consolidado. El cronograma marca como cumplidas fases iniciales y refiere campañas A--D; contrastar contra evidencia antes de afirmar viabilidad.
- El inventario de `pinnV3/results/` confirma campañas largas del 2 y 5 de noviembre de 2026, una mini-campana del 6 de noviembre y smokes del 6 y 7 de noviembre. Los nombres/fechas fueron localizados y, el 2026-11-07, sus JSON/logs fueron auditados (ver la seccion de hallazgos, E). No declarar ganador ni aceptacion de tesis por el nombre, fecha o campo `accepted`; verificar configuracion, duracion, completitud, referencia y criterios de calificacion.
- Hallazgos menores reportados en la revision previa, pendientes de confirmar al editar: redundancia entre introduccion y problematica; concordancia singular/plural en una frase del diagnostico; simbolos de porcentaje omitidos en dos frases; antecedente `aldulaimi2024` citado en el cuerpo pero no incluido en la tabla cronologica.

## Hallazgos de la sesion 2026-11-07

Alcance y cambios: se leyeron completas las dos guias de escritura, los capitulos 00 a 06 y el anexo 1; se verificaron copias de guias, rubrica, matriz rev1, hash rev2 y el ultimo log de compilacion; se inventariaron las 11 carpetas de `pinnV3/results/` de noviembre; se ejecutaron comprobaciones automaticas sobre el `.tex`. Cambios realizados en esta sesion: solo este log. No se modificaron `Plan/`, `plan_v2/capitulos/`, anexos, matriz ni PDF historicos. Archivos temporales fuera del workspace: `%TEMP%/ths_audit/` y `%TEMP%/ths_audit_*.py`; pueden borrarse.

### A. Fuentes y control de versiones

- [x] Guias de escritura: leidas completas. Las copias en `plan_v2/guias/` son identicas (SHA-256) a las de `Plan/guias/`. Reglas que aplican a la auditoria: tercera persona; no llamar mediciones a calculos; oraciones de hasta 45 palabras (tolerancia 5); sin adjetivos valorativos sin criterio; apertura y cierre por capitulo; fuente en cada figura; citas verificables; ningun estilo en el .bib.
- [x] Rubrica (`matriz_consistencia/rubrica_plan_tesis.pdf`, 3 paginas sin capa de texto): texto completo obtenido con OCR de Windows (es-MX). Pregunta unica de 20 puntos sobre el plan y cinco criterios de 4 puntos: (1) planteamiento, diagnostico y justificacion; (2) formulacion y coherencia de objetivos; (3) fundamentos teoricos y antecedentes; (4) metodologia y desarrollo de la solucion; (5) evaluacion, viabilidad y contribucion. Niveles: Excelente, Bueno, En desarrollo, Insuficiente.
- [~] Guia N°01 Plan.pdf (60 paginas): sin texto extraible. El OCR fue ilegible en las paginas 1, 2 y 10 a 12 con ambas rotaciones. Las miniaturas rotadas muestran un indice con estructuras para maestria de investigacion y para maestria de especializacion o profesionalizante, y los anexos I a V; no se pudo leer pagina por pagina (seccion I).
- [~] Matriz rev1: el XLSX (2026-10-30, 20.049 B) tiene la hoja visible `Matriz_de _Consistencia` (con espacio en el nombre) y dos hojas ocultas: `Operacionalización` y `Diseño de casos`. El PDF rev1 (2026-11-06) tiene una pagina; su texto coincide con la hoja visible, pero no aparece la columna Fase DSR / actividad (pendiente comprobar visualmente). Los archivos `..._Termica.xlsx` (2026-09-25) y `rev1 - Copy.xlsx` (2026-10-03) no se usaron como fuente.
- [x] Hash rev2: `Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev2.pdf` coincide con `rev2_SHA256.txt` (6f7c7e9b...). Sin cambios.
- [~] Ultima compilacion (`plan_v2/plan_tesis_cables_pinn.log`, 2026-11-06 20:40): Output written on plan_tesis_cables_pinn.pdf (67 pages); sin advertencias de referencias ni citas indefinidas. La rev2 tenia 76 paginas; la diferencia de 9 paginas debe explicarse antes de la Fase 10. El PDF de esa compilacion no aparece en `plan_v2/`.

### B. Correccion de un hallazgo previo

- El capitulo 1 si contiene la formulacion del problema (1.2.2: problema general y problemas especificos PE1 a PE4 con figura) y la justificacion y alcances (1.2.3: justificacion practica, tecnologica, metodologica y cientifica; alcances; limitaciones). La lectura previa se corto en la linea 400 y no debe usarse como evidencia.
- La estructura coincide con el resumen de la guia en `guia_estructura_y_contenido_plan_tesis.md` (seccion 3.3). Diferencia de nombre: el plan titula la seccion 1.1 INTRODUCCION y la guia (resumen) la llama Diagnostico. Confirmar con la guia impresa.
- Los textos de PE1 a PE4 (capitulo 1) y de OE1 a OE4 (capitulo 2) coinciden en lo sustantivo con la matriz rev1.

### C. Comprobaciones automaticas del `.tex`

- Etiquetas: 62. Referencias: 47. Referencias sin etiqueta: 0. Etiquetas duplicadas: 0. Etiquetas sin referencia: 15 (no es error).
- Citas: 49 claves citadas; 49 entradas en `referencias_plan_tesis.bib`; ninguna falta ni sobra. Hay 23 entradas en `referencias_plan_tesis_no_usadas.bib`. El .bib no contiene referencias de estilo (Lovei, Scholz, Van Emden): cumple la guia.
- Citas del capitulo 1 fuera de la tabla cronologica (que se presenta como antecedentes de 19 fuentes): aldulaimi2024, ariasVelasquez2023, cigre2022, cigre2023overheadLines, claves coes* y minem*, coesEstadisticasAnuales, coesMaximaDemanda, fariz2026, iea2025transmissiongrid, iec60502_2_2014, ieee442, mobius2025, olade2025generacionALC2024, osinergmin2024puntaLomitas, proinversion2026palian y thue1999electrical. Definir el criterio de la tabla o ampliarla.
- Terminos y estilo:
  - en esta tesis: 1 vez (capitulo 3, linea 403). No hay primera ni segunda persona.
  - importante: 2 veces (capitulo 3, lineas 173 y 314).
  - medida o medicion: 5 ocurrencias. Cuatro son correctas (negaciones o requisitos de validacion). En el capitulo 1, linea 205, resistividad medida de 4,9 atribuye una medicion a una fuente: reemplazar por reportada.
  - requieren construir: sujeto singular (una descripcion ... requieren) en el capitulo 1; corregir a requiere.
  - 19 a generacion eolica y solar y 11 a recursos no convencionales (capitulo 1, problematica): falta el simbolo de porcentaje.
  - Precisamiento (capitulo 4, titulo del parrafo de nivel 4): error de palabra; probable Precisiones.
  - Oraciones de prosa de mas de 50 palabras (heuristico): HE2 (63 palabras) y HE3 (57), capitulo 2.
- Simbolos o terminos sin definicion: k_eq (capitulo 1, figura de brecha, y capitulo 4, procedimiento A1); Omega_c (capitulo 4, formula de ampacidad); variantes PINN y anillos (capitulo 4, parrafo Precisamiento); margen termico (sin definicion operativa).
- Juicio de expertos: aparece solo en la matriz (anexo, F4: calidad de la optimizacion por juicio de expertos, rango 1-10). Los capitulos 2 y 4 no contienen protocolo.
- Balance energetico y NRMSE: aparecen en tablas e hipotesis, pero el capitulo 4 no los define; la seccion de metricas menciona RMSE, MAE y balance sin formula.
- Alcance electromagnetico frente a I^2 R: el capitulo 1 (limitaciones) dice que no se resolvera la interaccion electromagnetica completa y que las perdidas se representan como fuente de Joule proporcional a I^2 R. El parrafo Precisamiento del capitulo 4 indica que se estimara el efecto piel a 60 Hz y que se contrastara una distribucion electromagnetica. Ambos textos requieren aclaracion y el parrafo del capitulo 4 no tiene citas.

### D. Umbrales en conflicto (requiere decision)

| Fuente | NRMSE | Error de Tmax | Balance energetico | Observacion |
|---|---|---|---|---|
| Capitulo 2, HE2 | <= 5 % | error relativo <= 5 % | <= 2 % | incluye la frase de revisar umbrales en la fase piloto |
| Capitulo 4, tabla de criterios | <= 5 % | e_Tmax <= 5 % | <= 2 % | misma regla que HE2 |
| Matriz rev1, F2 | < 1 % | Delta Tmax < 0,1 K (absoluto) | no indicado | |
| Matriz rev1, F3 | < 2,5 % | Delta Tmax < 0,5 K (absoluto) | no indicado | |
| pinnV3, parada por precision (`accuracy_stop`) | 5 % | Delta T por caso: 5,94 K (cable_disk), 0,84 K (xlpe_single) | 2 % | usa la referencia FEM para parar |
| pinnV3, campo `accepted` (`pinn_v3/evaluation/metrics.py`) | 10 % | Tmax_rise_pct 10 % | no incluido | genera el campo accepted |
| pinnV3, documentos del 27-oct-2026 (`FINAL_CAMPAIGN`, `CONVERGENCE_FINDINGS`) | < 0,25 % | < 0,05 K | < 0,5 % | calificacion vigente segun esos documentos |

Hay al menos cinco juegos de umbrales distintos. Ademas, `accepted` no incluye el balance y usa 10 %, por lo que una corrida marcada como aceptada puede no cumplir los umbrales del plan.

### E. Inventario de campañas pinnV3 de noviembre

Todos los resumenes con `summary.json` tienen smoke_only = true y official_qualification = false.

| # | Carpeta (`pinnV3/results/`) | Escritura | Adam + L-BFGS | Corridas registradas / planificadas | Hallazgos | Clasificacion |
|---|---|---|---|---|---|---|
| 1 | pinn_campaign_8000adam_ext8000x2_400lbfgs_all_mms_cables_20261102 (+ logs de 03 y 05) | 02-nov 08:27 a 05-nov 20:13 | 8000 + 400 (extensible a 24000) | 222 / 729 (218 completadas, 4 fallidas) | 27 casos planificados, 9 con corridas (8 MMS y aras_flat); 164 con accepted=True, de ellas 35 incumplen balance > 2 % o NRMSE > 5 %; M2 acepta 6 de 25; aras_flat 0 de 6 | Parcial (30 %) |
| 2 | pinn_campaign_8000adam_..._M1_M1EXACT_fem_none_20261105 | 05-nov | n.d. | 0 / n.d. | carpeta sin archivos | Sin datos |
| 3 | pinn_campaign_8000adam_..._M1_M1EXACT_fem_none_CABLES_20261105 | 05-nov | 8000 + 400 | 1 / 228 | unica corrida: aras_flat M1 s0, NRMSE 13,7 %, Delta T 21,2 K, presupuesto agotado | Interrumpida (0,4 %) |
| 4 | pinn_campaign_mini_400adam_ext400x2_50lbfgs_..._CABLES_20261106 | 06-nov | 400 + 50 | 23 / 76 | 4 corridas con lbfgs_non_finite; M1_EXACT con balance mediano 99,8 %; 4 accepted, 1 incumple | Smoke parcial (30 %) |
| 5 | m1exact_mms_cycle_smoke_20261102 | 02-nov | 2 + 1 | 1 / n.d. | prueba tecnica | Smoke |
| 6 | pilot_smoke_50adam_5lbfgs_20261106 | 06-nov | 50 + 5 | 3 / 4 | sin aceptacion | Smoke |
| 7 | smoke_all_networks_50x5_20261106_pilot | 06-nov | 50 + 5 | 12 / 12 | 8 completadas; 4 fallidas (DeepXDE) | Smoke |
| 8 | smoke_all_networks_50x5_mms_cables_annulus_slab_20261106_partial_energy100 | 06-nov | 50 + 5 | 43 / 952 | 37 completadas; 6 fallidas; 2 accepted, ambas incumplen | Smoke parcial (4,5 %) |
| 9 | smoke_all_networks_50x5_pde_bc_20261106_pilot | 06-nov | 50 + 5 | 2 / 2 | piloto | Smoke |
| 10 | smoke_all_networks_50x5_pde_bc_mms_cables_annulus_slab_20261106 | log 06-nov 15:51; resultados hasta 07-nov 10:03 | 50 + 5 | 952 / 952 | 730 completadas, 114 fallidas, 108 omitidas por duracion; 128 accepted, 112 incumplen; M1_EXACT_BC acepta 23 de 56 | Smoke completa en su registro, con fallos y omisiones |
| 11 | smoke_rings_20261107 | 06-nov 14:27 a 14:29 | 400 + 25 | 2 / 2 | M1: NRMSE 27,0 %, Delta T 118 K, balance 1,6 %; M1_EXACT: balance 98,9 % | Smoke |

Hallazgos del inventario:

- Ninguna corrida de noviembre tiene calificacion oficial. En las campañas largas, los casos de cable se limitan a aras_flat: 6 corridas en la fila 1 y 1 en la fila 3; ninguna se acepta. La fila 1 planifico 27 casos, pero solo 9 tienen corridas registradas.
- Balance anomalo de M1_EXACT en cables: 98,9 % (fila 11), 99,8 % mediano (fila 4) y 96,5 % mediano en 56 corridas (fila 10). Antes de usar la variante para seleccion, verificar el calculo de balance y la implementacion.
- La parada por precision usa la referencia FEM (reference_used_for_stopping = true en la fila 11) y detiene las corridas entre 55 y 24.399 pasos. Por eso las comparaciones entre arquitecturas no tienen el mismo presupuesto, y las metricas finales no son independientes del criterio de parada.
- Errores registrados: IndexError en 54 corridas (fila 10); ImportError de deepxde_solver y NotImplementedError de RBF-FD en DeepXDE (filas 7, 8 y 10); RuntimeError de forma (200x3 frente a 6x12 o 7x12): 4 corridas en la fila 8 y 4 en la fila 10; ValueError FEM readout requires one output-layer evaluation per region en 3 corridas M1_EXACT de mms_robin (fila 1); PermissionError de Windows al escribir resultados (fila 1).
- Campañas A-D del cronograma: las carpetas adaptive_all_cases_20261026, 20261026b, 20261026c y 20261026d (26 y 27 de octubre) y adaptive_all_cases_20261027_final (27 y 28 de octubre) son candidatas, no confirmadas. Las filas de `pinnV3/CAMPAIGN_CLASSIFICATION_2026-10-27.md` muestran requires_more_steps. Si corresponden, las fechas son de octubre, no de septiembre como indica la tabla de cronograma.

### F. Evaluacion frente a la rubrica

| Criterio | Exigencia (resumen) | Estado en el plan | Brecha principal |
|---|---|---|---|
| 1. Planteamiento, justificacion | problema real, relevancia, antecedentes, justificacion y alcance | presente: problematica, tabla de antecedentes, PG y PE1 a PE4, justificacion en cuatro ejes, alcances y limitaciones | cifras de contexto sin verificar contra datos_fuente; nombre INTRODUCCION frente a Diagnostico |
| 2. Objetivos | claros, medibles, coherentes | OG y OE1 a OE4 coherentes con PE1 a PE4 | OE sin indicadores ni umbrales en el capitulo 2; OG incluye reproducibilidad sin producto; umbrales en conflicto (seccion D) |
| 3. Fundamentos y antecedentes | pertinentes, solidos y relacionados con la propuesta | capitulo 3 con bases fisicas, PINN, DSR, definiciones y tabla de uso | la frase de posicionamiento como mejora de metodos PINN no esta sustentada; paginas de neher1957 inconsistentes (752-760 frente a 752-772) |
| 4. Metodologia | coherente y viable; tecnicas, datos, herramientas y etapas | procedimiento DSR, fases, criterios, diseno discreto de casos | sin herramientas nombradas (el texto las omite); NRMSE, balance y margen sin definicion; juicio de expertos sin protocolo; nomenclatura de fases distinta de la matriz; parrafo de fisica mal ubicado y sin citas |
| 5. Evaluacion, viabilidad, contribucion | indicadores, metricas o criterios; viabilidad; resultados esperados | metricas, criterios, cronograma, presupuesto y principios de diseno como proposiciones | umbrales sin unificar; presupuesto sin montos; cronograma anterior a la fecha actual; evidencia computacional de noviembre solo de smoke o parcial |

### G. Recomendaciones (ninguna aplicada)

1. Fijar un unico conjunto de umbrales (decision D1) y registrarlo antes del piloto; usar la misma regla en la matriz, en los capitulos 2 y 4 y en el campo accepted de pinnV3.
2. En pinnV3: que accepted incluya el balance y los umbrales elegidos; separar la referencia de parada de la referencia de verificacion; igualar el presupuesto de pasos entre arquitecturas en las comparaciones.
3. Investigar el balance cercano a 100 % de M1_EXACT en cables y los IndexError, RuntimeError y ValueError antes de otra campaña.
4. Capitulo 2: reescribir HE2 y HE3 con umbrales unicos, quitar la frase que pospone la revision de umbrales a la fase piloto y dividir las oraciones de mas de 50 palabras; asignar indicador y umbral a cada OE; decidir el destino de la reproducibilidad en el OG.
5. Capitulo 4: nombrar herramientas y versiones (PyTorch, FEniCSx y Gmsh, y WSL si se usan) y el hardware; definir NRMSE, balance, margen termico y Delta I; definir e_Tmax en kelvin o como elevacion sobre la temperatura ambiente (con grados Celsius el error relativo es inestable); definir k_eq y Omega_c; mover el parrafo Precisamiento a Modelo fisico, corregir el titulo, anadir citas y resolver el alcance electromagnetico frente a I^2 R; unificar la nomenclatura de fases con la matriz (F0 a F4); redactar el protocolo de juicio de expertos (evaluadores, criterios, anclajes 1 a 10 y agregacion) o retirarlo de la matriz.
6. Capitulo 1: corregir requieren a requiere; anadir el simbolo de porcentaje en 19 y 11; cambiar resistividad medida por reportada; revisar la redundancia entre la introduccion y la identificacion del problema (la triada problema real, subyacente y tecnico aparece dos veces); aclarar la frase sobre las condiciones reales de temperatura del terreno en el caso de Arias Velasquez.
7. Capitulo 3: reformular el posicionamiento como mejora de metodos PINN en una afirmacion delimitada; revisar los dos usos de importante; reducir el catalogo de definiciones a los terminos que usan el modelo o la metodologia; unificar las paginas de neher1957.
8. Capitulo 5: actualizar el cronograma a la fecha actual (la ronda E y el hito H3 de octubre ya pasaron); sustentar o corregir cumplido a septiembre con evidencia; presentar montos o justificar que los rubros no son monetarios; no afirmar viabilidad con las corridas de noviembre, que son smoke o parciales.
9. Anexo 1: corregir la celda HE1 (texto suelto y parentesis sin abrir); 5,% en HE2 (matriz y anexo); cuadratico y limite sin tilde; aceptción y (mperios) en el XLSX; sinteticos en F2 del anexo frente al XLSX; sincronizar umbrales con D1; decidir si Operacionalización y Diseño de casos pasan a ANEXO 2 y 3 (la guia las recomienda) o se eliminan, porque la hoja oculta cita anx:diseno-casos, etiqueta que no existe en el .tex.
10. Matriz: revisar el nombre de la hoja (`Matriz_de _Consistencia`) y comprobar la columna de fases en el PDF rev1; regenerar el PDF solo despues de corregir el XLSX.
11. Fase 10: explicar la diferencia de paginas (76 frente a 67) y la ubicacion del PDF de compilacion.

### H. Decisiones pendientes del usuario

- D1. Umbrales unicos (seccion D): fuente de verdad.
- D2. Reproducibilidad en el objetivo general: cerrar, reformular o convertir en producto.
- D3. Juicio de expertos: incluir protocolo o retirar de la matriz.
- D4. Confirmar si las campañas A-D son las adaptativas de 26 y 27 de octubre, y corregir el cronograma.
- D5. Alcance electromagnetico (I^2 R, efecto piel): escoger una formulacion y reflejarla en los capitulos 1 y 4.
- D6. Anexos 2 y 3 (hojas ocultas) y las 23 entradas del .bib no usadas.
- D7. Formato de entrega: la memoria del usuario indica entrega en .docx; el plan se compila en LaTeX/PDF.
- D8. Autorizacion para editar capitulos (Fases 2 a 9) despues de resolver D1 a D5.

### H2. Resoluciones del usuario (2026-11-07) y criterio de edicion

- D1 (umbrales): fuente de verdad, la ultima version de la matriz de consistencia (rev1; XLSX y PDF con el mismo contenido). Valores usados en el texto: HE2 (error de Tmax y NRMSE no mayores a 5 %; desequilibrio no mayor a 2 %), F2 (|Delta Tmax| menor que 0,1 K; NRMSE menor que 1 %) y F3 (|Delta Tmax| menor que 0,5 K; NRMSE menor que 2,5 %). En las campanas de calculo, el informe declara el criterio de parada y de aceptacion que aplico cada campana. La ultima corrida (smoke_rings_20261107) usa parada con NRMSE 5 %, balance 2 % y Delta T por caso, y aceptacion con NRMSE 10 % y Tmax_rise 10 %; cuando difiere de la matriz, el texto lo indica. Pendiente fuera de plan_v2: sincronizar el campo accepted de pinnV3 y precisar en la matriz que Delta Tmax es valor absoluto.
- D2 (reproducibilidad): el objetivo general se reformula. La reproducibilidad se sustenta en un repositorio publico con el codigo fuente, los datos empleados y sus fuentes. Pendiente: la misma redaccion en la matriz XLSX (no se edita en esta pasada).
- D3 (expertos): se mantiene el indicador F4 en la matriz y se redacta un protocolo preliminar como ANEXO 2 (borrador; sin umbral definitivo; no validado).
- D4 (campanas): A-D se identifican con las carpetas adaptive_all_cases_20261026, 20261026b, 20261026c y 20261026d (sufijos a-d); la adaptativa final es adaptive_all_cases_20261027_final. Extension registrada: A 39 corridas, 4 casos, 4 arquitecturas, 3 semillas; B 38, 4, 4, 3; C 40, 4, 4, 3; D 42, 4, 4, 3; final 22 corridas, 2 casos, 5 arquitecturas, 3 semillas. La configuracion es comun en A-D; la correspondencia de letras con los sufijos es una suposicion a confirmar. Las campañas del 25 de octubre (long_v2) y la selection_campaign_1200_50 no tienen resumen con el formato esperado y no se cuentan. El cronograma pasa la fase 3 a octubre de 2026 y su extension a noviembre.
- D5 (electromagnetismo): formulacion minima. Perdidas Joule I^2 R(T), con la resistencia referida a 20 C. Las ecuaciones de Maxwell no se resuelven. El efecto piel y el de proximidad solo entran por la resistencia en corriente alterna del caso. Aplicar en capitulos 1 y 4.
- D6 (anexos y bibliografia): las hojas ocultas de la matriz no se usan como anexos. Las 23 entradas no citadas se conservan en el archivo .bib no usado; el listado de referencias solo incluye lo citado.
- D7 (formato): LaTeX compilado a PDF; no se entrega en .docx.
- D8 (edicion): autorizada. Se edita por capitulo, con registro antes y despues en la bitacora.

### I. Verificaciones pendientes

- Guia N°01 pagina por pagina a resolucion completa: estructura de especializacion (capitulos 1 a 5), requisitos de presupuesto y financiamiento, formato (Anexo III), anexos requeridos y limite de paginas.
- Matriz PDF rev1: comprobar visualmente la columna de fases.
- Cifras del capitulo 1 (MINEM, COES, OLADE, Punta Lomitas, Palian): contrastar con `plan_v2/datos_fuente/` (extractos_texto, extractos_figuras, descargas).
- Fuentes primarias: paginas y valores de khumalo2025, ariasVelasquez2023, kim2025, atoccsa2024, neher1957 y fariz2026 (paginas 14842 a 14851).
- Causa del balance de M1_EXACT y de los errores de codigo: no revisada en `pinn_v3`.
- Reanudacion o parada manual de las filas 3 y 10 de la tabla E (fechas de escritura y logs).

### J. Proxima accion al retomar (detalle)

Confirmar D1 a D8 con el usuario. Despues: (1) resolver D1 y D5 en el texto y en la matriz; (2) revisar la causa del balance de M1_EXACT y los errores de implementacion; (3) verificar la Guia N°01 pagina a pagina; (4) contrastar las cifras del capitulo 1 con datos_fuente; (5) editar por capitulo (Fases 2 a 9), registrando cada cambio antes y despues en la bitacora.

## Juicio de expertos: criterio de trabajo acordado

El juicio experto sera una evaluacion externa estructurada de las soluciones de calculo termico generadas por el artefacto, comparadas con soluciones convencionales aplicables a cables enterrados. Antes de recolectar puntuaciones se elaborara y fijara un protocolo reproducible, alineado con la matriz vigente. Como minimo, el protocolo debe especificar:

- perfil y seleccion de evaluadores, numero previsto, experiencia pertinente y manejo de conflictos de interes;
- casos y resultados comparados, datos de entrada comunes, metodos convencionales de referencia y materiales entregados a los evaluadores;
- criterios con definiciones y anclajes observables para cada valor de la escala 1--10, incluyendo exactitud/adecuacion tecnica, consistencia fisica, utilidad para el calculo de ampacidad, trazabilidad y claridad de limites;
- procedimiento de evaluacion independiente, cegamiento cuando sea viable, registro de justificaciones y forma de resolver discrepancias;
- agregacion de puntuaciones, acuerdo entre evaluadores, umbral/regla de aceptacion y tratamiento de resultados inconclusos;
- relacion explicita con las metricas numericas: la puntuacion experta aporta juicio de utilidad y adecuacion profesional, mientras errores, residuos, balances y comparacion con referencias sustentan la verificacion.

No fijar aun cantidad de expertos, ponderaciones ni umbral global sin comprobar la matriz, las guias y la viabilidad. El protocolo se desarrollara antes de la evaluacion final y no se presentara como validado hasta aplicarlo y analizar sus resultados.

## Plan integral de revision

- [~] **Fase 0. Control de versiones y fuentes.** Preservar `Plan/` y los artefactos historicos; confirmar matriz mas reciente; releer guias UNI, guias de escritura, rubrica y fuentes primarias. Registrar decisiones de fuente.
- [~] **Fase 1. Auditoria de consistencia.** Construir una correspondencia unica entre problema general/especificos, objetivo general/especificos, hipotesis, VI/VD, fases F1--F4, indicadores, umbrales, instrumentos y productos. Resolver contradicciones con la matriz vigente, no con umbrales inventados.
- [ ] **Fase 2. Revision critica del planteamiento (capitulo 1).** Evaluar realidad, relevancia, alcance y evidencia del problema; revisar antecedentes y brecha; agregar/actualizar antecedentes PINN/ML para cables solo tras verificar fuentes; acortar repeticiones y corregir errores factuales/gramaticales.
- [~] **Fase 3. Objetivos e hipotesis (capitulo 2).** Alinear OE/HE con la matriz; cerrar o retirar reproducibilidad del objetivo general si no se convierte en producto verificable; robustecer HE1; distinguir umbrales de verificacion (F2) de los de evaluacion de escenarios (F3); definir cuando una diferencia es relevante solo si la matriz y el protocolo lo permiten.
- [ ] **Fase 4. Marco teorico (capitulo 3).** Revisar cobertura, pertinencia, actualidad y sintesis; incorporar antecedentes faltantes verificados; explicitar aporte y limitaciones de cada antecedente; eliminar catalogos o explicaciones repetidas que no sostengan el modelo o el metodo.
- [~] **Fase 5. Metodologia (capitulo 4).** Nombrar herramientas realmente usadas (PINN, FEM, malla, bibliotecas y hardware con version cuando corresponda); definir catalogo y seleccion de escenarios, niveles de contraste/extension/proximidad, semillas, configuracion candidata y protocolo para calcular `Imax` con tolerancias. Basar la eleccion de modelo en comparacion justa de las corridas `pinnV3` recientes. Alinear formulacion AC/DC, perdidas y limitaciones.
- [~] **Fase 6. Evaluacion y juicio experto.** Definir formulas reproducibles para cada metrica exigida (Tmax, NRMSE, balance, ampacidad, cambio porcentual, margen, robustez); establecer reglas de aceptacion separadas por fase; disenar el protocolo experto descrito arriba y precisar como su evidencia contribuye a aceptar/rechazar la propuesta sin reemplazar verificacion numerica.
- [~] **Fase 7. Viabilidad y administracion (capitulo 5).** Sustentar resultados preliminares con las campanas mas recientes de `pinnV3`, limitando las afirmaciones al estado real de cada corrida; actualizar cronograma; estimar y totalizar presupuesto o explicar los rubros no monetarios; incluir riesgos, mitigaciones y alternativas tecnicas realistas.
- [~] **Fase 8. Matriz y anexos.** Actualizar la matriz LaTeX solo despues de resolver la fuente; asegurar equivalencia literal de problemas, objetivos, hipotesis, variables, metricas, umbrales, juicio experto y controles con los capitulos. Revisar anexos restantes y referencias cruzadas.
- [ ] **Fase 9. Edicion de estilo y formato.** Aplicar las dos guias de escritura; reducir redundancia con parafrasis breves sin perder evidencia; mantener tercera persona, terminologia estable, citas verificables y estructura UNI. Corregir erratas, unidades, porcentajes y concordancia.
- [ ] **Fase 10. Verificacion final.** Compilar solo la fuente de `plan_v2` con LuaLaTeX/Biber segun instrucciones del proyecto; revisar referencias, tablas, figuras, bibliografia, paginas, errores y warnings relevantes. Hacer una lectura final severa contra cada criterio de la rubrica y dejar una lista de brechas residuales; no generar/reescribir PDF historicos sin autorizacion.

## Proxima accion al retomar

Confirmar con el usuario las decisiones D1 a D8 de la seccion de hallazgos (H). Despues seguir la seccion J: resolver umbrales y alcance electromagnetico, revisar la causa del balance de M1_EXACT, verificar la Guia N°01 pagina a pagina, contrastar las cifras del capitulo 1 y editar por capitulo registrando cada cambio antes y despues. La rubrica y el inventario de pinnV3 ya estan hechos.

## Bitacora de eventos

| Fecha y hora local | Estado | Evento y evidencia |
|---|---|---|
| 2026-11-07 10:28:05 -05:00 | Hecho y verificado | Creado este archivo tras confirmar que no existia. Se verifico el inventario de `plan_v2`, las rutas de guias y matriz, los criterios visibles en el anexo, el contenido actual de capitulos 2, 4 y 5 y los nombres de las campañas recientes de `pinnV3`. No se editaron aun capitulos, anexos ni matriz. |
| 2026-11-07 15:04 a 15:19 -05:00 | Hecho y verificado | Lectura completa de las guias de redaccion y de estructura, de los capitulos 00 a 06 y del anexo 1. Copias de guias identicas en Plan/guias y plan_v2/guias (SHA-256). Sin cambios en Plan/, capitulos, anexos ni matriz. |
| 2026-11-07 15:04 a 15:19 -05:00 | Hecho y verificado | Rubrica leida por OCR (3 paginas). Guia N°01 (60 pp.) sin texto extraible: OCR ilegible; verificacion pagina a pagina pendiente. Matriz rev1 XLSX leida (3 hojas, 2 ocultas) y PDF rev1 extraido. Hash rev2 verificado. |
| 2026-11-07 15:04 a 15:19 -05:00 | Hecho y verificado | Inventario de 11 carpetas de pinnV3 (2 a 7 de noviembre): clasificacion, contradicciones de accepted y umbrales contrastados con pinn_v3/evaluation/metrics.py y con los documentos de pinnV3. |
| 2026-11-07 15:04 a 15:19 -05:00 | Hecho y verificado | Comprobaciones automaticas del .tex: 0 referencias rotas; 49 citas = 49 entradas del .bib; terminos, oraciones largas y simbolos sin definir. Correccion de hallazgo previo: el capitulo 1 si contiene formulacion y justificacion/alcances. |
| 2026-11-07 15:19 -05:00 | Pendiente | Decisiones D1 a D8 y verificaciones de la seccion I. No se aplicaron cambios de contenido. |
| 2026-11-07 15:59 -05:00 | En curso | Resoluciones D1 a D8 registradas (seccion H2). Recuento de campañas pinnV3 por extension. Inicio de la edicion por capitulo: capitulo 2 (OG, HE1 a HE3 y umbrales), capitulo 4 (metricas, formulacion Joule y resistencia, criterios, referencia al protocolo), anexo 2 nuevo y registro en el anexo 06. Pendientes: capitulos 1, 3 y 5, y anexo 1. |
| 2026-11-07 (sesion de edicion, tras 15:59 -05:00) | Hecho (verificacion automatica OK; sin compilar) | Capitulo 2: OG reformulado con repositorio publico (D2); HE1 con cobertura mayor a 90 % de requisitos criticos; HE2 y HE3 con umbrales de la matriz (D1) y HE3 dividida en dos oraciones; retirada la frase que pospone la revision de umbrales al piloto; criterios F2 y F3 en el parrafo de umbrales. Capitulo 4: retirado el parrafo Precisamiento de metodos; formulacion Joule con R(T) referida a 20 C y limite electromagnetico en modelo fisico (D5); herramientas nombradas (PyTorch, FEniCSx, Gmsh, WSL); mapeo de fases a F0-F4; NRMSE, desequilibrio energetico y margen definidos; e_Tmax referido a la elevacion termica; tabla de criterios con umbrales de la matriz; referencia al protocolo de expertos (D3). Anexo 2 nuevo (borrador de protocolo preliminar) incluido en el anexo 06. Verificacion: 64 etiquetas, 49 referencias sin rotura, 49 citas sin faltantes ni sobrantes. Pendiente: capitulos 1, 3 y 5, anexo 1 (sincronizar XLSX fuera de plan_v2) y compilacion. |
| 2026-11-07 (compilacion de control) | Hecho y verificado | Compilacion de plan_v2 en copia temporal (%TEMP%), sin tocar el repositorio: 71 paginas, sin errores, sin referencias ni citas indefinidas, biber sin advertencias. |
| 2026-11-07 (antes de editar capitulo 1) | En curso | Objetivo en `capitulos/01_planteamiento_del_problema.tex`: (a) condensar en la introduccion la triada de problemas, que la seccion 1.2 repite; (b) requieren a requiere; (c) simbolo de porcentaje en 19 y 11; (d) resistividad medida a reportada y aclarar la frase de temperatura del terreno (Arias Velasquez); (e) definir k_eq antes de la figura de brecha; (f) limitaciones alineadas con D5; (g) incorporar a la tabla de antecedentes y a los apartados tematicos los estudios tecnicos citados fuera de ella (aldulaimi2024, ariasVelasquez2023, mobius2025, fariz2026) y declarar el criterio para las fuentes de contexto; (h) mencionar el anexo 2 en el cierre de la introduccion. Las paginas 752-760 de neher1957 se mantienen: son un subrango del articulo (752-772 en el .bib). |
| 2026-11-07 (capitulo 1 aplicado) | Hecho (verificacion por busqueda; sin compilar) | Capitulo 1: (a) introduccion condensada, la triada queda solo en 1.2 con referencia cruzada; (b) requiere; (c) 19 % y 11 %; (d) resistividad reportada y aclaracion de la temperatura del terreno en Arias Velasquez; (e) k_eq definido en la dificultad metodologica; (f) limitaciones con I^2 R(T), R a 20 C, sin Maxwell (D5); (g) tabla de antecedentes con 23 fuentes (ariasVelasquez2023, aldulaimi2024, mobius2025, fariz2026) y criterio para fuentes de contexto; parrafos tematicos ampliados; (h) cierre de la introduccion con anexos 1 y 2. Pendiente: capitulos 3 y 5, anexo 1 y compilacion. |
| 2026-11-07 (capitulo 3 aplicado) | Hecho (verificacion por busqueda; sin compilar) | Capitulo 3: posicionamiento reformulado como formulacion PINN delimitada al dominio 2D estacionario y a las referencias del capitulo 4; los dos usos de importante reemplazados; en esta tesis retirado; definiciones FDM y FVM retiradas por duplicar el parrafo de metodos de calculo. DTR, PAC y el resto del catalogo se mantienen porque el texto los usa. Neher1957 se mantiene en 752-760 segun la decision del capitulo 1. Pendiente: capitulo 5, anexo 1 y compilacion. |
| 2026-11-07 (capitulo 5 aplicado) | Hecho (verificacion por busqueda; sin compilar) | Capitulo 5: fase 3 movida a octubre con extension a noviembre (D4); parrafo de extension de las campañas A-D (corridas por campaña, casos, arquitecturas y semillas), con proposito de identificar las arquitecturas PINN con mejor desempeno; hitos H1-H3 con fechas corregidas; leyenda del cronograma y fecha de revision 7-nov-2026; presupuesto sin montos inventados, indicando que queda abierto hasta obtener cotizaciones. Pendientes: (1) fases 1 y 2 en julio-septiembre siguen en el cronograma sin evidencia verificada; (2) la correspondencia de letras A-D con los sufijos de carpeta es suposicion a confirmar (D4); (3) montos del presupuesto requieren datos del usuario. Pendiente: anexo 1 y compilacion. |
| 2026-11-07 (anexos 1 y 2 aplicados) | Hecho (verificacion por busqueda) | Anexo 1: celda HE1 corregida; umbrales de HE2, F2 y F3 alineados con la matriz rev1 (5 % y 2 %; 0,1 K y 1 %; 0,5 K y 2,5 %); cuadratico y limite corregidos; sinteticos retirado de F2 para coincidir con la matriz; referencia al anexo 2 en F4. Anexo 2: protocolo condensado y cuestionario por caso de cinco preguntas; metodos: mediana por criterio y ICC de dos vias. Pendiente: sincronizar el XLSX (typos y nombre de hoja) con permiso del usuario. |
| 2026-11-07 (compilacion de control 2) | Hecho y verificado | Copia temporal de plan_v2 con LuaLaTeX y biber: 73 paginas; sin errores, sin referencias ni citas indefinidas, sin advertencias de biber; una caja desbordada de 0,2 pt. |
| 2026-11-07 (revision final de rubrica y anexo 1) | Hecho | Revision rapida: criterios 1 y 3 cumplidos en lo esencial; criterios 2, 4 y 5 parciales. Pendientes: verificar cifras del capitulo 1 contra datos_fuente, indicador por objetivo especifico, versiones y configuracion de entrenamiento en el capitulo 4, y sincronizar el XLSX. |
| 2026-11-08 (capitulo 2: indicadores por objetivo) | Hecho (verificacion por busqueda; sin compilar) | Cada objetivo especifico OE1-OE4 recibe una frase con su indicador tomado de la matriz (anexo 1): OE1 cobertura de requisitos criticos mayor de 90 % y trazabilidad; OE2 |Delta Tmax| menor que 0,1 K y NRMSE menor que 1 %; OE3 |Delta Tmax| menor que 0,5 K, NRMSE menor que 2,5 % y posicion del punto caliente; OE4 ampacidad, Delta I, margen y juicio de expertos 1-10 (anexo 2). Pendientes: (1) verificar las cifras del capitulo 1 contra datos_fuente (indicio a confirmar: el enlace subterraneo de 220 kV del informe del COES no coincide con la reconfiguracion Chavarria-Santa Rosa-Carapongo citada); (2) sincronizar el XLSX con el documento, si Excel no lo bloquea. |
| 2026-11-08 (capitulo 1: Punta Lomitas) | Hecho y verificado (fuentes en linea; compilacion limpia) | Correccion: la central tiene 57 aerogeneradores, 50 del proyecto original de 260 MW (COES 2021 y ENGIE 2023) mas 7 de la expansion de 36,4 MW (COES 2022, dos circuitos de 33 kV). Se retiro la cifra de 290 km de cables XLPE, que no aparece en las fuentes consultadas; la linea de transmision de 60 km hacia el SEIN es la cifra de ENGIE 2023. Claves nuevas en referencias_plan_tesis.bib: coes2022puntaLomitasExpansion y engie2023puntaLomitas. Compilacion en carpeta temporal nueva: 73 paginas, 0 errores, 0 referencias ni citas indefinidas. Pendientes: (a) la frase de siete centrales (cuatro eolicas y tres solares, unos 1250 MW) no esta verificada; la nota del MINEM de 2025 reporta dos solares y ninguna eolica por 717 MW; (b) las longitudes de 8266 y 17179 km y la maxima demanda de 7942 MW de 2025 no tienen respaldo (el anuario da 5863 km en 2010 y 15024 km en 2024); (c) presentacion_plan_tesis_cables_pinn.tex conserva los 290 km; (d) las compilaciones de control previas usaron copias antiguas de capitulos en %TEMP%\ths_build_check porque TeX busca primero en la carpeta de salida; desde ahora compilar en carpeta nueva. |
| 2026-11-08 (capitulo 1: cable aproximado) | Hecho y verificado (compilacion en carpeta nueva: 73 paginas, 0 errores, 0 referencias ni citas indefinidas) | Cifra de cable de 33 kV: el proyecto original tiene 15 circuitos y unos 75 km de trazado (COES 2021, tabla 9: 50 tramos, 75,2 km). Supuesto del autor: los dos circuitos de la expansion tienen longitud similar (COES 2022 confirma solo que son dos). Total: unos 85 km de trazado y aproximadamente 255 km de cable con tres conductores por circuito trifasico. El estudio COES 2021 no menciona XLPE; la nota general sobre XLPE se conserva sin atribuirla a Punta Lomitas. Matriz rev2: verificacion parcial por busqueda; rev1 tiene 'aceptcion', '5,%', 'cuadratico', 'Temperatura limite' y '(mperios)' y rev2 ya no los tiene. Pendientes: (a) frase de siete centrales sin verificar; (b) longitudes 8266/17179 km y demanda 7942 MW sin respaldo; (c) presentacion_plan_tesis_cables_pinn.tex conserva 290 km. |
