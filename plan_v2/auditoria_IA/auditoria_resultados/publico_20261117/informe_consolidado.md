# Auditoria de los capitulos I a V: consolidado de cierre

**Evaluacion de una version posterior:** el PDF editorial fue evaluado por separado el 17 de noviembre de 2026. Sus resultados y comparacion estan en [la evaluacion editorial](../version_editorial_20261117/informe_consolidado.md). Este documento conserva los resultados de la version anterior; no deben atribuirse al PDF actualizado.

**Cierre de alcance:** 17 de noviembre de 2026, por indicacion del usuario.

**Actualizacion del capitulo I:** en la nueva consulta del 17 de noviembre a las 19:12:19 UTC, el mismo informe muestra **36,8/100, Low risk y 23 de 23 motores**, con coincidencia entre pantalla y JSON. Se conserva abajo la tabla historica; no se sustituye su evidencia. El portal reutilizo el informe, por lo que no se acredita un segundo calculo independiente. Detalle y archivos en la [segunda consulta del capitulo I](../reanalisis_capitulo_1_20261117/informe_consolidado.md).

La auditoria se cierra con los cinco capitulos, correspondientes a las paginas PDF 12 a 72 (61 paginas). Preliminares, referencias y anexos quedan fuera del alcance final. Se detuvo el recolector local y no se autorizaron nuevos envios. La solicitud de preliminares ya enviada se conserva como antecedente; no se afirma que se haya cancelado en el servidor del portal. Las ocho entradas originales se mantienen intactas.

Las puntuaciones se transcriben del portal; no prueban autoria ni calidad academica.
No se promedian capitulos para fabricar una puntuacion de todo el documento.

## Resultados Completos Verificados

**Actualizacion:** 17 de noviembre de 2026, 19:25:58 UTC. Se recuperaron los informes completos de III a V sin reenviar textos. Combinados con las capturas completas de I y II, los cinco capitulos tienen ahora 23 resultados por motor y coincidencia entre pantalla y JSON en sus versiones verificadas.

| Capitulo | Pantalla /100 | JSON /100 | Motores puntuados | Evidencia actual |
|---|---:|---:|---|---|
| I. Planteamiento del problema | 36,8 | 36,8 | 23 de 23 | [Segunda consulta](../reanalisis_capitulo_1_20261117/informe_consolidado.md) |
| II. Objetivos | 17,3 | 17,3 | 23 de 23 | [JSON completo](02_objetivos_exportacion_20261117T154424_931660Z.json) |
| III. Marco teorico | 16,3 | 16,3 | 23 de 23 | [JSON recuperado](../recuperacion_III_V_20261117/03_marco_teorico_exportacion_20261117T192553_726339Z.json) |
| IV. Metodologia | 22,1 | 22,1 | 23 de 23 | [JSON recuperado](../recuperacion_III_V_20261117/04_metodologia_exportacion_20261117T192556_755309Z.json) |
| V. Administracion | 27,3 | 27,3 | 23 de 23 | [JSON recuperado](../recuperacion_III_V_20261117/05_administracion_exportacion_20261117T192558_187820Z.json) |

La discrepancia de las exportaciones antiguas de III--V ya no aparece en los resultados actuales. La [recuperacion documentada](../recuperacion_III_V_20261117/informe_consolidado.md) conserva capturas y exportaciones originales. Son consultas posteriores de los mismos informes, no pruebas independientes ni mejoras atribuibles a cambios del texto.

## Registro Historico De La Primera Captura

La tabla siguiente conserva las capturas iniciales y no representa el estado actual. No se borran ni sustituyen las evidencias parciales.

| Unidad | Paginas PDF | Estado | Pantalla /100 | Exportacion /100 | Evidencia |
|---|---|---|---|---|---|
| Planteamiento del problema | 12 a 28 | informe_disponible_motores_parciales | 96.0 | 99.3 | [Captura](01_planteamiento_resultados.json); [JSON original](01_planteamiento_exportacion_original.json) |
| Objetivos de la investigacion | 29 a 35 | informe_disponible_cobertura_no_acreditada | 17.3 | 17.3 | [Captura](02_objetivos_captura_20261117T154424_931660Z.json); [JSON original](02_objetivos_exportacion_20261117T154424_931660Z.json) |
| Marco teorico | 36 a 49 | informe_disponible_cobertura_no_acreditada | 16.3 | 99.4 | [Captura](03_marco_teorico_captura_20261117T160125_528155Z.json); [JSON original](03_marco_teorico_exportacion_20261117T160125_528155Z.json) |
| Metodologia de la investigacion | 50 a 68 | informe_disponible_cobertura_no_acreditada | 22.1 | 99.3 | [Captura](04_metodologia_captura_20261117T163427_775372Z.json); [JSON original](04_metodologia_exportacion_20261117T163427_775372Z.json) |
| Administracion del plan de tesis | 69 a 72 | informe_disponible_cobertura_no_acreditada | 27.3 | 99.3 | [Captura](05_administracion_captura_20261117T171529_702726Z.json); [JSON original](05_administracion_exportacion_20261117T171529_702726Z.json) |

Informes archivados: **5 de 5 capitulos incluidos en el alcance final**. El cierre acredita la conservacion de informes y evidencias, no una validacion de autoria, plagio, calidad academica o cobertura completa de los 23 motores.

## Interpretacion Del Capitulo I: Captura Inicial

**El valor de 96,0/100 de la captura inicial era una senal alta del detector; no una nota academica ni un 96 % demostrado de contenido generado por IA.** El portal mostro `Slop detected`, pero esa captura solo contiene puntuaciones en 14 de 23 motores, con nueve en `Waiting` aunque la cabecera indicaba `Complete`. La consulta posterior, documentada arriba, ya muestra 36,8 y los 23 motores puntuados.

Los motores discrepan: TMR Detector muestra 95,2, Structural 51,3, OpenAI Detector 14,3 y SuperAnnotate 0,1. No se convierten esos valores en votos equivalentes ni se promedian localmente. La exportacion del mismo informe devuelve 99,3 y solo seis motores. No se ha establecido la causa de la diferencia entre pantalla y exportacion.

El resultado justifica una revision humana de la redaccion y de la trazabilidad, pero **no permite concluir que el capitulo este mal, que carezca de aporte propio o que su autoria haya sido demostrada por el detector**. Tampoco basta para certificar que todo esta correcto. La evaluacion academica debe contrastar afirmaciones, citas y fuentes, coherencia del argumento, limites y contribucion propia; esta auditoria de deteccion no reemplaza esa evaluacion.

No se modifico el plan para reducir puntuaciones ni se respondio al cuestionario de autoria del portal. El [informe individual del capitulo I](01_planteamiento_informe.md) conserva el detalle disponible por motor de la captura inicial. Las discrepancias de esa captura historica impedian interpretar sus columnas como una medicion consistente; las consultas posteriores de I y III--V, documentadas arriba, contienen ya los 23 motores y coinciden con la pantalla.

## Observaciones Historicas

- Planteamiento del problema: El JSON exportado tiene 99.3 y seis motores; la captura visible conserva 96.0 y catorce. No se sustituyen ni se fusionan.
- Planteamiento del problema: https://sloptotal.com/report/?id=3de8ee861873
- Objetivos de la investigacion: https://sloptotal.com/report/?id=26b29dad10dd
- Marco teorico: Pantalla: 16.3; exportacion: 99.4. Se conservan ambas; causa no establecida.
- Marco teorico: https://sloptotal.com/report/?id=98c56e99c33f
- Metodologia de la investigacion: Pantalla: 22.1; exportacion: 99.3. Se conservan ambas; causa no establecida.
- Metodologia de la investigacion: https://sloptotal.com/report/?id=94bbcc8e9790
- Administracion del plan de tesis: Pantalla: 27.3; exportacion: 99.3. Se conservan ambas; causa no establecida.
- Administracion del plan de tesis: https://sloptotal.com/report/?id=b07fd08fe60a

Los textos incluyen tablas, formulas, referencias y encabezados extraidos del PDF.
La cobertura interna y el truncamiento de cada motor no estan acreditados.
Un estado Complete no garantiza que los 23 motores tengan puntuacion.
El portal declara que elimina los informes despues de 30 dias.
Las capturas historicas no se sustituyen por resultados de otra fecha.
No se afirma que la redaccion academica en espanol, las tablas o las citas hayan causado la puntuacion: esa causalidad no se comprobo. Tampoco se realizaron deteccion de plagio ni comprobacion externa de todas las fuentes mediante estos resultados.
