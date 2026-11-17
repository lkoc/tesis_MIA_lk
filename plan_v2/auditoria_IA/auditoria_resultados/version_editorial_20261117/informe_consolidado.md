# SlopTotal: evaluacion de la version editorial

Actualizado: 2026-11-17T22:00:59.453565+00:00.

Las puntuaciones se transcriben del portal; no prueban autoria ni calidad academica.
No se promedian capitulos para fabricar una puntuacion de todo el documento.

## Version Y Alcance

Se evaluaron los capitulos I a V del [PDF editorial evaluado y preservado](../../../old/plan_previo_afinamiento_IV_V_20261117_ce82f135.pdf), SHA-256 `ce82f135ae0e05d1e5700c5a37355bbd26275b4537c7af390f9dc2b721d8a824`. Las cinco unidades abarcan las paginas PDF 12 a 72 (61 paginas). Preliminares, referencias y anexos quedaron fuera del envio.

**Afinado posterior de IV/V:** el PDF de trabajo se actualizo despues de esta evaluacion y todavia no tiene una nueva puntuacion. El [analisis de cambios de IV y V](analisis_IV_V.md) documenta el contraste de motores, los ajustes de exposicion y las huellas de ambas versiones. Las cifras de este informe siguen describiendo exclusivamente el PDF editorial preservado.

Se realizaron cinco envios secuenciales mediante el formulario normal. Los cinco identificadores difieren de los de la version anterior. Cada entrada se verifico mediante SHA-256 antes de enviarla; se guardaron el texto exacto, su ubicacion, el registro de envio, la captura del navegador y la exportacion JSON original. Los informes permanentes ya terminados se recargaron antes de exportar. No se repitieron envios por demora y el recolector quedo detenido al finalizar.

## Resultados Completos

| Unidad | Paginas PDF | Estado | Pantalla /100 | Exportacion /100 | Evidencia |
|---|---|---|---|---|---|
| I. Planteamiento del problema | 12 a 28 | 23 de 23 motores puntuados | 29,5 | 29,5 | [Captura](01_planteamiento_captura_20261117T215152_890925Z.json); [JSON original](01_planteamiento_exportacion_20261117T215152_890925Z.json) |
| II. Objetivos | 29 a 35 | 23 de 23 motores puntuados | 11,8 | 11,8 | [Captura](02_objetivos_captura_20261117T215319_608778Z.json); [JSON original](02_objetivos_exportacion_20261117T215319_608778Z.json) |
| III. Marco teorico | 36 a 49 | 23 de 23 motores puntuados | 22,7 | 22,7 | [Captura](03_marco_teorico_captura_20261117T215542_891255Z.json); [JSON original](03_marco_teorico_exportacion_20261117T215542_891255Z.json) |
| IV. Metodologia | 50 a 68 | 23 de 23 motores puntuados | 30,4 | 30,4 | [Captura](04_metodologia_captura_20261117T215844_980367Z.json); [JSON original](04_metodologia_exportacion_20261117T215844_980367Z.json) |
| V. Administracion | 69 a 72 | 23 de 23 motores puntuados | 47,0 | 47,0 | [Captura](05_administracion_captura_20261117T220059_441439Z.json); [JSON original](05_administracion_exportacion_20261117T220059_441439Z.json) |

Informes archivados y verificados: **5 de 5 capitulos en alcance**, con **115 resultados individuales**. Para cada uno se verificaron 23 nombres de motor distintos y 23 puntuaciones numericas tanto en pantalla como en JSON, ademas de la coincidencia del identificador y del agregado.

## Comparacion Con La Version Anterior

La referencia es la [evaluacion completa anterior](../publico_20261117/informe_consolidado.md), no las primeras exportaciones parciales de seis motores. Aquel PDF se conserva en [la copia previa a la revision editorial](../../../old/plan_previo_revision_editorial_20261117_3f9d80b9.pdf).

| Capitulo | Anterior /100 | Editorial /100 | Diferencia en puntos | Etiqueta actual del portal |
|---|---:|---:|---:|---|
| I | 36,8 | 29,5 | -7,3 | Clean - likely human-written |
| II | 17,3 | 11,8 | -5,5 | Clean - likely human-written |
| III | 16,3 | 22,7 | +6,4 | Clean - likely human-written |
| IV | 22,1 | 30,4 | +8,3 | Low risk |
| V | 27,3 | 47,0 | +19,7 | Suspicious |

**El resultado es mixto:** bajaron I y II y subieron III, IV y V. V presenta el mayor aumento y el portal lo clasifica como `Suspicious`; se conserva esa clasificacion sin suavizarla. La comparacion es descriptiva: no demuestra que una correccion concreta cause el cambio ni que una version sea mas humana, mas original o de mejor calidad que la otra. No se dispone de repeticiones controladas para estimar la variabilidad del detector.

No se modifico el plan durante esta evaluacion ni se busco una puntuacion menor mediante reenvios sucesivos. Un resultado bajo no prueba autoria humana; un resultado alto no prueba autoria por IA. Estas cifras no representan una proporcion demostrada de palabras generadas por IA. El capitulo V merece una lectura editorial independiente, pero no una reescritura automatica basada solo en el numero.

## Observaciones

- Planteamiento del problema: https://sloptotal.com/report/?id=1dd6a94c35af
- Objetivos de la investigacion: https://sloptotal.com/report/?id=11c69fa013a3
- Marco teorico: https://sloptotal.com/report/?id=3bf37e0427dc
- Metodologia de la investigacion: https://sloptotal.com/report/?id=f0c4c9a146f6
- Administracion del plan de tesis: https://sloptotal.com/report/?id=5addc08d07b8

Los textos incluyen tablas, formulas, referencias y encabezados extraidos del PDF.
La cobertura interna y el truncamiento de cada motor no estan acreditados.
No se infiere cobertura solo por el estado Complete: en esta ejecucion se comprobaron los 23 resultados numericos por capitulo. Esto no acredita que cada motor haya procesado todas las palabras del texto.
El portal declara que elimina los informes despues de 30 dias.
Las capturas historicas no se sustituyen por resultados de otra fecha.
