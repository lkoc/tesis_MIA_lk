# IV y V: contraste de versiones y afinado de redaccion

**Fecha:** 17 de noviembre de 2026.

## Alcance Y Limite

Se comparan la version anterior a la revision editorial y la version editorial evaluada con SlopTotal. Ambas tienen informes completos de 23 motores por capitulo. El contraste identifica cambios de texto y de resultados, pero no permite demostrar que una frase o una estructura hayan causado una variacion del agregado.

La version editorial evaluada tiene SHA-256 `ce82f135ae0e05d1e5700c5a37355bbd26275b4537c7af390f9dc2b721d8a824`. Las correcciones puntuales posteriores descritas al final son una tercera version. Al escribir este analisis todavia no se habian evaluado; la [evaluacion posterior del afinado](../afinamiento_IV_V_20261117/informe_consolidado.md) obtuvo IV 33,9 y V 41,0, con 23 motores y coincidencia pantalla/JSON. Los resultados de las versiones anteriores se conservan sin cambios.

## Que Cambio En Los Textos Evaluados

| Capitulo | Cambio observado frente a la version previa | Aspecto que merece revision humana |
|---|---|---|
| IV | La explicacion del nivel paso de una oracion compuesta a varias frases con sujetos y estructuras semejantes. | Vincular cada clasificacion con el procedimiento concreto, sin repetir una definicion general en cada frase. |
| IV | Se separo la explicacion de metricas y normalizacion y se anadio un cierre del capitulo. | Mantener los bloques que separan ideas distintas; evitar que las transiciones solo anuncien lo que hace el documento. |
| V | Se amplio la apertura para explicar la funcion del cronograma y del presupuesto. | Abrir con las necesidades de ejecucion, en lugar de una descripcion general de las partes del capitulo. |
| V | La prosa presupuestaria paso a varios bloques explicativos y el financiamiento comenzo con una frase corta seguida de su desarrollo. | Nombrar los conceptos concretos y evitar reiterar lo que ya dice la tabla, sin perder las advertencias sobre estimaciones. |

Son observaciones editoriales, no rasgos exclusivos de un LLM. Dividir una oracion o un parrafo puede mejorar su comprension; no se considera un defecto por si mismo.

## Que Cambio En Los Motores

Agregados del portal: IV paso de **22,1 a 30,4** y V de **27,3 a 47,0**. Las puntuaciones individuales siguientes se expresan en escala de 0 a 100; en los JSON originales figuran en escala de 0 a 1.

| Capitulo | Motor | Antes | Editorial | Cambio en puntos |
|---|---|---:|---:|---:|
| IV | TMR Detector | 72,9 | 94,8 | +21,9 |
| IV | Desklib DeBERTa | 35,9 | 3,7 | -32,2 |
| IV | OpenAI Detector | 22,4 | 13,9 | -8,5 |
| IV | Burstiness | 0,0 | 4,3 | +4,3 |
| V | BERT-tiny RAID | 88,6 | 71,7 | -16,9 |
| V | OpenAI Detector | 16,3 | 31,0 | +14,7 |
| V | Structural Analysis | 45,3 | 41,0 | -4,3 |
| V | Burstiness | 61,2 | 53,6 | -7,6 |

En V disminuyeron tanto Structural Analysis como Burstiness mientras el agregado aumento. En IV hubo grandes variaciones de signo contrario entre clasificadores. Por ello, **no se sostiene que el aumento se explique simplemente por menor variedad de oraciones, mas muletillas o peor redaccion**. Tampoco se calculan contribuciones causales de los motores al agregado, pues no se ha verificado el modelo de calibracion utilizado en estas solicitudes.

Los detalles de Burstiness declaran muestras de 18 oraciones para IV y 15 para V en la version editorial. Esas muestras no equivalen a una lectura completa de ambos capitulos. Los indicadores de legibilidad y estructura tampoco identifican por si solos parrafos problematicos ni certifican autoria.

## Recursos De I A III Que Se Pueden Aprovechar

- **De I:** exponer una decision o una limitacion y explicar su consecuencia en el problema concreto. Aplicacion en IV: relacionar representacion, calculo y evidencia requerida.
- **De II:** distinguir la accion propuesta de su criterio de comprobacion. Aplicacion en IV: separar la clasificacion metodologica del modo en que se contrastaran los pares.
- **De III:** explicar una definicion por su uso en el modelo, con sujetos concretos y precision terminologica. Aplicacion en V: nombrar conectividad, energia y derechos de sustentacion, en vez de referencias vagas a consumos y tramites.
- **En ambos:** conservar la continuidad entre oraciones y las repeticiones tecnicas necesarias. No introducir sinonimos, errores o longitudes artificiales para aparentar autoria humana.

## Ajustes Puntuales Aplicados Despues Del Contraste

En IV se afinan la apertura, el nivel de investigacion y la introduccion de DSR. El funcionamiento de la PINN se presenta como objeto de evaluacion, no como exito que deba confirmarse. En V se concreta la apertura y la prosa presupuestaria y se elimina una frase redundante del financiamiento.

Se siguen las guias Markdown del plan. Las formulas, cifras, referencias, citas, figuras y tablas se comparan con una copia tomada antes de este afinado; el protocolo, los importes y los hitos se conservan. I--III no se modifican en esta ronda. No se vuelve a enviar el documento a SlopTotal ni se promete una puntuacion menor.

Verificacion final: las ocho filas numericas de este informe coinciden con los JSON originales. El control de IV/V conserva los elementos tecnicos protegidos; LuaLaTeX/Biber compilaron 85 paginas sin errores ni referencias indefinidas y sin advertencias de Biber. La inspeccion visual de las paginas afectadas no mostro recortes ni solapamientos.

El [PDF de trabajo afinado](../../../plan_tesis_cables_pinn.pdf) tiene SHA-256 `9123017df6815b867c10145947f2d828b84fd7575cfda478c0c6e970f37644fc`. El [PDF editorial evaluado](../../../old/plan_previo_afinamiento_IV_V_20261117_ce82f135.pdf), al que corresponden 30,4 y 47,0, se conserva con su huella original. El PDF afinado fue evaluado despues: IV 33,9 y V 41,0. Ese resultado no acredita una causa especifica para las variaciones observadas.

## Evidencia

- IV anterior: [JSON completo](../recuperacion_III_V_20261117/04_metodologia_exportacion_20261117T192556_755309Z.json).
- IV editorial: [JSON completo](04_metodologia_exportacion_20261117T215844_980367Z.json).
- V anterior: [JSON completo](../recuperacion_III_V_20261117/05_administracion_exportacion_20261117T192558_187820Z.json).
- V editorial: [JSON completo](05_administracion_exportacion_20261117T220059_441439Z.json).
- Resultados y limites de la evaluacion editorial: [informe consolidado](informe_consolidado.md).

La prioridad es una exposicion precisa y verificable. Una subida del detector no prueba que el texto haya empeorado academicamente, del mismo modo que una bajada no demuestra autoria humana.