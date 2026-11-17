# Verificacion final de las observaciones del asesor

**Fecha:** 17 de noviembre de 2026.

**Documento rector:** [Informe de evaluacion del 13 de noviembre](Informe_Evaluacion_Plan_Tesis_Koc_Melendez_PINN.md).

**Version contrastada:** [PDF final del plan](../plan_tesis_cables_pinn.pdf), 85 paginas, SHA-256 `baa97227721ad14407759c90149bf9da85d6df62a1a695993ac352da9c530a89`.

## Dictamen De Cierre

**Nueve observaciones estan atendidas documentalmente y una permanece parcial: la numero 9, por falta de cotizaciones y tarifas verificadas.** La version puede remitirse al asesor como plan corregido, identificando esa salvedad; no corresponde afirmar que las diez observaciones estan formalmente absueltas.

Atendida documentalmente significa que el plan ya explicita la definicion, el procedimiento, el limite o la evidencia exigida. No significa que las corridas de confirmacion, la validacion de campo o los hitos futuros se hayan ejecutado. La aprobacion y el cierre formal corresponden al asesor y a la instancia academica.

## Cotejo De Las Diez Observaciones

| N. | Observacion del informe | Estado | Evidencia y conclusion |
|---|---|---|---|
| 1 | Unificar HE2, criterios y matriz | Atendida documentalmente | [HE2](../capitulos/02_objetivos.tex#L69), [criterios](../capitulos/04_metodologia.tex#L416) y [matriz integrada](../capitulos/anexos/anexo_1_matriz_consistencia.tex#L147) comparten F2: error absoluto E_T menor que 0,1 K y NRMSE menor que 1 %. F3 exige E_T menor que 0,5 K y NRMSE menor que 2,5 %. No hay puerta alternativa del 5 %; se distingue error numerico de efecto fisico. |
| 2 | Formalizar perdidas y ampacidad | Atendida documentalmente | [Resistencia y fuente Joule](../capitulos/04_metodologia.tex#L188): ley R(T), fuente uniforme por conductor, temperaturas medias, unidades y modos DC/AC excluyentes. Se declaran las perdidas excluidas y se evita duplicar piel/proximidad. La [busqueda acotada](../capitulos/04_metodologia.tex#L207) fija intervalo, tolerancias, limites de iteracion, control FEM e inconclusos. |
| 3 | Completar el entrenamiento PINN | Atendida documentalmente | [Protocolo y tabla de configuracion](../capitulos/04_metodologia.tex#L215): red por subdominio material, capas, neuronas, activacion, inicializacion, precision, escalas, colocacion, pesos, Adam/L-BFGS, semillas, seleccion y parada. Se fija w_d=0 y se reservan referencias para evaluar, no para entrenar o detener la confirmacion. |
| 4 | Precisar la homogenizacion | Atendida documentalmente | [Regla por area del entorno](../capitulos/04_metodologia.tex#L300), sin incluir capas de cable ni calibrar a resultados. Los pares conservan geometria y contornos; VD3 usa perdidas iguales y VD4 conserva la ley R(T), no el valor de las perdidas. HE4 admite refutacion y el promedio no se presenta como cota universal. |
| 5 | Convergencia y referencia FEM | Atendida documentalmente | [Referencia FEM](../capitulos/04_metodologia.tex#L402): FEniCSx/Gmsh, elementos de grado dos, PETSc/LU/MUMPS, residuo algebraico, refinamientos y tolerancias termicas, energeticas y de ampacidad. Se verifican casos exactos, maximos y cuadratura; se separan estabilidad de referencia y error PINN, sin llamar cota rigurosa a una estimacion numerica. |
| 6 | Relevancia y variabilidad | Atendida documentalmente | [Reglas de decision](../capitulos/04_metodologia.tex#L380): efectos minimos de 1 K y 2 %, cinco semillas pareadas, mediana/rango, envolventes con margen numerico y categorias relevantes, no relevantes o inconclusas. No se eliminan semillas fallidas ni se presentan envolventes como intervalos de confianza. |
| 7 | Acotar la interpretacion externa | Atendida documentalmente | [Limitaciones](../capitulos/01_planteamiento_del_problema.tex#L534): sin mediciones de campo ni certificacion operacional, con conclusiones limitadas a geometria, materiales, fuentes y contornos verificados. Se conserva el alcance 2D estacionario y de perdidas simplificadas. |
| 8 | Comparar antecedentes y revisar pertinencia | Atendida documentalmente | [Sintesis critica](../capitulos/01_planteamiento_del_problema.tex#L401): metodo, geometria/datos, limites y decision de diseno. La referencia de lineas aereas no esta activa en las citas compiladas. Biber resuelve las citas sin advertencias. Esto no equivale a auditoria externa de DOI, autenticidad o indexacion. |
| 9 | Cotizar presupuesto y completar riesgos | Parcial | El [presupuesto](../capitulos/05_administracion.tex#L67) tiene subtotal S/ 2940, contingencia S/ 588 y total S/ 3528; la [tabla de riesgos](../capitulos/05_administracion.tex#L106) fija mitigaciones y reglas de continuidad. El texto sigue declarando importes pendientes de contrastar con cotizaciones y tarifas. No se encontraron precios acreditados dentro del plan que permitan cerrar esa parte de la observacion. |
| 10 | Alinear cronograma e hitos | Atendida documentalmente | [Cronograma y hitos](../capitulos/05_administracion.tex#L8): corte al 17-nov-2026, cierre programado en junio de 2027, campanas historicas clasificadas como exploratorias e hitos con evidencia exigida. Se aclaro H1 para que los documentos aprobados se lean como requisitos para cerrarlo, no como una aceptacion ya lograda. |

## Pendiente Que Requiere Respaldo Externo

Para cerrar completamente la observacion 9 se requiere contrastar los importes de impresion/encuadernacion, conectividad/energia, derechos institucionales y movilidad con cotizaciones, tarifas o bases documentadas; la ruta de publicacion sin APC debe explicitarse como supuesto mientras no haya revista definida. Los respaldos deben identificar fuente o proveedor, fecha, unidad, cantidad y condiciones aplicables.

Si los montos comprobados cambian, se actualizaran subtotal, contingencia, total y financiamiento. Si no se obtienen antes del envio, debe declararse al asesor que el presupuesto esta completo como estimacion, pero aun no cotizado. Esta verificacion no inventa cotizaciones ni convierte provisiones en tarifas oficiales.

## Comprobaciones Ejecutadas

- Compilacion limpia con LuaLaTeX/Biber: 85 paginas, sin errores ni citas/referencias indefinidas, sin advertencias de Biber. Salida de comprobacion: `%TEMP%/plan_tesis_cierre_asesor_20261117/`.
- Bibliografia compilada: 50 claves unicas, 21 de tipo articulo; no se duplicaron registros repetidos del auxiliar ni se contaron normas/libros como articulos. El tipo bibliografico no certifica revision por pares ni indexacion.
- Matriz externa [rev2](../../matriz_consistencia/Matriz_de_Consistencia_Tesis_Cables_Heterogeneidad_Termica_rev2.xlsx) comprobada con `python scripts/sincronizar_matriz_plan_rev2.py --check`: las tres hojas conservan el protocolo sincronizado. No se reescribio el Excel en esta ronda.
- Correccion puntual de H1 confirmada en el texto extraido y en la pagina PDF 70 (pagina impresa 59). La tabla se inspecciono sin recortes ni solapamientos.
- Se mantienen dos desbordamientos de URL bibliograficas inferiores a 0,2 pt, ya presentes antes; no se introdujeron cambios de margenes o tipografia para ocultarlos.
- El [PDF evaluado antes de este cierre](../old/plan_previo_cierre_asesor_20261117_9123017d.pdf), SHA-256 `9123017df6815b867c10145947f2d828b84fd7575cfda478c0c6e970f37644fc`, quedo preservado. Las puntuaciones de SlopTotal corresponden a esa version, no certifican el cumplimiento del informe del asesor y no se atribuyen a este PDF final.

## Alcance De Este Cierre

Se comprueba el diseno documental del plan, no la ejecucion satisfactoria de la investigacion. El entorno exacto versionado, las fichas y manifiestos de ejecucion, las corridas de confirmacion y la aplicacion del juicio experto siguen siendo productos de las fases previstas; no se presentan como ya realizados. Tampoco se certifica el cumplimiento literal de toda la Guia N. 1 ni una validacion experimental de campo.

La salvedad presupuestaria y la ratificacion del asesor deben mantenerse visibles al remitir la version corregida.