# Resumen de cambios aplicables para levantar las observaciones

**Fecha de revisión:** 13 de noviembre de 2026  
**Documentos contrastados:** [informe de evaluación](./Informe_Evaluacion_Plan_Tesis_Koc_Melendez_PINN.md) y [archivo principal del plan](./plan_tesis_cables_pinn.tex).

## Conclusión ejecutiva

El plan ya incorpora varias correcciones señaladas por el informe: formula la resistencia del conductor como función de la temperatura, acota las pérdidas eléctricas y reconoce que no habrá validación experimental de campo. También desarrolla criterios por fases, registra campañas de entrenamiento y amplía la comparación de antecedentes. Por ello, no conviene aplicar las observaciones como si el texto estuviera intacto: algunas están atendidas y otras requieren cerrar definiciones que todavía son ambiguas o no reproducibles.

Las prioridades antes de congelar el protocolo son: **aclarar la jerarquía de umbrales; cerrar el modelo de pérdidas y la búsqueda de ampacidad; especificar el protocolo reproducible de la PINN; definir la homogenización y la verificación FEM; y fijar de antemano qué cuenta como efecto térmico relevante.**

> El archivo principal carga capítulos y anexos separados. Las modificaciones de contenido deben aplicarse en los `.tex` incluidos, no concentrarse en el preámbulo de `plan_tesis_cables_pinn.tex`.

## Observaciones y acciones propuestas

| # | Prioridad | Estado en el plan actual | Cambio aplicable para levantar la observación |
|---|---|---|---|
| 1 | Alta | **Parcial; requiere aclaración.** La metodología distingue un límite general de HE2 (error relativo de \(T_{\max}\) y NRMSE ≤ 5 %) de criterios más estrictos por fase: F2 (< 0,1 K y NRMSE < 1 %) y F3 (< 0,5 K y NRMSE < 2,5 %). La matriz también presenta el límite general y los límites por fase. | Indicar expresamente si los límites por fase son puertas de aceptación más estrictas que se deben cumplir además del límite general, o si lo sustituyen. Mantener una definición única de cada métrica: \(\Delta T_{\max}\) absoluto en K, error relativo normalizado por elevación térmica y NRMSE. Alinear HE2, Tabla de criterios y Anexo 1 con esa jerarquía, sin mezclar “error absoluto” con “error porcentual”. |
| 2 | Alta | **Parcialmente atendida.** Ya están \(R(T)\), la fuente Joule \(I^2R(T)\), el uso de resistencia AC y la aclaración de que los efectos piel/proximidad no se calculan aparte. | Completar una ficha por benchmark con \(R_{20}\), \(\alpha_{20}\), límite térmico, geometría, corriente y fuente volumétrica equivalente. Declarar explícitamente si se excluyen o incorporan pérdidas dieléctricas y de pantalla; si se excluyen, mantenerlo como límite del estudio. Evitar contabilizar dos veces piel/proximidad cuando el dato de entrada ya sea resistencia AC. Precisar el criterio de parada y la tolerancia de la búsqueda de \(I_{\max}\). |
| 3 | Alta | **Parcialmente atendida.** Se describen la pérdida compuesta, el ajuste de pesos y el registro de configuración y semillas; el cronograma consigna campañas y arquitecturas evaluadas. No queda consolidado en el texto un protocolo completo de entrenamiento. | Incorporar una tabla de configuración: arquitectura (capas y neuronas), activación, optimizador y secuencia de entrenamiento, normalización, inicialización, muestreo de colocación por dominio/interfaz, número de semillas, búsqueda de pesos de pérdida y regla para seleccionar el modelo. Definir qué configuración se congela para las evaluaciones F2–F4 y cómo se registrará cada ejecución. |
| 4 | Alta | **Parcial; falta la regla operativa.** Se plantean pares heterogéneo–homogéneo y alternativas equivalentes, conservadoras u optimistas, pero no queda definida la derivación de \(k_{eq}\). HE4 se presenta como expectativa contrastable. | Definir la regla de homogenización (por ejemplo, promedio u otra equivalencia térmica justificada), sus datos de entrada y el tratamiento de interfaces. En cada par, documentar qué propiedades se conservan y cuál es la única representación que cambia. Mantener HE4 como hipótesis que puede confirmarse o rechazarse, no como resultado asumido. |
| 5 | Alta | **Parcialmente atendida.** El plan exige referencias FEM convergentes y registra malla y tolerancia como metadatos, pero no se explicita una regla cuantitativa completa de convergencia. | Especificar solver y versión, secuencia de refinamiento de malla, magnitudes observadas, tolerancia de convergencia e independencia de malla. Documentar condiciones de contorno y parámetros comunes con la PINN. Separar el error de la referencia FEM del error PINN y describir cómo se verifica FEM frente a soluciones analíticas o manufacturadas. |
| 6 | Alta | **Parcial; falta fijar el umbral de relevancia.** Se definen métricas, estados para VD3/VD4 y el requisito de que la variabilidad sea menor que el efecto relevante, pero no se concreta el efecto mínimo ni la regla de agregación entre semillas. | Establecer antes del análisis el efecto mínimo relevante en temperatura y ampacidad, la estadística de resumen entre semillas y el criterio de variabilidad/incertidumbre. Traducir esas reglas a las categorías “sin efecto relevante”, “efecto relevante” y “condición crítica” de VD3/VD4, con ejemplos de decisión y casos inconclusos. |
| 7 | Alta | **Atendida; preservar el límite.** El plan excluye mediciones directas de campo, define la evaluación como verificación numérica y restringe la generalización a los casos evaluados. | Mantener esta limitación en metodología, alcances y conclusiones. No describir el prototipo como validado para operación real ni extrapolar fuera de las geometrías, materiales, fuentes y contornos verificados. |
| 8 | Media | **Parcialmente atendida.** El capítulo presenta una cronología de 23 antecedentes y síntesis temáticas sobre ampacidad, heterogeneidad, PINN y DSR. La cita a Van Dyke no aparece en los capítulos `.tex` actuales, aunque la entrada permanece en la bibliografía `.bib`. | Añadir una comparación crítica compacta (método, geometría/datos, limitaciones y consecuencia para el diseño propuesto) y vincular cada brecha con una decisión metodológica. Revisar la correspondencia entre citas y referencias y el formato APA 7. Confirmar si la entrada de Van Dyke sigue citada en otra fuente compilada; si no, retirarla o justificar su pertinencia antes de depurar referencias. |
| 9 | Media | **Parcial; presupuesto abierto.** Los rubros están identificados, pero varios montos y el total figuran como “por confirmar”. | Obtener cotizaciones o tarifas para impresión, conectividad/energía, trámites, movilidad y publicación; completar el total y declarar la contingencia o el supuesto usado cuando un monto dependa de la revista. Añadir riesgos de cómputo, convergencia y disponibilidad de referencias con acciones de mitigación. |
| 10 | Media | **Parcialmente atendida.** Se consignan campañas ejecutadas en octubre y los hitos H1–H6, pero el cronograma debe reflejar su estado a la fecha de evaluación y distinguir lo completado de lo pendiente. | Actualizar el estado de H1/H2 y la ronda E al 13 de noviembre de 2026; enlazar cada hito ya cumplido con su evidencia (especificación, benchmarks, informes de campañas y confirmación de semillas). Reprogramar hitos pendientes si las fechas reales difieren de las previstas y mantener consistencia entre texto, Gantt y productos VD1–VD4. |

## Orden recomendado de trabajo

1. **Cerrar criterios de aceptación:** resolver la jerarquía de umbrales y fijar el efecto mínimo relevante y su variabilidad.
2. **Cerrar la especificación física:** completar los supuestos de pérdidas, entradas por benchmark y cálculo de \(I_{\max}\).
3. **Congelar el protocolo numérico:** documentar entrenamiento PINN, homogenización y verificación FEM.
4. **Actualizar trazabilidad y viabilidad:** reforzar comparación bibliográfica, presupuesto y estado real del cronograma.
5. **Conservar los límites ya bien declarados:** mantener la ausencia de validación de campo y restringir las conclusiones al dominio verificado.

## Archivos fuente a modificar

- [Metodología](./capitulos/04_metodologia.tex): pérdidas, entrenamiento, homogenización, FEM, umbrales y relevancia térmica.
- [Objetivos e hipótesis](./capitulos/02_objetivos.tex): consistencia de HE2–HE4 y clasificación de resultados.
- [Matriz de consistencia](./capitulos/anexos/anexo_1_matriz_consistencia.tex): alineación final de hipótesis, indicadores y umbrales.
- [Antecedentes y alcances](./capitulos/01_planteamiento_del_problema.tex): comparación crítica, citas y límites externos.
- [Administración](./capitulos/05_administracion.tex): cronograma actualizado, hitos y presupuesto.

Este resumen identifica acciones; no modifica el contenido del plan ni sustituye la aprobación del asesor sobre los criterios experimentales.
