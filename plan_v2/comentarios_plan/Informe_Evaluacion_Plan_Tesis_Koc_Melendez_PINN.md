**INFORME DE EVALUACIÓN DEL PLAN DE TESIS**

# SECCIÓN A. IDENTIFICACIÓN DEL PLAN DE TESIS

Título: Diseño y evaluación de una red neuronal informada por física para estimar la temperatura y la ampacidad de cables eléctricos enterrados en entornos heterogéneos.

Tesistas: Luis Enrique Koc Góngora y Herbert Antonio Meléndez García. | Asesor: Samuel Oporto.

Área: IA científica, modelado computacional y sistemas eléctricos de potencia. Contexto: configuraciones representativas de cables XLPE enterrados, en sección transversal 2D; desarrollo computacional en FIIS-UNI y referencias documentadas.

Problema principal: cuantificar de forma reproducible cómo la heterogeneidad espacial de la conductividad térmica del suelo y los materiales de relleno modifica el campo térmico, la temperatura máxima y la ampacidad frente a una representación homogénea. Solución: artefacto PINN 2D con ecuación de conducción, condiciones de contorno e interfaces, comparado con referencias analíticas, manufacturadas, FEM y casos publicados.

Fecha de evaluación: 13 de noviembre de 2026.

# SECCIÓN B. RESUMEN EJECUTIVO

El plan presenta una propuesta profesionalizante técnicamente pertinente, con un problema de ingeniería delimitado, evidencia contextual del sector eléctrico peruano y una brecha explícita. Destacan la separación entre objeto físico y artefacto, la formulación estacionaria 2D, la continuidad de temperatura y flujo en interfaces, la comparación pareada heterogéneo–homogéneo y la decisión expresa de distinguir verificación numérica de validación experimental de campo.

Las principales debilidades son corregibles, pero afectan la evaluación: (i) umbrales contradictorios entre HE2 y la matriz; (ii) especificación aún incompleta de pérdidas eléctricas y de la función R(T) que alimenta el cálculo de ampacidad; (iii) falta de detalle operativo sobre arquitectura PINN, muestreo, entrenamiento y ponderación de pérdidas; y (iv) ausencia de validación de campo, que restringe la generalización y el uso operativo directo. El nivel de madurez es bueno en formulación y diseño conceptual, intermedio en especificación reproducible de implementación. La contribución es moderada y situada, no un algoritmo nuevo.

# SECCIÓN C. VERIFICACIÓN DE ESTRUCTURA Y GUÍA N.º 1

El documento contiene carátula y datos generales (pp. 1–2), planteamiento (pp. 1–15), objetivos e hipótesis (pp. 16–22), marco teórico (pp. 23–36), metodología (pp. 37–47), administración (pp. 48–49), referencias (pp. 50–54) y matriz de consistencia (pp. 55–58). La Guía N.º 1 no fue adjuntada; por ello no es posible certificar el cumplimiento literal de sus pp. 13–14, 38–41 y 54–57. La matriz se evalúa como requisito del curso, no como exigencia expresa de la guía para esta modalidad.

| **Componente**          | **Estado**               | **Observación**                                                                                            |
| ----------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Preliminares            | Cumple                   | Autores, asesor, programa, lugar, duración y artefacto (pp. 1–2).                                          |
| Cap. I: problema        | Cumple                   | Diagnóstico, antecedentes, formulación, justificación y límites (pp. 1–15).                                |
| Cap. II: objetivos      | Cumple                   | Objetivo general, cuatro específicos, hipótesis y dimensiones (pp. 16–22).                                 |
| Cap. III: marco teórico | Cumple                   | Bases físicas, PINN, DSR y términos (pp. 23–36).                                                           |
| Cap. IV: metodología    | Cumple con observaciones | DSR, unidad de análisis, fuentes, métricas y fases; faltan especificaciones de implementación (pp. 37–47). |
| Cap. V: administración  | Parcial                  | Cronograma y financiamiento definidos; varios importes del presupuesto quedan por cotizar (pp. 48–49).     |
| Referencias y anexos    | Cumple                   | Referencias y matriz A.1–A.2 (pp. 50–58).                                                                  |

# SECCIÓN D. COHERENCIA Y TRAZABILIDAD

Problema–objetivos: correspondencia global adecuada. PG se traduce en OG; PE1–PE4 se relacionan respectivamente con OE1–OE4 y productos VD1–VD4 (pp. 11–17 y matriz pp. 55–58). OE1 especifica requisitos; OE2 construye/verifica; OE3 analiza escenarios; OE4 compara ampacidad. La secuencia es lógica y verificable.

Matriz: existe y es amplia. Sin embargo, A.2 presenta para HE2 límites de ΔTmáx <0,1 K y NRMSE <1,0 %, mientras que HE2 (p. 18) y la Tabla 4.2 (p. 46) establecen error de Tmáx y NRMSE ≤5 %. Debe unificarse la definición de error (absoluto/relativo) y el umbral. También debe precisarse cómo se determina el efecto mínimo relevante y la variabilidad entre semillas.

Objetivos–método–resultados: alineamiento general satisfactorio; VD1–VD4 son productos y resultados verificables. La hipótesis HE4 anticipa que la representación homogénea sobreestimará la ampacidad ante zonas locales de baja conductividad; debe quedar como hipótesis contrastable, explicitando la regla de homogenización (media aritmética, equivalente térmico u otra), no como resultado asumido. No se exige población estadística ni hipótesis adicionales por tratarse de casos numéricos DSR.

# SECCIÓN E. PUBLICACIONES CIENTÍFICAS

La bibliografía final contiene 42 referencias (pp. 50–54). Se identifican al menos 21 artículos académicos en revistas o actas; 12 corresponden a 2022–2026 y 9 son anteriores. Por tanto, supera el mínimo de 15 publicaciones y las publicaciones recientes predominan ligeramente dentro de ese conjunto. La Tabla 1.1 de antecedentes resume 19 fuentes, aunque incluye normas, métodos y documentos técnicos, que no deben contabilizarse automáticamente como artículos científicos.

La selección es pertinente para ampacidad, conducción térmica, PINN y DSR; combina trabajos recientes con fundamentos clásicos y normas IEC/IEEE/CIGRÉ. Debe reforzarse la comparación crítica de limitaciones y la conexión entre antecedentes y decisiones de diseño. Se observa pertinencia discutible de Van Dyke (2023), cuyo título versa sobre líneas aéreas, para sustentar usos de cables enterrados. No se verificaron externamente DOI, autenticidad ni indexación; por tanto, no se afirma validación bibliográfica independiente. Conviene revisar consistencia APA 7 y formato de editoriales/URLs.

# SECCIÓN F. METODOLOGÍA Y SOLUCIÓN DE IA

Fortalezas: enfoque cuantitativo, aplicado y tecnológico; DSR con siete fases; unidad de análisis definida como sección 2D; casos intencionales y comparaciones pareadas; fuentes analíticas, manufacturadas, FEM convergentes y publicadas; métricas térmicas, residuos, balance energético, ampacidad y robustez (pp. 37–47). La PDE estacionaria ∇·(k(x,y)∇T)+Q=0, la continuidad de temperatura/flujo y la pérdida compuesta están explícitas (pp. 41–42).

Aspectos por completar: (1) especificar modelo de pérdidas del cable, resistencia eléctrica en función de temperatura R(T), tratamiento de pérdidas AC, piel/proximidad y pérdidas de pantalla/dieléctricas cuando apliquen; (2) definir geometría y parámetros de cable, fuentes y condiciones de contorno de cada benchmark; (3) detallar arquitectura por material o estrategia de dominio, activaciones, optimizador, muestreo de colocación, normalización, inicialización, semillas y búsqueda de pesos de pérdida; (4) fijar convergencia FEM, tolerancias y procedimiento de bisección para Imáx. El plan reconoce el carácter estacionario y la ausencia de validación de campo (pp. 14–15, 44–46); esto es admisible si las conclusiones se restringen al dominio verificado.

# SECCIÓN G. INNOVACIÓN Y CONTRIBUCIÓN

Contribución MODERADA: integración y evaluación de PINN para el arreglo cable–entorno térmico multimaterial, con transmisión de temperatura/flujo en interfaces y análisis trazable de contraste, patrón, extensión y proximidad. El valor potencial es identificar cuándo la simplificación homogénea distorsiona Tmáx e Imáx. No se acredita novedad algorítmica general; la contribución profesionalizante reside en el artefacto, el protocolo reproducible y los principios de diseño derivados de la evidencia (pp. 11, 14, 31 y 39–40).

# SECCIÓN H. RÚBRICA DE EVALUACIÓN

| **N.º** | **Criterio**                             | **Máx.** | **Obtenido** | **Justificación**                                                                  |
| ------- | ---------------------------------------- | -------- | ------------ | ---------------------------------------------------------------------------------- |
| 1       | Estructura y cumplimiento de Guía N.º 1  | 2        | 1,5          | Estructura completa; guía no adjunta para contraste literal.                       |
| 2       | Planteamiento del problema               | 2        | 2,0          | Problema real, evidenciado y delimitado; riesgo operativo explicado.               |
| 3       | Coherencia problema–objetivos            | 2        | 1,5          | Trazabilidad lógica PG/PE–OG/OE; mejorar precisión de criterios.                   |
| 4       | Antecedentes y publicaciones científicas | 2        | 1,5          | Supera 15; buen núcleo técnico, pero requiere auditoría APA y pertinencia puntual. |
| 5       | Matriz de consistencia                   | 2        | 1,5          | Existe y es detallada; contradicción numérica de umbrales HE2.                     |
| 6       | Metodología                              | 2        | 1,5          | DSR y verificación pertinentes; faltan decisiones reproducibles de implementación. |
| 7       | Solución basada en IA                    | 2        | 1,5          | PINN y física bien formuladas; arquitectura y pérdidas eléctricas incompletas.     |
| 8       | Resultados esperados                     | 2        | 1,5          | VD1–VD4 e indicadores; criterios inconsistentes y validación de campo excluida.    |
| 9       | Innovación y contribución                | 2        | 1,5          | Contribución aplicada moderada; diferenciación frente a PINN/FEM debe concretarse. |
| 10      | Viabilidad y calidad formal              | 2        | 1,5          | 12 meses y equipo disponible; presupuesto parcial y riesgos de implementación.     |

**TOTAL: 15,5 / 20**

# SECCIÓN I. OBSERVACIONES Y RECOMENDACIONES DE MEJORA

Las siguientes correcciones prioritarias deben incorporarse antes de congelar el protocolo experimental.

1\. Unificar umbrales de HE2. Ubicación: Cap. II, p. 18; Cap. IV, p. 46; Anexo 1, p. 58. La matriz fija <0,1 K y <1 %, pero texto/Tabla 4.2 usan ≤5 %. Definir una única versión, diferenciar error absoluto y relativo y actualizar hipótesis, matriz y criterios. Prioridad: ALTA.

2\. Formalizar el cálculo de ampacidad. Ubicación: Cap. IV, pp. 41–42. La condición Imáx depende de pérdidas y Tmáx; falta especificar R(T), pérdidas AC aplicables y modelo de fuente. Incorporar ecuaciones, parámetros y supuestos por tipo de cable, sin duplicar correcciones de efecto piel. Prioridad: ALTA.

3\. Completar protocolo de entrenamiento PINN. Ubicación: Cap. III, pp. 28–30; Cap. IV, pp. 39–42. Definir arquitectura, estrategia multimaterial/interfaz, activaciones, optimizador, muestreo, escalamiento, semillas y ajuste de pesos de pérdida. Prioridad: ALTA.

4\. Definir con precisión la homogenización. Ubicación: Cap. II, p. 18; Cap. IV, pp. 42–45; Anexo, pp. 57–58. Especificar cómo se obtiene keq y construir pares homogéneos comparables; justificar la expectativa de sobreestimación de ampacidad. Prioridad: ALTA.

5\. Precisar criterios de convergencia y referencia FEM. Ubicación: Cap. IV, pp. 42–46. Documentar refinamiento de malla, tolerancias, solver, independencia de malla y compatibilidad de condiciones de borde; separar error de referencia del error PINN. Prioridad: ALTA.

6\. Operacionalizar relevancia térmica. Ubicación: Cap. II, pp. 18–21; Cap. IV, pp. 44–46. Fijar antes del análisis el efecto mínimo relevante, regla para variabilidad entre semillas y umbrales de clasificación de VD3/VD4. Prioridad: ALTA.

7\. Acotar la interpretación externa. Ubicación: Cap. I, pp. 14–15; Cap. IV, pp. 44–46. Mantener explícito que se verifica con casos numéricos y publicados, sin medición de campo; no presentar el artefacto como validado para operación real. Prioridad: ALTA.

8\. Fortalecer antecedentes comparativos. Ubicación: Cap. I, pp. 8–11 y Referencias, pp. 50–54. Añadir una síntesis comparativa de métodos, datos, geometrías, límites y brecha; revisar la pertinencia de Van Dyke (2023) y correspondencia cita–referencia. Prioridad: MEDIA.

9\. Completar presupuesto y riesgos. Ubicación: Cap. V, pp. 48–49. Cotizar impresión, conectividad/energía, trámites, movilidad y publicación; estimar costo total y prever riesgos de cómputo, convergencia y disponibilidad de casos. Prioridad: MEDIA.

10\. Alinear cronograma e hitos. Ubicación: Cap. V, p. 48. Precisar evidencias de los hitos declarados cumplidos a septiembre de 2026 y reprogramar actividades pendientes respecto de la fecha de evaluación. Prioridad: MEDIA.

# SECCIÓN J. DICTAMEN FINAL

**APROBADO CON OBSERVACIONES**

El plan es pertinente, coherente en su estructura general y viable como proyecto profesionalizante de investigación y desarrollo de un artefacto PINN 2D. La nota de 15,5/20 refleja fortalezas claras en el diagnóstico, la delimitación, la formulación física y la secuencia DSR, junto con deficiencias específicas que impiden considerarlo listo para ejecución sin ajustes. Las correcciones críticas son unificar los umbrales de aceptación, cerrar el modelo de pérdidas/ampacidad y completar el protocolo reproducible de entrenamiento y verificación. La ausencia de validación experimental no invalida el alcance declarado, pero obliga a limitar las conclusiones al dominio numérico verificado.

Nota final: 15,5 / 20. La calificación aplica exclusivamente los diez criterios de la rúbrica proporcionada; no se ha establecido un umbral adicional de aprobación.