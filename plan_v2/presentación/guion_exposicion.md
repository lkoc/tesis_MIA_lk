# Guion de exposición del plan de tesis

Duración objetivo: 17 minutos. Las diapositivas 2, 3, 11, 20, 24 y 25 se entregan como respaldo, pero no se exponen.

## Secuencia oral

| Diapositiva | Tiempo | Mensaje que debe quedar claro |
|---:|---:|---|
| 1 | 0:25 | Se diseñará y evaluará un artefacto PINN 2D para un problema térmico de cables enterrados. |
| 4 | 0:55 | La situación ocurre en el SEIN e involucra a operadores, transmisores, generadores y usuarios. |
| 5 | 1:00 | La nueva generación y las restricciones urbanas incrementan el uso de infraestructura enterrada. |
| 6 | 1:10 | Los cuatro estudios no se comparan entre sí en amperios; cada par muestra la sensibilidad de una instalación a su entorno térmico. |
| 7 | 0:55 | El problema subyacente no es solo el suelo: también es la forma homogénea con la que se representa. |
| 8 | 1:00 | El objeto real es el arreglo físico cable--entorno; la PINN será el artefacto, no el objeto. |
| 9 | 0:45 | Las propiedades se capturan mediante normas, planos y casos documentados; no habrá campaña de sensores ni validación de campo. |
| 10 | 0:45 | La instancia modelada relaciona el mapa de conductividad con el campo térmico y el punto caliente. |
| 12 | 1:00 | El problema técnico se limita a construir y evaluar una formulación integrada, verificada y reproducible. |
| 13 | 1:05 | Los cuatro problemas específicos forman una cadena; la sistematización posterior es una actividad metodológica de DSR, no un quinto problema. |
| 14 | 0:45 | El objetivo general se despliega en cuatro paquetes discretos VI1--VI4 y cuatro productos discretos VD1--VD4. |
| 15 | 1:10 | Cada objetivo produce un objeto que se conserva como entrada heredada de la etapa siguiente; además, cada nueva etapa puede requerir otro objeto de entrada. |
| 16 | 1:05 | VI1 es el paquete inicial; VI2 combina VD1 y referencias; VI3 combina VD2 y escenarios; VI4 combina VD3 y representaciones pareadas. |
| 17 | 0:45 | La conductividad, el contraste, la extensión y la proximidad son parámetros físicos del caso; se distinguen de controles, decisiones numéricas y metadatos. |
| 18 | 1:05 | VD1--VD4 son objetos discretos con estados declarados: especificación, artefacto verificado, expediente térmico y expediente de ampacidad; las métricas son sus indicadores. |
| 19 | 0:55 | Las hipótesis son relaciones técnicas falsables y los umbrales son criterios iniciales del proyecto. |
| 21 | 1:10 | Recorrer una fila completa de la matriz para demostrar el alineamiento problema--objetivo--producto--variable--evidencia. |
| 22 | 1:00 | Cerrar con la cadena contexto, evidencia, brecha y contribución prevista. |
| 23 | 0:45 | Mostrar dominio del alcance declarando explícitamente qué no resolverá la tesis. |

## Respuestas técnicas previsibles

**¿Por qué usar una PINN si FEM ya representa geometrías heterogéneas?**  
FEM será una referencia de verificación, no un competidor que deba ser reemplazado. La investigación evaluará si una formulación PINN puede integrar la PDE, las interfaces, la conductividad variable y la estimación de ampacidad de manera reproducible. La utilidad y los límites deben demostrarse; no se presuponen.

**¿Por qué se habla de verificación y no de validación?**  
La tesis comprobará que el artefacto implementa correctamente la formulación mediante soluciones analíticas o manufacturadas, FEM convergente y casos publicados. No habrá mediciones independientes en una instalación real; por ello, no corresponde afirmar validación experimental en campo.

**¿Cuáles son exactamente las variables independientes?**
Son cuatro paquetes categóricos y discretos, uno por etapa. VI1 es el paquete inicial \(P_1\). VI2 es \(\{\mathrm{VD1},R_2\}\): la especificación trazable heredada más un conjunto de referencias. VI3 es \(\{\mathrm{VD2},E_3\}\): el artefacto verificado más un catálogo de escenarios. VI4 es \(\{\mathrm{VD3},C_4\}\): el expediente térmico más un catálogo de representaciones pareadas. Así se mantienen pocas VI sin perder las entradas nuevas de cada bloque.

**¿Qué estados o realizaciones pueden tomar las VI y VD?**
VI1 puede ser básica, completa o completa documentada; \(R_2\), analítico/manufacturado, FEM, publicado o mixto; \(E_3\), base, estratificado, localizado, mejorado, seco o multimaterial; y \(C_4\), par equivalente, conservador u optimista. VD1 puede ser incompleta, completa pendiente o trazable aceptada; VD2, rechazada, aceptada con restricciones o aceptada; VD3, inconclusa, sin efecto relevante, con efecto relevante no crítico o crítica; y VD4, sin discrepancia relevante, con reducción, con aumento o inconclusa/fuera del dominio.

**¿Por qué una VD también aparece en la VI siguiente?**
Porque ambas son objetos discretos de la construcción. La VD es el producto del bloque actual y se convierte, sin perder su identidad y versión, en la parte heredada del paquete de entrada siguiente: \(\mathrm{VD1}\equiv\mathrm{VI}_{2,h}\), \(\mathrm{VD2}\equiv\mathrm{VI}_{3,h}\) y \(\mathrm{VD3}\equiv\mathrm{VI}_{4,h}\).

**¿Entonces \(k(x,y)\), el contraste, la extensión y la proximidad no son variables independientes?**
No. Son parámetros físicos que caracterizan las realizaciones del catálogo de escenarios \(E_3\) y las comparaciones \(C_4\). Se documentan y pueden recorrerse en una malla de sensibilidad, pero no son objetos de transición de la construcción del artefacto.

**¿Las épocas, capas o neuronas son variables independientes?**  
No. Son decisiones numéricas de diseño que se fijan después del piloto y se registran para asegurar trazabilidad y reproducibilidad.

**¿El estudio mantiene un diseño factorial?**
No como diseño de variables independientes. Se utiliza un diseño discreto de objetos para combinar VD2 con realizaciones de \(E_3\), y VD3 con realizaciones de \(C_4\). Una malla de cobertura recorre parámetros físicos como \(C_k\), \(f_h\) y \(d_h/D\); sirve para sensibilidad física, no para redefinirlos como VI ni para análisis factorial estadístico.

**¿Se pueden comparar directamente los porcentajes de la diapositiva 6?**  
Solo como evidencia de que el entorno térmico importa. Cada reducción pertenece a un cable, tensión, disposición y escenario distintos. No se debe construir un ranking de tecnologías ni extrapolar un porcentaje al Perú.

**¿De dónde provienen los umbrales de 5 %, 5 % y 2 %?**  
Son criterios iniciales de aceptación del proyecto para error de temperatura máxima, NRMSE y balance energético. No son constantes universales de las PINN. Se revisarán durante el piloto antes de observar los resultados finales y cualquier cambio quedará documentado.

**¿Qué aporta DSR?**  
DSR conecta el problema real y el vacío subyacente con la construcción, demostración, evaluación y comunicación de un artefacto. Además del prototipo, exige requisitos, evidencia, límites y principios de diseño reproducibles.

**¿Por qué la sistematización no es un quinto problema u objetivo específico?**
Porque solo puede ejecutarse cuando el artefacto ya fue construido y evaluado y existe evidencia que integrar. Es una actividad de síntesis y comunicación del ciclo DSR; sus productos son el procedimiento reproducible, la guía, el repositorio depurado, los límites y los principios de diseño derivados.

## Versión abreviada

Si el tiempo disponible se reduce a 12 minutos, omitir las diapositivas 10, 17 y 19 durante la exposición y explicar sus ideas al presentar las diapositivas 9, 16 y 18, respectivamente.
