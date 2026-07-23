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
| 13 | 1:05 | Los cinco problemas específicos forman una cadena; cada producto habilita el paso siguiente. |
| 14 | 0:45 | El objetivo general tiene entradas físicas explícitas y resultados medibles. |
| 15 | 1:05 | Cada objetivo específico es un hito con producto verificable, no una actividad de programación. |
| 16 | 0:55 | La variable independiente principal es la configuración espacial de la conductividad térmica. |
| 17 | 0:45 | Los factores físicos se diferencian de los controles y de los hiperparámetros de entrenamiento. |
| 18 | 0:55 | Se medirán la respuesta térmica, el efecto operativo sobre ampacidad y la calidad del artefacto. |
| 19 | 0:55 | Las hipótesis son relaciones técnicas falsables y los umbrales son criterios iniciales del proyecto. |
| 21 | 1:10 | Recorrer una fila completa de la matriz para demostrar el alineamiento problema--objetivo--producto--variable--evidencia. |
| 22 | 1:00 | Cerrar con la cadena contexto, evidencia, brecha y contribución prevista. |
| 23 | 0:45 | Mostrar dominio del alcance declarando explícitamente qué no resolverá la tesis. |

## Respuestas técnicas previsibles

**¿Por qué usar una PINN si FEM ya representa geometrías heterogéneas?**  
FEM será una referencia de verificación, no un competidor que deba ser reemplazado. La investigación evaluará si una formulación PINN puede integrar la PDE, las interfaces, la conductividad variable y la estimación de ampacidad de manera reproducible. La utilidad y los límites deben demostrarse; no se presuponen.

**¿Por qué se habla de verificación y no de validación?**  
La tesis comprobará que el artefacto implementa correctamente la formulación mediante soluciones analíticas o manufacturadas, FEM convergente y casos publicados. No habrá mediciones independientes en una instalación real; por ello, no corresponde afirmar validación experimental en campo.

**¿Cuál es exactamente la variable independiente?**  
La representación espacial de la conductividad térmica del entorno, operacionalizada mediante contraste, patrón, extensión y proximidad. La geometría, las pérdidas y los contornos permanecen constantes en cada comparación pareada.

**¿Las épocas, capas o neuronas son variables independientes?**  
No son factores físicos del objeto. Forman parte de la búsqueda de diseño del artefacto y se registran para asegurar trazabilidad y reproducibilidad.

**¿Se pueden comparar directamente los porcentajes de la diapositiva 6?**  
Solo como evidencia de que el entorno térmico importa. Cada reducción pertenece a un cable, tensión, disposición y escenario distintos. No se debe construir un ranking de tecnologías ni extrapolar un porcentaje al Perú.

**¿De dónde provienen los umbrales de 5 %, 5 % y 2 %?**  
Son criterios iniciales de aceptación del proyecto para error de temperatura máxima, NRMSE y balance energético. No son constantes universales de las PINN. Se revisarán durante el piloto antes de observar los resultados finales y cualquier cambio quedará documentado.

**¿Qué aporta DSR?**  
DSR conecta el problema real y el vacío subyacente con la construcción, demostración, evaluación y comunicación de un artefacto. Además del prototipo, exige requisitos, evidencia, límites y principios de diseño reproducibles.

## Versión abreviada

Si el tiempo disponible se reduce a 12 minutos, omitir las diapositivas 10, 17 y 19 durante la exposición y explicar sus ideas al presentar las diapositivas 9, 16 y 18, respectivamente.
