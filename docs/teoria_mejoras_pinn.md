# Formulaciones PINN implementadas y evidencia de selección

Este documento describe el código activo en `Benchmarks/pinn.py`. Los informes
previos de 64×4, 128×5 y destilación son antecedentes; no constituyen una
comparación común cuando difieren los datos, las pérdidas o la referencia FEM.
Los resultados completos se consultan en el [informe interno](../Benchmarks/INFORME_INTERNO.md).

## Ecuación y representación

Se resuelve −div(k grad T) = Q. Para k espacial variable se incluye grad(k)·grad(T).
En interfaces exactas se exige continuidad de temperatura y de flujo normal.
En el suelo exterior a los cables Q = 0; cada contorno circular recibe una
potencia lineal, cuya unidad W/m se convierte a flujo W/m² dividiendo por 2πr.

| Variante | Implementación | Ventaja que se evalúa | Limitación |
|---|---|---|---|
| Directa | MLP tanh para el campo completo. | Base simple para ablación. | Puede fallar al representar fuentes pequeñas en un dominio grande. |
| Enriquecida | Fondo de fuentes e imágenes más corrección MLP. | Incorpora la estructura de una solución homogénea. | El fondo no satisface la PDE completa si k varía; deben conservarse sus derivadas. |
| Conservación integral | Añade residuo del balance de energía. | Detecta y penaliza déficit global. | Un balance correcto no garantiza un campo correcto. No es por sí sola cPINN. |
| Redes por material | Una MLP por estrato; continuidad de T y k∂T/∂n. | Permite derivadas laterales distintas en saltos exactos. | Requiere declarar la geometría de interfaz y comprobar ambas trazas. |
| Multipolar | Tres órdenes armónicos por cable más MLP. | Representa variaciones angulares e interacción próxima. | Más incógnitas; sensibilidad persistente en heterogeneidad compleja. |
| Electrotérmica | Potencias individuales aprendidas con R(Tc); corriente aprendida en ampacidad. | Consistencia entre generación y temperatura actual. | Residuos eléctricos y térmicos deben aprobarse conjuntamente. |

La base de residuos y diferenciación automática se relaciona con Raissi et al.
(2017: 4–5). La descomposición por material se contrasta con Shukla et al.
(2021: 1, 5–6, versión arXiv). El fondo y el enriquecimiento multipolar son
adaptaciones del artefacto; no se atribuye su mejora a un artículo que no
evaluó estos casos.

## Selección finita y controles

C01–C08 varían balance, peso de PDE, anchura, profundidad, tasa de Adam y
densidad de puntos. Se mantiene un caso y la semilla 5 durante la exploración.
C05 obtiene el menor error de esa campaña; C04 ofrece un compromiso compacto
con 609 parámetros, frente a 8577 de C05. La transferencia y la sensibilidad
se observan con semillas 11, 23 y 37; no prueban un óptimo universal.

El control multipolar conserva arquitectura y presupuesto para distinguir
su efecto del aumento de iteraciones. La red global de interfaz se compara
con subredes. Los pesos fijos se motivan por la competencia entre gradientes
documentada por Wang et al. (2020: 9–10), pero no implementan su algoritmo
adaptativo de ponderación. Las variantes débiles, mixtas y fronteras exactas
están en la discusión bibliográfica y no se presentan como ejecutadas.

La colocación sí incorpora una adaptación inspirada en RAD, comparada con
gradiente térmico y renovación aleatoria (Wu et al., 2022: 6–8). Conserva
la mitad de la nube geométrica inicial y redistribuye la otra mitad en dos
pasos Adam; no aumenta el total ni adapta las fronteras. La puntuación por
residuo conserva grad(k)·grad(T). Un gradiente térmico alto puede ser correcto
y no garantiza que priorizarlo mejore el resultado. El [expediente específico](../Benchmarks/MUESTREO_ADAPTATIVO.md)
presenta los 18 entrenamientos, seis controles, comparaciones entre semillas
y gráficos. Se archivan estados intermedios, candidatos y probabilidades;
la mejora se decide mediante métricas externas y aceptación completa.

## Acoplamiento eléctrico

Cada conductor satisface Pj = I²R20[1 + α(Tcj − 20)], con α = 0,00393 K⁻¹.
La ley y el coeficiente del cobre se contrastan con CIGRÉ WG B1.56 (2022: 132).
La PINN aprende potencias individuales; FEniCSx resuelve su actualización
mediante respuesta térmica, punto fijo y comprobación matricial independiente.
En ampacidad solo el conductor más caliente se fija a 90 °C.

La configuración inicial acoplada usa 32×3, 1200 pasos Adam, hasta 1600
iteraciones L-BFGS y peso eléctrico 100. Los fallos heterogéneos motivan
una campaña explícita con peso 1000 y mayor presupuesto; se conserva un
control de presupuesto. Toda elección posterior debe mostrar sus resultados
por semilla y distinguir ajuste sobre el propio caso de generalización.

La aceptación exige NRMSE ≤ 5 %, error del incremento térmico ≤ 5 %, balance
≤ 2 % y residuo eléctrico ≤ 0,1 %. En ampacidad se añade error de corriente
≤ 5 % y distancia a la temperatura límite ≤ 0,1 K. No se eliminan semillas
fallidas para aparentar robustez; solo se filtran al informar una corriente
aceptada, junto con el denominador total.

## Reproducibilidad y alcance

Los puntos de evaluación son comunes e independientes del entrenamiento.
Se conservan JSON físicos, configuraciones, pesos, historiales, campos y
fuentes por huella. La revisión distingue el código de entrenamiento del
código de reevaluación, y registra las limitaciones de procedencia de los
primeros pilotos. Los tiempos con procesos concurrentes no prueban aceleración.

La verificación sigue la distinción de CIGRÉ WG B1.87 (2025: 93–99). El
modelo resuelve suelo 2D y reconstruye radialmente el conductor; no verifica
un campo interno 2D de cada capa ni pérdidas AC completas. FNO y PINO no
forman parte de los resultados PINN de la tesis.

## Referencias

Las referencias completas y las versiones consultadas se encuentran al final
de cada cuaderno y del informe interno. Las nuevas fuentes primarias sobre
formulaciones están en `Benchmarks/references_selection.bib`; las fuentes de
cables y CIGRÉ están en `Tesis_LaTeX_Borrador_UNI/referencias.bib`.
