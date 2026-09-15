# Auditoría de tesis, datos y artefacto térmico

Fecha: 10 de septiembre de 2026. El estado previo se conserva en
`auditoria/estado_inicial` y en el registro inicial de Git. Los cambios que ya
existían al comenzar el trabajo se conservaron. Este informe distingue una
auditoría técnica de una aprobación académica o de una validación de campo.

## Fuentes y criterios revisados

Se contrastaron el Plan, las guías locales de redacción y estructura, la copia
de la Guía UNI N.° 2, el borrador LaTeX, la carpeta `docs`, los ejemplos, la
información bibliográfica y el catálogo de dieciséis instancias del Plan.
Los textos extraídos, las páginas visuales de la guía y el inventario de
código se conservan en `auditoria/`.

Se consultaron directamente la
[entrega del Plan en Drive](https://drive.google.com/file/d/1nxj1wEjYxxdAGLyGREdUeDWGj5mNzIrv/view)
y la [carpeta de entrega G91](https://drive.google.com/drive/folders/1FJToqarrDDXG1JfYUV3FpsK-j2Y9_klr).
El informe de datos entregado se obtuvo de
[este archivo](https://drive.google.com/file/d/1Vyrnu-B8ezAoO4tkvUpljHwMwqpLfQeO/view).
Las transcripciones están en `drive_plan_entregado.txt` y
`drive_datos_entregados.md`. No se sustituyeron las entregas históricas.

La entrega de Drive y el Plan establecen una investigación DSR sobre el
artefacto, con exactitud térmica, conservación, heterogeneidad y ampacidad.
El borrador conservaba esa estructura, pero metodología ejecutada, resultados,
discusión y conclusiones eran instrucciones pendientes. Los ejemplos no
aportaban una cadena uniforme de datos, malla, entrenamiento y comparación.

### Requisitos UNI y redacción

| Criterio | Decisión aplicada |
|---|---|
| Estructura profesionalizante | Se mantiene el esquema detallado de cuatro capítulos de las pp. impresas 12–13 de la Guía N.° 2. |
| Inconsistencia de la guía | El diagrama de la p. 8 muestra cinco capítulos. Se prioriza el esquema detallado específico; no se afirma que todas las láminas coincidan. |
| Hipótesis | Las pp. 21–22 distinguen la evaluación profesionalizante de la contrastación de hipótesis. Se conservan criterios computacionales y objetivos. |
| Propuesta y desarrollo | Presentación, desarrollo, análisis de resultados, discusión e impacto permanecen en el capítulo III. |
| Estilo | Redacción impersonal, afirmaciones acotadas, unidades, fuentes y separación de evidencia de expectativas. |
| Fórmulas | Numeración, referencia en el texto y leyenda inferior mediante el entorno `formula`. |
| Tablas y figuras | Se generan desde registros; incluyen fuente y referencia en el texto. |
| Referencias | APA mediante biblatex/biber; se distingue artículo, norma, informe y prepublicación. |
| Documentos personales | Dedicatoria, agradecimientos y constancias requieren información auténtica de los autores; no se inventan. |

La ficha general de evaluación incluye apartados de hipótesis que pertenecen
al esquema académico. No se trasladan automáticamente al esquema
profesionalizante. La decisión de estructura se documenta para la revisión
del asesor, sin convertirla en una exigencia de autorización para corregir
el borrador.

## Hallazgos técnicos y correcciones

| Hallazgo | Efecto | Corrección o clasificación |
|---|---|---|
| Fondo analítico desconectado de autograd por defecto | Se omitía su contribución a la PDE cuando variaba k. | Se conserva el grafo en el modelo residual; la batería verifica el operador completo. |
| Neumann tratado como derivada, sin signo o conductividad | Se confundían K/m y W/m². | El residuo usa `-k grad(T)·n - q`. |
| Robin convertido en temperatura prescrita en `train_custom` | Se resolvía otra condición de contorno. | Se incorpora flujo más intercambio convectivo; prueba con T distinta de T∞. |
| Neumann omitido en `train_custom` | Parte de la frontera no contribuía al entrenamiento. | Se implementa y se prueba un flujo no nulo. |
| Agrupación de todos los contornos bajo el peso Dirichlet | Los pesos Neumann/Robin declarados no actuaban. | Se separan los grupos de pérdida por tipo de frontera. |
| Derivadas segundas de campos afines | Autograd podía perder la dependencia necesaria para derivar una constante. | Se conserva una dependencia nula; prueba de laplaciano cero. |
| Misma derivada a ambos lados de un salto | Se penalizaba `(k1-k2) dT/dr`, suprimiendo transmisión de calor. | Trazas separadas en el entrenador previo; redes por material en la batería verificada. |
| Muestreo de interfaz con dispersión radial | Algunos puntos no estaban en la interfaz declarada. | Radios exactos y evaluación de trazas laterales explícitas. |
| Conductividad escalar distinta de la tensorial | Referencia analítica y PINN podían usar otro suelo. | Se unifican las transiciones suaves; prueba NumPy/PyTorch. |
| Iteración R(T) con una temperatura para todos los cables | Se atribuía el punto caliente a todos los conductores. | Actualización individual, pérdida dieléctrica incluida en potencia total y error explícito si no converge. |
| Entrada de varios cables reducida silenciosamente al primero | Se omitían fuentes sin avisar. | El ejecutor de un cable rechaza esa entrada; la batería usa el conjunto completo. |
| Normalización FNO por máximo de cada caso | Casos con distintas magnitudes podían recibir entradas equivalentes. | Escalas físicas comunes para k y potencia. |
| Pérdida PDE FNO sin vínculo con la predicción | Una ejecución supervisada podía denominarse PINO. | Se rechaza `w_pde > 0` mientras no exista esa formulación; FNO no sustenta resultados PINN. |
| Metadatos y modelos FEM históricos incompatibles | Una diferencia de Tmax se presentaba como verificación. | Se reconstruye una batería con datos y evaluación comunes. |
| Errores relativos calculados con Celsius absoluto | El porcentaje dependía del origen de temperatura. | Normalización con el incremento respecto al ambiente. |

El inventario inicial comprendió 79 archivos Python y 25 085 líneas en núcleo,
ejemplos y scripts. La revisión combinó lectura de formulaciones y rutas de
ejecución, búsqueda de patrones, pruebas y ejecuciones independientes. No
constituye una prueba formal de corrección de cada línea ni valida módulos
fuera de la batería estacionaria, como el acoplamiento transitorio completo.

### Resultados históricos y ejemplos

`examples/kim_2024_154kv_optim_C/run_fem_C.py` emplea scikit-fem y una
representación simplificada; su nombre no lo convierte en FEniCS.
`fem_fenicsx_colab.py` sí usa FEniCSx, pero sus capas, temperaturas de borde,
frecuencia y pérdidas difieren de las del PINN histórico. Los valores cercanos
a 70–71 °C no constituyen una verificación común mientras difieran esos datos.

Los informes de comparación 64×4, 128×5, destilación y otros experimentos se
conservan como antecedentes. No se trasladan a la tesis como evidencia de
superioridad ni se mezclan con la batería actual. `examples/` mantiene los
datos originales utilizados para reconstruir los escenarios controlados.

## Bibliografía y cobertura del Plan

El archivo `doi_metadata.json` registra consultas de metadatos, incluidos
errores HTTP. Un DOI no resuelto por Crossref no se clasifica automáticamente
como falso: el artículo de Chen aparece en la plataforma editorial SciOpen.
Las respuestas 429 representan una restricción de acceso, no evidencia
bibliográfica negativa.

| Referencia o registro | Resultado de revisión |
|---|---|
| Kim | Geothermics 125, 103151, año 2025; DOI `10.1016/j.geothermics.2024.103151`. Los nombres de carpetas con 2024 son históricos. |
| Coeficiente α de Kim | El PDF original imprime 0,0393. Se declara la adopción de 0,00393 para el acoplamiento DC R(T); no se oculta como una transcripción propia. |
| Aras | DOI correcto `10.1080/15325000590964425`; se retira el DOI erróneo de la documentación activa. |
| Atoccsa | El índice histórico cita `en17174356`; la referencia de la tesis es `10.3390/en17051023`. Se conserva la errata en el registro de procedencia. |
| Ocłoń | El artículo de optimización de bedding corresponde a `10.1016/j.energy.2015.04.100`, no al DOI del índice histórico. |
| Al-Dulaimi | FEM–BPNN aporta un sustituto supervisado; no se clasifica como PINN por el hecho de usar resultados térmicos. |
| Weiss | Se distingue la edición española consultada de los metadatos de la edición inglesa; se enlaza el PDF español del BID. |
| Familias PINN | Se agregan fuentes primarias sobre cPINN, gradientes, muestreo, primer orden, VPINN y fronteras exactas. Las alternativas no ejecutadas se identifican. |

Las referencias nuevas están en `Benchmarks/references_selection.bib` y se
incorporan a la compilación de la tesis. Esta actualización no modifica una
biblioteca Zotero externa ni pretende certificar cada nota de lectura antigua.

| Instancia del índice del Plan | Uso en la batería o motivo de delimitación |
|---|---|
| Kim 2025 | Geometría de seis cables y propiedades; adaptación controlada, nueve capas radiales. |
| Khumalo 2025 | Sustento del análisis de suelo seco; escenarios sintéticos, sin calibración experimental. |
| Al-Dulaimi 2024 | Antecedente de suelo multicapa y FEM–BPNN; no reproducción de la red supervisada. |
| Atoccsa 2024 | Antecedente de relleno y optimización; optimización económica/PSO fuera del alcance ejecutado. |
| Ocłoń 2015 | Antecedente de relleno localizado; no se atribuyen sus resultados al prototipo. |
| Aras 2005 | Geometría y disposición: un cable y tres cables planos. |
| Quan 2019 | Disposición y k(T) como extensión; el modelo actual usa k espacial independiente de T. |
| Oladunjoye 2012 | Datos de propiedades; no es por sí mismo una solución de referencia de campo. |
| Möbius 2025 | Contexto estacional; serie climática y transitorios no ejecutados. |
| CIGRE 880 | Referencia normativa; no se certifican sus catorce variantes IEC mediante pérdidas DC simplificadas. |
| CIGRE 963, caso 1 | Orienta controles de campo, dominio y malla; ductos completos no reproducidos. |
| IEC 60853, anexo A | Fuera del alcance estacionario. |
| Pan, placa | Sustento de PINN térmica; inversión de frontera fuera del problema directo ejecutado. |
| CIGRE, anillo | Prueba analítica ejecutada con PINN radial y FEM 2D. |
| Hahn, rectángulo Robin | La condición convectiva se verifica con una solución manufacturada propia; no se atribuye a la serie de Hahn. |
| MMS del Plan | Conductividad variable y salto exacto, ampliados con variación continua 2D, contraste 10:1 y estrato horizontal. |

## Formato, selección y evidencia actual

`Benchmarks/FORMAT.md` documenta valores constantes, expresiones y estratos
con discontinuidad exacta o suavizado declarado. La fuente manufacturada puede
derivarse automáticamente de la solución y de k. Los nuevos casos se resuelven
con ambos métodos; las capacidades del formato no se confunden con un dominio
ya verificado para cualquier número de estratos o geometría.

La campaña C01–C08 compara balance, peso de PDE, anchura, profundidad, tasa de
aprendizaje y muestreo. Las ablaciones adicionales comparan PINN directa,
red global frente a redes por material y enriquecimiento multipolar. Se
conservan todos los resultados, las semillas y los controles con igual
presupuesto cuando se evalúa una modificación del enriquecimiento.

La elección es la mejor alternativa observada bajo un criterio declarado,
no un óptimo universal. La campaña de un cable separa mínima complejidad de
mínimo error; los casos con varios cables requieren otra representación.
La evidencia por caso y la regla de selección se encuentran en
`Benchmarks/selection.json` y `Benchmarks/summary/`.

Las salidas trazables incluyen JSON de métricas, NPZ de campos, PT de pesos,
fuentes archivadas por huella, CSV agregados, figuras y cuadernos ejecutados.
Las tablas de la tesis proceden de `report.py`; `ampacity.py` conserva el
índice histórico con resistencia común a 90 °C. Los tiempos
incluyen carga concurrente y diferencias Windows/WSL; no demuestran una
aceleración respecto de FEM.

## Alcance alcanzado y límites que permanecen

Se desarrolla y verifica un artefacto de conducción estacionaria con modelo
reducido de cables. Las conclusiones distinguen suelos continuos y discretos,
resultados admitidos y fallos. La temperatura de conductor es reconstruida;
no se afirma haber verificado un campo 2D completo dentro de cada capa.

La corriente límite se calcula con actualización individual R(T), tanto en
FEniCSx como en PINN. La primera técnica usa respuestas térmicas unitarias,
iteración y bisección; la segunda incorpora potencia y corriente como
incógnitas y conserva sus residuos eléctricos. Se ejecutan tres mallas FEM
y tres semillas PINN por escenario y modo. Los resultados a R20 fija se
conservan como pruebas térmicas controladas, identificadas explícitamente.
La ampacidad corresponde al modelo DC reducido; no incluye pérdidas AC,
validación experimental ni acoplamiento humedad–calor. OE4 se concreta en
ese dominio y no certifica una instalación.

La tesis conserva espacios administrativos y personales que no pueden
completarse con datos supuestos. La conformidad final corresponde a la
evaluación académica; el informe no sustituye la aprobación de la universidad.

## Errata de adjunto bibliográfico

El PDF local identificado como Raissi (2019) contiene realmente la versión 1
de *Physics informed deep learning (Part I)*, publicada en arXiv en 2017.
Se conserva la referencia de 2019 como artículo con metadatos editoriales
verificados, pero las citas de páginas de los cuadernos se atribuyen a la
prepublicación efectivamente consultada. No se fabrican páginas del artículo.
La ley R(T) y α del cobre se contrastan con TB 880, p. 132, del PDF local.

## Verificación de cifras de contexto

Se contrastó la producción nacional de 65 109 GWh y la participación solar/eólica de 9,5 % con el boletín MINEM de enero de 2026, cuadro 4, guardado en Plan/_fuentes_tmp/minem_2025_dic.pdf. Se retiró del diagnóstico la cifra de máxima demanda 7942 MW: el JSON COES conservado para 2025 contiene un máximo mensual de 7966,448 MW y no se acreditó la equivalencia de criterios entre ambas cifras. Esta discrepancia no afecta los cálculos térmicos.

## Cierre de ejecución y revisión

- Batería física: 19 casos; 57 soluciones FEM de verificación y 66 expedientes acoplados, con tres mallas. Todas las referencias finas cumplen los controles declarados.
- Entrenamientos conservados: 290. La selección principal acepta 53/57 ejecuciones térmicas o electrotérmicas y 30/33 de corriente límite. Los fallos no se eliminan.
- Verificación automatizada: 346 pruebas aprobadas, incluido el análisis de derivadas de propagación; detalle en `auditoria/tests_final.xml`.
- Auditoría de artefactos: RMSE y máximos reconstruidos desde campos; comprobación de la respuesta FEM, R(T), puntos, modos y fuentes. Las 90 ejecuciones principales tienen fuentes completas. Quedan 54 registros históricos con archivo de código original incompleto, identificados individualmente.
- Reproducción: reinstalación Python aislada y dos casos recalculados; FEniCSx recalculado en WSL. No se afirma una reinstalación limpia de Conda.
- Resolución PINN: 36 entrenamientos con 16/32/64 neuronas, tres densidades interiores y presupuesto duplicado. Se conservan tasas empíricas, comparaciones entre semillas y descomposición térmica/eléctrica. Detalle en `Benchmarks/ANALISIS_NUMERICO.md`.
- Colocación adaptativa: 18 entrenamientos nuevos y seis controles fijos. Se comparan renovación aleatoria, residuo y gradiente térmico, con igual número interior y presupuesto. Se reconstruyeron las selecciones y se comprobaron puntuaciones desde estados intermedios. Detalle en `Benchmarks/MUESTREO_ADAPTATIVO.md`.
- Entrega revisable: 19 cuadernos ejecutados con gráficos y citas; PDF de 74 páginas, sin errores de referencia ni desbordamientos detectados por el registro de compilación. La revisión visual se conserva en `auditoria/pdf_review/`.
- Metodología, resultados, discusión, conclusiones y resumen se completaron desde el informe interno y las tablas; se mantienen únicamente los espacios personales y administrativos auténticos.

Casos nominales con aceptación parcial: kim_layered, kim_pac. Casos sin ninguna
ampacidad PINN aceptada: ninguno. Estas
limitaciones impiden presentar el artefacto como reemplazo general de FEM.
La operación usa R(Tc) individual en ambas técnicas; el R20 del JSON es una
referencia y los ensayos a potencia fija conservan su carácter de verificación.

La comparación de redes más y menos densas es evidencia de insensibilidad en
los rangos probados, condicionada a error externo y a entrenamiento. Los
órdenes observados FEM se separan de las pendientes PINN. La propagación del
error es determinista dentro del modelo reducido; no se inventa incertidumbre
estadística de propiedades físicas ni se aplica GCI a la anchura neuronal.
