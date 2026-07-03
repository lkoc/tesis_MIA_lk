# Guía de redacción del plan de tesis

Esta guía define el estilo de redacción que debe seguir el documento `plan_tesis_cables_pinn.tex`. Su objetivo es mantener una escritura clara, directa y trazable.

La guía sirve para dos usos. Primero, orienta a una persona que revisa el texto. Segundo, funciona como instrucción de estilo para un LLM que deba redactar, corregir o reescribir fragmentos del plan.

## Numeración y tabla de contenido

El documento LaTeX debe usar tres niveles numéricos (`\section`, `\subsection`, `\subsubsection`) y un cuarto nivel con numeración por letras. El cuarto nivel se implementa como `\paragraph` y debe mostrarse como `a)`, `b)`, `c)` en el texto y en el índice.

- Asegurarse de que el preámbulo incluya `\setcounter{tocdepth}{4}` y `\setcounter{secnumdepth}{4}` para que los cuatro niveles aparezcan en el TOC y estén numerados.
- El formato de `\paragraph` usado en el preámbulo debe forzar la numeración en letras mediante `\renewcommand\theparagraph{\alph{paragraph})}`.
- En los fragmentos que un LLM genere, respetar este esquema: usar `\paragraph` para subdividir hasta el cuarto nivel y numerar con letras entre paréntesis de cierre.

Ejemplo de configuración (preambulo):

```tex
% Incluir hasta cuarto nivel en el índice y numeración de secciones
\setcounter{tocdepth}{4}
\setcounter{secnumdepth}{4}
% Numeración de cuarto nivel por letras
\renewcommand\theparagraph{\alph{paragraph})}
	itleformat{\paragraph}{\normalfont\bfseries}{\theparagraph}{0.75em}{}
```

Ver la implementación en [Plan/plan_tesis_cables_pinn.tex](Plan/plan_tesis_cables_pinn.tex).

## Regla Maestra

La redacción debe ser académica, técnica y formal, pero tan simple como sea posible. El texto debe estar siempre en tercera persona.

La forma preferida es impersonal o institucional:

- `La investigación analiza...`
- `El estudio propone...`
- `La tesis evalúa...`
- `El artefacto permite...`
- `Se observa...`
- `Se evaluará...`

No usar primera persona ni segunda persona:

- no usar `yo`, `mi`, `me`, `nosotros`, `nuestro`, `nuestra`, `presentamos`, `analizamos`.
- no usar `tú`, `usted`, `ustedes`, `debes`, `puedes`.
- no dirigirse al lector.

## Uso Como Skill Para LLM

Cuando un LLM use esta guía, debe aplicar estas reglas antes de generar el texto:

1. Conservar el contenido técnico, las citas, las etiquetas LaTeX, las fórmulas y los datos numéricos.
2. Reescribir en tercera persona, con tono académico, técnico y formal.
3. Mantener lenguaje simple, directo y preciso.
4. Separar ideas largas en oraciones de 25 a 35 palabras cuando sea posible.
5. Usar conectores para mostrar causa, contraste, secuencia o consecuencia.
6. No inventar citas, datos, resultados ni fuentes.
7. No agregar referencias de estilo al archivo `.bib` ni al documento de tesis.
8. No convertir una afirmación débil en una conclusión fuerte.
9. No usar adjetivos valorativos si no existe criterio o evidencia.
10. Revisar que cada párrafo tenga una función: contexto, problema, evidencia, análisis, método, resultado esperado o límite.
11. Mantener redacción discursiva, con unidad temática, coherencia, cohesión y propósito comunicativo claro.
12. Verificar que toda figura tenga fuente en el caption.
13. Usar referencias cruzadas internas cuando ayuden a conectar capítulos, secciones, figuras, tablas o anexos.
14. Incluir una apertura y un cierre discursivo en cada capítulo; en secciones extensas, aplicar la misma regla.

Salida esperada del LLM:

- texto corregido en español académico.
- redacción en tercera persona.
- párrafos breves.
- citas técnicas conservadas.
- sin explicación externa, salvo que se pida una justificación de cambios.

## Marco de revisión

Cada sección debe revisarse con cinco acciones: pensar, organizar, relacionar ideas, argumentar y comunicar. Estas acciones permiten controlar la lógica antes de corregir solo palabras.

### Pensar

Antes de escribir, definir qué se quiere demostrar, qué evidencia existe y cuál es la idea principal. También se debe separar lo que puede concluirse de lo que todavía es un supuesto.

Preguntas guía:

- ¿Qué debe demostrar la sección?
- ¿Qué evidencia sostiene la afirmación?
- ¿Cuál es la idea principal?
- ¿Qué conclusión es válida con la evidencia disponible?

### Organizar

Las ideas deben seguir una secuencia lógica. No basta con tener buenas ideas; deben aparecer en un orden que permita seguir el razonamiento.

Secuencia recomendada:

1. Problema.
2. Evidencia.
3. Análisis.
4. Conclusión.

La matriz de consistencia funciona como organizador central. Por tanto, debe alinear problema, objetivos, hipótesis, variables, evidencia y controles.

### Relacionar Ideas

Las ideas no deben aparecer como frases independientes. Cada oración debe conectarse con la anterior mediante una relación clara.

Relaciones esperadas:

- causa y efecto.
- comparación.
- contraste.
- secuencia.
- consecuencia.

Los conectores ayudan a mostrar esas relaciones. Deben usarse cuando aclaran la lógica, no como relleno.

### Argumentar

Toda afirmación técnica debe estar sustentada. La argumentación convierte una opinión en una conclusión defendible.

Evitar frases como:

> Este algoritmo es mejor.

Preferir una afirmación que responda:

- ¿Por qué?
- ¿Con qué evidencia?
- ¿Comparado con qué?
- ¿En qué condiciones?

### Comunicar

El razonamiento debe expresarse con lenguaje claro, preciso, objetivo, coherente y conciso. La gramática, el vocabulario y el estilo hacen visible la lógica del texto.

Un texto bien escrito no necesariamente está bien razonado. Por tanto, primero se revisa la lógica y luego se pule la forma.

## Redacción Discursiva

El documento debe tener estilo discursivo. Esto significa que el texto no debe leerse como una lista de ideas aisladas, sino como una explicación organizada alrededor de un tema central.

La redacción discursiva exige seis elementos:

- unidad temática.
- coherencia.
- cohesión.
- intencionalidad comunicativa.
- adecuación al público y al contexto.
- integración de tipos de redacción.

### Unidad Temática

Cada capítulo, sección y párrafo debe girar alrededor del problema central: estimar temperatura y ampacidad en cables enterrados con heterogeneidad térmica mediante un artefacto PINN verificable.

Las partes del texto deben contribuir al desarrollo de esa idea. Si un párrafo no explica el problema, la evidencia, el método, el artefacto, la evaluación o el alcance, debe revisarse.

### Coherencia

Las ideas deben organizarse de manera lógica y ordenada. El lector debe poder seguir esta ruta general:

1. sistema físico.
2. problema térmico.
3. brecha de representación.
4. propuesta PINN.
5. objetivos e hipótesis.
6. metodología DSR.
7. evidencia, métricas y anexos.

Cada sección debe iniciar con una idea que conecte con la anterior. Además, debe cerrar o avanzar hacia la siguiente parte del argumento.

### Cohesión

La cohesión se logra con conectores, referencias internas y repetición controlada de términos clave. Los términos `sistema cable--instalación--entorno térmico`, `heterogeneidad térmica`, `\kx`, `T(x,y)`, `\Tmax`, `\Imax`, `artefacto PINN` y `DSR` deben mantener continuidad.

Los conectores deben señalar la relación entre ideas:

- `por tanto`: consecuencia.
- `sin embargo`: contraste.
- `además`: suma relacionada.
- `debido a`: causa.
- `en cambio`: contraste entre opciones.
- `así`: resultado.
- `primero`, `luego`, `finalmente`: secuencia.

### Intencionalidad Comunicativa

Cada sección debe tener un propósito visible. En este plan, los propósitos principales son informar, explicar, justificar, delimitar y argumentar.

La redacción no debe buscar persuadir con adjetivos. Debe convencer mediante evidencia, orden lógico, criterios de evaluación y citas técnicas.

### Adecuación Al Público Y Contexto

El público previsto es académico y técnico. Por ello, el lenguaje debe ser formal, preciso y verificable, pero no debe volverse innecesariamente complejo.

Los términos especializados se aceptan cuando son necesarios. Sin embargo, deben aparecer definidos en la sección de términos o explicados por el contexto.

### Integración De Tipos De Redacción

La tesis puede combinar exposición, descripción y argumentación. La exposición explica conceptos; la descripción precisa sistemas, variables o escenarios; la argumentación justifica decisiones metodológicas.

La narración solo debe usarse cuando ordena una secuencia de trabajo, como las fases DSR o el procedimiento de evaluación.

## Hilo Conductor

El hilo conductor debe poder seguirse entre capítulos. La lectura esperada es:

1. El problema surge porque el entorno térmico real del cable enterrado es heterogéneo.
2. Esa heterogeneidad puede cambiar el campo de temperatura y la ampacidad.
3. Las representaciones homogéneas pueden ocultar ese cambio.
4. La tesis propone un artefacto PINN para representar $\kx$ y estimar $T(x,y)$.
5. Los objetivos, hipótesis y dimensiones operativas traducen la brecha en tareas evaluables.
6. La metodología DSR organiza la construcción, verificación y evaluación del artefacto.
7. El marco teórico aporta las bases físicas, normativas, numéricas y metodológicas.
8. Los anexos mantienen la trazabilidad entre problema, objetivos, variables, evidencia y controles.

Cada capítulo debe contener al menos una frase puente que indique su relación con el capítulo anterior o con el objetivo general. Esa conexión puede reforzarse con referencias cruzadas internas cuando la relación sea útil para el lector.

## Referencias Cruzadas Internas

El hilo conductor también se verifica mediante referencias cruzadas internas. Estas referencias ayudan a mostrar que una idea no aparece aislada, sino conectada con otra parte del documento.

Usar referencias cruzadas cuando:

- una sección depende de una definición presentada antes.
- una hipótesis deriva de un problema u objetivo.
- una decisión metodológica se apoya en una dimensión operativa.
- una tabla o matriz organiza ideas desarrolladas en el texto.
- un anexo respalda una decisión, una variable o un diseño de escenarios.

No abusar de las referencias cruzadas. Deben aparecer solo cuando orientan al lector o evitan repetir una explicación.

Formas recomendadas:

- `como se definió en la Sección~\ref{sec:...}`.
- `esta relación se retoma en la Sección~\ref{sec:...}`.
- `la trazabilidad se organiza en el Anexo~\ref{anx:...}`.
- `la Figura~\ref{fig:...} resume esta relación`.
- `la Tabla~\ref{tab:...} convierte esta lógica en criterios`.

Evitar formas mecánicas o forzadas:

- `ver sección...` sin explicar por qué importa.
- referencias al final de una oración sin relación clara.
- varias referencias seguidas en una misma oración.

Regla práctica:

> Una referencia cruzada debe explicar una relación, no solo señalar una ubicación.

## Aperturas Y Cierres

Cada capítulo debe iniciar con un párrafo breve que explique qué se desarrolla y cómo se relaciona con el hilo conductor. Ese párrafo no debe adelantar resultados; debe orientar la lectura.

Estructura recomendada de apertura:

1. indicar el tema del capítulo.
2. explicar su función dentro del plan.
3. conectar con el capítulo anterior, el problema general o el objetivo general.

Cada capítulo debe cerrar con un párrafo breve de síntesis. El cierre debe resumir lo tratado o indicar la conclusión general que prepara el capítulo siguiente.

Estructura recomendada de cierre:

1. resumir la idea principal del capítulo.
2. indicar qué queda establecido.
3. conectar con la siguiente parte del documento.

En secciones largas, se debe aplicar la misma lógica. La apertura explica qué se desarrollará; el cierre resume la función de esa sección dentro del argumento.

Ejemplo de apertura:

> Este capítulo convierte el problema de investigación en objetivos, hipótesis y dimensiones operativas. Por tanto, conecta la brecha técnica descrita en la Sección~\ref{sec:problematica} con la metodología desarrollada en la Sección~\ref{sec:metodologia}.

Ejemplo de cierre:

> En síntesis, el capítulo define qué se evaluará, con qué variables y bajo qué controles. Esta organización prepara la metodología de verificación descrita en la Sección~\ref{sec:metodologia}.

## Principio general

El texto debe explicar ideas técnicas con lenguaje simple. La complejidad debe venir de la relación lógica entre ideas, no de palabras rebuscadas.

Por tanto, cada afirmación debe cumplir una función clara dentro del argumento. Si una frase no aporta evidencia, relación lógica o precisión técnica, debe eliminarse o reescribirse.

La simplicidad no significa informalidad. El texto puede ser simple y, al mismo tiempo, académico, técnico y formal.

## Afirmaciones justificadas

Evitar afirmaciones genéricas como:

- "Esto es importante."
- "El método es eficiente."
- "La herramienta es robusta."
- "El problema es relevante."

Cada afirmación debe indicar por qué, para qué o con base en qué fuente se sostiene.

Ejemplo:

Antes:

> La heterogeneidad térmica es importante para la ampacidad.

Después:

> La heterogeneidad térmica modifica la ampacidad porque cambia el camino de disipación del calor. Este efecto se observa cuando la resistividad del suelo aumenta por secado \parencite[1,6--9]{khumalo2025}.

## Lenguaje directo

Usar palabras comunes siempre que no se pierda precisión técnica.

Preferir:

- "usar" en lugar de "emplear" cuando no haya diferencia técnica.
- "mostrar" en lugar de "evidenciar" si el sentido es simple.
- "cambiar" en lugar de "modificar" cuando el contexto sea general.
- "por eso" o "por tanto" cuando se necesita consecuencia.

En el documento de tesis se prefiere `por tanto` sobre `por eso`, porque conserva un tono más académico.

Mantener términos técnicos solo cuando sean necesarios para el campo:

- ampacidad
- conductividad térmica
- resistividad térmica
- PDE
- PINN
- FEM
- DSR
- \textit{bedding}
- \textit{backfill}

Estos términos deben estar definidos en la sección de definición de términos.

## Oraciones

Las oraciones deben tener, en lo posible, entre 25 y 35 palabras. Se acepta una oración algo más larga cuando contiene una cita o una expresión técnica necesaria.

La estructura preferida es:

> sujeto + verbo + complemento

Ejemplo:

> La conductividad térmica del suelo cambia la temperatura máxima del conductor.

Se pueden usar oraciones subordinadas, pero solo si ayudan a relacionar ideas.

Ejemplo:

> La aproximación homogénea puede ser útil cuando el entorno es simple, pero pierde precisión cuando existen zonas secas cercanas al cable.

Evitar oraciones que unan demasiadas funciones. Si una oración define, justifica, compara y concluye al mismo tiempo, debe dividirse.

## Párrafos

Cada párrafo debe desarrollar una sola idea. En general, debe tener una o dos oraciones.

La primera oración presenta la idea. La segunda la justifica, la conecta o indica su consecuencia.

Ejemplo:

> La matriz de consistencia organiza la lógica del plan. Por tanto, relaciona problema, objetivos, hipótesis, variables, evidencia y controles.

Evitar párrafos que mezclen:

- contexto
- problema
- método
- resultado esperado
- justificación

Si aparecen varias funciones en un mismo párrafo, dividirlo.

## Conectores

Usar conectores para mostrar la relación entre ideas. No deben usarse como adorno.

Conectores recomendados:

- `por tanto`: consecuencia lógica.
- `sin embargo`: contraste o límite.
- `además`: suma de una idea relacionada.
- `debido a`: causa concreta.
- `por ello`: consecuencia práctica.
- `en cambio`: contraste entre dos opciones.
- `así`: resultado de una decisión metodológica.
- `primero`, `luego`, `finalmente`: secuencia.

Ejemplo:

> Las normas IEC aportan una referencia de cálculo. Sin embargo, no representan todas las heterogeneidades del entorno. Por tanto, se requiere una comparación con modelos de campo.

## Organización lógica

Las ideas deben avanzar en una secuencia clara:

1. Contexto.
2. Problema.
3. Brecha.
4. Propuesta.
5. Evidencia o criterio de evaluación.
6. Límite o alcance.

La matriz de consistencia debe funcionar como organizador de ideas. Por tanto, no debe verse como un anexo aislado.

Debe conectar:

- problema general y problemas específicos.
- objetivos general y específicos.
- hipótesis.
- variables o factores.
- evidencia esperada.
- controles.

## Citas

Las citas deben sostener afirmaciones específicas. No deben colocarse al final de un párrafo que contiene varias ideas distintas.

Regla práctica:

Si una oración afirma un dato, una relación física o una recomendación técnica, debe tener una fuente cercana.

Ejemplo:

> La resistividad térmica del suelo aumenta cuando baja la humedad, y ese aumento puede reducir la ampacidad \parencite[1,6--9]{khumalo2025}.

Evitar:

> La humedad, la profundidad, el relleno y la temperatura son importantes para el diseño, la operación y la seguridad de los cables enterrados \parencite{fuente}.

La oración anterior mezcla muchas afirmaciones. Debe dividirse.

## Adjetivos

Evitar adjetivos exagerados o no justificados:

- importante
- fundamental
- crítico
- novedoso
- robusto
- eficiente
- óptimo
- significativo

Se pueden usar si se explica el criterio.

Ejemplo:

Antes:

> El modelo FEM tiene un costo computacional alto.

Después:

> El modelo FEM tiene mayor costo computacional porque requiere malla, verificación y repetición del cálculo para cada escenario.

## Registro Académico, Técnico y Formal

El tono debe ser académico, técnico y formal, pero no solemne. El texto debe sonar como una explicación técnica clara.

Evitar:

> La presente investigación propende a la elucidación rigurosa del fenómeno térmico.

Preferir:

> Esta investigación analiza cómo la heterogeneidad térmica cambia la temperatura y la ampacidad.

Reglas de registro:

- usar tercera persona o forma impersonal.
- preferir verbos concretos: `analiza`, `evalúa`, `compara`, `estima`, `registra`, `verifica`.
- evitar expresiones conversacionales: `básicamente`, `en realidad`, `obviamente`, `de alguna manera`.
- evitar metáforas si existe una expresión técnica simple.
- evitar promesas absolutas: `garantiza`, `demuestra definitivamente`, `resuelve por completo`.
- preferir conclusiones delimitadas: `permite evaluar`, `permite comparar`, `dentro del dominio definido`.

Ejemplos:

Antes:

> En este trabajo mostramos que el método puede ser mejor.

Después:

> La tesis evaluará si el método reduce el error frente a una referencia FEM dentro del dominio definido.

Antes:

> El modelo es muy robusto.

Después:

> El modelo se considerará estable si la variación de $\Tmax$ entre semillas permanece dentro del criterio definido.

## Figuras y tablas

Toda figura o tabla debe estar conectada con el texto.

Antes de la figura o tabla, explicar para qué se usa. Después, indicar qué idea organiza o qué relación muestra.

Ejemplo:

> La Figura 1 ordena la relación entre el entorno real, la simplificación homogénea y el riesgo de decisión. Por tanto, sirve como puente entre la problemática y el artefacto propuesto.

Toda figura debe incluir su fuente en el caption. Si la figura fue creada para el documento, debe indicarse `Fuente: Elaboración propia.`.

Si la figura fue adaptada desde una referencia, debe indicarse `Fuente: Elaboración propia a partir de \textcite{...}.`.

Si la figura se toma directamente de una referencia, debe indicarse `Fuente: \textcite[fig.~x]{...}.`.

Las tablas también deben permitir rastrear su origen. Cuando sean matrices o tablas construidas para la tesis, se puede usar el texto previo o una nota metodológica para indicar que forman parte de la organización propia del estudio.

## Listas

Usar listas cuando los elementos sean paralelos. Cada ítem debe tener una estructura similar.

Evitar ítems excesivamente largos. Si un ítem necesita más de una idea, dividirlo o redactarlo en dos oraciones.

En listas metodológicas, cada ítem debe iniciar con un sustantivo o un verbo en infinitivo. No mezclar estilos dentro de la misma lista.

Ejemplo uniforme:

- definir requisitos.
- construir el artefacto.
- verificar referencias.
- evaluar escenarios.

## Reglas Para Editar LaTeX

Al corregir el archivo `.tex`, se deben conservar:

- comandos LaTeX.
- etiquetas `\label`.
- referencias `\ref`.
- citas `\parencite`, `\textcite` y variantes.
- fórmulas y símbolos.
- unidades `\SI`.
- tablas, figuras y entornos.

No cambiar una cita de lugar si no se revisa la oración que sostiene. No eliminar una cita técnica sin comprobar que la afirmación ya no la necesita.

## Referencias De Estilo

Estas referencias orientan el estilo de escritura científica y técnica. No forman parte de la tesis, no deben agregarse al archivo `.bib` y no deben citarse en `plan_tesis_cables_pinn.tex`.

- Lövei, G. L. (2021). *Writing and Publishing Scientific Papers: A Primer for the Non-English Speaker*. Open Book Publishers. https://doi.org/10.11647/obp.0235
- Scholz, F. (2022). *Writing and publishing a scientific paper*. *ChemTexts*, 8. https://doi.org/10.1007/s40828-022-00160-7
- Van Emden, J., & Becker, L. (2018). *Writing for Engineers* (4th ed.). Palgrave.

Principios adaptados para esta guía:

- escribir para que el lector siga el razonamiento sin esfuerzo innecesario.
- formular una idea principal por párrafo.
- sostener cada afirmación técnica con evidencia o criterio.
- usar estructura convencional y explícita en textos científicos.
- preferir claridad y precisión sobre ornamentación verbal.
- revisar el texto desde la necesidad del lector, no desde la comodidad del autor.

## Revisión rápida

Antes de cerrar una sección, verificar:

- El texto está en tercera persona.
- El tono es académico, técnico y formal.
- El lenguaje es simple sin volverse coloquial.
- El estilo es discursivo y mantiene un hilo conductor.
- Cada capítulo se conecta con el problema central y con el objetivo general.
- Cada capítulo tiene una apertura que orienta la lectura.
- Cada capítulo tiene un cierre que sintetiza lo tratado o prepara la siguiente parte.
- Las referencias cruzadas internas aparecen cuando ayudan a conectar ideas entre secciones, capítulos, figuras, tablas o anexos.
- Las referencias cruzadas internas se usan con moderación y de manera natural.
- La sección responde a una pregunta clara.
- Las ideas siguen la secuencia problema, evidencia, análisis y conclusión.
- Las oraciones muestran relaciones de causa, contraste, secuencia o consecuencia.
- Cada párrafo desarrolla una sola idea.
- Las oraciones no son innecesariamente largas.
- Las afirmaciones técnicas tienen fuente o justificación.
- Los conectores muestran relación lógica.
- No hay adjetivos exagerados sin criterio.
- La matriz de consistencia conecta las partes del plan.
- Todas las figuras tienen fuente en el caption.
- Las secciones están numeradas si deben aparecer en el índice.
- Los términos técnicos aparecen definidos en la sección correspondiente.

## Regla final

El documento debe leerse como una cadena lógica. Cada oración debe responder una de estas preguntas:

- ¿Qué ocurre?
- ¿Por qué ocurre?
- ¿Qué consecuencia tiene?
- ¿Cómo se evaluará?
- ¿Con qué evidencia se sostiene?

Si una oración no responde ninguna de estas preguntas, debe revisarse.
