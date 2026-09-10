from pathlib import Path
root=Path('Tesis_LaTeX_Borrador_UNI/capitulos')
p=root/'01_planteamiento_del_problema.tex';s=p.read_text(encoding='utf-8')
s=s.replace(r'\left|T_{\max}^{\mathrm{ref}}\right|',r'\left|T_{\max}^{\mathrm{ref}}-T_0\right|')
s=s.replace('La primera expresión mide la diferencia relativa de temperatura máxima frente a una referencia.','La primera expresión normaliza el error de temperatura máxima con el incremento respecto al ambiente. Esta elección evita que el resultado dependa del origen de la escala Celsius.')
a=s.index(r'\pendiente{Cerrar la metodología ejecutada');b=s.index('\n',a)
s=s[:a]+r'''La metodología ejecutada comprende quince casos y tres mallas FEM por caso. Las configuraciones se comparan mediante una campaña exploratoria de ocho candidatos y variantes específicas de interfaz y enriquecimiento geométrico. Las semillas 11, 23 y 37 evalúan sensibilidad; el detalle de selección se presenta en la Sección~\ref{sec:seleccion-pinn}.

El alcance físico ejecutado resuelve el suelo en dos dimensiones y reconstruye radialmente el interior de los cables. Las pérdidas base se fijan mediante resistencia DC a 20 °C. La comparación de corriente corresponde a un índice condicionado a resistencia común a 90 °C, por lo que el alcance operativo del OE4 es parcial.

Los umbrales de 5\,\pct{} para campo y temperatura y 2\,\pct{} para balance se conservan como reglas del proyecto. Se explicita que los errores relativos usan el incremento térmico respecto al ambiente. Las referencias FEM se someten además a un cambio máximo entre mallas inferior a 0,5\,\pct{} de dicho incremento.
'''+s[b:]
s=s.replace('Esta adaptación evita presentar como demostrado un resultado que todavía debe evaluarse.','Esta adaptación vincula las conclusiones con la evidencia computacional y con los límites observados.')
s=s.replace('En síntesis, el capítulo establece el problema, los objetivos y el protocolo que debe producir evidencia.','El capítulo establece el problema, los objetivos y el protocolo de producción y evaluación de evidencia.')
s=s.replace('Los resultados no podrán generalizarse','Los resultados no pueden generalizarse')
p.write_text(s,encoding='utf-8')
p=root/'02_marco_teorico.tex';s=p.read_text(encoding='utf-8');a=s.index(r'\pendiente{Actualizar el cierre');b=s.index('\n',a)
s=s[:a]+r'''\subsection{SÍNTESIS CRÍTICA Y CRITERIOS DE SELECCIÓN}

La revisión se actualizó para sustentar las decisiones de arquitectura y entrenamiento. Se consultaron fuentes primarias sobre continuidad de flujo, ponderación de pérdidas, muestreo, formulación variacional y variables de primer orden. La revisión es narrativa y dirigida al diseño del artefacto; no se presenta como revisión sistemática ni metaanálisis.

Una red entrenada con pares generados por FEM constituye un sustituto supervisado cuando su pérdida no incorpora la ecuación física. El enfoque FEM--BPNN de \textcite{aldulaimi2024} aporta antecedentes de representación térmica y aproximación, pero no demuestra por sí mismo el funcionamiento de una PINN. La misma distinción se aplica a operadores FNO entrenados solamente con etiquetas.

La literatura ofrece respuestas diferentes a dificultades distintas. La descomposición aborda interfaces; los pesos adaptativos abordan desequilibrios de optimización; y el muestreo adaptativo redistribuye puntos de evaluación del residuo \parencites{jagtap2020cpinn}{wang2020gradients}{wu2022sampling}. Ninguna de estas opciones garantiza superioridad en el arreglo de cables sin una comparación específica.

La brecha que se estudia es la falta de una comparación trazable para las configuraciones y supuestos seleccionados. No se afirma que PINN, heterogeneidad térmica o cálculo de ampacidad carezcan de antecedentes. La contribución consiste en especificar el mismo problema para PINN y FEM y clasificar el dominio de aceptación del artefacto.
'''+s[b:]
s=s.replace('La contribución abstracta prevista comprende','La contribución metodológica comprende')
s=s.replace('Son proposiciones sujetas a evaluación, no conclusiones anticipadas.','Su evaluación se documenta mediante los casos y las decisiones del Capítulo III.')
s=s.replace('En síntesis, la cadena física parte','La cadena física parte')
s=s.replace('la evidencia que todavía debe producirse','la evidencia obtenida y sus límites')
p.write_text(s,encoding='utf-8')
p=root/'00_introduccion.tex';s=p.read_text(encoding='utf-8')
s=s.replace('La investigación propone un artefacto','La investigación desarrolla y evalúa un artefacto')
s=s.replace('El Capítulo III presenta la solución y reserva la evidencia de desarrollo, resultados, discusión e impacto. El Capítulo IV se destina a las conclusiones y recomendaciones que podrán formularse después de evaluar el artefacto.','El Capítulo III presenta la solución, la selección de configuraciones, los resultados, la discusión y el impacto. El Capítulo IV formula conclusiones y recomendaciones dentro del modelo estacionario reducido, y distingue el índice de corriente DC de una ampacidad normativa completa.')
p.write_text(s,encoding='utf-8')
print('Context chapters updated')
