"""Integrate the executed common model into the thesis; numerical tables are generated."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];CH=ROOT/'Tesis_LaTeX_Borrador_UNI/capitulos'
p=CH/'03_propuesta_y_desarrollo.tex';s=p.read_text(encoding='utf-8')
s=s.replace('varios cables del mismo tipo y una potencia común por escenario','varios cables del mismo tipo y una corriente común por escenario, con potencias individuales dependientes de la temperatura')
s=s.replace('Las potencias de referencia se calculan con $P=I^2R_{20}$.','Las potencias de referencia se calculan con $P_{20}=I^2R_{20}$. Los resultados de operación actualizan la resistencia de cada conductor con su temperatura; las pruebas a potencia fija se conservan como controles térmicos.')
s=s.replace('quince casos: cuatro soluciones manufacturadas, un anillo analítico y diez escenarios de cables','diecinueve casos: siete soluciones manufacturadas, un anillo analítico y once escenarios de cables')
needle=r'\subsection{REFERENCIA FEM Y CONTROL DE DISCRETIZACIÓN}'
insert=r'''\subsection{PARÁMETROS CONTINUOS E INTERFACES DISCRETAS}

La conductividad admite un valor constante, una fórmula en coordenadas físicas o una lista ordenada de estratos. El lenguaje de expresiones restringe operadores y funciones; no ejecuta código arbitrario. NumPy, PyTorch y UFL evalúan la misma definición, registrada en el archivo del caso.

La verificación continua incorpora $k=1+x$, $k=1+0,4\sin(2\pi x)\cos(2\pi y)$ y $k=\exp[\ln(10)(x+y)/2]$. En el cuadrado unitario, los dos últimos campos cubren intervalos de $0,6$ a $1,4$ y de $1$ a $10$, respectivamente. La fuente manufacturada se deriva de la solución prescrita y de cada conductividad.

La evaluación de $\nabla\cdot(k\nabla T)$ conserva el término $\nabla k\cdot\nabla T$. Omitirlo resolvería otra ecuación cuando los parámetros varían espacialmente. La derivación simbólica de la fuente se contrasta con diferenciación automática en puntos independientes.

Los estratos con salto exacto se declaran mediante posición, eje y valores laterales, con ancho de suavizado nulo. FEniCSx ajusta la geometría a la interfaz y PINN utiliza una subred por región, con continuidad de temperatura y flujo. Se incluyen interfaces manufacturadas vertical y horizontal, además de un cable sobre dos estratos de suelo.

Un ancho de transición positivo define un material continuo distinto, cuyo gradiente se incorpora a la ecuación. Este parámetro pertenece a los datos físicos y no se modifica para ocultar un fallo del entrenamiento. La capacidad del formato para describir más estratos no implica verificación automática de configuraciones no ejecutadas.

'''
assert needle in s;s=s.replace(needle,insert+needle)
s=s.replace('w_e\\mathcal L_{\\mathrm{energía}}.','w_e\\mathcal L_{\\mathrm{energía}}+w_R\\mathcal L_{R(T)}+w_I\\mathcal L_{T_{\\lim}}.')
a=s.index(r'\subsection{ÍNDICE DE CORRIENTE Y REPRODUCIBILIDAD}');b=s.index(r'\section{ANÁLISIS DE LOS DATOS Y RESULTADOS}',a)
s=s[:a]+r'''\subsection{ACOPLAMIENTO ELECTROTÉRMICO Y AMPACIDAD DC}
\label{sec:acoplamiento}

La pérdida Joule depende de la temperatura del conductor que la produce. Se emplea la ley DC lineal de la Fórmula~\ref{for:perdidas-acopladas}, con $\alpha=0,00393$ K$^{-1}$ para el cobre, documentada en \textcite[132]{cigre2022}. La misma relación se impone en ambos métodos y se evalúa individualmente.

\begin{formula}[htbp]
\centering
\begin{equation}
\label{for:perdidas-acopladas}
R_j(T_{c,j})=R_{20}[1+\alpha(T_{c,j}-20)],\qquad
P_j=I^2R_j(T_{c,j}).
\end{equation}
\caption{Actualización de resistencia y pérdida lineal por conductor.}
\end{formula}

Fijar $R_{20}$ desacopla la generación y la temperatura. Esa condición se conserva para verificar el operador térmico y comparar arquitecturas con una fuente idéntica. Los resultados operativos y de corriente límite emplean la relación acoplada; no se asigna a todos los conductores la temperatura del más caliente.

FEniCSx obtiene una matriz de respuesta térmica mediante una excitación de un vatio por metro en cada cable. La reconstrucción radial se incorpora a su diagonal. Con propiedades térmicas independientes de la temperatura, la relación entre potencias y temperaturas resulta afín, como muestra la Fórmula~\ref{for:respuesta-acoplada}.

\begin{formula}[htbp]
\centering
\begin{equation}
\label{for:respuesta-acoplada}
\boldsymbol T_c=T_0\boldsymbol 1+A\boldsymbol P,\qquad
[\mathbb I-I^2R_{20}\alpha A]\boldsymbol P
=I^2R_{20}[1+\alpha(T_0-20)]\boldsymbol 1.
\end{equation}
\caption{Respuesta térmica y comprobación matricial del acoplamiento DC.}
\end{formula}

La iteración de potencias se detiene cuando el cambio térmico es menor que $10^{-5}$ K y se contrasta con la solución matricial. Se rechaza un punto fijo inestable según el radio espectral del acoplamiento. Una solución FEM final con las potencias convergidas verifica campo, temperatura y balance.

La PINN añade una potencia entrenable por conductor a su enriquecimiento analítico y multipolar. La temperatura de superficie y la reconstrucción radial determinan el residuo de la ley eléctrica. El entrenamiento conjunto conserva las derivadas respecto a esas potencias y no recibe temperaturas ni matrices FEM como etiquetas.

La campaña acoplada mantiene tres capas de 32 neuronas, 1200 pasos Adam y hasta 1600 iteraciones L-BFGS. La penalización eléctrica tiene peso inicial 100; sus variaciones se identifican como comparaciones adicionales. Se exige un residuo eléctrico relativo máximo de 0,1\,\pct{}, junto con los criterios térmicos.

Para ampacidad, FEniCSx aplica bisección sobre la corriente y resuelve el acoplamiento individual en cada evaluación. La PINN incorpora una corriente positiva desconocida y una penalización de la temperatura límite, con peso inicial 100. Ambos métodos buscan la condición de la Fórmula~\ref{for:indice-corriente}.

\begin{formula}[htbp]
\centering
\begin{equation}
\label{for:indice-corriente}
\max_j T_{c,j}(I,\boldsymbol R(\boldsymbol T_c))=90\ ^\circ\mathrm C.
\end{equation}
\caption{Condición de corriente límite con resistencias individuales actualizadas.}
\end{formula}

La bisección FEM utiliza tolerancia térmica de $10^{-4}$ K y conserva cada paso. La aceptación PINN exige además una distancia máxima de $0,1$ K al límite y un error de corriente menor o igual que 5\,\pct{} frente a FEM. Estos son criterios numéricos de esta evaluación, no tolerancias de una norma de instalación.

El resultado es una ampacidad DC del modelo reducido. No incluye pérdidas dieléctricas, pantallas, efecto pelicular ni proximidad. El índice histórico que fija todas las resistencias a 90 °C se conserva como antecedente computacional y no sustituye la solución individual acoplada.

\subsection{REPRODUCIBILIDAD Y EXPEDIENTE DIGITAL}

Cada expediente contiene los datos físicos, la ley eléctrica, fuentes archivadas por SHA-256, configuraciones, pesos PINN, mallas, campos y métricas. Los cuadernos muestran los resultados guardados y permiten ejecutar nuevamente ambos solucionadores. Se distingue la revisión del cuaderno del entrenamiento previo que produjo la evidencia.

El informe interno reúne todas las variantes, incluidas las rechazadas, con ventajas, limitaciones y gráficos. Las tablas de la tesis se generan desde esos mismos registros. Las versiones de dependencias y los comandos de reproducción se conservan en el repositorio, sin atribuir al archivo de un entorno una prueba de reinstalación que no se haya realizado.

'''+s[b:]
s=s.replace('Se ejecutaron cuarenta y cinco soluciones FEM, correspondientes a quince casos y tres mallas por caso.','La verificación térmica reúne cincuenta y siete soluciones FEM: diecinueve casos y tres mallas por caso. El acoplamiento añade sesenta y seis expedientes de solución, correspondientes a once escenarios, tres mallas y dos modos: operación nominal y corriente límite.')
s=s.replace(r'\input{tablas/benchmark_fem}',r'''\input{tablas/benchmark_fem}

La convergencia del modelo acoplado se registra por separado en la Tabla~\ref{tab:benchmark-fem-acoplado}. Para la ampacidad se examina también el cambio de corriente entre las mallas finas. Las respuestas unitarias necesarias para construir la matriz térmica son soluciones intermedias y no se cuentan como escenarios adicionales.

\input{tablas/benchmark_coupled_fem}''')
s=s.replace('La Tabla~\\ref{tab:benchmark-pinn} presenta la mediana','La Tabla~\\ref{tab:benchmark-pinn} usa los casos manufacturados y el anillo como verificación, y los cables con pérdidas acopladas como operación. Presenta la mediana')
a=s.index(r'\subsection{EFECTO DE LA HETEROGENEIDAD}');b=s.index(r'\section{DISCUSIÓN E INTERPRETACIÓN DE LOS RESULTADOS}',a)
s=s[:a]+r'''\subsection{EFECTO DE LA HETEROGENEIDAD Y DE LAS PÉRDIDAS}

La Tabla~\ref{tab:benchmark-perdidas} contrasta las temperaturas FEM con resistencia de referencia fija y con actualización individual. La discrepancia refleja la realimentación entre pérdida y temperatura. Los controles térmicos no deben utilizarse como predicciones de operación cuando se requiere dicha actualización.

\input{tablas/benchmark_losses}

\input{tablas/benchmark_operational_findings}

Las comparaciones mantienen dominio, corriente y contornos, modificando el entorno térmico declarado. Por ello permiten atribuir diferencias a esos escenarios controlados. No reproducen exactamente las condiciones de los artículos ni demuestran un efecto universal de cualquier suelo seco o material de relleno.

\subsection{COMPARACIÓN DE AMPACIDAD DC ACOPLADA}

La Tabla~\ref{tab:benchmark-ampacidad} presenta las corrientes que satisfacen la condición límite con R(T) individual. La mediana PINN incorpora únicamente semillas que cumplen todos los criterios, mientras el error máximo incluye los intentos evaluados. La cantidad aceptada permite distinguir una estimación estable de una aceptación parcial.

\input{tablas/benchmark_ampacity}

La modificación de corriente surge del cambio de resistencia térmica y de la actualización eléctrica simultánea. En varios cables, la temperatura de cada uno determina sus pérdidas; solo el más caliente alcanza el límite impuesto. La coincidencia con FEM verifica el problema definido, sin certificar pérdidas AC ni una instalación real.

'''+s[b:]
s=s.replace('mientras OE4 aporta un índice DC condicionado y no una ampacidad normativa completa','mientras OE4 aporta una ampacidad DC con actualización individual R(T), circunscrita al modelo reducido')
p.write_text(s,encoding='utf-8')
# Bibliographic identity of the consulted preprint and Spanish edition.
p=ROOT/'Tesis_LaTeX_Borrador_UNI/referencias.bib';s=p.read_text(encoding='utf-8')
a=s.index('@techreport{weiss2021regulacion,');b=s.index('\n@',a+1);part=s[a:b]
part=part.replace('Chueca, J. Enrique and Carvalho Metanias Hallack, Michelle','Chueca Montuenga, Enrique and Hallack, Michelle')
part=re.sub(r'  doi\s*=.*\n','',part)
part=re.sub(r'  url\s*=.*\n','  url         = {https://publications.iadb.org/publications/spanish/document/impacto-de-la-regulacion-en-la-calidad-del-servicio-de-distribucion-de-la-energia-electrica-en-ameri.pdf},\n',part)
part=part.replace('  type        =', '  number      = {IDB-TN-2328},\n  type        =')
s=s[:a]+part+s[b:]
if '@online{raissi2017,' not in s:s+=r'''
@online{raissi2017,
 author = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
 title = {Physics informed deep learning (Part {I}): Data-driven solutions of nonlinear partial differential equations},
 year = {2017},
 eprint = {1711.10561},
 eprinttype = {arxiv},
 note = {Prepublicación, versión 1 consultada},
 url = {https://arxiv.org/abs/1711.10561v1}
}
'''
p.write_text(s,encoding='utf-8')
