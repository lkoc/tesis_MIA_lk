# -*- coding: utf-8 -*-
"""Reestructura el Plan de tesis segun la Guia N.o 01 (R-02..R-10 de la auditoria general).

Solo edita texto LaTeX; no ejecuta calculos. Cada reemplazo verifica que el patron aparezca el numero
esperado de veces. Uso: python plan_reestructurar.py
"""
import re, os

P = r'c:\usr\ths_mia_fiis\tesis_MIA_lk\Plan\plan_tesis_cables_pinn.tex'
raw = open(P, encoding='utf-8', newline='').read()
crlf = '\r\n' in raw
S = raw.replace('\r\n', '\n')
ENV0 = {e: (len(re.findall(r'\\begin\{%s\}' % e, S)), len(re.findall(r'\\end\{%s\}' % e, S)))
        for e in ('figure', 'table', 'longtable', 'procedure', 'formula', 'enumerate', 'itemize', 'description',
                  'landscape', 'singlespace', 'tikzpicture')}


def rep(old, new, count=1):
    global S
    n = S.count(old)
    assert n == count, (n, count, old[:90])
    S = S.replace(old, new)


def between(start, end, frm=0):
    i = S.index(start, frm)
    j = S.index(end, i)
    return i, j


C = '% ===========================================================================\n'

# ---------------------------------------------------------------- preambulo: fuente Poppins
rep('\\setsansfont{Arial}\n', '\\setsansfont{Arial}\n% Fuente con licencia OFL para la linea institucional de la caratula (Guia N.o 01).\n'
    '\\newfontfamily\\poppins[Path=fonts/,BoldFont=Poppins-Bold.ttf]{Poppins-Bold.ttf}\n')

# ---------------------------------------------------------------- caratula (R-09)
i, j = between('\\begin{titlepage}', '\\end{titlepage}')
j += len('\\end{titlepage}')
CARATULA = r'''\begin{titlepage}
\thispagestyle{empty}
\begin{singlespace}
\setlength{\parskip}{0pt}
\begin{center}
{\poppins\bfseries\fontsize{22}{26}\selectfont UNIVERSIDAD NACIONAL DE INGENIERÍA\par}
\vspace{0.35cm}
{\bfseries\fontsize{16}{19}\selectfont UNIDAD DE POSGRADO DE LA FACULTAD DE INGENIERÍA INDUSTRIAL Y DE SISTEMAS\par}
{\bfseries\fontsize{16}{19}\selectfont ESCUELA DE POSGRADO\par}
\vspace{0.35cm}
\includegraphics[width=6.7cm]{logo_UNI.png}
\vspace{0.25cm}

{\bfseries\fontsize{14}{17}\selectfont PLAN DE TESIS\par}
\vspace{0.25cm}
{\bfseries\fontsize{12}{14.4}\selectfont\MakeUppercase{\tituloTesis}\par}

\vfill
{\fontsize{12}{14.4}\selectfont
PARA OBTENER EL GRADO ACADÉMICO DE\par
MAESTRO CON MENCIÓN EN:\par
INTELIGENCIA ARTIFICIAL\par}

\vspace{0.55cm}
{\bfseries\fontsize{12}{14.4}\selectfont ELABORADA POR:\par}
{\bfseries\fontsize{12}{14.4}\selectfont LUIS ENRIQUE KOC GÓNGORA\par}
{\bfseries\fontsize{12}{14.4}\selectfont HERBERT ANTONIO MELÉNDEZ GARCÍA\par}

\vspace{0.5cm}
{\bfseries\fontsize{12}{14.4}\selectfont ASESOR: SAMUEL OPORTO\par}

\vfill
{\bfseries\fontsize{14}{17}\selectfont LIMA, PERÚ\par}
{\bfseries\fontsize{12}{14.4}\selectfont 2026\par}
\end{center}
\end{singlespace}
\end{titlepage}'''
S = S[:i] + CARATULA + S[j:]

# ---------------------------------------------------------------- datos generales (R-06)
rep('''Tesistas & \\mbox{Luis Enrique Koc Góngora};\\newline
           \\mbox{Herbert Antonio Meléndez García}. \\\\
\\addlinespace[5pt]
''', '''Tesistas & \\mbox{Luis Enrique Koc Góngora};\\newline
           \\mbox{Herbert Antonio Meléndez García}. \\\\
\\addlinespace[5pt]
Asesor & Samuel Oporto. \\\\
\\addlinespace[5pt]
''')
rep('''Duración estimada & Doce meses, desde julio de 2026 hasta junio de 2027. \\\\
\\addlinespace[5pt]
''', '''Duración estimada & Doce meses, desde julio de 2026 hasta junio de 2027. \\\\
\\addlinespace[5pt]
Fecha de inicio & Julio de 2026. \\\\
\\addlinespace[5pt]
Fecha de culminación prevista & Junio de 2027. \\\\
\\addlinespace[5pt]
''')

# ---------------------------------------------------------------- Cap. I
rep('la máxima demanda del SEIN aumentó de', 'la máxima demanda del Sistema Eléctrico Interconectado Nacional (SEIN) aumentó de')
rep('del Sistema Eléctrico Nacional (SEIN) aumentó de', 'del SEIN aumentó de')
rep('\\section{INTRODUCCIÓN}\n', '\\section{DIAGNÓSTICO}\n\\label{sec:diagnostico}\n')
i, j = between('El primer capítulo desarrolla la evidencia de la problemática', '\n\n\\section{PROBLEMÁTICA}')
S = S[:i] + ('El primer capítulo desarrolla el diagnóstico, identifica el problema con sus antecedentes bibliográficos, lo formula y '
             'delimita su justificación y sus alcances. El segundo convierte esa formulación en objetivos, hipótesis y dimensiones '
             'operativas. El tercero reúne las bases teóricas y la definición de términos. El cuarto presenta la metodología DSR y la '
             'estrategia de verificación. El quinto administra el plan: cronograma, presupuesto y financiamiento. Las referencias y los '
             'anexos documentan las fuentes, la consistencia, la operacionalización, el diseño discreto de casos y la revisión documental.') + S[j:]
rep('\\section{PROBLEMÁTICA}\n', '\\subsection{PROBLEMÁTICA}\n')
rep('compararon el uso de PAC ($k=', 'compararon el uso de hormigón de agregado precolocado (PAC, $k=')
rep('%\\section{IDENTIFICACIÓN Y DESCRIPCIÓN ABSTRACTA DEL PROBLEMA DE ESTUDIO}\n\n\\section{PLANTEAMIENTO DEL PROBLEMA}\n\\label{sec:planteamiento-problema}\n',
    '\\section{IDENTIFICACIÓN Y DESCRIPCIÓN DEL PROBLEMA DE ESTUDIO}\n\\label{sec:planteamiento-problema}\n')

# --- antecedentes: cortar del Marco teorico y reinsertar como 1.2.1
i, j = between('\\section{ANTECEDENTES DE LA INVESTIGACIÓN}\n\\label{sec:antecedentes}\n', '\\section{BASES CONCEPTUALES DEL PROBLEMA}')
S = S[:i] + S[j:]
ANT = r'''\subsection{ANTECEDENTES BIBLIOGRÁFICOS}
\label{sec:antecedentes}

La Tabla~\ref{tab:antecedentes-cronologia} ordena los antecedentes de la fuente más antigua a la más reciente; los apartados siguientes los agrupan por tema y también siguen ese orden dentro de cada tema, salvo las revisiones, que se citan junto con el trabajo al que se refieren. Ocho de las 19 fuentes son anteriores a 2021 y se conservan por ser clásicos (\textcite{neher1957}), normas de la Comisión Electrotécnica Internacional (\textit{International Electrotechnical Commission}, IEC) \parencite{iec60853}, fundamentos metodológicos (\textcite{peffers2007}; \textcite{gregor2013}), el enfoque PINN original (\textcite{raissi2019}) o estudios y revisiones que siguen siendo referencia del tema (\textcite{aras2005}; \textcite{oclon2015}; \textcite{enescu2020}). Las 11 restantes son de 2021 o posteriores.

\begingroup\footnotesize\setlength{\tabcolsep}{3pt}
\begin{longtable}{L{1.1cm}L{3.4cm}L{7.6cm}L{2.4cm}}
\caption{Evolución cronológica de los antecedentes, de la fuente más antigua a la más reciente. Fuente: Elaboración propia a partir de las referencias citadas.}\label{tab:antecedentes-cronologia}\\
\toprule
Año & Fuente & Aporte al problema & Naturaleza \\
\midrule\endfirsthead
\toprule
Año & Fuente & Aporte al problema & Naturaleza \\
\midrule\endhead
1957 & \textcite{neher1957} & Formulación clásica de pérdidas, resistencias térmicas y elevación de temperatura. & Clásico \\
2002 & \textcite{iec60853} & Corriente cíclica y de emergencia: ciclos de carga y secado parcial. & Norma \\
2005 & \textcite{aras2005} & Comparación de enfoques analíticos y numéricos para cables enterrados; FEM representa fuentes cercanas. & Estudio de referencia \\
2007 & \textcite{peffers2007} & Seis actividades de la ciencia del diseño. & Método \\
2013 & \textcite{gregor2013} & Artefacto situado y contribuciones abstractas; evaluación por validez, utilidad, calidad y eficacia. & Método \\
2015 & \textcite{oclon2015} & Optimización del lecho de asiento: una región de alta conductividad mejora la disipación. & Estudio técnico \\
2019 & \textcite{raissi2019} & Enfoque PINN para problemas directos e inversos. & Fundamento PINN \\
2020 & \textcite{enescu2020} & Revisión de la evaluación térmica y del rating; métodos analíticos, circuitales y de campo. & Revisión \\
2021 & \textcite{enescu2021} & Revisión de conceptos y métodos de rating térmico dinámico; influencia del suelo y del secado. & Revisión \\
2022 & \textcite{lawal2022} & Revisión sistemática de PINN: ventajas y dificultades de entrenamiento. & Revisión \\
2023 & \textcite{xing2023} & PINN para conducción de calor anisótropa estacionaria 3D. & Estudio técnico \\
2023 & \textcite{iec60287} & Ecuaciones de corriente admisible y pérdidas en régimen permanente. & Norma \\
2024 & \textcite{atoccsa2024} & Optimización de ampacidad con relleno térmico en cables de alta tensión. & Estudio técnico \\
2024 & \textcite{delport2024} & Ciclos de evaluación durante el desarrollo; criterios de abstracción, originalidad, justificación y beneficio. & Método \\
2025 & \textcite{khumalo2025} & Relación entre humedad, resistividad, profundidad y temperatura del suelo. & Estudio técnico \\
2025 & \textcite{kim2025} & Reducción de temperatura con un material de relleno de mayor conductividad. & Estudio técnico \\
2025 & \textcite{pan2025} & Reconstrucción de un campo térmico 2D y de sus condiciones de borde con datos limitados. & Estudio técnico \\
2025 & \textcite{ren2025} & Revisión de la evolución metodológica de las PINN y de sus dificultades de entrenamiento. & Revisión \\
2025 & \textcite{cigre2025} & Casos heterogéneos y verificación de las herramientas de rating por elementos finitos. & Recomendación técnica \\
\bottomrule
\end{longtable}\endgroup

\subsubsection{MODELOS TÉRMICOS Y AMPACIDAD}

\textcite[752--772]{neher1957} establecieron una formulación clásica para relacionar pérdidas, resistencias térmicas y elevación de temperatura. Por tanto, su trabajo es una base histórica de los métodos de \textit{rating}.

La IEC 60853, de 2002, amplía el tratamiento a ciclos de carga y secado parcial \parencite[cláusulas 2--4]{iec60853}. \textcite[1387--1400]{aras2005} compararon enfoques analíticos y numéricos para cables enterrados; sus resultados muestran que FEM representa fuentes cercanas y distribuciones de temperatura mediante geometría explícita.

Las revisiones de Enescu y colaboradores ordenan los métodos analíticos, circuitales y de campo \parencite[4,11,17]{enescu2020}. Además, señalan que la conductividad del suelo, las capas y el secado son factores dominantes. La revisión sobre capacidad térmica dinámica añade una relación directa para el estudio: la resistividad térmica del suelo influye en la ampacidad, y un valor mayor la reduce \parencite[3--6]{enescu2021}.

La IEC 60287, en su edición de 2023, formaliza ecuaciones de corriente admisible y pérdidas para régimen permanente \parencite[cláusulas 4--5]{iec60287}. Además, las recomendaciones del Consejo Internacional de Grandes Redes Eléctricas (CIGRÉ) de 2025 documentan casos heterogéneos y exigen verificación antes de confiar en una herramienta de \textit{rating} \parencite[3--4,41--42,93--99]{cigre2025}. Por ello, la verificación será un criterio central del estudio.

Estos métodos son referencias necesarias para comparación. Sin embargo, no sustituyen una solución de campo cuando la topología térmica no cumple sus supuestos \parencites{iec60287}{iec60853}[3--4,93--99]{cigre2025}.

\subsubsection{HETEROGENEIDAD, SUELO NATIVO Y SUELO DE RELLENO}

Los trabajos de optimización del lecho de asiento y del suelo de relleno muestran que una región de alta conductividad puede mejorar la disipación \parencites[1--3]{oclon2015}[1--4]{atoccsa2024}. Sin embargo, se apoyan principalmente en FEM y optimización externa.

\textcite[1,6--9]{khumalo2025} aportan evidencia sobre humedad, resistividad, profundidad y temperatura ambiente del suelo. Por su parte, \textcite[1,10,16]{kim2025} cuantifican la reducción de temperatura lograda con un suelo de relleno más conductor. En conjunto, ambos trabajos muestran que $k$ no es una constante secundaria. Por tanto, debe tratarse como variable de diseño e incertidumbre. El aporte previsto de este estudio es distinto, porque busca trasladar la física a una PINN y evaluar reglas sobre contraste, posición y extensión de regiones heterogéneas.

\subsubsection{INTELIGENCIA ARTIFICIAL INFORMADA POR LA FÍSICA}

\textcite[686--690]{raissi2019} consolidaron el enfoque PINN para problemas directos e inversos al combinar datos y residuos de ecuaciones diferenciales. Las revisiones posteriores confirman su expansión, pero también identifican entrenamiento lento, competencia entre términos de pérdida, dificultad multiescala y necesidad de estrategias adaptativas o descomposición de dominios \parencites[1--2,14--17]{lawal2022}[2--6]{ren2025}.

En conducción de calor, \textcite[1,3,17]{xing2023} demostraron una PINN para un problema anisotrópico estacionario 3D y reportaron, en uno de sus ejemplos, error relativo de 0,23\,\pct. \textcite[1,3,16]{pan2025} reconstruyeron un campo térmico 2D y condiciones de borde con datos limitados, con error absoluto inferior a \SI{1}{\celsius} en el dominio de validación declarado. Estos resultados prueban viabilidad en problemas térmicos controlados, pero no eliminan la necesidad de evaluar interfaces, pérdidas internas y geometrías de cables.

\subsubsection{ANTECEDENTES METODOLÓGICOS DSR}

\textcite[54--56]{peffers2007} proponen seis actividades: identificar y motivar el problema, definir objetivos de solución, diseñar y desarrollar, demostrar, evaluar y comunicar. La secuencia no es rígida y permite iterar cuando la evidencia de evaluación no alcanza los objetivos.

\textcite[342--350]{gregor2013} distinguen el artefacto situado de las contribuciones más abstractas, tales como métodos y principios de diseño. También proponen evaluar validez, utilidad, calidad y eficacia, y comunicar con claridad el problema, el artefacto y la evidencia. \textcite[198--203]{delport2024} complementan esta visión con ciclos continuos de evaluación durante el desarrollo y con los principios de abstracción, originalidad, justificación y beneficio.

\subsubsection{UBICACIÓN DEL PROYECTO}

El proyecto se ubica en la intersección de tres líneas de trabajo: el cálculo térmico y la ampacidad de cables enterrados, las redes neuronales informadas por física y la ciencia del diseño de artefactos. Se desarrollará como tesis profesionalizante del programa de Maestría en Inteligencia Artificial de la Unidad de Posgrado de la Facultad de Ingeniería Industrial y de Sistemas de la Universidad Nacional de Ingeniería.

Respecto de la primera línea, no reemplaza las normas ni la referencia FEM: usará la IEC 60287 como línea base cuando el caso sea comparable y FEM como referencia independiente del campo. Respecto de la segunda, aportará una comparación trazable de una formulación PINN por material, con transmisión de temperatura y de flujo en las interfaces, aplicada a configuraciones de cable enterrado. Respecto de la tercera, corresponde a un artefacto situado con principios de diseño derivados, en el sentido de \textcite[342--350]{gregor2013}. La brecha descrita en la Sección~\ref{sec:planteamiento-problema} delimita el aporte: la comparación específica entre esas tres líneas, y no una novedad general sobre ninguna de ellas.

'''
rep('\\subsection{FORMULACIÓN DEL PROBLEMA}\n', ANT + '\\subsection{FORMULACIÓN DEL PROBLEMA}\n')
rep('del ANEXO 5.1, la cual funciona', 'del \\ref{anx:matriz-consistencia}, la cual funciona')
rep('descritas en la Sección~\\ref{sec:metodologia}.', 'descritas en el Capítulo~\\ref{ch:metodologia}.')
rep('sostienen la metodología de la sección~\\ref{sec:metodologia}', 'sostienen la metodología del Capítulo~\\ref{ch:metodologia}')

# justificacion y alcances (1.2.3)
rep('\\subsection{JUSTIFICACIÓN}\n\n\\subsubsection{JUSTIFICACIÓN PRÁCTICA}\n',
    '\\subsection{JUSTIFICACIÓN Y ALCANCES}\n\n\\subsubsection{JUSTIFICACIÓN}\n\n\\paragraph{Justificación práctica.}\n')
rep('\\subsubsection{JUSTIFICACIÓN TECNOLÓGICA}', '\\paragraph{Justificación tecnológica.}')
rep('\\subsubsection{JUSTIFICACIÓN METODOLÓGICA}', '\\paragraph{Justificación metodológica.}')
rep('\\subsubsection{JUSTIFICACIÓN CIENTÍFICA Y ACADÉMICA}', '\\paragraph{Justificación científica y académica.}')
rep('\\section{ALCANCES Y LIMITACIONES}\n\n\\subsection{ALCANCES}\n', '\\subsubsection{ALCANCES Y LIMITACIONES}\n\n\\paragraph{Alcances.}\n')
rep('\\subsection{LIMITACIONES}\n', '\\paragraph{Limitaciones.}\n')

# ---------------------------------------------------------------- Cap. II
rep('\\section{OBJETIVOS}\n\n\\subsection{OBJETIVO GENERAL}\n', '\\section{OBJETIVO GENERAL}\n')
rep('\\subsection{OBJETIVOS ESPECÍFICOS}\n', '\\section{OBJETIVOS ESPECÍFICOS}\n')
rep('error relativo de $\\Tmax$ no mayor a 5\\,\\pct{}, NRMSE no mayor a 5\\,\\pct{}',
    'error relativo de $\\Tmax$ no mayor a 5\\,\\pct{}, error cuadrático medio normalizado (NRMSE) no mayor a 5\\,\\pct{}')

# llamadas a figuras y tablas sin cita (R-10)
rep('\\subsection{DISEÑO DISCRETO DE CASOS Y PARÁMETROS CONTROLADOS}\n\n',
    '\\subsection{DISEÑO DISCRETO DE CASOS Y PARÁMETROS CONTROLADOS}\n\nLa Figura~\\ref{fig:modelo-conceptual} muestra el modelo entrada--proceso--salida de las dimensiones operativas y la Tabla~\\ref{tab:dimensiones-operativas} registra las variables discretas y sus estados por etapa.\n\n')
rep('\\label{subsec:sintesis-dsr}\n\n',
    '\\label{subsec:sintesis-dsr}\n\nLa Figura~\\ref{fig:sintesis-metodologica-dsr} ubica la sistematización y los principios de diseño dentro de la metodología DSR.\n\n')
rep('\\label{anx:operacionalizacion-variables}\n\\thispagestyle{empty}\n\n',
    '\\label{anx:operacionalizacion-variables}\n\\thispagestyle{empty}\n\nLa Tabla~\\ref{tab:operacionalizacion-variables} presenta la matriz de operacionalización por etapa.\n\n')
rep('\\label{anx:diseno-casos}\n\\thispagestyle{empty}\n\n',
    '\\label{anx:diseno-casos}\n\\thispagestyle{empty}\n\nLa Tabla~\\ref{tab:diseno-casos} presenta el diseño discreto de casos y los parámetros físicos de cobertura.\n\n')

# ---------------------------------------------------------------- Metodologia <-> Marco teorico
MET_START = C + '\\chapter{METODOLOGÍA DE LA INVESTIGACIÓN}'
MT_START = C + '\\chapter{MARCO TEÓRICO}'
BIB_START = C + '\\begin{singlespace}\n\\tesisbodyfont\n\\printbibliography'
assert S.count(MET_START) == 1 and S.count(MT_START) == 1 and S.count(BIB_START) == 1
a = S.index(MET_START); b = S.index(MT_START); c = S.index(BIB_START)
assert a < b < c
PRE, MET, MT, POST = S[:a], S[a:b], S[b:c], S[c:]

# --- Marco teorico
def rep_in(block, old, new, count=1):
    n = block.count(old)
    assert n == count, (n, count, old[:90])
    return block.replace(old, new)

MT = rep_in(MT, 'Este capítulo reúne los antecedentes y las bases que sostienen el problema, el artefacto y su evaluación. Integra',
            'Este capítulo reúne las bases teóricas que sostienen el problema, el artefacto y su evaluación, y precisa los términos que se usarán. Integra')
MT = rep_in(MT, 'con el modelo físico y la metodología.\n',
            'con el modelo físico y la metodología. Los antecedentes bibliográficos se presentan en la Sección~\\ref{sec:antecedentes}.\n')
k = MT.index('\\section{BASES CONCEPTUALES DEL PROBLEMA}')
d = MT.index('\\section{DEFINICIÓN DE TÉRMINOS}')
mid = MT[k:d]
assert '\\subsubsection' not in mid
mid = mid.replace('\\subsection{', '\\subsubsection{')
mid = mid.replace('\\section{BASES CONCEPTUALES DEL PROBLEMA}', '\\subsection{BASES CONCEPTUALES DEL PROBLEMA}')
mid = mid.replace('\\section{TÉCNICAS APLICADAS}', '\\subsection{TÉCNICAS APLICADAS}')
assert '\\section{' not in mid
INTRO_BT = ('\\section{BASES TEÓRICAS}\n\\label{sec:bases-teoricas}\n\n'
            'Las bases teóricas comprenden las bases conceptuales del problema (Sección~\\ref{sec:bases-conceptuales}) y las técnicas '
            'aplicadas (Sección~\\ref{sec:tecnicas-aplicadas}).\n\n')
MT = MT[:k] + INTRO_BT + mid + MT[d:]

# --- Metodologia
MET = rep_in(MET, '\\section{METODOLOGÍA}\n\\label{sec:metodologia}\n\n', '@@SEC_METODOLOGIA@@')
i1 = MET.index('@@SEC_METODOLOGIA@@')
p_prec_start = MET.index('\\paragraph{Precisamiento de la formulación térmica y eléctrica.}')
p_tipo_start = MET.index('La investigación tendrá enfoque cuantitativo')
p_met_start = MET.index('Como maestría de especialización')
sub_adapt = MET.index('\\subsection{ADAPTACIÓN DE LA CIENCIA DEL DISEÑO}')
assert i1 < p_prec_start < p_tipo_start < p_met_start < sub_adapt
P_PREC = MET[p_prec_start:p_tipo_start]
P_TIPO = MET[p_tipo_start:p_met_start]
P_MET = MET[p_met_start:sub_adapt]
NIVEL = ('El nivel de la investigación será explicativo y comparativo, con orientación tecnológica. Es explicativo porque busca establecer '
         'cómo el contraste, el patrón, la extensión y la proximidad de las heterogeneidades de $\\kx$ modifican $T(x,y)$, $\\Tmax$ e $\\Imax$; '
         'es comparativo porque contrasta pares controlados de representación heterogénea y homogénea equivalente; y tiene orientación '
         'tecnológica porque incluye la construcción y la verificación del artefacto PINN.\n\n')
NEW_HEAD = ('\\section{TIPO DE INVESTIGACIÓN}\n\\label{sec:metodologia}\n\n' + P_TIPO.rstrip() + '\n\n'
            '\\section{NIVEL DE INVESTIGACIÓN}\n\n' + NIVEL +
            '\\section{MÉTODOS DE TRABAJO}\n\n' + P_MET.rstrip() + '\n\n' + P_PREC.rstrip() + '\n\n')
MET = MET[:i1] + NEW_HEAD + MET[sub_adapt:]
MET = rep_in(MET, '\\subsection{CONTEXTO DE APLICACIÓN Y UNIDAD DE ANÁLISIS}',
             '\\section{POBLACIÓN Y MUESTRA}\n\n'
             'La población de referencia es el conjunto de configuraciones de cables enterrados en entornos térmicamente heterogéneos dentro del dominio 2D estacionario delimitado. '
             'Por tratarse de una maestría de especialización, no se extraerá una muestra probabilística: se seleccionará de forma intencional un conjunto de casos, '
             'realizaciones de $R_2$, $E_3$ y $C_4$, descrito en el \\ref{anx:diseno-casos}.\n\n'
             '\\subsection{CONTEXTO DE APLICACIÓN Y UNIDAD DE ANÁLISIS}')
MET = rep_in(MET, '\\subsection{EXPERIMENTACIÓN NUMÉRICA DEL ARTEFACTO DENTRO DE DSR}',
             '\\section{TIPO DE DISEÑO}\n\n'
             'El diseño es de ciencia del diseño (DSR), con ciclos formativos de evaluación y comparaciones pareadas entre representaciones heterogéneas y homogéneas equivalentes '
             'sobre casos discretos. No es un diseño experimental de campo ni un diseño factorial estadístico.\n\n'
             '\\subsection{EXPERIMENTACIÓN NUMÉRICA DEL ARTEFACTO DENTRO DE DSR}')
MET = rep_in(MET, '\\subsection{EVIDENCIA, INSTRUMENTOS Y TRAZABILIDAD}',
             '\\section{TÉCNICAS E INSTRUMENTOS DE RECOLECCIÓN DE DATOS}\n\n'
             'Los datos se recolectarán mediante revisión documental y registro de ejecuciones computacionales. Los instrumentos y la trazabilidad se describen a continuación.\n\n'
             '\\subsection{EVIDENCIA, INSTRUMENTOS Y TRAZABILIDAD}')
MET = rep_in(MET, '\\subsection{MÉTRICAS DE EVALUACIÓN}',
             '\\section{TÉCNICAS E INSTRUMENTOS DE ANÁLISIS Y PROCESAMIENTO DE DATOS}\n\n'
             'El análisis comparará los resultados del artefacto con referencias independientes mediante las métricas y los criterios que siguen.\n\n'
             '\\subsection{MÉTRICAS DE EVALUACIÓN}')
MET = rep_in(MET, '\\subsection{ETAPAS DE INTERVENCIÓN DEL ESTUDIO}', '\\section{ETAPAS DE INTERVENCIÓN DEL ESTUDIO}')
MET = rep_in(MET, 'MAE o RMSE espacial', 'el error absoluto medio (MAE) o el error cuadrático medio (RMSE) espacial')
MET = rep_in(MET, 'IEC se usará como línea base normativa', 'La IEC 60287 se usará como línea base normativa')

# ---------------------------------------------------------------- Cap. V
CAPV = r'''% ===========================================================================
\chapter{ADMINISTRACIÓN DEL PLAN DE TESIS}
\label{ch:administracion}
\thispagestyle{empty}

Este capítulo administra el plan: programa las actividades en el tiempo, identifica los rubros de gasto y declara la fuente de financiamiento.

\section{CRONOGRAMA}
\label{sec:cronograma}

El estudio se desarrolla en doce meses, de julio de 2026 a junio de 2027. La Tabla~\ref{tab:cronograma} presenta el diagrama de Gantt de octubre de 2026 a junio de 2027; el primer bloque resume julio a septiembre de 2026, ya cumplido, con las fases del Procedimiento~\ref{proc:etapas}. Las fechas posteriores a septiembre de 2026 son una programación sujeta a la aprobación del asesor.

\providecommand{\gc}{\cellcolor{darkblue}}
\providecommand{\gp}{\cellcolor{darkblue!35}}
\begin{table}[htbp]
\centering
\caption{Cronograma del plan de tesis en diagrama de Gantt. Azul oscuro: cumplido a septiembre de 2026; azul claro: programado. Fuente: Elaboración propia.}
\label{tab:cronograma}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{C{0.7cm}L{4.3cm}*{10}{C{0.82cm}}}
\toprule
& & & \multicolumn{3}{c}{\textbf{2026}} & \multicolumn{6}{c}{\textbf{2027}} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-12}
\textbf{Fase} & \textbf{Actividad} & \textbf{\makecell{Jul--\\sep}} & \textbf{Oct} & \textbf{Nov} & \textbf{Dic} & \textbf{Ene} & \textbf{Feb} & \textbf{Mar} & \textbf{Abr} & \textbf{May} & \textbf{Jun} \\
\midrule
@@GANTT_ROWS@@
\bottomrule
\end{tabular}
\end{table}

Los hitos del plan son: H1, especificación trazable y referencias FEM verificadas (cumplido en el primer bloque); H2, campaña A--D y confirmación con semillas nuevas (cumplido a septiembre de 2026); H3, decisión sobre la ronda E (octubre de 2026); H4, expedientes térmico y comparativo de ampacidad (diciembre de 2026 a febrero de 2027); H5, borrador completo de la tesis (marzo de 2027); y H6, revisión del asesor, depósito y sustentación (abril a junio de 2027).

\section{PRESUPUESTO}
\label{sec:presupuesto}

La Tabla~\ref{tab:presupuesto} organiza los gastos en los rubros de la Guía N.° 01: equipos, materiales e insumos, servicios, y viáticos y publicación. El equipo computacional ya existe y el software es de código abierto, por lo que no generan desembolso adicional. Los montos por confirmar dependen de tarifas y cotizaciones que los tesistas verificarán antes de ejecutar el gasto; el total se consolidará con esos montos.

\begin{table}[htbp]
\centering
\caption{Presupuesto del plan de tesis por rubros. Fuente: Elaboración propia.}
\label{tab:presupuesto}
\scriptsize
\begin{tabularx}{\textwidth}{L{3.0cm}Y L{2.5cm}L{2.2cm}}
\toprule
\textbf{Rubro} & \textbf{Concepto} & \textbf{Situación} & \textbf{Monto (S/)} \\
\midrule
a) Equipos & Computadora personal con CPU Intel Core Ultra 7 165U para entrenamientos y referencias FEM. & Existente, de los tesistas & 0,00 \\
b) Materiales e insumos & Software de código abierto: FEniCSx, Gmsh, PETSc, PyTorch y LaTeX. & Sin costo de licencia & 0,00 \\
b) Materiales e insumos & Impresión y encuadernación de los ejemplares del informe final. & Por cotizar & Por confirmar \\
c) Servicios & Acceso a Internet y energía eléctrica durante el desarrollo. & Existente, de los tesistas & Por confirmar \\
c) Servicios & Derechos académicos y administrativos de sustentación y grado. & Según el tarifario de la Universidad & Por confirmar \\
d) Viáticos y publicación & Movilidad a reuniones de asesoría y a la sustentación. & Por definir & Por confirmar \\
d) Viáticos y publicación & Publicación del manuscrito derivado de la tesis. & Sujeta a la revista elegida & Por confirmar \\
\bottomrule
\end{tabularx}
\end{table}

\section{FINANCIAMIENTO}
\label{sec:financiamiento}

El plan será autofinanciado: los tesistas asumirán los gastos con fondos propios. No se han solicitado fondos concursables ni financiamiento de terceros, y no se requiere un compromiso económico adicional de la Universidad.

'''

# filas del Gantt: columnas 0 = jul-sep 2026 (cumplido), 1..9 = oct 2026 .. jun 2027
GANTT = [
    ('1', 'Revisión y especificación', [0], []),
    ('2', 'Diseño lógico y prototipo', [0], []),
    ('3', 'Construcción, verificación y piloto (campaña A--D)', [0], []),
    ('3E', 'Ronda E de desarrollo, opcional y sujeta a aprobación', [], [1, 2]),
    ('4', 'Demostración y evaluación térmica', [], [1, 2, 3]),
    ('5', 'Comparación de ampacidad', [], [2, 3, 4, 5]),
    ('6', 'Síntesis DSR', [], [5, 6]),
    ('7', 'Comunicación: redacción, revisión del asesor y sustentación', [], [6, 7, 8, 9]),
]
rows = []
for fase, act, done, plan in GANTT:
    cells = []
    for k in range(10):
        cells.append('\\gc' if k in done else '\\gp' if k in plan else '')
    rows.append(fase + ' & ' + act + ' & ' + ' & '.join(cells) + ' \\\\')
CAPV = CAPV.replace('@@GANTT_ROWS@@', '\n'.join(rows))
assert '@@' not in CAPV

S = PRE + MT + MET + CAPV + POST

# ---------------------------------------------------------------- referencias (bloque propio) y anexos
rep('\\printbibliography[heading=subbibnumbered,title={REFERENCIAS}]', '\\printbibliography[heading=bibintoc,title={REFERENCIAS}]')
rep('''\\begin{singlespace}
\\tesisbodyfont
\\renewcommand{\\thesection}{ANEXO \\thechapter.\\arabic{section}}
\\titleformat{\\section}
  {\\normalfont\\bfseries\\fontsize{11}{13.2}\\selectfont}
  {\\thesection.}{0.75em}{\\MakeUppercase}
\\titlespacing*{\\section}{0pt}{1.0\\baselineskip}{0.25\\baselineskip}
\\begin{landscape}
\\chapter{ANEXOS}
\\thispagestyle{empty}
''', '''\\renewcommand{\\thetable}{A.\\arabic{table}}
\\renewcommand{\\thefigure}{A.\\arabic{figure}}
\\renewcommand{\\theHtable}{anexo.\\arabic{table}}
\\renewcommand{\\theHfigure}{anexo.\\arabic{figure}}
\\renewcommand{\\theHsection}{anexo.\\arabic{section}}
\\setcounter{table}{0}
\\setcounter{figure}{0}
\\setcounter{section}{0}
\\begin{singlespace}
\\tesisbodyfont
\\renewcommand{\\thesection}{ANEXO \\arabic{section}}
\\titleformat{\\section}
  {\\normalfont\\bfseries\\fontsize{11}{13.2}\\selectfont}
  {\\thesection.}{0.75em}{\\MakeUppercase}
\\titlespacing*{\\section}{0pt}{1.0\\baselineskip}{0.25\\baselineskip}
\\begin{landscape}
\\chapter*{ANEXOS}
\\addcontentsline{toc}{chapter}{ANEXOS}
\\thispagestyle{empty}
''')
rep('\\section{MATRIZ DE CONSISTENCIA}\n\\thispagestyle{empty}\n', '\\section{MATRIZ DE CONSISTENCIA}\n\\label{anx:matriz-consistencia}\n\\thispagestyle{empty}\n')
rep('\\renewcommand{\\thesection}{ANEXO \\thechapter.\\arabic{section}}', '\\renewcommand{\\thesection}{ANEXO \\arabic{section}}', count=1)

# leyendas sin 'Fuente' (R-10)
n_fuente = 0
def _f(m):
    global n_fuente
    n_fuente += 1
    return m.group(1) + m.group(2) + ' Fuente: Elaboración propia.}'
S = re.sub(r'(?m)^(\\caption(?:\[[^\]]*\])?\{)((?!.*Fuente).*)\}$', _f, S)

# ---------------------------------------------------------------- verificaciones
ENV1 = {e: (len(re.findall(r'\\begin\{%s\}' % e, S)), len(re.findall(r'\\end\{%s\}' % e, S))) for e in ENV0}
for e in ENV0:
    b0, e0 = ENV0[e]; b1, e1 = ENV1[e]
    assert b1 == e1, (e, b1, e1)
    if e in ('figure', 'procedure', 'formula', 'longtable', 'landscape', 'singlespace', 'tikzpicture', 'description', 'enumerate', 'itemize'):
        pass
print('captions con Fuente anadida:', n_fuente)
print('entornos antes/despues:', {e: (ENV0[e][0], ENV1[e][0]) for e in ENV0})
out = S.replace('\n', '\r\n') if crlf else S
open(P, 'w', encoding='utf-8', newline='').write(out)
print('ok; crlf =', crlf)
