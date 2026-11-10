"""One-time coherent revision of existing theoretical and problem chapters."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CH=ROOT/'Tesis_LaTeX_Borrador_UNI/capitulos'

def main():
    path=CH/'01_planteamiento_del_problema.tex';text=path.read_text(encoding='utf-8')
    old='La tesis profesionalizante no plantea una contrastación estadística de hipótesis.'
    if old in text:
        first=text.index(old);last=text.index('\\section{METODOLOGÍA}',first)
        text=text[:first]+r'''\section{HIPÓTESIS Y CRITERIOS DE CONTRASTE}

Se conservan las hipótesis del plan como proposiciones de diseño contrastables mediante evidencia computacional. No se reemplazan por una declaración genérica de éxito ni se realizan inferencias sobre una población de instalaciones a partir de semillas aleatorias. La hipótesis general vincula representación explícita de $k(x,y)$, reproducción de referencias y diferencias térmicas y operativas frente a representaciones homogéneas.

\begin{enumerate}[label=HE\arabic*.]
\item Un paquete físico completo y documentado permite obtener VD1: especificación trazable aceptada, con requisitos vinculados a pruebas.
\item La especificación y las referencias válidas permiten obtener VD2: artefacto aceptado en el dominio declarado, con error de temperatura y NRMSE no mayores a 5\,\pct{}, balance no mayor a 2\,\pct{} y variabilidad menor que el efecto relevante.
\item El artefacto verificado permite obtener VD3: expediente térmico concluyente; una región de baja conductividad cercana aumenta la temperatura y un relleno de mayor conductividad la reduce en los pares controlados.
\item VD3 y los pares de representación permiten obtener VD4: expediente comparativo de ampacidad; promediar una zona local de baja conductividad puede sobreestimar la corriente admisible.
\end{enumerate}

Los estados posibles incluyen aceptación restringida, rechazo e inconclusión. Las tendencias de HE3 y HE4 se contrastan en pares definidos, sin convertirlas en leyes universales. La incertidumbre numérica y la variabilidad entre semillas deben ser menores que la diferencia que se interpreta. No se considera cumplido OE4 por mostrar únicamente temperaturas a corriente fija.

'''+text[last:]
    marker='La metodología ejecutada comprende diecinueve casos'
    if marker in text:
        first=text.index(marker);last=text.index('El capítulo establece',first)
        text=text[:first]+r'''La metodología ejecutada se organiza en selección de arquitectura y tamaño, estudio de colocaciones, ajuste de entrenamiento y confirmación con semillas nuevas. Los casos de desarrollo y confirmación, presupuestos, criterios y reglas de selección se fijan antes de observar la campaña correspondiente y se detallan en la Sección~\ref{sec:seleccion-pinn}. Las referencias FEM conservan exactamente geometría, conductividades, fuentes y contornos del problema PINN.

La ecuación de calor se resuelve en el conductor, cada capa y el suelo. No se reconstruye la temperatura interior mediante una resistencia térmica equivalente. La representación distingue la fuente prescrita, el acoplamiento eléctrico DC local y las estimaciones condicionadas de efecto piel. La futura extensión temporal requiere capacidades y condiciones iniciales documentadas; no se presenta como resultado ejecutado.

Además de los umbrales del plan, la aceptación exige control por material, continuidad térmica y flujo normal en cada interfaz. Las referencias FEM deben cambiar menos de 0,5\,\pct{} de la elevación térmica entre las dos últimas mallas, tanto en Tmax como en RMSE por región; se exige asimismo cambio de Tmax menor a 0,1 K y balance menor a 0,2\,\pct{}. Estas diferencias son indicadores de convergencia observada, no cotas demostradas del error exacto.

'''+text[last:]
    path.write_text(text,encoding='utf-8')
    path=CH/'00_introduccion.tex';text=path.read_text(encoding='utf-8')
    text=text.replace('El Capítulo IV formula conclusiones y recomendaciones dentro del modelo estacionario reducido, y distingue la ampacidad DC acoplada de una ampacidad normativa completa.',
        'El Capítulo IV vincula las conclusiones con los cuatro objetivos y con la evidencia de la formulación explícita. La selección preliminar de red y colocaciones precede a la evaluación final; las conclusiones distinguen fuentes prescritas, acoplamiento DC, sensibilidad al efecto piel y límites pendientes de validación.')
    text=text.replace('El artefacto aproxima $T(x,y)$, estima $\\Tmax$ e $\\Imax$ y conserva registros de verificación.',
        'El artefacto aproxima $T(x,y)$ en conductor, capas y suelo, y obtiene la temperatura máxima del campo interior. El cálculo de $\\Imax$ requiere resolver adicionalmente la condición de corriente límite con la misma ley de pérdidas. Se investiga qué representación y tamaño de red satisfacen los criterios, cuántas colocaciones son necesarias y dónde conviene concentrarlas; una red mayor o una pérdida menor no garantizan una solución más fiel.')
    path.write_text(text,encoding='utf-8')
    path=CH/'02_marco_teorico.tex';text=path.read_text(encoding='utf-8')
    marker=r'\subsection{INTERIOR EXPLÍCITO, ESCALAS Y REPRESENTACIÓN}'
    if marker not in text:
        anchor=r'\subsection{CONDUCTIVIDAD, RESISTIVIDAD Y HETEROGENEIDAD}'
        addition=r'''\subsection{INTERIOR EXPLÍCITO, ESCALAS Y REPRESENTACIÓN}

El dominio se descompone en regiones abiertas $\Omega_i$ de conductor, aislamiento, pantalla, cubierta y suelo. Las propiedades son suaves dentro de cada región; el salto de conductividad se trata mediante trazas laterales y condiciones de transmisión. Diferenciar automáticamente una función indicadora de material no representa la derivada distribucional del salto. La ecuación y las interfaces se imponen por separado.

En un anillo, las coordenadas polares conservan ambas direcciones de difusión:
\begin{formula}[htbp]
\begin{equation}
\rho c_p T_t-\frac{1}{r}\partial_r(rkT_r)-\frac{1}{r^2}\partial_\varphi(kT_\varphi)=Q.
\label{for:polar-completa}
\end{equation}
\caption{Ecuación de calor polar completa en las capas anulares.}
\end{formula}
Eliminar el término angular solo es admisible para una prueba axisimétrica. La proximidad de otros cables y las heterogeneidades del suelo rompen generalmente esa simetría. El núcleo que contiene $r=0$ se representa en cartesianas, donde la regularidad es natural. Una solución 2D regular no exige anular toda derivada radial direccional en el centro.

Para $a\le r\le b$ se emplea $\eta=\log(r/a)/\log(b/a)$, junto con $\cos\varphi$ y $\sin\varphi$. La transformación amplía numéricamente una capa milimétrica sin aumentar su espesor físico. La diferenciación respecto a $x,y$ conserva los jacobianos y el operador completo. En suelo se normalizan las coordenadas globales y se añaden distancias logarítmicas a los cables; estas son entradas geométricas, no temperaturas analíticas impuestas. La combinación de normalización local y descomposición tiene antecedentes en \textcite{moseley2021fbpinn}, aunque no se reproduce su método de ventanas solapadas.

Los residuos se adimensionalizan por región: con longitud $\ell_i$, flujo $q_i^*$ y conductividad representativa $k_i^*$ se toman $T_i^*=q_i^*\ell_i/k_i^*$ y $r_i^*=q_i^*/\ell_i$. Así se evita sumar cantidades con unidades distintas o ponderar un aislamiento únicamente por su pequeña área. El número de puntos de cada región se especifica de forma independiente del área del dominio.

\subsection{FUENTE ELÉCTRICA, RESISTENCIA DC Y EFECTO PIEL}

La ley térmica conserva la PDE incluso si la fuente volumétrica es uniforme. Para acoplamiento DC, con conductor largo y potencial axial común en cada sección, se utiliza la conductividad eléctrica local $\sigma(T)$ y la corriente prescrita:
\begin{formula}[htbp]
\begin{equation}
\sigma(T)=\frac{1}{\rho_{e,0}[1+\alpha(T-T_{e,0})]},\qquad
E_z=\frac{I}{\int_{\Omega_c}\sigma(T)\,dA},\qquad Q=\sigma(T)E_z^2.
\label{for:fuente-dc-local}
\end{equation}
\caption{Fuente Joule DC dependiente de la temperatura local.}
\end{formula}
La integral garantiza la corriente impuesta; no sustituye el problema térmico ni reconstruye la temperatura. Se distingue $\rho_e$, resistividad eléctrica, de $\rho c_p$, capacidad térmica volumétrica. La relación entre resistencia por unidad de longitud y resistividad es $\rho_{e,0}=R_{0}A_c$ bajo la hipótesis homogénea declarada.

Un dato DC no es automáticamente una resistencia AC. Para conductor circular macizo homogéneo aislado se puede resolver el campo electromagnético cilíndrico. Con fasores RMS, $\delta=\sqrt{\rho_e/(\pi f\mu)}$, $z=(1-\mathrm i)a/\delta$ y funciones de Bessel $J_0,J_1$:
\begin{formula}[htbp]
\begin{equation}
\frac{R_{\mathrm{AC}}}{R_{\mathrm{DC}}}=\Re\!\left[\frac{zJ_0(z)}{2J_1(z)}\right],\qquad
J_z(r)=\frac{I\gamma J_0(\gamma r)}{2\pi aJ_1(\gamma a)},\quad
Q(r)=\rho_e|J_z(r)|^2,\quad\gamma=(1-\mathrm i)/\delta.
\label{for:piel-volumetrica}
\end{equation}
\caption{Estimación cilíndrica del efecto piel y su fuente volumétrica.}
\end{formula}
El integral de $Q$ da $I^2R_{\mathrm{AC}}$. Una resistencia ya medida a 60 Hz no se corrige otra vez por piel, y debe identificarse su temperatura. No se infiere automáticamente su dependencia térmica multiplicándola por el factor DC. Una resistencia AC tampoco identifica por sí sola el perfil de corriente. La proximidad, pantallas, armaduras y conductores segmentados exigen información adicional; \textcite{patel2013skin} presentan un tratamiento conjunto de piel y proximidad que no se reemplaza aquí por la estimación aislada.

Se separan dos sensibilidades: cambiar la potencia total DC a AC y redistribuir la misma potencia dentro del conductor. La comparación a igual potencia evita atribuir a la distribución espacial un cambio causado por pérdidas distintas. La pequeña diferencia térmica estacionaria de un perfil no justifica eliminar el interior de la PDE, ni demuestra una equivalencia para transitorios. La estimación eléctrica usa temperatura prescrita; el acoplamiento electromagnético AC con temperatura no uniforme queda fuera de esta implementación.

'''
        text=text.replace(anchor,addition+anchor)
    text=text.replace('Esta forma permite representar discontinuidades entre materiales sin sustituirlas por una propiedad global.',
        'Esta forma se aplica dentro de cada región material; las discontinuidades se completan con condiciones de transmisión, sin sustituirlas por una propiedad global.')
    text=text.replace('La Fórmula~\\ref{for:perdida-pinn} reúne restricciones físicas y datos de referencia cuando existen.',
        'La Fórmula~\\ref{for:perdida-pinn} describe una familia general. En la investigación ejecutada se fija $w_d=0$: las temperaturas FEM no son etiquetas de entrenamiento. La variante mixta añade la ley constitutiva del flujo y todas las variantes conservan la PDE interior.')
    text=text.replace('Su uso debe justificarse mediante evidencia del prototipo básico y no asumirse como requisito desde el inicio.',
        'Para interfaces con flujo no nulo y salto de k, la descomposición permite derivadas normales distintas. Una única red global C1 comparte la derivada a ambos lados y no puede satisfacer exactamente ese salto; se considera un control de limitación arquitectónica, no una candidata admisible por aumentar sus neuronas.')
    path.write_text(text,encoding='utf-8')

if __name__=='__main__':main()
