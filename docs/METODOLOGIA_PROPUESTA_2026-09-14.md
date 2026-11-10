# Metodología para la tesis: conducción 2D estacionaria en cables enterrados

Fecha: 14 de septiembre de 2026. Documento vivo de especificación y revisión técnica. Actualizado por instrucción del tesista: la ecuación de calor se resuelve directamente dentro de todas las capas en todas las variantes PINN activas. Referencia de alcance: `Plan/plan_tesis_cables_pinn.tex`, capítulos 1–3 y anexo de casos; contrastada con la copia textual `docs/auditoria/drive_plan_entregado.txt`. Las referencias científicas de esta propuesta son anteriores al corte bibliográfico de junio de 2026 del plan. El hash consignado en la auditoría previa identifica su estado anterior a esta actualización.

**Dictamen:** se aprueba técnicamente este protocolo para orientar el diseño y la auditoría. No se aprueba todavía el artefacto, ninguna arquitectura como ganadora experimental ni los resultados existentes. Esta revisión de asesoría no sustituye la aprobación institucional del asesor o jurado. El plan es coherente en alcance y secuencia DSR; necesita las precisiones operativas que siguen para sostener sus conclusiones.

## 1. Correspondencia con el plan

Se conserva investigación aplicada y cuantitativa bajo ciencia del diseño (DSR), sección transversal 2D, régimen estacionario, cables XLPE y heterogeneidad prescrita del entorno. No se agrega dinámica de humedad, simulación electromagnética completa, régimen transitorio, validación de campo ni un quinto objetivo. Un caso «seco» es un mapa de conductividad impuesto; no predice el proceso de secado.

| Objetivo del plan | Ejecución metodológica | Producto y decisión |
|---|---|---|
| OE1: especificar | Ficha física, geometría, PDE, fuentes, interfaces, contornos, procedencia y contrato de evaluación | VD1 aceptada solo si las entradas son completas y las unidades consistentes |
| OE2: construir y verificar | Casos analíticos y manufacturados, FEM convergente, piloto PINN, convergencia de puntos, semillas y dominio | VD2 aceptada, restringida o rechazada por caso y familia |
| OE3: evaluar heterogeneidad | Casos pareados de contraste, patrón, extensión y proximidad con configuración congelada | VD3: evidencia térmica y localización del máximo |
| OE4: comparar ampacidad | Búsqueda de corriente admisible con ley de pérdidas común y representaciones homogéneas declaradas | VD4: diferencias operativas con incertidumbre numérica |

Se conserva la nomenclatura de entradas y productos discretos del plan. Los parámetros físicos son factores de cobertura y sensibilidad; las semillas son repeticiones numéricas, no instalaciones independientes ni muestras para inferencia poblacional. DSR organiza la construcción; la verificación numérica justifica la aceptación del solucionador. La contribución defendible es identificar un diseño reproducible y sus límites para esta familia física, sin presuponer superioridad sobre FEM.

Precisiones necesarias del plan: definir la NRMSE; evitar que el porcentaje de temperatura dependa del cero de la escala; distinguir datos de entrenamiento de referencias de evaluación; precisar la representación de capas; definir equivalencia homogénea y criterios de convergencia; tratar las tendencias HE3–HE4 como proposiciones contrastables. El aumento de discrepancia con tamaño, contraste o proximidad no se impone como resultado universal: depende de geometría, contorno y regla de equivalencia.

## 2. Problema físico y unidades

El dominio es una sección perpendicular al eje de cables suficientemente largos, con variaciones axiales despreciables. Los más de 4 m se interpretan como extensión transversal del terreno. Si se refieren a longitud axial, esa longitud no es una coordenada de este modelo. Empalmes, terminaciones y cruces con efectos axiales quedan fuera del dominio verificado.

En cada región material Ωᵢ:

\[
\boldsymbol q_i=-k_i\nabla T_i,\qquad \nabla\cdot\boldsymbol q_i=Q_i,
\quad\text{equivalentemente}\quad-\nabla\cdot(k_i\nabla T_i)=Q_i.
\]

Se exige kᵢ>0, constante por material o función suave dentro de la región. No se sustituye el operador por kΔT cuando k varía espacialmente. En discontinuidades, se aplican las ecuaciones por región y las condiciones de transmisión; la diferenciación automática de un selector de materiales no representa la derivada distribucional de un salto.

Con una misma normal n orientada de i hacia j, contacto perfecto sin fuente superficial:

\[
T_i=T_j,\qquad (\boldsymbol q_i-\boldsymbol q_j)\cdot\boldsymbol n=0.
\]

No se exige continuidad del gradiente normal: cambia inversamente con k. Si existe resistencia de contacto documentada, se sustituye la igualdad de temperaturas por Tᵢ−Tⱼ=R''qₙ, con R'' en m² K/W. No se introduce esa resistencia sin evidencia.

Fuentes: pérdidas lineales p' en W/m, Q=p'/A en W/m³ para una región de sección A en m². En 2D, ∫ΩQ dA y ∮∂Ωq·n ds tienen unidades W/m. La pérdida Joule será p'c=I²R'ac(Tc), indicando si R' es AC o DC, referencia térmica, coeficiente térmico y tratamiento del conductor no isotermo. Si se usa temperatura media del conductor para R', se declara y verifica esa aproximación. Pantallas, cubiertas y pérdidas dieléctricas se incluyen solo si el caso las documenta; un caso Joule simplificado no se presenta como reproducción completa de una norma.

Los contornos deben coincidir entre PINN y referencia: T prescrita, flujo saliente prescrito o q·n=h(T−Taire). No se impone además una temperatura fija en un borde ya convectivo. Es necesaria una condición que determine el nivel de temperatura: con Neumann puro hay que verificar compatibilidad integral y fijar una referencia; generación positiva y todos los bordes adiabáticos no admiten equilibrio estacionario.

La expansión del dominio debe mantener la superficie y profundidad de enterramiento físicas; desplaza los bordes artificiales lateral e inferior. Se comparan al menos tres extensiones hasta que su efecto entre en el presupuesto de error. No basta declarar que 4 m es «suficientemente grande».

### Decisión vinculante: interior del cable explícito en todas las variantes

Se elimina la reconstrucción térmica por resistencias de la formulación activa. Conductor, aislamiento, pantalla y cubierta tienen dominio, conductividad, fuente volumétrica, puntos interiores y residual propios. La superficie exterior del cable es una interfaz de temperatura y flujo desconocidos; no recibe un flujo uniforme prescrito. El máximo se obtiene del campo resuelto dentro del conductor. Esta obligación se aplica a todas las arquitecturas; ninguna ablación puede sustituir la PDE interior por una resistencia o una temperatura reconstruida.

La física común se expresa mediante

\[
\rho_i c_{p,i}\partial_tT_i+\nabla\cdot\boldsymbol q_i-Q_i=0,
\qquad \boldsymbol q_i+k_i\nabla T_i=0.
\]

La campaña de tesis sigue siendo estacionaria (∂tT=0); el operador de almacenamiento y el contrato de capacidad térmica se preparan para extensión temporal. Una corrida temporal requiere ρcp positivo documentado en cada material, condición inicial y carga temporal. No se rellenan propiedades faltantes con valores inventados ni se presenta el soporte de un operador temporal como validación transitoria completa.

En coordenadas polares locales de un anillo, la misma ecuación es

\[
\rho c_p\partial_tT=\frac1r\partial_r(rk\partial_rT)
+\frac1{r^2}\partial_\phi(k\partial_\phi T)+Q.
\]

Se conserva el término angular. La radial 1D es admisible únicamente como prueba axisimétrica, no como modelo de un cable en un entorno heterogéneo general. El núcleo conductor se representa en cartesianas para evitar r=0; en una solución general regular no corresponde imponer ∂rT=0 para todos los ángulos en el origen. En anillos se usan coordenadas logradiales y sinφ,cosφ, manteniendo las derivadas físicas mediante la regla de la cadena.

Para pérdidas DC dependientes de temperatura se especifica resistividad eléctrica local ρe(T), conductividad σ=1/ρe y un campo axial Ez=I/∫Ωcσ(T)dA; así Jz=σEz y Q=σEz². Esta aproximación de cable largo DC permite distribución transversal de corriente y evita usar una temperatura reconstruida para las pérdidas. Sigue excluyendo efecto pelicular, proximidad y un problema electromagnético 3D, de acuerdo con el plan. La verificación térmica a fuente fija usa Q=I²R'20/A, declarado como fuente volumétrica prescrita.

**Revisión y aprobación de la modificación:** satisface conservación local, transmisión y almacenamiento; mantiene el alcance estacionario actual y evita un obstáculo para la extensión temporal. Se aprueba como física común obligatoria antes de modificar la implementación. Las campañas resistivas anteriores quedan como evidencia histórica identificada, no como resultados de la nueva formulación.

## 3. Esquemas PINN y elección defendible

| Familia | Aporte | Papel en esta tesis |
|---|---|---|
| PINN global fuerte, salida T | Referencia simple de la formulación original | Línea base; una MLP suave global no representa exactamente el quiebre del gradiente en interfaces |
| PINN por materiales, inspirada en cPINN/XPINN | Redes locales y transmisión explícita de temperatura y flujo | Candidata principal por correspondencia directa con la física |
| Formulación mixta o de primer orden, salidas T,qx,qy | Solo primeras derivadas en residuos; permite controlar ley de Fourier y balance por separado | Candidata principal dentro de la descomposición; compararla con salida T |
| VPINN/hp-VPINN o formulación variacional | Residuos integrales, integración por partes y refinamiento local | Alternativa si la forma fuerte o mixta no converge; exige cuadratura por material |
| FBPINN | Bases locales y normalización por subdominio para escalas múltiples | Alternativa para subdividir terreno grande; el solapamiento suave por sí solo no resuelve saltos materiales |
| Fourier features y enriquecimientos analíticos | Facilitan escalas espaciales o perfiles conocidos | Ablación posterior; no sustituyen condiciones de interfaz |
| PINN paramétrica, DeepONet/PINO y otros operadores | Respuesta para familias de entradas | Extensión opcional; no necesaria para OE1–OE4 y añade entrenamiento y generalización paramétrica |
| PINN transitoria con entrenamiento causal | Orden temporal | Fuera del alcance aprobado del plan |

La PINN original incorpora ecuaciones y condiciones como restricciones del entrenamiento [Raissi et al., 2019](https://doi.org/10.1016/j.jcp.2018.10.045). cPINN enfatiza conservación en interfaces y XPINN permite descomposición general; no son nombres intercambiables para cualquier red con varias salidas [Jagtap et al., 2020](https://doi.org/10.1016/j.cma.2020.113028), [Shukla et al., 2021](https://arxiv.org/abs/2104.10013). Esta elección para cables es una inferencia de diseño que deberá verificarse.

El esquema mixto propuesto aprende flujo físico, no solo derivadas auxiliares: impone q+k∇T=0 y ∇·q−Q=0. Es una aplicación de la idea de primer orden, no una reproducción literal del artículo [FO-PINNs](https://arxiv.org/abs/2210.14320). Evita segundas derivadas de T en el entrenamiento, pero aumenta salidas y restricciones; no garantiza mejor exactitud. Deben medirse balances tanto con q predicho como con −k∇T para impedir que una red conserve un flujo incompatible con su temperatura.

[hp-VPINN](https://arxiv.org/abs/2003.05385) aporta una alternativa variacional. [FBPINN](https://arxiv.org/abs/2107.07871) justifica normalizar localmente dominios con escalas múltiples. Las [Fourier features](https://arxiv.org/abs/2012.10047) se estudiarán solo si el error espacial lo justifica. No se ensayarán todas las familias indiscriminadamente.

Diseño mínimo comparativo: M0 global fuerte; M1 por materiales con salida T; M2 por materiales con salida T,qx,qy. M1 y M2 comparten geometría, puntos y evaluación; se reportan parámetros entrenables y costo, evitando confundir una mejora de formulación con una red mayor. Inicio de piloto: tanh, 3 capas ocultas de 32 neuronas por región, con comparación acotada a 3×64 y 4×32 cuando exista subajuste. Son valores propuestos, no óptimos bibliográficos. Las regiones disconexas podrán usar redes independientes; si comparten pesos se registra como elección distinta.

## 4. Milímetros frente a metros y contraste físico

Se combinan cuatro medidas: geometría explícita por capas, variables adimensionales, coordenadas locales y muestreo estratificado. Aumentar puntos sin estas medidas no corrige una representación geométrica equivocada.

Con x=L₀ξ, T=Ta+ΔT₀u, k=k₀κ, q=(k₀ΔT₀/L₀)v:

\[
v+\kappa\nabla_\xi u=0,\qquad
\nabla_\xi\cdot v=\widehat Q,
\qquad \widehat Q=QL_0^2/(k_0\Delta T_0).
\]

Se fijan L₀, k₀ y ΔT₀ en la configuración; ΔT₀ puede ser Tlim−Ta cuando sea positivo. La normalización no elimina el contraste κ ni el problema de las capas. Para coordenadas cartesianas locales x=cᵢ+Aᵢξᵢ, ∇x=Aᵢ⁻ᵀ∇ξᵢ; los factores se mantienen en PDE, flujo, normales e integrales. No se normalizan entradas y luego se usa un Laplaciano como si las escalas no hubieran cambiado.

Para un anillo a≤r≤b, se recomienda η=log(r/a)/log(b/a) y ángulo periódico mediante sinφ,cosφ o condiciones periódicas explícitas. En conducción radial estacionaria sin fuentes, T es afín en log r, lo que motiva esta coordenada. Deben conservarse métrica polar y jacobiano r dr dφ. El conductor que incluye r=0 se trata con coordenadas cartesianas o regularidad explícita; no se aplica allí log r.

Ejemplo ilustrativo, no dato del cable de la tesis: dominio 4×4 m, aislamiento de 1 mm y radio interior 20 mm. El anillo ocupa π(0.021²−0.020²)=1.288×10⁻⁴ m², aproximadamente 8.05×10⁻⁶ del dominio. Diez mil puntos uniformes aleatorios tendrían solo 0.0805 puntos esperados en esa capa; la probabilidad de ninguno ronda 92.3 %. Con 100 mil sigue siendo aproximadamente 44.7 %. Esta cuenta geométrica explica por qué ×10 global puede seguir siendo insuficiente. No es una estimación del error PINN.

El espesor, radios y contactos provienen de fichas verificadas. Cada capa recibe puntos propios y ambas interfaces. Los equivalentes radiales quedan retirados de todas las variantes activas. Una solución analítica radial puede servir como referencia externa de verificación en un caso simétrico; nunca reemplaza la PDE interior de la pérdida PINN ni el campo calculado para obtener Tmáx.

Las propiedades se guardan en SI con procedencia y rango. k y resistividad térmica 1/k no se confunden; esta última tiene unidades K m/W. Densidad y calor específico no intervienen en el modelo estacionario. No se añaden como entradas entrenables para aparentar mayor realismo. Las variaciones físicas se separan de variaciones numéricas. Si posteriormente se entrena una familia paramétrica, entradas positivas con varios órdenes de magnitud pueden parametrizarse logarítmicamente; eso no justifica promediar físicamente conductividades con esa transformación.

## 5. Puntos: cantidad, posición y convergencia

Se distinguen puntos interiores de colocación, puntos de interfaces y contornos, cuadratura para balances, puntos independientes de evaluación y posiciones de cables/inclusiones. Aumentar puntos de un mapa de salida solo mejora su visualización y la búsqueda del máximo; no equivale a entrenar con más restricciones. Mover cables cambia el caso físico y exige volver a verificar geometría, fuentes y distancias.

Wu et al. compararon diez estrategias en más de 6000 simulaciones y encontraron ventajas de RAD y RAR-D en sus problemas. La evidencia apoya estudiar distribución y adaptación, no una regla universal de duplicar, triplicar o multiplicar por diez [Wu et al., 2023](https://arxiv.org/abs/2207.10289). Tampoco prueba de antemano que RAD gane en este cable multimaterial.

Presupuesto inicial de piloto por caso: 2048 puntos interiores por región extensa de suelo/relleno, 512 por conductor o capa delgada, 256 por interfaz y 256 por segmento exterior. Se reporta el total real N₀, porque depende del número de materiales y cables. Son mínimos iniciales para estudiar convergencia, no criterios de suficiencia.

En cada material se muestrea por área con secuencias Sobol aleatorizadas o estratos reproducibles; en anillos, uniforme por área implica r=√(a²+u(b²−a²)). Se añaden estratos de proximidad si se quiere mayor resolución normal. Cada interfaz tiene posiciones angulares o de arco propias y normales verificadas. La densidad en un lado del salto no reemplaza puntos de transmisión.

Se estudian N₀,2N₀,4N₀,8N₀ sobre casos representativos: anillo, interfaz plana de alto contraste y cable con heterogeneidad cercana. ×3 o ×10 se usan solo para precisar una transición o extender una curva sin estabilización; no son objetivos en sí. Se modifica por separado Ninterior y Ninterfaz en una ablación para localizar el cuello de botella. La sucesión no presume orden algebraico de convergencia de una PINN.

Primero se compara estratificado fijo con redistribución adaptativa a igual presupuesto de puntos; después se compara crecimiento RAR. Como propuesta reproducible, mantener un 50 % de puntos de cobertura y redistribuir el otro 50 % dentro de cada material, con piso uniforme de probabilidad 0.2. El resto se asigna proporcionalmente a un indicador adimensional de residual PDE y constitutivo; las interfaces tienen indicador propio de salto de temperatura y flujo. Frecuencia inicial: cada 500 pasos Adam, bolsa candidata de 10 veces el presupuesto de cada estrato. Registrar todos estos valores; no se atribuyen al artículo como receta óptima.

La adaptación termina antes de L-BFGS: puntos y pesos se congelan para que su búsqueda de línea evalúe una misma función. Los pesos adaptativos se ensayan después de normalizar residuos y se documentan; el desequilibrio de gradientes está motivado por [Wang et al.](https://arxiv.org/abs/2001.04536). Si se usa muestreo no uniforme, se decide explícitamente entre una pérdida de optimización por estratos y una integral física estimada con pesos de importancia. Las métricas finales siempre usan cuadratura física independiente y pesos de área/longitud; no un promedio sin corregir sobre puntos concentrados.

Inicio de entrenamiento: Adam hasta 20000 pasos con tasa inicial 10⁻³ y calendario registrado; refinamiento L-BFGS hasta 2000 iteraciones y máximo de evaluaciones explícito. Parada anticipada requiere estabilidad de métricas independientes; agotar iteraciones no equivale a converger. El piloto puede modificar estos techos con justificación anterior al conjunto final. Comparar error alcanzable al converger y error a igual tiempo; a igual número de épocas los costos cambian con N.

Semillas: tres para búsqueda de diseño y cinco semillas nuevas predefinidas para confirmación, conservando resultados fallidos. La misma lista se emplea en variantes pareadas. No se selecciona retrospectivamente la semilla más favorable. Cinco es una decisión de proyecto, no una garantía estadística.

Se declara suficiencia solo si dos aumentos consecutivos mantienen Tmáx, ampacidad, métricas por región, saltos y balance dentro del presupuesto fijado. Para el piloto se propone estabilidad ≤0.2 K en Tmáx y ≤0.2 % en I; estos umbrales se endurecen si el efecto relevante es menor. Si el error se estanca por capacidad, acondicionamiento o interfaz incorrecta, se corrige esa causa antes de seguir aumentando N.

## 6. Verificación y métricas que evitan aceptación aparente

Escalera de verificación: V0 conducción homogénea manufacturada; V1 interfaz plana con T continua y gradiente discontinuo conocido; V2 conductor con generación y anillos concéntricos de solución analítica; V3 cable multicapa 2D con suelo homogéneo; V4 relleno, estratos o inclusión próxima; V5 caso publicado con todos los datos esenciales. V0–V2 verifican también FEM; V3–V4 requieren FEM convergente. Los casos publicados parcialmente especificados sirven como contraste cualitativo o caso reconstruido, no como reproducción exacta. CIGRÉ ofrece casos y orientación para verificación de herramientas [TB 880](https://www.e-cigre.org/publications/detail/880-power-cable-rating-examples-for-calculation-tool-verification.html), [TB 963](https://electra.cigre.org/340-june-2025/technical-brochures/finite-element-analysis-for-cable-rating-calculations.html).

FEM: interfaces y fuentes representadas con etiquetas verificadas, refinamiento local en espesor y alrededor de cables, al menos tres mallas y estudio de dominio separado. Comprobar área de conductor y calor integrado: una solución convergente sobre geometría equivocada no es referencia física válida. Reportar elementos efectivos a través del espesor; no inferir resolución local del tamaño nominal global. Orden observado/Richardson solo si existe régimen asintótico; si no, reportar diferencias y refinar. Contrastar flujos de interfaz con trazas apropiadas; un gradiente FEM nodal suavizado entre materiales oculta el salto.

En el método principal no se usan temperaturas FEM como etiquetas de entrenamiento. Si se ensaya PINN asistida por datos, se identifica como variante, se separan escenarios y puntos de ajuste/evaluación y se declara el costo de producir datos. Un campo FEM completo usado para entrenar no constituye evidencia independiente al evaluarlo en otros píxeles del mismo campo.

Sea ΔTref=Tmáx,ref−Ta para un caso con ambiente escalar y calentamiento no nulo:

\[
E_{T,abs}=|T_{max,P}-T_{max,ref}|,\quad
e_{\Delta T}=100E_{T,abs}/|\Delta T_{ref}|,
\]
\[
\mathrm{NRMSE}=100\sqrt{\sum_j w_j(T_{P,j}-T_{ref,j})^2/\sum_j w_j}/\Delta T_0.
\]

ΔT₀ es una escala física prefijada común a la comparación, independiente de la predicción. Si hay ambiente variable o calentamiento casi nulo, se usa esa escala prefijada y error absoluto; no se divide por un número próximo a cero. Se conserva como métrica histórica el porcentaje del plan con su escala declarada, pero la aceptación térmica usa elevación/escala física para evitar cambiar al pasar de °C a K.

Las reglas iniciales 5 % térmico, 5 % NRMSE y 2 % desequilibrio del plan se mantienen con estas definiciones. Se añaden mapas y error por conductor, aislamiento y terreno, percentil y máximo del error, y localización del punto caliente. Una NRMSE global pequeña puede ocultar errores grandes en el cable por su área diminuta. Tmáx relevante para ampacidad se busca en todos los conductores, refinando la evaluación local; no es necesariamente el máximo del dominio completo ni el de una grilla gruesa.

Balance global y por material:

\[
e_E=100\frac{|\oint_{\partial\Omega}\boldsymbol q\cdot n\,ds-\int_\Omega Q\,dA|}{P_*}.
\]

Para calentamiento positivo P*=∫ΩQdA. Para casos sin generación o con fuentes de signo mixto se fija una escala de transferencia no nula y se informa también el defecto absoluto. Las integrales de frontera se calculan con flujos obtenidos de la solución, no sustituyendo los valores prescritos de contorno; de otro modo el control sería tautológico. El balance puede cumplirse aun con campo incorrecto: nunca es el único filtro.

En interfaces se informa RMS y máximo de [T] en K y de [q·n] en W/m², normalizados con ΔT₀ y un flujo característico no nulo. Se propone 1 % para RMS de ambas restricciones y revisión del máximo, ajustando por presupuesto físico antes de confirmación. Es una exigencia adicional de proyecto. En M2, verificar además q+ k∇T y repetir el balance con flujo derivado de T.

Definir antes de evaluación final δTrel y δIrel, efectos mínimos relevantes. Inicio de piloto propuesto: 1 K y 1 % de ampacidad. Construir cotas empíricas de error por referencia, puntos, dominio, optimización y semillas, sin llamarlas intervalos probabilísticos. Para cada diferencia pareada, sumar conservadoramente las cotas de ambos miembros; exigir que sea <un tercio del efecto mínimo relevante para clasificar con margen. Si no se logra, refinar o declarar inconcluso. La variabilidad entre semillas también debe quedar por debajo de ese margen; los porcentajes generales del plan pueden ser demasiado laxos para detectar efectos pequeños.

## 7. Ampacidad y comparación homogénea

Se busca I tal que maxΩc T(I)=Tlim, con temperatura límite tomada de la especificación del cable. Bisección con intervalo documentado y evidencia de monotonía en el intervalo es una opción robusta. Cada evaluación actualiza fuentes dependientes de I y, si corresponde, de T, hasta convergencia interna electrotermal. Se guardan todas las corrientes, máximos y balances. Las tolerancias de esa iteración y de la búsqueda externa consumen parte del presupuesto de error operativo.

Solo con k y pérdidas resistivas independientes de T y operador/contornos lineales puede reutilizarse exactamente T(I)=T⁽⁰⁾+(I/I₀)²[T(I₀)−T⁽⁰⁾], con T⁽⁰⁾ calculada para las mismas fuentes independientes de I y contornos. Con ambiente uniforme y sin otras fuentes se simplifica a Ta+(I/I₀)²[T(I₀)−Ta]. No se usa una regla de raíz cuadrada general para materiales dependientes de temperatura o secado acoplado. El costo de entrenar o reutilizar debe contabilizarse de forma comparable en FEM y PINN.

Cada par conserva cables, corriente en evaluación térmica, ley de pérdidas, contornos y extensión de dominio; cambia exclusivamente la representación del entorno. Para ampacidad se conserva la ley física, aunque la corriente y por ello las pérdidas resultantes sean diferentes. No es posible exigir pérdidas numéricamente idénticas al comparar dos corrientes admisibles diferentes.

La regla de k equivalente se registra antes: promedio de área sobre una región de terreno físicamente fija, o hipótesis conservadora/optimista con k mínimo/máximo de esa región. No se incluye cobre o aislamiento en el promedio del suelo. El promedio aritmético es una representación comparativa, no una equivalencia universal de resistencia 2D. No se calibra k equivalente con Tmáx del propio caso usado para demostrar discrepancia. Una expansión artificial de dominio no debe alterar la región del promedio ni la fracción de inclusión definida físicamente.

Se sigue la cobertura del plan: patrones simples y niveles de Ck, fh, dh/D, hasta 108 escenarios si son físicamente aplicables. La distancia se define desde superficies o centros de manera única. Una capa que cruza toda la sección no tiene la misma noción de «proximidad» que una inclusión: las combinaciones inaplicables se excluyen y documentan. Se empieza por bases y extremos; la reducción de cobertura debe justificarse, sin elegir solo casos favorables.

## 8. Flujo reproducible: contrato mínimo

Una ejecución debe poder reconstruirse desde un manifiesto, sin editar cuadernos ni recuperar parámetros de memoria. Los cuadernos explican y visualizan; un punto de entrada automatizado ejecuta el mismo procedimiento.

| Archivo o grupo | Contenido obligatorio |
|---|---|
| `case.json` | Versión de esquema; unidades SI; geometría y regiones; interfaces y normales; k; ley de Q; corriente; contornos; Tlim; supuestos; fuente, página/tabla y estado de cada dato |
| `training.json` | Modelo, coordenadas y escalas; activación; capas; dtype; optimizadores y paradas; cuotas de puntos; adaptación; pesos; semillas separadas |
| `evaluation.json` | Referencia y convergencia; puntos independientes; cuadratura; búsqueda de máximos; métricas, denominadores, tolerancias y estados |
| `manifest.json` | Identificador único; fecha UTC; hash de cada entrada, código efectivo y salidas; entorno; hardware; comandos y estado completado/fallido |
| `points.npz` o HDF5 | Coordenadas SI, tipo de punto, material, interfaz, normal y peso; partición entrenamiento/evaluación; historial o estado suficiente para regenerar adaptación |
| `reference.*` | Malla con etiquetas, solución, niveles de refinamiento, solver y tolerancias; procedencia independiente |
| `checkpoint.*` | Pesos, estado de optimizador y generadores, paso, escalas y mapa de regiones necesarios para evaluar/reanudar |
| `metrics.json`, `history.csv`, `fields.npz` | Métricas con unidades; componentes de pérdida; valores T y flujo; puntos; estado de aceptación |
| `environment.lock` y registro | Versiones exactas, sistema operativo, backend, driver/runtime cuando corresponda, hilos y dispositivo |

JSON/CSV/HDF5/NPZ son decisiones prácticas propuestas, no obligación de adoptar todos. JSON usa UTF-8, números finitos y unidades explícitas; un campo faltante invalida la corrida cuando es necesario para interpretarla. Datos sintéticos se marcan como tales. Conservar PDFs/fichas originales y su hash por separado de los datos transcritos. El checksum prueba integridad, no veracidad de una fuente.

Un commit no describe cambios locales sin confirmar: guardar hash/snapshot del código efectivo y diff, incluidos archivos nuevos usados. Separar semilla de inicialización, muestreo, adaptación y evaluación; registrar estados al reanudar. La reproducibilidad entre versiones o dispositivos se verifica con tolerancias y no se promete identidad binaria. Una repetición independiente desde el manifiesto debe reproducir métricas y clasificación; no basta releer un JSON existente. No sobrescribir corridas ni seleccionar silenciosamente el último archivo por nombre.

Secuencia: extraer datos → validar contrato y geometría → construir/verificar referencia → piloto de formulación y puntos → congelar configuración → ejecutar semillas nuevas → evaluar independientemente → clasificar VD2 → cubrir casos OE3 → calcular pares OE4 → generar tablas desde los manifiestos aceptados. Los resultados que no pasan VD2 se conservan como fallos o límites, pero no sostienen conclusiones operativas aceptadas.

## 9. Paralelismo: criterio previo a inspeccionar la máquina

Se inventariarán CPU física/lógica, memoria, aceleradores, memoria de dispositivo, backend real y límites del entorno. Tener GPU no demuestra que la biblioteca la use; disponer de varias redes tampoco significa ejecución concurrente. Separar paralelismo de tensores, de casos/semillas, de subdominios y FEM/MPI.

Para CPU, estudiar primero hilos por proceso y luego varias semillas/casos independientes. Evitar procesos×hilos mayor que los núcleos útiles sin medición que lo justifique; controlar BLAS/OpenMP y memoria. Para una GPU, comenzar con una corrida vectorizada y medir tamaño de lote, transferencia y ocupación; múltiples entrenamientos pueden competir por memoria. Con varias GPU, una semilla/caso por GPU suele ser un punto de partida sencillo; descomposición distribuida necesita intercambio y sincronización de interfaces. La posibilidad MPI+CPU/GPU de cPINN/XPINN está documentada por [Shukla et al.](https://arxiv.org/abs/2104.10013); sus aceleraciones no se extrapolan a esta máquina.

Medir 1,2,4,… hilos o trabajadores hasta límites de hardware; tres repeticiones de tiempo como mínimo tras calentamiento, mismo caso, precisión y criterio de exactitud, sin otras cargas propias simultáneas. Reportar tiempo hasta solución aceptada, rendimiento de campaña, memoria y S(p)=t1/tp; eficiencia S(p)/p cuando p representa recursos comparables. En CPU heterogénea o dispositivos distintos se explica el denominador. La utilización instantánea no es una prueba de eficiencia.

En M1 se recomienda float64 de referencia para derivadas de segundo orden y escalas extremas. M2 permite evaluar float32; la elección final se decide por errores de campo y flujo. No habilitar precisión mixta o TF32 sin comparación física. Cuando se cronometra GPU, sincronizar antes y después del tramo. Incluir entrenamiento, adaptación, preparación y búsqueda de ampacidad; mostrar inferencia por separado. El tiempo sigue siendo metadato secundario, como dispone el plan.

## 10. Revisión crítica y puerta metodológica

| Pregunta de revisión | Resolución previa a auditar |
|---|---|
| ¿Se resuelve el problema del plan? | Sí: conducción 2D estacionaria, OE1–OE4, heterogeneidad impuesta y verificación sin campo |
| ¿La formulación admite interfaces y capas? | Sí: regiones explícitas, transmisión y escalas locales; equivalentes térmicos retirados de todas las variantes activas |
| ¿Se ha decidido «la mejor PINN» sin evidencia? | No: se elige candidata razonada y comparación M0–M2 con condiciones de selección |
| ¿Existe criterio para aumentar puntos? | Sí: curva por estratos, adaptación comparada, evaluación independiente y estabilidad física |
| ¿Se evita validación circular? | Sí: referencias independientes, balance calculado y particiones declaradas |
| ¿Se puede interpretar ampacidad? | Sí: ley de pérdidas y corriente externa/interna, límite conductor y presupuesto de error |
| ¿Se puede repetir y evaluar paralelismo? | Sí: contrato de entradas y manifiesto, semilla/estado, métricas y prueba de escalado |
| ¿Las conclusiones pueden ser negativas? | Sí: rechazo/restricción de VD2 e inconclusión en VD3–VD4 son resultados admisibles |

**Puerta M: aprobada para auditoría e implementación.** Se autoriza pasar a revisar el código, flujo, benchmark y resultados contra esta especificación, en ese orden lógico. No se atribuye al protocolo evidencia empírica aún no producida. Las decisiones de piloto se distinguen de sus resultados y deberán congelarse antes de confirmación. Si la implementación no cumple, se documentan brechas y solo se repiten cálculos capaces de responder una pregunta identificada.

## 11. Precisamiento eléctrico y cierre de la implementación

Se aprueba técnicamente el uso de la ecuación de calor interior en todas las variantes. El contrato identifica resistencia DC/AC, frecuencia, temperatura y procedencia; los datos DC a 60 Hz necesitan corrección de pérdidas y los datos AC no se corrigen dos veces. La distribución de calor es una fuente volumétrica de la PDE, nunca una sustitución del interior térmico. El perfil cilíndrico de piel es una estimación condicionada a la construcción y no incluye automáticamente proximidad. La fuente uniforme debe justificarse mediante sensibilidad a igual pérdida total. Las especificaciones, cifras reproducibles y límites están en [FULL_DOMAIN.md](../Benchmarks/FULL_DOMAIN.md); la verificación posterior se conserva en [REVISION_FISICA_EXPLICITA.md](REVISION_FISICA_EXPLICITA.md). La aprobación del método no equivale a aceptación de toda la campaña ni a aprobación institucional de la tesis.
