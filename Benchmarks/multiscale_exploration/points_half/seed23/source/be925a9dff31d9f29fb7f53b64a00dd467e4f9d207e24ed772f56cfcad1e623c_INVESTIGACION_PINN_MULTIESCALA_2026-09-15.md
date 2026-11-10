# Investigación separada: PINN multiescala con transmisión exacta

Fecha: 2026-09-15. Estado inicial: propuesta y protocolo exploratorio, anterior a los nuevos ensayos. Este documento no constituye una conclusión de tesis ni una afirmación de superioridad experimental. Las actualizaciones de resultados se añaden al final, sin alterar retrospectivamente el protocolo.

## 1. Auditoría del trabajo interrumpido

La sincronización Git terminó, pero **el trabajo científico anterior no estaba cerrado**. `Benchmarks/explicit_study/status.json` confirma A=36, B=24, C=12, C2=8 y D=21: 101 entrenamientos principales. D acepta solamente 7/21 ejecuciones. C2 eligió provisionalmente peso de continuidad 1000; ninguna receta pasó los cuatro ensayos de desarrollo. El resultado no permite llamar robusta a la arquitectura elegida.

Existen las referencias FEM de ampacidad, pero no `ampacity/status.json` ni las nueve confirmaciones PINN. Falta asimismo la auditoría integral final y la actualización/compilación definitiva de la tesis. No hay entrenamientos Python activos al iniciar esta revisión. Las tablas parciales y el PDF existente no equivalen al cierre científico. Se conserva toda la evidencia anterior: D pasa a ser evidencia exploratoria para esta nueva iteración; no puede reutilizarse como confirmación ciega de una arquitectura diseñada después de observarlo.

## 2. Qué dificultad debe resolver la arquitectura

Hay cuatro dificultades distintas: separación geométrica de escalas, contraste de conductividad, cambio de regularidad entre materiales y condicionamiento de la optimización. Aumentar puntos solamente ataca la cobertura espacial.

En `xlpe_single` el dominio real es **8 × 4 m**, no 4 × 4 m. La pantalla de 12 a 13 mm tiene 1 mm de espesor: relaciones 8000:1 y 4000:1 con las dimensiones exteriores. Su conductividad es 380 W/(m K), frente a 0,286 en XLPE; conductor 400. Una capa metálica muy conductora puede tener una caída de temperatura diminuta y transportar un flujo importante: una pérdida de temperatura pequeña no garantiza un flujo correcto.

Una cuadrícula uniforme de paso 1 mm requeriría aproximadamente 32 millones de celdas para 8 × 4 m. No es un número de colocaciones PINN necesario ni un criterio de precisión: ilustra el desperdicio de resolver el suelo lejano con la escala del espesor. Con puntos uniformes en área, una corona de radio 12–13 mm ocupa aproximadamente 7,854e-5 m²; su fracción en 32 m² es 2,454e-6. Diez mil puntos globales tendrían sólo 0,0245 puntos esperados allí. El muestreo por material evita este problema directamente.

Para contacto perfecto, T es continua y k ∂nT es continuo. Cuando k salta y hay flujo normal no nulo, ∂nT debe saltar. Una MLP global suave en x,y impone una regularidad incorrecta. Una red única como **modelo compuesto**, con expertos por región y variables compartidas, sí puede representar este campo. No es necesario exigir una única MLP lisa.

## 3. Bibliografía primaria y decisión que informa

La búsqueda comprende conducción heterogénea, interfaces elípticas, reacción-difusión y problemas oscilatorios. Es una revisión dirigida, no una revisión sistemática exhaustiva. Se distinguen publicaciones de sus versiones preprint.

| Fuente primaria | Qué demuestra o propone | Uso y límite para esta tesis |
|---|---|---|
| Moseley, Markham y Nissen-Meyer, FBPINNs, publicación 2023; [preprint 2021](https://arxiv.org/abs/2107.07871), [DOI](https://doi.org/10.1007/s10444-023-10065-9) | Redes locales en subdominios solapados y normalización local para problemas grandes/multiescala. | Motiva parches locales; sumar funciones globalmente suaves no resuelve por sí solo el salto de gradiente en una interfaz material. |
| Dolean, Heinlein, Mishra y Moseley, [multilevel FBPINNs](https://arxiv.org/abs/2306.05486), [publicación 2024](https://doi.org/10.1016/j.cma.2024.117116) | Añade niveles de descomposición y comunicación global a las aproximaciones locales. | Motiva una corrección global del suelo y parches por cable; sus resultados no prueban eficacia en estas capas metálicas. |
| S. Wu y B. Lu, [INN, JCP 470, 111588, 2022](https://lsec.cc.ac.cn/~lubz/Publication/2022JCP-INN.pdf) | Descomposición por interfaces y balance de pérdidas mediante información de múltiples gradientes en problemas elípticos. | Fundamenta separar materiales y diagnosticar competencia de términos; todavía requiere optimizar condiciones de transmisión. |
| Z. Wu, Jiang, Sun y Li, [HCD-PINN, 2024](https://doi.org/10.1109/TCPMT.2024.3416523), [registro institucional del autor](https://ece.mst.edu/media/academic/ece/documents/facultyprofiles/resumesandcvs/CurriculumVita_LJ_2025NOV_Combined.pdf) | Conducción transitoria no homogénea mediante restricciones y descomposición. | Antecedente cercano: no se puede reivindicar como nueva la idea general de PINN térmica con interfaces duras. Se verificaron datos bibliográficos y resumen; la descarga íntegra NSF falló. |
| Wang, Wang y Perdikaris, [Fourier multiescala, 2021](https://arxiv.org/abs/2012.10047), [DOI](https://doi.org/10.1016/j.cma.2021.113938) | Explica sesgo espectral y propone características Fourier a varias escalas, con ejemplos de ondas/reacción-difusión. | Útil para variación angular o fuentes oscilatorias. Frecuencias globales altas no sustituyen geometría ni transmisión. El `fourier` anterior de una frecuencia π no ensayó toda esta familia. |
| Kharazmi, Zhang y Karniadakis, [hp-VPINNs](https://arxiv.org/abs/2003.05385), [publicación 2021](https://doi.org/10.1016/j.cma.2020.113547) | Formulación variacional con pruebas locales y refinamiento h/p. | Alternativa prioritaria si las derivadas segundas y los coeficientes delgados dominan el coste. La forma débil conserva la PDE; no es un equivalente resistivo. |
| Wu et al., [estudio de muestreo adaptativo](https://arxiv.org/abs/2207.10289) | Compara políticas no adaptativas y basadas en residuos en varios problemas PINN. | Justifica ensayar adaptación; no establece un factor universal 2×, 3× o 10× para todo problema. |

**Decisión:** la candidata principal es un modelo compuesto por materiales, con coordenadas locales y trazas compartidas de temperatura y flujo; se agrega una representación global/local del suelo. La alternativa de contraste es una hp-VPINN por material con integración local conservativa. Fourier multiescala es un enriquecimiento opcional, no la arquitectura principal elegida por nombre o popularidad.

## 4. Transformaciones sin alterar la física

Se mantiene en cada material:

\[
\rho_m c_m\partial_tT_m-\nabla\cdot(k_m\nabla T_m)=Q_m,
\quad [T]=0,\quad[k\nabla T\cdot n]=0.
\]

La campaña estacionaria anula sólo ∂tT. En DC dependiente de temperatura se conserva Q=σ(T)E_z² y E_z=I/∫σ(T)dA. Una resistencia eléctrica documentada puede determinar la generación Joule; no reemplaza la conducción térmica interior.

Para una transformación x=F(ξ), Jacobiano J y determinante positivo:

\[
\rho c T_t-\frac1{\det J}\nabla_\xi\cdot\left(\det J J^{-1}kJ^{-T}\nabla_\xi T\right)=Q.
\]

Normalizar las entradas no autoriza a borrar esos factores. En el código activo se diferencia respecto de x,y físicos y la regla de la cadena los conserva.

En una corona a≤r≤b, s=log(r/a)/ℓ, ℓ=log(b/a):

\[
\nabla\cdot(k\nabla T)=\frac1{r^2\ell^2}\partial_s(kT_s)+\frac1{r^2}\partial_\theta(kT_\theta),
\qquad q_r=-\frac{k}{r\ell}T_s.
\]

La dirección angular permanece; no se impone simetría radial. Cada espesor ocupa s∈[0,1]. Las entradas periódicas son cosθ,sinθ. En el núcleo se usa (x−xc)/a,(y−yc)/a y polinomios regulares, sin log r ni división por r en el centro. Para un estrato plano se normaliza su espesor por separado de la coordenada tangencial.

La transformación logarítmica mejora la representación radial, pero cuando b−a≪a sigue existiendo anisotropía en el operador transformado: no constituye por sí sola una cura del condicionamiento.

## 5. Aporte propuesto: aprender las trazas una sola vez

En cada circunferencia Γj se introducen **incógnitas compartidas** τj(θ,t)=T y gj(θ,t)=qr, con normal radial común hacia fuera. Pueden representarse con pocos armónicos aprendibles o una red periódica. Ni τ ni g se obtienen de FEM. Los materiales vecinos usan exactamente las mismas variables, aunque k difiera.

En una corona se construye una interpolación de Hermite más una corrección neural:

\[
T(s,\theta)=H_{00}(s)\tau_a+H_{01}(s)\tau_b
+H_{10}(s)d_a+H_{11}(s)d_b
+16s^2(1-s)^2 A_mN_m(s,\cos\theta,\sin\theta),
\]

con H00=2s³−3s²+1, H01=−2s³+3s², H10=s³−2s²+s, H11=s³−s²; da=−ℓ a ga/k(a,θ) y db=−ℓ b gb/k(b,θ). La corrección y su derivada normal se anulan en ambos extremos. **Por construcción**, las dos caras comparten temperatura y flujo. La PDE completa determina las trazas y la corrección interior: Hermite no es una solución térmica prescrita ni una red de resistencias. La capacidad interior permanece libre y se puede aumentar.

En el núcleo, un modo angular de orden n se extiende como ρⁿ(An+Bnρ²), ρ=r/a. Si τn y dn=−gn/k son sus datos de borde, Bn=(a dn−n τn)/2 y An=τn−Bn. Expresado mediante las partes real/imaginaria de (x+iy)ⁿ es regular en r=0. Se añade (1−ρ²)²N(x/a,y/a), que no cambia valor ni derivada en r=a.

En el suelo, una red global G representa el campo de largo alcance. Un collar alrededor de cada cable corrige su valor y derivada para compartir τ y g. Para R≤r≤S, h=S−R y s=(r−R)/h:

\[
T=G+w_0(s)[\tau-G(R,\theta)]
+h w_1(s)[-g/k(R,\theta)-G_r(R,\theta)],
\]

w0=1−10s³+15s⁴−6s⁵; w1=s−6s³+8s⁴−3s⁵. Los correctores valen cero con sus dos primeras derivadas en S. Se recupera G fuera del collar sin introducir una interfaz física nueva. El collar debe estar dentro del dominio, no solapar otros cables y no cruzar discontinuidades materiales sin tratamiento específico. Esta geometría es una limitación explícita del primer prototipo.

Para varios cables se requiere comprobar separación de los collares. Si no existe espacio, se usan parches conjuntos o una partición compatible, no una suma ciega. Para estratos planos, la misma construcción Hermite usa la coordenada normal del estrato; las intersecciones y esquinas exigen compatibilidad de trazas. El primer prototipo circular **no** valida automáticamente esos casos.

La ampliación multilevel añadiría al suelo correctores locales de soporte compacto, normalizados por el radio del parche, y un nivel global compartido. El nivel global comunica cables distantes; resolver dos problemas independientes con una temperatura arbitraria en su separación sería incorrecto. La partición cambia la representación y el algoritmo, no la PDE ni el acoplamiento.

**Novedad defendible provisional:** combinación y evaluación de trazas térmicas/de flujo compartidas, cartas logradiales, extensión regular del núcleo y corrección global/local en contraste milímetro–metro. No se afirma prioridad científica: las piezas tienen antecedentes y hace falta revisar métodos de restricciones duras antes de cualquier reivindicación de originalidad.

## 6. Muestreo inteligente y convergencia

1. Reservar un mínimo de puntos por material y por interfaz, independiente de su área. Usar coordenadas locales para distribuir puntos radiales y angulares. En suelo lejano, cobertura Sobol/estratificada; cerca de cables, bandas de distancia logarítmica.
2. Distinguir dos conjuntos: colocaciones para optimización y cuadraturas independientes para auditoría. Una frontera exactamente satisfecha sigue auditándose: un error de implementación puede romperla.
3. Probar colocaciones adicionales con residuo PDE **adimensional por material**, error de flujo/constitutiva cuando corresponda y desequilibrio local. No usar solamente |∇T|: en metal de alta k puede ser pequeño aunque el transporte importe. Si la transmisión se impone exactamente, su salto no sirve como indicador de dónde refinar; deben usarse residuos interiores y balances en subregiones.
4. Conservar una fracción exploratoria uniforme (p. ej., 30–50%, a calibrar) y distribuir el resto entre celdas/parches según indicador/coste. Para una pérdida que aproxima una integral física, utilizar pesos de importancia según la densidad de muestreo; si se omiten, declarar que se cambia la medida de entrenamiento. Los pesos grandes necesitan control de varianza.
5. Refinar 1×,2×,4× primero. Llegar a 8×/10× sólo si mejora error independiente por segundo. Separar aumento radial, angular y suelo: multiplicar todos los conjuntos oculta cuál era el cuello de botella. Añadir armónicos si falla variación angular, parches si falla suelo, y ancho local si persiste residuo interior.
6. Para afirmar independencia respecto de puntos, repetir refinamiento con la **receta final**, varias semillas y una malla FEM convergida. La campaña B anterior no mostró insensibilidad en 8/8 comparaciones; no se debe presentar como convergente ni extrapolar su resultado a la nueva arquitectura.

El presupuesto óptimo es el menor que cumple simultáneamente los criterios físicos y FEM con estabilidad entre semillas. La bibliografía consultada no autoriza un multiplicador universal.

## 7. Protocolo exploratorio fijado antes del nuevo prototipo

Primero se verificará algebraica y numéricamente la transmisión exacta, la regularidad del centro y la conservación del operador 2D. Estas comprobaciones no constituyen un benchmark de eficiencia ni una validación FEM de la solución.

Después se realizará un piloto de la construcción circular en `xlpe_single`, con referencia FEM ya convergida, semillas 11 y 23, las mismas colocaciones y presupuesto de C: 512 suelo, 256 por capa, 192 por interfaz; Adam 1500, L-BFGS 3000, lr=0,0005; float64, CPU, un hilo por ensayo. Red local 32×3, seis armónicos de traza y collar exterior S=min(0,15 m, distancias geométricas admisibles). Mismos criterios de aceptación del estudio anterior. Los coeficientes de traza se inicializan en T0 y flujo cero para no introducir una solución de referencia. No se entrenará con etiquetas FEM.

Se reutilizarán los resultados C de las mismas semillas como comparación exploratoria de presupuesto, declarando que cambia el número de parámetros y el coste de derivación. La nueva implementación y configuración quedarán archivadas con SHA256. Ningún fracaso se reemplazará silenciosamente. Los resultados sólo decidirán si se justifica una campaña nueva; dos semillas de desarrollo no establecen robustez de producción.

Para una campaña posterior: verificar coaxial con variación angular y soluciones manufacturadas de capa fina; desarrollar con XLPE, estratos y un caso de parches; reservar casos nuevos, posiciones no usadas, espesores 0,25–2 mm y semillas nuevas (p. ej., 211,223,227) para confirmación. Las geometrías modificadas requieren nuevas referencias FEM y convergencia de malla. Comparar por tiempo hasta tolerancia, fracción aceptada, memoria, parámetros y error; no sólo por épocas. El método debe superar al FEM sólo si se demuestra, y puede ser útil como sustituto paramétrico aunque pierda en una única solución directa.

## 8. Revisión severa antes de trasladar a tesis

Se acepta esta propuesta como **hipótesis metodológica verificable**, no como mejor red probada. Riesgos: trazas Fourier insuficientes, coste de derivar la proyección de G, condicionamiento de Hermite con gran contraste, errores en normales, y geometría de collares/estratos. La PDE sigue siendo la autoridad: satisfacer interfaces exactamente no garantiza el campo correcto.

La futura extensión temporal conserva ρcTt y trazas dependientes de tiempo; requiere datos de capacidad, condición inicial y pruebas temporales, posiblemente ventanas causales. No se promete que el prototipo estacionario ya la resuelva. La forma variacional o el sistema de primer orden se consideran si la diferenciación de alto orden domina; ambos conservan la física.

La ejecución actual usa CPU float64. Paralelismo entre ensayos es apropiado para reproducibilidad y búsqueda; el entrenamiento distribuido entre subdominios requiere medir coste de comunicación y gradientes compartidos. Una GPU disponible no implica aceleración de derivadas pequeñas en doble precisión. No se cambia de dispositivo en una comparación sin registrarlo.

## 9. Bitácora y resultados posteriores

Pendiente al registrar este protocolo: implementación del prototipo, verificaciones y dos pilotos. Los hallazgos se agregarán aquí y se conservarán fuera de la tesis hasta su evaluación.

### 9.1. Construcción verificada y ampliación bibliográfica

`construction_audit.json` registra una comprobación de las cuatro interfaces con coeficientes de traza aleatorios, incluyendo modos angulares no nulos. El salto máximo de T es 1,14e-13 K y el de flujo 9,72e-7 W/m²; el operador logpolar con k variable coincide con el cartesiano con error relativo 6,06e-16. El núcleo tiene derivadas finitas en el centro y el error Dirichlet exterior es cero. Son verificaciones de representación, no resultados de entrenamiento.

La búsqueda ampliada identificó antecedentes que restringen la reivindicación de novedad:

- Chung et al., [Hard-constrained PINNs for Interface Problems, preprint v2, mayo 2026](https://arxiv.org/abs/2604.08453v2): compara construcciones con ventanas y correcciones buffer. Advierte dificultades de solapamiento y esquinas en 2D; no basta trasladar una construcción 1D. Se consultó el resumen; no se afirma haber reproducido el método.
- Lai et al., [Hard-Constraint PINNs for Interface Optimal Control Problems](https://arxiv.org/abs/2308.06709), publicación [SIAM 2025](https://doi.org/10.1137/23M1601249): antecedente elíptico y parabólico para separar restricciones de interfaz del aprendizaje de la PDE.
- Dong y Li, [locELM, 2021](https://arxiv.org/abs/2012.02895), [artículo del autor](https://www.math.purdue.edu/~sdong/PDF/locelm_CMAME2021.pdf): redes locales con características ocultas fijas y solución de coeficientes por mínimos cuadrados.
- Chi, Chen y Yang, [Random Feature Method for Interface Problems](https://arxiv.org/abs/2308.04330): aborda interfaces lineales, incluidas geometrías complejas. Es un antecedente directamente pertinente para una alternativa neuronal sin entrenamiento completo de pesos ocultos.

### 9.2. Segundo piloto registrado antes de su ejecución: aprovechar linealidad

Las interfaces exactas no eliminan el condicionamiento interior. Con temperaturas de traza escaladas todas por 50 K, representar una caída pequeña en metal requiere cancelar números de escala mucho mayor. Un cambio de variables a temperatura común y diferencias escaladas por q*h/k puede mejorar ese aspecto sin imponer su valor: esas diferencias seguirían siendo incógnitas. Esta mejora queda propuesta, no implementada en el primer piloto.

Para **k y Q prescritos**, la construcción completa es lineal en las trazas y en los coeficientes de salida si se congelan las capas ocultas. Se añade un ensayo locELM/PIELM de esa misma representación: semillas 11 y 23, ancho 32, profundidad 3, mismos puntos y normalizaciones. Resolver A z≈b mediante SVD con normalización de columnas y umbral relativo 1e-12. Conservar la PDE fuerte en todos los materiales y los balances regionales; los términos de salto son cero por construcción y se verifican externamente. No usar FEM para montar A ni b. Comprobar numéricamente la afinidad y la equivalencia entre suma de residuos cuadrados y pérdida física.

Se registrarán tiempo de construcción de A, tiempo de solución, rango numérico, espectro singular, residual y métricas FEM. El número de coeficientes ajustados se distinguirá de los pesos ocultos fijos. No denominar este ensayo entrenamiento completo de una PINN ni extrapolar linealidad a Q(T). Para DC no lineal se requeriría Picard/Newton o proyección variable; eso sería otro estudio.

### 9.3. Resultado inicial SVD y refinamiento exploratorio registrado

Las dos ejecuciones SVD de 32×3 aceptan todos los criterios del FEM: error Tmax 0,035315 K (semilla 11) y 0,102976 K (23), en 54,34 y 50,06 s medidos. Se ajustan 269 coeficientes; se conservan otros pesos fijos. El sistema de la semilla 11 tiene 1541 filas y rango 269. La equivalencia entre la pérdida física y el sistema lineal se verifica independientemente. Este resultado motiva una ampliación exploratoria; no estaba prevista antes de esas dos observaciones.

Antes de ejecutar la ampliación se fija: mantener seis armónicos, mismos criterios, solver y geometría; comparar 32×3 con n=256/512/1024 suelo, n_capa=128/256/512 y n_interfaz=96/192/384, semillas 11 y 23. El nivel central reutiliza los dos resultados anteriores. Ejecutar además 16×2 con el nivel central, ambas semillas. Son seis nuevas ejecuciones. Comparar todos los campos sobre los mismos puntos FEM; medir cambios máximos por región normalizados por el incremento térmico FEM y cambios de Tmax, con umbral orientativo 0,5% para ambos. No confundir insensibilidad a colocaciones con exactitud: también deben cumplirse todos los criterios FEM. Los conjuntos son reproducibles con la misma semilla, pero no se afirma que sean anidados.

Las dos ejecuciones Adam/L-BFGS continúan con su presupuesto registrado aunque SVD ya sea prometedor. No se retiran resultados por ser desfavorables. Para la elección final se distinguirán número de coeficientes, pesos totales, densidad, error por material y tiempo observado.
