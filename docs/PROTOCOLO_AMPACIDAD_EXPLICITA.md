# Extensión de verificación: corriente límite DC con interior explícito

Fijado el 15 de septiembre de 2026 antes de entrenar las PINN de corriente límite. Complementa el protocolo A–D sin modificar sus decisiones ni usar los resultados de confirmación para ajustar la receta.

Se compara XLPE homogéneo, XLPE con zona seca cercana y una representación homogénea cuyo k es el promedio aritmético espacial del **suelo**, excluyendo los discos de cable. La media se obtiene por integración de las ventanas del mapa de conductividad y cuadratura de los discos; no se calibra para igualar una temperatura. Dominio, cable, corriente común desconocida, ley eléctrica y contornos permanecen iguales en cada par.

FEM resuelve la PDE multimaterial con sigma(T) local y busca por acotación/bisección la corriente que produce Tmax=90 °C. Se exige residuo térmico ≤0,01 K e intervalo relativo de corriente ≤0,05 %. Se comparan al menos tres mallas; la variación de corriente fina debe ser ≤0,5 % y el balance ≤0,2 %. Los ensayos intermedios de la raíz guardan metadatos; se conserva el campo completo de la solución final de cada malla.

La PINN aprende la corriente positiva parametrizada como I=I0 exp(eta), junto con T y, si corresponde, q. La ley local de Joule sigue siendo Q=sigma(T)Ez², con Ez=I/integral(sigma dA). La temperatura límite se impone sobre muestras interiores y el centro de todos los conductores. No se usa corriente, temperatura ni respuesta térmica FEM dentro del entrenamiento.

Se conserva la arquitectura, distribución y tasa seleccionadas en C. Para este problema no lineal adicional se fija **antes de observar sus resultados** el doble de los pasos Adam y del límite L-BFGS de C. Semillas de confirmación: 71, 83 y 97; muestreos 1001, 1003 y 1007. No se selecciona una semilla favorable. Se mantienen las puertas térmicas e interfaces y se agregan error de corriente frente a FEM ≤2 % y desviación de Tmax respecto a 90 °C ≤0,1 K. El presupuesto adicional no se presenta como un efecto aislado de arquitectura.

Esta extensión calcula ampacidad DC del modelo explícito. No es ampacidad AC normativa. La escala nominal de potencia se conserva solo para adimensionalizar las pérdidas PINN; la fuente física usa la corriente aprendida. La identidad del problema de corriente desconocida se distingue de su estado solucionado y de un problema de corriente prescrita.

Si un caso falla, se registra aceptación restringida o inconclusión de OE4; no se recuperan como sustituto los resultados reducidos anteriores. Los pares aceptados permiten contrastar HE4 dentro del promedio y la zona seca especificados, sin afirmar que todo promedio homogéneo produzca la misma tendencia.
