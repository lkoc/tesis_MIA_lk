<!-- explicit-heat-2d-v1 -->
**Contrato vigente:** [FULL_DOMAIN.md](FULL_DOMAIN.md). `explicit_results` usa identidad de física, etiquetas de material y temperatura interior explícita; este contrato histórico no debe aplicarse automáticamente a esos archivos.
<!-- /explicit-heat-2d-v1 -->

# Especificación común de materiales: valores, estratos y fórmulas

La versión extendida del formato JSON admite `conductivity` como número,
expresión o colección ordenada de estratos. La misma definición se evalúa en
NumPy para inspección, en PyTorch para PINN y en UFL para FEniCSx. El lector
está en `expressions.py`; no usa `eval`, `exec` ni ejecución de código del JSON.

## Valor constante

```json
"conductivity": 1.365
```

Las unidades son W/(m K). Los archivos anteriores usan la abreviatura `k` y
las definiciones `patch` y `bands`; siguen siendo legibles. Para una nueva
definición `conductivity`, no deben mezclarse `patch` o `bands` activos: el
validador rechaza esa ambigüedad.

## Estratos discretos

```json
"conductivity": {
  "type": "layers",
  "axis": "y",
  "interfaces": [-1.76, -0.56],
  "values": [1.517, 1.351, 1.804],
  "smoothing": 0.0
}
```

Las interfaces están ordenadas de menor a mayor coordenada. Cada valor
corresponde al intervalo situado a continuación: aquí, suelo profundo,
intermedio y superficial. La cantidad de valores debe superar en uno a la
cantidad de interfaces. `smoothing: 0.0` significa **salto exacto**.

Para estos estratos la PINN usa una subred por material y penaliza por separado
los saltos de temperatura y flujo. FEniCSx divide geométricamente el dominio
de suelo en las interfaces. La versión de geometría de cables exige que una
interfaz no atraviese un cable; lo rechaza explícitamente si ocurre.

Se han incluido pruebas con interfaces verticales, horizontales y con un cable
situado por encima de una interfaz de suelo. La capacidad de expresar más estratos
no demuestra automáticamente su exactitud: cada nuevo caso necesita su propia
convergencia FEM y evaluación PINN.

## Transiciones continuas entre estratos

El mismo objeto admite, por ejemplo, `"smoothing": 0.1`. En ese caso se usa una
transición con tangente hiperbólica de ancho característico 0,1 m. **Este es
otro modelo físico**, con gradiente continuo y sin interfaz de salto.
La diferencia queda en el JSON y cambia su huella; no se activa suavizado
automáticamente por dificultades del entrenamiento.

## Variación espacial mediante fórmulas

```json
"conductivity": {
  "type": "expression",
  "expression": "1 + 0.4*sin(2*pi*x)*cos(2*pi*y)"
}
```

También se admite `"expression": "exp(log(10)*(x+y)/2)"` para el cuadrado
unitario. La primera expresión varía entre 0,6 y 1,4; la segunda entre 1 y 10.
Las coordenadas `x` e `y` son valores físicos en metros, por lo que los
coeficientes de la fórmula deben conservar la coherencia dimensional.

Se admiten `x`, `y`, `pi`, números finitos, `+`, `-`, `*`, `/`, `**`, `sin`,
`cos`, `exp`, `log`, `sqrt`, `tanh` y `where(condición,a,b)`. No se permiten
atributos, índices, importaciones ni llamadas a otras funciones. La
conductividad se comprueba en una cuadrícula antes de resolver; esta muestra
no sustituye una demostración de positividad en todo el dominio.

PyTorch conserva el grafo de la expresión y evalúa **div(k grad(T))**, que
incluye `grad(k) · grad(T)`. UFL incorpora la misma expresión en su formulación
débil. Una fórmula con `where` puede crear un salto; para saltos materiales
debe emplearse `type: layers`, de modo que ambos métodos conozcan también
la geometría de las interfaces.

## Solución manufacturada y fuente derivada

```json
{
  "id": "mi_caso_continuo",
  "kind": "mms",
  "bounds": [0.0, 1.0, 0.0, 1.0],
  "T0": 20.0,
  "scale": 30.0,
  "conductivity": {
    "type": "expression",
    "expression": "exp(log(10)*(x+y)/2)"
  },
  "exact_expression": "20 + 30*sin(pi*x)*sin(pi*y)",
  "source_expression": "manufactured",
  "source": "Elaboración propia"
}
```

Con `source_expression: manufactured`, el código deriva simbólicamente
`Q = -div(k grad(T_exacta))` dentro del lenguaje restringido. Así, cambiar la
conductividad modifica coherentemente la fuente de la prueba. También puede
proporcionarse una fórmula explícita de `Q`. La solución exacta fija los
contornos y sirve para evaluar; **no es una etiqueta interior del entrenamiento**.

En estratos discontinuos la fuente se deriva por región. La temperatura y el
flujo exactos deben ser continuos en cada interfaz para que no aparezca una
fuente superficial no declarada. Las pruebas contrastan la derivación
simbólica con autograd y verifican ambas trazas de la solución manufacturada.

El generador FEM manufacturado actual usa el cuadrado unitario con interfaces
alineadas con su malla; los ejemplos incluidos sitúan el salto en 0,5. Los
escenarios de cables utilizan Gmsh y coordenadas generales del JSON.

## Parámetros cubiertos

El modelo estacionario usa conductividad y fuente térmica; ambas admiten
variaciones espaciales mediante las expresiones anteriores. La capacidad
calorífica y la densidad no intervienen en una PDE estacionaria. Incorporarlas
en un problema transitorio requiere ampliar y verificar el solucionador,
además de registrar sus fórmulas: escribirlas en el archivo no las vuelve
activas automáticamente.

Los casos revisables son `mms_variable`, `mms_smooth_2d`, `mms_high_contrast`,
`mms_interface`, `mms_layered_y` y `xlpe_discrete_layers`. Sus cuadernos,
metadatos y tres mallas FEM permiten comprobar el comportamiento de cada
representación.

## Pérdidas eléctricas dependientes de la temperatura

`current` (A), `R20` (Ω/m) y `alpha` (K⁻¹) describen el conductor común del
escenario. `power` es la referencia I²R20, no la potencia final de operación.
El archivo común `electrical_model.json` declara la ley DC lineal, temperatura
de referencia, tolerancias y límite térmico. Ambos solucionadores archivan
ese archivo y su SHA-256 junto con los datos del caso.

En modo acoplado cada conductor tiene su propia temperatura y potencia:
`P[j] = I**2 * R20 * (1 + alpha * (Tc[j] - 20))`. Los resultados guardan
`conductor_C`, `powers_W_m`, `resistance_ohm_m`, `current_A` y
`electrical_residual_pct`. La hipótesis de flujo circular uniforme y la
reconstrucción radial son comunes a FEM y PINN. Un modelo que resuelva el
interior conductor debe usar generación volumétrica local J·E en W/m³;
no puede introducir directamente una potencia lineal W/m en esa ecuación.

La temperatura escalar usada en R(T) es la reconstrucción del centro del
conductor. Se supone representativa del conductor para la ley eléctrica;
no se resuelve la distribución local de resistividad dentro de su sección.
El modelo de cables exige Q=0 en suelo y rechaza fuentes volumétricas
adicionales. Las fuentes prescritas y manufacturadas pertenecen a los casos
MMS; no se ignoran silenciosamente en el solucionador acoplado.

La conductividad térmica espacial k(x,y) y la resistencia eléctrica R(Tc)
son propiedades distintas. La primera admite las fórmulas anteriores; la
segunda usa la ley explícita común. No se implementa aún k(x,y,T), ni cables
con diferentes secciones o corrientes dentro de un mismo escenario.

## Configuración numérica del muestreo

La física del caso permanece separada de las decisiones de entrenamiento.
El JSON de configuración PINN admite `sampling`: `fixed`, `uniform`,
`residual` o `gradient`; `adapt_steps`: `[400, 800]`;
`candidate_multiplier`: `4`; y `adaptive_fraction`: `0.5`.
Los pasos deben quedar dentro de Adam. `uniform` significa renovación
aleatoria sobre la misma familia geométrica de candidatos, no uniformidad
espacial del suelo. `n` controla los puntos globales; `n_interior` en los
metadatos registra el total efectivo con muestras próximas a cables.
La adaptación conserva ese total. Los resultados incluyen historial,
probabilidades, nubes y estados intermedios para repetir la selección.
