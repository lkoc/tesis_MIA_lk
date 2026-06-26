# Informe de datos de casos de estudio extraídos de papers

**Parte 2 - Ejemplos de datos recolectados del objeto de estudio**

Este informe se preparó a partir de los PDFs usados en el documento de plan de tesis y de la matriz bibliográfica local. La selección sigue la definición final del objeto de estudio del plan: el sistema cable-instalación-entorno térmico en cables eléctricos subterráneos, entendido como la unidad física formada por cable XLPE multicapa, disposición plana o trébol, bedding, backfill, suelo nativo, profundidad y condiciones térmicas de operación.

Se incluyeron los papers que entregan datos numéricos o geométricos suficientes para representar una instalación de cables o un caso térmico comparable. Se priorizaron datos que puedan alimentar una simulación FEM/PINN: geometría, materiales, conductividad térmica, resistividad térmica, humedad, condiciones de frontera, corriente, temperatura máxima y ampacidad.

## Resumen de papers con datos directos

| Clave         | Paper                    | Caso de instalación                                                                                              | Datos aprovechables                                                                                             |
| ------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| kim2025       | Kim et al. (2025)        | UPCS 154 kV con seis cables, dos capas planas, duct bank, bedding PAC/NC/arena y tres estratos de suelo natural. | Cable multicapa, geometria de instalación, suelos, bedding, fronteras térmicas, temperatura máxima y capacidad. |
| khumalo2025   | Khumalo et al. (2025)    | Cable MV XLPE bajo secado de suelo; compara humedad, resistividad, profundidad, temperatura de suelo y rating.   | Resistividad del suelo, humedad, profundidad, temperatura ambiente del suelo, ampacidad.                        |
| aldulaimi2024 | Al-Dulaimi et al. (2024) | Cable 132 kV XLPE con FEM-BPNN; arreglos plano/trébol, suelos mono/multicapa, backfill SCMB/FTB y zonas secas.   | Geometria 2D, configuraciones, materiales, conductividades, condiciones de frontera, Tmax.                      |
| atoccsa2024   | Atoccsa et al. (2024)    | Cable 220 kV XLPE con backfill térmico optimizado por PSO dinámico.                                              | Datos del cable, variables de zanja/backfill, costos, ampacidad con y sin backfill.                             |
| oclon2015     | Ocłoń et al. (2015)      | Sistema 400 kV en formación plana con tubos HDPE, SBM y FTB optimizado por PSO.                                  | Capas del cable, ductos, conductividades, dominio FEM, profundidad, separación, Tmax.                           |
| aras2005      | Aras et al. (2005)       | Cable 154 kV XLPE comparado por IEC, FEM y ensayo térmico de laboratorio.                                        | Geometria, propiedades térmicas, profundidad, dominio FEM, ampacidad/flujo térmico.                             |

## Kim et al. (2025) - UPCS 154 kV con bedding PAC/NC/arena

**Referencia.** Kim, Y.-S., Cong, H. N., Dinh, B. H., & Kim, H.-K. (2025). Effect of ambient air and ground temperatures on heat transfer in underground power cable system buried in newly developed cable bedding material. *Geothermics, 125*, 103151. https://doi.org/10.1016/j.geothermics.2024.103151

El paper representa con bastante detalle el objeto de estudio de la tesis: un sistema de cables enterrados, multicapa, dentro de una instalación 2D/3D con duct bank, material de bedding y estratos de suelo. Es el caso más completo para construir un benchmark de temperatura máxima porque conecta geometría, materiales, clima local y resultado térmico.

| Bloque           | Dato extraído                              | Valor o descripción                                                                           |
| ---------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| Sistema          | Tensión del cable                          | 154 kV                                                                                        |
| Sistema          | Arreglo                                    | Seis cables en dos capas de formación plana dentro de duct bank.                              |
| Dominio          | Entorno                                    | Bloque de bedding + tres capas de suelo natural.                                              |
| Frontera         | Borde inferior                             | Temperatura constante del terreno: 15.2 °C.                                                   |
| Frontera         | Superficie                                 | Convección con aire ambiente; velocidad de viento usada: 1.32 m/s invierno y 1.17 m/s verano. |
| Clima            | Casos ambientales                          | Verano: aire 27.2 °C; invierno: aire 5.4 °C.                                                  |
| Criterio térmico | Temperatura máxima admisible del conductor | 90 °C.                                                                                        |
| Operación        | Caso crítico reportado                     | Julio: 1026 A, aire 27.2 °C y temperatura del suelo alrededor del cable 17.0 °C.              |

| Capa / dato del cable    | Material                          | Dimensión                              | Conductividad k, W/(m K) |
| ------------------------ | --------------------------------- | -------------------------------------- | ------------------------ |
| Conductor                | Cobre                             | dc = 42.4 mm; sección nominal 1200 mm² | 400                      |
| Pantalla del conductor   | Semiconductivo                    | d = 46.4 mm                            | 0.2857                   |
| Aislamiento              | XLPE                              | d = 80.4 mm                            | 0.2857                   |
| Pantalla del aislamiento | Semiconductivo                    | d = 83.0 mm                            | 0.2857                   |
| Cinta semiconductiva     | Tape                              | d = 85.0 mm                            | 0.167                    |
| Vaina metálica           | Aluminio                          | d = 90.0 mm                            | 237                      |
| Cubierta externa         | PE                                | d = 100.0 mm                           | 0.2857                   |
| Tubo/casing              | PE                                | d = 220 mm; espesor 10 mm              | 0.2857                   |
| Dato eléctrico           | Resistencia DC a 20 °C            | 0.0151 Ω/km                            |                          |
| Dato eléctrico           | Coeficiente térmico del conductor | 0.0393                                 |                          |

| Suelo  | USCS | Peso unitario natural, kN/m³ | Humedad, % | k, W/(m K) | Resistividad térmica, °C cm/W |
| ------ | ---- | ---------------------------- | ---------- | ---------- | ----------------------------- |
| Capa 1 | SC   | 18.081                       | 23.25      | 1.804      | 55.44                         |
| Capa 2 | CL   | 18.884                       | 26.27      | 1.351      | 74.01                         |
| Capa 3 | CL   | 19.987                       | 23.14      | 1.517      | 65.94                         |

| Material de bedding/backfill | Densidad seca, kg/m³ | Fluidez / bleeding            | Resistencia compresiva, MPa | k, W/(m K) |
| ---------------------------- | -------------------- | ----------------------------- | --------------------------- | ---------- |
| Arena natural                | 1603                 | No reportada                  | No reportada                | 1.365      |
| Concreto normal (NC)         | 2093                 | Fluidez 180 mm                | 15.0                        | 2.093      |
| PAC                          | 1410                 | Bleeding 1.7%; grout 382 mm   | 1.56                        | 2.094      |
| CLSM-RS10                    | 2735                 | Bleeding 3.0%; fluidez 235 mm | 2.735                       | 2.150      |

| Condición                                         | Arena natural | PAC / NC                | Dato útil para la tesis                         |
| ------------------------------------------------- | ------------- | ----------------------- | ----------------------------------------------- |
| Temperatura máxima del conductor en verano        | 77.6 °C       | 70.6 °C                 | Backfill de mayor k reduce Tmax en 7.0 °C.      |
| Temperatura máxima del conductor en invierno      | 57.1 °C       | 49.9 °C                 | Reducción aproximada de 7.2 °C.                 |
| Diferencia conductor central-superior en verano   | 1.0 °C        | 1.4 °C                  | Evidencia de gradiente interno por posición.    |
| Diferencia conductor central-superior en invierno | 1.3 °C        | 1.6 °C                  | Confirma sensibilidad geométrica del arreglo.   |
| Capacidad en caso crítico                         | No aplica     | 1026 A con Tmax 70.6 °C | Escenario calibrable para comparación FEM/PINN. |

**Aporte para la tesis.** Proporciona un caso con cable XLPE multicapa, suelos reales, bedding de distinta conductividad y resultados de temperatura máxima. Es especialmente útil para escenarios de relleno de alta conductividad y para validar que el artefacto PINN capture la influencia del clima y de la frontera superior.

## Khumalo et al. (2025) - rating bajo secado de suelo

**Referencia.** Khumalo, N. Q., Naidoo, R. M., Mbungu, N. T., & Bansal, R. C. (2025). A critical assessment of cable rating methods under soil drying out conditions. *International Transactions on Electrical Energy Systems, 2025*, Article 5946564. https://doi.org/10.1155/etep/5946564

Este paper aporta datos muy directos para el componente suelo nativo/estado hídrico del objeto de estudio. Su valor principal es cuantificar cómo la resistividad térmica aumenta cuando el suelo se seca y cómo eso reduce la ampacidad. También da sensibilidad a profundidad y temperatura del suelo.

| Componente              | Dato extraído                  | Valor o descripción                                                                                                                                                                     |
| ----------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cable                   | Tipo                           | Cable MV XLPE de tres núcleos.                                                                                                                                                          |
| Cable                   | Capas identificadas            | Conductor de cobre, pantalla semiconductiva, XLPE, pantalla del núcleo, cinta semiconductiva, pantalla de cobre, relleno, bedding FR-PVC, armadura de acero galvanizado y cubierta PVC. |
| Condición de referencia | Resistividad térmica del suelo | 1.2 K m/W.                                                                                                                                                                              |
| Condición de referencia | Temperatura ambiente del suelo | 25 °C.                                                                                                                                                                                  |
| Condición de referencia | Profundidad de tendido         | 850 mm.                                                                                                                                                                                 |

| Humedad del suelo, % | Resistividad térmica, K m/W | Lectura para el objeto de estudio   |
| -------------------- | --------------------------- | ----------------------------------- |
| 14.5                 | 0.596                       | Suelo húmedo / condición favorable. |
| 5                    | 1.646                       | Secado parcial relevante.           |
| 2                    | 2.341                       | Secado severo.                      |
| 0                    | 3.720                       | Suelo seco / condición crítica.     |

| Condición                    | Resistividad, K m/W | Cable A, A | Cable B, A | Cable C, A | Promedio, A |
| ---------------------------- | ------------------- | ---------- | ---------- | ---------- | ----------- |
| Suelo 14.5% humedad          | 0.596               | 518.746    | 519.540    | 516.723    | 518.34      |
| Suelo importado / referencia | 1.200               | 382.371    | 382.869    | 381.143    | 382.13      |
| Suelo 5% humedad             | 1.646               | 330.663    | 331.070    | 329.667    | 330.47      |
| Suelo 2% humedad             | 2.341               | 280.169    | 280.495    | 279.372    | 280.01      |
| Suelo seco                   | 3.720               | 224.330    | 224.582    | 223.730    | 224.21      |

| Variable                       | Valores evaluados             | Resultado extraído                                            |
| ------------------------------ | ----------------------------- | ------------------------------------------------------------- |
| Temperatura ambiente del suelo | 24, 25, 26, 27 y 28 °C        | Para Cable A: 386.597, 382.371, 378.099, 373.777 y 369.405 A. |
| Profundidad de tendido         | 750, 850, 950, 1000 y 1150 mm | Para Cable A: 382.321, 382.371, 377.308, 375.040 y 369.062 A. |
| Peor caso                      | 3.72 K m/W; 1150 mm; 28 °C    | Cable A/B/C: 208.70, 208.93 y 208.19 A.                       |
| Caso ideal                     | 1.2 K m/W; 850 mm; 25 °C      | Cable A/B/C: 382.37, 382.87 y 381.14 A.                       |
| Reducción promedio             | Peor caso frente a referencia | 45.40%.                                                       |

**Aporte para la tesis.** Es una fuente cuantitativa para construir escenarios de baja conductividad/alta resistividad por secado. Permite definir contrastes de k, validar sensibilidad de ampacidad y justificar la necesidad de representar la conductividad espacialmente variable en vez de asumir un suelo homogéneo fijo.

## Al-Dulaimi et al. (2024) - FEM-BPNN con suelos y backfill variables

**Referencia.** Al-Dulaimi, A. A., Guneser, M. T., Hameed, A. A., García Márquez, F. P., & Gouda, O. E. (2024). Adaptive FEM-BPNN model for predicting underground cable temperature considering varied soil composition. *Engineering Science and Technology, an International Journal, 51*, 101658. https://doi.org/10.1016/j.jestch.2024.101658

El paper es relevante porque transforma muchos escenarios FEM de cables enterrados en una base de aprendizaje para una red neuronal. Aunque el modelo de red no es PINN, la estructura de datos de entrada y salida se alinea con la tesis: geometría, configuración plana/trébol, profundidad, distancia entre cables, backfill, suelo y temperatura máxima.

| Bloque                    | Dato extraído                    | Valor o descripción                                |
| ------------------------- | -------------------------------- | -------------------------------------------------- |
| Cable                     | Tensión                          | 132 kV.                                            |
| Cable                     | Sección geométrica del conductor | 749.9 mm².                                         |
| Criterio térmico          | Temperatura máxima admisible     | 90 °C para conductor XLPE en régimen estacionario. |
| Dominio FEM               | Área de simulación               | 4 m x 4 m.                                         |
| Backfill                  | Ancho del bloque                 | 2 m.                                               |
| Profundidad H             | Valores                          | 0.8, 1.0 y 1.2 m.                                  |
| Separación S              | Formación plana                  | 0.1 a 0.4 m.                                       |
| Separación S              | Formación trébol                 | 0.008 a 0.2 m según el paper.                      |
| Temperatura superficial   | Tsg                              | 20 a 50 °C.                                        |
| Temperatura inicial/suelo | Tsoil                            | 14 °C.                                             |
| Carga                     | Máxima reportada para dataset    | 945 A.                                             |

| Material / zona               | Conductividad k, W/(m K) | Observación                                            |
| ----------------------------- | ------------------------ | ------------------------------------------------------ |
| Conductor de cobre            | 386                      | Capa interna conductora.                               |
| Aislamiento XLPE              | 0.2875                   | Capa dieléctrica.                                      |
| Cubierta HDPE                 | 0.2875                   | Cubierta externa.                                      |
| Pantalla/vaina de cobre       | 386                      | Ruta metálica de alta k.                               |
| Suelo medio                   | 2.28                     | Condición de suelo usada como referencia.              |
| SCMB                          | 1.00                     | Sand-cement mixture backfill.                          |
| FTB                           | 1.54                     | Fluidized thermal backfill.                            |
| Suelo nativo / sand clay loam | Variable                 | El paper evalúa estados húmedos/secos y composiciones. |

| Escenario FEM                       | Backfill | Tmax reportada, °C | Lectura                                        |
| ----------------------------------- | -------- | ------------------ | ---------------------------------------------- |
| Una capa homogénea, formación plana | SCMB     | 84.31              | Por debajo del límite de 90 °C.                |
| Una capa homogénea, formación plana | FTB      | 82.99              | FTB reduce Tmax frente a SCMB.                 |
| Una capa homogénea, trébol          | SCMB     | 86.10              | Mayor Tmax por proximidad térmica.             |
| Una capa homogénea, trébol          | FTB      | 84.64              | Reducción con FTB, pero sigue mayor que plana. |
| Multicapa homogénea, plana          | SCMB     | 82.58              | La estratificación cambia el resultado.        |
| Multicapa homogénea, plana          | FTB      | 81.22              | Mejor disipación.                              |
| Multicapa homogénea, trébol         | SCMB     | 84.64              | Mayor concentración térmica.                   |
| Multicapa homogénea, trébol         | FTB      | 83.15              | Mejor que SCMB.                                |
| Multicapa no homogénea, plana       | SCMB     | 83.65              | Caso heterogéneo explícito.                    |
| Multicapa no homogénea, plana       | FTB      | 82.42              | Backfill de mayor k reduce Tmax.               |
| Multicapa no homogénea, trébol      | SCMB     | 86.26              | Caso más exigente entre los reportados.        |
| Multicapa no homogénea, trébol      | FTB      | 84.89              | FTB reduce el máximo.                          |

| Variable del dataset | Uso para una versión PINN / benchmark                            |
| -------------------- | ---------------------------------------------------------------- |
| Tsg                  | Condición de frontera térmica superior.                          |
| S                    | Distancia entre cables; controla proximidad de fuentes.          |
| H                    | Profundidad de instalación.                                      |
| kbackfill            | Contraste térmico entre relleno y suelo.                         |
| Código de entorno    | Identifica suelo homogéneo, multicapa, no homogéneo o zona seca. |
| Tmax                 | Salida escalar para comparar con campo térmico 2D.               |

**Aporte para la tesis.** Es una plantilla de generación de escenarios paramétricos para evaluar el artefacto: permite variar geometría, patrón de heterogeneidad y material de relleno, y comparar la temperatura máxima de una PINN con resultados FEM.

## Atoccsa et al. (2024) - optimización de ampacidad con backfill térmico

**Referencia.** Atoccsa, B. A., Puma, D. W., Mendoza, D., Urday, E., Ronceros, C., & Palma, M. T. (2024). Optimization of ampacity in high-voltage underground cables with thermal backfill using dynamic PSO and adaptive strategies. *Energies, 17*(5), 1023. https://doi.org/10.3390/en17051023

El paper aporta un caso de diseño donde el backfill térmico no solo se modela, sino que se optimiza junto con la geometría de instalación. Para la tesis sirve como caso de alta tensión donde la ampacidad cambia de forma importante al modificar el entorno térmico alrededor de los cables.

| Dato del cable 220 kV XLPE          | Valor                                    |
| ----------------------------------- | ---------------------------------------- |
| Conductor                           | Cobre Milliken de 5 segmentos, recocido. |
| Sección del conductor               | 2000 mm².                                |
| Diámetro del conductor dc           | 54.5 mm.                                 |
| Espesor de pantalla semiconductiva  | 3.5 mm.                                  |
| Espesor de aislamiento XLPE         | 24.0 mm.                                 |
| Diámetro externo del aislamiento Di | 107.1 mm.                                |
| Espesor de vaina de aluminio        | 2.8 mm.                                  |
| Diámetro externo de vaina Ds        | 137.4 mm.                                |
| Espesor de cubierta externa         | 5.0 mm.                                  |
| Diámetro externo del cable De       | 147.7 mm.                                |
| Temperatura máxima del conductor    | 90 °C.                                   |
| Frecuencia                          | 60 Hz.                                   |
| Resistencia del conductor a 20 °C   | 0.009 Ω/km.                              |
| Tensión nominal fase-fase           | 220 kV.                                  |

| Variable de diseño | Significado                         | Límite inferior | Límite superior |
| ------------------ | ----------------------------------- | --------------- | --------------- |
| L                  | Profundidad del cable               | 0.5 m           | 2.0 m           |
| LG                 | Profundidad del centro del backfill | 0.6 m           | 4.0 m           |
| w                  | Ancho del backfill                  | 1.2 m           | 4.0 m           |
| h                  | Espesor/altura del backfill         | 0.6 m           | 3.0 m           |
| s                  | Separación entre cables             | De ≈ 0.147 m    | 2.0 m           |
| s1                 | Separación auxiliar                 | 0.3 m           | 2.0 m           |

| Componente de costo       | Costo unitario | Expresión geométrica usada |
| ------------------------- | -------------- | -------------------------- |
| Excavación                | 16.5 USD/m³    | w·LG + w·h/2               |
| Remoción de tierra        | 13.15 USD/m³   | w·LG + w·h/2               |
| Backfill de arena térmica | 28.5 USD/m³    | w·h - (3/4)πDe²            |

| Resultado optimizado     | Valor                                 |
| ------------------------ | ------------------------------------- |
| L                        | 0.500 m                               |
| LG                       | 0.872 m                               |
| w                        | 3.562 m                               |
| h                        | 1.344 m                               |
| s                        | 1.481 m                               |
| λ1                       | 2.667                                 |
| Costo total              | 300 USD                               |
| Costo de backfill        | 94.7 USD                              |
| Ampacidad con backfill   | 1156.915 A                            |
| Ampacidad sin backfill   | 969.9 A                               |
| Incremento aproximado    | 18.45% respecto al caso sin backfill. |
| Pérdidas dieléctricas Wd | 3 x 3.546 W/m                         |
| Pérdidas de carga Wl     | 3 x 17.67 W/m                         |

**Aporte para la tesis.** Aporta un caso donde la variable de interés no es solo Tmax, sino la ampacidad resultante de una geometría y un backfill. Es útil para verificar que el artefacto pueda reproducir tendencias de diseño: mayor k y geometría favorable elevan Imax.

## Ocłoń et al. (2015) - bedding FTB optimizado en sistema 400 kV

**Referencia.** Ocłoń, P., Cisek, P., Taler, D., Pilarczyk, M., & Szwarc, T. (2015). Optimizing of the underground power cable bedding using momentum-type particle swarm optimization method. *Energy, 92*(2), 230–239. https://doi.org/10.1016/j.energy.2015.04.100

Este paper modela tres cables subterráneos de 400 kV en formación plana, dentro de tubos HDPE llenos con mezcla SBM y rodeados por backfill FTB. Es muy útil para la tesis porque separa claramente suelo madre, ducto, mezcla de relleno dentro del tubo y bedding térmico exterior.

| Elemento del sistema         | Dato extraído                                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Configuración                | Tres cables subterráneos de 400 kV en formación plana/in-line.                                                                        |
| Ducto                        | Tubos HDPE con diámetro externo 278 mm y espesor 14 mm.                                                                               |
| Relleno dentro del ducto     | SBM: mezcla sand-bentonite, densidad 1700 kg/m³, k = 0.95 W/(m K).                                                                    |
| Bedding exterior             | FTB: mezcla SGFC con 41% agregado fino, 49% agregado grueso, 2.5% cemento y 7.5% fly ash; densidad seca 2187 kg/m³; k = 1.54 W/(m K). |
| Suelo madre                  | k = 1.00 W/(m K).                                                                                                                     |
| Ducto HDPE                   | k = 0.48 W/(m K).                                                                                                                     |
| Dominio FEM                  | Modelo 2D estacionario; dominio cuadrado de 10 m x 10 m usando simetría.                                                              |
| Profundidad                  | H = 2 m desde el nivel de cruce vial.                                                                                                 |
| Frontera superior            | Temperatura Tg = 30 °C.                                                                                                               |
| Fronteras laterales/inferior | Adiabáticas por simetría/aislamiento térmico en el dominio.                                                                           |
| Carga máxima asumida         | 1145 A.                                                                                                                               |
| Criterio de operación        | Topt = 65 °C; el máximo del conductor no debe exceder ese valor.                                                                      |

| Dato del cable 400 kV                     | Valor        |
| ----------------------------------------- | ------------ |
| Sección del conductor                     | 1600 mm².    |
| Diámetro del conductor                    | 49.6 mm.     |
| Espesor total de aislamiento              | 27 mm.       |
| Diámetro externo                          | 127.9 mm.    |
| Resistencia DC a 20 °C                    | 0.0113 Ω/km. |
| Resistencia AC a 65 °C y 50 Hz            | 0.0157 Ω/km. |
| Current loading in ground in-line a 65 °C | 1145 A.      |

| Capa del cable | Material | Espesor / radio            | k, W/(m K) |
| -------------- | -------- | -------------------------- | ---------- |
| Conductor      | Cobre    | dc = 49.6 mm; rc = 24.8 mm | 400        |
| Aislamiento    | XLPE     | 30.5 mm; rins = 55.3 mm    | 0.2875     |
| Vaina/pantalla | Cobre    | 6.4 mm; rsh = 61.7 mm      | 400        |
| Cubierta       | HDPE     | 5.1 mm; rj = 66.8 mm       | 0.2875     |

| Variable optimizada                    | Rango / resultado                                                                                    |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| l, separación entre ejes de cables     | Rango 0.3 a 0.6 m; óptimo 0.600 m.                                                                   |
| s, distancia al borde derecho del FTB  | Rango 0.2 a 0.4 m; óptimo 0.2000-0.20044 m.                                                          |
| b, distancia al borde superior del FTB | Rango 0.2 a 0.4 m; óptimo 0.31327-0.31366 m.                                                         |
| p, distancia al borde inferior del FTB | Rango 0.2 a 0.4 m; óptimo 0.2000-0.20044 m.                                                          |
| Tmax                                   | 65 °C en el conductor central.                                                                       |
| Área media FTB modelada Ab             | 0.3195-0.3196 m²; el área real por simetría es aprox. 0.6390 m².                                     |
| Interpretación del resultado           | El conductor central queda más caliente por interacción térmica y peor disipación que los laterales. |

**Aporte para la tesis.** Es un caso muy controlado para estudiar proximidad de fuentes y backfill localizado. La geometría permite crear escenarios 2D con contraste entre suelo madre, FTB, SBM y HDPE, exactamente el tipo de heterogeneidad espacial que el plan exige evaluar.

## Aras et al. (2005) - comparación IEC/FEM/ensayo en cable 154 kV

**Referencia.** Aras, F., Oysu, C., & Yilmaz, G. (2005). An assessment of the methods for calculating ampacity of underground power cables. *Electric Power Components and Systems, 33*(12), 1385–1402. https://doi.org/10.1080/15325000590964425

Este paper es más antiguo, pero sirve como benchmark clásico porque compara métodos de cálculo de ampacidad con FEM y con una prueba térmica. No tiene el nivel de detalle de suelos y backfill de los papers recientes, pero sí entrega una instalación homogénea útil para verificación inicial.

| Variable                             | Dato extraído                                           |
| ------------------------------------ | ------------------------------------------------------- |
| Cable                                | 154 kV XLPE.                                            |
| Temperatura de suelo                 | 20 °C para el caso del norte de Turquía.                |
| Profundidad de enterramiento         | 1.2 m.                                                  |
| Medio                                | Suelo homogéneo.                                        |
| Temperatura máxima de operación XLPE | 90 °C.                                                  |
| Dominio FEM                          | 18 m de ancho x 10 m de profundidad; fronteras a 20 °C. |
| Pérdida dieléctrica                  | 3.57 W/m.                                               |
| Conductividad del suelo              | 1.2 W/(m K).                                            |
| Conductividad XLPE                   | 0.2857 W/(m K).                                         |
| Conductividad pantalla               | 384.6 W/(m K).                                          |

| Dimensión del cable           | Valor     |
| ----------------------------- | --------- |
| Diámetro del conductor        | 37.7 mm.  |
| Diámetro XLPE + semiconductor | 81.7 mm.  |
| Diámetro de pantalla          | 98.7 mm.  |
| Diámetro de cubierta          | 106.7 mm. |
| Profundidad h                 | 1200 mm.  |

| Resultado / referencia       | Dato                                                                                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Mallas FEM ensayadas         | 557, 690, 1074 y 1486 elementos; se selecciona 1074 elementos.                                                                           |
| Comparación FEM vs IEC       | Diferencia aproximada del orden de 1% para el caso principal.                                                                            |
| Sensibilidad del aislamiento | Reducir el espesor de XLPE de 22 mm a 17 mm incrementa la ampacidad en 2.9%.                                                             |
| Flujo térmico de referencia  | Para 22 mm de aislamiento XLPE se reporta 0.585 W/m² en la superficie del conductor.                                                     |
| Ensayo experimental          | 15 m de cable 154 kV XLPE; ciclos térmicos aproximados de 310 K a 380 K; suelo representado por una capa de papel con k = 0.475 W/(m K). |

**Aporte para la tesis.** Sirve como caso de verificación base con suelo homogéneo y geometría sencilla. Puede preceder a los casos heterogéneos: primero se comprueba la conducción radial/enterrada y luego se agregan backfill, estratos o zonas secas.

## Variables consolidadas para la parte 2

La extracción permite organizar los datos de los papers como una biblioteca de casos para construir ejemplos reproducibles del objeto de estudio. La tabla siguiente resume cómo deberían mapearse a archivos de datos o escenarios de simulación.

| Grupo de datos          | Variables extraídas                                                                                          | Papers fuente                         | Uso recomendado                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------- |
| Cable multicapa         | Diámetros, espesores, materiales, k de conductor/aislamiento/pantallas/cubierta.                             | Kim, Al-Dulaimi, Atoccsa, Ocłoń, Aras | Definir geometría interna o equivalente térmico del cable.                |
| Instalación             | Profundidad H/L, separación S/l/s, formación plana o trébol, ductos, arreglo de 1/3/6 cables.                | Kim, Al-Dulaimi, Atoccsa, Ocłoń, Aras | Generar dominios 2D reproducibles y variar proximidad de fuentes.         |
| Suelo nativo            | k, resistividad térmica, humedad, clasificación USCS, temperatura de suelo.                                  | Kim, Khumalo, Aras, Ocłoń             | Modelar condición homogénea, estratificada o seca.                        |
| Bedding/backfill        | PAC, NC, arena, SCMB, FTB, SBM, HDPE, conductividad y dimensiones.                                           | Kim, Al-Dulaimi, Atoccsa, Ocłoń       | Construir heterogeneidades de alta/baja k y escenarios de mejora térmica. |
| Condiciones de frontera | Temperatura superficial/ambiente, convección, viento, temperatura inferior constante, fronteras adiabáticas. | Kim, Al-Dulaimi, Ocłoń, Aras          | Definir problemas directos comparables con FEM/PINN.                      |
| Resultados térmicos     | Tmax, diferencias por posición, campo de temperatura, criterio 65/90 °C.                                     | Kim, Al-Dulaimi, Ocłoń, Aras          | Validar salida de campo y localización del punto caliente.                |
| Resultados operativos   | Ampacidad Imax, reducción por secado, incremento con backfill.                                               | Khumalo, Atoccsa, Aras, Kim           | Evaluar estimación de Imax y sensibilidad a k.                            |

| Nivel de uso                     | Caso recomendado | Razón                                                                         |
| -------------------------------- | ---------------- | ----------------------------------------------------------------------------- |
| Verificación simple              | Aras 2005        | Suelo homogéneo, dominio FEM claro y comparación IEC/FEM.                     |
| Heterogeneidad por humedad       | Khumalo 2025     | Datos numéricos directos de resistividad-humedad-ampacidad.                   |
| Backfill de alta conductividad   | Kim 2025         | Compara arena natural, NC, PAC y CLSM con temperaturas máximas.               |
| Escenarios paramétricos          | Al-Dulaimi 2024  | Incluye profundidad, separación, configuración, suelos multicapa y backfills. |
| Optimización de ampacidad        | Atoccsa 2024     | Relaciona dimensiones de backfill con Imax y costo.                           |
| Backfill localizado y proximidad | Ocłoń 2015       | Dominio 2D con FTB/SBM/HDPE y fuente central crítica.                         |

## Papers usados sin tabla de caso directo

Los demás papers académicos del documento se usan como soporte metodológico, revisión del estado del arte o base de PINN/DSR. No se incluyeron como tablas de caso porque no entregan un escenario de instalación de cable con datos suficientes de geometría, suelo/relleno y resultado térmico-operativo comparable.

| Tipo                                 | Ejemplos                                                                | Motivo                                                                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Base analítica / normativa histórica | Neher y McGrath (1957)                                                  | Aporta formulación de cálculo de temperatura y capacidad, no un caso moderno de instalación con datos completos de suelo/backfill. |
| Revisiones de cable rating           | Enescu et al. (2020, 2021); systematic mapping de DTR; overview térmico | Sirven para justificar variables y brechas, pero consolidan literatura en vez de reportar un único dataset de instalación.         |
| PINN y SciML                         | Raissi et al.; Xing et al.; papers de conducción directa/inversa        | Aportan método computacional, no datos de cables subterráneos.                                                                     |
| Ciencia del diseño                   | Peffers; Hevner; Gregor                                                 | Soporte metodológico de DSR, sin datos físicos de cable.                                                                           |

## Notas de trazabilidad

- Los valores provienen de tablas, figuras y texto técnico de los PDFs locales copiados en `Entrega_Google_Drive/01_papers_usados`.

- Cuando un paper reporta varios resultados equivalentes, se priorizaron los que describen mejor el objeto de estudio de la tesis: cable, instalación, suelo/relleno, frontera térmica y respuesta Tmax/Imax.

- Las tablas de este informe son una extracción de trabajo para preparar datos de ejemplo. Para una publicación o capítulo final conviene volver a citar página, tabla o figura exacta en cada valor numérico.