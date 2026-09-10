---
title: "Informe de datos de casos de estudio y referencias de verificación térmica"
author: "Luis Enrique Koc Góngora y Herbert Antonio Meléndez García"
lang: es
---

El informe organiza casos obtenidos de los PDF usados en el plan de tesis, la matriz bibliográfica local y los documentos almacenados en Zotero. La selección distingue entre instancias físicas del sistema cable--instalación--entorno térmico y casos matemáticos o normativos para verificar el artefacto PINN antes de aplicarlo al objeto físico.

El conjunto incluye fuentes con datos suficientes para reproducir una instalación, un cálculo normativo o un problema de conducción de calor. Para cada caso se priorizan la geometría, los materiales, las propiedades térmicas, las condiciones de frontera y una solución de referencia analítica, numérica o publicada.

## Resumen de fuentes con instancias directas

| Clave         | Fuente                   | Caso de instalación                                                                                              | Datos aprovechables                                                                                             |
| ------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| kim2025       | Kim et al. (2025)        | UPCS 154 kV con seis cables, dos capas planas, duct bank, bedding PAC/NC/arena y tres estratos de suelo natural. | Cable multicapa, geometria de instalación, suelos, bedding, fronteras térmicas, temperatura máxima y capacidad. |
| khumalo2025   | Khumalo et al. (2025)    | Cable MV XLPE bajo secado de suelo; compara humedad, resistividad, profundidad, temperatura de suelo y rating.   | Resistividad del suelo, humedad, profundidad, temperatura ambiente del suelo, ampacidad.                        |
| aldulaimi2024 | Al-Dulaimi et al. (2024) | Cable 132 kV XLPE con FEM-BPNN; arreglos plano/trébol, suelos mono/multicapa, backfill SCMB/FTB y zonas secas.   | Geometria 2D, configuraciones, materiales, conductividades, condiciones de frontera, Tmax.                      |
| atoccsa2024   | Atoccsa et al. (2024)    | Cable 220 kV XLPE con backfill térmico optimizado por PSO dinámico.                                              | Datos del cable, variables de zanja/backfill, costos, ampacidad con y sin backfill.                             |
| oclon2015     | Ocłoń et al. (2015)      | Sistema 400 kV en formación plana con tubos HDPE, SBM y FTB optimizado por PSO.                                  | Capas del cable, ductos, conductividades, dominio FEM, profundidad, separación, Tmax.                           |
| aras2005      | Aras et al. (2005)       | Cable 154 kV XLPE comparado por IEC, FEM y ensayo térmico de laboratorio.                                        | Geometria, propiedades térmicas, profundidad, dominio FEM, ampacidad/flujo térmico.                             |
| quan2019      | Quan et al. (2019)       | Tres cables enterrados en formación plana o trébol, con suelo nativo o FTB y conductividad dependiente de T.    | Dominio 2D, geometría, fronteras, profundidad, separación, temperatura y ampacidad.                             |
| oladunjoye2012 | Oladunjoye et al. (2012) | Diez mediciones in situ de suelo en una central eléctrica de Nigeria.                                           | Resistividad, conductividad, temperatura, humedad, densidad y porosidad del suelo.                              |
| mobius2025    | Möbius et al. (2025)     | Tres cables MV en trébol, enterrados en Hamburgo y evaluados con 32 años de datos de suelo.                     | Composición y humedad del suelo, temperatura, resistividad y ampacidad estacional.                              |
| cigre2022_case0 | CIGRE TB 880 (2022)    | Cable 132 kV enterrado; caso base y 14 variantes de instalación y pérdidas.                                    | Geometría, parámetros IEC, pérdidas, temperaturas y ampacidades convergidas.                                    |
| cigre2025_case1 | CIGRE TB 963 (2025)    | Tres cables 132 kV en ductos HDPE, backfill y suelo; nueve variantes FEM 2D.                                   | Dominio, cable multicapa, materiales, fronteras, pérdidas y ampacidades.                                        |
| iec60853_annexA | IEC 60853-3 (2002)     | Cable tripolar 132 kV sometido a carga cíclica y secado parcial del suelo.                                     | Dimensiones, resistencias y capacitancias térmicas, ciclo de carga y corriente pico.                            |
| pan2025_plate | Pan et al. (2025)         | Placa rectangular 2D sin fuente con una condición de borde desconocida.                                       | Dominio, PDE de Laplace, fronteras, referencia FEM, muestreo y errores PINN.                                    |
| cigre2025_annulus | CIGRE TB 963 (2025)  | Anillo conductor homogéneo con flujo interior y temperatura exterior prescrita.                               | Solución analítica completa para verificar el campo de temperatura.                                            |
| hahn2012_rect | Hahn y Özişik (2012)      | Rectángulo 2D estacionario con tres bordes a temperatura fija y uno convectivo.                                | Solución analítica en serie para verificar una condición de Robin.                                             |
| mms2d_suite   | Elaboración propia        | Dos problemas manufacturados 2D: conductividad variable e interfaz multimaterial.                             | Solución exacta, fuente compatible, fronteras e interfaces para pruebas unitarias de la PINN.                  |

## Kim et al. (2025) - UPCS 154 kV con bedding PAC/NC/arena

**Referencia.** Kim, Y.-S., Cong, H. N., Dinh, B. H., & Kim, H.-K. (2025). Effect of ambient air and ground temperatures on heat transfer in underground power cable system buried in newly developed cable bedding material. *Geothermics, 125*, 103151. https://doi.org/10.1016/j.geothermics.2024.103151

Kim et al. representan un sistema de cables multicapa enterrados en un banco de ductos, rodeados por material de \textit{bedding} y varios estratos de suelo. El caso relaciona la geometría y las propiedades térmicas de la instalación con el clima local, la temperatura máxima y la ampacidad.

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

**Aporte a la tesis.** La investigación usa este caso para evaluar rellenos con distinta conductividad y condiciones climáticas en la frontera superior. La combinación de un cable XLPE multicapa, estratos de suelo y resultados térmicos publicados permite comparar la respuesta del artefacto PINN con una instalación físicamente definida.

## Khumalo et al. (2025) - rating bajo secado de suelo

**Referencia.** Khumalo, N. Q., Naidoo, R. M., Mbungu, N. T., & Bansal, R. C. (2025). A critical assessment of cable rating methods under soil drying out conditions. *International Transactions on Electrical Energy Systems, 2025*, Article 5946564. https://doi.org/10.1155/etep/5946564

Khumalo et al. cuantifican la relación entre el estado hídrico del suelo, la resistividad térmica y la ampacidad. El estudio también analiza la sensibilidad de la respuesta frente a la profundidad de instalación y la temperatura del terreno.

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

**Aporte a la tesis.** Estos datos permiten construir escenarios de baja conductividad y alta resistividad asociados con el secado. Así, el artefacto puede evaluarse frente a contrastes de $k(x,y)$ y cambios de ampacidad que una representación homogénea fija no describe.

## Al-Dulaimi et al. (2024) - FEM-BPNN con suelos y backfill variables

**Referencia.** Al-Dulaimi, A. A., Guneser, M. T., Hameed, A. A., García Márquez, F. P., & Gouda, O. E. (2024). Adaptive FEM-BPNN model for predicting underground cable temperature considering varied soil composition. *Engineering Science and Technology, an International Journal, 51*, 101658. https://doi.org/10.1016/j.jestch.2024.101658

Al-Dulaimi et al. convierten varios escenarios FEM de cables enterrados en una base de aprendizaje para una red neuronal. Aunque la red no es una PINN, sus entradas y salidas reúnen las variables del estudio: geometría, configuración, profundidad, separación, \textit{backfill}, suelo y temperatura máxima.

| Bloque                    | Dato extraído                    | Valor o descripción                                |
| ------------------------- | -------------------------------- | -------------------------------------------------- |
| Cable                     | Tensión                          | 132 kV.                                            |
| Cable                     | Sección geométrica del conductor | 749.9 mm².                                         |
| Criterio térmico          | Temperatura máxima admisible     | 90 °C para conductor XLPE en régimen estacionario. |
| Dominio FEM               | Área de simulación               | 4 m x 4 m.                                         |
| Backfill                  | Ancho del bloque                 | 2 m.                                               |
| Profundidad H             | Valores                          | 0.8, 1.0 y 1.2 m.                                  |
| Separación S              | Formación plana                  | 0.1 a 0.4 m.                                       |
| Separación S              | Formación trébol                 | 0.008 a 0.2 m según el artículo.                   |
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
| Suelo nativo / sand clay loam | Variable                 | El estudio evalúa estados húmedos/secos y composiciones. |

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

| Variable del conjunto de datos | Uso en el artefacto PINN o en el caso de verificación         |
| -------------------- | ---------------------------------------------------------------- |
| Tsg                  | Condición de frontera térmica superior.                          |
| S                    | Distancia entre cables; controla proximidad de fuentes.          |
| H                    | Profundidad de instalación.                                      |
| kbackfill            | Contraste térmico entre relleno y suelo.                         |
| Código de entorno    | Identifica suelo homogéneo, multicapa, no homogéneo o zona seca. |
| Tmax                 | Salida escalar para comparar con campo térmico 2D.               |

**Aporte a la tesis.** La estructura paramétrica del caso permite variar la geometría, el patrón de heterogeneidad y el material de relleno. La investigación puede usar esas variaciones para comparar la temperatura máxima estimada por el artefacto PINN con una referencia FEM.

## Atoccsa et al. (2024) - optimización de ampacidad con backfill térmico

**Referencia.** Atoccsa, B. A., Puma, D. W., Mendoza, D., Urday, E., Ronceros, C., & Palma, M. T. (2024). Optimization of ampacity in high-voltage underground cables with thermal backfill using dynamic PSO and adaptive strategies. *Energies, 17*(5), 1023. https://doi.org/10.3390/en17051023

Atoccsa et al. analizan un caso de diseño en el que el \textit{backfill} térmico se optimiza junto con la geometría de instalación. Los resultados muestran cómo la modificación del entorno próximo al cable cambia la ampacidad de un sistema de alta tensión.

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

**Aporte a la tesis.** Este caso incorpora la ampacidad como respuesta de una geometría y un \textit{backfill} definidos, además de la temperatura máxima. Por tanto, permite verificar si el artefacto reproduce las tendencias de diseño asociadas con la conductividad térmica y las dimensiones del relleno.

## Ocłoń et al. (2015) - bedding FTB optimizado en sistema 400 kV

**Referencia.** Ocłoń, P., Cisek, P., Taler, D., Pilarczyk, M., & Szwarc, T. (2015). Optimizing of the underground power cable bedding using momentum-type particle swarm optimization method. *Energy, 92*(2), 230–239. https://doi.org/10.1016/j.energy.2015.04.100

Ocłoń et al. modelan tres cables subterráneos de 400 kV en formación plana, instalados en tubos HDPE con mezcla SBM y rodeados por \textit{backfill} FTB. La geometría separa el suelo nativo, el ducto, el relleno interior y el \textit{bedding} térmico exterior.

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

**Aporte a la tesis.** La separación de materiales permite estudiar la proximidad entre fuentes térmicas y rellenos localizados. A partir de esta geometría se construyen escenarios 2D con contrastes entre suelo nativo, FTB, SBM y HDPE, de acuerdo con la heterogeneidad espacial definida en el plan.

## Aras et al. (2005) - comparación IEC/FEM/ensayo en cable 154 kV

**Referencia.** Aras, F., Oysu, C., & Yilmaz, G. (2005). An assessment of the methods for calculating ampacity of underground power cables. *Electric Power Components and Systems, 33*(12), 1385–1402. https://doi.org/10.1080/15325000590964425

Aras et al. comparan métodos de cálculo de ampacidad con FEM y con un ensayo térmico. Aunque la instalación no incorpora el detalle de suelos y rellenos de los estudios posteriores, sus condiciones homogéneas proporcionan una referencia inicial para la verificación.

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

**Aporte a la tesis.** La investigación usa esta instalación como caso base con suelo homogéneo y geometría simple. Primero se verifica la conducción radial y enterrada; luego se incorporan \textit{backfill}, estratos o zonas secas para evaluar la respuesta frente a una complejidad creciente.

## Quan et al. (2019) - formaciones plana y trébol con suelo o FTB

**Referencia.** Quan, L., Fu, C., Si, W., & Yang, J. (2019). Numerical study of heat transfer in underground power cable system. *Energy Procedia, 158*, 5317–5322. https://doi.org/10.1016/j.egypro.2019.01.636

Quan et al. resuelven en COMSOL un modelo estacionario 2D de tres cables enterrados y comparan la geometría, la profundidad y el relleno térmico. El dominio mide 20 m × 10 m, la superficie se mantiene a 20 °C y las demás fronteras son adiabáticas.

| Variable | Valor o escenario |
| -------- | ----------------- |
| Configuración | Tres cables en formación plana o trébol. |
| Profundidad | 1.0 a 2.0 m. |
| Separación entre ejes, formación plana | 0.2 a 0.6 m. |
| Radios exteriores conductor/aislamiento/pantalla/cubierta | 24.8, 55.3, 61.7 y 66.8 mm. |
| Corriente del análisis térmico | 1145 A. |
| Criterios térmicos | 65 °C como temperatura óptima y 90 °C como límite. |
| Medio próximo | Suelo nativo o relleno térmico fluidizado (FTB), ambos con k dependiente de la temperatura. |

| Formación | Ampacidad sin FTB, A | Ampacidad con FTB, A | Incremento con FTB |
| --------- | --------------------: | -------------------: | -----------------: |
| Plana | 1396.4 | 1599.3 | 14.5% |
| Trébol | 1248.5 | 1500.9 | 20.2% |

**Aporte a la tesis.** El caso permite verificar la interacción entre la configuración, la profundidad y la heterogeneidad próxima al cable. Además, los valores publicados de ampacidad permiten comparar el efecto del FTB en las formaciones plana y trébol.

## Oladunjoye et al. (2012) - propiedades térmicas medidas in situ

**Referencia.** Oladunjoye, M. A., Sanuade, O. A., & Olaojo, A. A. (2012). In situ determination of thermal resistivity of soil: Case study of Olorunsogo Power Plant, Southwestern Nigeria. *ISRN Civil Engineering, 2012*, 591450. https://doi.org/10.5402/2012/591450

Oladunjoye et al. caracterizan el entorno térmico de la central de turbinas de gas Olorunsogo mediante diez calicatas de aproximadamente 1.5 m y mediciones con un equipo KD2 Pro. Aunque el estudio no simula un cable específico, proporciona propiedades del suelo para escenarios heterogéneos.

| Variable medida | Rango | Promedio reportado |
| --------------- | ----- | ------------------ |
| Resistividad térmica | 0.3407 a 0.7188 K m/W | 0.5643 K m/W |
| Conductividad térmica | 1.391 a 2.935 W/(m K) | — |
| Temperatura del suelo | 28.72 a 35.39 °C | 32.11 °C |
| Humedad óptima | 13.00 a 16.20% | — |
| Densidad seca máxima | 1725.05 a 1930.00 kg/m³ | 1855.61 kg/m³ |
| Porosidad | 39.74 a 45.64% | — |

**Aporte a la tesis.** Las mediciones de campo permiten parametrizar $k(x,y)$ y la temperatura del terreno con variabilidad espacial observada. Este caso complementa los escenarios construidos con propiedades bibliográficas o de laboratorio.

## Möbius et al. (2025) - ampacidad con datos climáticos de Hamburgo

**Referencia.** Möbius, P., Michael, L.-H., Plenz, M., Schräder, J., & Gockel, C. (2025). Energy cable ampacity: Impact of seasonal and climate-related changes. *Renewable and Sustainable Energy Reviews, 212*, 115348. https://doi.org/10.1016/j.rser.2025.115348

El caso aplica IEC 60287 a tres cables unipolares N2XS2Y 1×35/16 de 6/10 kV, instalados en trébol a 0.5 m de profundidad. Usa 32 años de registros de 30 estaciones del área metropolitana de Hamburgo y un suelo de referencia con 30% de arena, 55% de limo, 15% de arcilla y densidad seca de 1.2 t/m³.

| Condición | Temperatura del suelo | Resistividad térmica | Ampacidad |
| --------- | ---------------------: | -------------------: | --------: |
| Fría y húmeda | 0 °C | 1.0 K m/W | 245 A |
| Caliente y seca | 20 °C | 2.5 K m/W | 148 A |
| Valor declarado por el fabricante | — | — | 187 A |

El suelo de referencia tiene punto de marchitez de 12%, capacidad de campo de 24% y saturación máxima de 37%. Para el análisis de sensibilidad, los autores variaron la temperatura entre 0 y 45 °C y la resistividad entre 0.1 y 4.0 K m/W.

**Aporte a la tesis.** La serie histórica define una instancia operacional con variación estacional del suelo. La diferencia de 97 A entre las condiciones extremas permite evaluar el efecto de representar la temperatura y la resistividad mediante valores variables en lugar de constantes.

## Casos normativos y de verificación de IEC y CIGRE

Las fuentes normativas cumplen funciones complementarias dentro del conjunto de verificación. IEC 60287-1-1 aporta las ecuaciones de ampacidad estacionaria, pérdidas y parámetros térmicos; CIGRE TB 880 desarrolla casos para comprobar su aplicación. En cambio, IEC 60853-3 incorpora un cálculo cíclico completo en los anexos A y B.

| Fuente | Función en la verificación | Uso en el conjunto de casos |
| ------ | ------------------------------ | -------------------------- |
| IEC 60287-1-1 | Aporta formulación y parámetros para el cálculo estacionario. | Mantener como línea base normativa, no como instancia autónoma. |
| IEC 60853-3, anexos A y B | Sí; cálculo cíclico con secado parcial y datos completos. | Incorporar como instancia normativa transitoria. |
| CIGRE TB 880 | Sí; casos para verificar herramientas basadas en IEC. | Incorporar el caso introductorio 0 y sus variantes. |
| CIGRE TB 963 | Sí; ejemplo analítico y casos FEM. | Incorporar el anillo analítico y el caso 1 en ductos. |

## CIGRE TB 880 (2022) - caso introductorio 0 de cable 132 kV

**Referencia.** CIGRE Working Group B1.56 (2022). *Power cable rating examples for calculation tool verification*. Technical Brochure 880, caso introductorio 0, pp. 61--125.

El caso base representa tres cables unipolares de 132 kV y 630 mm² Cu, con aislamiento XLPE, pantalla laminada de aluminio y cubierta PE. Los cables, de 75.5 mm de diámetro exterior, están en trébol tocándose, directamente enterrados a 1.0 m hasta el centro del circuito. Se fija una temperatura del terreno de 20 °C, resistividad térmica de 1.0 K m/W y límite del conductor de 90 °C.

| Variante del caso base | Tratamiento de pérdidas de pantalla | Ampacidad convergida |
| ---------------------- | ----------------------------------- | -------------------: |
| Pantallas unidas en ambos extremos, resultado IEC | Corrientes circulantes; pérdidas por corrientes parásitas omitidas | 821.78 A |
| Pantallas unidas en un punto | Sin corrientes circulantes; incluye pérdidas por corrientes parásitas | 886.18 A |
| Pantallas unidas en ambos extremos, recomendación CIGRE | Corrientes circulantes y parásitas, con factor de reducción | 803.16 A |

La familia completa extiende el mismo cable a instalación directa, ductos HDPE, ductos PVC en concreto, aire con radiación solar y canal sin relleno. Esta secuencia permite reproducir primero las resistencias, las pérdidas y la ampacidad del circuito IEC, y luego evaluar los cambios de geometría, frontera y tratamiento de pérdidas.

**Aporte a la tesis.** El caso funciona como referencia normativa de regresión. Aunque no proporciona un campo 2D, permite comprobar la relación corriente--pérdidas--temperatura y comparar la ampacidad estimada por PINN o FEM con valores convergidos.

## CIGRE TB 963 (2025) - caso 1 FEM 2D con cables en ductos

**Referencia.** CIGRE Working Group B1.87 (2025). *Finite element analysis for cable rating calculations*. Technical Brochure 963, caso 1, pp. 103--121.

La configuración contiene tres cables de 132 kV y 630 mm² Cu en ductos HDPE llenos de aire, dispuestos horizontalmente dentro de un backfill y rodeados por suelo y una capa superficial de asfalto. El caso comparte el diseño de cable de TB 880, pero explicita el dominio y los materiales necesarios para una referencia FEM 2D.

| Parámetro | Valor |
| --------- | ----- |
| Diámetros exteriores conductor / pantallas semiconductoras / XLPE / pantalla metálica / cubierta | 30.30 / 33.30 / 64.30 / 66.90 / 68.50 / 75.50 mm |
| Profundidad al centro de los ductos | 1.0 m |
| Ducto HDPE | 140 mm exterior; 10.3 mm de espesor |
| Banco de ductos y separación | 1.0 m × 0.5 m; 0.3 m entre ejes |
| Capa de asfalto | 0.1 m |
| Temperaturas | Terreno 20 °C; conductor máximo 90 °C |
| Resistividades térmicas | Suelo 1.0 K m/W; backfill 0.5 K m/W |
| Dominio seleccionado tras sensibilidad | 40 m × 20 m |

| Variante | Descripción | Ampacidad |
| -------- | ----------- | ---------: |
| 0 | Modelo base; pantallas con unión multipunto | 615 A |
| 1 | Transferencia de calor del aire en ductos modelada explícitamente | 630 A |
| 2 | Pantallas con unión en un punto | 973 A |
| 3 | Ductos rellenos con bentonita | 700 A |
| 4 | Dominios infinitos en los bordes | 615 A |
| 5 | Fuentes térmicas definidas según IEC | 617 A |
| 6 | Secado del suelo | 587 A |
| 7 | Pantalla de alambres de cobre | 654 A |
| 8 | Carga variable transitoria; máximo de 85 °C | 615 A |

En la variante base, a 615 A las temperaturas de conductor de las tres fases son 90.0, 83.6 y 82.0 °C; las del aire de los ductos son 71.2, 68.4 y 66.1 °C. El barrido del dominio produce 615.84 A con 200 m × 100 m y 609.85 A con 10 m × 5 m, lo que permite evaluar sensibilidad a la frontera artificial.

**Aporte a la tesis.** El caso reúne los componentes físicos del artefacto propuesto: geometría 2D, varios materiales, fuentes dependientes de corriente, interfaces y resultados por fase. Por ello, permite comprobar el campo térmico, el punto caliente, el balance de energía, la sensibilidad al dominio y la ampacidad.

## IEC 60853-3 (2002) - ejemplo cíclico con secado parcial

**Referencia.** International Electrotechnical Commission (2002). *IEC 60853-3: Calculation of the cyclic and emergency current rating of cables--Part 3: Cyclic rating factor for cables of all voltages, with partial drying of the soil*, anexos A y B, pp. 21--29.

El ejemplo considera un cable tripolar de 132 kV, conductor de cobre de 400 mm², aislamiento de papel impregnado en aceite, cubierta de plomo, refuerzo y cubierta PVC. Como el alcance inicial es estacionario, esta instancia se reserva para evaluar posteriormente el término temporal y los efectos del secado.

| Grupo | Datos principales |
| ----- | ------------------ |
| Cable | Diámetro exterior 0.109 m; conductor 22.85 mm; aislamiento 9.5 mm; resistencia AC 61.5 µΩ/m |
| Operación | Ambiente 20 °C; conductor 85 °C; incremento admisible 65 K; pérdidas dieléctricas 6.03 W/m por cable |
| Circuito térmico | $T_1=0.835$ K m/W; $T_3=0.09$ K m/W; $T_A=0.278$ K m/W; $T_B=0.1021$ K m/W |
| Suelo | Profundidad 1.0 m; $\rho_w=1.0$ y $\rho_d=2.5$ K m/W; incremento crítico 30 K; difusividad húmeda $5\times10^{-7}$ m²/s |
| Ciclo | Factor de pérdidas de 24 h: 0.504; ordenadas de las seis horas previas: 0.992, 0.728, 0.640, 0.596, 0.593 y 0.796 |
| Resultado | Factor cíclico $M=1.218$; corregido por secado $M_1=1.27$; ampacidad estacionaria con secado 520 A; pico 640 A |

**Aporte a la tesis.** La evaluación inicial mantiene este caso separado de las referencias estacionarias. Su clasificación como caso normativo transitorio conserva los parámetros necesarios para una prueba posterior de la extensión temporal del artefacto.

## Pan et al. (2025) - placa 2D para reconstrucción térmica con PINN

**Referencia.** Pan, Y., Zhang, K., Ma, N., & Zhang, J. (2025). Research on the reconstruction of the temperature field in two-dimensional steady-state thermal conductivity based on physics-informed neural networks. *Eng, 6*, 99. https://doi.org/10.3390/eng6050099

Pan et al. estudian una placa rectangular $x\in[0,1]$ m, $y\in[0,0.5]$ m, sin fuente interna y con conductividad constante, por lo que satisface $\nabla^2T=0$. La temperatura es de 25 °C en el borde inferior y de 0 °C en los otros tres lados. En el problema inverso, la condición inferior se reconstruye a partir de datos interiores.

| Elemento | Valor |
| -------- | ----- |
| Referencia numérica | ANSYS/Fluent; 10 153 elementos y 10 368 nodos |
| Datos etiquetados | 20 puntos interiores, aproximadamente 0.2% de los nodos |
| Entrenamiento reportado | 100 000 iteraciones; diez repeticiones y selección del mejor resultado |
| Resultado base | Error relativo promedio aproximado de 7.8% |
| Generalización | Para bordes entre 10 y 40 °C: error relativo promedio menor de 10% y error absoluto menor de 1 °C |
| Ponderación adaptativa | Error absoluto máximo reducido a 0.6 °C y error relativo promedio reducido en unos 2 puntos porcentuales |

**Aporte a la tesis.** Esta instancia 2D permite comprobar la red, el muestreo y las métricas en una geometría simple que no depende de un cable. Por tanto, precede a los casos con fuentes cilíndricas, materiales discontinuos y cálculo de ampacidad.

## CIGRE TB 963 (2025) - caso analítico anular

En las pp. 52--55, CIGRE TB 963 define un anillo con radio interior 0.06 m y exterior 20 m. El material tiene resistividad térmica $\rho=0.8$ K m/W; el borde exterior está a 20 °C y el interior recibe un flujo de 106.1 W/m², equivalente a 40 W/m. La solución exacta es

$$
T(r)=20-\frac{40(0.8)}{2\pi}\ln\left(\frac{r}{20}\right),
\qquad r=\sqrt{x^2+y^2}.
$$

La temperatura exacta en el radio interior es aproximadamente 49.59 °C. Aunque la solución es radial, el anillo puede representarse como un dominio 2D en coordenadas cartesianas. Así, la solución proporciona una referencia exacta en todo el dominio y no solo en el punto de temperatura máxima.

**Aporte a la tesis.** La investigación adopta este anillo como caso inicial para verificar las derivadas automáticas, las condiciones de flujo y temperatura, el error de campo y la convergencia. Una vez superadas estas comprobaciones, la evaluación avanza hacia la geometría multicapa del cable.

## Hahn y Özişik (2012) - rectángulo 2D con frontera convectiva

**Referencia.** Hahn, D. W., & Özişik, M. N. (2012). *Heat Conduction* (3.ª ed.), ejemplo 3-3, pp. 92--95. Wiley. https://doi.org/10.1002/9781118411285

El ejemplo rectangular de Hahn y Özişik incorpora una condición mixta que complementa las fronteras prescritas de la placa de Pan. En $0<x<L$, $0<y<W$ se resuelve $\nabla^2T=0$, con $T=0$ en $x=0$, $x=L$ y $y=0$, y

$$
-k\frac{\partial T}{\partial y}=h[T(x,W)-T_\infty]
$$

en el borde superior. La solución exacta es

$$
T(x,y)=\sum_{n=1}^{\infty}C_n\sin(\lambda_n x)\sinh(\lambda_n y),
\qquad \lambda_n=\frac{n\pi}{L},
$$

$$
C_n=\frac{hT_\infty\int_0^L\sin(\lambda_nx)\,dx}
{[k\lambda_n\cosh(\lambda_nW)+h\sinh(\lambda_nW)]\int_0^L\sin^2(\lambda_nx)\,dx}.
$$

El caso inicial usa $L=1$ m, $W=0.5$ m, $k=1$ W/(m K), $h=10$ W/(m² K) y $T_\infty=40$ °C. El número de términos de la serie se registra para mantener el error de truncamiento por debajo del criterio exigido al artefacto PINN.

**Aporte a la tesis.** El caso verifica una condición de Robin que representa la convección equivalente en la superficie del terreno o en un cable expuesto. Además, permite separar el error del artefacto PINN del error de truncamiento de la serie.

## Fuentes generales de transferencia de calor

Las fuentes generales de transferencia de calor amplían la verificación más allá de las instalaciones de cables. Carslaw y Jaeger, Hahn y Özişik, y Kakac et al. desarrollan soluciones de conducción en coordenadas rectangulares y cilíndricas, mientras que Patankar aporta una referencia para la discretización conservativa y el balance de flujo.

Dentro del alcance 2D estacionario, el ejemplo rectangular de Hahn y Özişik se incorpora como instancia explícita porque combina una frontera convectiva con una solución analítica en serie. Los demás textos conforman un catálogo para futuras pruebas transitorias, cilíndricas o tridimensionales.

Xing et al. (2023) presentan tres problemas anisotrópicos con soluciones analíticas en dominios 3D. Estos problemas no se incorporan directamente porque la unidad de análisis es 2D; sin embargo, su estrategia de definir una solución exacta y deducir la ecuación y sus contornos orienta la suite manufacturada siguiente.

## Suite propuesta de soluciones manufacturadas 2D

Las dos instancias manufacturadas se definen a partir de la ecuación de conducción y del criterio de comparar PINN y FEM con soluciones exactas. Corresponden a una elaboración propia y se mantienen separadas de los resultados publicados. En cada caso, la fuente $Q$ se obtiene al sustituir una temperatura prescrita en

$$
-\nabla\cdot\left(k(x,y)\nabla T\right)=Q(x,y).
$$

### MMS-2D-01: conductividad espacialmente variable

En el cuadrado unitario se define

$$
T=T_0+A\sin(\pi x)\sin(\pi y),\qquad
k=k_0(1+\beta x),
$$

con $T=T_0$ en todo el borde. La fuente compatible es

$$
Q=2k_0(1+\beta x)A\pi^2\sin(\pi x)\sin(\pi y)
-k_0\beta A\pi\cos(\pi x)\sin(\pi y).
$$

El caso comprueba simultáneamente la implementación de $k(x,y)$, su gradiente, la fuente y las fronteras de Dirichlet. Los valores iniciales son $T_0=20$ °C, $A=30$ K, $k_0=1$ W/(m K) y $\beta=1$.

### MMS-2D-02: interfaz de dos materiales

El segundo caso define una interfaz vertical $x=a$, con $k_1$ a la izquierda y $k_2$ a la derecha. La temperatura se expresa como

$$
T=T_0+\phi(x)\sin(\pi y),
$$

$$
\phi(x)=
\begin{cases}
A x/k_1, & x\le a,\\
A\left[a/k_1+(x-a)/k_2\right], & x>a.
\end{cases}
$$

En cada material, $Q_i=k_i\pi^2\phi(x)\sin(\pi y)$. La construcción conserva la continuidad de $T$ y de $k\,\partial T/\partial x$ en la interfaz. Los valores $a=0.5$, $k_1=0.5$, $k_2=2.0$ W/(m K), $A=10$ W/m y $T_0=20$ °C definen el contraste inicial de conductividad.

| Prueba de aceptación común | Criterio que debe registrarse |
| -------------------------- | ----------------------------- |
| Exactitud de campo | MAE, RMSE, NRMSE y norma relativa $L_2$ frente a $T$ exacta |
| Punto caliente | Error absoluto de $T_{max}$ y distancia de su localización |
| Consistencia física | Norma del residuo de PDE y balance entre fuente y flujo de borde |
| Interfaz | Saltos de temperatura y flujo normal para MMS-2D-02 |
| Robustez | Distribución de errores y convergencia en repeticiones con semillas declaradas |

**Aporte a la tesis.** La suite evalúa dos condiciones que el anillo homogéneo no representa: una conductividad variable suave y una interfaz discontinua. Sus datos se registran como elaboración propia y se mantienen separados de los casos publicados.

## Variables consolidadas de los casos

Los datos extraídos se organizan como una biblioteca de casos reproducibles del objeto de estudio y de verificación del artefacto. La tabla siguiente relaciona cada grupo de variables con su fuente y con el uso previsto en los escenarios de simulación.

| Grupo de datos          | Variables extraídas                                                                                          | Fuentes                               | Uso recomendado                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------- |
| Cable multicapa         | Diámetros, espesores, materiales, k de conductor/aislamiento/pantallas/cubierta.                             | Kim, Al-Dulaimi, Atoccsa, Ocłoń, Aras, Quan | Definir geometría interna o equivalente térmico del cable.                |
| Instalación             | Profundidad H/L, separación S/l/s, formación plana o trébol, ductos, arreglo de 1/3/6 cables.                | Kim, Al-Dulaimi, Atoccsa, Ocłoń, Aras, Quan, Möbius | Generar dominios 2D reproducibles y variar proximidad de fuentes.         |
| Suelo nativo            | k, resistividad térmica, humedad, clasificación, temperatura y densidad del suelo.                           | Kim, Khumalo, Aras, Ocłoń, Oladunjoye, Möbius | Modelar condición homogénea, estratificada, seca o medida en campo.       |
| Bedding/backfill        | PAC, NC, arena, SCMB, FTB, SBM, HDPE, conductividad y dimensiones.                                           | Kim, Al-Dulaimi, Atoccsa, Ocłoń, Quan | Construir heterogeneidades de alta/baja k y escenarios de mejora térmica. |
| Condiciones de frontera | Temperatura superficial/ambiente, convección, viento, temperatura inferior constante, fronteras adiabáticas. | Kim, Al-Dulaimi, Ocłoń, Aras, Quan, Möbius | Definir problemas directos comparables con FEM/PINN.                      |
| Resultados térmicos     | Tmax, diferencias por posición, campo de temperatura, criterio 65/90 °C.                                     | Kim, Al-Dulaimi, Ocłoń, Aras, Quan    | Validar salida de campo y localización del punto caliente.                |
| Resultados operativos   | Ampacidad Imax, reducción por secado, incremento con backfill y variación estacional.                        | Khumalo, Atoccsa, Aras, Kim, Quan, Möbius | Evaluar estimación de Imax y sensibilidad a k y temperatura del suelo.    |
| Verificación normativa  | Resistencias, pérdidas, temperatura, factor cíclico y ampacidad convergida.                                    | CIGRE TB 880; IEC 60853-3              | Probar la cadena corriente--pérdidas--temperatura y una futura extensión transitoria. |
| Referencia FEM 2D       | Dominio, malla, materiales, fronteras, temperaturas por fase y sensibilidad.                                  | CIGRE TB 963, caso 1; Pan et al.       | Comparar campo, punto caliente, dominio artificial y reconstrucción inversa. |
| Referencia analítica    | Solución exacta de campo en anillo o rectángulo y condición de Robin.                                         | CIGRE TB 963, ejemplo 4.3; Hahn y Özişik | Medir error de campo sin incertidumbre de una malla FEM.                  |
| Verificación manufacturada | $T(x,y)$ exacta, $k(x,y)$, fuente compatible, fronteras e interfaces.                                      | Suite MMS-2D propuesta                 | Probar conductividad variable, derivadas automáticas y continuidad multimaterial. |

| Nivel de uso                     | Caso recomendado | Razón                                                                         |
| -------------------------------- | ---------------- | ----------------------------------------------------------------------------- |
| Verificación simple              | Aras 2005        | Suelo homogéneo, dominio FEM claro y comparación IEC/FEM.                     |
| Heterogeneidad por humedad       | Khumalo 2025     | Datos numéricos directos de resistividad-humedad-ampacidad.                   |
| Backfill de alta conductividad   | Kim 2025         | Compara arena natural, NC, PAC y CLSM con temperaturas máximas.               |
| Escenarios paramétricos          | Al-Dulaimi 2024  | Incluye profundidad, separación, configuración, suelos multicapa y backfills. |
| Optimización de ampacidad        | Atoccsa 2024     | Relaciona dimensiones de backfill con Imax y costo.                           |
| Backfill localizado y proximidad | Ocłoń 2015       | Dominio 2D con FTB/SBM/HDPE y fuente central crítica.                         |
| Geometría y FTB con k(T)          | Quan 2019        | Compara formaciones plana/trébol y cuantifica el aumento de ampacidad.        |
| Propiedades in situ               | Oladunjoye 2012  | Diez puntos de campo con k, resistividad, temperatura y propiedades físicas. |
| Variación estacional              | Möbius 2025      | Relaciona clima, estado del suelo y ampacidad con 32 años de observaciones.   |
| Regresión normativa IEC           | CIGRE TB 880, caso 0 | Presenta parámetros y ampacidades convergidas para variantes controladas.  |
| FEM 2D multimaterial publicado    | CIGRE TB 963, caso 1 | Reúne la geometría multimaterial 2D y las salidas previstas del artefacto. |
| Extensión transitoria             | IEC 60853-3, anexos A/B | Conserva cable, ciclo, suelo húmedo/seco y corriente pico reproducible. |
| PINN 2D antes del cable           | Pan 2025         | Aísla aprendizaje, muestreo e inversión de fronteras en una placa simple.    |
| Solución exacta con flujo         | CIGRE TB 963, anillo | Entrega $T(x,y)$ exacta para todo el dominio.                              |
| Solución exacta con convección    | Hahn y Özişik, ejemplo 3-3 | Verifica una condición de Robin mediante una serie analítica.        |
| Pruebas unitarias multimateriales | MMS-2D-01/02     | Aíslan $k(x,y)$ variable y continuidad de temperatura/flujo en interfaces.   |

## Fuentes usadas sin tabla de caso directo

Las demás fuentes brindan soporte metodológico o teórico. Una fuente no se incorpora como instancia cuando carece de un dominio, una ecuación, condiciones auxiliares o una referencia cuantitativa reproducible dentro del alcance 2D.

| Tipo                                 | Ejemplos                                                                | Motivo                                                                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Base analítica / normativa histórica | Neher y McGrath (1957)                                                  | Aporta formulación de cálculo de temperatura y capacidad, no un caso moderno de instalación con datos completos de suelo/backfill. |
| Revisiones de cable rating           | Enescu et al. (2020, 2021); systematic mapping de DTR; overview térmico | Sirven para justificar variables y brechas, pero consolidan literatura en vez de reportar un único dataset de instalación.         |
| PINN y SciML                         | Raissi et al.; Xing et al.; revisiones de conducción directa/inversa    | Aportan método o casos 3D; no todos corresponden a la unidad de análisis 2D.                                                       |
| Transferencia de calor general      | Carslaw y Jaeger; Kakac et al.; Patankar                                | Proporcionan familias de soluciones y métodos; se incorporaron solo los casos con ficha reproducible inmediata.                    |
| Ciencia del diseño                   | Peffers; Hevner; Gregor                                                 | Soporte metodológico de DSR, sin datos físicos de cable.                                                                           |

## Notas de trazabilidad

- Los valores se extraen de tablas, figuras y texto técnico de los PDF almacenados en la biblioteca local de Zotero.

- Cuando un artículo presenta varios resultados equivalentes, se priorizan los que describen el cable, la instalación, el suelo o relleno, la frontera térmica y la respuesta $T_{\max}$/$I_{\max}$.

- Las tablas organizan los datos necesarios para preparar los casos de ejemplo. En una publicación o capítulo posterior, cada valor numérico se vincula con su página, tabla o figura de origen.

- Las soluciones manufacturadas se identifican como elaboración propia y se separan de los resultados publicados. Sus fuentes y fronteras se generan en forma simbólica y se verifican antes del entrenamiento.

