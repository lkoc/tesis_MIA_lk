# Protocolo preinscrito de la ronda E (borrador, no ejecutado)

**Estado:** propuesta para aprobación del asesor. **No se ha ejecutado ningún entrenamiento de esta ronda.** Este documento fija, antes de observar cualquier resultado de E, los casos, los cambios permitidos, las semillas, las reglas de decisión y el reporte. Solo tiene efecto si se aprueba y se firma antes de ejecutar.

Origen: recomendación R-12 de la auditoría general del 29-sep-2026, aprobada por los tesistas. Complementa, sin editarlos, los protocolos [experimental A–D](PROTOCOLO_EXPERIMENTAL_EXPLICITO.md), [calibración C2](PROTOCOLO_CALIBRACION_INTERFACES.md) y [ampacidad explícita](PROTOCOLO_AMPACIDAD_EXPLICITA.md).

## 1. Motivo

La confirmación D aceptó la receta congelada en 7 de 21 ejecuciones. Con el diagnóstico de la restricción dominante (capítulo III, Sección «Diagnóstico de los rechazos», Tabla `tab:explicit-D-diagnosis`):

| Problema | Aceptadas | Restricción dominante (razón a la tolerancia, peor semilla) |
|---|---|---|
| `xlpe_discrete_layers` | 0/3 | Salto de flujo 1,88 y salto de T 1,70 (el campo y Tmax cumplen: 0,92) |
| `xlpe_backfill` | 0/3 | Error de Tmax y de campo 1,99; energía 1,46 |
| `xlpe_dry_near` | 1/3 | Error de Tmax y de campo 2,16; semilla 83 cumplió (0,76) |
| `aras_flat` | 0/3 | Campo 16,21 |
| `kim_sand` | 0/3 | Campo 6,29; Tmax 5,75; saltos de T y de flujo 2,51 y 2,02 |

Los tres primeros están cerca del umbral y su limitación es específica (continuidad en la interfaz; error de Tmax moderado). Aras plano y Kim en arena superan varias veces la tolerancia de campo y no se incluyen (ver §2).

## 2. Alcance

- **Casos incluidos:** `xlpe_discrete_layers`, `xlpe_backfill`, `xlpe_dry_near`.
- **Casos excluidos:** `aras_flat` y `kim_sand`. No se reabren sin un cambio de arquitectura o de presupuesto que se justifique por escrito, con su propia etapa de desarrollo.
- **Casos ya aceptados** (`xlpe_single`, `xlpe_dry_far`): no se reentrenan ni se sustituye su receta; la ronda E no modifica su estado.

## 3. Lo que no cambia

- La física, las ecuaciones, las normalizaciones y los contratos de caso (misma huella SHA-256 de identidad física).
- Las referencias FEM refinadas y sus puertas de convergencia.
- Los seis criterios de aceptación PINN y sus umbrales: NRMSE y error de Tmax ≤ 5 %, RMSE por material ≤ 5 %, balances ≤ 2 %, salto máximo de T ≤ 0,1 K, salto RMS de flujo ≤ 1 %.
- Las temperaturas FEM no intervienen en la pérdida.

## 4. Cambios permitidos a la receta (uno por limitación diagnosticada)

Cada cambio responde a la restricción dominante observada en D y se declara aquí, antes de E. No se admiten otros cambios.

| Caso | Limitación diagnosticada | Cambio permitido |
|---|---|---|
| `xlpe_discrete_layers` | Continuidad en la interfaz (salto de flujo y de T) | Duplicar las colocaciones por interfaz respecto de la receta D (de 192 a 384) y usar la política de bandas de interfaz ya implementada y evaluada en B |
| `xlpe_backfill`, `xlpe_dry_near` | Error de Tmax cercano al umbral | Presupuesto de colocaciones 4× respecto del inicial de A, es decir, el doble del de la receta D, que usa 2× (512 puntos por región de suelo, 256 por capa, 192 por interfaz), con la política ya evaluada en B |

El peso de continuidad térmica ($w_T=1000$), la arquitectura (`subdomain`, 32 neuronas, 3 capas por material), la tasa ($5\times10^{-4}$), los pasos de Adam y L-BFGS y la mezcla geométrica se mantienen. Un cambio de arquitectura, de $w_T$ o de tasa exige reabrir A o C con su propio protocolo y queda fuera de E.

## 5. Diseño

1. **E1 (desarrollo).** Para cada caso, comparar la receta D con la receta E de la tabla anterior, con las inicializaciones de desarrollo 11 y 23 y los muestreos 101 y 103 (pareados). Son 3 casos × 2 recetas × 2 semillas = 12 entrenamientos; solo pueden reutilizarse los 2 de la receta D en estratos con semillas de desarrollo (etapa C2), porque relleno y zona seca cercana no se entrenaron con semillas de desarrollo. El resto se ejecuta.
2. **Congelación.** Se conserva la receta E solo para los casos donde E1 la muestre no peor en la peor violación de los criterios; en caso contrario se conserva la receta D. La decisión se registra en un archivo con huella antes de E2.
3. **E2 (confirmación).** Semillas nuevas, no usadas en A–D ni en la extensión de ampacidad: inicializaciones **131, 149 y 167**; muestreos **2011, 2013 y 2017** (deben verificarse como no usadas antes de firmar). Son 3 casos × 3 semillas = 9 entrenamientos.

## 6. Reglas de decisión

- Un caso queda **aceptado** solo si cumple todos los criterios en las tres semillas de E2. Con 1 o 2 de 3 queda **restringido**; con 0 de 3, **no aceptado**.
- No se selecciona una semilla favorable ni se descarta una corrida. Se publican mediana, rango y cantidad aceptada de todas las semillas.
- Una sola ronda. Si algún caso sigue sin aceptarse, no se abre otra ronda sin un protocolo nuevo aprobado.
- El dominio de aceptación del capítulo III (Tabla `tab:dominio-aceptacion`) se actualiza únicamente con los resultados de E2. Los resultados de D no se reemplazan.

## 7. Reporte

- Tabla de restricción dominante por corrida y por caso, con la misma definición que la Tabla `tab:explicit-D-diagnosis`.
- Tabla de costo computacional por corrida (parámetros, segundos), como la Tabla `tab:costo-computacional`.
- Todas las corridas de E1 y E2 en el expediente completo (Anexo «Expediente completo de ejecuciones») y en el manifiesto de la campaña, con huellas.
- Los resultados se incorporan a la tesis como una etapa nueva, sin renumerar los 101 entrenamientos A–D.

## 8. Costo estimado

A partir de los tiempos registrados en D (XLPE de un cable: 496–672 s por corrida), 9 corridas de E2 suman del orden de 1,5 h de tiempo de corrida con el presupuesto de D; con presupuestos mayores será superior. E1 añade hasta 10 corridas nuevas. Estos valores son una guía y no un compromiso.

## 9. Aprobación

| Rol | Nombre | Fecha | Firma |
|---|---|---|---|
| Asesor | Samuel Oporto | | |
| Tesista | Luis Enrique Koc Góngora | | |
| Tesista | Herbert Antonio Meléndez García | | |

Hasta que este cuadro esté completo, la ronda E se considera **propuesta y no autorizada**.
