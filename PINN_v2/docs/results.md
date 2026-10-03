# Resultados de la Fase 6

Esta página resume lo que ya se midió. Los criterios de aceptación y las compuertas son **propuestos**,
no confirmados. La referencia FEM nunca se usó para entrenar.

## Referencia FEM (estudio p6_fem3)

La malla se considera convergente si el cambio entre los dos últimos niveles es 0.5 % del
calentamiento o menos. El estudio de dominio duplica el semiancho y la profundidad.

| caso | cambio de malla, % del calentamiento | cambio por dominio, % | sensible al dominio |
|---|---|---|---|
| xlpe_single | 0.056 | 0.31 | no |
| xlpe_backfill | 0.035 | 0.49 | no |
| xlpe_discrete_layers | 0.055 | 0.01 | no |
| aras_flat | 0.001 | 3.48 | sí |
| kim_v5_summer | 0.001 | 11.87 | sí |

Para `kim_v5_summer` se usaron los niveles 0 a 2 y la Tmax es 53.24 °C en el dominio ancho,
+4.04 K respecto al dominio base. El nivel 3 agotó la memoria de WSL; los estudios fallidos `p6_fem` y
`p6_fem2` se conservan.

## PINN M1 contra FEM (6000 Adam y 100 L-BFGS)

| caso | NRMSE, % | veredicto |
|---|---|---|
| xlpe_single | 0.558 | aceptado |
| xlpe_backfill | 2.14 | rechazado |
| xlpe_discrete_layers | 0.53 | rechazado |
| aras_flat | 0.981 | rechazado (peor región 4.83 %) |
| kim_v5_summer | 18.7 | rechazado |

## Entrenamientos largos de M1 (20000 Adam y 200 L-BFGS)

| caso | NRMSE, % | veredicto |
|---|---|---|
| xlpe_backfill | 2.43 | rechazado (fallan el balance y el salto de T) |
| xlpe_discrete_layers | 0.407 | rechazado (falla el salto de T en la interfaz) |
| aras_flat | 0.502 | aceptado (balance 0.096 %) |

Alargar el entrenamiento no mejoró `xlpe_backfill`.

### Kim, verano (M1, semilla 0)

El M1 no resuelve kim_v5_summer: NRMSE 13.4 %, error del ascenso de T 54.2 %, error de Tmax 18.4 K, balance 167 %, salto de T en interfaz 0.14 K, pérdida final 0.559 (12000 Adam, sin L-BFGS, 18892 s). Fallan cinco compuertas (Tmax, balance, salto de T, NRMSE, región); solo pasa el salto de flujo. La causa no está diagnosticada: no se atribuye a ninguna decisión de diseño hasta revisar el historial de pérdida y el balance por región.

## Cómo leerlo

- Un rechazo dice qué compuerta falló; el detalle está en el `report.md` de cada estudio.
- Los casos sensibles al dominio no tienen una referencia única: el error del PINN se mide contra
  el FEM del mismo dominio reducido.

### Kim, verano (M1, semilla 0): diagnostico

Hechos verificados sobre `results/phase6/p6_pinn_kim_long/kim_v5_summer/M1_s0`:

- La perdida total sigue bajando al final (6305, 2.57, 1.54, 0.861, 0.624, 0.559 en los pasos 0, 1200, 3000, 6000, 9000, 11999); no hubo L-BFGS.
- El residuo PDE normalizado del suelo es 0.099; el maximo en las capas de cable es 3e-4. El error se concentra en el suelo.
- El campo queda por debajo de la referencia casi en todas partes (99.9 % de los puntos), unos 17 a 22 K en los conductores. El punto caliente queda desplazado 0.40 m.
- Balance global 167 %. RMS en Dirichlet: izquierda 2.17 K, derecha 2.00 K, fondo 0.41 K.
- Los terminos de interfaz son pequenos, asi que el fallo no viene de las interfaces.

Hipotesis no verificada: entrenamiento insuficiente o capacidad limitada (red M1 de ancho 32 y profundidad 3, 256 puntos interiores). Falta una corrida exploratoria (mas pasos, L-BFGS, red mayor) en una carpeta nueva para confirmarlo o descartarlo.