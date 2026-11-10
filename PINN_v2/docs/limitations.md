# Limitaciones

Lo que este marco no hace o todavía no está confirmado.

## Criterios

- Las compuertas de aceptación (NRMSE 5 %, Tmax 5 %, región 5 %, balance 2 %, salto de T 0.1 K,
  flujo en interfaces 1 %) son **propuestas**. Falta que la persona responsable las confirme.
- Las compuertas de convergencia FEM (0.5 % de malla, 2 % de dominio) también son propuestas.

## Modelos

- CUSP, DCSNN, FBPINN, RBA y SA son variantes **adaptadas** de la idea de cada artículo, no
  reproducciones. Los DOI y arXiv citados en el código están marcados "por verificar".
- I-PINN, AdaI-PINN y VPINN no están implementados.
- La pérdida held-out de la búsqueda se compara bien dentro de una familia; entre familias distintas
  puede no ser comparable.

## Física y datos

- Solo régimen permanente en 2D.
- El caso de Kim solo tiene verano; no hay caso de invierno.
- El modo `dc_temperature` de FEM es un acople electrotérmico en corriente continua; el PINN usa la fuente fija.
- Un dato faltante rechaza el caso; no se inventa.

## Annulus fuera de la comparación principal

El caso `annulus` se conserva como diagnóstico, pero se excluye del agregado y del ranking de la
comparación principal de PINN. No se concluye que una PINN sea incapaz de resolver anillos; los
experimentos disponibles muestran que este caso exige una formulación y validación específicas:

- El radio interior es `0.06 m` y el exterior `20 m` (razón `333`). La solución exacta varía como
  $T(r)=T_0+P/(2\pi k)\ln(r_o/r)$, por lo que el gradiente radial se concentra cerca del borde interior.
- La frontera interior prescribe flujo Neumann, no temperatura. La traza de temperatura interior
  debe inferirse globalmente a partir de la PDE, la potencia y la frontera exterior Dirichlet.
- El hueco central está excluido del dominio, pero el muestreo uniforme por área original dejó
  `0/512` puntos de colocación con `r < 0.5 m` (radio mínimo `1.56391 m`). Así, imponer el flujo
  no garantiza que la red resuelva el collar estrecho que controla el máximo y el balance.
- El lifting Hermite lineal en `r` resulta incompatible con el perfil logarítmico: aun usando
  temperaturas y pendientes exactas en los dos radios, su error máximo radial fue `234.10 K`.
  El lifting y muestreo en log-radio redujeron mucho el error, pero no alcanzaron las compuertas.
  Incluso `CUSP_EXACT` con `20 000` pasos Adam y `200` llamadas L-BFGS terminó con NRMSE FEM
  `10.42 %`, error Tmax `12.49 K` y balance global `57.31 %`.

Estos resultados motivan excluir `annulus` de la comparación general: mezcla geometría con agujero,
gran razón de radios, condición Neumann con traza térmica desconocida y una solución con fuerte
variación radial. Su resolución requiere estudiar por separado el ansatz radial, la cobertura de
colocación y la identificabilidad de la temperatura interior. La referencia FEM concuerda con la
solución analítica (RMSE `0.000183 K`), de modo que la exclusión se debe a la dificultad del PINN,
no a una referencia FEM defectuosa. Ver [resultados diagnósticos](results.md#alcance-de-la-comparacion-principal).

## Entorno

- FEniCSx solo corre en WSL2 Ubuntu; sin WSL las pruebas con marca `fem` se omiten.
- El entrenamiento es en CPU y float64, y puede tardar horas en casos con muchas regiones.
- El nivel 3 de malla de Kim agota la memoria de WSL.