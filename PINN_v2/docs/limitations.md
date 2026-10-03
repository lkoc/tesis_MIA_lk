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

## Entorno

- FEniCSx solo corre en WSL2 Ubuntu; sin WSL las pruebas con marca `fem` se omiten.
- El entrenamiento es en CPU y float64, y puede tardar horas en casos con muchas regiones.
- El nivel 3 de malla de Kim agota la memoria de WSL.