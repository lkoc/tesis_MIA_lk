# Referencia FEM (FEniCSx)

FEniCSx solo está instalado en **WSL2**. El lado Windows lo invoca por subproceso y recibe JSON y NPZ.
Ningún dato FEM entra en la pérdida de un PINN.

## Entorno usado

| Elemento | Valor |
|---|---|
| Distribución WSL | `Ubuntu` (variable `PINN_V2_WSL_DISTRO`) |
| Python | `/home/lkoc/miniforge3/envs/fenicsx/bin/python` (variable `PINN_V2_FEM_PYTHON`) |
| dolfinx | 0.10.0 |
| gmsh | 4.15.2 |

## Correr FEM con el mismo caso

```bash
python -m pinn_v2 fem --case xlpe_single --levels 0 1 2 --output results/runs/fem_xlpe
python -m pinn_v2 fem --definition mi_caso.json --levels 0 1 2 --output results/runs/fem_mi_caso
```

`--levels` son niveles de refinamiento de la malla. `--mode` puede ser `fixed` (potencia prescrita, por defecto)
o `dc_temperature` (acoplamiento electrotérmico en corriente continua, con iteración sobre la conductividad eléctrica). El PINN solo usa la fuente prescrita.

Por cada nivel se escribe `fem_l<nivel>.json` (Tmax, grados de libertad y temperatura media por región) y `fem_l<nivel>.npz` (campo de temperatura en los puntos de evaluación).

## Evaluar un PINN contra el FEM

```bash
python -m pinn_v2 train --case xlpe_single --model M1 --reference results/runs/fem_xlpe/fem_l2.npz --output results/runs/m1_vs_fem
```

El NPZ debe traer `xy` y `T`. Si `xy` no coincide con los puntos de evaluación del problema, la corrida se
rechaza: comparar en puntos distintos no tendría sentido. La referencia solo se usa después de entrenar.

## Convergencia y dominio

El estudio de la fase 6 (`pinn_v2/experiments/fem_ladder.py`) llama a un resultado FEM "referencia" solo si
las dos mallas más finas difieren menos de 0.5 % del aumento de temperatura (criterio **propuesto**).
También duplica el semiancho y la profundidad del dominio para medir cuánto depende la respuesta del
dominio reducido. Los resultados están en [Resultados](results.md).

## Si no tienes WSL

El PINN funciona sin FEM. Puedes entrenar, buscar hiperparámetros y evaluar contra los casos con solución
exacta (`mms_*`). Las pruebas con la marca `fem` se omiten sin WSL.