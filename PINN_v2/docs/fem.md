# Referencia FEM (FEniCSx)

FEniCSx solo está instalado en **WSL2**. El lado Windows lo invoca por subproceso y recibe JSON y NPZ.
Ningún dato FEM entra en la pérdida de un PINN.

La compilación XPU de PyTorch en Windows solo acelera los PINN de Windows; no habilita la GPU de
FEniCSx dentro de WSL. El FEM actual usa PETSc con `preonly` + `LU` + `MUMPS`, una ruta de CPU.
Para aprovechar una GPU desde WSL harían falta acceso GPU visible dentro de Linux y un backend de
PETSc/FEniCSx compatible con esa GPU; la instalación actual no los configura.

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

## FEM del MMS heterogeneo para el reporte de tesis

Caso: `mms_interface`, dominio unitario 2D, conductividades 0.5 y 2 separadas en x=0.5,
fuente manufacturada no nula y contacto perfecto. El resultado FEM existente usa FEniCSx 0.10.0
y elementos P2. Hay niveles 0 y 1; el nivel 1 es el mas fino guardado de esta ejecucion y no
se etiqueta como referencia de malla convergida sin un estudio adicional.

| Caso | Metodo | Nivel | Celdas por lado | h m | Grados de libertad | Tmax en los puntos de evaluacion C |
|---|---|---:|---:|---:|---:|---:|
| mms_interface | FEM FEniCSx P2 | 1 | 32 | 0.03125 | 4225 | 32.483638 |

La Tmax procede del campo NPZ en los mismos 6000 puntos usados por las PINN, no de una
grilla de visualizacion. El desbalance energetico propio del FEM es 0.079783%; no es un error
de comparacion del FEM consigo mismo. Su tiempo registrado por nivel es 1.089022 s y no
representa todo el tiempo de lanzamiento, WSL o del trabajo del frontend.

El hash fisico y las coordenadas coinciden con los resultados M1_EXACT y DeepXDE_EXACT
comparados. Las NRMSE y errores PINN-FEM se recalcularon a partir de los campos guardados;
la fila FEM deja esos errores en blanco. Las auditorias originales contra la solucion
analitica permanecen separadas. Ningun campo FEM entro en las perdidas del entrenamiento.

### Seccion horizontal del annulus

Para comparar visualmente el anillo sin interpolar una nube dispersa, el FEM nivel1 se evaluo
directamente sobre1800 puntos de `y=0`, entre `ri*1.01` y `ro*0.995` en ambas ramas.
Se excluyen los limites exactos solo para que los puntos caigan de forma segura dentro de la malla
curva; la grafica analitica conserva la solucion exacta en ese mismo segmento muestreado.
RMSE FEM-analitica=`0.000183 K`; maximo absoluto=`0.000724 K`.

- [Grafica FEM y analitica](../results/phase7/p7_annulus_fem_line_20261021/fem/annulus_line_fem_vs_exact.png).
- [Muestras de la linea y error](../results/phase7/p7_annulus_fem_line_20261021/fem/annulus_line_fem_vs_exact.csv).
- [Metadatos FEM P2 nivel1](../results/phase7/p7_annulus_fem_line_20261021/fem/annulus_l1.json).
- [NPZ FEM con puntos de evaluacion de area y linea](../results/phase7/p7_annulus_fem_line_20261021/fem/annulus_l1.npz).

La [tabla de resultados](results.md) contiene la comparacion para la tesis y sus limites.
Procedencia: [metadatos del nivel 1](../front-end/runs/20261019_200551_fem_mms_interface/fem/mms_interface_l1.json),
[campo NPZ](../front-end/runs/20261019_200551_fem_mms_interface/fem/mms_interface_l1.npz) y
[resumen reproducible](../results/phase7/p7_deepxde_exact_mms_20261019/comparison_vs_fem_20261020.json).

## Convergencia y dominio

El estudio de la fase 6 (`pinn_v2/experiments/fem_ladder.py`) llama a un resultado FEM "referencia" solo si
las dos mallas más finas difieren menos de 0.5 % del aumento de temperatura (criterio **propuesto**).
También duplica el semiancho y la profundidad del dominio para medir cuánto depende la respuesta del
dominio reducido. Los resultados están en [Resultados](results.md).

## Si no tienes WSL

El PINN funciona sin FEM. Puedes entrenar, buscar hiperparámetros y evaluar contra los casos con solución
exacta (`mms_*`). Las pruebas con la marca `fem` se omiten sin WSL.