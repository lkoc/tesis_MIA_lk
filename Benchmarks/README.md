# Casos reproducibles de conducción térmica

Los archivos `cases/*.json` constituyen la especificación física común. PINN y
FEniCSx leen exactamente el mismo archivo, evalúan la misma conductividad y usan
los mismos puntos de comparación. Las opciones numéricas están separadas en
`configurations/*.json`. Los resultados conservan el caso completo y su SHA-256.

## Alcance físico

Se estudian quince problemas estacionarios: cuatro soluciones manufacturadas,
un anillo y diez escenarios de cables. En los escenarios de cables se resuelve
el suelo con agujeros circulares y flujo uniforme. Las capas internas se
recuperan con resistencias radiales en serie. Esta reducción permite comparar
dos solucionadores del **mismo problema**; no representa un cálculo 2D completo
de cada capa ni reproduce exactamente los artículos de Aras o Kim.

Las pérdidas son `power = current**2 * R20`, en W/m, fijadas a 20 °C. No incluyen
pérdidas dieléctricas, pantallas, proximidad ni efecto pelicular. `alpha` queda
registrado para extensiones electrotérmicas; no interviene en el campo base.
El suelo ocupa un rectángulo de 8 × 4 m con 20 °C en sus cuatro lados.

## Formato único, versión 1

El contrato de la versión 1 está definido por `cases.py:validate_case` y esta
documentación. JSON usa punto decimal, codificación UTF-8 y unidades SI. No se
admiten comentarios JSON; las explicaciones se guardan en `source` y `adaptation`.

| Campo | Contenido y unidad |
|---|---|
| `id`, `kind` | Identificador estable y familia de ecuaciones |
| `bounds` | `[xmin, xmax, ymin, ymax]`, m; superficie en y = 0 |
| `T0`, `scale` | Temperatura ambiente, °C; escala numérica, K |
| `cables` | Lista de centros `[x,y]`, m |
| `radius` | Radio exterior común, m |
| `layers` | Capas `[radio_interior, radio_exterior, k]`, m, m, W/(m K) |
| `current`, `R20`, `power` | A, Ω/m y W/m por cable |
| `k` | Conductividad del suelo de base, W/(m K) |
| `patch` | `[cx,cy,ancho,alto,k_relleno,epsilon]`; metros y W/(m K) |
| `bands` | Interfaces horizontales `[y,k_arriba,k_abajo]`, de arriba abajo |
| `band_smoothing` | Ancho de transición, m; predeterminado 0,1 m |
| `pair` | Identificador del control homogéneo, o `null` |
| `source`, `adaptation` | Procedencia y supuestos explícitos |

Los materiales localizados y estratos se interpolan mediante tangentes
hiperbólicas. `epsilon` es parte del modelo físico regularizado, no un ajuste
oculto del solucionador. El caso manufacturado de interfaz sí tiene un salto
exacto y usa dos subredes con continuidad de temperatura y flujo.

La versión actual admite un tipo de cable y una potencia común por escenario.
Para otro entorno basta copiar un JSON, cambiar `id`, centros, capas y
propiedades, y ejecutar ambos métodos. Los cables no pueden superponerse ni
tocar el borde exterior. Extender a varios tipos exige versionar el contrato
y actualizar ambos lectores; no debe ignorarse silenciosamente un campo.

## Ejecución

Desde la raíz del repositorio, en el entorno Windows usado en esta auditoría:

```powershell
python Benchmarks/pinn.py --cases mms_constant --seeds 11 23 37
python Benchmarks/campaign.py
python Benchmarks/pinn.py --config Benchmarks/configurations/C03_reference.json
```

FEniCSx 0.10.0 se ejecutó en Ubuntu/WSL, con Gmsh 4.15.2:

```powershell
wsl -d Ubuntu -- /home/lkoc/miniforge3/envs/fenicsx/bin/python /mnt/c/usr/ths_mia_fiis/tesis_MIA_lk/Benchmarks/fem.py --cases all --levels 0 1 2
```

En Linux con FEniCSx instalado: `python Benchmarks/fem.py --cases all`.
No se necesita PyTorch en el entorno FEM. Python/Windows contiene PyTorch
2.9.0+xpu, NumPy y Matplotlib; los entrenamientos usaron CPU, precisión doble y
dos hilos. Los cuadernos usan además nbformat, nbclient e ipykernel.

La reproducción puede cambiar ligeramente entre plataformas y bibliotecas.
Los manifiestos registran las versiones efectivamente utilizadas. Los tiempos
históricos incluyen procesos concurrentes y diferencias Windows/WSL: no sirven
para afirmar aceleración de PINN frente a FEM.

## Selección y auditoría

`campaign.py` define ocho candidatos, variando una decisión respecto a C03.
Se compara adición del balance, peso de PDE, anchura, profundidad, tasa de
aprendizaje y cantidad de puntos. La semilla 5 es exploratoria. Las semillas
11, 23 y 37 caracterizan sensibilidad; los casos usados en el ajuste se
identifican como tales. Una repetición con otra semilla no constituye por sí
sola validación fuera de distribución.

FEM nunca proporciona etiquetas al entrenamiento. La evaluación usa 6000
puntos independientes y contornos más densos. La aceptación térmica exige
NRMSE ≤ 5 %, error de incremento máximo de temperatura ≤ 5 % y desequilibrio
global ≤ 2 %. Los errores se normalizan con `Tmax_FEM - T0`, evitando cocientes
de temperaturas Celsius dependientes del origen de la escala. Se conservan
errores máximos, residuos y resultados por semilla incluso cuando se rechaza.

La temperatura del conductor es una reconstrucción radial a partir de la
temperatura media de su superficie externa. Los campos de ambas técnicas se
comparan fuera de los cables. Los puntos superficiales se desplazan 0,2 % del
radio para evitar ambigüedades de localización sobre la malla curva. En el
anillo se excluye del muestreo de área el borde exterior de 0,5 % y se evalúa
por separado un perfil cercano al radio interior. Estas decisiones son comunes
a ambos métodos y constan en `cases.py`.

`results/` contiene referencias FEM y la línea base PINN; `comparisons/`
conserva alternativas y pilotos. Los ficheros `.pt` guardan pesos y metadatos,
`.json` métricas e historial, y `.npz` campos numéricos. Los cuadernos muestran
estas salidas y permiten ejecutar de nuevo cada caso. Para una nueva campaña,
usar otro directorio `--output` conserva los registros anteriores.
