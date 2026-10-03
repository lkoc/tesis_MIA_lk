# Modelos

Cambiar de modelo es cambiar un argumento: `--model NOMBRE` (o `"model"` en el JSON).

| Nombre | Partición | Formulación | Idea |
|---|---|---|---|
| `M0` | global | fuerte | una red para todo el dominio; control |
| `M1` | por región | fuerte | una red por material o subdominio; interfaces como pérdidas |
| `M2` | por región | mixta | salidas $(T, q_x, q_y)$; la ley constitutiva $q=-k\nabla T$ es una pérdida |
| `CUSP` | adaptada | fuerte | variante de la fase 5 |
| `DCSNN` | adaptada | fuerte | variante de la fase 5 |
| `FBPINN` | adaptada | fuerte | descomposición del dominio adaptada |
| `<modelo>+rba` | igual que el base | igual | atención por residuo (Adam solamente) |
| `<modelo>+sa` | igual que el base | igual | pesos autoadaptativos (Adam solamente) |

Las variantes de la fase 5 son **adaptaciones** de las ideas originales, no reproducciones; las citas
quedan marcadas "por verificar" en `REFACTOR_PLAN.md`. I-PINN, AdaI-PINN y VPINN no están implementados.

## Qué comparten todos

- Ninguna condición de frontera está incrustada en la red: Dirichlet, Neumann y Robin son pérdidas blandas.
- La salida es `T_ref + amplitud x crudo`, con escalas por región para que las pérdidas sean adimensionales.
- La ecuación `-div(k grad T) = Q` se evalúa en cada región con su propia conductividad y fuente.

## Términos de la pérdida

| Término | Qué penaliza | Peso por defecto |
|---|---|---|
| `pde` | residuo de la ecuación en cada región | 1 |
| `constitutive` | `q + k grad T` (solo M2) | 1 |
| `interface_temperature` | salto de T en interfaces | 100 |
| `interface_flux` | salto del flujo normal | 10 |
| `dirichlet` | error de temperatura en lados Dirichlet | 100 |
| `neumann` | error de flujo en lados Neumann | 10 |
| `robin` | error de la ley de Robin | 10 |
| `energy` | balance de energía por región | 1 |

Cambiar un peso: `--weights dirichlet=50 interface_flux=20` o `"weights": {...}` en el JSON.

## Atención (RBA y SA)

Las dos modifican los pesos por punto del residuo interior de forma cerrada, a partir del residuo
sin gradiente. No leen ninguna solución de referencia. Como el peso cambia durante el entrenamiento,
L-BFGS no se puede usar; el marco fuerza `lbfgs_steps = 0` cuando hay atención.

## Agregar un modelo nuevo

1. Escribe una clase en `pinn_v2/solvers/pinn/variants.py` que implemente `field(region, xy)`,
   `temperature(xy, labels)` y `parameter_count()` (mira `PINNModel` en `networks.py`).
2. Regístrala en el diccionario `VARIANTS` con el nombre que usarás en `--model`.
3. Corre `pytest tests -q`; los tests de las fases 4 y 5 sirven de plantilla.

No hay que tocar ni el entrenador ni la pérdida: ambos trabajan con cualquier modelo que ofrezca esa interfaz.