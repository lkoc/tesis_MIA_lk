# Criterios geométricos de `pinn-v3-problem-1`

## Regla de prioridad

Cada elemento de `regions` debe declarar un entero `priority` único:

```json
{
  "id": "soil",
  "priority": 0,
  "shape": {"type": "rectangle", "bounds": [-4, 4, -4, 0]},
  "conductivity": 1.0,
  "source": 0.0
}
```

Si dos regiones se solapan, la región con mayor `priority` es la propietaria del punto o superficie solapada. Esta regla se aplica en ambos motores:

- PINN: `GenericProblem.owner_index()` ordena por `priority` antes de asignar la región.
- FEM/Gmsh: `generic_fem.build_mesh()` ordena las superficies por `priority` antes de asignar los grupos físicos.
- Muestreo: los puntos se aceptan solo si pertenecen al propietario calculado y están separados de la frontera regional.

Las prioridades deben ser únicas. No se debe depender del orden accidental de las listas JSON.

## ¿La conversión `disk` a `annulus` es automática?

No durante la ejecución. Es parte de la definición geométrica final del problema.

La transformación se realizó una vez mediante la herramienta de migración:

`tools/migrate_region_priorities.py`

La herramienta asignó prioridades y convirtió las capas concéntricas acumulativas:

```text
incorrecto para regiones físicas:
core       = disk(r1)
insulation = disk(r2)
sheath     = disk(r3)
```

en regiones disjuntas:

```text
core       = disk(r1)
insulation = annulus(r1, r2)
sheath     = annulus(r2, r3)
```

Los `problem.json` actuales deben contener ya la forma final. El runtime no reescribe geometría ni infiere capas faltantes.

## Solapamiento intencional frente a capas físicas

La prioridad es útil para superposiciones intencionales, por ejemplo:

- un parche de material sobre una región de suelo;
- una inclusión que reemplaza el material de fondo;
- una región de refinamiento o zona con una ley física distinta.

Para materiales concéntricos que deben aparecer todos en la malla FEM, las regiones deben ser disjuntas. Si se modelan todas como discos acumulativos, la última región puede cubrir las anteriores y Gmsh puede dejar una región sin superficies físicas. El FEM debe rechazar esa situación, no inventar una partición.

## Formas admitidas

- `rectangle`: `bounds: [x0, x1, y0, y1]`.
- `disk`: `center: [cx, cy]`, `radius_m`.
- `annulus`: `center`, `inner_radius_m`, `outer_radius_m`.

La forma de la región y la forma del dominio son independientes. Un dominio circular/anular debe declararse como tal en `domain`; no se representa como un rectángulo con regiones auxiliares si se espera conservar la topología FEM.

## Fronteras

Dominio rectangular:

- `bottom`, `top`, `left`, `right`.

Dominio anular:

- `inner`, `outer`.

Cada frontera declara explícitamente su tipo y sus parámetros:

- Dirichlet: `value`.
- Neumann: `q_n_W_m2`.
- Robin: `h_W_m2K` y `T_inf_C`.

No se debe convertir una frontera circular interna en una cara rectangular equivalente.

## Fuentes y campos espaciales

`conductivity`, `source` y `heat_capacity` pueden ser:

- un número constante;
- una expresión algebraica restringida;
- una tabla 1D interpolada.

La fuente se conserva por región y se evalúa en los mismos puntos físicos que usa el motor. La potencia integrada debe comprobarse frente al dato declarado cuando exista.

## Validación antes de resolver

Antes de lanzar PINN o FEM, el repositorio debe comprobar:

1. prioridades presentes y únicas;
2. formas dentro del dominio;
3. cobertura completa del dominio;
4. pertenencia inequívoca de puntos de muestreo;
5. regiones físicas no vacías después de aplicar prioridad;
6. fronteras compatibles con la topología del dominio;
7. campos finitos y conductividad positiva;
8. fuente integrada y potencia declarada, cuando estén disponibles.

La huella del `problem.json` validado es la identidad común que deben registrar PINN y FEM.
