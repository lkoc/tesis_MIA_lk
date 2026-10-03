# Formato de caso

## La pregunta de siempre: ¿qué formato de entrada se usa?

| Versión | Formato de entrada | Comentario |
|---|---|---|
| `pinn_cables` (anterior) | varios CSV (`domain`, `cables_placement`, `cable_layers`, `boundary_conditions`, `scenarios`, `soil_properties`, `solver_params`) y un YAML opcional | datos físicos y parámetros del solver en archivos separados |
| `Benchmarks` (anterior) | un JSON por caso, sin procedencia | formato heredado, sigue funcionando |
| **PINN_v2** | **un solo JSON por caso** (`pinn-v2-case-1`) | lo leen por igual el PINN y el FEM |

El catálogo activo se descubre recursivamente bajo `pinn_v2/cases/v2/`; contiene casos de cable y
casos generales 2D de verificación analítica/MMS. Los casos de cable construyen el mismo `FullDomain`
para FEM y PINN; las verificaciones usan su propio `Problem` y también pueden resolverse con ambos
motores. Los JSON planos v1 se conservan en `pinn_v2/cases/data/` para reproducir experimentos
antiguos, pero no se listan ni se aceptan como casos activos. `pinn_v2/cases/archive/v2/` contiene
copias históricas de las verificaciones, fuera del descubrimiento activo.

## Estructura de un caso v2

Ejemplo real: `pinn_v2/cases/v2/kim_v5_summer.json`.

```json
{
  "schema_version": "pinn-v2-case-1",
  "id": "kim_v5_summer",
  "title": "Kim 2025, seis cables en lecho PAC, verano",
  "physical": { "...": "geometría, capas, k, corriente" },
  "boundary_conditions": {
    "top": {"type": "robin",
            "h": {"value": 14.605, "unit": "W per m2 K"},
            "T_inf": {"value": 27.2, "unit": "degC"}},
    "bottom": {"type": "dirichlet", "temperature": {"value": 15.2, "unit": "degC"}}
  },
  "contact": [],
  "provenance": [{"id": "kim2025", "citation": "Kim et al., Geothermics 125:103151 (2025)"}],
  "data_status": {"geometry": {"status": "supuesto", "source": "legacy_kim_layered", "note": "..."}},
  "inconsistencies": [{"field": "alpha", "values": {"paper": 0.0393, "used": 0.00393}, "resolution": "..."}]
}
```

### Bloque `physical`

| Campo | Significado |
|---|---|
| `id`, `kind` | identificador y tipo (`cable`) |
| `bounds` | dominio `[xmin, xmax, ymin, ymax]` en metros; la superficie es `ymax = 0` |
| `T0` | temperatura de nivel (frontera Dirichlet por defecto), en °C |
| `scale` | escala de potencia usada para normalizar |
| `cables` | centros `[x, y]` de cada cable |
| `radius` | radio exterior del cable |
| `layers` | lista `[r_interior, r_exterior, k]` por capa, de adentro hacia afuera |
| `current`, `R20`, `alpha` | corriente, resistencia a 20 °C y coeficiente del cobre (0.00393 1/K) |
| `power` | potencia lineal del conductor en W por metro (`Q = p'/A`; se verifica que la integral de Q dA coincida) |
| `k` | conductividad del suelo; `bands` y `patch` definen estratos y parches |

### Fronteras

Cada lado (`top`, `bottom`, `left`, `right`) admite:

- `dirichlet`: `temperature` con valor y unidad.
- `neumann`: `flux` con valor y unidad.
- `robin`: `h` y `T_inf` con valor y unidad. El PINN impone $q\cdot n = h\,(T-T_\infty)$; nunca lo transforma en Dirichlet.

Si `boundary_conditions` no está, se usa el contrato heredado: Dirichlet `T0` en todo el borde.

### Estado de cada dato

Cada grupo (`geometry`, `materials`, `source`, `boundary_conditions`) lleva un estado:

| Estado | Significa |
|---|---|
| `publicado` | tomado de una publicación citada |
| `extraido` | leído de un archivo del proyecto |
| `inferido` | deducido de otros datos |
| `supuesto` | decisión de modelado; requiere nota |
| `faltante` | no existe; el caso se **rechaza** |

Todo estado distinto de `publicado` exige una nota, y toda fuente debe aparecer en `provenance`.

## Crear un caso nuevo

1. Copia un JSON activo, por ejemplo `pinn_v2/cases/v2/kim_v5_summer.json`, a tu carpeta de trabajo y edítalo.
2. Valídalo: `python -m pinn_v2 validate mi_caso.json`
3. Revísalo: `python -m pinn_v2 show mi_caso.json`
4. Úsalo con cualquier herramienta, sin tocar el código:

```bash
python -m pinn_v2 train --definition mi_caso.json --model M1 --output results/runs/mi_caso_m1
python -m pinn_v2 fem --definition mi_caso.json --levels 0 1 2 --output results/runs/mi_caso_fem
```

Para dejarlo disponible por id, guárdalo en cualquier subcarpeta de `pinn_v2/cases/v2/` con el nombre `<id>.json`.

> La huella `sha256` que imprime `show` identifica el contenido del caso. Cada corrida PINN guarda además
> `physics_sha256`, la huella de la física que realmente usó.