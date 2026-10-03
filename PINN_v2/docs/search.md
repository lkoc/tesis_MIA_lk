# Búsqueda de hiperparámetros

```bash
python -m pinn_v2 search --config configs/search_example.json --trials 8 --strategy random
```

## Archivo de búsqueda

```json
{
  "case": "xlpe_single",
  "trials": 8,
  "strategy": "random",
  "seed": 0,
  "base": {"adam_steps": 1500, "lbfgs_steps": 0, "init_seed": 0, "sampling_seed": 11},
  "space": {
    "model": ["M0", "M1", "M2"],
    "width": [16, 32, 64],
    "depth": [2, 3, 4],
    "lr": [0.001, 0.003, 0.01],
    "weights.interface_temperature": [30, 100, 300]
  }
}
```

- `base`: valores fijos para todas las pruebas (pasos, semillas, pesos, puntos).
- `space`: listas de valores. Las claves admitidas son `model`, `width`, `depth`, `lr`, `activation`,
  `adam_steps`, `lbfgs_steps`, `weights.<termino>` y `points.<nombre>`.
- `strategy`: `random` (sin repetir combinaciones, reproducible con `seed`) o `grid` (producto completo, recortado a `trials`).

## Cómo se elige la mejor prueba

Cada prueba es una corrida normal en su propia carpeta `trial_XXX`. Después se reevalúa la red entrenada
con la **pérdida física de validación**:

- mismos términos y **pesos fijos por defecto** para todas las pruebas;
- puntos de colocación nuevos, con una semilla que el entrenamiento nunca usa;
- sin ninguna solución de referencia (ni exacta, ni FEM, ni de artículos).

La pérdida de entrenamiento no sirve para comparar, porque cambia cuando cambias los pesos.

## Salidas

| Archivo | Contenido |
|---|---|
| `summary.json` | espacio, filas por prueba, mejor prueba, semillas y declaraciones de no uso de referencias |
| `report.md` | tabla ordenada por la pérdida de validación |
| `manifest.json` | huella de la búsqueda |
| `trial_XXX/` | corrida completa de cada prueba |

Una prueba que falla queda registrada con su mensaje y no detiene la búsqueda. Si la carpeta de salida
ya existe, la búsqueda se rechaza.

## Cuidados

- La pérdida de validación de M2 incluye el término constitutivo y la de M0 y M1 no; al comparar
  **entre** familias, trátala como una guía, no como un veredicto.
- Si pasas una referencia, su error se guarda en el resumen solo como información; nunca entra en el orden.
- Para decidir el modelo final, revisa la auditoría de las mejores pruebas (interfaces y balance), no solo la pérdida.