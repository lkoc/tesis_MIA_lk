# Reproducibilidad

Cada corrida deja suficiente información para repetirla y para saber qué código la produjo.

## Qué se guarda

En la carpeta de salida de un entrenamiento hay:

| archivo | contenido |
|---|---|
| `result.json` | configuración usada, historial de pérdida, evaluación y auditoría |
| `model.pt` | pesos entrenados |
| `collocation.npz` | puntos de colocación usados |
| `evaluation.npz` | campo predicho y, si hubo referencia, el campo de referencia |
| `manifest.json` | comando, semillas, versiones, estado de git y huellas sha256 |
| `source/` | copia de los fuentes tal como estaban al correr |

El manifest guarda el comando exacto, las semillas, las versiones de Python, NumPy, SciPy y PyTorch,
el commit de git y si la carpeta `PINN_v2` tenía cambios sin confirmar (`dirty`).

## Semillas

- `init_seed`: inicialización de la red.
- `sampling_seed`: puntos de colocación del entrenamiento.
- Auditoría: semilla fija 909091, que el entrenamiento nunca usa.
- Búsqueda de hiperparámetros: semilla fija 424243 para la pérdida held-out.

Con las mismas semillas, el mismo caso y los mismos ajustes se repite la corrida en CPU y float64.

## Huellas del caso

Un caso tiene dos huellas: `sha256` del archivo completo y `physics_sha256` solo de la parte física.
Si cambias el título o las notas, cambia la primera pero no la segunda. Úsala para comprobar que dos
corridas resolvieron la misma física.

## Nada se sobrescribe

- Una carpeta de salida que ya existe se rechaza.
- Los JSON se escriben de forma atómica y no reemplazan un archivo existente.
- Los estudios fallidos se conservan; no se borran.

## Repetir una corrida

1. Abre `manifest.json` y copia el campo `command`.
2. Usa una carpeta de salida nueva.
3. Compara `physics_sha256` y las semillas con las de la corrida original.

## Qué no entra al entrenamiento

La referencia FEM y los datos de artículos se pasan solo a la evaluación. El resultado dice
`reference_used_in_loss: false` y `fem_labels_used: 0`.