# Arquitectura del paquete

```
pinn_v2/
  cli.py, __main__.py       punto de entrada: python -m pinn_v2
  cases/                    definición de casos (JSON), catálogo y validación
    data/                   19 casos heredados (un JSON cada uno)
    v2/                     casos nativos del formato v2
  geometry/                 dominio completo: regiones, capas, fuentes
  physics/                  operadores, residuales, efecto piel
  sampling/                 puntos de colocación
  solvers/
    pinn/                   problema, redes, objetivo, entrenador, atención, variantes
    fem/                    adaptador de Windows y ejecutores de FEniCSx (WSL2)
    ampacity.py             capacidad de corriente
  evaluation/               métricas, auditoría independiente, lectura de referencias
  experiments/              estudios por fases y búsqueda de hiperparámetros
  manifest/                 registro de cada corrida (versiones, semillas, huellas)
configs/                    ejemplos de configuración JSON
tests/                      pruebas unitarias e integración
docs/                       esta documentación
results/                    una carpeta por corrida; nunca se sobrescribe
```

## Flujo de una corrida PINN

1. `load_problem(fuente)` lee el caso (id o archivo JSON) y construye el `Problem`: regiones, conductividades, fuentes, caras de interfaz y tramos de frontera.
2. `build_model(problem, arquitectura, ancho, profundidad, activación)` crea la red.
3. `Objective` genera los puntos de colocación (con su propia semilla) y calcula los términos de pérdida.
4. `trainer.fit` aplica Adam con decaimiento coseno y luego L-BFGS opcional. Si la nube interior es fija, restaura el menor loss de entrenamiento; con remuestreo, el mínimo se reinicia por nube y la validación física fija se conserva para comparar toda la corrida.
5. `evaluate_model` hace la auditoría independiente con puntos nuevos y, si hay referencia, el reporte de errores.
6. `trainer.run` escribe los archivos de la corrida y el manifest.

## Flujo de una corrida FEM

`python -m pinn_v2 fem` llama a `solvers/fem/adapter.py`, que ejecuta `full_fem.py` dentro de WSL2
con el mismo caso. Los resultados vuelven como JSON y NPZ. Las dos ramas comparten la definición
del caso y los puntos de evaluación, por eso se pueden comparar punto a punto.

## Datos que viajan entre módulos

| Objeto | Qué es |
|---|---|
| `CaseDefinition` | caso declarativo con procedencia y estado de cada dato |
| `Problem` | caso listo para el PINN (regiones, escalas, muestreo) |
| `Objective` | pérdida ponderada, nunca lee referencia |
| `result.json` | configuración, historia, pérdidas finales y evaluación |
| `manifest.json` | versiones, semillas, huellas de entradas y código |

## Cómo se ve cada capa

Cada capa del cable y cada material del suelo es una **región**. En M1, M2 y las variantes por región,
cada región tiene su propia red; en M0 hay una sola red para todo el dominio. Las interfaces entre
regiones se imponen como pérdidas de continuidad de temperatura y de flujo normal.