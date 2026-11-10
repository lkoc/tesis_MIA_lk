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
2. `build_model(problem, arquitectura, ancho, profundidad, activación)` crea la red. `M1_EXACT`
  elige trazas Fourier para interfaces circulares, trazas planas para MMS y levantamiento radial
  para annulus CIGRE. Sus entradas nuevas usan `unit_box_v1`, con features en `[0,1]`;
  el encoder queda versionado para reconstruir checkpoints.
3. `Objective` genera los puntos de colocación (con su propia semilla) y calcula los términos de pérdida.
  Para `M1_EXACT`, solo el residuo PDE contribuye al objetivo; la auditoría física permanece separada.
4. `trainer.fit` aplica Adam con decaimiento coseno y luego L-BFGS opcional. Selecciona estados post-actualizacion en los limites de registro. Con remuestreo, el minimo se reinicia por nube y la validacion fija se conserva. El ultimo estado completo para reanudar no se reemplaza por el mejor estado para inferir.
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
| `model.json` + `model.pt` | especificacion fisica/modelo y bundle de pesos/entrenamiento autosuficientes |

## Reconstruccion y persistencia

`DeepXDE_EXACT` compone el campo nativo exacto dentro de una red PyTorch de DeepXDE. La
autodiferenciacion/optimizacion pasa por `dde.Model`; la construccion de restricciones y el
ansatz se reutilizan, en vez de duplicar las ecuaciones de frontera. Los residuos por material
se normalizan para que cada media regional tenga el mismo peso que el objetivo nativo. El test
de paridad incluye gradientes. Exporta estado de campo para inferencia, no historia de reanudacion
del optimizador DeepXDE. Esta distincion queda en la configuracion efectiva del par.

`solvers/pinn/checkpoint.py` centraliza carga estricta, inferencia T/flujo, etiquetado de regiones,
RNG, nubes, estados adaptativos y serializacion nativa. La especificacion declarativa queda en
un JSON inmutable; su SHA256 se guarda dentro del PT. Solo el PT se reemplaza atomicamente,
evitando prometer una operacion atomica sobre dos nombres de archivo en Windows.

CLI, scoring de busqueda y frontend comparten este cargador. Las grillas, catalogo y referencias
no son datos obligatorios para inferir o reanudar. El codigo compatible instalado sigue siendo
necesario; no se serializan objetos Python arbitrarios. La lectura usa `weights_only=True`.
La compatibilidad historica de pesos no implica que se pueda recuperar un optimizador no guardado.

## Cómo se ve cada capa

Cada capa del cable y cada material del suelo es una **región**. En M1, M2 y las variantes por región,
cada región tiene su propia red; en M0 hay una sola red para todo el dominio. Normalmente las
interfaces entre regiones se imponen como pérdidas de continuidad de temperatura y de flujo normal.
En `M1_EXACT`, trazas compartidas fijan exactamente T y flujo normal; la conductividad permite el
salto correspondiente del gradiente. Las condiciones externas se incorporan al levantamiento según
la geometría y BC del caso. El modo conserva el mismo trainer y formatos, y valida compatibilidad
antes de construir el modelo.