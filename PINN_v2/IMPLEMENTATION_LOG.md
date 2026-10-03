# PINN_v2 - Bitacora de implementacion

## Fase 0 - Auditoria (2026-10-02)
- Hecho: lectura de Benchmarks (full_domain, full_pinn, full_fem, full_physics, full_sampling, expressions, skin_effect, cases, full_ampacity, fem), pinn_cables (pde.py), datos de Kim, docx de estado del arte.
- Verificado: FEniCSx 0.10.0 vive solo en WSL2 Ubuntu (env fenicsx); no hay Docker ni dolfinx en Windows.
- Verificado: capas 8 y 9 de Kim, formula de h, T_inferior (REFACTOR_PLAN seccion 4).
- Decision B8 (usuario): alpha del conductor = 0.00393 1/K (cobre); 0.0393 del articulo queda registrado como inconsistente y no se usa.
- Creado: REFACTOR_PLAN.md e IMPLEMENTATION_LOG.md. Nada mas fuera de PINN_v2.
- Linea base: 366 passed, 3 deselected (git limpio al inicio).
- Incidencia: el entorno bloqueo escrituras mientras la sesion estaba en modo plan; ademas, el filtro de comandos rechaza textos con una barra aislada rodeada de espacios (se evita en los documentos).
- Pendiente: Fase 1.
## Fase 1 - Esqueleto, copia y caracterizacion (completada)

- Paquete `pinn_v2` con submodulos cases, geometry, physics, sampling, solvers (pinn, fem), manifest. Procedencia de las 29 copias en `COPIED_FROM.json` (ruta origen, sha256 origen, destino).
- Se eliminaron los hacks de `sys.path`; el archivado de fuentes esta centralizado en `pinn_v2/manifest/provenance.py` (`archive_sources`).
- La geometria (`geometry/domain.py`, `cases`) no importa torch (probado en subproceso), porque el runner de WSL la importa.
- Fisica sin cambios: `PHYSICS_VERSION=explicit-heat-2d-v1`; las huellas coinciden con el legado en todos los casos de cable.
- Pruebas en `PINN_v2/tests`: caracterizacion portada de `test_full_domain.py`, paridad con `Benchmarks`, huellas de procedencia contra originales intactos, ausencia de referencias a paquetes legados, geometria sin torch.
- Prueba FEM condicional (marcador `fem`, WSL2 Ubuntu, entorno fenicsx): coaxial nivel 0, RMSE vs exacta 3.4e-4 K, error Tmax 7.1e-4 K, balance 0.47 %; se rechaza sobrescribir.
- Linea base legada sin cambios: 366 passed, 3 deselected. Originales y datos del paper intactos.
- Siguiente: Fase 2 (CaseDefinition v2), ver entrada siguiente.

## Fase 2 - CaseDefinition v2 (completada, 2026-10-02)

- `cases/definition.py`: `CaseDefinition` inmutable (schema_version `pinn-v2-case-1`, id, titulo, fisica, condiciones de contorno, contacto, proveniencia, estado de datos, inconsistencias), con hash sha256 canonico, `load`, `from_dict`, `from_legacy`, `solver_case`, `domain(mode)` y `validate`. `load_all()` envuelve los 19 casos heredados como `supuesto` (fuente `legacy_catalog`) y carga `cases/v2/*.json`.
- Estado de datos por grupo: publicado, extraido, inferido, supuesto o faltante. `faltante`, un grupo sin estado, una fuente vacia, la falta de nota en estados no publicados o proveniencia sin cita se rechazan con `MissingDataError`.
- `cases/units.py`: validacion dimensional (k en W/(m K), h en W/(m2 K), T en C o K, R'' en m2 K/W). Un k dado en K m/W se rechaza con aviso de que es resistividad; no hay conversion silenciosa.
- `cases/boundary.py`: Dirichlet, Neumann y Robin por lado (bottom 1001, top 1002, left 1003, right 1004; normal saliente, q.n positivo si el calor sale). Robin se conserva como Robin; todo-Neumann se rechaza por indeterminado; h <= 0 se rechaza. Sin `boundary_conditions` rige el Dirichlet heredado, de modo que las huellas legadas no cambian.
- Contacto imperfecto: solo esquema (`contact`, R'' en m2 K/W con fuente y estado). Interfaz inexistente, duplicada, sin fuente o R <= 0 se rechazan. La huella cambia y la especificacion declara el contacto; los solvers deben lanzar NotImplementedError en lugar de tratarlo como perfecto.
- `geometry/domain.py`: `bcs`, `contact`, `has_imperfect_contact`, `side_conditions`, `all_dirichlet`, `dirichlet_value`, `boundary_sides(n)` (puntos, normales, longitudes, etiquetas) y `source_integral(region)` (comprobacion de la integral de Q dA contra la potencia prescrita, W/m, en todos los casos de cable).
- `cases/v2/kim_v5_summer.json`: Kim 2025 verano. 9 capas (capa 8 CLSM-RS10 k=2.15; capa 9 casing PE k=0.2857), 1026 A, alpha=0.00393 (B8), Robin superior h=14.605 W/m2K y T_aire 27.2 C, Dirichlet 15.2 C en los otros lados (estado extraido). No se define caso de invierno: la temperatura del aire de invierno no se encontro y no se inventa.
- Inconsistencias registradas en el caso (no se corrigen en silencio): alpha 0.0393 del articulo (no usado), h=7.371 del CSV frente a la formula 14.605/15.289, README 865 A/1000 mm2 Al frente a 1026 A/1200 mm2 Cu, T_aire de invierno ausente.
- Pruebas: `tests/test_case_definition.py` (10): 20 definiciones, huellas heredadas intactas, hash estable y sensible, integral de Q igual a la potencia, caso Kim, rechazos por datos faltantes, unidades, reglas de contorno, normales y etiquetas de lados, contacto imperfecto.
- Resultados: PINN_v2 36 passed; linea base legada 366 passed, 3 deselected. Originales y datos de articulos intactos.
- Pendiente: Fase 3 (FEM con Neumann y Robin por lado, adaptador Windows, referencias V0-V2 con 3 mallas o mas).
## Fase 3 - Referencia FEM (completada, 2026-10-02)

- `solvers/fem/full_fem.py` (runner, solo WSL): Neumann y Robin por lado (grupos fisicos 1001-1004 en dominios rectangulares, 1000 en el circular); Robin se ensambla como `h u v ds` a la izquierda y `h T_inf v ds` a la derecha, Neumann como `-q_n v ds`; Dirichlet por lado con compatibilidad de esquinas comprobada. Sin `boundary_conditions` se conserva el Dirichlet heredado: el resultado coaxial legado no cambia (RMSE 3.4e-4 K nivel 0, 4.6e-5 K nivel 1, 6.2e-6 K nivel 2). Contacto imperfecto: `NotImplementedError` antes de mallar (probado por subproceso).
- Metricas nuevas por corrida: flujo por lado, `boundary_check` (Robin: salida constitutiva h(T-T_inf) frente a traccion; Neumann: prescrito frente a traccion), punto caliente y T media por conductor, y balance consistente de forma debil `balance_consistent_pct` (reaccion en lados Dirichlet obtenida con una funcion psi, mas salida Robin constitutiva, mas Neumann prescrito, contra la potencia).
- Hallazgo: el balance por traccion `-k grad T . n` no converge a cero cuando una esquina une un lado Dirichlet con uno Robin (singularidad de esquina; 1 % en los casos simetricos). La salida Robin constitutiva y el balance de forma debil si cierran (1e-8 %). Se reportan ambos y no se confunde el primero con un defecto del solver.
- `solvers/fem/mms_fem.py` (nuevo, WSL): V0 (`mms_constant`), V1 (`mms_interface`, k=0.5 y 2 con interfaz alineada a la malla) y `mms_robin` (Robin h=10 en el lado superior, nunca convertido a Dirichlet), mallas de 16*2^nivel celdas, P2, MUMPS. Errores L2, H1, maximo nodal y RMSE en puntos fijos; la solucion exacta solo mide el error. Resultado: orden L2 observado 3.00 en los tres casos (4 niveles); L2 de 4.0e-6 (V0), 7.2e-7 (V1) y 2.8e-7 K (Robin) en el nivel 3. V1: salto de flujo relativo 7.7e-3, 1.3e-3, 8.8e-5, 1.5e-4 (la sonda con desplazamiento de 1e-6 m limita la medida a ~1e-4; el salto de T de ~2.5e-5 K es artefacto de esa sonda).
- `cases/verification.py` (nuevo): `coaxial_robin`, `coaxial_dirichlet` (Dirichlet explicito igual al heredado, reproduce bit a bit el RMSE legado) y `symmetric_pair` (dominio completo con dos cables y semidominio con lado Neumann q.n=0). Los datos de contorno son elecciones de verificacion marcadas `supuesto`, no datos de articulos.
- Coaxial con Robin (h=15, T_inf=20): RMSE frente a `radial_exact` 3.5e-4, 4.7e-5 y 6.2e-6 K en 3 mallas, error Tmax 7.6e-4, 5.9e-5 y 3.9e-6 K. Simetria: cociente de potencias 2.000000; cociente de salida Robin 2.0039, 1.99997 y 1.99998; diferencia de T media entre semidominio y dominio completo 1.9e-2, 1.1e-2 y 9e-4 K (limitada por la asimetria de la malla gmsh).
- `solvers/fem/adapter.py` (Windows, sin torch): `run_module`, `run_cable_fem`, `run_mms_fem`, `environment`, `fem_available`, `wsl_path`. `solvers/fem/environment.py` imprime versiones. `manifest/run_manifest.py`: escritura atomica sin sobrescritura (archivo temporal mas `os.link`), estado git, versiones, digests de entradas y fuentes.
- `experiments/fem_reference.py`: estudio con id nuevo en `results/phase3/<id>`; rechaza directorios existentes; escribe `summary.json`, `convergence.md` y `manifest.json` (dolfinx 0.10.0, gmsh 4.15.2, petsc4py 3.25.0; `fem_labels_used_for_training=false`). Estudio de referencia: `results/phase3/fem_ref_2026-10-02_b`. El estudio `..._a` es una corrida previa sin el balance de forma debil; se conserva sin borrar.
- Pruebas: `tests/test_fem_phase3.py` (12; 6 puras y 6 con marcador fem). PINN_v2 completo: 48 passed. Linea base legada: 366 passed, 3 deselected; `git status` solo muestra `PINN_v2/` sin rastrear.
- Pendiente: V0-V2 con cables solo cubren Robin/Neumann y MMS; el estudio de convergencia de dominio y de etiquetas para V3-V4 corresponde a la Fase 6. Siguiente: Fase 4 (M0, M1 y M2 bajo una interfaz, perdidas suaves de Neumann y Robin, balance con salida Robin).

## Fase 4 - PINN M0, M1 y M2 (completada, 2026-10-02)

- `solvers/pinn/problem.py`: una interfaz (`Problem`) para todas las familias, con dos adaptadores: `MMSProblem` (cuadrado unidad, V0/V1 y Robin) y `CableProblem` (seccion de cable explicita via `FullDomain`). Regiones, caras de interfaz, piezas de contorno con normal saliente, fuente, escalas y puntos de evaluacion. La solucion exacta o de referencia solo se usa al evaluar, nunca en una perdida.
- `solvers/pinn/networks.py`: M0 = (global, fuerte), M1 = (region, fuerte), M2 = (region, mixta: salidas T, qx, qy). Sin envolvente dura: Dirichlet, Neumann y Robin son perdidas suaves del mismo tipo; Robin nunca se convierte en Dirichlet. Salida T = T_ref + amplitud por red.
- `solvers/pinn/objective.py`: terminos pde (por region, forma conservativa con k y Q de cada capa), constitutivo (M2), `interface_temperature` (salto de T nulo), `interface_flux` (salto de q.n nulo), contornos y energia por region (salida por caras y lados menos la integral de Q). Pesos por defecto declarados y validados; no hay resistencias equivalentes, ni reconstruccion radial de T del conductor, ni flujo prescrito en la superficie del cable.
- `solvers/pinn/trainer.py`: Adam con coseno y luego L-BFGS, se restaura el mejor estado finito; semillas separadas (`init_seed` para pesos, `sampling_seed` para puntos); directorio de salida sin sobrescritura; una corrida divergida o con error se conserva como `failure.json`; `result.json`, `manifest.json`, `evaluation.npz`, `collocation.npz` y `model.pt`.
- `evaluation/audit.py`: T maxima y punto caliente, T media por conductor, NRMSE global y por region, error de dT, saltos de T y de q.n en cada interfaz, residuo PDE por region, residuo constitutivo (M2), balance global y por region. La etiqueta de la referencia (`reference_label`) es siempre explicita.
- `experiments/pinn_baselines.py`: estudios con id nuevo bajo `results/FASE/ID`. Los criterios de aceptacion siguen siendo PROPUESTOS y no confirmados: NRMSE 5 % o menos, error de dT 5 % o menos, RMSE por region 5 % del incremento o menos, balance 2 % o menos, salto de T maximo 0.1 K o menos, salto de q.n RMS 1 % o menos.
- Resultados (`results/phase4/p4_v0`, `p4_v1`, `p4_v2`; perfil 4000 Adam + 60 L-BFGS; M2 en coaxiales 6000 + 100; 3 semillas en V0 y V1, 2 en V2):
  - V0 (`mms_constant`, `mms_robin`): M0, M1 y M2 aceptados; NRMSE maximo 0.016 % (M0, M1) y 0.063 % (M2).
  - V1 (`mms_interface`): M0 rechazado (NRMSE 6.2 a 6.5 %, salto de q.n RMS 17 %: el cambio de pendiente en la interfaz no cabe en una red suave unica); M1 aceptado (NRMSE maximo 0.012 %); M2 aceptado (maximo 0.13 %).
  - V2 (`coaxial_full`, `coaxial_robin`, `coaxial_bc_dirichlet`): M0 rechazado (NRMSE 13.5 a 23 %, error de dT cercano a 100 %: no resuelve las capas finas); M1 aceptado (maximo 0.017 %); M2 aceptado (maximo 0.071 %), con balance derivado hasta 0.23 %.
- Hallazgos: (1) M0 y M1 son numericamente identicos en casos de una sola region (misma red, V0); no son evidencia independiente. (2) M2 es menos preciso que M1 en T pero cumple los criterios; su balance con flujo derivado no esta garantizado (en una corrida larga previa llego a 12.9 % en coaxial). (3) El rechazo de M0 en V1 y V2 es una propiedad de la parametrizacion (una red suave), no de un ajuste pobre de hiperparametros.
- Pruebas: `tests/test_phase4_pinn.py` (29 sin marcador lento) y pruebas de integracion con marcador `slow`.

## Fase 5 - Variantes de PINN (completada, 2026-10-02)

- Todas las variantes son ADAPTACIONES al contrato fisico de este proyecto, no reproducciones fieles. Cada DOI o identificador esta marcado "por verificar": no hay acceso web para comprobarlo.

| Variante | Clase | Referencia (por verificar) | Estado |
|---|---|---|---|
| CUSP (`CuspModel`) | adaptada, inspirada | Tseng et al. 2023, arXiv:2210.08424 | implementada: entrada con distancia con signo a cada interfaz (quiebre en el nivel cero) |
| DCSNN (`DCSNNModel`) | adaptada | Hu, Lin, Lai 2022, 10.1016/j.jcp.2022.111576 | implementada: redes suaves por subdominio con continuidad en interfaz como perdida |
| FBPINN (`FBPINNModel`) | adaptada | Moseley et al. 2023, 10.1007/s10444-023-10065-9 | implementada con 2 partes y ventana de ancho 0.02 |
| RBA (`attention.py`, kind rba) | adaptada | Anagnostopoulos et al. 2024, arXiv:2307.00379 | implementada en el termino PDE; el valor de la perdida se reporta sin ponderar |
| SA-PINN (kind sa) | adaptada | McClenny y Braga-Neto, arXiv:2009.04544 | implementada, mismo mecanismo con otra regla de ascenso |
| I-PINN y AdaI-PINN | - | no verificado | NO implementada: no se pudo verificar el original y no se inventa |
| VPINN y hp-VPINN | - | - | NO implementada (opcional en el plan) |

- L-BFGS no se combina con pesos adaptativos (la perdida cambia): `trainer.run` lanza ValueError y queda `failure.json`.
- Resultados V1 (`results/phase5/p5_v1`, 3 semillas): CUSP aceptado (NRMSE 0.0048 a 0.0072 %); DCSNN aceptado (0.020 a 0.038 %); FBPINN rechazado (6.0 a 6.5 %); M0+rba y M0+sa rechazados (6.3 a 6.5 %); M1+rba y M1+sa aceptados (error de dT 0.36 a 0.58 %, peor que M1 sin atencion: sin ganancia aqui).
- Resultados V2 (`results/phase5/p5_v2`, 1 semilla, `coaxial_full`): CUSP 13.3 %, DCSNN 13.6 % y FBPINN 14.6 % de NRMSE (rechazados: parametrizacion global, no separa las capas del cable); M1+rba aceptado (0.046 %).
- Conclusion: solo las variantes que ven la interfaz (CUSP, DCSNN) resuelven V1; ninguna variante global resuelve capas concentricas finas. La atencion no mejora a M1.
- Pruebas: `tests/test_phase5_variants.py` (16).

## Fase 6 - Escalera V3-V5, CLI, docs y front-end (completada, 2026-10-02)

- FEM (results/phase6/p6_fem3, WSL2, dolfinx 0.10.0): convergente en 4 mallas para V3 y V4; Kim V5 converge en niveles 0-2 (el 3 agota memoria). aras_flat (3.48 % del ascenso) y Kim (11.9 %) son sensibles al dominio.
- PINN M1 (6000 Adam + 100 L-BFGS) en results/phase6/p6_pinn_m1; largo (20000 Adam + 200 L-BFGS) en p6_pinn_m1_long: xlpe_backfill NRMSE 2.43 % rechazado (balance 4.56 %, salto de T); xlpe_discrete_layers 0.407 % rechazado (salto de T en interfaz); aras_flat 0.502 % aceptado (balance 0.096 %).
- Kim verano (p6_pinn_kim_long, M1 semilla 0, 12000 Adam, sin L-BFGS, 18892 s): NRMSE 13.4 %, error de Tmax 18.4 K, balance 167 %: rechazado en 5 de 6 compuertas. Causa NO diagnosticada; pendiente revisar el historial de perdida y el balance por region. No se atribuye a ningun diseno hasta entonces.
- Criterios de aceptacion: siguen siendo propuestos, no confirmados por el usuario.
- CLI `python -m pinn_v2` (help, cases, show, validate, train, fem, search), busqueda de hiperparametros, hook `on_log` del entrenador, documentacion HTML local (docs/build_docs.py, 14 paginas) y front-end web (front-end/server.py: editar casos, ejecutar FEM o PINN con progreso en vivo, comparar con mapas de error, ayuda integrada). El PINN rechaza contacto imperfecto con NotImplementedError; el FEM lo admite.
- Pruebas: 109 pasan en PINN_v2. Logs de raiz movidos a results/logs_root. README.md de raiz creado.

## Fase 7 - Reportes agregados (completada, 2026-10-02)

- `pinn_v2/experiments/aggregate.py` lee los `summary.json` existentes (solo lectura, sin reentrenar) y escribe `results/phase7/aggregate/summary.md` y `all_runs.csv`: 79 corridas de p4_v0..p6_pinn_kim_long. Los estudios FEM (sin filas PINN) se omiten. No se sobrescribio ningun resultado previo.
- Prueba: `tests/test_phase7_aggregate.py`.

## Fase 8 - Cierre (completada, 2026-10-02)

- `python -m pytest tests` en PINN_v2: 110 pasan, incluidas las marcadas slow y fem (8 en total, FEM via WSL2).
- Linea base heredada (`python -m pytest Benchmarks pinn_cables -q`, desde la raiz): 366 pasadas, 3 deseleccionadas, intacta.
- `git status`: solo `PINN_v2/` sin seguimiento; no se modifico codigo heredado.
- Pendientes abiertos: diagnostico del fallo del M1 en `kim_v5_summer` (causa sin establecer); confirmacion de los criterios de aceptacion por el usuario; I-PINN, AdaI-PINN y VPINN siguen sin implementar por decision del plan.

## Diagnostico Kim (M1 verano)
Verificado: perdida aun decreciente, residuo PDE del suelo 0.099 contra 3e-4 en cables, campo bajo la referencia en 99.9 % de los puntos, balance 167 %, interfaces pequenas. Causa no confirmada; hipotesis de entrenamiento o capacidad insuficiente, pendiente de corrida exploratoria en carpeta nueva.
