# PINN_v2 - Plan de refactorizacion (Fase 0, auditoria de solo lectura)

Fecha: 2026-10-02. Rama base: main (HEAD 94d6c2f, arbol limpio). Linea base: pytest -> 366 passed, 3 deselected.
Alcance de esta fase: se leyo codigo y documentos; no se modifico, borro ni sobrescribio nada fuera de PINN_v2.

## 1. Decisiones del usuario (B1-B8)

| Id | Decision |
|----|----------|
| B1 | La carpeta nueva es PINN_v2 (paquete Python pinn_v2). |
| B2 | Se busca en Benchmarks y en pinn_cables; lo necesario se copia a PINN_v2. Los originales no se tocan. |
| B3 | FEniCSx es el ya instalado en el equipo (seccion 3). |
| B4 | Identidad real de las capas externas de Kim: se verifica contra el articulo (seccion 4). |
| B5 | h y T_inferior de Kim: se buscan en el articulo y en los resumenes .md (seccion 4). |
| B6 | Se ejecutan todas las fases 0-8. |
| B7 | Autorizada la lectura de todo lo necesario (docx de estado del arte incluido). |
| B8 | alpha: si corresponde al nucleo/conductor se usa el valor fisico del cobre 0.00393 1/K; en otro caso el del articulo (0.0393). En el modelo alpha solo interviene en R(T) del conductor, luego se usa 0.00393. El 0.0393 del articulo queda como dato inconsistente, sin usarse. |

## 2. Inventario y matriz de reutilizacion

| Origen | Contenido | Accion en PINN_v2 |
|--------|-----------|-------------------|
| Benchmarks/full_domain.py | Region, Interface, FullDomain, dc_conductivity, radial_exact | Copiar y extender (BC por lado, contacto, sin torch). |
| Benchmarks/cases.py y carpeta cases | catalogo, validate_case, MMS, field_k | Copiar; los JSON son entradas heredadas envueltas en CaseDefinition v2. |
| Benchmarks/expressions.py | lenguaje restringido de expresiones | Copiar sin cambios de logica. |
| Benchmarks/skin_effect.py | perfil de Bessel, validacion de perdidas | Copiar. |
| Benchmarks/full_physics.py | heat_residual, mixed_residual | Copiar; gradients y laplace_variable_k se vendorizan desde pinn_cables (modulo pinn/pde.py). |
| Benchmarks/full_sampling.py | politicas uniform, mixed, interface, residual | Copiar. |
| Benchmarks/full_pinn.py | FullPINN (9 variantes, varias alias), PhysicsLoss, evaluate, train | Refactorizar en solvers/pinn (M0/M1/M2 tras una interfaz comun) con BC suaves. |
| Benchmarks/full_fem.py y fem.py | FEM P2 gmsh/OCC, MUMPS, Picard | Copiar como solvers/fem/runner.py (solo WSL) mas adaptador Windows; anadir Neumann y Robin. |
| Benchmarks/full_ampacity.py | raiz de corriente | Reutilizar en la fase de ampacidad. |
| Pruebas de Benchmarks (9 modulos) | caracterizacion | Portar. |
| Resultados historicos y datos de articulos | evidencia | Solo lectura; nunca etiquetas. |
| build_final_thesis, report, notebooks, thesis_results, full_study_report | sobrescriben artefactos | No ejecutar sin inspeccion; no se copian. |
| pinn_cables | marco heredado (Kennelly, IEC 60287) | Solo contraste; no entra en variantes activas. |

## 3. Entorno de ejecucion (B3)

- Windows 11, Python 3.12, PyTorch 2.9.0+xpu en CPU float64; skfem instalado; sin dolfinx en Windows.
- FEniCSx solo en WSL2 Ubuntu: env fenicsx de miniforge del usuario lkoc (Python 3.12.13, dolfinx 0.10.0, gmsh 4.15.2, petsc4py 3.25.0, mpi4py 4.1.1, numpy 2.4.3, scipy 1.17.1). Fijado por Benchmarks/environment/conda-explicit-linux-64.txt.
- No es Docker: no hay marcador de contenedor, el demonio esta detenido y la distribucion WSL por defecto es docker-desktop; usar siempre wsl -d Ubuntu.
- Los JSON FEM historicos no guardan la plataforma; la conclusion procede de rutas, kernel, fechas de instalacion y comandos documentados.
- Politica: el entorno WSL no recibe pytest ni torch. Pruebas FEM condicionales (se omiten si WSL/dolfinx no responde), invocadas por subproceso; Windows consume NPZ y JSON. Una malla, un rango MPI, P2.
- Atribucion: un resultado solo se llama FEniCSx si lo produjo solvers/fem/runner.py en ese entorno (el manifiesto guarda versiones de dolfinx y gmsh).
## 4. Datos de Kim 2025 (B4, B5) y estado de los datos

Fuente: Kim et al., Geothermics 125:103151, doi 10.1016/j.geothermics.2024.103151.

| Dato | Valor | Estado |
|------|-------|--------|
| Capas 1-7 de kim_layered.json | conductor Cu k=400; XLPE y semiconductoras 0.2857; pantalla 0.167 y 237 | publicado |
| Capa 8 (r 0.05-0.10 m) | CLSM-RS10, k = 2.15 W/mK | publicado |
| Capa 9 (r 0.10-0.11 m) | tubo de PE (casing), k = 0.2857 W/mK | publicado |
| Corriente y conductor | 1026 A, Cu 1200 mm2 | publicado |
| Suelo | estratos k = 1.804, 1.351 y 1.517 W/mK | publicado |
| Parche de relleno | k = 2.094 | publicado (CSV del ejemplo) |
| Robin superior | h = 7.371 + 6.43 v^0.75; v=1.17 da 14.605 (verano), v=1.32 da 15.289 (invierno) | publicado (formula), calculado |
| T aire | 27.2 C (verano) | publicado |
| T inferior | 15.2 C | publicado |
| alpha del conductor | 0.00393 1/K (cobre) | decidido por el usuario (B8); el articulo imprime 0.0393, inconsistente, no usado |
| Dominio 8x4 m con Dirichlet 20 C en kim_layered.json | adaptacion controlada heredada | supuesto |
| Perdidas DC a 20 C, flujo circular uniforme | adaptacion heredada | supuesto |

Inconsistencias (marcadas inconsistente; nada se inventa ni se corrige en silencio):
1. alpha: articulo 0.0393 frente a JSON 0.00393; resuelta por B8 (0.00393, conductor de cobre).
2. README del ejemplo Kim: 865 A y 1000 mm2 Al; articulo: 1026 A y 1200 mm2 Cu.
3. boundary_conditions.csv usa h=7.371 constante (solo el termino sin viento).
4. Un informe previo intercambia las velocidades de viento de verano e invierno.
5. Una cifra h=14.0 y 15.0 citada en el chat era erronea; verificados: 14.605 y 15.289.

## 5. Estado del arte (docx, B7)

Documento de etapa 1 generado automaticamente; los DOI deben verificarse antes de citarlos. Cada variante se clasifica como fiel, inspirada o adaptada.

| Variante | Clase en PINN_v2 | Referencia clave |
|----------|------------------|------------------|
| M0 PINN fuerte global | control | Raissi 2019, 10.1016/j.jcp.2018.10.045 |
| M1 por subdominio o material | inspirada (cPINN, XPINN) | Jagtap 2020, CiCP 28:2002; I-PINN Sarma y Roy, CMAME 2024 |
| M2 mixta (T, qx, qy) | inspirada (primer orden) | contraste: Shukla 2021, arXiv:2104.10013 |
| I-PINN y AdaI-PINN | adaptada | Roy 2024, 10.4208/cicp.OA-2024-0131, arXiv:2406.04626 |
| Cusp-capturing PINN | adaptada | Tseng 2023, arXiv:2210.08424 |
| DCSNN | adaptada | Hu, Lin y Lai 2022, 10.1016/j.jcp.2022.111576 |
| Contacto imperfecto | adaptada | Oh y Jo 2025, 10.3934/math.2025364 |
| FBPINN | adaptada | Moseley 2023, 10.1007/s10444-023-10065-9 |
| SA-PINN y RBA | adaptada | McClenny arXiv:2009.04544; Ramirez arXiv:2405.06443 |
| VPINN y hp-VPINN | opcional | - |
| DeepONet, PINO y transitorio | fuera del alcance de la etapa 2 | - |

Brecha empirica heredada (ESTADO.md, 2026-09-29): casos homogeneos 3 de 3; backfill, discrete_layers, aras_flat y kim_sand 0 de 3.

## 6. Contrato fisico no negociable (se verifica con pruebas)

-div(k grad T)=Q explicita en cada capa; sin resistencias equivalentes, sin reconstruccion radial de T_conductor ni flujo prescrito en la superficie del cable; contacto perfecto (T y flujo normal continuos), imperfecto solo con R'' documentado; Dirichlet, Neumann y Robin (Robin nunca como Dirichlet); Q=p' dividido por A con comprobacion de la integral de Q dA; adimensionalizacion, float64 de referencia; k nunca confundida con resistividad; datos de articulos nunca como etiquetas.

## 7. Arquitectura destino

    PINN_v2
      pyproject.toml, REFACTOR_PLAN.md, IMPLEMENTATION_LOG.md
      pinn_v2
        cases        CaseDefinition v2, validacion, catalogo heredado, proveniencia
        geometry     Region, Interface, Domain (numpy puro; tambien lo importa el runner FEM)
        physics      expressions, skin_effect, operators (torch)
        sampling     politicas de colocacion
        solvers.pinn modelos M0, M1, M2 y variantes, perdidas, entrenamiento
        solvers.fem  runner (solo WSL) y adapter (Windows)
        evaluation   metricas comunes, balance energetico independiente
        paper_data   lectores de solo lectura
        experiments  benchmarks V0-V5, ampacidad
        reports      solo desde manifiestos
        manifest     manifiestos, no-overwrite, escritura atomica
      tests

## 8. Fases y criterios de salida

| Fase | Contenido | Salida |
|------|-----------|--------|
| 0 | Auditoria | este documento y bitacora |
| 1 | Copia de modulos y pruebas de caracterizacion | huellas, areas, MMS, integral de Q, normales, dc y piel, determinismo, no-overwrite |
| 2 | CaseDefinition v2 (Neumann, Robin, proveniencia, hash, validacion dimensional, rechazo de faltantes) | pruebas de esquema (completada) |
| 3 | Referencia FEM V0-V2, 3 mallas o mas, via WSL (completada) | JSON y NPZ con manifiesto |
| 4 | M0, M1 y M2 bajo una interfaz | resultados V0-V2 (completada; results/phase4) |
| 5 | Variantes nuevas una a una, con DOI y pruebas | tabla fiel, inspirada, adaptada (completada; CUSP, DCSNN, FBPINN, RBA y SA-PINN adaptadas, DOI por verificar; I-PINN, AdaI-PINN y VPINN NO implementadas) |
| 6 | Escalera V3-V5 (Kim V5: Robin superior, T=15.2 C inferior) | completada: FEM en results/phase6/p6_fem3; M1 corto y largo en p6_pinn_m1, p6_pinn_m1_long, p6_pinn_kim_long (aras_flat aceptado; xlpe_backfill, xlpe_discrete_layers y kim_v5_summer rechazados); CLI, busqueda de hiperparametros, docs HTML y front-end web listos |
| 7 | Campanas y reportes desde manifiestos | completada: pinn_v2/experiments/aggregate.py -> results/phase7/aggregate (summary.md, all_runs.csv; 79 corridas) |
| 8 | Cierre: suite completa, linea base 366 intacta | completada: PINN_v2 110 pruebas (incl. slow y fem) pasan; linea base 366 pasadas, 3 deseleccionadas; git status solo PINN_v2 sin seguimiento |

## 9. Riesgos

- Kim V5 en el dominio real (45.5 m de semiancho) es costoso; se documentara la reduccion de dominio y su efecto.
- Los PINN heredados fallan en casos heterogeneos (0 de 3): no se prometen mejoras; los fallos se reportan.
- DOI del docx sin verificar: marcar por verificar.
- Windows CP1252: evitar caracteres no ASCII en salidas por consola.

## 10. Escalera de benchmarks V0-V5 (registro por benchmark)

Criterios de aceptacion: propuestos, heredados de los gates de `explicit-heat-2d-v1` (NRMSE global <= 5 %, error relativo de dT <= 5 %, RMSE por region <= 5 %, balance <= 2 %, salto de T en interfaz <= 0.1 K, salto de flujo <= 1 %); se confirman con el usuario antes de la Fase 6. Un FEM solo cuenta como referencia si es convergente (3 mallas o mas) con etiquetas verificadas.

| Id | Problema | Datos | Tipo de reproduccion | Referencia | Solvers aplicables | Estado FEM |
|----|----------|-------|----------------------|------------|--------------------|------------|
| V0 | MMS homogeneo en el cuadrado unidad (`mms_constant`; tambien `mms_robin`) | completos | exacta (solucion fabricada) | analitica | FEM, M0, M1, M2 | verificado, orden 3.00 (Fase 3) |
| V1 | Interfaz plana, T continua, gradiente discontinuo (`mms_interface`) | completos | exacta | analitica | FEM, M0, M1, M2, I-PINN, cusp | verificado, orden 3.00 (Fase 3) |
| V2 | Conductor con generacion y anillos concentricos (`coaxial_full`, `coaxial_robin`) | completos | exacta | `radial_exact` | FEM, M0, M1, M2 | verificado, rmse 6e-6 K (Fase 3) |
| V3 | Cable multicapa 2D en suelo homogeneo (`xlpe_single`) | completos (adaptacion heredada) | reconstruccion documentada | solo FEM convergente | todos | convergente, 4 mallas (Fase 6, p6_fem3) |
| V4 | Relleno, estratos, inclusion o heterogeneidad fuerte (`backfill`, `discrete_layers`, `aras_flat`) | completos (adaptacion heredada) | contraste cualitativo/reconstruccion | solo FEM convergente con estudio de dominio | todos | convergente; aras_flat sensible al dominio (3.48 % del ascenso) (Fase 6) |
| V5 | Caso publicado suficientemente especificado: Kim 2025 verano (`kim_v5_summer`) | parcial: invierno sin T_aire; dominio reducido 8 x 4 m | reconstruccion documentada con dominio reducido | FEM convergente; articulo solo como contraste | todos | convergente en niveles 0-2 (nivel 3 agota memoria WSL); muy sensible al dominio (11.9 % del ascenso) (Fase 6) |

Lecciones de la Fase 3 que aplican a V3-V5: el balance por traccion no cierra en esquinas Dirichlet-Robin (usar salida constitutiva y balance de forma debil); la simetria mediante un lado Neumann se verifica contra el dominio completo.
