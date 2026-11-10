# Estado del proyecto (fuente única de cifras)

Actualizado el 29-sep-2026. Los demás documentos deben remitirse a este archivo en lugar de repetir cifras.
Estas cifras se registraron a mano a partir de los resultados y de las compilaciones en copia temporal; un generador automático queda pendiente (R-39).
Decisiones y pendientes: [PENDIENTES.md](Tesis_LaTeX_Borrador_UNI/PENDIENTES.md). Auditoría: [AUDITORIA_GENERAL_2026-09-29.md](auditoria%20general/AUDITORIA_GENERAL_2026-09-29.md).

## Documentos

| Documento | Estado | Cifra |
|---|---|---|
| Tesis (`Tesis_LaTeX_Borrador_UNI/tesis.tex`) | Compila en copia temporal sin referencias ni citas indefinidas | 88 páginas con interlineado `\setstretch{1.4}` (82 con `\onehalfspacing`); límite institucional: 130 |
| Plan (`Plan/plan_tesis_cables_pinn.tex` y `Plan/capitulos/`) | Estructura de la Guía N.° 01 (Capítulos I a V, Referencias y Anexos); compila sin referencias ni citas indefinidas | 76 páginas |
| Título vigente | El más reciente (D3); **no enviado a la Unidad de Posgrado** | Carátulas y metadatos de tesis y Plan |

## Campaña explícita (etapas A a D)

| Etapa | Entrenamientos | Contenido |
|---|---|---|
| A | 36 | Arquitectura y tamaño |
| B | 24 | Número y localización de colocaciones |
| C | 12 | Tasa de aprendizaje |
| C2 | 8 nuevos (más 4 reutilizados de C) | Peso de continuidad térmica |
| D | 21 | Confirmación con semillas nuevas 71, 83 y 97 |
| **Total** | **101** | Suma de tiempos de corrida: 15,0 h (8,0 h en D) |

Confirmación D: **7 de 21** ejecuciones aceptadas.

| Problema | Aceptadas |
|---|---|
| `xlpe_single` | 3/3 |
| `xlpe_dry_far` | 3/3 |
| `xlpe_dry_near` | 1/3 (restringido) |
| `xlpe_backfill` | 0/3 |
| `xlpe_discrete_layers` | 0/3 |
| `aras_flat` | 0/3 |
| `kim_sand` | 0/3 |

Búsqueda de corriente límite con la formulación explícita: **no ejecutada** (9 corridas previstas). Ronda E: **propuesta, no autorizada ni ejecutada** ([protocolo](docs/PROTOCOLO_RONDA_E.md)).

## Pruebas y bibliografía

- Pruebas recolectadas: 366 (307 + 59) según la auditoría general; `pytest` en la raíz no ejecuta ambos conjuntos (R-16 pendiente).
- Bibliografía del Plan: 49 entradas citadas en `referencias_plan_tesis.bib`; las 23 no citadas están en `referencias_plan_tesis_no_usadas.bib`.

## Advertencia

`Benchmarks/full_study_report.py` reescribe archivos de la tesis con ediciones manuales (véase PENDIENTES.md). No ejecutarlo sin actualizarlo antes.
