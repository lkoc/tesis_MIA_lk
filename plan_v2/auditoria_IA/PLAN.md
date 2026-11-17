---
goal: Auditoria reproducible de documentos con sloptotal local
version: 1.0
date_created: 2026-11-13
last_updated: 2026-11-13
owner: GitHub Copilot
status: In progress
tags: [audit, documents, notebook, sloptotal]
---

# Introduction

Estado: In progress. Implementar un flujo general dentro de esta carpeta, sin
modificar documentos de entrada. Las puntuaciones describen el comportamiento
del detector; no prueban autoria ni miden el porcentaje de palabras creadas por IA.

## 1. Requirements & Constraints

- **REQ-001**: Leer PDF con texto, TXT, Markdown y DOCX. Rechazar DOC antiguo con instrucciones de conversion; no tratar binarios como texto.
- **REQ-002**: Conservar todo el texto por defecto y registrar exclusiones solo cuando el usuario las configure. Detectar encabezados y permitir segmentacion por palabras, sin capitulos ni rutas especificos del plan.
- **REQ-003**: Evaluar texto completo, fragmentos o ambos; guardar entrada exacta, ubicaciones, hashes, respuestas crudas, errores y cobertura.
- **REQ-004**: Cuaderno con parametros iniciales comentados, diagnostico Docker, arranque del contenedor existente, previsualizacion y generacion del Markdown.
- **SEC-001**: Solo API loopback, sin redirecciones externas; no descargar modelos, desactivar TLS, cambiar certificados ni recrear/reiniciar contenedores automaticamente.
- **CON-001**: No inventar puntuaciones ni atribuir causalidad a correlaciones editoriales. No promediar fragmentos para fabricar una puntuacion global.
- **PAT-001**: Separar extraccion/segmentacion en documentos.py; API, ejecucion e informe en auditar_plan.py; cuaderno como interfaz de los mismos componentes.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Definir contratos y verificar fuentes antes de implementar.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Inspeccionar script y definir arquitectura y contratos en este documento. | Si | 2026-11-13 |
| TASK-002 | Consultar API local y fuente oficial para confirmar endpoints, limites y campos de resultados. | No | |

### Implementation Phase 2

- GOAL-002: Implementar componentes independientes con agentes de menor capacidad.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-003 | Agente lector: crear documentos.py y test_documentos.py, validar conservacion de palabras y limites de fragmento. | No | |
| TASK-004 | Agente cuaderno: crear auditoria_sloptotal.ipynb, README.md y actualizar requirements.txt segun contrato. | No | |
| TASK-005 | Integrador: sustituir auditar_plan.py por CLI generica, diagnostico, cliente API, persistencia e informe. | No | |

### Implementation Phase 3

- GOAL-003: Verificar flujo y dejar evidencia real, distinguiendo bloqueos externos.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-006 | Ejecutar pruebas de formatos, chunking, respuesta API y errores usando fixtures locales y servidor simulado. | No | |
| TASK-007 | Ejecutar cuaderno con PDF real y diagnostico local; intentar puntuaciones solo dentro de limites de tiempo y sin saturar colas. | No | |
| TASK-008 | Revisar informe Markdown, cobertura y hashes de entrada; actualizar estado y explicar requisitos no verificados. | No | |

## Arquitectura Y Contratos

`documentos.py` expone `extract_document(path: Path) -> Document`,
`make_chunks(document: Document, max_words=400, max_chars=5000, mode='structure') -> list[TextChunk]`.
`Document` tiene `text: str`, `blocks: list[TextBlock]`, `metadata: dict`, `warnings: list[str]`.
`TextBlock` tiene `text: str`, `location: str`, `section: str`, `heading: bool`.
`TextChunk` tiene `chunk_id: str`, `text: str`, `locations: list[str]`, `sections: list[str]`.
Sin solapamiento por defecto: concatenar palabras de chunks debe reconstruir las de Document.text.
Si una seccion excede los limites se divide sin cortar palabras, salvo tokens individuales que
excedan max_chars (error explicito). Los titulos nunca se eliminan del texto.

`auditar_plan.py` expone `AuditConfig`, `prepare_document(config) -> (Document, list[TextChunk])`,
`diagnose_service(config) -> dict`, `run_audit(config, document=None, chunks=None, diagnostics=None) -> Path`.
Configuracion: `input_path`, `output_dir`, `api_url='http://127.0.0.1:8000'`,
`container_name='sloptotal'`, `start_container=True`, `startup_timeout=60`,
`mode='both'` (whole/chunks/both), `chunk_mode='structure'` (structure/words),
`max_words=400`, `max_chars=5000`, `request_timeout=120`, `review_threshold=65.0`,
`dry_run=False`, `max_chunks=None` (sin limite), `stop_after_timeouts=1`.
El cuaderno puede fijar la ruta del PDF del usuario como ejemplo en sus parametros, pero el
script no debe contenerla. CLI: `python auditar_plan.py INPUT --output-dir OUTPUT [--dry-run]`.
Salidas por ejecucion aislada: `informe.md`, `manifest.json`, `texto_extraido.txt`,
`fragmentos.json`, `resultados.json`, `raw/*.json`; puntero `ultimo_informe.txt` en output_dir.

## 3. Alternatives

- **ALT-001**: Analizar LaTeX crudo; descartado por mezclar comandos y prosa. Para LaTeX se recomienda compilar a PDF y auditar ese PDF.
- **ALT-002**: Usar puntuacion rapida para chunks y completa para global; descartado por cambiar de motores. Se usa el mismo endpoint de analisis para ambos, secuencialmente.
- **ALT-003**: Excluir automaticamente referencias/portada; descartado por alterar silenciosamente el alcance y requerir supuestos sobre cada documento.

## 4. Dependencies

- **DEP-001**: Python 3.10+, PyMuPDF para PDF, python-docx para Word, requests para API; nbformat/ipykernel/nbclient para cuaderno y verificacion.
- **DEP-002**: Docker con contenedor sloptotal existente; API alcanzable y motores cargados para obtener puntuaciones reales.

## 5. Files

- **FILE-001**: auditar_plan.py, entrada generica conservando el nombre existente.
- **FILE-002**: documentos.py, parsers y segmentacion.
- **FILE-003**: auditoria_sloptotal.ipynb, interfaz repetible.
- **FILE-004**: test_documentos.py y test_auditar_plan.py, pruebas focalizadas.
- **FILE-005**: README.md y requirements.txt, uso y dependencias.

## 6. Testing

- **TEST-001**: Fixtures TXT/MD/PDF/DOCX sin capitulos prefijados; estructura y texto conservados; entradas vacias/formatos no soportados.
- **TEST-002**: Fragmentos cumplen limites, tienen ubicaciones y conservan palabras sin duplicados.
- **TEST-003**: Una puntuacion de un motor no se interpreta como global; cero es valido, pendientes/errores no cuentan como completados; timeouts no reenvian POST automaticamente.
- **TEST-004**: Cuaderno valido y ejecutable de principio a fin; modo seco no envia texto; informe registra cobertura real y limitaciones.

## 7. Risks & Assumptions

- **RISK-001**: API puede responder health healthy con modelos fallidos o solicitudes bloqueadas. Registrar diagnostico y cortar nuevas solicitudes tras timeout.
- **RISK-002**: Detectores pueden truncar tokens internamente. Registrar limites conocidos y no llamar evaluacion integral a una cobertura interna desconocida.
- **RISK-003**: PDF puede mezclar formulas/tablas/orden de columnas o carecer de OCR. Advertir y permitir inspeccion antes del envio.
- **ASSUMPTION-001**: El usuario autoriza enviar su texto a su instancia local, pero no modificarlo ni descargar modelos por rutas alternativas a politicas de red.

## 8. Related Specifications / Further Reading

- https://github.com/pablocaeg/sloptotal
- https://github.com/pablocaeg/sloptotal/blob/master/examples/python_client.py
- Contrato instalado: http://127.0.0.1:8000/openapi.json

## Auditoria Publica Autorizada: 17 De Noviembre De 2026

El usuario autorizo el envio de la version final del plan a SlopTotal. Este flujo
es independiente del cliente local anterior: no modifica su restriccion de API
loopback ni intenta sortear el bloqueo del acceso programatico al portal.

- Entradas y evidencias: `auditoria_resultados/publico_20261117/`.
- Fuente de progreso: `manifest.json`; conservar tickets y huellas al reanudar.
- `recolector_publico.py` recibe capturas exclusivamente en `127.0.0.1`; no envia
	documentos ni solicita analisis al portal. Usa solo la biblioteca estandar.
- `secuencia_navegador.js` se ejecuta con el objeto `page` de las herramientas
	Playwright del navegador integrado. Usa el formulario normal y espera su
	resultado antes de enviar otra unidad. No responde al cuestionario de autoria.
- Inicio del recolector: `python recolector_publico.py --port 8767`.
- La secuencia debe recibir ese mismo origen como segundo argumento:
	`secuenciaAuditoria(page, "http://127.0.0.1:8767", 6)`.
- Un `envio_iniciado` sin ticket requiere inspeccionar la pagina antes de
	reanudar: no equivale a una solicitud rechazada ni autoriza a reenviarla.
- La captura guarda DOM, tabla de motores, fecha, URL y JSON exportado. No se
	fusionan puntuaciones distintas entre pantalla y exportacion.
- El consolidado se actualiza al recibir cada evento. El recolector y la pagina
	deben seguir abiertos mientras se ejecuta la secuencia; no es una tarea
	permanente independiente del navegador. Se detiene ante errores y limita a
	sesenta minutos la espera de finalizacion por unidad, sin reenviar por demora.

Comprobaciones del flujo publico: prueba focalizada de reanudacion y rechazo de
duplicados en `test_auditoria.py`; huellas de las ocho entradas comprobadas;
conexion navegador--recolector y guardado real del capitulo II verificados. El
entorno `.venv-benchmarks-repro` no tiene `requests`, por lo que cinco pruebas del
cliente local anterior fallan por importacion; no se modificaron sus dependencias.
No hay Node.js disponible en PATH: la sintaxis de la secuencia se comprobo con el
motor JavaScript del navegador integrado y los diagnosticos del editor.