"""Configuración e interfaz de auditoría con contrato estandarizado.

Proporciona AuditConfig y funciones de alto nivel que envuelven auditar_plan.py
para normalizar el flujo de auditoría desde notebooks y otros clientes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from documentos import Document, TextChunk, extract_document, make_chunks


@dataclass
class AuditConfig:
    """Configuración reproducible de auditoría de documentos.

    Atributos:
        input_path: ruta del PDF a auditar (absoluta o relativa al directorio actual)
        output_dir: directorio de salida para reportes y artefactos
        api_url: URL de la API sloptotal local (default: http://127.0.0.1:8000)
        container_name: nombre del contenedor Docker (default: sloptotal)
        start_container: iniciar contenedor si está detenido (default: True)
        startup_timeout: segundos de espera para que el contenedor responda (default: 60)
        mode: 'whole' (global), 'chunks' (fragmentos), 'both' (ambos) (default: 'both')
        chunk_mode: 'structure' (párrafos), 'words' (por palabras) (default: 'structure')
        max_words: máximo de palabras por fragmento para chunking por palabras (default: 400)
        max_chars: máximo de caracteres por fragmento (default: 5000)
        request_timeout: segundos de timeout para requests HTTP (default: 300)
        review_threshold: puntuación mínima para priorizar lectura humana (default: 65.0)
        dry_run: True no envía requests ni inicia Docker; solo valida config (default: False)
        max_chunks: máximo de fragmentos a procesar; None procesa todos (default: None)
        stop_after_timeouts: detener auditoría después de N timeouts consecutivos (default: 1)
    """
    input_path: Path | str
    output_dir: Path | str
    api_url: str = "http://127.0.0.1:8000"
    container_name: str = "sloptotal"
    start_container: bool = True
    startup_timeout: int = 60
    mode: str = "both"  # 'whole', 'chunks', 'both'
    chunk_mode: str = "structure"  # 'structure', 'words'
    max_words: int = 400
    max_chars: int = 5000
    request_timeout: int = 300
    report_timeout: int = 900
    review_threshold: float = 65.0
    dry_run: bool = False
    max_chunks: int | None = None
    stop_after_timeouts: int = 1

    def __post_init__(self):
        """Normalizar rutas a Path absoluto."""
        if self.mode not in {"whole", "chunks", "both"}:
            raise ValueError("mode debe ser whole, chunks o both")
        if self.chunk_mode not in {"structure", "words"}:
            raise ValueError("chunk_mode debe ser structure o words")
        for name in ("max_words", "max_chars", "startup_timeout", "request_timeout", "report_timeout", "stop_after_timeouts"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo")
        if self.max_chunks is not None and self.max_chunks <= 0:
            raise ValueError("max_chunks debe ser positivo o None")
        if not 0 <= self.review_threshold <= 100:
            raise ValueError("review_threshold debe estar entre 0 y 100")
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if not self.input_path.is_absolute():
            self.input_path = self.input_path.resolve()
        if not self.output_dir.is_absolute():
            self.output_dir = self.output_dir.resolve()

    def to_dict(self) -> dict[str, Any]:
        """Convertir a diccionario para auditar_plan.run_audit()."""
        d = asdict(self)
        d["input_path"] = str(self.input_path)
        d["output_dir"] = str(self.output_dir)
        return d


def prepare_document(config: AuditConfig) -> tuple[Document, list[TextChunk]]:
    """Extraer un documento compatible y segmentar segun configuracion.

    Args:
        config: configuración de auditoría

    Returns:
        tupla (Document, lista de TextChunk)

    Raises:
        FileNotFoundError: si el PDF no existe
        ValueError: si no se puede extraer texto del PDF
    """
    if not config.input_path.is_file():
        raise FileNotFoundError(f"No se encontró documento: {config.input_path}")

    document = extract_document(config.input_path)
    if not document.text.strip():
        raise ValueError("El documento no contiene texto extraible; revisar si requiere OCR.")
    chunks = make_chunks(document, config.max_words, config.max_chars, config.chunk_mode)
    document.metadata["audit_total_chunks"] = len(chunks)

    if config.max_chunks is not None:
        chunks = chunks[: config.max_chunks]

    return document, chunks


def diagnose_service(config: AuditConfig) -> dict[str, Any] | None:
    """Diagnosticar estado de la API y contenedor Docker.

    Args:
        config: configuración de auditoría

    Returns:
        diccionario con diagnóstico completo (docker + health), o None si dry_run=True
    """
    if config.dry_run:
        return None

    import auditar_plan

    api_url = auditar_plan.ensure_local_url(config.api_url)
    docker, health = auditar_plan.diagnose_service(
        api_url,
        config.container_name,
        config.start_container,
        config.startup_timeout,
    )

    result = {"docker": docker, "health": health}
    return result


def run_audit(
    config: AuditConfig,
    document: Document | None = None,
    chunks: list[TextChunk] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> Path:
    """Ejecutar auditoría de documento.

    Si document y chunks no se proporcionan, se extraen del config.input_path.
    El resultado incluye:
      - Reporte Markdown: auditoria_resultado_<timestamp>.md
      - Copia actual del reporte: auditoria_resultado_actual.md
      - JSON de salud (si disponible)
      - JSON de análisis global y fragmentos
      - Índice de fragmentos y texto extraído

    Args:
        config: configuración de auditoría
        document: Document pre-extraído (opcional)
        chunks: TextChunk pre-segmentados (opcional)
        diagnostics: diagnóstico pre-obtenido (opcional)

    Returns:
        ruta del archivo de reporte Markdown generado

    Raises:
        FileNotFoundError: si no se puede encontrar el PDF
        ValueError: si no se puede extraer texto o diagnosticar el servicio
    """
    import auditar_plan

    # Preparar documento si no se proporciona
    if document is None or chunks is None:
        document, chunks = prepare_document(config)

    # Diagnosticar si no se proporciona diagnóstico
    if diagnostics is None and not config.dry_run:
        diagnostics = diagnose_service(config)

    return auditar_plan.run_audit(config.to_dict(), document, chunks, diagnostics)
