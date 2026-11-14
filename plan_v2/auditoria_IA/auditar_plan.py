"""Generic local sloptotal client, live progress and traceable Markdown reports."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.parse import quote, urlparse

import requests

from documentos import Document, TextChunk


def ensure_local_url(api_url: str) -> str:
    parsed = urlparse(api_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Solo se permite una API local para proteger el documento.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("La URL local no debe contener credenciales, query ni fragmento.")
    return api_url.rstrip("/")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def get_json(api_url: str, endpoint: str, timeout: int = 15) -> Any:
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f"{api_url}{endpoint}", timeout=(5, timeout))
        response.raise_for_status()
        return response.json()


def post_json(api_url: str, endpoint: str, payload: dict[str, Any], timeout: int) -> Any:
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(f"{api_url}{endpoint}", json=payload, timeout=(10, timeout))
        response.raise_for_status()
        return response.json()


def docker_state(container_name: str) -> dict[str, Any]:
    try:
        result = subprocess.run(["docker", "inspect", container_name], capture_output=True, text=True, timeout=15)
        if result.returncode:
            return {"exists": False, "error": result.stderr.strip()}
        container = json.loads(result.stdout)[0]
        return {"exists": True, "state": container["State"], "image": container["Config"]["Image"], "ports": container["NetworkSettings"]["Ports"]}
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as error:
        return {"exists": False, "error": str(error)}


def diagnose_service(api_url: str, container_name: str, start_if_stopped: bool, startup_timeout: int) -> tuple[dict[str, Any], Any]:
    api_url = ensure_local_url(api_url)
    docker = docker_state(container_name)
    if docker.get("exists") and not docker.get("state", {}).get("Running") and start_if_stopped:
        try:
            started = subprocess.run(["docker", "start", container_name], capture_output=True, text=True, timeout=60)
            docker["start_exit_code"] = started.returncode
            docker["start_output"] = (started.stdout + started.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            docker["start_error"] = str(error)
    deadline = time.monotonic() + startup_timeout
    while True:
        try:
            health = get_json(api_url, "/health", timeout=5)
            engines = health.get("engines", {}) if isinstance(health, dict) else {}
            print(f"sloptotal: {engines.get('loaded', '?')}/{engines.get('total', '?')} motores cargados.", flush=True)
            return docker, health
        except (requests.RequestException, ValueError) as error:
            docker["health_error"] = str(error)
            if time.monotonic() >= deadline:
                return docker, None
            time.sleep(min(2, max(0, deadline - time.monotonic())))


def report_body(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if "overall_score" in payload:
        return payload
    for name in ("report", "result", "data"):
        nested = payload.get(name)
        if isinstance(nested, dict):
            body = report_body(nested)
            if body:
                return body
    return payload


def find_score(payload: Any) -> float | None:
    value = report_body(payload).get("overall_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) and 0 <= score <= 100 else None


def wait_for_report(api_url: str, result: Any, timeout: int) -> Any:
    deadline = time.monotonic() + timeout
    latest = result
    while isinstance(latest, dict):
        body = report_body(latest)
        status = str(body.get("status", latest.get("status", ""))).lower()
        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"sloptotal devolvio estado {status}: {body.get('error', body.get('detail', ''))}")
        if find_score(latest) is not None and status not in {"queued", "pending", "processing", "running", "in_progress"}:
            return latest
        ticket = latest.get("ticket_id")
        report_id = latest.get("report_id") or body.get("report_id") or body.get("id")
        if status in {"completed", "complete", "done"} and find_score(latest) is None and not ticket:
            return latest
        if not ticket and not report_id:
            return latest
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise requests.Timeout("Tiempo de espera agotado para el informe en cola.")
        print(f"  Esperando sloptotal ({status or 'pendiente'}); no se vuelve a enviar el texto.", flush=True)
        time.sleep(min(3, remaining))
        endpoint = f"/api/queue/ticket/{quote(str(ticket), safe='')}" if ticket else f"/api/report/{quote(str(report_id), safe='')}"
        latest = get_json(api_url, endpoint, timeout=max(1, min(15, int(remaining))))
    return latest


def heuristic_comments(text: str) -> list[str]:
    comments = []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    long_count = sum(len(sentence.split()) >= 45 for sentence in sentences)
    if long_count:
        comments.append(f"Revisar carga sintactica: {long_count} oracion(es) de al menos 45 palabras; segmentacion aproximada.")
    if len(text.split()) < 150:
        comments.append("Fragmento corto: la puntuacion puede ser inestable por falta de contexto.")
    if text.count("\n") > max(5, len(text.split()) // 12):
        comments.append("Muchos saltos de linea: comprobar tablas, formulas y orden de extraccion antes de interpretar.")
    if not comments:
        comments.append("Sin alerta editorial automatica; comprobar precision, fuentes y coherencia en contexto.")
    return comments


def engine_rows(payload: Any) -> list[tuple[str, Any, Any]]:
    body = report_body(payload)
    values = body.get("engine_results", body.get("engines", []))
    rows = []
    if isinstance(values, dict):
        values = [dict(value, engine=name) if isinstance(value, dict) else {"engine": name, "score": value} for name, value in values.items()]
    if isinstance(values, list):
        for value in values:
            if isinstance(value, dict):
                rows.append((str(value.get("engine", value.get("name", "motor"))), value.get("score"), value.get("error", value.get("status", ""))))
    return rows


def md_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(config: dict[str, Any], document: Document, chunks: list[TextChunk], diagnostics: Any, results: list[dict[str, Any]]) -> str:
    scored = [item for item in results if item["score"] is not None]
    errors = [item for item in results if item["status"] in {"error", "timeout", "no_score"}]
    lines = ["# Auditoria de documento con sloptotal", "", f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}", f"Documento: {config['input_path']}", "", "## Alcance y limitaciones", "", f"Tareas seleccionadas: {len(results)}. Con puntuacion: {len(scored)}. Con error/sin puntuacion: {len(errors)}.", f"Fragmentos seleccionados: {len(chunks)} de {document.metadata.get('audit_total_chunks', len(chunks))}.", "La puntuacion 0-100 del detector NO es el porcentaje de palabras creadas por IA ni prueba de autoria.", "Documento global y fragmentos usan /api/analyze; sus puntuaciones no se suman ni son directamente equivalentes.", "Los comentarios editoriales son heuristicas independientes: no explican causalmente la puntuacion.", "No se modifica ni se reescribe la fuente. No hay calibracion propia para espanol academico ni umbral garantizado.", ""]
    health = (diagnostics or {}).get("health") or {}
    engines = health.get("engines", {})
    lines.extend(["## Servicio", "", f"Motores cargados: {engines.get('loaded', 'desconocido')}/{engines.get('total', 'desconocido')}.", f"Motores fallidos: {', '.join(engines.get('failed', [])) or 'ninguno reportado / desconocido'}.", "Una cobertura incompleta puede cambiar la puntuacion respecto a una instalacion completa.", "", "## Extraccion", "", f"Formato: {document.metadata.get('format')}. Palabras: {len(document.text.split())}. Caracteres: {len(document.text)}."])
    lines.extend(f"- {warning}" for warning in document.warnings)
    lines.extend(["", "## Resultados", "", "| Tarea | Ubicacion | Estado | Score 0-100 | Atencion |", "| --- | --- | --- | --- | --- |"])
    for item in results:
        attention = "revisar en contexto" if item["score"] is not None and item["score"] >= config["review_threshold"] else "sin prioridad por score"
        lines.append(f"| {item['id']} | {md_text(', '.join(item['locations']))} | {item['status']} | {item['score'] if item['score'] is not None else 'no disponible'} | {attention} |")
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    for item in results:
        lines.extend(["", f"## {item['id']}", "", f"Estado: {item['status']}. Score: {item['score'] if item['score'] is not None else 'no disponible'}.", f"Secciones: {md_text(', '.join(item['sections'])) or 'no detectadas'}."])
        if item.get("error"):
            lines.append(f"Error: {item['error']}")
        lines.extend(f"- {comment}" for comment in item["comments"])
        rows = engine_rows(item.get("response"))
        if rows:
            lines.extend(["", "| Motor | Score individual (escala original) | Estado/error |", "| --- | --- | --- |"])
            lines.extend(f"| {md_text(name)} | {score} | {md_text(status)} |" for name, score, status in rows)
        if item["id"] in chunk_map:
            lines.extend(["", "Texto analizado:", ""])
            lines.extend("> " + line for line in chunk_map[item["id"]].text.splitlines())
    lines.extend(["", "## Metodo de revision humana", "", "1. Verificar el texto extraido y sus ubicaciones contra el documento original.", "2. Leer los fragmentos priorizados con su contexto; conservar citas, formulas y significado tecnico.", "3. Revisar claridad, respaldo documental y consistencia, sin alterar estilo solo para cambiar un score.", "4. Registrar decisiones humanas y versiones. Evaluar falsos positivos con textos humanos comparables, no con un score aislado.", "", "Configuracion, hashes, texto extraido y respuestas originales estan junto a este informe.", ""])
    return "\n".join(lines)


def run_audit(config: dict[str, Any], document: Document, chunks: list[TextChunk], diagnostics: Any = None) -> Path:
    api_url = ensure_local_url(config["api_url"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "diagnostics.json", diagnostics)
    write_json(run_dir / "chunks.json", [asdict(chunk) for chunk in chunks])
    write_json(run_dir / "document.json", {"metadata": document.metadata, "warnings": document.warnings, "source_sha256": hashlib.sha256(Path(config["input_path"]).read_bytes()).hexdigest(), "text_sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest()})
    (run_dir / "texto_extraido.txt").write_text(document.text, encoding="utf-8")
    tasks = []
    if config["mode"] in {"whole", "both"}:
        tasks.append(("global", document.text, ["documento completo"], []))
    if config["mode"] in {"chunks", "both"}:
        tasks.extend((chunk.chunk_id, chunk.text, chunk.locations, chunk.sections) for chunk in chunks)
    results = []
    consecutive_timeouts = 0
    health = (diagnostics or {}).get("health")
    print(f"[0/{len(tasks)} 0.0%] Inicio: {len(document.text.split())} palabras; {len(chunks)} fragmentos seleccionados.", flush=True)
    for index, (task_id, text, locations, sections) in enumerate(tasks, start=1):
        item = {"id": task_id, "status": "pending", "score": None, "locations": locations, "sections": sections, "comments": heuristic_comments(text)}
        started = time.monotonic()
        if config["dry_run"]:
            item["status"] = "dry_run"
        elif not health:
            item.update(status="skipped", error="API sin diagnostico de salud utilizable.")
        elif consecutive_timeouts >= config["stop_after_timeouts"]:
            item.update(status="skipped", error="Se detuvieron nuevos envios despues de timeouts; una solicitud anterior puede seguir activa en el servidor.")
        elif len(text.strip()) < 50:
            item.update(status="skipped", error="Menos de 50 caracteres: contexto insuficiente para el endpoint.")
        else:
            print(f"  Enviando {task_id} ({len(text.split())} palabras) a /api/analyze...", flush=True)
            try:
                initial = post_json(api_url, "/api/analyze", {"text": text}, config["request_timeout"])
                write_json(run_dir / f"{task_id}_initial.json", initial)
                response = wait_for_report(api_url, initial, config.get("report_timeout", 900))
                item["response"] = response
                item["score"] = find_score(response)
                item["status"] = "completed" if item["score"] is not None else "no_score"
                consecutive_timeouts = 0
            except requests.Timeout as error:
                item.update(status="timeout", error=str(error))
                consecutive_timeouts += 1
            except (requests.RequestException, ValueError, RuntimeError) as error:
                item.update(status="error", error=str(error))
                consecutive_timeouts = 0
        item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        results.append(item)
        write_json(run_dir / f"{task_id}.json", item)
        write_json(run_dir / "results.json", results)
        score_text = f"{item['score']:.2f}/100" if item["score"] is not None else "sin puntuacion"
        priority = " ATENCION: lectura humana prioritaria." if item["score"] is not None and item["score"] >= config["review_threshold"] else ""
        print(f"[{index}/{len(tasks)} {100 * index / len(tasks):.1f}%] {task_id}: {item['status']}, {score_text}.{priority} {item['comments'][0]}", flush=True)
        if item.get("error"):
            print(f"  {item['error']}", flush=True)
    report = render_report(config, document, chunks, diagnostics, results)
    report_path = run_dir / "auditoria_resultado.md"
    report_path.write_text(report, encoding="utf-8")
    (output / "auditoria_resultado_actual.md").write_text(report, encoding="utf-8")
    successful = sum(item["score"] is not None for item in results)
    print(f"Finalizado: {successful}/{len(tasks)} tareas con puntuacion. Informe: {report_path}", flush=True)
    return report_path


def main() -> None:
    from auditoria_config import AuditConfig, run_audit as audit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("auditoria_resultados"))
    parser.add_argument("--mode", choices=["whole", "chunks", "both"], default="both")
    parser.add_argument("--chunk-mode", choices=["structure", "words"], default="structure")
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument("--max-chars", type=int, default=5000)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--report-timeout", type=int, default=900)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--container-name", default="sloptotal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    audit(AuditConfig(**vars(args)))


if __name__ == "__main__":
    main()