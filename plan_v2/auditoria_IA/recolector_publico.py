"""Recibe evidencias del navegador en loopback; nunca envia solicitudes al portal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse


ORDER = (
    "01_planteamiento", "02_objetivos", "03_marco_teorico", "04_metodologia",
    "05_administracion", "00_preliminares", "06_referencias", "07_anexos",
)


def load_manifest(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


def normalized_text(folder: Path, unit: dict) -> str:
    source = (folder / unit["archivo"]).read_bytes()
    if hashlib.sha256(source).hexdigest() != unit["sha256"]:
        raise ValueError(f"La entrada cambio: {unit['id']}")
    return source.decode("utf-8").replace("\r\n", "\n").strip()


def next_job(folder: Path) -> dict:
    manifest = load_manifest(folder)
    if manifest.get("cierre_auditoria", {}).get("nuevos_envios_autorizados") is False:
        return {"done": True, "closed": True}
    for identifier in manifest.get("orden_ejecucion", ORDER):
        unit = next(unit for unit in manifest["unidades"] if unit["id"] == identifier)
        if unit["estado"].startswith("informe_disponible"):
            continue
        if unit["estado"] == "envio_iniciado":
            raise ValueError(f"Envio de resultado desconocido: {identifier}; no reenviar")
        text = normalized_text(folder, unit)
        return {**unit, "texto_formulario": text,
                "sha256_formulario_esperado": hashlib.sha256(text.encode()).hexdigest()}
    return {"done": True}


def save_manifest(folder: Path, manifest: dict) -> None:
    temporary = folder / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(folder / "manifest.json")
    lines = [
        "# Auditoria publica del plan: consolidado", "",
        f"Actualizado: {datetime.now(timezone.utc).isoformat()}.", "",
        "Las puntuaciones se transcriben del portal; no prueban autoria ni calidad academica.",
        "No se promedian capitulos para fabricar una puntuacion de todo el documento.", "",
        "| Unidad | Paginas PDF | Estado | Pantalla /100 | Exportacion /100 | Evidencia |",
        "|---|---|---|---|---|---|",
    ]
    available = 0
    for unit in manifest["unidades"]:
        ready = unit["estado"].startswith("informe_disponible")
        available += int(ready)
        evidence = unit.get("captura_navegador", unit.get("resultado_capturado"))
        link = f"[Captura]({evidence})" if evidence else "Pendiente"
        if unit.get("exportacion_original"):
            link += f"; [JSON original]({unit['exportacion_original']})"
        lines.append(
            f"| {unit['titulo']} | {unit['pagina_pdf_inicial']} a {unit['pagina_pdf_final']} "
            f"| {unit['estado']} | {unit.get('puntuacion_portal', 'No disponible')} "
            f"| {unit.get('puntuacion_exportacion', 'No disponible')} | {link} |"
        )
    lines += ["", f"Informes archivados: **{available} de {len(manifest['unidades'])}**.", "",
              "## Observaciones", ""]
    for unit in manifest["unidades"]:
        if unit.get("discrepancia_exportacion"):
            lines.append(f"- {unit['titulo']}: {unit['discrepancia_exportacion']}")
        if unit.get("url_informe"):
            lines.append(f"- {unit['titulo']}: {unit['url_informe']}")
    lines += ["", "Los textos incluyen tablas, formulas, referencias y encabezados extraidos del PDF.",
              "La cobertura interna y el truncamiento de cada motor no estan acreditados.",
              "Un estado Complete no garantiza que los 23 motores tengan puntuacion.",
              "El portal declara que elimina los informes despues de 30 dias.",
              "Las capturas historicas no se sustituyen por resultados de otra fecha.", ""]
    (folder / "informe_consolidado.md").write_text("\n".join(lines), encoding="utf-8")


def record_event(folder: Path, event: dict) -> dict:
    manifest = load_manifest(folder)
    if manifest.get("cierre_auditoria", {}).get("nuevos_envios_autorizados") is False:
        raise ValueError("La auditoria esta cerrada; no se modifican sus evidencias")
    unit = next(unit for unit in manifest["unidades"] if unit["id"] == event["id"])
    action = event["action"]
    timestamp = datetime.now(timezone.utc).isoformat()
    if action == "started":
        if unit["estado"] != "preparado_no_enviado" or unit.get("ticket"):
            raise ValueError("La unidad ya se envio o tiene un envio de estado desconocido")
        active = [other for other in manifest["unidades"]
                  if other["id"] != unit["id"] and other["estado"] in
                  {"envio_iniciado", "en_cola", "enviado_en_proceso"}]
        if active:
            raise ValueError("Ya hay una unidad pendiente; no se acumulan solicitudes")
        text = normalized_text(folder, unit)
        if hashlib.sha256(text.encode()).hexdigest() != event["sha256"]:
            raise ValueError("El texto del formulario no coincide con la entrada")
        unit.update(estado="envio_iniciado", inicio_envio_utc=timestamp,
                    sha256_texto_formulario=event["sha256"])
    elif action == "submitted":
        parsed = urlparse(event["url"])
        ticket = parse_qs(parsed.query).get("ticket", [""])[0]
        if parsed.scheme != "https" or parsed.netloc != "sloptotal.com" or not re.fullmatch(r"[a-f0-9]{16}", ticket):
            raise ValueError("URL de ticket no valida")
        if unit.get("ticket") and unit["ticket"] != ticket:
            raise ValueError("Se intento sustituir un ticket existente")
        if unit["estado"] not in {"envio_iniciado", "en_cola"}:
            raise ValueError("No consta el inicio del envio")
        unit.update(estado="en_cola", ticket=ticket, url_ticket=event["url"],
                    url_informe=event["url"], fecha_envio_utc=timestamp)
    elif action == "reused":
        parsed = urlparse(event["url"])
        report_id = parse_qs(parsed.query).get("id", [""])[0]
        if parsed.scheme != "https" or parsed.netloc != "sloptotal.com" or not re.fullmatch(r"[a-f0-9]{12}", report_id):
            raise ValueError("URL de informe reutilizado no valida")
        if unit["estado"] != "envio_iniciado":
            raise ValueError("No consta el inicio del envio")
        unit.update(estado="informe_reutilizado_pendiente_captura", id_informe_reutilizado=report_id,
                    url_informe=event["url"], respuesta_directa_utc=timestamp,
                    evaluacion_independiente_acreditada=False)
    elif action == "capture":
        if not unit.get("ticket") and not unit.get("id_informe_reutilizado"):
            raise ValueError("No consta ticket ni informe reutilizado para esta captura")
        capture = event["capture"]
        export = json.loads(capture["export_text"])
        if not re.fullmatch(r"[a-f0-9]{12}", export.get("id", "")):
            raise ValueError("Exportacion sin identificador de informe valido")
        if capture.get("report_id") != export["id"]:
            raise ValueError("La exportacion y la captura no identifican el mismo informe")
        if unit.get("id_informe_reutilizado") and unit["id_informe_reutilizado"] != export["id"]:
            raise ValueError("La captura no corresponde al informe reutilizado")
        if not capture.get("report_text") or not isinstance(export.get("overall_score"), (int, float)):
            raise ValueError("La captura no contiene un resultado")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        export_name = f"{unit['id']}_exportacion_{stamp}.json"
        capture_name = f"{unit['id']}_captura_{stamp}.json"
        (folder / export_name).write_text(capture["export_text"], encoding="utf-8")
        (folder / capture_name).write_text(json.dumps(capture, ensure_ascii=False, indent=2), encoding="utf-8")
        unit.update(estado="informe_disponible_cobertura_no_acreditada",
                    ultima_observacion_utc=timestamp, captura_navegador=capture_name,
                    exportacion_original=export_name, puntuacion_exportacion=export["overall_score"],
                    motores_exportacion=len(export.get("engine_results", [])),
                    url_informe=f"https://sloptotal.com/report/?id={export['id']}")
        if capture.get("visible_score") is not None:
            unit["puntuacion_portal"] = capture["visible_score"]
        if capture.get("visible_score") != export["overall_score"]:
            unit["discrepancia_exportacion"] = (
                f"Pantalla: {capture.get('visible_score')}; exportacion: {export['overall_score']}. "
                "Se conservan ambas; causa no establecida."
            )
    else:
        raise ValueError("Evento no permitido")
    save_manifest(folder, manifest)
    return {"ok": True, "id": unit["id"], "estado": unit["estado"]}


def serve(folder: Path, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, value: dict) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "https://sloptotal.com")
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self) -> None:
            if self.headers.get("Origin") != "https://sloptotal.com":
                self.respond(403, {"error": "Origen no permitido"})
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "https://sloptotal.com")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Audit-Local")
            self.end_headers()

        def do_GET(self) -> None:
            try:
                if self.path == "/health":
                    self.respond(200, {"ok": True, "mode": "local_evidence_only"})
                elif self.path == "/sequence":
                    source = Path(__file__).with_name("secuencia_navegador.js")
                    self.respond(200, {"code": source.read_text(encoding="utf-8")})
                elif self.path == "/next":
                    self.respond(200, next_job(folder))
                else:
                    self.respond(404, {"error": "Ruta no disponible"})
            except (ValueError, KeyError, StopIteration) as error:
                self.respond(409, {"error": str(error)})

        def do_POST(self) -> None:
            if self.path != "/event" or self.headers.get("X-Audit-Local") != "1":
                self.respond(403, {"error": "Solo eventos locales explicitos"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 2_000_000:
                self.respond(413, {"error": "Tamano no permitido"})
                return
            try:
                self.respond(200, record_event(folder, json.loads(self.rfile.read(length))))
            except (ValueError, KeyError, StopIteration) as error:
                self.respond(409, {"error": str(error)})

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Recolector local: http://127.0.0.1:{port}; sin conexiones al portal", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).parent / "auditoria_resultados/publico_20261117")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.folder)
    for unit in manifest["unidades"]:
        normalized_text(args.folder, unit)
    if args.check:
        print(f"Entradas verificadas: {len(manifest['unidades'])}; siguiente: {next_job(args.folder).get('id', 'ninguna')}")
    else:
        serve(args.folder, args.port)


if __name__ == "__main__":
    main()