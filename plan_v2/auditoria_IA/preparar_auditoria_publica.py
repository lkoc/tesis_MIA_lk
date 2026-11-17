"""Prepara entradas trazables del PDF sin enviar datos a servicios externos."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(source: Path, destination: Path) -> dict:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("Se requiere pdftotext en PATH; no se instalaran paquetes.")
    extraction = subprocess.run(
        [executable, "-layout", "-enc", "UTF-8", str(source), "-"],
        capture_output=True, check=True,
    )
    pages = extraction.stdout.decode("utf-8").split("\f")
    if not pages[-1].strip():
        pages.pop()
    headings = [
        ("01_planteamiento", r"CAPITULO\s+1\.", "Planteamiento del problema"),
        ("02_objetivos", r"CAPITULO\s+2\.", "Objetivos de la investigacion"),
        ("03_marco_teorico", r"CAPITULO\s+3\.", "Marco teorico"),
        ("04_metodologia", r"CAPITULO\s+4\.", "Metodologia de la investigacion"),
        ("05_administracion", r"CAPITULO\s+5\.", "Administracion del plan de tesis"),
        ("06_referencias", r"(?m)^[ \t]*REFERENCIAS[ \t]*\r?$", "Referencias"),
        ("07_anexos", r"(?m)^[ \t]*ANEXOS[ \t]*\r?$", "Anexos"),
    ]
    starts = [(0, "00_preliminares", "Preliminares")]
    for identifier, pattern, title in headings:
        matches = []
        for page_index, text in enumerate(pages):
            header = unicodedata.normalize("NFKD", text[:700]).encode("ascii", "ignore").decode()
            if re.search(pattern, header):
                matches.append(page_index)
        if len(matches) != 1:
            raise ValueError(f"Limite ambiguo para {title}: {matches}")
        starts.append((matches[0], identifier, title))
    if [start[0] for start in starts] != sorted({start[0] for start in starts}):
        raise ValueError("Limites de unidades no consecutivos.")
    destination.mkdir(parents=True, exist_ok=False)
    units = []
    for unit_index, (first, identifier, title) in enumerate(starts):
        stop = starts[unit_index + 1][0] if unit_index + 1 < len(starts) else len(pages)
        text = "\f".join(pages[first:stop])
        payload = text.encode("utf-8")
        filename = f"{identifier}.txt"
        (destination / filename).write_bytes(payload)
        units.append({
            "id": identifier, "titulo": title, "archivo": filename,
            "pagina_pdf_inicial": first + 1, "pagina_pdf_final": stop,
            "palabras": len(text.split()), "caracteres": len(text),
            "sha256": sha256(payload), "estado": "preparado_no_enviado",
        })
    manifest = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "fuente": str(source.resolve()), "fuente_sha256": sha256(source.read_bytes()),
        "paginas": len(pages), "extraccion": "pdftotext -layout -enc UTF-8",
        "exclusiones": [], "unidades": units,
        "advertencias": [
            "Se conserva el texto de tablas, formulas, encabezados, pies y referencias.",
            "La extraccion PDF puede alterar el orden de columnas y simbolos matematicos.",
            "Una puntuacion del detector no prueba autoria ni valida calidad academica.",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    reconstructed = "\f".join((destination / unit["archivo"]).read_bytes().decode("utf-8") for unit in units)
    if reconstructed != "\f".join(pages):
        raise ValueError("La segmentacion no conserva el texto extraido.")
    for unit in units:
        if sha256((destination / unit["archivo"]).read_bytes()) != unit["sha256"]:
            raise ValueError(f"Huella invalida: {unit['id']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.source, args.destination)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()