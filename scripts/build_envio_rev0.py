from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(r"C:\usr\ths_mia_fiis\tesis_MIA_lk")
PLAN = ROOT / "Plan"
DEST = PLAN / "envío_rev0"
GROUP_ID = 6532672
API = f"http://127.0.0.1:23119/api/groups/{GROUP_ID}"
ZOTERO_STORAGE = Path(r"C:\Users\QU1267\Zotero\storage")


def api_json(endpoint: str):
    command = [
        "curl.exe",
        "-sS",
        "--fail",
        "--max-time",
        "30",
        "-H",
        "Zotero-API-Version: 3",
        f"{API}{endpoint}",
    ]
    last_error = ""
    for _ in range(3):
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode == 0:
            return json.loads(completed.stdout)
        last_error = completed.stderr.strip()
    raise RuntimeError(f"No se pudo leer Zotero: {endpoint}: {last_error}")


def api_all(endpoint: str, page_size: int = 100):
    rows = []
    start = 0
    while True:
        separator = "&" if "?" in endpoint else "?"
        page = api_json(f"{endpoint}{separator}limit={page_size}&start={start}")
        rows.extend(page)
        if len(page) < page_size:
            return rows
        start += page_size


def split_bib_entries(text: str):
    entries = {}
    pattern = re.compile(r"(?m)^\s*@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,")
    for match in pattern.finditer(text):
        start = match.start()
        brace = text.find("{", match.start())
        depth = 0
        escaped = False
        end = None
        for index in range(brace, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end:
            entries[match.group(2)] = {
                "type": match.group(1),
                "raw": text[start:end].strip(),
            }
    return entries


def bib_field(raw: str, field: str) -> str:
    match = re.search(rf"(?mi)^\s*{re.escape(field)}\s*=\s*", raw)
    if not match:
        return ""
    index = match.end()
    while index < len(raw) and raw[index].isspace():
        index += 1
    if index >= len(raw):
        return ""
    if raw[index] == "{":
        depth = 0
        escaped = False
        for end in range(index, len(raw)):
            char = raw[end]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return raw[index + 1 : end].strip()
    if raw[index] == '"':
        escaped = False
        for end in range(index + 1, len(raw)):
            char = raw[end]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return raw[index + 1 : end].strip()
    end = raw.find(",", index)
    return raw[index : end if end >= 0 else len(raw)].strip()


def cited_keys(*bcf_paths: Path):
    keys = set()
    for path in bcf_paths:
        tree = ET.parse(path)
        for element in tree.iter():
            if element.tag.endswith("citekey") and element.text:
                keys.add(element.text.strip())
    return keys


def normalize_doi(value: str) -> str:
    value = value or ""
    value = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", value.strip())
    return value.strip(" {}\t\r\n.").lower()


def normalize_text(value: str) -> str:
    value = value or ""
    value = re.sub(r"\\(?:['\"`^~=.uvHckbdtr])\s*\{?([A-Za-z])\}?", r"\1", value)
    value = value.replace(r"\%", "%").replace(r"\&", " and ")
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", value)
    value = value.replace("{", "").replace("}", "")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def collection_paths(collections):
    by_key = {row["key"]: row["data"] for row in collections}

    def path(key: str) -> str:
        names = []
        seen = set()
        while key and key not in seen and key in by_key:
            seen.add(key)
            data = by_key[key]
            names.insert(0, data.get("name", ""))
            key = data.get("parentCollection") or ""
        return " / ".join(name for name in names if name)

    return {key: path(key) for key in by_key}


def safe_name(value: str, limit: int = 175) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value)
    value = re.sub(r"\s+", "_", value).strip(" ._")
    return (value or "archivo")[:limit]


def first_year(date_value: str) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", date_value or "")
    return match.group(0) if match else ""


def creators_bib(creators) -> str:
    names = []
    for creator in creators or []:
        if creator.get("name"):
            names.append(creator["name"])
        else:
            last = creator.get("lastName", "")
            first = creator.get("firstName", "")
            names.append(f"{last}, {first}".strip(" ,"))
    return " and ".join(name for name in names if name)


def extra_bib_entry(key: str, data: dict) -> str:
    fields = [
        ("author", creators_bib(data.get("creators"))),
        ("title", data.get("title", "")),
        ("journaltitle", data.get("publicationTitle", "")),
        ("year", first_year(data.get("date", ""))),
        ("volume", data.get("volume", "")),
        ("pages", data.get("pages", "")),
        ("doi", data.get("DOI", "")),
        ("url", data.get("url", "")),
    ]
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields if value)
    return f"@article{{{key},\n{body}\n}}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8", newline="\n")


def main():
    if DEST.exists():
        resolved = DEST.resolve()
        expected = (PLAN / "envío_rev0").resolve()
        if resolved != expected or resolved.parent != PLAN.resolve() or resolved.name != "envío_rev0":
            raise RuntimeError(f"Verificación de destino fallida; no se elimina: {resolved}")
        shutil.rmtree(resolved)

    # Fuente bibliográfica y claves realmente procesadas por Biber.
    bib_path = PLAN / "referencias_plan_tesis.bib"
    bib_text = bib_path.read_text(encoding="utf-8")
    bib_entries = split_bib_entries(bib_text)
    keys = cited_keys(
        PLAN / "plan_tesis_cables_pinn.bcf",
        PLAN / "presentación" / "presentacion_plan_tesis_cables_pinn.bcf",
    )

    # Exportación actual desde la API local de Zotero.
    collections = api_all("/collections")
    all_items = api_all("/items")
    top_items = [row for row in all_items if row.get("data", {}).get("itemType") not in {"attachment", "note", "annotation"}]
    item_by_key = {row["key"]: row for row in top_items}
    children_by_parent = {}
    for row in all_items:
        parent = row.get("data", {}).get("parentItem")
        if parent:
            children_by_parent.setdefault(parent, []).append(row)

    doi_index = {}
    title_index = {}
    citation_index = {}
    for row in top_items:
        data = row["data"]
        doi = normalize_doi(data.get("DOI", ""))
        title = normalize_text(data.get("title", ""))
        citation_key = data.get("citationKey", "")
        if doi:
            doi_index.setdefault(doi, []).append(row)
        if title:
            title_index.setdefault(title, []).append(row)
        if citation_key:
            citation_index[citation_key] = row

    selected = []
    missing_bib_entries = []
    unmatched = []
    normalized_zotero_titles = [(normalize_text(row["data"].get("title", "")), row) for row in top_items]
    known_citation_aliases = {
        "iec60287": "iec60287_1_1_2023",
    }

    def prefer_pdf(candidates):
        return max(
            candidates,
            key=lambda row: sum(
                1
                for child in children_by_parent.get(row["key"], [])
                if child.get("data", {}).get("contentType") == "application/pdf"
            ),
        )

    for key in sorted(keys):
        entry = bib_entries.get(key)
        if not entry:
            missing_bib_entries.append(key)
            continue
        doi = normalize_doi(bib_field(entry["raw"], "doi"))
        title = normalize_text(bib_field(entry["raw"], "title"))
        match = None
        method = ""
        if doi and doi_index.get(doi):
            match = prefer_pdf(doi_index[doi])
            method = "DOI"
        elif title and title_index.get(title):
            match = prefer_pdf(title_index[title])
            method = "título"
        elif key in known_citation_aliases and known_citation_aliases[key] in citation_index:
            match = citation_index[known_citation_aliases[key]]
            method = "alias bibliográfico verificado"
        elif title:
            scored = [
                (difflib.SequenceMatcher(None, title, candidate_title).ratio(), row)
                for candidate_title, row in normalized_zotero_titles
                if candidate_title
            ]
            score, candidate = max(scored, key=lambda pair: pair[0])
            if score >= 0.93:
                match = candidate
                method = f"título aproximado ({score:.3f})"
        if match:
            selected.append({"bib_key": key, "item": match, "match_method": method, "bib_entry": entry["raw"]})
        else:
            unmatched.append({"bib_key": key, "title": bib_field(entry["raw"], "title"), "doi": doi})

    # Dos fuentes adicionales usadas en el informe mejorado de datos.
    extras = [
        ("quan2019", "quan2019numerical"),
        ("oladunjoye2012", "oladunjoyeSituDeterminationThermal2012"),
    ]
    for bib_key, zotero_citation_key in extras:
        row = citation_index.get(zotero_citation_key)
        if not row:
            unmatched.append({"bib_key": bib_key, "title": "", "doi": "", "reason": "extra no encontrado"})
            continue
        selected.append(
            {
                "bib_key": bib_key,
                "item": row,
                "match_method": "citationKey de Zotero",
                "bib_entry": extra_bib_entry(bib_key, row["data"]),
            }
        )

    DEST.mkdir(parents=True)
    ref_dir = DEST / "01_Referencias_Zotero"
    report_dir = DEST / "02_Informe"
    presentation_dir = DEST / "03_Presentacion"
    data_dir = DEST / "04_Datos_objeto_estudio"
    pdf_dir = ref_dir / "01_PDFs"
    supplemental_pdf_dir = ref_dir / "01_PDFs_sin_registro_Zotero"
    for directory in (pdf_dir, supplemental_pdf_dir, report_dir, presentation_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = collection_paths(collections)
    export_rows = []
    csv_rows = []
    copied_pdfs = []
    supplemental_pdfs = []
    missing_pdfs = []
    unique_items = {}

    for record in selected:
        item = record["item"]
        unique_items[item["key"]] = item
        data = item["data"]
        item_collections = [paths.get(key, key) for key in data.get("collections", [])]
        pdf_exports = []
        pdf_children = [
            child
            for child in children_by_parent.get(item["key"], [])
            if child.get("data", {}).get("itemType") == "attachment"
            and child.get("data", {}).get("contentType") == "application/pdf"
        ]
        for attachment_index, attachment in enumerate(pdf_children, start=1):
            attachment_data = attachment["data"]
            original_name = attachment_data.get("filename", "")
            source = ZOTERO_STORAGE / attachment["key"] / original_name
            if not source.exists() and attachment_data.get("path"):
                candidate = Path(attachment_data["path"].replace("attachments:", str(ZOTERO_STORAGE) + "\\"))
                if candidate.exists():
                    source = candidate
            if source.exists():
                destination_name = safe_name(
                    f"01_{len(copied_pdfs) + 1:03d}_{record['bib_key']}_{attachment_index}_{original_name}"
                )
                if not destination_name.lower().endswith(".pdf"):
                    destination_name += ".pdf"
                destination = pdf_dir / destination_name
                shutil.copy2(source, destination)
                pdf_exports.append(destination_name)
                copied_pdfs.append(destination)
            else:
                missing_pdfs.append(
                    {
                        "bib_key": record["bib_key"],
                        "zotero_item_key": item["key"],
                        "attachment_key": attachment["key"],
                        "filename": original_name,
                    }
                )

        export_rows.append(
            {
                "bib_key": record["bib_key"],
                "match_method": record["match_method"],
                "zotero": item,
                "classification": item_collections,
                "pdf_files_in_package": pdf_exports,
            }
        )
        csv_rows.append(
            {
                "bib_key": record["bib_key"],
                "zotero_citation_key": data.get("citationKey", ""),
                "zotero_item_key": item["key"],
                "title": data.get("title", ""),
                "DOI": data.get("DOI", ""),
                "collections": " | ".join(item_collections),
                "tags": " | ".join(tag.get("tag", "") for tag in data.get("tags", [])),
                "pdfs": " | ".join(pdf_exports),
                "match_method": record["match_method"],
            }
        )

    # PDF locales de referencias citadas que no forman parte de la biblioteca Zotero actual.
    supplemental_sources = {
        "delport2024": PLAN / "Delport et al. - 2024 - Methodological Guidelines for Design Science Research.pdf",
        "gregor2013": PLAN / "GregorHevnerMISQ2013.pdf",
        "peffers2007": PLAN / "Peffers et al. - 2007 - A design science research methodology for information systems research.pdf",
        "iec60287": ROOT / "Entrega_Google_Drive" / "03_normas_estandares_guias_manuales" / "IEC 60287-1-1 2023. Electric cables--Calculation of the current rating--Part 1-1 Current rating equations and calculation of losses.pdf",
        "minem2025anuario2024": PLAN / "_fuentes_tmp" / "minem_anuario_2024.pdf",
        "minem2025indicadoresDiciembre": PLAN / "_fuentes_tmp" / "minem_2025_dic.pdf",
    }
    unmatched_by_key = {row["bib_key"]: row for row in unmatched}
    for bib_key, source in supplemental_sources.items():
        if bib_key not in unmatched_by_key or not source.exists():
            continue
        destination = supplemental_pdf_dir / safe_name(f"01_{bib_key}_{source.name}")
        if not destination.name.lower().endswith(".pdf"):
            destination = destination.with_suffix(".pdf")
        shutil.copy2(source, destination)
        supplemental_pdfs.append(destination)
        unmatched_by_key[bib_key]["local_pdf_in_package"] = destination.name
        entry = bib_entries.get(bib_key, {})
        csv_rows.append(
            {
                "bib_key": bib_key,
                "zotero_citation_key": "",
                "zotero_item_key": "",
                "title": bib_field(entry.get("raw", ""), "title"),
                "DOI": bib_field(entry.get("raw", ""), "doi"),
                "collections": "NO LOCALIZADA EN ZOTERO",
                "tags": "",
                "pdfs": destination.name,
                "match_method": "PDF local suplementario",
            }
        )

    # Registrar también las referencias sin Zotero ni PDF para que el inventario sea completo.
    classified_bib_keys = {row["bib_key"] for row in csv_rows}
    for row in unmatched:
        if row["bib_key"] in classified_bib_keys:
            continue
        csv_rows.append(
            {
                "bib_key": row["bib_key"],
                "zotero_citation_key": "",
                "zotero_item_key": "",
                "title": row.get("title", ""),
                "DOI": row.get("doi", ""),
                "collections": "NO LOCALIZADA EN ZOTERO",
                "tags": "",
                "pdfs": "",
                "match_method": "sin registro Zotero ni PDF local",
            }
        )

    full_export = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": "Zotero local API (solo lectura)",
        "library": {"type": "group", "id": GROUP_ID, "name": "Tesis_MIA_2026"},
        "collections_current": collections,
        "records": export_rows,
        "unmatched_bibliography": unmatched,
        "missing_bib_entries": missing_bib_entries,
        "missing_pdf_files": missing_pdfs,
    }
    write_text(ref_dir / "01_referencias_usadas_zotero.json", json.dumps(full_export, ensure_ascii=False, indent=2))
    write_text(ref_dir / "01_colecciones_actuales_zotero.json", json.dumps(collections, ensure_ascii=False, indent=2))
    with (ref_dir / "01_clasificacion_actual_zotero.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["bib_key"])
        writer.writeheader()
        writer.writerows(csv_rows)

    bib_output = "% Exportación completa de las referencias usadas en el envío rev0.\n\n"
    bib_output += "\n\n".join(bib_entries[key]["raw"] for key in sorted(keys) if key in bib_entries)
    extra_entries = [record["bib_entry"] for record in selected if record["bib_key"] in {"quan2019", "oladunjoye2012"}]
    if extra_entries:
        bib_output += "\n\n" + "\n\n".join(extra_entries)
    bib_output += "\n"
    write_text(ref_dir / "01_referencias_usadas.bib", bib_output)

    manifest = f"""# 01 Referencias de Zotero

Exportación generada desde la API local de solo lectura de Zotero para la biblioteca de grupo `Tesis_MIA_2026`.

- Claves citadas procesadas por Biber: {len(keys)}
- Registros citados localizados en Zotero: {len(selected) - len(extras)}
- Registros adicionales del informe de datos: {len(extras)}
- Ítems Zotero únicos exportados: {len(unique_items)}
- PDF copiados desde Zotero: {len(copied_pdfs)}
- PDF locales suplementarios sin registro Zotero: {len(supplemental_pdfs)}
- Referencias no localizadas en Zotero: {len(unmatched)}
- Adjuntos PDF registrados pero no hallados en disco: {len(missing_pdfs)}

`01_referencias_usadas_zotero.json` conserva los metadatos Zotero, las colecciones actuales, etiquetas y adjuntos. `01_clasificacion_actual_zotero.csv` facilita la revisión tabular. `01_referencias_usadas.bib` contiene las 49 referencias citadas y las dos fuentes añadidas al informe de datos. `01_PDFs` reúne los adjuntos de Zotero; `01_PDFs_sin_registro_Zotero` separa los PDF locales suplementarios.

## Referencias no localizadas

{json.dumps(unmatched, ensure_ascii=False, indent=2) if unmatched else "Ninguna."}
"""
    write_text(ref_dir / "01_MANIFIESTO.md", manifest)

    # Informe principal y fuentes mínimas.
    report_sources = report_dir / "02_Fuentes_LaTeX"
    report_images = report_sources / "02_imagenes"
    report_images.mkdir(parents=True)
    shutil.copy2(PLAN / "plan_tesis_cables_pinn.pdf", report_dir / "02_Informe_plan_tesis_cables_pinn.pdf")
    report_tex = (PLAN / "plan_tesis_cables_pinn.tex").read_text(encoding="utf-8")
    report_tex = report_tex.replace(r"\graphicspath{{imagenes/}}", r"\graphicspath{{02_imagenes/}}")
    report_tex = report_tex.replace(r"\addbibresource{referencias_plan_tesis.bib}", r"\addbibresource{02_referencias_plan_tesis.bib}")
    report_assets = {
        "logo_UNI.png": "02_logo_UNI.png",
        "kim2025_cable_xsection.png": "02_kim2025_cable_xsection.png",
        "enescu_trench_xsection_es.png": "02_enescu_trench_xsection_es.png",
    }
    for original_name, packaged_name in report_assets.items():
        report_tex = report_tex.replace(original_name.removesuffix(".png"), packaged_name.removesuffix(".png"))
    write_text(report_sources / "02_plan_tesis_cables_pinn.tex", report_tex)
    shutil.copy2(bib_path, report_sources / "02_referencias_plan_tesis.bib")
    for original_name, packaged_name in report_assets.items():
        shutil.copy2(PLAN / "imagenes" / original_name, report_images / packaged_name)
    write_text(
        report_sources / "02_compilar.bat",
        "@echo off\r\nlatexmk -lualatex -interaction=nonstopmode 02_plan_tesis_cables_pinn.tex\r\n",
    )
    write_text(
        report_sources / "02_LEEME.md",
        "# Reproducción del informe\n\nRequiere una distribución LaTeX con LuaLaTeX, `latexmk`, Biber y la fuente Arial instalada. Ejecute `02_compilar.bat`. La carpeta contiene solo el `.tex`, la bibliografía y las tres imágenes usadas; no incluye auxiliares.\n",
    )

    # Presentación y fuentes mínimas.
    presentation_sources = presentation_dir / "03_Fuentes_LaTeX"
    presentation_images = presentation_sources / "03_imagenes"
    presentation_images.mkdir(parents=True)
    presentation_original = PLAN / "presentación"
    shutil.copy2(
        presentation_original / "presentacion_plan_tesis_cables_pinn.pdf",
        presentation_dir / "03_Presentacion_plan_tesis_cables_pinn.pdf",
    )
    presentation_tex = (presentation_original / "presentacion_plan_tesis_cables_pinn.tex").read_text(encoding="utf-8")
    presentation_tex = presentation_tex.replace(r"\addbibresource{../referencias_plan_tesis.bib}", r"\addbibresource{03_referencias_plan_tesis.bib}")
    presentation_tex = presentation_tex.replace(
        r"\graphicspath{{../imagenes/}{../../examples/kim_2024_154kv_bedding/results_multilayer_research/}}",
        r"\graphicspath{{03_imagenes/}}",
    )
    presentation_assets = {
        "logo_UNI.png": "03_logo_UNI.png",
        "enescu_trench_xsection_es.png": "03_enescu_trench_xsection_es.png",
    }
    for original_name, packaged_name in presentation_assets.items():
        presentation_tex = presentation_tex.replace(original_name.removesuffix(".png"), packaged_name.removesuffix(".png"))
    write_text(presentation_sources / "03_presentacion_plan_tesis_cables_pinn.tex", presentation_tex)
    shutil.copy2(bib_path, presentation_sources / "03_referencias_plan_tesis.bib")
    for original_name, packaged_name in presentation_assets.items():
        shutil.copy2(PLAN / "imagenes" / original_name, presentation_images / packaged_name)
    write_text(
        presentation_sources / "03_compilar.bat",
        "@echo off\r\nlatexmk -lualatex -interaction=nonstopmode 03_presentacion_plan_tesis_cables_pinn.tex\r\n",
    )
    write_text(
        presentation_sources / "03_LEEME.md",
        "# Reproducción de la presentación\n\nRequiere una distribución LaTeX con LuaLaTeX, `latexmk`, Biber y la fuente Arial instalada. Ejecute `03_compilar.bat`. La carpeta contiene solo el `.tex`, la bibliografía y las dos imágenes usadas; no incluye auxiliares.\n",
    )

    # Datos mejorados: fuentes autoritativas sin auxiliares ni DOCX desactualizado.
    data_source = ROOT / "Entrega_Google_Drive" / "02_ejemplos_datos_objeto_estudio"
    shutil.copy2(data_source / "Informe_datos_casos_estudio_papers.pdf", data_dir / "04_Informe_datos_casos_estudio_papers.pdf")
    shutil.copy2(data_source / "Informe_datos_casos_estudio_papers.tex", data_dir / "04_Informe_datos_casos_estudio_papers.tex")
    shutil.copy2(data_source / "Informe_datos_casos_estudio_papers.md", data_dir / "04_Informe_datos_casos_estudio_papers.md")

    data_rows = [
        ["kim2025", "Kim et al. (2025)", "UPCS 154 kV; seis cables; PAC/arena; tres estratos", "objeto_fisico", "Campo térmico y ampacidad con bedding", "10.1016/j.geothermics.2024.103151"],
        ["khumalo2025", "Khumalo et al. (2025)", "Cable MV XLPE; secado de suelo y ampacidad", "objeto_fisico", "Sensibilidad a humedad y resistividad", "10.1155/etep/5946564"],
        ["aldulaimi2024", "Al-Dulaimi et al. (2024)", "Cable 132 kV; suelo multicapa; SCMB/FTB; FEM-BPNN", "objeto_fisico", "Geometría y materiales heterogéneos", "10.1016/j.jestch.2024.101658"],
        ["atoccsa2024", "Atoccsa et al. (2024)", "Cable 220 kV; backfill optimizado con PSO", "objeto_fisico", "Ampacidad y optimización de backfill", "10.3390/en17174356"],
        ["oclon2015", "Ocłoń et al. (2015)", "Sistema 400 kV; FTB/SBM/HDPE; PSO", "objeto_fisico", "Interfaces y backfill localizado", "10.1016/j.enconman.2015.07.015"],
        ["aras2005", "Aras et al. (2005)", "Cable 154 kV; comparación IEC/FEM/ensayo", "objeto_fisico", "Verificación base homogénea", "10.1080/15325000590964425"],
        ["quan2019", "Quan et al. (2019)", "Tres cables; plana/trébol; suelo/FTB; k(T)", "objeto_fisico", "Geometría, profundidad y k(T)", "10.1016/j.egypro.2019.01.636"],
        ["oladunjoye2012", "Oladunjoye et al. (2012)", "Diez calicatas; propiedades térmicas in situ", "datos_de_campo", "Parametrización espacial del suelo", "10.5402/2012/591450"],
        ["mobius2025", "Möbius et al. (2025)", "Tres cables MV; Hamburgo; 32 años de suelo y clima", "objeto_fisico", "Escenarios estacionales", "10.1016/j.rser.2025.115348"],
        ["cigre2022_case0", "CIGRE TB 880 (2022)", "Cable 132 kV; caso base y 14 variantes IEC", "verificacion_normativa", "Regresión corriente-pérdidas-temperatura", "CIGRE TB 880"],
        ["cigre2025_case1", "CIGRE TB 963 (2025)", "Tres cables 132 kV en ductos; nueve variantes FEM 2D", "benchmark_fem_2d", "Campo, dominio, interfaces y ampacidad", "CIGRE TB 963"],
        ["iec60853_annexA", "IEC 60853-3 (2002)", "Cable tripolar 132 kV; ciclo y secado parcial", "verificacion_normativa_transitoria", "Extensión temporal futura", "IEC 60853-3:2002"],
        ["pan2025_plate", "Pan et al. (2025)", "Placa 1 m x 0.5 m; Laplace; borde desconocido", "benchmark_pinn_2d", "Muestreo e inversión de frontera", "10.3390/eng6050099"],
        ["cigre2025_annulus", "CIGRE TB 963 (2025)", "Anillo 2D con flujo interior y solución exacta", "benchmark_analitico", "Error de campo y condición de flujo", "CIGRE TB 963"],
        ["hahn2012_rect", "Hahn y Özişik (2012)", "Rectángulo 2D con frontera convectiva y serie exacta", "benchmark_analitico", "Condición de Robin", "10.1002/9781118411285"],
        ["mms2d_suite", "Elaboración propia", "Conductividad variable e interfaz de dos materiales", "benchmark_manufacturado", "PDE, k(x,y), fuentes e interfaces", "Elaboración propia"],
    ]
    with (data_dir / "04_indice_instancias.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["clave", "fuente", "instancia", "tipo", "uso_pinn", "doi_o_identificador"])
        writer.writerows(data_rows)
    write_text(
        data_dir / "04_LEEME.md",
        "# Datos del objeto de estudio y referencias de verificación\n\nVersión revisada y ampliada a 16 instancias: nueve físicas o de campo y siete de verificación normativa, FEM, analítica o manufacturada. La ampliación incorpora CIGRE TB 880, CIGRE TB 963, IEC 60853-3, Pan et al. (2025), Hahn y Özişik (2012) y dos soluciones manufacturadas 2D. El PDF, el `.tex` y el Markdown son las versiones autoritativas; se excluyeron auxiliares y el DOCX anterior porque no refleja esta revisión.\n",
    )

    # ZIP de referencias, excluyendo el propio ZIP.
    zip_path = ref_dir / "01_Referencias_Zotero_con_PDFs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(ref_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, Path("01_Referencias_Zotero") / path.relative_to(ref_dir))

    # Índice y sumas de comprobación del envío.
    index = f"""# Envío rev0

1. `01_Referencias_Zotero`: exportación de Zotero, clasificación actual, bibliografía, {len(copied_pdfs)} PDF de Zotero, {len(supplemental_pdfs)} PDF locales suplementarios y ZIP.
2. `02_Informe`: informe principal de 68 páginas y fuentes LaTeX mínimas reproducibles.
3. `03_Presentacion`: presentación de 33 láminas y fuentes LaTeX mínimas reproducibles.
4. `04_Datos_objeto_estudio`: informe ampliado a 16 instancias físicas y de verificación, fuentes y CSV índice.

Los archivos auxiliares de LaTeX fueron excluidos deliberadamente.
"""
    write_text(DEST / "01_INDICE_envio_rev0.md", index)

    checksum_rows = []
    for path in sorted(DEST.rglob("*")):
        if path.is_file() and path.name != "01_SHA256SUMS.txt":
            checksum_rows.append(f"{sha256(path)}  {path.relative_to(DEST).as_posix()}")
    write_text(DEST / "01_SHA256SUMS.txt", "\n".join(checksum_rows) + "\n")

    summary = {
        "destination": str(DEST),
        "cited_keys": len(keys),
        "selected_records": len(selected),
        "unique_zotero_items": len(unique_items),
        "pdfs_from_zotero": len(copied_pdfs),
        "supplemental_local_pdfs": len(supplemental_pdfs),
        "unmatched": unmatched,
        "missing_pdf_files": missing_pdfs,
        "files": sum(1 for path in DEST.rglob("*") if path.is_file()),
        "bytes": sum(path.stat().st_size for path in DEST.rglob("*") if path.is_file()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
