import html
import json
import random
import re
import sqlite3
import string
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DB = Path(r"C:\Users\QU1267\Zotero\zotero.sqlite")
NOTE_TITLE = "[Tesis MIA] Aporte a la tesis"
LIBRARY_ID = 2


CATEGORY_TAGS = {
    "PINNs": "tema-pinn",
    "FEM": "tema-fem-numerico",
    "Fundamentos": "tema-transferencia-calor",
    "Cables": "tema-cables-ampacidad",
    "Suelo": "tema-suelo-backfill-humedad",
    "DTR": "tema-dtr-cargas-ciclicas",
    "Normas": "tema-normas-estandares",
    "Otros": "tema-otros",
}

HTML_CATEGORY_TAGS = {
    "PINN": "tema-pinn",
    "FEM/Numerico": "tema-fem-numerico",
    "Fundamentos TC": "tema-transferencia-calor",
    "Cable/Ampacidad": "tema-cables-ampacidad",
    "Suelo/Backfill": "tema-suelo-backfill-humedad",
    "DTR/Dinamico": "tema-dtr-cargas-ciclicas",
    "Norma": "tema-normas-estandares",
}

REFINED_TYPE_TAGS = {
    "estado_arte": ("tipo-estado-del-arte",),
    "aporte_teorico": ("tipo-aporte", "tipo-aporte-teorico"),
    "aplicacion": ("tipo-aplicacion",),
    "normativa": ("tipo-otros", "tipo-normativa"),
    "guia_tecnica": ("tipo-aporte", "tipo-guia-tecnica"),
    "apoyo": ("tipo-aporte", "tipo-apoyo-didactico"),
    "fuente_clasica": ("tipo-aporte", "tipo-fuente-clasica"),
}

TITLE_TYPE_RULES = [
    (r"\breview\b|\boverview\b|systematic|mapping|assessment of the methods", "estado_arte"),
    (r"\biec\b|\bieee\b|\bastm\b|\bstandard\b|\bguide\b|\btb\s*\d+|cigr", "normativa"),
    (r"\bhandbook\b|examples|calculation tool verification|catalog|cables de aluminio", "guia_tecnica"),
    (r"\bheat equation\b|diffusion equation|separation of variables", "apoyo"),
    (r"conduction of heat in solids|numerical heat transfer and fluid flow|neher|thermal properties of soils", "fuente_clasica"),
    (r"model|simulation|optimization|application|analysis|monitoring|rating|ampacity|temperature|conductivity", "aplicacion"),
]


def norm_doi(value):
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    value = value.strip().strip(".")
    return "" if value in {"", "n/a", "na", "none"} else value


def norm_text(value):
    value = value or ""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def short_words(text, max_words=95):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,.;:") + "."


def parse_markdown_records():
    md = (ROOT / "ANALISIS_BIBLIOGRAFICO_ZOTERO.md").read_text(encoding="utf-8")
    records = []

    for block in re.split(r"(?=^## )", md, flags=re.M):
        if not block.startswith("## "):
            continue
        category = block.splitlines()[0][3:].strip()
        if category.startswith("PINNs para"):
            continue

        for part in re.split(r"(?=^### )", block, flags=re.M)[1:]:
            heading = part.splitlines()[0][4:].strip()
            bits = heading.split("\u2014", 1)
            author_year = bits[0].strip() if len(bits) > 1 else ""
            title = (bits[1].strip() if len(bits) > 1 else heading).replace("...", "")

            doi = re.search(r"\*\*DOI/URL:\*\*\s*(.+)", part)
            keywords = re.search(r"\*\*Keywords:\*\*\s*(.+)", part)
            aporte = re.search(r"\*\*Aporte al proyecto:\*\*\s*(.+)", part)

            records.append(
                {
                    "category_full": category,
                    "author_year": author_year,
                    "title": title,
                    "doi": norm_doi(doi.group(1) if doi else ""),
                    "keywords": keywords.group(1).strip() if keywords else "",
                    "aporte": short_words(aporte.group(1).strip() if aporte else ""),
                }
            )
    return records


def parse_html_records():
    records = []
    for filename in ("zotero_network.html", "zotero_offline/zotero_network.html"):
        path = ROOT / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"const NODES=(\[.*?\]);", text, flags=re.S)
        if not match:
            continue
        nodes = json.loads(match.group(1))
        for node in nodes:
            records.append(
                {
                    "category": node.get("category") or "",
                    "paper_type": node.get("paper_type") or "",
                    "title": node.get("title_full") or "",
                    "doi": norm_doi(node.get("doi_link") or ""),
                    "tags": node.get("tags") or [],
                }
            )
        break
    return records


def best_by_title(item, records, threshold):
    item_title = norm_text(item["title"])
    best = None
    best_score = 0.0

    for record in records:
        record_title = norm_text(record.get("title", ""))
        if not record_title:
            continue
        score = SequenceMatcher(None, item_title, record_title).ratio()
        if record_title and (item_title.startswith(record_title) or record_title.startswith(item_title[: len(record_title)])):
            score = max(score, 0.92)
        if score > best_score:
            best = record
            best_score = score

    return (best, best_score) if best_score >= threshold else (None, best_score)


def category_tag(md_record, html_record):
    if html_record and html_record.get("category") in HTML_CATEGORY_TAGS:
        return HTML_CATEGORY_TAGS[html_record["category"]]

    category = md_record.get("category_full", "") if md_record else ""
    for needle, tag in CATEGORY_TAGS.items():
        if needle.lower() in category.lower():
            return tag
    return "tema-otros"


def guess_paper_type(item, md_record, html_record):
    if html_record and html_record.get("paper_type"):
        return html_record["paper_type"]

    haystack = norm_text(
        " ".join(
            [
                item.get("title", ""),
                item.get("typeName", ""),
                md_record.get("keywords", "") if md_record else "",
                md_record.get("category_full", "") if md_record else "",
            ]
        )
    )
    for pattern, paper_type in TITLE_TYPE_RULES:
        if re.search(pattern, haystack):
            return paper_type
    return "aporte_teorico"


def keyword_tags(md_record):
    if not md_record or not md_record.get("keywords"):
        return []
    tags = []
    for raw in re.split(r",|;", md_record["keywords"]):
        tag = raw.strip()
        if not tag:
            continue
        if len(tag) > 80:
            continue
        tags.append(tag)
    return tags


def get_top_level_items(cur):
    query = """
    SELECT
        i.itemID,
        i.key,
        it.typeName,
        COALESCE(v_title.value, '') AS title,
        COALESCE(v_doi.value, '') AS doi,
        COALESCE(v_date.value, '') AS date,
        COALESCE(v_abs.value, '') AS abstractNote
    FROM items i
    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
    LEFT JOIN itemData d_title ON d_title.itemID = i.itemID AND d_title.fieldID = 1
    LEFT JOIN itemDataValues v_title ON v_title.valueID = d_title.valueID
    LEFT JOIN itemData d_doi ON d_doi.itemID = i.itemID AND d_doi.fieldID = 8
    LEFT JOIN itemDataValues v_doi ON v_doi.valueID = d_doi.valueID
    LEFT JOIN itemData d_date ON d_date.itemID = i.itemID AND d_date.fieldID = 6
    LEFT JOIN itemDataValues v_date ON v_date.valueID = d_date.valueID
    LEFT JOIN itemData d_abs ON d_abs.itemID = i.itemID AND d_abs.fieldID = 2
    LEFT JOIN itemDataValues v_abs ON v_abs.valueID = d_abs.valueID
    WHERE i.libraryID = ?
      AND i.itemID NOT IN (SELECT itemID FROM itemAttachments)
      AND i.itemID NOT IN (SELECT itemID FROM itemNotes)
      AND i.itemTypeID NOT IN (1, 3, 28)
    ORDER BY i.itemID
    """
    return [dict(row) for row in cur.execute(query, (LIBRARY_ID,))]


def get_existing_item_tags(cur, item_id):
    rows = cur.execute(
        """
        SELECT t.name
        FROM itemTags it
        JOIN tags t ON t.tagID = it.tagID
        WHERE it.itemID = ?
        """,
        (item_id,),
    )
    return {row[0] for row in rows}


def ensure_tag(cur, tag_name, tag_lookup):
    tag_name = tag_name.strip()
    key = tag_name.lower()
    if key in tag_lookup:
        return tag_lookup[key]
    cur.execute("INSERT INTO tags(name) VALUES (?)", (tag_name,))
    tag_id = cur.lastrowid
    tag_lookup[key] = tag_id
    return tag_id


def attach_tag(cur, item_id, tag_id):
    cur.execute(
        "INSERT OR IGNORE INTO itemTags(itemID, tagID, type) VALUES (?, ?, 0)",
        (item_id, tag_id),
    )
    return cur.rowcount > 0


def sync_queue_item(cur, library_id, key):
    cur.execute(
        """
        INSERT OR IGNORE INTO syncQueue(libraryID, key, syncObjectTypeID, lastCheck, tries)
        VALUES (?, ?, 3, NULL, 0)
        """,
        (library_id, key),
    )


def unique_key(cur):
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    while True:
        key = "".join(random.choice(alphabet) for _ in range(8))
        exists = cur.execute("SELECT 1 FROM items WHERE libraryID = ? AND key = ?", (LIBRARY_ID, key)).fetchone()
        if not exists:
            return key


def upsert_note(cur, item, note_html, now):
    existing = cur.execute(
        """
        SELECT i.itemID, i.key
        FROM itemNotes n
        JOIN items i ON i.itemID = n.itemID
        WHERE n.parentItemID = ? AND n.title = ?
        """,
        (item["itemID"], NOTE_TITLE),
    ).fetchone()

    if existing:
        cur.execute(
            "UPDATE itemNotes SET note = ?, title = ? WHERE itemID = ?",
            (note_html, NOTE_TITLE, existing["itemID"]),
        )
        cur.execute(
            """
            UPDATE items
            SET dateModified = ?, clientDateModified = ?, synced = 0
            WHERE itemID = ?
            """,
            (now, now, existing["itemID"]),
        )
        sync_queue_item(cur, LIBRARY_ID, existing["key"])
        return "updated"

    note_key = unique_key(cur)
    cur.execute(
        """
        INSERT INTO items(itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
        VALUES (28, ?, ?, ?, ?, ?, 0, 0)
        """,
        (now, now, now, LIBRARY_ID, note_key),
    )
    note_id = cur.lastrowid
    cur.execute(
        "INSERT INTO itemNotes(itemID, parentItemID, note, title) VALUES (?, ?, ?, ?)",
        (note_id, item["itemID"], note_html, NOTE_TITLE),
    )
    sync_queue_item(cur, LIBRARY_ID, note_key)
    return "inserted"


def build_note(md_record, category, paper_type):
    aporte = md_record.get("aporte") if md_record else ""
    if not aporte:
        aporte = "Este trabajo aporta contexto metodológico o técnico para el modelamiento térmico de cables subterráneos con PINNs."
    aporte = short_words(aporte, 80)
    note_text = (
        f"<p><strong>Aporte a la tesis:</strong> {html.escape(aporte)}</p>"
        f"<p><strong>Clasificación:</strong> {html.escape(category)}; {html.escape(paper_type)}.</p>"
    )
    return f'<div class="zotero-note znv1">{note_text}</div>'


def main():
    if not DB.exists():
        raise SystemExit(f"No existe la base Zotero: {DB}")

    md_records = parse_markdown_records()
    html_records = parse_html_records()
    md_by_doi = {}
    html_by_doi = {}
    for record in md_records:
        if record["doi"]:
            md_by_doi.setdefault(record["doi"], []).append(record)
    for record in html_records:
        if record["doi"]:
            html_by_doi.setdefault(record["doi"], []).append(record)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = DB.with_name(f"zotero.sqlite.codex-backup-{timestamp}")
    source = sqlite3.connect(str(DB))
    destination = sqlite3.connect(str(backup_path))
    with destination:
        source.backup(destination)
    source.close()
    destination.close()

    con = sqlite3.connect(str(DB), timeout=20)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("PRAGMA busy_timeout = 10000")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    items = get_top_level_items(cur)
    tag_lookup = {row["name"].lower(): row["tagID"] for row in cur.execute("SELECT tagID, name FROM tags")}

    summary = {
        "backup": str(backup_path),
        "items": len(items),
        "notes_inserted": 0,
        "notes_updated": 0,
        "tags_added": 0,
        "md_matched": 0,
        "html_matched": 0,
        "categories": Counter(),
        "paper_types": Counter(),
        "duplicates_by_doi": [],
    }

    doi_seen = Counter(norm_doi(item["doi"]) for item in items if norm_doi(item["doi"]))
    summary["duplicates_by_doi"] = [doi for doi, count in doi_seen.items() if count > 1]

    cur.execute("BEGIN IMMEDIATE")
    try:
        for item in items:
            doi = norm_doi(item["doi"])
            md_record = (md_by_doi.get(doi) or [None])[0] if doi else None
            md_score = 1.0 if md_record else 0.0
            if not md_record:
                md_record, md_score = best_by_title(item, md_records, 0.45)

            html_record = (html_by_doi.get(doi) or [None])[0] if doi else None
            html_score = 1.0 if html_record else 0.0
            if not html_record:
                html_record, html_score = best_by_title(item, html_records, 0.60)

            if md_record:
                summary["md_matched"] += 1
            if html_record:
                summary["html_matched"] += 1

            theme_tag = category_tag(md_record, html_record)
            paper_type = guess_paper_type(item, md_record, html_record)
            paper_type_tags = REFINED_TYPE_TAGS.get(paper_type, ("tipo-otros",))

            category_label = md_record.get("category_full", "Otros") if md_record else "Otros"
            summary["categories"][theme_tag] += 1
            summary["paper_types"][paper_type] += 1

            desired_tags = [
                "tesis-mia-clasificado",
                theme_tag,
                *paper_type_tags,
            ]
            desired_tags.extend(keyword_tags(md_record))

            title_keywords = norm_text(item["title"] + " " + (md_record.get("keywords", "") if md_record else ""))
            if any(token in title_keywords for token in ["pinn", "physics informed", "neural network", "bpnn", "cnn lstm"]):
                desired_tags.extend(["acm-machine-learning", "acm-neural-networks"])
            if any(token in title_keywords for token in ["deep learning", "cnn lstm"]):
                desired_tags.append("acm-deep-learning")
            if "finite element" in title_keywords or "fem" in title_keywords:
                desired_tags.append("metodo-fem")
            if "thermal conductivity" in title_keywords or "heat conduction" in title_keywords:
                desired_tags.append("conduccion-de-calor")
            if "underground cable" in title_keywords or "power cable" in title_keywords:
                desired_tags.append("cables-subterraneos")

            existing_lower = {tag.lower() for tag in get_existing_item_tags(cur, item["itemID"])}
            for tag in dict.fromkeys(tag.strip() for tag in desired_tags if tag and tag.strip()):
                if tag.lower() in existing_lower:
                    continue
                tag_id = ensure_tag(cur, tag, tag_lookup)
                if attach_tag(cur, item["itemID"], tag_id):
                    summary["tags_added"] += 1
                    existing_lower.add(tag.lower())

            note_html = build_note(md_record, category_label, paper_type)
            note_result = upsert_note(cur, item, note_html, now)
            summary[f"notes_{note_result}"] += 1

            cur.execute(
                """
                UPDATE items
                SET dateModified = ?, clientDateModified = ?, synced = 0
                WHERE itemID = ?
                """,
                (now, now, item["itemID"]),
            )
            sync_queue_item(cur, LIBRARY_ID, item["key"])

        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
