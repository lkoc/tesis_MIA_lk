import html
import json
import random
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DB = Path(r"C:\Users\QU1267\Zotero\zotero.sqlite")
LIBRARY_ID = 2
ROOT_COLLECTION = "Tesis_MIA_2026"
OLD_NOTE_TITLE = "[Tesis MIA] Aporte a la tesis"
NOTE_TITLE = "[Tesis MIA] Aporte a la investigación"

COLLECTION_THEME = {
    "01 tema-pinn - PINNs": "tema-pinn",
    "02 tema-transferencia-calor": "tema-transferencia-calor",
    "03 tema-suelo-backfill-humedad": "tema-suelo-backfill-humedad",
    "04 tema-fem-numerico": "tema-fem-numerico",
    "05 tema-cables-ampacidad": "tema-cables-ampacidad",
    "06 tema-dtr-cargas-ciclicas": "tema-dtr-cargas-ciclicas",
    "07 tema-normas-estandares": "tema-normas-estandares",
}

THEME_LABELS = {
    "tema-pinn": "PINNs e IA física",
    "tema-transferencia-calor": "transferencia de calor",
    "tema-suelo-backfill-humedad": "suelo/backfill/humedad",
    "tema-fem-numerico": "FEM y métodos numéricos",
    "tema-cables-ampacidad": "cables y ampacidad",
    "tema-dtr-cargas-ciclicas": "DTR y cargas cíclicas",
    "tema-normas-estandares": "normas y estándares",
}

TYPE_LABELS = {
    "tipo-estado-del-arte": "estado del arte",
    "tipo-aporte": "aporte metodológico/teórico",
    "tipo-aplicacion": "aplicación",
    "tipo-normativa": "normativa",
    "tipo-guia-tecnica": "guía técnica",
    "tipo-apoyo": "apoyo académico",
}

TYPE_OVERRIDES_BY_DOI = {
    "10.3390/en14092591": "tipo-estado-del-arte",
    "10.1016/j.epsr.2021.107395": "tipo-aporte",
    "10.1080/15325000590964425": "tipo-estado-del-arte",
    "10.1016/j.rser.2025.115348": "tipo-aplicacion",
    "10.1109/tia.2015.2475244": "tipo-aplicacion",
    "10.1109/61.796253": "tipo-estado-del-arte",
}

TYPE_OVERRIDES_BY_TITLE = [
    ("thermal capacity and loading assessment", "tipo-aplicacion"),
    ("thermal bottleneck determination", "tipo-aplicacion"),
    ("dynamic rating of power cables based upon transient temperature calculations", "tipo-aplicacion"),
    ("power cable rating examples for calculation tool verification", "tipo-aporte"),
    ("rating of electric power cables ampacity computations", "tipo-aporte"),
    ("power cable rating calculations a historical perspective", "tipo-estado-del-arte"),
]


def norm_text(value):
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def norm_doi(value):
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    return "" if value in {"", "n/a", "na", "none"} else value.rstrip(".")


def backup_db():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = DB.with_name(f"zotero.sqlite.codex-clean-tags-backup-{timestamp}")
    shutil.copy2(DB, backup)
    return backup


def unique_key(cur, table):
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    while True:
        key = "".join(random.choice(alphabet) for _ in range(8))
        exists = cur.execute(f"SELECT 1 FROM {table} WHERE libraryID = ? AND key = ?", (LIBRARY_ID, key)).fetchone()
        if not exists:
            return key


def sync_queue(cur, key, object_type_id):
    cur.execute(
        """
        INSERT OR IGNORE INTO syncQueue(libraryID, key, syncObjectTypeID, lastCheck, tries)
        VALUES (?, ?, ?, NULL, 0)
        """,
        (LIBRARY_ID, key, object_type_id),
    )


def get_fields(cur):
    return {row["fieldName"]: row["fieldID"] for row in cur.execute("SELECT fieldID, fieldName FROM fields")}


def get_thesis_items(cur, fields):
    return cur.execute(
        """
        WITH RECURSIVE tree(collectionID) AS (
            SELECT collectionID FROM collections WHERE collectionName = ?
            UNION ALL
            SELECT c.collectionID FROM collections c JOIN tree t ON c.parentCollectionID = t.collectionID
        )
        SELECT DISTINCT
            i.itemID, i.key, it.typeName,
            COALESCE(vt.value, '') AS title,
            COALESCE(vd.value, '') AS doi,
            COALESCE(va.value, '') AS abstractNote
        FROM collectionItems ci
        JOIN tree tr ON tr.collectionID = ci.collectionID
        JOIN items i ON i.itemID = ci.itemID
        JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
        LEFT JOIN itemData dt ON dt.itemID = i.itemID AND dt.fieldID = ?
        LEFT JOIN itemDataValues vt ON vt.valueID = dt.valueID
        LEFT JOIN itemData dd ON dd.itemID = i.itemID AND dd.fieldID = ?
        LEFT JOIN itemDataValues vd ON vd.valueID = dd.valueID
        LEFT JOIN itemData da ON da.itemID = i.itemID AND da.fieldID = ?
        LEFT JOIN itemDataValues va ON va.valueID = da.valueID
        WHERE i.libraryID = ?
          AND i.itemID NOT IN (SELECT itemID FROM itemAttachments)
          AND i.itemID NOT IN (SELECT itemID FROM itemNotes)
        ORDER BY title COLLATE NOCASE
        """,
        (ROOT_COLLECTION, fields["title"], fields["DOI"], fields["abstractNote"], LIBRARY_ID),
    ).fetchall()


def get_item_tags(cur, item_id):
    return [row["name"] for row in cur.execute(
        "SELECT t.name FROM itemTags it JOIN tags t ON t.tagID = it.tagID WHERE it.itemID = ?",
        (item_id,),
    )]


def get_item_collections(cur, item_id):
    return [row["collectionName"] for row in cur.execute(
        """
        SELECT c.collectionName
        FROM collectionItems ci JOIN collections c ON c.collectionID = ci.collectionID
        WHERE ci.itemID = ?
        """,
        (item_id,),
    )]


def add_if(haystack, tags, tag, *needles):
    if any(needle in haystack for needle in needles):
        tags.add(tag)


def infer_type(item, title_norm, tags_norm):
    doi = norm_doi(item["doi"])
    if doi in TYPE_OVERRIDES_BY_DOI:
        return TYPE_OVERRIDES_BY_DOI[doi]
    for title_fragment, paper_type in TYPE_OVERRIDES_BY_TITLE:
        if title_fragment in title_norm:
            return paper_type

    haystack = f"{title_norm} {' '.join(tags_norm)} {norm_text(item['typeName'])}"

    if re.search(r"\b(iec\s*60[0-9]+|ieee std|astm|standard test method)\b", haystack):
        return "tipo-normativa"

    if (
        "tipo systematic review" in haystack
        or "tipo estado del arte" in haystack
        or re.search(r"\b(review|overview|systematic|mapping|assessment|survey|historical perspective|critical review)\b", haystack)
    ):
        return "tipo-estado-del-arte"

    if (
        "tipo aplicacion" in haystack
        or re.search(r"\b(application|case study|numerical study|experiment|experimental|measurement|modeling and analysis|effect of|comparison of methods|practical considerations)\b", haystack)
    ):
        return "tipo-aplicacion"

    if re.search(r"\b(handbook|guide|calculation tool verification|catalog|catalogo|examples)\b", haystack):
        return "tipo-guia-tecnica"

    if re.search(r"\b(heat equation examples|educational|didactic|apoyo)\b", haystack):
        return "tipo-apoyo"

    return "tipo-aporte"


def infer_canonical_tags(item, current_tags, collection_names):
    title_norm = norm_text(item["title"])
    tags_norm = [norm_text(tag) for tag in current_tags]
    haystack = " ".join([title_norm, norm_text(item["abstractNote"]), *tags_norm])

    final = set()

    for collection in collection_names:
        if collection in COLLECTION_THEME:
            final.add(COLLECTION_THEME[collection])

    final.add(infer_type(item, title_norm, tags_norm))

    # Controlled technical tags.
    add_if(haystack, final, "metodo-pinn", "pinn", "physics informed", "scientific machine learning")
    add_if(haystack, final, "metodo-machine-learning", "machine learning", "deep learning", "neural network", "bpnn", "cnn lstm")
    add_if(haystack, final, "metodo-fem", "finite element", "fem", "spectral finite")
    add_if(haystack, final, "metodo-fdm-fvm", "finite difference", "finite volume", "fdm", "fvm")
    add_if(haystack, final, "metodo-analitico", "analytical", "neher", "green", "separation of variables", "fourier")
    add_if(haystack, final, "metodo-optimizacion", "optimization", "pso", "particle swarm")
    add_if(haystack, final, "metodo-medicion", "measurement", "thermal probe", "test method", "in situ", "needle probe")
    add_if(haystack, final, "metodo-benchmark", "benchmark", "verification", "calculation tool")
    add_if(haystack, final, "metodo-dtr", "dynamic thermal rating", "dtr", "dynamic rating")
    add_if(haystack, final, "metodo-iec-60287", "iec 60287")
    add_if(haystack, final, "metodo-iec-60853", "iec 60853")
    add_if(haystack, final, "metodo-ieee", "ieee")
    add_if(haystack, final, "metodo-cigre", "cigre")

    add_if(haystack, final, "revision-sistematica", "systematic review", "systematic mapping", "prisma", "search string", "bibliometric")
    add_if(haystack, final, "objeto-cables-subterraneos", "underground cable", "buried cable", "underground power cable")
    add_if(haystack, final, "material-suelo-backfill", "soil", "backfill", "bedding", "unsaturated")
    add_if(haystack, final, "propiedad-ampacidad", "ampacity", "current rating", "thermal rating", "load capability")
    add_if(haystack, final, "propiedad-conductividad-termica", "thermal conductivity", "heat conduction", "conduction")
    add_if(haystack, final, "propiedad-resistividad-termica", "thermal resistivity", "resistivity")
    add_if(haystack, final, "fenomeno-temperatura-conductor", "conductor temperature", "temperature rise", "temperature field")
    add_if(haystack, final, "condicion-cargas-ciclicas", "cyclic loading", "cyclic rating", "fluctuating load", "emergency rating", "transient temperature")
    add_if(haystack, final, "condicion-heterogeneidad-termica", "heterogeneous", "inhomogeneous", "multi layered", "varied soil", "soil composition")

    if "thesis core" in haystack or "thesis-core" in current_tags:
        final.add("tesis-core")

    return sorted(final)


def ensure_tag(cur, cache, name):
    if name in cache:
        return cache[name]
    cur.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,))
    tag_id = cur.execute("SELECT tagID FROM tags WHERE name = ?", (name,)).fetchone()["tagID"]
    cache[name] = tag_id
    return tag_id


def replace_item_tags(cur, item, desired_tags, tag_cache):
    current = {
        row["name"]: row["tagID"]
        for row in cur.execute(
            "SELECT t.name, t.tagID FROM itemTags it JOIN tags t ON t.tagID = it.tagID WHERE it.itemID = ?",
            (item["itemID"],),
        )
    }

    desired = {tag: ensure_tag(cur, tag_cache, tag) for tag in desired_tags}
    current_ids = set(current.values())
    desired_ids = set(desired.values())

    for tag_id in current_ids - desired_ids:
        cur.execute("DELETE FROM itemTags WHERE itemID = ? AND tagID = ?", (item["itemID"], tag_id))

    added = 0
    for tag_id in desired_ids - current_ids:
        cur.execute("INSERT OR IGNORE INTO itemTags(itemID, tagID, type) VALUES (?, ?, 0)", (item["itemID"], tag_id))
        added += cur.rowcount

    return {
        "removed": len(current_ids - desired_ids),
        "added": added,
        "before": len(current_ids),
        "after": len(desired_ids),
    }


def build_aporte_note(item, tags):
    themes = [tag for tag in tags if tag.startswith("tema-")]
    type_tag = next((tag for tag in tags if tag in TYPE_LABELS), "tipo-aporte")

    type_intro = {
        "tipo-estado-del-arte": "ubica el trabajo dentro del estado del arte",
        "tipo-aporte": "aporta una formulación, método o fundamento técnico",
        "tipo-aplicacion": "aplica una metodología a un caso concreto",
        "tipo-normativa": "define criterios normativos o de referencia",
        "tipo-guia-tecnica": "sirve como guía técnica y referencia de verificación",
        "tipo-apoyo": "apoya el marco conceptual y didáctico",
    }[type_tag]

    theme_bits = []
    if "tema-pinn" in themes:
        theme_bits.append("sustenta el uso de modelos informados por física o híbridos para estimar campos térmicos")
    if "tema-transferencia-calor" in themes:
        theme_bits.append("refuerza el modelo de conducción de calor y las condiciones de frontera")
    if "tema-suelo-backfill-humedad" in themes:
        theme_bits.append("caracteriza cómo suelo, humedad y backfill modifican la transferencia térmica")
    if "tema-fem-numerico" in themes:
        theme_bits.append("sirve como referencia numérica para validar el modelo 2D")
    if "tema-cables-ampacidad" in themes:
        theme_bits.append("conecta temperatura del conductor con ampacidad y diseño de cables enterrados")
    if "tema-dtr-cargas-ciclicas" in themes:
        theme_bits.append("extiende el análisis hacia cargas variables y rating térmico dinámico")
    if "tema-normas-estandares" in themes:
        theme_bits.append("aporta la línea base normativa para contrastar resultados")

    if not theme_bits:
        theme_bits.append("aporta contexto técnico para el modelado térmico de cables subterráneos")

    aporte = f"Este trabajo {type_intro}; {', '.join(theme_bits[:3])}."
    if len(theme_bits) > 3:
        aporte += " Además, complementa otras líneas temáticas de la tesis."

    theme_label = ", ".join(THEME_LABELS.get(tag, tag) for tag in themes)
    type_label = TYPE_LABELS.get(type_tag, type_tag)

    return (
        '<div class="zotero-note znv1">'
        f"<p><strong>Aporte a la investigación:</strong> {html.escape(aporte)}</p>"
        f"<p><strong>Clasificación depurada:</strong> {html.escape(type_label)}; {html.escape(theme_label)}.</p>"
        "</div>"
    )


def upsert_standard_note(cur, item, note_html, now):
    notes = cur.execute(
        """
        SELECT i.itemID, i.key, n.title
        FROM itemNotes n JOIN items i ON i.itemID = n.itemID
        WHERE n.parentItemID = ?
          AND (n.title = ? OR n.title = ?)
        ORDER BY CASE WHEN n.title = ? THEN 0 ELSE 1 END, i.itemID
        """,
        (item["itemID"], NOTE_TITLE, OLD_NOTE_TITLE, NOTE_TITLE),
    ).fetchall()

    if notes:
        keeper = notes[0]
        cur.execute("UPDATE itemNotes SET title = ?, note = ? WHERE itemID = ?", (NOTE_TITLE, note_html, keeper["itemID"]))
        cur.execute(
            "UPDATE items SET dateModified = ?, clientDateModified = ?, synced = 0 WHERE itemID = ?",
            (now, now, keeper["itemID"]),
        )
        sync_queue(cur, keeper["key"], 3)
        deleted = 0
        for extra in notes[1:]:
            cur.execute("DELETE FROM itemNotes WHERE itemID = ?", (extra["itemID"],))
            cur.execute("DELETE FROM items WHERE itemID = ?", (extra["itemID"],))
            deleted += 1
        return "updated", deleted

    note_key = unique_key(cur, "items")
    cur.execute(
        """
        INSERT INTO items(itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
        VALUES ((SELECT itemTypeID FROM itemTypes WHERE typeName = 'note'), ?, ?, ?, ?, ?, 0, 0)
        """,
        (now, now, now, LIBRARY_ID, note_key),
    )
    note_id = cur.lastrowid
    cur.execute(
        "INSERT INTO itemNotes(itemID, parentItemID, note, title) VALUES (?, ?, ?, ?)",
        (note_id, item["itemID"], note_html, NOTE_TITLE),
    )
    sync_queue(cur, note_key, 3)
    return "inserted", 0


def main():
    if not DB.exists():
        raise SystemExit(f"No existe la base Zotero: {DB}")

    backup = backup_db()
    con = sqlite3.connect(str(DB), timeout=30)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("PRAGMA busy_timeout = 30000")

    fields = get_fields(cur)
    items = get_thesis_items(cur, fields)
    tag_cache = {row["name"]: row["tagID"] for row in cur.execute("SELECT tagID, name FROM tags")}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "backup": str(backup),
        "items": len(items),
        "tag_links_removed": 0,
        "tag_links_added": 0,
        "max_tags_before": 0,
        "max_tags_after": 0,
        "notes_inserted": 0,
        "notes_updated": 0,
        "duplicate_standard_notes_deleted": 0,
        "type_counts": Counter(),
        "theme_counts": Counter(),
    }

    cur.execute("BEGIN IMMEDIATE")
    try:
        for item in items:
            current_tags = get_item_tags(cur, item["itemID"])
            collections = get_item_collections(cur, item["itemID"])
            desired_tags = infer_canonical_tags(item, current_tags, collections)
            tag_result = replace_item_tags(cur, item, desired_tags, tag_cache)
            summary["tag_links_removed"] += tag_result["removed"]
            summary["tag_links_added"] += tag_result["added"]
            summary["max_tags_before"] = max(summary["max_tags_before"], tag_result["before"])
            summary["max_tags_after"] = max(summary["max_tags_after"], tag_result["after"])

            note_html = build_aporte_note(item, desired_tags)
            note_status, deleted = upsert_standard_note(cur, item, note_html, now)
            summary[f"notes_{note_status}"] += 1
            summary["duplicate_standard_notes_deleted"] += deleted

            for tag in desired_tags:
                if tag.startswith("tema-"):
                    summary["theme_counts"][tag] += 1
                if tag in TYPE_LABELS:
                    summary["type_counts"][tag] += 1

            cur.execute(
                "UPDATE items SET dateModified = ?, clientDateModified = ?, synced = 0 WHERE itemID = ?",
                (now, now, item["itemID"]),
            )
            sync_queue(cur, item["key"], 3)

        # Remove tags no longer used anywhere, so Zotero's tag pane stays clean.
        orphan_rows = cur.execute(
            "SELECT tagID FROM tags WHERE tagID NOT IN (SELECT DISTINCT tagID FROM itemTags)"
        ).fetchall()
        for row in orphan_rows:
            cur.execute("DELETE FROM tags WHERE tagID = ?", (row["tagID"],))
        summary["orphan_tags_deleted"] = len(orphan_rows)

        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    summary["type_counts"] = dict(summary["type_counts"])
    summary["theme_counts"] = dict(summary["theme_counts"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
