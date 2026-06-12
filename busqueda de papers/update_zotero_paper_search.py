import hashlib
import json
import random
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIR = Path(__file__).resolve().parent
DB = Path(r"C:\Users\QU1267\Zotero\zotero.sqlite")
STORAGE = Path(r"C:\Users\QU1267\Zotero\storage")
LIBRARY_ID = 2

TOPIC_COLLECTIONS = {
    "tema-pinn": "01 tema-pinn - PINNs",
    "tema-transferencia-calor": "02 tema-transferencia-calor",
    "tema-suelo-backfill-humedad": "03 tema-suelo-backfill-humedad",
    "tema-fem-numerico": "04 tema-fem-numerico",
    "tema-cables-ampacidad": "05 tema-cables-ampacidad",
    "tema-dtr-cargas-ciclicas": "06 tema-dtr-cargas-ciclicas",
    "tema-normas-estandares": "07 tema-normas-estandares",
}

TYPE_TAGS = {
    "estado_arte": ["tipo-estado-del-arte"],
    "estado_arte_sistematico": ["tipo-estado-del-arte", "tipo-systematic-review"],
    "aporte": ["tipo-aporte", "tipo-aporte-teorico"],
    "aporte_metodologico": ["tipo-aporte", "tipo-aporte-metodologico"],
    "aplicacion": ["tipo-aplicacion"],
    "aplicacion_numerica": ["tipo-aplicacion", "tipo-aplicacion-numerica"],
    "aplicacion_normativa": ["tipo-aplicacion", "tipo-aplicacion-comparativa"],
}

PDFS = {
    "10.1007/s10915-022-01939-z": {
        "path": ROOT / "Estado del arte" / "scientific_ml_through_pinns_10.1007_s10915-022-01939-z.pdf",
        "url": "https://link.springer.com/content/pdf/10.1007/s10915-022-01939-z.pdf",
    },
    "10.1007/s11831-023-09911-2": {
        "path": ROOT / "Estado del arte" / "spectral_fem_review_10.1007_s11831-023-09911-2.pdf",
        "url": "https://link.springer.com/content/pdf/10.1007/s11831-023-09911-2.pdf",
    },
    "10.1007/s11831-026-10509-7": {
        "path": ROOT
        / "Estado del arte"
        / "soil_thermal_conductivity_measurement_review_10.1007_s11831-026-10509-7.pdf",
        "url": "https://link.springer.com/content/pdf/10.1007/s11831-026-10509-7.pdf",
    },
}

CANDIDATES = [
    {
        "doi": "10.1007/s10915-022-01939-z",
        "themes": ["tema-pinn", "tema-fem-numerico"],
        "paper_type": "estado_arte_sistematico",
        "extra_tags": ["PINN", "scientific machine learning", "Scopus search string"],
    },
    {
        "doi": "10.1007/s11831-023-09911-2",
        "themes": ["tema-fem-numerico"],
        "paper_type": "estado_arte",
        "extra_tags": ["spectral finite element method", "review", "Scopus"],
    },
    {
        "doi": "10.1007/s11831-024-10083-w",
        "themes": ["tema-fem-numerico"],
        "paper_type": "estado_arte",
        "extra_tags": ["finite element method", "spectral methods", "review"],
    },
    {
        "doi": "10.1016/0045-7825(92)90020-K",
        "themes": ["tema-fem-numerico"],
        "paper_type": "estado_arte",
        "extra_tags": ["adaptive finite element method", "computational mechanics", "review"],
    },
    {
        "doi": "10.1007/s11831-026-10509-7",
        "themes": ["tema-suelo-backfill-humedad", "tema-transferencia-calor"],
        "paper_type": "estado_arte_sistematico",
        "extra_tags": ["soil thermal conductivity", "measurement techniques", "Scopus", "Web of Science"],
    },
    {
        "doi": "10.1007/s10706-015-9843-2",
        "themes": ["tema-suelo-backfill-humedad", "tema-transferencia-calor"],
        "paper_type": "estado_arte",
        "extra_tags": ["unsaturated soils", "thermal conductivity models", "critical review"],
    },
    {
        "doi": "10.1016/j.ijthermalsci.2017.03.013",
        "themes": ["tema-suelo-backfill-humedad", "tema-transferencia-calor"],
        "paper_type": "estado_arte",
        "extra_tags": ["soil thermal conductivity", "predictive models", "review"],
    },
    {
        "doi": "10.1109/61.252604",
        "themes": ["tema-transferencia-calor", "tema-fem-numerico", "tema-cables-ampacidad"],
        "paper_type": "aporte_metodologico",
        "extra_tags": ["multi-layered soil", "thermal model", "underground power cables"],
    },
    {
        "doi": "10.1109/61.252605",
        "themes": ["tema-transferencia-calor", "tema-fem-numerico", "tema-cables-ampacidad"],
        "paper_type": "aplicacion_numerica",
        "extra_tags": ["multi-layered soil", "practical considerations", "underground power cables"],
    },
    {
        "doi": "10.1109/MIAS.2015.2417094",
        "themes": ["tema-cables-ampacidad", "tema-normas-estandares"],
        "paper_type": "estado_arte",
        "extra_tags": ["power cable rating", "historical perspective", "ampacity"],
    },
    {
        "doi": "10.1016/j.epsr.2015.12.005",
        "themes": ["tema-cables-ampacidad"],
        "paper_type": "estado_arte",
        "extra_tags": ["high current rating", "insulated cables", "review"],
    },
    {
        "doi": "10.1016/j.rser.2018.04.001",
        "themes": ["tema-dtr-cargas-ciclicas"],
        "paper_type": "estado_arte",
        "extra_tags": ["dynamic thermal rating", "review", "transmission lines"],
    },
    {
        "doi": "10.1109/61.796253",
        "themes": ["tema-normas-estandares", "tema-cables-ampacidad"],
        "paper_type": "estado_arte",
        "extra_tags": ["IEEE", "CIGRE", "ampacity standards", "comparison"],
    },
    {
        "doi": "10.1109/TIA.2015.2475244",
        "themes": ["tema-normas-estandares", "tema-cables-ampacidad"],
        "paper_type": "aplicacion_normativa",
        "extra_tags": ["cable ampacity", "comparison of methods", "standards"],
    },
    {
        "doi": "10.1016/j.epsr.2021.107395",
        "themes": ["tema-dtr-cargas-ciclicas", "tema-transferencia-calor"],
        "paper_type": "aporte_metodologico",
        "extra_tags": ["dynamic thermal analysis", "fluctuating load", "van Wormer coefficient"],
    },
    {
        "doi": "10.1109/TPWRD.2018.2849017",
        "themes": ["tema-dtr-cargas-ciclicas", "tema-suelo-backfill-humedad"],
        "paper_type": "aporte_metodologico",
        "extra_tags": ["cyclic loading", "backfill thermal resistivity", "specific heat"],
    },
]

EXISTING_UPDATES = [
    {
        "doi": "10.1016/j.jestch.2024.101658",
        "themes": ["tema-pinn", "tema-fem-numerico", "tema-transferencia-calor"],
        "type_tags": TYPE_TAGS["aplicacion_numerica"],
        "extra_tags": ["FEM-BPNN", "underground cable temperature", "soil composition"],
    },
    {
        "doi": "10.3390/a18100600",
        "themes": ["tema-pinn", "tema-cables-ampacidad"],
        "type_tags": TYPE_TAGS["aplicacion"],
        "extra_tags": ["physics-enhanced AI", "condition monitoring", "underground cables"],
    },
    {
        "doi": "10.3390/en14092591",
        "themes": ["tema-dtr-cargas-ciclicas", "tema-transferencia-calor", "tema-cables-ampacidad"],
        "type_tags": TYPE_TAGS["estado_arte"],
        "extra_tags": ["dynamic thermal rating", "underground power cables", "review"],
    },
    {
        "doi": "10.3390/en13205319",
        "themes": ["tema-transferencia-calor", "tema-cables-ampacidad"],
        "type_tags": TYPE_TAGS["estado_arte"],
        "extra_tags": ["thermal assessment", "cable current rating", "overview"],
    },
    {
        "doi": "10.1080/15325000590964425",
        "themes": ["tema-cables-ampacidad", "tema-transferencia-calor"],
        "type_tags": TYPE_TAGS["estado_arte"],
        "extra_tags": ["ampacity methods", "underground power cables", "assessment"],
    },
    {
        "doi": "10.1016/j.egypro.2019.01.636",
        "themes": ["tema-transferencia-calor", "tema-fem-numerico", "tema-cables-ampacidad"],
        "type_tags": TYPE_TAGS["aplicacion_numerica"],
        "extra_tags": ["heat transfer", "underground power cable system", "numerical study"],
    },
    {
        "doi": "10.1016/j.geothermics.2024.103151",
        "themes": ["tema-transferencia-calor", "tema-suelo-backfill-humedad", "tema-cables-ampacidad"],
        "type_tags": TYPE_TAGS["aplicacion_numerica"],
        "extra_tags": ["ambient air temperature", "ground temperature", "cable bedding"],
    },
    {
        "doi": "10.1155/etep/5946564",
        "themes": ["tema-suelo-backfill-humedad", "tema-cables-ampacidad"],
        "type_tags": TYPE_TAGS["estado_arte"],
        "extra_tags": ["soil drying out", "cable rating methods", "critical assessment"],
    },
    {
        "doi": "10.1109/access.2025.3650354",
        "themes": ["tema-dtr-cargas-ciclicas", "tema-cables-ampacidad"],
        "type_tags": TYPE_TAGS["estado_arte_sistematico"],
        "extra_tags": ["PRISMA", "dynamic thermal rating", "search string"],
    },
]


def norm_doi(value):
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    return "" if value in {"", "n/a", "na", "none"} else value.rstrip(".")


def load_crossref_records():
    path = SEARCH_DIR / "busqueda_candidates_raw.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {norm_doi(row.get("doi")): row for row in data if row.get("title")}


def unique_key(cur, table):
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    while True:
        key = "".join(random.choice(alphabet) for _ in range(8))
        if not cur.execute(f"SELECT 1 FROM {table} WHERE libraryID = ? AND key = ?", (LIBRARY_ID, key)).fetchone():
            return key


def sync_queue(cur, key, object_type_id):
    cur.execute(
        """
        INSERT OR IGNORE INTO syncQueue(libraryID, key, syncObjectTypeID, lastCheck, tries)
        VALUES (?, ?, ?, NULL, 0)
        """,
        (LIBRARY_ID, key, object_type_id),
    )


def value_id(cur, value):
    value = (value or "").strip()
    if not value:
        return None
    cur.execute("INSERT OR IGNORE INTO itemDataValues(value) VALUES (?)", (value,))
    return cur.execute("SELECT valueID FROM itemDataValues WHERE value = ?", (value,)).fetchone()["valueID"]


def set_field(cur, item_id, field_id, value):
    vid = value_id(cur, value)
    if vid is None:
        return
    cur.execute(
        "INSERT OR REPLACE INTO itemData(itemID, fieldID, valueID) VALUES (?, ?, ?)",
        (item_id, field_id, vid),
    )


def ensure_creator(cur, name):
    name = " ".join((name or "").split())
    if not name:
        return None
    parts = name.split()
    first = " ".join(parts[:-1]) if len(parts) > 1 else ""
    last = parts[-1] if len(parts) > 1 else name
    field_mode = 0
    cur.execute(
        "INSERT OR IGNORE INTO creators(firstName, lastName, fieldMode) VALUES (?, ?, ?)",
        (first, last, field_mode),
    )
    return cur.execute(
        "SELECT creatorID FROM creators WHERE firstName = ? AND lastName = ? AND fieldMode = ?",
        (first, last, field_mode),
    ).fetchone()["creatorID"]


def attach_creators(cur, item_id, authors, creator_type_id):
    for idx, author in enumerate(authors or []):
        creator_id = ensure_creator(cur, author)
        if creator_id is None:
            continue
        cur.execute(
            """
            INSERT OR IGNORE INTO itemCreators(itemID, creatorID, creatorTypeID, orderIndex)
            VALUES (?, ?, ?, ?)
            """,
            (item_id, creator_id, creator_type_id, idx),
        )


def ensure_tag(cur, tag_name, cache):
    tag_name = tag_name.strip()
    if not tag_name:
        return None
    key = tag_name.lower()
    if key in cache:
        return cache[key]
    cur.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (tag_name,))
    tag_id = cur.execute("SELECT tagID FROM tags WHERE name = ?", (tag_name,)).fetchone()["tagID"]
    cache[key] = tag_id
    return tag_id


def add_tags(cur, item_id, tags, cache):
    added = 0
    for tag in dict.fromkeys(tags):
        tag_id = ensure_tag(cur, tag, cache)
        if tag_id is None:
            continue
        cur.execute(
            "INSERT OR IGNORE INTO itemTags(itemID, tagID, type) VALUES (?, ?, 0)",
            (item_id, tag_id),
        )
        added += cur.rowcount
    return added


def date_from_record(record):
    parts = (record.get("issued") or record.get("published_online") or record.get("published_print") or [[]])[0]
    if not parts:
        return ""
    if len(parts) >= 3:
        return f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
    if len(parts) == 2:
        return f"{parts[0]:04d}-{parts[1]:02d}"
    return str(parts[0])


def item_by_doi(cur, doi, fields):
    if not doi:
        return None
    rows = cur.execute(
        """
        SELECT i.itemID, i.key, v.value AS doi
        FROM items i
        JOIN itemData d ON d.itemID = i.itemID AND d.fieldID = ?
        JOIN itemDataValues v ON v.valueID = d.valueID
        WHERE i.libraryID = ?
          AND i.itemID NOT IN (SELECT itemID FROM itemAttachments)
          AND i.itemID NOT IN (SELECT itemID FROM itemNotes)
        """,
        (fields["DOI"], LIBRARY_ID),
    )
    for row in rows:
        if norm_doi(row["doi"]) == doi:
            return row
    return None


def create_item(cur, record, fields, item_type_id, creator_type_id, now):
    key = unique_key(cur, "items")
    cur.execute(
        """
        INSERT INTO items(itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
        VALUES (?, ?, ?, ?, ?, ?, 0, 0)
        """,
        (item_type_id, now, now, now, LIBRARY_ID, key),
    )
    item_id = cur.lastrowid
    set_field(cur, item_id, fields["title"], record.get("title"))
    set_field(cur, item_id, fields["DOI"], record.get("doi"))
    set_field(cur, item_id, fields["url"], record.get("URL") or f"https://doi.org/{record.get('doi')}")
    set_field(cur, item_id, fields["publicationTitle"], record.get("container"))
    set_field(cur, item_id, fields["date"], date_from_record(record))
    set_field(cur, item_id, fields["volume"], record.get("volume"))
    set_field(cur, item_id, fields["issue"], record.get("issue"))
    set_field(cur, item_id, fields["pages"], record.get("page"))
    set_field(cur, item_id, fields["publisher"], record.get("publisher"))
    set_field(cur, item_id, fields["abstractNote"], record.get("abstract"))
    set_field(cur, item_id, fields["accessDate"], datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    attach_creators(cur, item_id, record.get("authors"), creator_type_id)
    sync_queue(cur, key, 3)
    return {"itemID": item_id, "key": key, "created": True}


def collection_ids(cur):
    parent = cur.execute(
        """
        SELECT collectionID
        FROM collections
        WHERE libraryID = ? AND parentCollectionID IS NULL AND collectionName = 'Tesis_MIA_2026'
        """,
        (LIBRARY_ID,),
    ).fetchone()
    if not parent:
        raise RuntimeError("No existe la coleccion raiz Tesis_MIA_2026")
    ids = {}
    for tag, name in TOPIC_COLLECTIONS.items():
        row = cur.execute(
            """
            SELECT collectionID, key FROM collections
            WHERE libraryID = ? AND parentCollectionID = ? AND collectionName = ?
            """,
            (LIBRARY_ID, parent["collectionID"], name),
        ).fetchone()
        if not row:
            raise RuntimeError(f"No existe la coleccion {name}")
        ids[tag] = row
    return ids


def add_to_collections(cur, item_id, item_key, theme_tags, col_ids, now):
    added = 0
    for theme in theme_tags:
        row = col_ids[theme]
        cur.execute(
            "INSERT OR IGNORE INTO collectionItems(collectionID, itemID, orderIndex) VALUES (?, ?, 0)",
            (row["collectionID"], item_id),
        )
        added += cur.rowcount
        cur.execute("UPDATE collections SET clientDateModified = ?, synced = 0 WHERE collectionID = ?", (now, row["collectionID"]))
        sync_queue(cur, row["key"], 1)
    sync_queue(cur, item_key, 3)
    return added


def has_pdf_attachment(cur, item_id):
    return cur.execute(
        """
        SELECT 1 FROM itemAttachments
        WHERE parentItemID = ? AND lower(COALESCE(contentType, '')) = 'application/pdf'
        LIMIT 1
        """,
        (item_id,),
    ).fetchone() is not None


def add_pdf_attachment(cur, item_id, pdf_info, fields, attachment_type_id, now):
    pdf_path = pdf_info["path"]
    if not pdf_path.exists() or pdf_path.stat().st_size < 10000:
        return False
    key = unique_key(cur, "items")
    storage_dir = STORAGE / key
    storage_dir.mkdir(parents=True, exist_ok=True)
    dest = storage_dir / pdf_path.name
    if not dest.exists():
        shutil.copy2(pdf_path, dest)
    data = dest.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    mtime_ms = int(dest.stat().st_mtime * 1000)
    cur.execute(
        """
        INSERT INTO items(itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
        VALUES (?, ?, ?, ?, ?, ?, 0, 0)
        """,
        (attachment_type_id, now, now, now, LIBRARY_ID, key),
    )
    attachment_id = cur.lastrowid
    cur.execute(
        """
        INSERT INTO itemAttachments(
            itemID, parentItemID, linkMode, contentType, charsetID, path,
            syncState, storageModTime, storageHash, lastProcessedModificationTime, lastRead
        )
        VALUES (?, ?, 1, 'application/pdf', NULL, ?, 0, ?, ?, NULL, NULL)
        """,
        (attachment_id, item_id, f"storage:{pdf_path.name}", mtime_ms, md5),
    )
    set_field(cur, attachment_id, fields["title"], "Full Text PDF")
    set_field(cur, attachment_id, fields["url"], pdf_info.get("url"))
    set_field(cur, attachment_id, fields["accessDate"], datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    sync_queue(cur, key, 3)
    return True


def backup_db():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = DB.with_name(f"zotero.sqlite.codex-paper-search-backup-{timestamp}")
    src = sqlite3.connect(str(DB))
    dst = sqlite3.connect(str(backup))
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    return backup


def main():
    records = load_crossref_records()
    if not records:
        raise SystemExit("No se encontro busqueda_candidates_raw.json con metadatos Crossref.")
    backup = backup_db()

    con = sqlite3.connect(str(DB), timeout=30)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("PRAGMA busy_timeout = 30000")
    fields = {row["fieldName"]: row["fieldID"] for row in cur.execute("SELECT fieldID, fieldName FROM fields")}
    item_type_id = cur.execute("SELECT itemTypeID FROM itemTypes WHERE typeName = 'journalArticle'").fetchone()["itemTypeID"]
    attachment_type_id = cur.execute("SELECT itemTypeID FROM itemTypes WHERE typeName = 'attachment'").fetchone()["itemTypeID"]
    creator_type_id = cur.execute("SELECT creatorTypeID FROM creatorTypes WHERE creatorType = 'author'").fetchone()["creatorTypeID"]
    tag_cache = {row["name"].lower(): row["tagID"] for row in cur.execute("SELECT tagID, name FROM tags")}
    col_ids = collection_ids(cur)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "backup": str(backup),
        "created_items": 0,
        "updated_existing_items": 0,
        "collection_memberships_added": 0,
        "tags_added": 0,
        "pdfs_attached": 0,
        "missing_pdf": [],
        "missing_metadata": [],
    }

    cur.execute("BEGIN IMMEDIATE")
    try:
        for candidate in CANDIDATES:
            doi = norm_doi(candidate["doi"])
            record = records.get(doi)
            if not record:
                summary["missing_metadata"].append(doi)
                continue
            item = item_by_doi(cur, doi, fields)
            if item:
                item = {"itemID": item["itemID"], "key": item["key"], "created": False}
                summary["updated_existing_items"] += 1
            else:
                record = dict(record)
                record["doi"] = doi
                item = create_item(cur, record, fields, item_type_id, creator_type_id, now)
                summary["created_items"] += 1

            tags = [
                "tesis-mia-busqueda-2026-06-05",
                *candidate["themes"],
                *TYPE_TAGS[candidate["paper_type"]],
                *candidate.get("extra_tags", []),
            ]
            summary["tags_added"] += add_tags(cur, item["itemID"], tags, tag_cache)
            summary["collection_memberships_added"] += add_to_collections(
                cur, item["itemID"], item["key"], candidate["themes"], col_ids, now
            )

            pdf_info = PDFS.get(doi)
            if pdf_info and not has_pdf_attachment(cur, item["itemID"]):
                if add_pdf_attachment(cur, item["itemID"], pdf_info, fields, attachment_type_id, now):
                    summary["pdfs_attached"] += 1
                else:
                    summary["missing_pdf"].append(doi)
            elif not pdf_info:
                summary["missing_pdf"].append(doi)

            cur.execute(
                "UPDATE items SET dateModified = ?, clientDateModified = ?, synced = 0 WHERE itemID = ?",
                (now, now, item["itemID"]),
            )
            sync_queue(cur, item["key"], 3)

        for update in EXISTING_UPDATES:
            doi = norm_doi(update["doi"])
            item = item_by_doi(cur, doi, fields)
            if not item:
                continue
            summary["updated_existing_items"] += 1
            tags = [
                "tesis-mia-busqueda-2026-06-05",
                *update["themes"],
                *update["type_tags"],
                *update.get("extra_tags", []),
            ]
            summary["tags_added"] += add_tags(cur, item["itemID"], tags, tag_cache)
            summary["collection_memberships_added"] += add_to_collections(
                cur, item["itemID"], item["key"], update["themes"], col_ids, now
            )
            cur.execute(
                "UPDATE items SET dateModified = ?, clientDateModified = ?, synced = 0 WHERE itemID = ?",
                (now, now, item["itemID"]),
            )
            sync_queue(cur, item["key"], 3)

        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
