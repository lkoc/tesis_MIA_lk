import json
import random
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DB = Path(r"C:\Users\QU1267\Zotero\zotero.sqlite")
LIBRARY_ID = 2
PARENT_COLLECTION_NAME = "Tesis_MIA_2026"
OLD_PINN_COLLECTION_NAME = "PINNs_calculo_termico_cables_enterrados_Tesis_MIA_2026"

TOPIC_COLLECTIONS = [
    ("tema-pinn", "01 tema-pinn - PINNs"),
    ("tema-transferencia-calor", "02 tema-transferencia-calor"),
    ("tema-suelo-backfill-humedad", "03 tema-suelo-backfill-humedad"),
    ("tema-fem-numerico", "04 tema-fem-numerico"),
    ("tema-cables-ampacidad", "05 tema-cables-ampacidad"),
    ("tema-dtr-cargas-ciclicas", "06 tema-dtr-cargas-ciclicas"),
    ("tema-normas-estandares", "07 tema-normas-estandares"),
]


def unique_key(cur, table):
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    while True:
        key = "".join(random.choice(alphabet) for _ in range(8))
        exists = cur.execute(
            f"SELECT 1 FROM {table} WHERE libraryID = ? AND key = ?",
            (LIBRARY_ID, key),
        ).fetchone()
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


def top_level_item_ids_for_tag(cur, tag_name):
    rows = cur.execute(
        """
        SELECT DISTINCT i.itemID, i.key
        FROM items i
        JOIN itemTags it ON it.itemID = i.itemID
        JOIN tags t ON t.tagID = it.tagID
        WHERE i.libraryID = ?
          AND t.name = ?
          AND i.itemID NOT IN (SELECT itemID FROM itemAttachments)
          AND i.itemID NOT IN (SELECT itemID FROM itemNotes)
          AND i.itemTypeID NOT IN (1, 3, 28)
        ORDER BY i.itemID
        """,
        (LIBRARY_ID, tag_name),
    )
    return [(row["itemID"], row["key"]) for row in rows]


def ensure_topic_collection(cur, parent_id, collection_name, now):
    existing = cur.execute(
        """
        SELECT collectionID, key, collectionName
        FROM collections
        WHERE libraryID = ? AND parentCollectionID = ? AND collectionName = ?
        """,
        (LIBRARY_ID, parent_id, collection_name),
    ).fetchone()
    if existing:
        return existing["collectionID"], existing["key"], "existing"

    if collection_name == TOPIC_COLLECTIONS[0][1]:
        old = cur.execute(
            """
            SELECT collectionID, key
            FROM collections
            WHERE libraryID = ? AND parentCollectionID = ? AND collectionName = ?
            """,
            (LIBRARY_ID, parent_id, OLD_PINN_COLLECTION_NAME),
        ).fetchone()
        if old:
            cur.execute(
                """
                UPDATE collections
                SET collectionName = ?, clientDateModified = ?, synced = 0
                WHERE collectionID = ?
                """,
                (collection_name, now, old["collectionID"]),
            )
            sync_queue(cur, old["key"], 1)
            return old["collectionID"], old["key"], "renamed"

    key = unique_key(cur, "collections")
    cur.execute(
        """
        INSERT INTO collections(collectionName, parentCollectionID, clientDateModified, libraryID, key, version, synced)
        VALUES (?, ?, ?, ?, ?, 0, 0)
        """,
        (collection_name, parent_id, now, LIBRARY_ID, key),
    )
    collection_id = cur.lastrowid
    sync_queue(cur, key, 1)
    return collection_id, key, "created"


def set_collection_membership(cur, collection_id, item_rows, now):
    desired_item_ids = {item_id for item_id, _ in item_rows}
    existing_item_ids = {
        row["itemID"]
        for row in cur.execute("SELECT itemID FROM collectionItems WHERE collectionID = ?", (collection_id,))
    }

    removed = existing_item_ids - desired_item_ids
    added = desired_item_ids - existing_item_ids

    for item_id in removed:
        cur.execute(
            "DELETE FROM collectionItems WHERE collectionID = ? AND itemID = ?",
            (collection_id, item_id),
        )

    for order_index, item_id in enumerate(sorted(added), start=1):
        cur.execute(
            "INSERT OR IGNORE INTO collectionItems(collectionID, itemID, orderIndex) VALUES (?, ?, ?)",
            (collection_id, item_id, order_index),
        )

    for item_id, item_key in item_rows:
        cur.execute(
            """
            UPDATE items
            SET dateModified = ?, clientDateModified = ?, synced = 0
            WHERE itemID = ?
            """,
            (now, now, item_id),
        )
        sync_queue(cur, item_key, 3)

    return len(added), len(removed)


def main():
    if not DB.exists():
        raise SystemExit(f"No existe la base Zotero: {DB}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = DB.with_name(f"zotero.sqlite.codex-collections-backup-{timestamp}")
    src = sqlite3.connect(str(DB))
    dst = sqlite3.connect(str(backup_path))
    with dst:
        src.backup(dst)
    src.close()
    dst.close()

    con = sqlite3.connect(str(DB), timeout=20)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("PRAGMA busy_timeout = 10000")

    parent = cur.execute(
        """
        SELECT collectionID, key
        FROM collections
        WHERE libraryID = ? AND parentCollectionID IS NULL AND collectionName = ?
        """,
        (LIBRARY_ID, PARENT_COLLECTION_NAME),
    ).fetchone()
    if not parent:
        raise SystemExit(f"No existe la colección raíz {PARENT_COLLECTION_NAME!r}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    summary = {
        "backup": str(backup_path),
        "parent_collection_id": parent["collectionID"],
        "collections_created": 0,
        "collections_renamed": 0,
        "collections_existing": 0,
        "memberships_added": 0,
        "memberships_removed": 0,
        "topic_counts": {},
        "collection_ids": {},
    }

    cur.execute("BEGIN IMMEDIATE")
    try:
        for tag_name, collection_name in TOPIC_COLLECTIONS:
            collection_id, collection_key, status = ensure_topic_collection(
                cur,
                parent["collectionID"],
                collection_name,
                now,
            )
            summary[f"collections_{status}"] += 1
            summary["collection_ids"][collection_name] = collection_id

            item_rows = top_level_item_ids_for_tag(cur, tag_name)
            added, removed = set_collection_membership(cur, collection_id, item_rows, now)
            summary["memberships_added"] += added
            summary["memberships_removed"] += removed
            summary["topic_counts"][collection_name] = len(item_rows)

            cur.execute(
                """
                UPDATE collections
                SET clientDateModified = ?, synced = 0
                WHERE collectionID = ?
                """,
                (now, collection_id),
            )
            sync_queue(cur, collection_key, 1)

        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
