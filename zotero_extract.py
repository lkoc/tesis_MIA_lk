"""
Extrae todos los ítems de la base de datos Zotero y genera un JSON completo.
"""
import sqlite3
import json
import os

DB = r"C:\Users\QU1267\Zotero\zotero.sqlite"

def get_field_id(cur, name):
    cur.execute("SELECT fieldID FROM fields WHERE fieldName=?", (name,))
    row = cur.fetchone()
    return row[0] if row else None

def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = con.cursor()

    # Obtener fieldIDs dinámicamente
    fields = {
        "title": get_field_id(cur, "title"),
        "date": get_field_id(cur, "date"),
        "abstractNote": get_field_id(cur, "abstractNote"),
        "DOI": get_field_id(cur, "DOI"),
        "url": get_field_id(cur, "url"),
        "publicationTitle": get_field_id(cur, "publicationTitle"),
        "volume": get_field_id(cur, "volume"),
        "pages": get_field_id(cur, "pages"),
    }
    print("FieldIDs:", fields)

    # Query principal
    cur.execute("""
        SELECT i.itemID, it.typeName
        FROM items i
        JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
        WHERE it.typeName NOT IN ('attachment','note')
        ORDER BY i.itemID
    """)
    items_raw = cur.fetchall()
    print(f"Items encontrados: {len(items_raw)}")

    # Para cada item obtener datos
    items = []
    for item_id, type_name in items_raw:
        data = {"id": item_id, "type": type_name}

        for fname, fid in fields.items():
            if fid is None:
                data[fname] = None
                continue
            cur.execute("""
                SELECT iv.value FROM itemData id
                JOIN itemDataValues iv ON id.valueID = iv.valueID
                WHERE id.itemID=? AND id.fieldID=?
            """, (item_id, fid))
            row = cur.fetchone()
            data[fname] = row[0] if row else None

        # Autores
        cur.execute("""
            SELECT c.firstName, c.lastName
            FROM itemCreators ic
            JOIN creators c ON ic.creatorID = c.creatorID
            WHERE ic.itemID=?
            ORDER BY ic.orderIndex
        """, (item_id,))
        authors = [f"{r[0]} {r[1]}".strip() for r in cur.fetchall()]
        data["authors"] = authors

        # Tags
        cur.execute("""
            SELECT t.name FROM itemTags it2
            JOIN tags t ON it2.tagID = t.tagID
            WHERE it2.itemID=?
        """, (item_id,))
        data["tags"] = [r[0] for r in cur.fetchall()]

        # Colecciones
        cur.execute("""
            SELECT col.collectionName FROM collectionItems ci
            JOIN collections col ON ci.collectionID = col.collectionID
            WHERE ci.itemID=?
        """, (item_id,))
        data["collections"] = [r[0] for r in cur.fetchall()]

        items.append(data)

    con.close()

    # Guardar JSON
    out = "zotero_library.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\nGuardado: {out}  ({len(items)} ítems)")

    # Mostrar los primeros 10 títulos
    print("\n--- Primeros 10 títulos ---")
    for it in items[:10]:
        print(f"  [{it['date']}] {it['title']}")

if __name__ == "__main__":
    main()
