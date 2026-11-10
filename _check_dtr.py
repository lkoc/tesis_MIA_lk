"""
1. Find the DTR paper node in NODES
2. Check if 3YNDYEIQ is in the DB
3. Also: for all nodes with local_pdf but not matched, show what att key they have
4. Then build the full patch script
"""
import sqlite3, re, json
from pathlib import Path

DB = r"C:\Users\QU1267\AppData\Local\Temp\zotero_ro.sqlite"
GROUP_ID = 6532672
SLUG = "tesis_mia_2026"
BASE = f"https://www.zotero.org/groups/{GROUP_ID}/{SLUG}/items"

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = conn.cursor()

# Get libraryID
cur.execute("SELECT libraryID FROM groups WHERE groupID=?", (GROUP_ID,))
lib_id = cur.fetchone()[0]

# Check if 3YNDYEIQ exists
cur.execute("SELECT itemID, key FROM items WHERE libraryID=? AND key='3YNDYEIQ'", (lib_id,))
row = cur.fetchone()
print(f"3YNDYEIQ in DB: {row}")

# Check L9K8V826
cur.execute("SELECT itemID, key FROM items WHERE libraryID=? AND key='L9K8V826'", (lib_id,))
row2 = cur.fetchone()
print(f"L9K8V826 in DB: {row2}")

# If 3YNDYEIQ doesn't exist in this library, maybe it's in another library
cur.execute("SELECT itemID, key, libraryID FROM items WHERE key='3YNDYEIQ'")
all_rows = cur.fetchall()
print(f"3YNDYEIQ in ANY library: {all_rows}")

cur.execute("SELECT itemID, key, libraryID FROM items WHERE key='L9K8V826'")
all_rows2 = cur.fetchall()
print(f"L9K8V826 in ANY library: {all_rows2}")

# Search for DTR mapping paper by title
cur.execute("""
    SELECT i.itemID, i.key, ifv.value
    FROM items i
    JOIN itemData id2 ON i.itemID=id2.itemID
    JOIN itemDataValues ifv ON id2.valueID=ifv.valueID
    JOIN fields f ON id2.fieldID=f.fieldID
    WHERE i.libraryID=? AND f.fieldName='title' AND ifv.value LIKE '%Systematic Mapping%Dynamic Thermal Rating%'
""", (lib_id,))
rows = cur.fetchall()
print(f"\nDTR Systematic Mapping items: {rows}")

# Also search in all libraries
cur.execute("""
    SELECT i.itemID, i.key, i.libraryID, ifv.value
    FROM items i
    JOIN itemData id2 ON i.itemID=id2.itemID
    JOIN itemDataValues ifv ON id2.valueID=ifv.valueID
    JOIN fields f ON id2.fieldID=f.fieldID
    WHERE f.fieldName='title' AND ifv.value LIKE '%Systematic%Dynamic Thermal Rating%'
""")
all_dtr = cur.fetchall()
print(f"DTR Systematic Mapping in ALL libraries: {len(all_dtr)}")
for r in all_dtr:
    print(f"  itemID={r[0]}, key={r[1]}, libID={r[2]}: {r[3][:80]}")

# Get all items in library 2 with their titles for the DTR category
cur.execute("""
    SELECT i.key, ifv.value
    FROM items i
    JOIN itemData id2 ON i.itemID=id2.itemID
    JOIN itemDataValues ifv ON id2.valueID=ifv.valueID
    JOIN fields f ON id2.fieldID=f.fieldID
    WHERE i.libraryID=? AND f.fieldName='title' AND ifv.value LIKE '%Dynamic%Rating%'
""", (lib_id,))
dyn_items = cur.fetchall()
print(f"\nDynamic Rating items in lib {lib_id}:")
for k, t in dyn_items:
    print(f"  key={k}: {t[:80]}")

conn.close()
