import sqlite3, json
DB = r"C:\Users\QU1267\Zotero\zotero.sqlite"
STORAGE = r"C:\Users\QU1267\Zotero\storage"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = con.cursor()
# Get all PDFs with parent item IDs
cur.execute("""
    SELECT i.key, ia.parentItemID, ia.path
    FROM items i
    JOIN itemAttachments ia ON i.itemID = ia.itemID
    WHERE ia.contentType = 'application/pdf' AND ia.path LIKE 'storage:%'
    ORDER BY ia.parentItemID
""")
rows = cur.fetchall()
print(f"Total PDF attachments: {len(rows)}")
# Map parentItemID -> local path
import os
for att_key, parent_id, path_str in rows[:5]:
    fname = path_str[8:]
    full = os.path.join(STORAGE, att_key, fname)
    print(f"parent={parent_id} exists={os.path.exists(full)}")
    print(f"  URL: file:///{full.replace(chr(92),'/')}")
print("...")
# Get all parent IDs with attachments
parent_ids = {r[1] for r in rows}
print(f"Unique parent IDs with PDFs: {len(parent_ids)}")
con.close()
