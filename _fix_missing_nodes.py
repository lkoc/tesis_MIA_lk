"""
Extraer nodos 16-31 del Copy y reinyectarlos en zotero_network.html
con sus campos zotero_cloud.
"""
import re, sqlite3, shutil
from pathlib import Path

copy_path  = Path(r"C:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network - Copy.html")
html_path  = Path(r"C:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html")
db_src     = Path(r"C:\Users\QU1267\Zotero\zotero.sqlite")
tmp_db     = Path(r"C:\usr\ths_mia_fiis\tesis_MIA_lk\_tmp4\z.sqlite")

GROUP_ID   = "6532672"
GROUP_SLUG = "tesis_mia_2026"

# 1. Get att -> parent map
tmp_db.parent.mkdir(exist_ok=True)
shutil.copy2(db_src, tmp_db)
conn = sqlite3.connect(str(tmp_db))
att_to_parent = {r[1]: r[0] for r in conn.execute("""
    SELECT p.key, a.key
    FROM itemAttachments ia
    JOIN items a ON a.itemID = ia.itemID
    JOIN items p ON p.itemID = ia.parentItemID
    WHERE ia.parentItemID IS NOT NULL
""").fetchall()}
conn.close()
shutil.rmtree(tmp_db.parent, ignore_errors=True)
print(f"att->parent: {len(att_to_parent)}")

# 2. Extract nodes 16-31 from Copy file
copy_text = copy_path.read_text(encoding='utf-8')

# Find the segment: everything from }, {"id": 16, ... up to (but not including) }, {"id": 32,
# Both are on the big const NODES=[...] line
m_from = re.search(r',\s*(\{"id":\s*16\b)', copy_text)
m_to   = re.search(r',\s*(\{"id":\s*32\b)', copy_text)
if not m_from or not m_to:
    raise SystemExit(f"Could not find id=16 ({bool(m_from)}) or id=32 ({bool(m_to)}) in Copy file")

# The extracted chunk: from "}, {"id": 16" up to (not including) ", {"id": 32"
chunk_16_31 = copy_text[m_from.start() : m_to.start()]
print(f"Extracted chunk length: {len(chunk_16_31)} chars")
# Quick sanity: count nodes
node_ids = re.findall(r'"id":\s*(\d+)', chunk_16_31)
print(f"Node ids in chunk: {node_ids}")

# 3. Inject zotero_cloud into the chunk
# Remove any existing zotero_cloud fields first
chunk_16_31 = re.sub(r',\s*"zotero_cloud":\s*"[^"]*"', '', chunk_16_31)

replaced = skipped = 0

def add_cloud(m):
    global replaced, skipped
    full    = m.group(1)
    att_key = m.group(2)
    parent_key = att_to_parent.get(att_key)
    if not parent_key:
        skipped += 1
        return full + ', "zotero_cloud": ""'
    url = (f"https://www.zotero.org/groups/{GROUP_ID}/{GROUP_SLUG}"
           f"/items/{parent_key}/attachment/{att_key}/reader")
    replaced += 1
    return full + f', "zotero_cloud": "{url}"'

chunk_16_31 = re.sub(r'("local_pdf":\s*"file:///[^"]+/storage/([A-Z0-9]+)/[^"]*")',
                     add_cloud, chunk_16_31)
chunk_16_31 = re.sub(r'("local_pdf":\s*"")', r'\1, "zotero_cloud": ""', chunk_16_31)
print(f"Cloud injected: {replaced}, skipped: {skipped}")

# 4. Insert the chunk into zotero_network.html before {"id": 32,
html = html_path.read_text(encoding='utf-8')

# Verify nodes 16-31 are indeed missing
missing = [i for i in range(16, 32)
           if not re.search(rf'"id":\s*{i}\b(?!.*"from")', html[:html.find('const EDGES')])]
print(f"Missing node ids confirmed: {missing}")

# Find the injection point: right before }, {"id": 32,  in the NODES section
nodes_section_end = html.find('const EDGES')
nodes_section = html[:nodes_section_end]
m_32 = re.search(r'(,\s*\{"id":\s*32\b)', nodes_section)
if not m_32:
    raise SystemExit("Cannot find id=32 in current html NODES section")

insert_pos = m_32.start()  # position of the comma before {"id": 32
html = html[:insert_pos] + chunk_16_31 + html[insert_pos:]
print(f"Chunk inserted at position {insert_pos}")

# 5. Verify all 78 nodes (0-77) now present
nodes_section2 = html[:html.find('const EDGES')]
found_ids = set(int(x) for x in re.findall(r'"id":\s*(\d+)(?=.*?"label")', nodes_section2))
expected  = set(range(78))
still_missing = expected - found_ids
print(f"Nodes found: {len(found_ids)}, still missing: {sorted(still_missing)}")

# 6. Write
html_path.write_text(html, encoding='utf-8')
print(f"DONE: {html_path}")
