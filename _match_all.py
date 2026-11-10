"""
Comprehensive Zotero cloud URL mapper:
- Match nodes by: local_pdf storage key, DOI, and title similarity
- Output complete mapping to update the HTML
"""
import sqlite3, re, json
from pathlib import Path

DB = r"C:\Users\QU1267\AppData\Local\Temp\zotero_ro.sqlite"
GROUP_ID = 6532672
SLUG = "tesis_mia_2026"
BASE = f"https://www.zotero.org/groups/{GROUP_ID}/{SLUG}/items"

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = conn.cursor()

cur.execute("SELECT libraryID FROM groups WHERE groupID=?", (GROUP_ID,))
lib_id = cur.fetchone()[0]

# ── Get all items in the library with title, DOI, key ──────────────────────────
cur.execute("""
    SELECT i.itemID, i.key, it.typeName
    FROM items i
    JOIN itemTypes it ON i.itemTypeID=it.itemTypeID
    WHERE i.libraryID=? AND it.typeName != 'attachment'
""", (lib_id,))
regular_items = {row[0]: {'key': row[1], 'type': row[2]} for row in cur.fetchall()}

# Get titles and DOIs
for field_name in ('title', 'DOI', 'url'):
    cur.execute("""
        SELECT id2.itemID, ifv.value
        FROM itemData id2
        JOIN itemDataValues ifv ON id2.valueID=ifv.valueID
        JOIN fields f ON id2.fieldID=f.fieldID
        WHERE f.fieldName=? AND id2.itemID IN ({})
    """.format(','.join(['?']*len(regular_items))),
    [field_name] + list(regular_items.keys()))
    for iid, val in cur.fetchall():
        if iid in regular_items:
            regular_items[iid][field_name] = val

# ── Get attachment items with parent relationships ─────────────────────────────
cur.execute("""
    SELECT i.itemID, i.key, ia.parentItemID, ia.path
    FROM items i
    JOIN itemTypes it ON i.itemTypeID=it.itemTypeID
    JOIN itemAttachments ia ON i.itemID=ia.itemID
    WHERE i.libraryID=?
""", (lib_id,))
all_att = cur.fetchall()
conn.close()

# Build: parentItemID -> (attKey, attPath)
parent_to_att = {}  # parentID -> (attKey, path)
att_key_to_parent_key = {}  # attKey -> parentKey
for att_id, att_key, par_id, att_path in all_att:
    if par_id and par_id in regular_items:
        par_key = regular_items[par_id]['key']
        att_key_to_parent_key[att_key] = par_key
        if par_id not in parent_to_att:
            parent_to_att[par_id] = (att_key, att_path or '')

print(f"Parent items with attachments: {len(parent_to_att)}")

# ── Read the HTML and extract nodes ───────────────────────────────────────────
html = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html").read_text(encoding='utf-8')
nodes_text = html.splitlines()[134]  # line 135

# Extract per-node fields: id, local_pdf, doi_link, zotero_cloud
# We need to split into individual node objects
def extract_nodes(line):
    """Extract list of dicts with id, local_pdf, doi_link, zotero_cloud, title_full"""
    nodes = []
    # Each node starts with {"id": N,
    pattern = re.compile(r'\{"id":\s*(\d+),.*?"local_pdf":\s*"([^"]*)".*?"zotero_cloud":\s*"([^"]*)"', re.DOTALL)
    # Since it's all on one line, let's find each node's id and its local_pdf & doi_link
    for m in re.finditer(r'\{"id":\s*(\d+),', nodes_text):
        nid = int(m.group(1))
        # Get the slice of text starting here
        node_start = m.start()
        # Find the matching } for this node
        depth = 0
        in_str = False
        i = node_start
        while i < len(nodes_text):
            c = nodes_text[i]
            if in_str:
                if c == '\\': i += 2; continue
                elif c == '"': in_str = False
            else:
                if c == '"': in_str = True
                elif c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        node_str = nodes_text[node_start:i+1]
                        break
            i += 1
        
        # Extract fields from node_str
        def get_field(s, field):
            m2 = re.search(r'"' + field + r'":\s*"([^"]*)"', s)
            return m2.group(1) if m2 else ''
        
        nodes.append({
            'id': nid,
            'local_pdf': get_field(node_str, 'local_pdf'),
            'doi_link': get_field(node_str, 'doi_link'),
            'zotero_cloud': get_field(node_str, 'zotero_cloud'),
        })
    return nodes

nodes = extract_nodes(nodes_text)
print(f"Parsed {len(nodes)} nodes from HTML")

# ── Match nodes to Zotero items ───────────────────────────────────────────────
# Strategy 1: match via local_pdf storage key
# Strategy 2: match via DOI
# Strategy 3: match remaining items manually if parent_to_att has unmatched items

cloud_map = {}  # node_id -> cloud_url

# Build reverse map: storage_att_key -> parent_key
storage_key_to_parent = {}
for par_id, (att_key, att_path) in parent_to_att.items():
    par_key = regular_items[par_id]['key']
    # att_path might look like: "storage:FILENAME" or just filename
    storage_key_to_parent[att_key] = par_key

# Strategy 1: local_pdf contains /storage/ATTKEY/
for n in nodes:
    lp = n['local_pdf']
    m = re.search(r'/storage/([A-Z0-9]{8})/', lp, re.IGNORECASE)
    if m:
        att_key = m.group(1).upper()
        par_key = att_key_to_parent_key.get(att_key)
        if par_key:
            url = f"{BASE}/{par_key}/attachment/{att_key}/reader"
            cloud_map[n['id']] = url

print(f"After strategy 1 (local_pdf): {len(cloud_map)} nodes mapped")

# Strategy 2: DOI matching
# Build DOI -> (parentKey, attKey) from Zotero
doi_to_keys = {}
for par_id, info in regular_items.items():
    doi = info.get('DOI', '').strip().lower()
    if doi and par_id in parent_to_att:
        att_key = parent_to_att[par_id][0]
        doi_to_keys[doi] = (info['key'], att_key)

# Also try with URL field that might be a DOI URL
url_doi_to_keys = {}
for par_id, info in regular_items.items():
    url_val = info.get('url', '').strip()
    doi_in_url = re.search(r'doi\.org/(.+?)$', url_val, re.IGNORECASE)
    if doi_in_url and par_id in parent_to_att:
        att_key = parent_to_att[par_id][0]
        url_doi_to_keys[doi_in_url.group(1).lower()] = (info['key'], att_key)

for n in nodes:
    if n['id'] in cloud_map:
        continue
    doi_link = n['doi_link']
    # Extract DOI from link
    doi_match = re.search(r'doi\.org/(.+?)$', doi_link, re.IGNORECASE)
    if doi_match:
        doi = doi_match.group(1).strip().lower()
        if doi in doi_to_keys:
            pk, ak = doi_to_keys[doi]
            cloud_map[n['id']] = f"{BASE}/{pk}/attachment/{ak}/reader"
        elif doi in url_doi_to_keys:
            pk, ak = url_doi_to_keys[doi]
            cloud_map[n['id']] = f"{BASE}/{pk}/attachment/{ak}/reader"

print(f"After strategy 2 (DOI): {len(cloud_map)} nodes mapped")

# Strategy 3: Check all parent items that have attachments and find unmatched node
# Print which parent items have attachments but no node matched
mapped_parent_keys = set()
for url in cloud_map.values():
    m = re.search(r'items/([A-Z0-9]+)/attachment', url)
    if m:
        mapped_parent_keys.add(m.group(1))

print(f"\nUnmatched Zotero items with attachments (should be manually checked):")
for par_id, (att_key, att_path) in parent_to_att.items():
    par_key = regular_items[par_id]['key']
    if par_key not in mapped_parent_keys:
        title = regular_items[par_id].get('title', 'NO_TITLE')[:70]
        doi = regular_items[par_id].get('DOI', '')
        print(f"  {par_key} -> {att_key} | {title} | DOI: {doi}")

# Save the final mapping
result = {str(k): v for k, v in sorted(cloud_map.items())}
Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\_zotero_cloud_map.json").write_text(
    json.dumps(result, indent=2), encoding='utf-8')
print(f"\nFinal mapping: {len(cloud_map)} nodes")
print("Saved to _zotero_cloud_map.json")
for nid, url in sorted(cloud_map.items()):
    print(f"  node {nid}: {url}")
