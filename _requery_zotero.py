"""
Re-query Zotero SQLite to get ALL attachment->parentKey mappings for group 6532672,
then match against nodes in the HTML by Zotero key embedded in local_pdf paths,
and print a mapping dict of {node_id: cloud_url}
"""
import sqlite3, re, json
from pathlib import Path

DB = r"C:\Users\QU1267\AppData\Local\Temp\zotero_ro.sqlite"
GROUP_ID = 6532672
SLUG = "tesis_mia_2026"
BASE = f"https://www.zotero.org/groups/{GROUP_ID}/{SLUG}/items"

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = conn.cursor()

# Get the libraryID for the group
cur.execute("SELECT libraryID FROM groups WHERE groupID=?", (GROUP_ID,))
row = cur.fetchone()
if not row:
    print("Group not found!")
    exit(1)
lib_id = row[0]
print(f"Library ID for group {GROUP_ID}: {lib_id}")

# Get all items and their keys in this library
# Items table: itemID, itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced
cur.execute("""
    SELECT i.itemID, i.key, it.typeName
    FROM items i
    JOIN itemTypes it ON i.itemTypeID=it.itemTypeID
    WHERE i.libraryID=?
    ORDER BY i.itemID
""", (lib_id,))
all_items = cur.fetchall()
print(f"Total items in library: {len(all_items)}")

# Build key->itemID and itemID->key maps
key2id = {row[1]: row[0] for row in all_items}
id2key = {row[0]: row[1] for row in all_items}
id2type = {row[0]: row[2] for row in all_items}

# Get attachment items
attachments = {iid: key for iid, key, typ in all_items if typ == 'attachment'}
print(f"Attachment items: {len(attachments)}")

# Get parent relationships: itemID -> parentItemID
cur.execute("""
    SELECT itemID, parentItemID FROM itemAttachments
    WHERE itemID IN ({})
""".format(','.join(['?']*len(attachments))), list(attachments.keys()))
att_parents = {row[0]: row[1] for row in cur.fetchall()}
print(f"Attachment->parent mappings: {len(att_parents)}")

# Build: parentKey -> attKey mapping (only first attachment per parent)
parent_to_att = {}
for att_id, par_id in att_parents.items():
    if par_id in id2key:
        par_key = id2key[par_id]
        att_key = attachments[att_id]
        if par_key not in parent_to_att:
            parent_to_att[par_key] = att_key

print(f"\nParent->Attachment mappings: {len(parent_to_att)}")
print("\nSample mappings:")
for pk, ak in list(parent_to_att.items())[:5]:
    url = f"{BASE}/{pk}/attachment/{ak}/reader"
    print(f"  {pk} -> {ak} | {url}")

# Now read the HTML and find all nodes with their Zotero keys from local_pdf paths
html = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html").read_text(encoding='utf-8')
lines = html.splitlines()
nodes_line = lines[134]  # line 135

# Extract all node id, local_pdf, zotero_cloud, title values
# We need to find Zotero item keys from local_pdf URLs
# local_pdf contains paths like: file:///C:/Users/QU1267/Zotero/storage/ABCD1234/filename.pdf
# The ABCD1234 is the ATTACHMENT key

# Parse nodes more carefully: split on node boundaries
# Since we can't use json.loads (mojibake), we parse manually
node_pattern = re.compile(r'"id":\s*(\d+).*?"local_pdf":\s*"(.*?)".*?"zotero_cloud":\s*"(.*?)".*?"title_full":\s*"(.*?)"', re.DOTALL)

# Actually nodes are on one line, so use a simpler per-node extraction
# Split into individual node strings
all_nodes_text = nodes_line

# Find each node's key data
print("\n--- Matching nodes to Zotero attachments ---")
found = {}
no_key = []
already_has = []

for m in re.finditer(r'"id":\s*(\d+),.*?"local_pdf":\s*"([^"]*)"', all_nodes_text):
    nid = int(m.group(1))
    lpdf = m.group(2)
    
    # Extract att_key from local_pdf
    att_match = re.search(r'/storage/([A-Z0-9]{8})/', lpdf, re.IGNORECASE)
    if att_match:
        att_key = att_match.group(1).upper()
        # Find parent of this attachment
        # Actually we need reverse: att_key -> parent_key
        # Build reverse map
        att_key_to_parent = {v: k for k, v in parent_to_att.items()}
        par_key = att_key_to_parent.get(att_key)
        if par_key:
            cloud_url = f"{BASE}/{par_key}/attachment/{att_key}/reader"
            found[nid] = cloud_url
        else:
            # The att_key might be from a different parent - search all attachments
            for aid, ak in attachments.items():
                if ak.upper() == att_key:
                    par_id = att_parents.get(aid)
                    if par_id and par_id in id2key:
                        par_key2 = id2key[par_id]
                        cloud_url2 = f"{BASE}/{par_key2}/attachment/{ak}/reader"
                        found[nid] = cloud_url2
                        break

print(f"Nodes with local_pdf attachment key found: {len(found)}")
print(f"Nodes without match: {len(no_key)}")

# Save mapping
Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\_zotero_cloud_map.json").write_text(
    json.dumps(found, indent=2), encoding='utf-8')
print(f"\nSaved mapping to _zotero_cloud_map.json")
print("\nAll found mappings:")
for nid, url in sorted(found.items()):
    print(f"  node {nid}: {url}")

conn.close()
