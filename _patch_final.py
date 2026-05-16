"""
Apply clean cloud URL mapping to zotero_network.html:
- Only use local_pdf storage key matching (22 nodes - reliable)
- Plus DOI matching only for real DOIs (starting with 10.)
- Then update zotero_cloud fields for all 78 nodes
- Move links to TOP of showD detail panel
- Fix mojibake display via JS function
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

# Get all items with type, title, DOI
cur.execute("""
    SELECT i.itemID, i.key, it.typeName
    FROM items i
    JOIN itemTypes it ON i.itemTypeID=it.itemTypeID
    WHERE i.libraryID=? AND it.typeName != 'attachment'
""", (lib_id,))
regular_items = {row[0]: {'key': row[1], 'type': row[2]} for row in cur.fetchall()}

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

cur.execute("""
    SELECT i.itemID, i.key, ia.parentItemID, ia.path
    FROM items i
    JOIN itemTypes it ON i.itemTypeID=it.itemTypeID
    JOIN itemAttachments ia ON i.itemID=ia.itemID
    WHERE i.libraryID=? AND it.typeName='attachment'
""", (lib_id,))
all_att = cur.fetchall()
conn.close()

parent_to_att = {}
att_key_to_parent_key = {}
for att_id, att_key, par_id, att_path in all_att:
    if par_id and par_id in regular_items:
        par_key = regular_items[par_id]['key']
        att_key_to_parent_key[att_key] = par_key
        if par_id not in parent_to_att:
            parent_to_att[par_id] = (att_key, att_path or '')

# Read HTML
html_path = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html")
html = html_path.read_text(encoding='utf-8')
nodes_text = html.splitlines()[134]

# Extract nodes
def extract_nodes(line):
    nodes = []
    for m in re.finditer(r'\{"id":\s*(\d+),', line):
        nid = int(m.group(1))
        node_start = m.start()
        depth = 0; in_str = False; i = node_start
        node_str = ''
        while i < len(line):
            c = line[i]
            if in_str:
                if c == '\\': i += 2; continue
                elif c == '"': in_str = False
            else:
                if c == '"': in_str = True
                elif c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        node_str = line[node_start:i+1]
                        break
            i += 1
        def gf(s, field):
            m2 = re.search(r'"' + field + r'":\s*"([^"]*)"', s)
            return m2.group(1) if m2 else ''
        nodes.append({'id': nid, 'local_pdf': gf(node_str, 'local_pdf'),
                      'doi_link': gf(node_str, 'doi_link')})
    return nodes

nodes = extract_nodes(nodes_text)
print(f"Parsed {len(nodes)} nodes")

# Strategy 1: local_pdf storage key
cloud_map = {}
for n in nodes:
    m = re.search(r'/storage/([A-Z0-9]{8})/', n['local_pdf'], re.IGNORECASE)
    if m:
        ak = m.group(1).upper()
        pk = att_key_to_parent_key.get(ak)
        if pk:
            cloud_map[n['id']] = f"{BASE}/{pk}/attachment/{ak}/reader"

print(f"After strategy 1 (local_pdf): {len(cloud_map)}")

# Strategy 2: DOI — ONLY for valid DOIs starting with "10."
doi_to_keys = {}
for par_id, info in regular_items.items():
    doi = info.get('DOI', '').strip()
    if doi and doi.startswith('10.') and par_id in parent_to_att:
        ak = parent_to_att[par_id][0]
        doi_to_keys[doi.lower()] = (info['key'], ak)
    # Also try URL field for DOI
    url_val = info.get('url', '').strip()
    doi_match = re.search(r'doi\.org/(10\..+?)$', url_val, re.IGNORECASE)
    if doi_match and par_id in parent_to_att:
        ak = parent_to_att[par_id][0]
        doi_norm = doi_match.group(1).lower()
        doi_to_keys[doi_norm] = (info['key'], ak)

for n in nodes:
    if n['id'] in cloud_map:
        continue
    doi_link = n['doi_link']
    doi_match = re.search(r'doi\.org/(10\..+?)$', doi_link, re.IGNORECASE)
    if doi_match:
        doi = doi_match.group(1).lower()
        if doi in doi_to_keys:
            pk, ak = doi_to_keys[doi]
            cloud_map[n['id']] = f"{BASE}/{pk}/attachment/{ak}/reader"

print(f"After strategy 2 (DOI, valid only): {len(cloud_map)}")

# Print unmapped nodes count
total_no_cloud = 78 - len(cloud_map)
print(f"Nodes with no cloud URL: {total_no_cloud} (no attachment in Zotero)")
for n in nodes:
    if n['id'] not in cloud_map:
        print(f"  node {n['id']}: no match")

# ─── NOW PATCH THE HTML ────────────────────────────────────────────────────────
# 1. Update all zotero_cloud fields
# 2. Move links div to TOP of showD (after author/year line)
# 3. Add mojibake fix function

print("\n=== Patching HTML ===")

# Update zotero_cloud fields in NODES (raw string replacement)
lines = html.splitlines()
nodes_line = lines[134]

for nid, url in cloud_map.items():
    # Find the node with this id and replace its zotero_cloud value
    # The pattern is unique enough: find id=N with its zotero_cloud field
    # We look for: "id": N, ... "zotero_cloud": "OLD_VALUE"
    # Since all on one line, use regex to find and replace for this specific node
    # Strategy: find the node, replace its zotero_cloud in context
    old_pattern = re.compile(
        r'("id":\s*' + str(nid) + r',.*?"zotero_cloud":\s*)"([^"]*)"',
        re.DOTALL
    )
    if old_pattern.search(nodes_line):
        nodes_line = old_pattern.sub(
            lambda m: m.group(1) + '"' + url + '"',
            nodes_line, count=1
        )
    else:
        print(f"  WARNING: Could not find node {nid} to update zotero_cloud")

lines[134] = nodes_line
print(f"  Updated {len(cloud_map)} zotero_cloud fields")

# 2. Move the links div to TOP in showD
# Current bottom of showD has: <div>${doi}${pdf}${cld}</div>
# We want it right after the year/pub line
old_html = '\n'.join(lines)

# The showD function currently has the links at the bottom
# Move links+badges div to right after author/year info
# Current structure:
#   <div class="dt">${d.title_full}</div>
#   <div class="dm"><strong>Autores:</strong>...</div>
#   <div class="dm"><strong>Anio:</strong> ... </div>
#   <div style="margin:8px 0">...(category badges)...</div>
#   ... abstract, aporte, tags, LINKS, nbH, collections
# We want links BEFORE abstract/aporte (right after badges)

old_showD_links_bottom = '    <div>${doi}${pdf}${cld}</div>\n    ${nbH}'
new_showD_links_top_removed = '    ${nbH}'  # remove from bottom

# Insert links+mojibake fix BEFORE abstract, after badge line
# The badge line is: <div style="margin:8px 0"><span class="bk"...
# We insert the links div right after the badge line and before abstract
old_badge_to_abstract = '''    <div style="margin:8px 0"><span class="bk" style="background:${ci.color||'#999'}">${ci.label||d.category}</span><span class="bk" style="background:${d.comm_color}">${d.comm_label}</span></div>
    ${d.abstract?'''
new_badge_with_links = '''    <div style="margin:8px 0"><span class="bk" style="background:${ci.color||'#999'}">${ci.label||d.category}</span><span class="bk" style="background:${d.comm_color}">${d.comm_label}</span></div>
    <div style="margin:6px 0">${doi}${pdf}${cld}</div>
    ${d.abstract?'''

old_html = old_html.replace(old_showD_links_bottom, new_showD_links_top_removed)
old_html = old_html.replace(old_badge_to_abstract, new_badge_with_links)

# 3. Add mojibake fix function and apply to displayed text
# Add fixMoji JS function and use it in showD for title, authors, abstract, aporte
# The mojibake is a double-UTF8 encoding issue (Latin-1 bytes interpreted as UTF-8)
# Fix: decodeURIComponent(escape(str)) decodes mojibake back to proper UTF-8
# BUT: escape() is deprecated, use a manual approach instead
#
# The correct fix: use TextDecoder if available
# Actually the simplest approach in modern browsers:
# function fixMoji(s){ try{return decodeURIComponent(escape(s))}catch(e){return s} }

# Check what the current showD function looks like in terms of title usage
# We need to use fixMoji() around displayed text fields

# First add the fixMoji function near the top of the script
# It should be before showD function

old_before_showD = '/* ─── DETAIL PANEL ─── */\nfunction showD(d,nid){'
new_with_fixmoji = '''/* ─── DETAIL PANEL ─── */
function fixMoji(s){if(!s)return s;try{return decodeURIComponent(escape(s));}catch(e){return s;}}
function showD(d,nid){'''

# Also apply fixMoji to the displayed fields in showD
# Current: ${d.title_full}, ${d.authors}, ${d.abstract}, ${d.aporte}
# New: ${fixMoji(d.title_full)}, ${fixMoji(d.authors)}, etc.
# But only apply to text content, not to badge/category/color fields

# Look for the actual showD lines
old_html = old_html.replace(old_before_showD, new_with_fixmoji)

# Apply fixMoji to displayed text in the innerHTML template
old_dt = '    <div class="dt">${d.title_full}</div>'
new_dt = '    <div class="dt">${fixMoji(d.title_full)}</div>'

old_authors = '    <div class="dm"><strong>Autores:</strong> ${d.authors||\'Desconocido\'}</div>'
new_authors = '    <div class="dm"><strong>Autores:</strong> ${fixMoji(d.authors)||\'Desconocido\'}</div>'

old_abstract = '${d.abstract?`<div class="da">${d.abstract}${d.abstract.length>=700?\'...\':\'\'}` :'
new_abstract = '${d.abstract?`<div class="da">${fixMoji(d.abstract)}${d.abstract.length>=700?\'...\':\'\'}` :'

old_aporte = '`<div class="dap"><div class="dapl">Aporte al proyecto</div>${d.aporte}</div>`'
new_aporte = '`<div class="dap"><div class="dapl">Aporte al proyecto</div>${fixMoji(d.aporte)}</div>`'

# Also fix neighbor titles in the neighbors list
old_nb_title = 'title="${cn.title_full}">${au}'
new_nb_title = 'title="${fixMoji(cn.title_full)}">${au}'

old_nb_titlefull = '${cn.title_full.substring(0,50)}${cn.title_full.length>50'
new_nb_titlefull = '${fixMoji(cn.title_full).substring(0,50)}${fixMoji(cn.title_full).length>50'

old_html = old_html.replace(old_dt, new_dt)
old_html = old_html.replace(old_authors, new_authors)

# For abstract, need to handle carefully
if old_abstract in old_html:
    old_html = old_html.replace(old_abstract, new_abstract)
else:
    print("  WARNING: Could not find abstract pattern to fix")

if old_aporte in old_html:
    old_html = old_html.replace(old_aporte, new_aporte)
else:
    print("  WARNING: Could not find aporte pattern to fix")

old_html = old_html.replace(old_nb_title, new_nb_title)
old_html = old_html.replace(old_nb_titlefull, new_nb_titlefull)

# Check counts of replaced items
for old, new, name in [
    (old_dt, new_dt, 'title_full'),
    (old_authors, new_authors, 'authors'),
    (old_nb_title, new_nb_title, 'nb_title'),
]:
    count = old_html.count(new)
    print(f"  fixMoji applied to {name}: {count} occurrence(s)")

print(f"  Links moved to top (after badge)")
print(f"  fixMoji function added")

# Write patched HTML
html_path.write_text(old_html, encoding='utf-8')
print(f"\nPatched HTML saved: {html_path}")

# Verify
lines2 = old_html.splitlines()
print(f"Total lines: {len(lines2)}")

# Quick check for syntax
import re
script_m = re.search(r'<script>(.*?)</script>', old_html, re.DOTALL)
if script_m:
    script = script_m.group(1)
    print(f"Script block: {len(script)} chars")
else:
    print("WARNING: No script block found!")
