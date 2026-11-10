import re, sqlite3, shutil
from pathlib import Path

zotero_root = Path(r"C:\Users\QU1267\Zotero")
db_src = zotero_root / "zotero.sqlite"
tmp_db = Path(r"C:\usr\ths_mia_fiis\tesis_MIA_lk\_tmp3\z.sqlite")
html_path = Path(r"C:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html")

GROUP_ID = "6532672"
GROUP_SLUG = "tesis_mia_2026"

# 1. copy DB and query keys
tmp_db.parent.mkdir(exist_ok=True)
shutil.copy2(db_src, tmp_db)
conn = sqlite3.connect(str(tmp_db))
cur = conn.cursor()
cur.execute("""
  SELECT p.key, a.key
  FROM itemAttachments ia
  JOIN items a ON a.itemID = ia.itemID
  JOIN items p ON p.itemID = ia.parentItemID
  WHERE ia.parentItemID IS NOT NULL
""")
att_to_parent = {r[1]: r[0] for r in cur.fetchall()}
conn.close()
shutil.rmtree(tmp_db.parent, ignore_errors=True)
print(f"att->parent: {len(att_to_parent)}")

# 2. read HTML
html = html_path.read_text(encoding='utf-8')

# 3. Remove any existing zotero_cloud fields (idempotent re-run support)
html = re.sub(r',\s*"zotero_cloud":\s*"[^"]*"', '', html)

# 4. Inject zotero_cloud after each "local_pdf": "file:///.../storage/{KEY}/..."
replaced = 0
skipped = 0

def replacer(m):
    global replaced, skipped
    full = m.group(1)
    att_key = m.group(2)
    parent_key = att_to_parent.get(att_key)
    if not parent_key:
        skipped += 1
        return full + ', "zotero_cloud": ""'
    cloud = (
        f"https://www.zotero.org/groups/{GROUP_ID}/{GROUP_SLUG}"
        f"/items/{parent_key}/attachment/{att_key}/reader"
    )
    replaced += 1
    return full + f', "zotero_cloud": "{cloud}"'

html = re.sub(r'("local_pdf":\s*"file:///[^"]+/storage/([A-Z0-9]+)/[^"]*")', replacer, html)
print(f"zotero_cloud injected: {replaced}, skipped (no parent): {skipped}")

# Add empty zotero_cloud to nodes with empty local_pdf
html = re.sub(r'("local_pdf":\s*"")', r'\1, "zotero_cloud": ""', html)

# 5. Add .lb3 CSS if not present
css_target = '.lb2{background:#b71c1c;color:#fff}.lb2:hover{background:#c62828}'
lb3_css = '\n.lb3{background:#2e7d32;color:#fff}.lb3:hover{background:#388e3c}'
if '.lb3{' not in html:
    html = html.replace(css_target, css_target + lb3_css)
    print("CSS .lb3 added")
else:
    print("CSS .lb3 already present")

# 6. Patch showD() to define const cld= and include ${cld} in render
if 'const cld=' not in html:
    # Find the pdf line and insert cld after it
    idx = html.find('const pdf=d.local_pdf')
    if idx >= 0:
        # find end of that statement: look for the closing `;` after the template literal
        end_idx = html.index("';", idx) + 2
        cld_stmt = "\n  const cld=d.zotero_cloud?`<a href=\"${d.zotero_cloud}\" target=\"_blank\" class=\"lb lb3\">&#x2601; Zotero</a>`:'';"
        html = html[:end_idx] + cld_stmt + html[end_idx:]
        print("showD: cld statement inserted after pdf line")
    else:
        print("ERROR: could not find 'const pdf=d.local_pdf' in showD()")
else:
    print("showD: const cld= already present")

# 7. Patch render line to include ${cld}
old_render = '<div>${doi}${pdf}</div>'
new_render = '<div>${doi}${pdf}${cld}</div>'
if old_render in html:
    html = html.replace(old_render, new_render)
    print("render line patched: ${doi}${pdf}${cld}")
elif '${cld}' in html:
    print("render line already has ${cld}")
else:
    print("WARNING: render line '<div>${doi}${pdf}</div>' not found")

# 8. write back
html_path.write_text(html, encoding='utf-8')
print(f"DONE: {html_path}")
