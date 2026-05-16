"""
Recorre todos los nodos del HTML y borra local_pdf cuando el archivo
no existe realmente en disco. Así el botón "PDF Local" solo aparece
cuando el archivo está disponible.
"""
import re, json, os, urllib.parse, pathlib

HTML = r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html"

html = pathlib.Path(HTML).read_text(encoding="utf-8")

# Extraer la línea que contiene NODES
m = re.search(r"(const NODES=)(\[.*?\])(;)", html, re.DOTALL)
if not m:
    print("ERROR: no se encontró const NODES=")
    exit(1)

prefix, nodes_json, suffix = m.group(1), m.group(2), m.group(3)
nodes = json.loads(nodes_json)

fixed = missing = 0
for n in nodes:
    lp = n.get("local_pdf", "")
    if not lp:
        continue
    # Convertir file:// URL a path de Windows
    path = lp.replace("file:///", "").replace("%20", " ")
    # urllib decode completo por si hay otros escapes
    path = urllib.parse.unquote(path)
    if not os.path.isfile(path):
        print(f"  ✗ MISSING node {n['id']:>3}: {os.path.basename(path)}")
        n["local_pdf"] = ""
        missing += 1
    else:
        print(f"  ✓ OK     node {n['id']:>3}: {os.path.basename(path)}")
        fixed += 1

print(f"\nResultado: {fixed} OK, {missing} borrados")

# Reconstruir NODES como JSON compacto (sin saltos de línea)
new_nodes = json.dumps(nodes, ensure_ascii=False, separators=(',', ':'))
new_html = html[:m.start()] + prefix + new_nodes + suffix + html[m.end():]

pathlib.Path(HTML).write_text(new_html, encoding="utf-8")
print("HTML actualizado OK")
