"""
Reconstruye local_pdf usando:
1. Para nodos con zotero_cloud: extrae ATT_KEY del URL → busca en storage/{ATT_KEY}/*.pdf
2. Para nodos sin zotero_cloud: busca por similitud de nombre de archivo en todo el storage
Verifica existencia real en disco antes de asignar.
"""
import re, json, os, glob, pathlib

HTML   = r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html"
STORE  = r"C:\Users\QU1267\Zotero\storage"
PREFIX = "file:///"

html = pathlib.Path(HTML).read_text(encoding="utf-8")

# ── extraer NODES ──────────────────────────────────────────────────────────────
m = re.search(r"(const NODES=)(\[.*?\])(;)", html, re.DOTALL)
nodes = json.loads(m.group(2))

# ── índice completo de PDFs en disco: {att_key: full_path} ───────────────────
disk_map = {}          # att_key (carpeta) → full path
name_map = {}          # filename sin extensión (lower) → full path  (para fallback)

for pdf_path in glob.glob(STORE + r"\*\*.pdf"):
    parts = pdf_path.replace("\\", "/").split("/")
    att_key  = parts[-2]
    filename = parts[-1]
    disk_map[att_key] = pdf_path
    name_map[filename.lower().replace(" ", "")] = pdf_path

# ── función fallback: busca nombre más similar ──────────────────────────────
def find_by_name(old_path):
    """Intenta encontrar el PDF correcto por similitud de nombre."""
    if not old_path:
        return None
    old_name = old_path.split("/")[-1].lower().replace(" ", "")
    # coincidencia exacta en name_map
    if old_name in name_map:
        return name_map[old_name]
    # coincidencia parcial: nombre de archivo sin clave al inicio
    # Zotero a veces tiene "authorYEARtitle__original.pdf" → comparar original
    if "__" in old_name:
        suffix = old_name.split("__", 1)[1]
        for k, v in name_map.items():
            if suffix in k:
                return v
    return None

# ── reconstruir local_pdf ────────────────────────────────────────────────────
fixed = skipped = fallback = 0
for n in nodes:
    cloud = n.get("zotero_cloud", "")
    old_pdf = n.get("local_pdf", "")  # vacío o path incorrecto

    # Estrategia 1: extraer ATT_KEY del cloud URL
    if cloud:
        cm = re.search(r"/attachment/([A-Z0-9]+)/reader", cloud)
        if cm:
            att_key = cm.group(1)
            if att_key in disk_map:
                path = disk_map[att_key].replace("\\", "/")
                n["local_pdf"] = PREFIX + path
                fixed += 1
                print(f"  ✓ cloud  node {n['id']:>3}  [{att_key}]  {os.path.basename(path)}")
                continue
            else:
                print(f"  ✗ cloud  node {n['id']:>3}  [{att_key}]  NOT ON DISK")

    # Estrategia 2: buscar por nombre de archivo (nodos sin cloud)
    found = find_by_name(old_pdf)
    if found:
        path = found.replace("\\", "/")
        n["local_pdf"] = PREFIX + path
        fallback += 1
        print(f"  ~ fname  node {n['id']:>3}  {os.path.basename(path)}")
        continue

    # Sin PDF local verificado
    n["local_pdf"] = ""
    skipped += 1

print(f"\nResultado: {fixed} por cloud-key, {fallback} por nombre, {skipped} sin PDF local")

# ── escribir HTML ─────────────────────────────────────────────────────────────
new_nodes = json.dumps(nodes, ensure_ascii=False, separators=(",", ":"))
new_html  = html[:m.start()] + m.group(1) + new_nodes + m.group(3) + html[m.end():]
pathlib.Path(HTML).write_text(new_html, encoding="utf-8")
print("HTML actualizado OK")
