"""
Genera una versión completamente offline de zotero_network.html:
  - Embebe vis-network JS y CSS inline (no CDN)
  - Copia los 65 PDFs a pdfs/
  - Reemplaza rutas absolutas local_pdf → rutas relativas ./pdfs/filename
Salida: zotero_offline/zotero_network.html  +  zotero_offline/pdfs/*.pdf
"""
import re, json, os, shutil, pathlib, urllib.request

SRC_HTML  = r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html"
OUT_DIR   = r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_offline"
OUT_HTML  = os.path.join(OUT_DIR, "zotero_network.html")
PDFS_DIR  = os.path.join(OUT_DIR, "pdfs")
STORE     = r"C:\Users\QU1267\Zotero\storage"

VIS_JS  = "https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"
VIS_CSS = "https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css"

os.makedirs(PDFS_DIR, exist_ok=True)

# ── 1. Descargar vis-network ───────────────────────────────────────────────────
print("Descargando vis-network JS...", end=" ", flush=True)
with urllib.request.urlopen(VIS_JS, timeout=30) as r:
    vis_js_text = r.read().decode("utf-8")
print(f"OK ({len(vis_js_text)//1024} KB)")

print("Descargando vis-network CSS...", end=" ", flush=True)
with urllib.request.urlopen(VIS_CSS, timeout=30) as r:
    vis_css_text = r.read().decode("utf-8")
print(f"OK ({len(vis_css_text)//1024} KB)")

# ── 2. Leer HTML fuente ────────────────────────────────────────────────────────
html = pathlib.Path(SRC_HTML).read_text(encoding="utf-8")

# ── 3. Reemplazar <script src CDN> → <script> inline </script> ────────────────
_js_block = f'<script>\n{vis_js_text}\n</script>'
html = re.sub(
    r'<script src="https://unpkg\.com/vis-network[^"]*"></script>',
    lambda m: _js_block,
    html
)

# ── 4. Reemplazar <link href CDN css> → <style> inline </style> ───────────────
_css_block = f'<style>\n{vis_css_text}\n</style>'
html = re.sub(
    r'<link\s+href="https://unpkg\.com/vis-network[^"]*"\s+rel="stylesheet">',
    lambda m: _css_block,
    html
)

# ── 5. Actualizar NODES: local_pdf absoluto → relativo ./pdfs/filename ─────────
m = re.search(r"(const NODES=)(\[.*?\])(;)", html, re.DOTALL)
if not m:
    print("ERROR: no se encontró const NODES="); exit(1)

nodes = json.loads(m.group(2))
copied = skipped = 0

for n in nodes:
    lp = n.get("local_pdf", "")
    if not lp:
        continue
    # lp es  file:///C:/Users/.../storage/KEY/filename.pdf
    src_path = lp.replace("file:///", "").replace("/", os.sep)
    filename = os.path.basename(src_path)
    dst_path = os.path.join(PDFS_DIR, filename)

    if os.path.isfile(src_path):
        shutil.copy2(src_path, dst_path)
        n["local_pdf"] = f"./pdfs/{filename}"
        copied += 1
        print(f"  ✓ {filename}")
    else:
        n["local_pdf"] = ""
        skipped += 1
        print(f"  ✗ MISSING {filename}")

print(f"\nPDFs copiados: {copied}, faltantes: {skipped}")

# ── 6. Reconstruir NODES en HTML ──────────────────────────────────────────────
new_nodes = json.dumps(nodes, ensure_ascii=False, separators=(",", ":"))
html = html[:m.start()] + m.group(1) + new_nodes + m.group(3) + html[m.end():]

# ── 7. Escribir HTML de salida ─────────────────────────────────────────────────
pathlib.Path(OUT_HTML).write_text(html, encoding="utf-8")
size_mb = os.path.getsize(OUT_HTML) / 1024 / 1024
print(f"\nEscrito: {OUT_HTML}  ({size_mb:.1f} MB)")
print(f"PDFs en: {PDFS_DIR}")
print("Listo — carpeta zotero_offline es completamente autónoma")
