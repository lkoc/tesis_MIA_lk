"""
Diagnostica y corrige el stipo fugado.
"""
import pathlib

HTML = pathlib.Path(r'c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html')
html = HTML.read_text(encoding='utf-8')

# Localizar el div stipo
idx_stipo = html.find('id="stipo"')
if idx_stipo == -1:
    print("ERROR: no encontró id=stipo")
    exit(1)

# Buscar el ltipo_guia_tecnica item dentro de stipo
idx_guia = html.find('ltipo_guia_tecnica', idx_stipo)
if idx_guia == -1:
    print("ERROR: no encontró ltipo_guia_tecnica")
    exit(1)

# Desde ltipo_guia_tecnica, encontrar el cierre del item (</div>)
idx_close_item = html.find('</div>', idx_guia)
# Ese cierra el .li de guia_tecnica
# El siguiente </div> cierra el stipo
idx_close_stipo = html.find('</div>', idx_close_item + 6)

print("Tras cierre stipo (chars 0-200):")
after = idx_close_stipo + 6
print(repr(html[after:after+200]))

# Encontrar el newline después del cierre del stipo
nl = html.find('\n', after)
if nl == -1:
    nl = len(html)

leaked = html[after:nl]
print(f"\nContenido fugado ({len(leaked)} chars):")
print(repr(leaked[:150]))

if not leaked.strip():
    print("\nNo hay contenido fugado — ya está bien.")
    exit(0)

# Eliminar el contenido fugado (hasta el newline)
html_fixed = html[:after] + html[nl:]
HTML.write_text(html_fixed, encoding='utf-8')
print("\n✓ Layout restaurado — HTML guardado OK")
