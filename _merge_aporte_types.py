"""
Fusiona aporte_metod → aporte_teorico y renombra la etiqueta a
"Aporte teórico / metodológico" en NODES, PAPER_TYPES y el panel stipo.
"""
import re, json, pathlib, collections

HTML = r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html"
html = pathlib.Path(HTML).read_text(encoding='utf-8')

# ── 1. Actualizar NODES ───────────────────────────────────────────────────────
m = re.search(r'(const NODES=)(\[.*?\])(;)', html, re.DOTALL)
nodes = json.loads(m.group(2))

changed = 0
for n in nodes:
    if n.get('paper_type') == 'aporte_metod':
        n['paper_type'] = 'aporte_teorico'
        changed += 1

print(f'Nodos reclasificados: {changed}')

# Contar para reconstruir stipo
type_counts = collections.Counter(n.get('paper_type', '') for n in nodes)
print('Distribución actualizada:')
for k, v in type_counts.most_common():
    if k: print(f'  {v:>3}  {k}')

new_nodes = json.dumps(nodes, ensure_ascii=False, separators=(',', ':'))
html = html[:m.start()] + m.group(1) + new_nodes + m.group(3) + html[m.end():]
print('✓ NODES actualizados')

# ── 2. Actualizar PAPER_TYPES constant en el JS ───────────────────────────────
# Buscar y reemplazar el objeto PAPER_TYPES en el JS
pt_m = re.search(r'const PAPER_TYPES=(\{.*?\});', html, re.DOTALL)
if pt_m:
    pt_old = json.loads(pt_m.group(1))
    # Renombrar label de aporte_teorico
    if 'aporte_teorico' in pt_old:
        pt_old['aporte_teorico']['label'] = 'Aporte teórico / metodológico'
    # Eliminar aporte_metod
    pt_old.pop('aporte_metod', None)
    pt_new_js = json.dumps(pt_old, ensure_ascii=False, separators=(',', ':'))
    html = html[:pt_m.start()] + 'const PAPER_TYPES=' + pt_new_js + ';' + html[pt_m.end():]
    print('✓ PAPER_TYPES actualizado')
else:
    print('WARN: PAPER_TYPES no encontrado')

# ── 3. Reconstruir el div stipo ───────────────────────────────────────────────
# Obtener PAPER_TYPES del HTML actualizado (ya sin aporte_metod)
pt_m2 = re.search(r'const PAPER_TYPES=(\{.*?\});', html, re.DOTALL)
paper_types = json.loads(pt_m2.group(1))

stipo_items = []
for key, info in paper_types.items():
    cnt = type_counts.get(key, 0)
    if cnt == 0:
        continue
    stipo_items.append(
        f'<div class="li" id="ltipo_{key}" onclick="toggleTipoF(\'{key}\')">'
        f'<span class="ld" style="background:{info["color"]}"></span>'
        f'<span class="lt">{info["label"]}</span>'
        f'<span class="lc">{cnt}</span></div>'
    )

new_stipo = '<div class="ls" id="stipo" style="display:none">' + ''.join(stipo_items) + '</div>'

# Reemplazar el div stipo existente — contar divs para encontrar el cierre correcto
idx_open = html.find('<div class="ls" id="stipo"')
idx_content = html.find('>', idx_open) + 1  # tras el '>' de apertura
depth, i = 1, idx_content
while depth > 0 and i < len(html):
    if html[i:i+5] == '<div ':
        depth += 1; i += 5
    elif html[i:i+6] == '</div>':
        depth -= 1; i += 6
    else:
        i += 1
html = html[:idx_open] + new_stipo + html[i:]
print('✓ stipo div reconstruido')

# ── Guardar ───────────────────────────────────────────────────────────────────
pathlib.Path(HTML).write_text(html, encoding='utf-8')
print('\nHTML guardado OK')
