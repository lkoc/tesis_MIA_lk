"""
Agrega paper_type a los NODES y añade:
  1. Campo paper_type en cada nodo (derivado de tags tipo-*)
  2. Badge en panel de detalle
  3. Tab "Tipos" en sidebar con filtro interactivo
  4. filterBy() actualizado para tipo
"""
import re, json, pathlib, collections

HTML = r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html"

# ── Mapeo prefijo → macro-clave (orden importa: más específico primero) ───────
PREFIXES = [
    ('tipo-systematic-review',  'estado_arte'),
    ('tipo-estado-del-arte',    'estado_arte'),
    ('tipo-aporte-teorico',     'aporte_teorico'),
    ('tipo-aporte-metodologico','aporte_metod'),
    ('tipo-aporte-didactico',   'apoyo'),
    ('tipo-apoyo',              'apoyo'),
    ('tipo-aplicacion',         'aplicacion'),
    ('tipo-normativa',          'normativa'),
    ('tipo-fuente-clasica',     'fuente_clasica'),
    ('tipo-guia-tecnica',       'guia_tecnica'),
    ('tipo-catalogo',           'guia_tecnica'),
]

PAPER_TYPES = {
    'estado_arte':    {'label': 'Estado del arte',      'color': '#00838f'},
    'aporte_teorico': {'label': 'Aporte teórico',        'color': '#6a1b9a'},
    'aporte_metod':   {'label': 'Aporte metodológico',   'color': '#283593'},
    'aplicacion':     {'label': 'Aplicación',            'color': '#e65100'},
    'normativa':      {'label': 'Normativa / Estándar',  'color': '#37474f'},
    'fuente_clasica': {'label': 'Fuente clásica',        'color': '#4e342e'},
    'apoyo':          {'label': 'Apoyo / Didáctico',     'color': '#558b2f'},
    'guia_tecnica':   {'label': 'Guía técnica',          'color': '#0277bd'},
}

def get_paper_type(tags):
    for tag in tags:
        tl = tag.lower()
        for prefix, key in PREFIXES:
            if tl.startswith(prefix):
                return key
    return ''

# ── Leer HTML ─────────────────────────────────────────────────────────────────
html = pathlib.Path(HTML).read_text(encoding='utf-8')

# ── 1. Parsear NODES y agregar paper_type ─────────────────────────────────────
m = re.search(r'(const NODES=)(\[.*?\])(;)', html, re.DOTALL)
nodes = json.loads(m.group(2))

type_counts = collections.Counter()
for n in nodes:
    tags = n.get('tags', [])
    if isinstance(tags, str): tags = [tags]
    pt = get_paper_type(tags)
    n['paper_type'] = pt
    type_counts[pt or '_sin_tipo'] += 1

print('Distribución de tipos:')
for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
    lbl = PAPER_TYPES[k]['label'] if k in PAPER_TYPES else 'Sin clasificar'
    print(f'  {v:>3}  {lbl} [{k}]')

new_nodes = json.dumps(nodes, ensure_ascii=False, separators=(',', ':'))
html = html[:m.start()] + m.group(1) + new_nodes + m.group(3) + html[m.end():]
print('✓ NODES actualizados')

# ── 2. Agregar PAPER_TYPES constant antes de EDGES ───────────────────────────
pt_js = json.dumps(PAPER_TYPES, ensure_ascii=False, separators=(',', ':'))
assert 'const EDGES=' in html, "EDGES not found"
html = html.replace('const EDGES=', f'const PAPER_TYPES={pt_js};\nconst EDGES=', 1)
print('✓ PAPER_TYPES constant')

# ── 3. Agregar función ptBadge() antes de fixMoji ────────────────────────────
badge_fn = ('function ptBadge(pt){'
            'const p=PAPER_TYPES[pt];if(!p)return\'\';'
            'return\'<span class="bk" style="background:\'+p.color'
            '+\';font-style:italic">\'+p.label+\'</span>\';}\n')
assert 'function fixMoji(s)' in html
html = html.replace('function fixMoji(s)', badge_fn + 'function fixMoji(s)', 1)
print('✓ ptBadge() agregado')

# ── 4. Actualizar showD: badge después de comm badge ─────────────────────────
OLD_BADGES = '<span class="bk" style="background:${d.comm_color}">${d.comm_label}</span></div>'
NEW_BADGES  = '<span class="bk" style="background:${d.comm_color}">${d.comm_label}</span>${ptBadge(d.paper_type)}</div>'
assert OLD_BADGES in html
html = html.replace(OLD_BADGES, NEW_BADGES, 1)
print('✓ Badge tipo en showD')

# ── 5. Construir HTML del panel stipo ─────────────────────────────────────────
stipo_items = []
for key, info in PAPER_TYPES.items():
    cnt = type_counts.get(key, 0)
    if cnt == 0:
        continue
    stipo_items.append(
        f'<div class="li" id="ltipo_{key}" onclick="toggleTipoF(\'{key}\')">'
        f'<span class="ld" style="background:{info["color"]}"></span>'
        f'<span class="lt">{info["label"]}</span>'
        f'<span class="lc">{cnt}</span></div>'
    )
stipo_div = '    <div class="ls" id="stipo" style="display:none">' + ''.join(stipo_items) + '</div>'

# ── 6. Insertar tab "Tipos" en ptabs ─────────────────────────────────────────
OLD_TAB = "onclick=\"swTab('comm')\">Grupos TF-IDF</div>"
NEW_TAB  = OLD_TAB + '\n      <div class="pt" id="ttipo" onclick="swTab(\'tipo\')">Tipos</div>'
assert OLD_TAB in html
html = html.replace(OLD_TAB, NEW_TAB, 1)
print('✓ Tab Tipos')

# ── 7. Insertar stipo div después de la línea scomm ──────────────────────────
scomm_marker = '<div class="ls" id="scomm"'
idx = html.find(scomm_marker)
assert idx != -1, "scomm not found"
end_of_line = html.find('\n', idx)
if end_of_line == -1:
    end_of_line = len(html)
html = html[:end_of_line] + '\n' + stipo_div + html[end_of_line:]
print('✓ stipo div insertado')

# ── 8. Agregar cTipoF y toggleTipoF ──────────────────────────────────────────
tipo_filter_js = (
    'let cTipoF=null;\n'
    'function toggleTipoF(tipo){'
    'document.querySelectorAll(\'[id^="ltipo_"]\').forEach(el=>el.classList.remove(\'on\'));'
    'if(cTipoF===tipo){cTipoF=null;showAll();return;}'
    'cTipoF=tipo;document.getElementById(\'ltipo_\'+tipo)?.classList.add(\'on\');'
    'filterBy(\'tipo\',tipo);}\n'
)
assert 'function swTab(t){' in html
html = html.replace('function swTab(t){', tipo_filter_js + 'function swTab(t){', 1)
print('✓ toggleTipoF agregado')

# ── 9. Actualizar swTab para manejar 3 tabs ───────────────────────────────────
OLD_SW = ("document.getElementById('scomm').style.display=t==='comm'?'':'none';"
          "document.getElementById('tcat').classList.toggle('on',t==='cat');"
          "document.getElementById('tcomm').classList.toggle('on',t==='comm');}")
NEW_SW = ("document.getElementById('scomm').style.display=t==='comm'?'':'none';"
          "document.getElementById('stipo').style.display=t==='tipo'?'':'none';"
          "document.getElementById('tcat').classList.toggle('on',t==='cat');"
          "document.getElementById('tcomm').classList.toggle('on',t==='comm');"
          "document.getElementById('ttipo').classList.toggle('on',t==='tipo');}")
assert OLD_SW in html, f"swTab pattern not found"
html = html.replace(OLD_SW, NEW_SW, 1)
print('✓ swTab actualizado')

# ── 10. Actualizar filterBy para tipo ────────────────────────────────────────
OLD_F1 = "hidden:type==='cat'?n.category!==val:n.community!==val"
NEW_F1  = "hidden:type==='cat'?n.category!==val:type==='comm'?n.community!==val:n.paper_type!==val"
assert OLD_F1 in html
html = html.replace(OLD_F1, NEW_F1, 1)

OLD_F2 = "const aM=type==='cat'?nA.category===val:nA.community===val;"
NEW_F2  = "const aM=type==='cat'?nA.category===val:type==='comm'?nA.community===val:nA.paper_type===val;"
assert OLD_F2 in html
html = html.replace(OLD_F2, NEW_F2, 1)

OLD_F3 = "const bM=type==='cat'?nB.category===val:nB.community===val;"
NEW_F3  = "const bM=type==='cat'?nB.category===val:type==='comm'?nB.community===val:nB.paper_type===val;"
assert OLD_F3 in html
html = html.replace(OLD_F3, NEW_F3, 1)
print('✓ filterBy actualizado')

# ── Guardar ───────────────────────────────────────────────────────────────────
pathlib.Path(HTML).write_text(html, encoding='utf-8')
print('\nHTML guardado OK')
