"""
Genera una visualización interactiva HTML de la biblioteca Zotero
usando vis.js Network. Lee zotero_library.json y produce
zotero_network.html con grafo navegable por temas.
"""
import json, re
from pathlib import Path

# ───────────────────────────────────────────────
# 1. CARGAR DATOS
# ───────────────────────────────────────────────
with open("zotero_library.json", encoding="utf-8") as f:
    raw = json.load(f)

# ───────────────────────────────────────────────
# 2. DEDUPLICAR (misma clave title+year)
# ───────────────────────────────────────────────
seen = {}
for it in raw:
    key = (it.get("title") or "").strip().lower()[:80]
    yr  = (it.get("date") or "")[:4]
    uid = f"{key}|{yr}"
    if uid not in seen:
        seen[uid] = it
    else:
        # Merge collections and tags
        existing = seen[uid]
        existing["collections"] = list(set(
            (existing.get("collections") or []) + (it.get("collections") or [])))
        existing["tags"] = list(set(
            (existing.get("tags") or []) + (it.get("tags") or [])))
        # prefer longer abstract
        if len(it.get("abstractNote") or "") > len(existing.get("abstractNote") or ""):
            existing["abstractNote"] = it["abstractNote"]
        if not existing.get("DOI") and it.get("DOI"):
            existing["DOI"] = it["DOI"]
        if not existing.get("url") and it.get("url"):
            existing["url"] = it["url"]

items = list(seen.values())

# ───────────────────────────────────────────────
# 3. CATEGORIZAR
# ───────────────────────────────────────────────
CATEGORIES = {
    "PINN":            {"color": "#9c27b0", "label": "PINNs",                  "order": 6},
    "FEM/Numérico":    {"color": "#ff9800", "label": "Métodos Numéricos FEM",  "order": 2},
    "Fundamentos TC":  {"color": "#e91e63", "label": "Fundamentos T. Calor",   "order": 1},
    "Cable/Ampacidad": {"color": "#ffc107", "label": "Cables & Ampacidad",     "order": 3},
    "Suelo/Backfill":  {"color": "#4caf50", "label": "Suelo & Backfill",       "order": 4},
    "DTR/Dinámico":    {"color": "#2196f3", "label": "DTR & Cargas Cíclicas",  "order": 5},
    "Híbrido":         {"color": "#00bcd4", "label": "Aplicación Híbrida",     "order": 7},
    "Norma":           {"color": "#607d8b", "label": "Normas / Estándares",    "order": 8},
    "Otro":            {"color": "#9e9e9e", "label": "Otro",                   "order": 9},
}

def categorize(it):
    tags_str  = " ".join(it.get("tags") or []).lower()
    title_str = (it.get("title") or "").lower()
    colls     = " ".join(it.get("collections") or []).lower()
    combined  = f"{tags_str} {title_str} {colls}"

    if "pinn" in combined or "physics-informed" in combined or "physics informed" in combined:
        return "PINN"
    if ("iec 60287" in combined or "iec 60853" in combined or "ieee std" in combined
            or "astm" in combined or "ieee 442" in combined or "ieee 835" in combined
            or "iec 60502" in combined):
        return "Norma"
    if ("fem" in combined or "finite element" in combined or "fdm" in combined
            or "fvm" in combined or "finite difference" in combined
            or "finite volume" in combined or "metodos numericos" in combined
            or "numerical heat" in combined or "coleccion-02" in combined):
        return "FEM/Numérico"
    if ("coleccion-01" in combined or "fundamentos transferencia" in combined
            or "heat conduction" in title_str or "heat equation" in title_str
            or "heat transfer handbook" in title_str):
        return "Fundamentos TC"
    if ("backfill" in combined or "bedding" in combined or "suelos" in combined
            or "coleccion-04" in combined or "soil thermal" in combined
            or "soil moisture" in combined or "thermal resistivity" in combined
            or "thermal properties of soils" in title_str):
        return "Suelo/Backfill"
    if ("dtr" in combined or "dynamic thermal" in combined or "dynamic rating" in combined
            or "cyclic" in combined or "transient" in combined
            or "coleccion-05" in combined or "fluctuant load" in combined):
        return "DTR/Dinámico"
    if ("ampacity" in combined or "cable rating" in combined or "cable ampacity" in combined
            or "coleccion-03" in combined or "current rating" in combined
            or "ampacidad" in combined or "underground power cable" in combined
            or "xlpe" in combined):
        return "Cable/Ampacidad"
    if ("cnn" in combined or "lstm" in combined or "neural network" in combined
            or "machine learning" in combined or "deep learning" in combined
            or "bpnn" in combined or "pso" in combined):
        return "Híbrido"
    return "Otro"

# Assign IDs and categories
nodes_data = []
for idx, it in enumerate(items):
    cat = categorize(it)
    yr  = (it.get("date") or "????")[:4]
    authors = it.get("authors") or []
    first_author = authors[0].split()[-1] if authors else "?"
    all_authors  = "; ".join(authors[:4]) + (" et al." if len(authors) > 4 else "")
    title  = it.get("title") or "Sin título"
    label  = f"{first_author}\n{yr}"
    doi    = it.get("DOI") or ""
    url    = it.get("url") or ""
    link   = (f"https://doi.org/{doi}" if doi else url) or ""
    abstr  = (it.get("abstractNote") or "")[:600].replace('"', "'")
    tags   = [t for t in (it.get("tags") or [])
              if not t.startswith("#") and t not in ("nosource","#nosource","reference")]
    colls  = it.get("collections") or []
    pub    = it.get("publicationTitle") or ""

    # Importance weight (thesis-core gets bigger nodes)
    weight = 10
    combined_tags = " ".join(tags).lower()
    if "thesis-core" in combined_tags:         weight = 22
    elif "foundational" in combined_tags:       weight = 20
    elif "pinn" in combined_tags:               weight = 16
    elif "ampacity" in combined_tags:           weight = 14
    elif "iec 60287" in combined_tags:          weight = 14
    elif "review" in combined_tags or "systematic" in combined_tags: weight = 15
    elif "dynamic thermal rating" in combined_tags: weight = 14

    nodes_data.append({
        "id":       idx,
        "label":    label,
        "title_full": title,
        "year":     yr,
        "authors":  all_authors,
        "category": cat,
        "tags":     tags,
        "collections": colls,
        "abstract": abstr,
        "link":     link,
        "doi":      doi,
        "pub":      pub,
        "color":    CATEGORIES[cat]["color"],
        "size":     weight,
    })

# ───────────────────────────────────────────────
# 4. CREAR ARISTAS (por tags compartidos relevantes)
# ───────────────────────────────────────────────
LINK_TAGS = [
    "PINN", "ampacity", "IEC 60287", "FEM", "backfill", "soil thermal",
    "dynamic thermal rating", "Dynamic Thermal Rating", "cable rating",
    "underground power cables", "underground cables", "heat conduction",
    "heat transfer", "cyclic loads", "thermal conductivity",
    "inverse problem", "thesis-core", "climate change", "Climate change",
    "XLPE", "xlpe", "cable construction", "ampacidad y normas",
    "cargas cíclicas y operación", "bedding y humedad",
]

# Build tag → node_ids index
tag_index = {}
for n in nodes_data:
    for t in n["tags"]:
        if t in LINK_TAGS:
            tag_index.setdefault(t, []).append(n["id"])

edges_set = set()
edges_data = []
edge_id = 0
for tag, nids in tag_index.items():
    # Avoid too many edges for large groups: limit to 15 per tag
    group = nids[:15]
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            a, b = min(group[i], group[j]), max(group[i], group[j])
            key  = (a, b, tag)
            if key not in edges_set:
                edges_set.add(key)
                edges_data.append({"id": edge_id, "from": a, "to": b, "label": tag})
                edge_id += 1

# Known manual connections (important relationships)
MANUAL_EDGES = [
    # PINN foundational chain
    (124, 125, "Raissi→surveys"),   # Raissi → Lawal
    (124, 126, "Raissi→surveys"),   # Raissi → Ren
    (124, 127, "xPINN"),            # Raissi → Jagtap
    (124, 128, "parallel PINN"),    # Raissi → Shukla
    # PINN → cable applications
    (124, 65,  "PINN→FEM-BPNN"),   # Raissi → Al-Dulaimi
    (124, 23,  "PINN→DTR"),        # Raissi → Enescu 2021
    # Neher → IEC chain
    (53, 32, "Neher→IEC"),         # Neher-McGrath → IEC 60287
    (53, 11, "Neher→Aras"),        # Neher-McGrath → Aras 2005
    (32, 33, "IEC60287 update"),   # IEC 60287 old → 2023
    # Standards → Applications
    (9,  11, "Anders→Aras"),       # Anders 1997 → Aras 2005
    (32, 9,  "IEC→Anders"),        # IEC 60287 → Anders
]
# Map original item indices to node IDs (they're the same here since we enumerate)
for a, b, lbl in MANUAL_EDGES:
    if a < len(nodes_data) and b < len(nodes_data):
        na, nb = min(a,b), max(a,b)
        key = (na, nb, lbl)
        if key not in edges_set:
            edges_set.add(key)
            edges_data.append({"id": edge_id, "from": na, "to": nb, "label": lbl})
            edge_id += 1

# ───────────────────────────────────────────────
# 5. GENERAR HTML
# ───────────────────────────────────────────────
nodes_json = json.dumps(nodes_data, ensure_ascii=False)
edges_json = json.dumps(edges_data, ensure_ascii=False)
cats_json  = json.dumps(CATEGORIES, ensure_ascii=False)

# Build legend HTML
legend_items = ""
for cat_key, cat_val in sorted(CATEGORIES.items(), key=lambda x: x[1]["order"]):
    n_papers = sum(1 for n in nodes_data if n["category"] == cat_key)
    if n_papers == 0:
        continue
    legend_items += f"""
    <div class="legend-item" onclick="filterCategory('{cat_key}')">
      <span class="legend-dot" style="background:{cat_val['color']}"></span>
      <span>{cat_val['label']} ({n_papers})</span>
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Red Bibliográfica – Tesis MIA: PINNs para Cables Enterrados</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<link href="https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }}
  #header {{ background: #1a1a2e; padding: 10px 20px; border-bottom: 2px solid #9c27b0; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  #header h1 {{ font-size: 1.1em; color: #ce93d8; flex: 1; min-width: 200px; }}
  #header .stats {{ font-size: 0.8em; color: #888; }}
  #search-box {{ padding: 6px 12px; border-radius: 20px; border: 1px solid #444; background: #2a2a3e; color: #eee; width: 220px; font-size: 0.85em; }}
  #search-box:focus {{ outline: none; border-color: #9c27b0; }}
  #main {{ display: flex; flex: 1; overflow: hidden; }}
  #network {{ flex: 1; background: #12121e; }}
  #sidebar {{ width: 340px; background: #1a1a2e; border-left: 1px solid #333; display: flex; flex-direction: column; overflow: hidden; }}
  #legend {{ padding: 12px; border-bottom: 1px solid #333; }}
  #legend h3 {{ font-size: 0.78em; color: #aaa; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 6px; cursor: pointer; font-size: 0.8em; transition: background 0.2s; }}
  .legend-item:hover {{ background: #252540; }}
  .legend-item.active {{ background: #2d1b4e; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  #detail-panel {{ flex: 1; overflow-y: auto; padding: 14px; }}
  #detail-panel h3 {{ font-size: 0.9em; color: #ce93d8; margin-bottom: 10px; }}
  .detail-title {{ font-size: 0.95em; font-weight: bold; color: #fff; margin-bottom: 6px; line-height: 1.4; }}
  .detail-authors {{ font-size: 0.8em; color: #a0c4ff; margin-bottom: 4px; }}
  .detail-year {{ font-size: 0.8em; color: #ffd54f; margin-bottom: 4px; }}
  .detail-pub {{ font-size: 0.78em; color: #80cbc4; margin-bottom: 8px; font-style: italic; }}
  .detail-cat {{ display: inline-block; font-size: 0.72em; padding: 2px 10px; border-radius: 12px; color: #fff; margin-bottom: 8px; }}
  .detail-abstract {{ font-size: 0.78em; color: #bbb; line-height: 1.6; margin-bottom: 10px; max-height: 200px; overflow-y: auto; border-left: 3px solid #444; padding-left: 8px; }}
  .detail-tags {{ display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; }}
  .tag-badge {{ font-size: 0.7em; padding: 2px 8px; border-radius: 10px; background: #252540; color: #ce93d8; border: 1px solid #444; }}
  .detail-link {{ display: inline-block; padding: 6px 14px; background: #7b1fa2; color: #fff; border-radius: 16px; text-decoration: none; font-size: 0.8em; margin-top: 4px; transition: background 0.2s; }}
  .detail-link:hover {{ background: #9c27b0; }}
  .detail-colls {{ font-size: 0.72em; color: #666; margin-top: 6px; }}
  #placeholder {{ display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #555; text-align: center; padding: 20px; }}
  #placeholder .icon {{ font-size: 3em; margin-bottom: 12px; }}
  #controls {{ padding: 8px 12px; border-bottom: 1px solid #333; display: flex; gap: 8px; flex-wrap: wrap; }}
  .ctrl-btn {{ padding: 4px 12px; border-radius: 12px; border: 1px solid #444; background: #252540; color: #ccc; cursor: pointer; font-size: 0.75em; transition: all 0.2s; }}
  .ctrl-btn:hover {{ background: #3a3a5c; color: #fff; }}
  #node-count {{ font-size: 0.72em; color: #666; padding: 4px 12px; }}
  .neighbors-list {{ font-size: 0.78em; margin-top: 8px; }}
  .neighbors-list summary {{ cursor: pointer; color: #888; margin-bottom: 4px; }}
  .neighbors-list ul {{ list-style: none; padding-left: 0; }}
  .neighbors-list li {{ padding: 3px 6px; cursor: pointer; color: #a0c4ff; border-radius: 4px; }}
  .neighbors-list li:hover {{ background: #252540; }}
  #filter-active {{ font-size: 0.72em; color: #ff9800; padding: 4px 12px; min-height: 22px; }}
</style>
</head>
<body>
<div id="header">
  <h1>🔬 Red Bibliográfica — Tesis MIA: PINNs para Cálculo Térmico de Cables Enterrados</h1>
  <span class="stats" id="stats-label"></span>
  <input type="text" id="search-box" placeholder="🔍 Buscar autor, título, tag..." oninput="searchNodes(this.value)">
</div>
<div id="main">
  <div id="network"></div>
  <div id="sidebar">
    <div id="controls">
      <button class="ctrl-btn" onclick="resetView()">↺ Reset</button>
      <button class="ctrl-btn" onclick="toggleEdges()">Conexiones</button>
      <button class="ctrl-btn" onclick="showAll()">Mostrar todos</button>
      <button class="ctrl-btn" onclick="fitNetwork()">Ajustar</button>
    </div>
    <div id="filter-active"></div>
    <div id="legend">
      <h3>Categorías (click para filtrar)</h3>
      {legend_items}
    </div>
    <div id="node-count"></div>
    <div id="detail-panel">
      <div id="placeholder">
        <div class="icon">📚</div>
        <p>Haz clic en un nodo para ver los detalles del paper</p>
        <p style="font-size:0.8em;margin-top:8px;color:#444">Usa el scroll para zoom · Arrastra para mover</p>
      </div>
    </div>
  </div>
</div>

<script>
const NODES_DATA = {nodes_json};
const EDGES_DATA = {edges_json};
const CATEGORIES = {cats_json};

// Build vis DataSets
const nodesDS = new vis.DataSet(NODES_DATA.map(n => ({{
  id: n.id,
  label: n.label,
  color: {{
    background: n.color,
    border: lightenColor(n.color, 40),
    highlight: {{ background: lightenColor(n.color, 60), border: '#fff' }},
    hover: {{ background: lightenColor(n.color, 30), border: '#fff' }}
  }},
  size: n.size,
  font: {{ color: '#fff', size: 10, strokeWidth: 2, strokeColor: '#000' }},
  shape: 'dot',
  borderWidth: 1.5,
  _data: n
}})));

const edgesDS = new vis.DataSet(EDGES_DATA.map(e => ({{
  id: e.id, from: e.from, to: e.to,
  label: '',
  color: {{ color: '#333355', highlight: '#9c27b0', hover: '#7e57c2' }},
  width: 1,
  smooth: {{ type: 'continuous', roundness: 0.3 }},
  _label: e.label
}})));

const options = {{
  nodes: {{ borderWidth: 1, shadow: {{ enabled: true, color: 'rgba(0,0,0,0.5)', size: 8 }} }},
  edges: {{ smooth: {{ type: 'continuous' }}, hoverWidth: 2, selectionWidth: 3 }},
  physics: {{
    enabled: true,
    forceAtlas2Based: {{
      gravitationalConstant: -60,
      centralGravity: 0.008,
      springLength: 120,
      springConstant: 0.08,
      damping: 0.5,
      avoidOverlap: 0.6
    }},
    solver: 'forceAtlas2Based',
    stabilization: {{ iterations: 200, updateInterval: 25 }}
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 200,
    navigationButtons: false,
    keyboard: {{ enabled: true, speed: {{ x: 10, y: 10, zoom: 0.03 }} }}
  }}
}};

const container = document.getElementById('network');
const network   = new vis.Network(container, {{ nodes: nodesDS, edges: edgesDS }}, options);
let edgesVisible = true;
let activeFilter = null;

// Update stats
document.getElementById('stats-label').textContent =
  `${{NODES_DATA.length}} papers · ${{EDGES_DATA.length}} conexiones`;
updateNodeCount();

// Click on node
network.on('click', function(params) {{
  if (params.nodes.length > 0) {{
    const node = nodesDS.get(params.nodes[0]);
    showDetail(node._data, params.nodes[0]);
  }} else {{
    document.getElementById('detail-panel').innerHTML = `
      <div id="placeholder">
        <div class="icon">📚</div>
        <p>Haz clic en un nodo para ver los detalles</p>
      </div>`;
  }}
}});

// Hover tooltip
network.on('hoverNode', function(params) {{
  const node = nodesDS.get(params.node);
  const d = node._data;
  container.title = `${{d.title_full}}\\n${{d.authors}} (${{d.year}})`;
}});

function showDetail(d, nodeId) {{
  const catInfo = CATEGORIES[d.category] || {{}};
  const link = d.link ? `<a href="${{d.link}}" target="_blank" class="detail-link">🔗 Abrir paper</a>` : '';
  
  // Get connected papers
  const connectedEdges = network.getConnectedEdges(nodeId);
  const connectedNodes = network.getConnectedNodes(nodeId);
  let neighborsHtml = '';
  if (connectedNodes.length > 0) {{
    const items = connectedNodes.slice(0, 12).map(nid => {{
      const cn = nodesDS.get(nid)._data;
      return `<li onclick="jumpToNode(${{nid}})" title="${{cn.title_full}}">${{cn.authors.split(';')[0].trim().split(' ').pop()}} (${{cn.year}}) – ${{cn.title_full.substring(0,55)}}...</li>`;
    }}).join('');
    neighborsHtml = `<details class="neighbors-list"><summary>📎 ${{connectedNodes.length}} papers conectados</summary><ul>${{items}}</ul></details>`;
  }}
  
  const tagsHtml = d.tags.filter(t=>t&&t!='nosource').slice(0,12)
    .map(t => `<span class="tag-badge">${{t}}</span>`).join('');
  const collsHtml = d.collections.map(c=>`<span style="font-style:italic">${{c}}</span>`).join(', ');

  document.getElementById('detail-panel').innerHTML = `
    <h3>📄 Detalle del Paper</h3>
    <div class="detail-title">${{d.title_full}}</div>
    <div class="detail-authors">👤 ${{d.authors || 'Autor desconocido'}}</div>
    <div class="detail-year">📅 ${{d.year || 'Sin año'}}${{d.pub ? ' · ' + d.pub : ''}}</div>
    <span class="detail-cat" style="background:${{catInfo.color || '#555'}}">${{catInfo.label || d.category}}</span>
    ${{d.abstract ? `<div class="detail-abstract">${{d.abstract}}${{d.abstract.length>=600?'…':''}}</div>` : '<div style="font-size:0.78em;color:#555;margin-bottom:8px;font-style:italic">Sin abstract disponible</div>'}}
    <div class="detail-tags">${{tagsHtml}}</div>
    ${{link}}
    ${{neighborsHtml}}
    <div class="detail-colls">📁 ${{collsHtml || 'Sin colección'}}</div>
  `;
}}

function jumpToNode(nid) {{
  network.focus(nid, {{ scale: 1.5, animation: {{ duration: 800, easingFunction: 'easeInOutQuad' }} }});
  network.selectNodes([nid]);
  const node = nodesDS.get(nid);
  showDetail(node._data, nid);
}}

function filterCategory(cat) {{
  activeFilter = activeFilter === cat ? null : cat;
  document.getElementById('filter-active').textContent =
    activeFilter ? `Filtro activo: ${{CATEGORIES[activeFilter]?.label || activeFilter}}` : '';
  
  // Update legend active state
  document.querySelectorAll('.legend-item').forEach((el, i) => el.classList.remove('active'));
  
  const updates = NODES_DATA.map(n => ({{
    id: n.id,
    hidden: activeFilter ? n.category !== activeFilter : false
  }}));
  nodesDS.update(updates);
  
  const edgeUpdates = EDGES_DATA.map(e => {{
    const fromNode = NODES_DATA[e.from];
    const toNode   = NODES_DATA[e.to];
    const hidden = activeFilter
      ? (fromNode?.category !== activeFilter && toNode?.category !== activeFilter)
      : false;
    return {{ id: e.id, hidden }};
  }});
  edgesDS.update(edgeUpdates);
  updateNodeCount();
  network.fit({{ animation: true }});
}}

function showAll() {{
  activeFilter = null;
  document.getElementById('filter-active').textContent = '';
  nodesDS.update(NODES_DATA.map(n => ({{ id: n.id, hidden: false }})));
  edgesDS.update(EDGES_DATA.map(e => ({{ id: e.id, hidden: false }})));
  updateNodeCount();
  fitNetwork();
}}

function toggleEdges() {{
  edgesVisible = !edgesVisible;
  edgesDS.update(EDGES_DATA.map(e => ({{ id: e.id, hidden: !edgesVisible }})));
}}

function resetView() {{
  showAll();
  network.setOptions({{ physics: {{ enabled: true }} }});
  setTimeout(() => network.setOptions({{ physics: {{ enabled: false }} }}), 3000);
}}

function fitNetwork() {{
  network.fit({{ animation: {{ duration: 800, easingFunction: 'easeInOutQuad' }} }});
}}

function searchNodes(query) {{
  const q = query.toLowerCase().trim();
  if (!q) {{ showAll(); return; }}
  const matched = new Set();
  NODES_DATA.forEach(n => {{
    const text = (n.title_full + ' ' + n.authors + ' ' + n.tags.join(' ')).toLowerCase();
    if (text.includes(q)) matched.add(n.id);
  }});
  nodesDS.update(NODES_DATA.map(n => ({{ id: n.id, hidden: !matched.has(n.id) }})));
  edgesDS.update(EDGES_DATA.map(e => ({{ id: e.id, hidden: !matched.has(e.from) || !matched.has(e.to) }})));
  document.getElementById('filter-active').textContent =
    matched.size > 0 ? `${{matched.size}} resultados para "${{q}}"` : `Sin resultados para "${{q}}"`;
  updateNodeCount();
  network.fit({{ animation: true }});
}}

function updateNodeCount() {{
  const visible = NODES_DATA.filter(n => !nodesDS.get(n.id).hidden).length;
  document.getElementById('node-count').textContent =
    `Mostrando ${{visible}} de ${{NODES_DATA.length}} papers`;
}}

function lightenColor(hex, amount) {{
  let r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  r = Math.min(255, r + amount); g = Math.min(255, g + amount); b = Math.min(255, b + amount);
  return '#' + [r,g,b].map(v => v.toString(16).padStart(2,'0')).join('');
}}

// Stop physics after stabilization
network.once('stabilizationIterationsDone', () => {{
  network.setOptions({{ physics: {{ enabled: false }} }});
  fitNetwork();
}});
</script>
</body>
</html>"""

out = Path("zotero_network.html")
out.write_text(html, encoding="utf-8")
print(f"✅ Generado: {out}  ({len(nodes_data)} nodos, {len(edges_data)} aristas)")
print(f"   Categorías: { {c: sum(1 for n in nodes_data if n['category']==c) for c in CATEGORIES} }")
