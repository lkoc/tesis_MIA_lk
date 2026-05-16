"""
Red bibliografica interactiva - Tesis MIA: PINNs para Cables Enterrados
v2: analisis semantico TF-IDF, comunidades, fondo blanco, PDFs locales, lazo.
"""
import json, re, math, sqlite3, random
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import quote

DATA_FILE      = "zotero_library.json"
OUTPUT_FILE    = "zotero_network.html"
ZOTERO_DB      = r"C:\Users\QU1267\Zotero\zotero.sqlite"
ZOTERO_STORAGE = r"C:\Users\QU1267\Zotero\storage"
MD_FILE        = "ANALISIS_BIBLIOGRAFICO_ZOTERO.md"

# ── 1. CARGAR Y DEDUPLICAR ─────────────────────────────────────────────────
with open(DATA_FILE, encoding="utf-8") as f:
    raw = json.load(f)

seen = {}
for it in raw:
    tk  = (it.get("title") or "").strip().lower()[:80]
    yr  = (it.get("date") or "")[:4]
    uid = f"{tk}|{yr}"
    iid = it.get("id")
    if uid not in seen:
        d = dict(it)
        d["_all_ids"] = [iid] if iid else []
        seen[uid] = d
    else:
        ex = seen[uid]
        if iid and iid not in ex["_all_ids"]:
            ex["_all_ids"].append(iid)
        ex["collections"] = list(set((ex.get("collections") or []) + (it.get("collections") or [])))
        ex["tags"]        = list(set((ex.get("tags") or [])        + (it.get("tags") or [])))
        if len(it.get("abstractNote") or "") > len(ex.get("abstractNote") or ""):
            ex["abstractNote"] = it["abstractNote"]
        for fld in ("DOI", "url"):
            if not ex.get(fld) and it.get(fld):
                ex[fld] = it[fld]

items = list(seen.values())
N     = len(items)
print(f"Papers unicos: {N}")

# ── 2. PDF LOCALES DESDE SQLITE ────────────────────────────────────────────
attach_map = {}
try:
    con = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("""
        SELECT i.key, ia.parentItemID, ia.path
        FROM items i JOIN itemAttachments ia ON i.itemID = ia.itemID
        WHERE ia.contentType = 'application/pdf' AND ia.path LIKE 'storage:%'
    """)
    for att_key, parent_id, path_str in cur.fetchall():
        if parent_id is not None and parent_id not in attach_map:
            fname = path_str[8:]
            base  = ZOTERO_STORAGE.replace("\\", "/")
            attach_map[parent_id] = f"file:///{base}/{att_key}/{quote(fname)}"
    con.close()
    covered = sum(1 for it in items for iid in it.get("_all_ids",[]) if iid in attach_map)
    print(f"PDFs en Zotero: {len(attach_map)} -> {covered} papers unicos con PDF local")
except Exception as e:
    print(f"Warning SQLite: {e}")

for it in items:
    it["local_pdf"] = ""
    for iid in it.get("_all_ids", []):
        if iid and iid in attach_map:
            it["local_pdf"] = attach_map[iid]
            break

# ── 3. CATEGORIZAR ─────────────────────────────────────────────────────────
CATEGORIES = {
    "PINN":            {"color": "#7b1fa2", "label": "PINNs",                 "order": 1},
    "FEM/Numerico":    {"color": "#bf360c", "label": "Metodos FEM/Numericos", "order": 2},
    "Fundamentos TC":  {"color": "#b71c1c", "label": "Fundamentos T. Calor",  "order": 3},
    "Cable/Ampacidad": {"color": "#e65100", "label": "Cables & Ampacidad",    "order": 4},
    "Suelo/Backfill":  {"color": "#1b5e20", "label": "Suelo & Backfill",      "order": 5},
    "DTR/Dinamico":    {"color": "#0d47a1", "label": "DTR & Cargas Ciclicas", "order": 6},
    "Hibrido":         {"color": "#006064", "label": "Aplic. Hibrida/ML",     "order": 7},
    "Norma":           {"color": "#37474f", "label": "Normas / Estandares",   "order": 8},
    "Otro":            {"color": "#757575", "label": "Otro",                  "order": 9},
}

def categorize(it):
    tg  = " ".join(it.get("tags") or []).lower()
    ti  = (it.get("title") or "").lower()
    co  = " ".join(it.get("collections") or []).lower()
    al  = f"{tg} {ti} {co}"
    if "pinn" in al or "physics-informed" in al or "physics informed" in al:
        return "PINN"
    if ("iec 60287" in al or "iec 60853" in al or "ieee std" in al
            or "astm" in al or "ieee 442" in al or "ieee 835" in al
            or "iec 60502" in al):
        return "Norma"
    if ("fem" in al or "finite element" in al or "fdm" in al
            or "finite difference" in al or "finite volume" in al
            or "numerical heat" in al or "coleccion-02" in al):
        return "FEM/Numerico"
    if ("coleccion-01" in al or "fundamentos transferencia" in al
            or "heat conduction" in ti or "heat equation" in ti
            or "heat transfer handbook" in ti):
        return "Fundamentos TC"
    if ("backfill" in al or "bedding" in al or "suelos" in al
            or "coleccion-04" in al or "soil thermal" in al
            or "soil moisture" in al or "thermal resistivity" in al
            or "thermal properties of soils" in ti):
        return "Suelo/Backfill"
    if ("dtr" in al or "dynamic thermal" in al or "dynamic rating" in al
            or "cyclic" in al or "transient" in al
            or "coleccion-05" in al or "fluctuant load" in al):
        return "DTR/Dinamico"
    if ("ampacity" in al or "cable rating" in al or "cable ampacity" in al
            or "coleccion-03" in al or "current rating" in al
            or "ampacidad" in al or "underground power cable" in al
            or "xlpe" in al):
        return "Cable/Ampacidad"
    if ("cnn" in al or "lstm" in al or "neural network" in al
            or "machine learning" in al or "deep learning" in al
            or "bpnn" in al or "pso" in al):
        return "Hibrido"
    return "Otro"

# ── 4. TF-IDF SEMANTICO ────────────────────────────────────────────────────
STOP = {
    "the","a","an","of","in","and","or","to","for","is","are","was","were",
    "this","that","these","those","with","from","at","by","be","been","as",
    "which","on","into","its","it","can","has","have","both","also","through",
    "their","they","such","we","our","not","all","more","when","than","two",
    "one","three","five","ten","new","high","low","large","small","good",
    "different","various","other","same","well","thus","here","each","first",
    "second","third","show","shows","shown","based","using","used","proposed",
    "results","paper","study","approach","analysis","data","system","method",
    "methods","model","models","performance","problem","about","between",
    "under","over","some","many","most","will","may","has","have","been",
    "de","la","las","los","el","en","para","con","una","por","que","del",
    "se","al","es","un","una","number","type","also","used","time","very",
}

def tok(text):
    return [w for w in re.findall(r"[a-z\xe1\xe9\xed\xf3\xfa\xfc\xf1]{3,}", text.lower())
            if w not in STOP]

def doc_text(it):
    return " ".join([
        (it.get("title") or "") * 3,
        " ".join(it.get("tags") or []) * 2,
        it.get("abstractNote") or "",
        it.get("publicationTitle") or "",
    ])

docs_tok = [tok(doc_text(it)) for it in items]

df_ctr = Counter()
for tokens in docs_tok:
    for w in set(tokens):
        df_ctr[w] += 1
idf = {w: math.log((N+1)/(df+1))+1 for w, df in df_ctr.items() if df >= 2}

def tfidf(tokens):
    tf   = Counter(tokens)
    tot  = max(len(tokens), 1)
    vec  = {w: (cnt/tot)*idf[w] for w, cnt in tf.items() if w in idf}
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
    return {w: v/norm for w, v in vec.items()}

def cosine(v1, v2):
    com = set(v1) & set(v2)
    return sum(v1[w]*v2[w] for w in com) if com else 0.0

vecs = [tfidf(t) for t in docs_tok]

SIM_THR = 0.07
TOP_K   = 7
print("Calculando similitud semantica TF-IDF...")
sim_edges = {}
for i in range(N):
    sims = [(cosine(vecs[i], vecs[j]), j) for j in range(i+1, N)]
    sims = [(s,j) for s,j in sims if s >= SIM_THR]
    sims.sort(reverse=True)
    for s, j in sims[:TOP_K]:
        k = (i,j)
        if k not in sim_edges or sim_edges[k] < s:
            sim_edges[k] = round(s, 4)
print(f"Aristas semanticas: {len(sim_edges)}")

# ── 5. COMUNIDADES (Spherical K-Means sobre vectores TF-IDF) ──────────────
K_COMM = 7  # numero de comunidades objetivo

random.seed(42)
# Inicializar centros con k-means++ (mejor distribucion inicial)
center_vecs = [vecs[random.randrange(N)]]
for _ in range(K_COMM - 1):
    dists = []
    for i in range(N):
        d = 1.0 - max((cosine(vecs[i], cv) for cv in center_vecs), default=0.0)
        dists.append(d)
    total = sum(dists)
    r = random.random() * total
    cumul = 0.0
    for i, d in enumerate(dists):
        cumul += d
        if cumul >= r:
            center_vecs.append(vecs[i])
            break
    else:
        center_vecs.append(vecs[-1])

assignments = {i: 0 for i in range(N)}
for _ in range(40):
    # Asignar cada paper al centro mas cercano (coseno)
    new_assign = {}
    for i in range(N):
        sims = [cosine(vecs[i], cv) for cv in center_vecs]
        new_assign[i] = sims.index(max(sims))
    if new_assign == assignments:
        break
    assignments = new_assign
    # Recalcular centros (promedio normalizado)
    new_cvecs = []
    for k in range(K_COMM):
        members = [i for i, a in assignments.items() if a == k]
        if not members:
            new_cvecs.append(center_vecs[k])
            continue
        merged = defaultdict(float)
        for i in members:
            for w, v in vecs[i].items():
                merged[w] += v
        n = len(members)
        avg = {w: v/n for w, v in merged.items()}
        norm = math.sqrt(sum(v*v for v in avg.values())) or 1.0
        new_cvecs.append({w: v/norm for w, v in avg.items()})
    center_vecs = new_cvecs

comms = dict(assignments)
n_comm = K_COMM
cs = Counter(comms.values())
sc = sorted(range(K_COMM), key=lambda k: -cs[k])
print(f"Comunidades semanticas (k-means k={K_COMM}):")

CPAL = ["#c62828","#6a1b9a","#1565c0","#2e7d32","#e65100",
        "#00695c","#827717","#4527a0","#1a237e","#004d40",
        "#bf360c","#880e4f","#33691e","#0277bd","#4e342e"]

def cc(cid): return CPAL[cid % len(CPAL)]

skip_kw = {"heat","cable","cables","thermal","power","underground","pinn",
           "method","model","equation","neural","rating","current","field",
           "temperature","conduction","transfer","soil","backfill","network",
           "based","using","proposed","results"}

def comm_info(cid):
    members = [items[i] for i,c in comms.items() if c == cid]
    top_cat = Counter(categorize(m) for m in members).most_common(1)[0][0]
    all_tok = []
    for m in members: all_tok.extend(tok(doc_text(m)))
    top_w = [w for w,_ in Counter(all_tok).most_common(25) if w not in skip_kw][:3]
    cnt = cs[cid]
    return {"label": f"G{cid+1}: {', '.join(top_w) if top_w else top_cat}",
            "cat": top_cat, "count": cnt, "color": cc(cid)}

comm_infos = [comm_info(cid) for cid in range(n_comm)]
print("Grupos semanticos:")
for cid, info in enumerate(comm_infos):
    print(f"  [{cid+1:2}] {info['label']}  ({info['count']} papers, dom: {info['cat']})")

# ── 6. APORTES DESDE MD ────────────────────────────────────────────────────
aportes = {}
try:
    md = Path(MD_FILE).read_text(encoding="utf-8")
    for sec in re.split(r'\n(?=### )', md):
        mh = re.match(r'###\s+(\S+)\s+\((\d{4}|0000)\)', sec)
        if not mh: continue
        key = f"{mh.group(1).lower()}|{mh.group(2)}"
        ma  = re.search(r'\*\*Aporte al proyecto:\*\*\s*(.+?)(?=\n-\s\*\*|\n###|\Z)',
                        sec, re.DOTALL)
        if ma:
            aportes[key] = ma.group(1).strip().replace('"', "'")
    print(f"Aportes cargados: {len(aportes)}")
except Exception as e:
    print(f"Warning aportes: {e}")

# ── 7. NODOS Y ARISTAS ─────────────────────────────────────────────────────
nodes_data = []
for idx, it in enumerate(items):
    cat      = categorize(it)
    yr       = (it.get("date") or "????")[:4]
    authors  = it.get("authors") or []
    first_au = authors[0].split()[-1] if authors else "Anonimo"
    all_au   = "; ".join(authors[:4]) + (" et al." if len(authors) > 4 else "")
    title    = it.get("title") or "Sin titulo"
    doi      = it.get("DOI") or ""
    url      = it.get("url") or ""
    doi_link = (f"https://doi.org/{doi}" if doi else url) or ""
    lpdf     = it.get("local_pdf") or ""
    abstr    = (it.get("abstractNote") or "")[:700].replace('"',"'").replace("\n"," ").replace("\\","\\\\")
    tags     = [t for t in (it.get("tags") or [])
                if not t.startswith("#") and t not in ("nosource","reference","#nosource")]
    colls    = it.get("collections") or []
    pub      = (it.get("publicationTitle") or "").replace('"',"'")
    cid      = comms[idx]
    aporte   = aportes.get(f"{first_au.lower()}|{yr}", "")

    ctags = " ".join(tags).lower()
    if   "thesis-core"  in ctags: size = 24
    elif "foundational" in ctags: size = 20
    elif "review"  in ctags or "systematic" in ctags: size = 16
    elif "pinn"         in ctags: size = 15
    elif "ampacity" in ctags or "iec 60287" in ctags: size = 14
    elif "dynamic thermal rating" in ctags: size = 13
    else: size = 10

    nodes_data.append({
        "id":         idx,
        "label":      f"{first_au}\n{yr}",
        "title_full": title,
        "year":       yr,
        "authors":    all_au,
        "category":   cat,
        "community":  cid,
        "comm_label": comm_infos[cid]["label"],
        "tags":       tags,
        "collections":colls,
        "abstract":   abstr,
        "doi_link":   doi_link,
        "local_pdf":  lpdf,
        "pub":        pub,
        "color":      CATEGORIES[cat]["color"],
        "comm_color": cc(cid),
        "size":       size,
        "aporte":     aporte,
    })

edges_data = [
    {"id": eid, "from": a, "to": b, "score": s,
     "width": max(1.0, round(s * 14, 1))}
    for eid, ((a,b), s) in enumerate(sorted(sim_edges.items()))
]
print(f"Nodos: {len(nodes_data)}, Aristas: {len(edges_data)}")

# ── 8. LEYENDAS ────────────────────────────────────────────────────────────
cat_counts = Counter(n["category"] for n in nodes_data)

cat_leg = ""
for key, val in sorted(CATEGORIES.items(), key=lambda x: x[1]["order"]):
    cnt  = cat_counts.get(key, 0)
    if cnt == 0: continue
    safe = re.sub(r"[^a-z0-9]", "_", key.lower())
    cat_leg += (f'<div class="li" id="lcat_{safe}" onclick="toggleCatF(\'{key}\')">'
                f'<span class="ld" style="background:{val["color"]}"></span>'
                f'<span class="lt">{val["label"]}</span>'
                f'<span class="lc">{cnt}</span></div>')

comm_leg = ""
for cid, info in enumerate(comm_infos):
    if not info["count"]: continue
    col = info["color"]
    comm_leg += (f'<div class="li" id="lcomm_{cid}" onclick="toggleCommF({cid})">'
                 f'<span class="ld" style="background:{col}"></span>'
                 f'<span class="lt" style="font-size:.83em">{info["label"]}</span>'
                 f'<span class="lc">{info["count"]}</span></div>')

# ── 9. JSON ────────────────────────────────────────────────────────────────
nodes_json     = json.dumps(nodes_data, ensure_ascii=False)
edges_json     = json.dumps(edges_data, ensure_ascii=False)
cats_json      = json.dumps(CATEGORIES, ensure_ascii=False)
comm_infos_json= json.dumps(comm_infos, ensure_ascii=False)
palette_json   = json.dumps(CPAL, ensure_ascii=False)

# ── 10. HTML ───────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Red Bibliografica - Tesis MIA: PINNs para Cables Enterrados</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<link  href="https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#fff;color:#333;height:100vh;display:flex;flex-direction:column;overflow:hidden}
/* topbar */
#top{background:linear-gradient(135deg,#1a237e 0%,#283593 60%,#3949ab 100%);color:#fff;padding:9px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;box-shadow:0 2px 10px rgba(0,0,0,.3);z-index:200;flex-shrink:0}
#top h1{font-size:.88em;font-weight:600;flex:1;min-width:180px}
.sb{background:rgba(255,255,255,.18);padding:3px 10px;border-radius:10px;font-size:.71em;white-space:nowrap}
#si{padding:5px 14px;border-radius:18px;border:none;background:rgba(255,255,255,.2);color:#fff;width:200px;font-size:.8em}
#si::placeholder{color:rgba(255,255,255,.55)}
#si:focus{outline:none;background:rgba(255,255,255,.3)}
.tb{padding:5px 11px;border-radius:12px;border:1px solid rgba(255,255,255,.3);background:rgba(255,255,255,.1);color:#fff;cursor:pointer;font-size:.71em;transition:all .2s;white-space:nowrap}
.tb:hover{background:rgba(255,255,255,.25)}
.tb.on{background:rgba(255,255,255,.35);border-color:#fff}
.ts{width:1px;height:18px;background:rgba(255,255,255,.2);flex-shrink:0}
/* main */
#main{display:flex;flex:1;overflow:hidden;min-height:0}
/* left panel */
#lp{width:215px;background:#fafafa;border-right:1px solid #e0e0e0;display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
#ptabs{display:flex;border-bottom:1px solid #e0e0e0;flex-shrink:0}
.pt{flex:1;padding:8px 4px;text-align:center;cursor:pointer;color:#888;background:#f5f5f5;border-bottom:2px solid transparent;font-size:.74em;transition:all .2s}
.pt.on{color:#1a237e;border-bottom-color:#1a237e;background:#fff;font-weight:600}
.ls{padding:8px;overflow-y:auto;flex:1}
.li{display:flex;align-items:center;gap:7px;padding:5px 7px;border-radius:6px;cursor:pointer;transition:background .15s;margin-bottom:2px}
.li:hover{background:#e8eaf6}
.li.on{background:#c5cae9}
.ld{width:11px;height:11px;border-radius:50%;flex-shrink:0}
.lt{flex:1;font-size:.77em;color:#444;line-height:1.3}
.lc{font-size:.71em;color:#aaa;min-width:18px;text-align:right}
/* canvas */
#cw{flex:1;position:relative;background:#f8f9fc;min-width:0}
#network{width:100%;height:100%}
#lcv{position:absolute;top:0;left:0;pointer-events:none;width:100%;height:100%;z-index:10}
/* graph tools */
#gt{position:absolute;top:10px;left:50%;transform:translateX(-50%);display:flex;gap:5px;background:#fff;padding:6px 12px;border-radius:20px;box-shadow:0 2px 12px rgba(0,0,0,.15);z-index:20;flex-wrap:wrap;justify-content:center;max-width:calc(100% - 24px)}
.gb{padding:4px 10px;border-radius:10px;border:1px solid #ddd;background:#fff;color:#555;cursor:pointer;font-size:.7em;transition:all .15s;white-space:nowrap}
.gb:hover{border-color:#3949ab;color:#3949ab;background:#e8eaf6}
.gb.on{background:#e8eaf6;border-color:#3949ab;color:#3949ab;font-weight:600}
.gb.w:hover{border-color:#c62828;color:#c62828;background:#ffebee}
.gs{width:1px;height:16px;background:#ddd;flex-shrink:0;align-self:center}
/* info overlays */
#sei{position:absolute;bottom:10px;left:10px;background:#fff;padding:5px 12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.12);font-size:.72em;color:#555;display:none;z-index:15}
#fb{position:absolute;top:52px;left:50%;transform:translateX(-50%);background:#ff6f00;color:#fff;padding:3px 14px;border-radius:12px;font-size:.7em;display:none;cursor:pointer;z-index:20;box-shadow:0 1px 6px rgba(0,0,0,.2)}
/* right panel */
#rp{width:340px;background:#fff;border-left:1px solid #e0e0e0;display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
#dh{padding:10px 14px;border-bottom:1px solid #f0f0f0;background:#fafafa;flex-shrink:0}
#dh h3{font-size:.74em;color:#888;text-transform:uppercase;letter-spacing:.6px}
#db{flex:1;overflow-y:auto;padding:14px}
.dt{font-size:.91em;font-weight:700;color:#1a1a2e;margin-bottom:8px;line-height:1.4}
.dm{font-size:.76em;color:#666;margin-bottom:3px}
.dm strong{color:#444}
.bk{display:inline-block;padding:2px 10px;border-radius:9px;font-size:.67em;color:#fff;margin-right:4px;margin-bottom:6px;font-weight:500}
.da{font-size:.76em;color:#555;line-height:1.65;margin:8px 0;padding:8px 10px;background:#f9f9f9;border-radius:6px;border-left:3px solid #ddd;max-height:150px;overflow-y:auto}
.dap{font-size:.76em;color:#1a237e;line-height:1.65;margin:8px 0;padding:9px 11px;background:#e8eaf6;border-radius:6px;border-left:3px solid #3949ab}
.dapl{font-size:.68em;font-weight:700;color:#3949ab;text-transform:uppercase;margin-bottom:5px;letter-spacing:.5px}
.tw{display:flex;flex-wrap:wrap;gap:3px;margin:8px 0}
.tg{font-size:.66em;padding:2px 7px;border-radius:7px;background:#f0f0f0;color:#666;border:1px solid #e0e0e0}
.lb{display:inline-block;padding:5px 13px;border-radius:12px;text-decoration:none;font-size:.72em;margin-right:5px;margin-top:6px;transition:all .2s;font-weight:500}
.lb1{background:#1a237e;color:#fff}.lb1:hover{background:#283593}
.lb2{background:#b71c1c;color:#fff}.lb2:hover{background:#c62828}
.nb{margin-top:12px}
.nb summary{font-size:.75em;color:#666;cursor:pointer;padding:4px 0;user-select:none}
.nb ul{list-style:none;margin-top:5px;max-height:150px;overflow-y:auto}
.nb li{font-size:.73em;padding:4px 8px;cursor:pointer;border-radius:5px;color:#3949ab;line-height:1.35}
.nb li:hover{background:#e8eaf6}
.emp{display:flex;flex-direction:column;align-items:center;justify-content:center;height:220px;color:#ccc;text-align:center}
.emp .ico{font-size:2.8em;margin-bottom:10px}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-thumb{background:#ccc;border-radius:3px}
</style>
</head>
<body>
<div id="top">
  <h1>&#x1F52C; Red Bibliografica &#8212; Tesis MIA: PINNs para Calculo Termico de Cables Enterrados</h1>
  <span class="sb" id="slbl"></span>
  <input type="text" id="si" placeholder="Buscar autor, titulo, tag..." oninput="onSearch(this.value)" onkeydown="if(event.key==='Escape'){this.value='';onSearch('');}">
  <div class="ts"></div>
  <button class="tb" onclick="resetAll()">&#x21BA; Reset</button>
  <button class="tb" onclick="fitNet()">Ajustar</button>
  <button class="tb on" id="bphy" onclick="togglePhy()">Fisica ON</button>
  <button class="tb" id="bcm"  onclick="toggleCM()">Color: Categoria</button>
</div>
<div id="main">
  <div id="lp">
    <div id="ptabs">
      <div class="pt on" id="tcat"  onclick="swTab('cat')">Categorias</div>
      <div class="pt"    id="tcomm" onclick="swTab('comm')">Grupos TF-IDF</div>
    </div>
    <div class="ls" id="scat">__CAT_LEG__</div>
    <div class="ls" id="scomm" style="display:none">__COMM_LEG__</div>
  </div>
  <div id="cw">
    <canvas id="lcv"></canvas>
    <div id="gt">
      <button class="gb" onclick="showAll()">Mostrar todo</button>
      <div class="gs"></div>
      <button class="gb" id="blasso" onclick="toggleLasso()">&#x2B1A; Lazo</button>
      <button class="gb" onclick="isolateSel()">Aislar seleccion</button>
      <button class="gb" onclick="expNbrs(1)">Vecinos 1&#x00B0;</button>
      <button class="gb" onclick="expNbrs(2)">Vecinos 2&#x00B0;</button>
      <div class="gs"></div>
      <button class="gb w" onclick="clearSel()">&#x2715; Limpiar</button>
    </div>
    <div id="fb" onclick="showAll()">Filtro activo &#8212; click para limpiar</div>
    <div id="sei"></div>
    <div id="network"></div>
  </div>
  <div id="rp">
    <div id="dh"><h3>&#x1F4C4; Detalle del Paper</h3></div>
    <div id="db">
      <div class="emp">
        <div class="ico">&#x1F4DA;</div>
        <p style="font-size:.81em">Haz clic en un nodo para ver los detalles</p>
        <p style="font-size:.71em;margin-top:8px;color:#ddd">Scroll = zoom | Arrastrar = mover</p>
      </div>
    </div>
  </div>
</div>
<script>
const NODES=__NODES__;
const EDGES=__EDGES__;
const CATS=__CATS__;
const CINFOS=__CINFOS__;
const CPAL=__CPAL__;

let cmode='category', phOn=true, lasso=false;
let lStart=null, lRect=null;

function nc(n,m){
  const f=m==='community'?n.comm_color:n.color;
  const b=m==='community'?n.color:n.comm_color;
  return{background:f,border:b,highlight:{background:lc(f,35),border:'#fff'},hover:{background:lc(f,20),border:'#888'}};
}
const nodesDS=new vis.DataSet(NODES.map(n=>({id:n.id,label:n.label,color:nc(n,cmode),size:n.size,font:{color:'#222',size:10,strokeWidth:2,strokeColor:'#fff'},shape:'dot',borderWidth:2,shadow:{enabled:true,color:'rgba(0,0,0,.12)',size:5},_data:n})));
const edgesDS=new vis.DataSet(EDGES.map(e=>({id:e.id,from:e.from,to:e.to,width:e.width,color:{color:'#c8cce8',highlight:'#3949ab',hover:'#7986cb'},smooth:{type:'continuous',roundness:.2},hoverWidth:3})));
const container=document.getElementById('network');
const net=new vis.Network(container,{nodes:nodesDS,edges:edgesDS},{
  nodes:{borderWidth:2},
  edges:{smooth:{type:'continuous'},hoverWidth:2,selectionWidth:3},
  physics:{enabled:true,forceAtlas2Based:{gravitationalConstant:-65,centralGravity:.006,springLength:130,springConstant:.06,damping:.5,avoidOverlap:.7},solver:'forceAtlas2Based',stabilization:{iterations:280,updateInterval:20}},
  interaction:{hover:true,multiselect:true,tooltipDelay:180,navigationButtons:false,keyboard:{enabled:true,speed:{x:10,y:10,zoom:.03}}}
});
net.on('stabilizationIterationsDone',()=>{net.setOptions({physics:{enabled:false}});phOn=false;document.getElementById('bphy').classList.remove('on');fitNet();});
document.getElementById('slbl').textContent=`${NODES.length} papers | ${EDGES.length} aristas semanticas TF-IDF`;

net.on('click',p=>{if(p.nodes.length>0){const nd=nodesDS.get(p.nodes[0]);showD(nd._data,p.nodes[0]);}else if(!lasso){clearD();}updSel();});
net.on('selectNode',updSel);net.on('deselectNode',updSel);

/* ── LASSO ── */
const lcv=document.getElementById('lcv');
const lctx=lcv.getContext('2d');
function resizeLCV(){lcv.width=container.offsetWidth;lcv.height=container.offsetHeight;}
resizeLCV();new ResizeObserver(resizeLCV).observe(container);
lcv.addEventListener('mousedown',e=>{if(!lasso)return;e.preventDefault();lStart={x:e.offsetX,y:e.offsetY};lRect=null;});
lcv.addEventListener('mousemove',e=>{
  if(!lasso||!lStart)return;
  const x=Math.min(lStart.x,e.offsetX),y=Math.min(lStart.y,e.offsetY);
  const w=Math.abs(e.offsetX-lStart.x),h=Math.abs(e.offsetY-lStart.y);
  lRect={x,y,w,h};
  lctx.clearRect(0,0,lcv.width,lcv.height);
  lctx.strokeStyle='#3949ab';lctx.lineWidth=1.5;lctx.setLineDash([5,3]);
  lctx.strokeRect(x,y,w,h);lctx.fillStyle='rgba(57,73,171,.08)';lctx.fillRect(x,y,w,h);
});
lcv.addEventListener('mouseup',()=>{
  if(!lasso||!lRect){lStart=null;return;}
  const pos=net.getPositions();const sel=[];
  for(const[id,cp]of Object.entries(pos)){
    const dp=net.canvasToDOM(cp);
    if(dp.x>=lRect.x&&dp.x<=lRect.x+lRect.w&&dp.y>=lRect.y&&dp.y<=lRect.y+lRect.h)sel.push(parseInt(id));
  }
  net.selectNodes(sel,true);
  lctx.clearRect(0,0,lcv.width,lcv.height);lRect=null;lStart=null;updSel();
  if(sel.length===1){const nd=nodesDS.get(sel[0]);showD(nd._data,sel[0]);}
});

/* ── FILTERS ── */
function filterBy(type,val){
  document.getElementById('fb').style.display='block';
  nodesDS.update(NODES.map(n=>({id:n.id,hidden:type==='cat'?n.category!==val:n.community!==val})));
  edgesDS.update(EDGES.map(e=>{const nA=NODES[e.from],nB=NODES[e.to];const aM=type==='cat'?nA.category===val:nA.community===val;const bM=type==='cat'?nB.category===val:nB.community===val;return{id:e.id,hidden:!aM&&!bM};}));
  net.fit({animation:true});updStats();
}
function showAll(){
  document.getElementById('fb').style.display='none';
  nodesDS.update(NODES.map(n=>({id:n.id,hidden:false})));
  edgesDS.update(EDGES.map(e=>({id:e.id,hidden:false})));
  document.querySelectorAll('.li').forEach(el=>el.classList.remove('on'));
  updStats();fitNet();
}
function isolateSel(){
  const sel=net.getSelectedNodes();if(!sel.length)return;
  const keep=new Set(sel);
  sel.forEach(nid=>net.getConnectedNodes(nid).forEach(c=>keep.add(c)));
  nodesDS.update(NODES.map(n=>({id:n.id,hidden:!keep.has(n.id)})));
  edgesDS.update(EDGES.map(e=>({id:e.id,hidden:!keep.has(e.from)||!keep.has(e.to)})));
  net.fit({animation:true});updStats();
  const fb=document.getElementById('fb');fb.style.display='block';
  fb.textContent=`${keep.size} nodos aislados - click para limpiar`;
}
function expNbrs(depth){
  let sel=new Set(net.getSelectedNodes());if(!sel.size)return;
  for(let d=0;d<depth;d++){const add=new Set();sel.forEach(nid=>net.getConnectedNodes(nid).forEach(c=>add.add(c)));add.forEach(n=>sel.add(n));}
  net.selectNodes([...sel]);updSel();
}
function clearSel(){net.unselectAll();updSel();}
function onSearch(q){
  q=q.toLowerCase().trim();
  if(!q){showAll();return;}
  nodesDS.update(NODES.map(n=>({id:n.id,hidden:!(n.title_full.toLowerCase().includes(q)||n.authors.toLowerCase().includes(q)||n.tags.some(t=>t.toLowerCase().includes(q))||n.year.includes(q))})));
  edgesDS.update(EDGES.map(e=>({id:e.id,hidden:nodesDS.get(e.from)?.hidden||nodesDS.get(e.to)?.hidden})));
  updStats();
}
function toggleLasso(){lasso=!lasso;document.getElementById('blasso').classList.toggle('on',lasso);lcv.style.pointerEvents=lasso?'auto':'none';lcv.style.cursor=lasso?'crosshair':'default';}
function togglePhy(){phOn=!phOn;net.setOptions({physics:{enabled:phOn}});document.getElementById('bphy').classList.toggle('on',phOn);document.getElementById('bphy').textContent=phOn?'Fisica ON':'Fisica OFF';}
function toggleCM(){cmode=cmode==='category'?'community':'category';document.getElementById('bcm').textContent='Color: '+(cmode==='category'?'Categoria':'Comunidad');nodesDS.update(NODES.map(n=>({id:n.id,color:nc(n,cmode)})));}
let cCatF=null;
function toggleCatF(cat){document.querySelectorAll('[id^="lcat_"]').forEach(el=>el.classList.remove('on'));if(cCatF===cat){cCatF=null;showAll();return;}cCatF=cat;const safe=cat.replace(/[^a-z0-9]/gi,'_').toLowerCase();document.getElementById('lcat_'+safe)?.classList.add('on');filterBy('cat',cat);}
let cCommF=null;
function toggleCommF(cid){document.querySelectorAll('[id^="lcomm_"]').forEach(el=>el.classList.remove('on'));if(cCommF===cid){cCommF=null;showAll();return;}cCommF=cid;document.getElementById('lcomm_'+cid)?.classList.add('on');filterBy('comm',cid);}
function swTab(t){document.getElementById('scat').style.display=t==='cat'?'':'none';document.getElementById('scomm').style.display=t==='comm'?'':'none';document.getElementById('tcat').classList.toggle('on',t==='cat');document.getElementById('tcomm').classList.toggle('on',t==='comm');}
function fitNet(){net.fit({animation:{duration:700,easingFunction:'easeInOutQuad'}});}
function resetAll(){showAll();if(!phOn){togglePhy();setTimeout(togglePhy,3500);}}
function jumpTo(nid){net.focus(nid,{scale:1.4,animation:{duration:600,easingFunction:'easeInOutQuad'}});net.selectNodes([nid]);const nd=nodesDS.get(nid);showD(nd._data,nid);}
function updSel(){const sel=net.getSelectedNodes();const info=document.getElementById('sei');if(sel.length){info.style.display='block';info.textContent=`${sel.length} nodo${sel.length>1?'s':''} seleccionado${sel.length>1?'s':''}`;}else{info.style.display='none';}}
function updStats(){const v=NODES.filter(n=>!nodesDS.get(n.id)?.hidden).length;document.getElementById('slbl').textContent=`${v}/${NODES.length} papers | ${EDGES.length} aristas semanticas TF-IDF`;}

/* ── DETAIL PANEL ── */
function showD(d,nid){
  const ci=CATS[d.category]||{};
  const doi=d.doi_link?`<a href="${d.doi_link}" target="_blank" class="lb lb1">DOI / URL</a>`:'';
  const pdf=d.local_pdf?`<a href="${d.local_pdf}" target="_blank" class="lb lb2">PDF Local</a>`:'';
  const tH=d.tags.filter(Boolean).slice(0,15).map(t=>`<span class="tg">${t}</span>`).join('');
  const apH=d.aporte?`<div class="dap"><div class="dapl">Aporte al proyecto</div>${d.aporte}</div>`:'';
  const cns=net.getConnectedNodes(nid);
  let nbH='';
  if(cns.length){
    const its=cns.slice(0,14).map(nid2=>{const cn=nodesDS.get(nid2)?._data;if(!cn)return'';const au=cn.authors.split(';')[0].trim().split(' ').pop();return`<li onclick="jumpTo(${nid2})" title="${cn.title_full}">${au} (${cn.year}) - ${cn.title_full.substring(0,50)}${cn.title_full.length>50?'...':''}</li>`;}).join('');
    nbH=`<details class="nb"><summary>${cns.length} papers conectados semanticamente</summary><ul>${its}</ul></details>`;
  }
  document.getElementById('db').innerHTML=`
    <div class="dt">${d.title_full}</div>
    <div class="dm"><strong>Autores:</strong> ${d.authors||'Desconocido'}</div>
    <div class="dm"><strong>Anio:</strong> ${d.year}${d.pub?' &middot; <em>'+d.pub+'</em>':''}</div>
    <div style="margin:8px 0"><span class="bk" style="background:${ci.color||'#999'}">${ci.label||d.category}</span><span class="bk" style="background:${d.comm_color}">${d.comm_label}</span></div>
    ${d.abstract?`<div class="da">${d.abstract}${d.abstract.length>=700?'...':''}</div>`:'<div style="font-size:.74em;color:#aaa;font-style:italic;margin:6px 0">Sin abstract disponible</div>'}
    ${apH}
    <div class="tw">${tH}</div>
    <div>${doi}${pdf}</div>
    ${nbH}
    <div style="font-size:.68em;color:#bbb;margin-top:10px">Colecciones: ${d.collections.join(', ')||'ninguna'}</div>`;
}
function clearD(){document.getElementById('db').innerHTML=`<div class="emp"><div class="ico">&#x1F4DA;</div><p style="font-size:.81em">Haz clic en un nodo para ver los detalles</p><p style="font-size:.71em;margin-top:8px;color:#ddd">Scroll = zoom | Arrastrar = mover</p></div>`;}

/* ── UTILS ── */
function lc(hex,a){const n=parseInt(hex.replace('#',''),16);const r=Math.min(255,(n>>16)+a);const g=Math.min(255,((n>>8)&0xff)+a);const b=Math.min(255,(n&0xff)+a);return'#'+[r,g,b].map(v=>v.toString(16).padStart(2,'0')).join('');}
</script>
</body>
</html>
"""

# ── 11. INYECTAR Y GUARDAR ─────────────────────────────────────────────────
html = (HTML
        .replace("__NODES__",    nodes_json)
        .replace("__EDGES__",    edges_json)
        .replace("__CATS__",     cats_json)
        .replace("__CINFOS__",   comm_infos_json)
        .replace("__CPAL__",     palette_json)
        .replace("__CAT_LEG__",  cat_leg)
        .replace("__COMM_LEG__", comm_leg))

out = Path(OUTPUT_FILE)
out.write_text(html, encoding="utf-8")
size_kb = out.stat().st_size // 1024
print(f"\n  {out}  ({len(nodes_data)} nodos, {len(edges_data)} aristas, {size_kb} KB)")
cat_c = {c: sum(1 for n in nodes_data if n['category']==c)
         for c in CATEGORIES if sum(1 for n in nodes_data if n['category']==c)>0}
print(f"  Categorias: {cat_c}")
print(f"  PDFs locales: {sum(1 for n in nodes_data if n['local_pdf'])}/{len(nodes_data)}")
print(f"  Aportes:      {sum(1 for n in nodes_data if n['aporte'])}/{len(nodes_data)}")
