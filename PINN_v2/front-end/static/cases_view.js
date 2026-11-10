'use strict';
// Pestana Casos (6/6, parte A): editor JSON y dibujo de la geometria.

function syncJson() {
  const box = $('#jsonText');
  if (box && document.activeElement !== box) box.value = JSON.stringify(S.doc, null, 2);
}

function renderJson() {
  const msg = h('div', { class: 'small', id: 'jsonMsg' }, '');
  const box = h('textarea', { id: 'jsonText', class: 'mono', spellcheck: false, rows: 32, readOnly: S.readonly, oninput: debounce((e) => {
    try {
      S.doc = JSON.parse(e.target.value);
      msg.textContent = 'JSON correcto';
      msg.className = 'small ok-text';
      S.dirty = true;
      detectAuto();
      drawPreview();
      updateHead();
      validateSoon();
    } catch (err) { msg.textContent = 'JSON no valido: ' + err.message; msg.className = 'small bad-text'; }
  }, 350) }, '');
  box.value = JSON.stringify(S.doc, null, 2);
  mount($('#jsonBox'), h('div', { class: 'muted small' }, 'Editor directo del archivo. Los cambios se validan al escribir; usa el formulario para volver a la vista guiada.'), box, msg);
}

const NS = 'http://www.w3.org/2000/svg';
function sv(tag, attrs, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const k of Object.keys(attrs || {})) if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
  for (const kid of kids) if (kid !== null && kid !== undefined) e.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
  return e;
}

function geoWindow(p, view) {
  const b = p.bounds || (p.kind === 'annulus' ? [-p.ro, p.ro, -p.ro, p.ro] : null);
  if (!b) return [0, 1, 0, 1];
  if (view !== 'zoom' || !(p.cables || []).length) return b.slice();
  const r = p.radius || 0.02;
  const xs = p.cables.map((c) => c[0]);
  const ys = p.cables.map((c) => c[1]);
  const pad = Math.max(3 * r, 0.02);
  return [Math.max(b[0], Math.min(...xs) - r - pad), Math.min(b[1], Math.max(...xs) + r + pad), Math.max(b[2], Math.min(...ys) - r - pad), Math.min(b[3], Math.max(...ys) + r + pad)];
}

const LAYER_FILL = ['#f59e0b', '#93c5fd', '#a7f3d0', '#fca5a5', '#ddd6fe', '#fde68a'];
const BC_COLOR = { dirichlet: '#2563eb', neumann: '#64748b', robin: '#ea580c' };

function geoSvg(doc, view, collocation) {
  const p = doc.physical;
  const w = geoWindow(p, view);
  const sx = (w[1] - w[0]) || 1;
  const sy = (w[3] - w[2]) || 1;
  const k = Math.min(420 / sx, 520 / sy);
  const X = (x) => (x - w[0]) * k + 6;
  const Y = (y) => (w[3] - y) * k + 6;
  const svg = sv('svg', { class: 'geo', viewBox: `0 0 ${sx * k + 12} ${sy * k + 22}` });
  const b = p.bounds || (p.kind === 'annulus' ? [-p.ro, p.ro, -p.ro, p.ro] : null);
  svg.appendChild(sv('rect', { x: X(b[0]), y: Y(b[3]), width: (b[1] - b[0]) * k, height: (b[3] - b[2]) * k, fill: '#f8f1e4', stroke: '#334155', 'stroke-width': 1 }));
  if (p.patch) {
    const [cx, cy, pw, ph] = p.patch;
    svg.appendChild(sv('rect', { x: X(cx - pw / 2), y: Y(cy + ph / 2), width: pw * k, height: ph * k, fill: 'rgba(14,165,233,.15)', stroke: '#0ea5e9', 'stroke-dasharray': '4 3' }));
  }
  for (const bd of p.bands || []) {
    if (bd[0] < w[2] || bd[0] > w[3]) continue;
    svg.appendChild(sv('line', { x1: X(b[0]), x2: X(b[1]), y1: Y(bd[0]), y2: Y(bd[0]), stroke: '#a16207', 'stroke-dasharray': '6 4' }));
    svg.appendChild(sv('text', { x: X(b[0]) + 4, y: Y(bd[0]) - 3, 'font-size': 9, fill: '#a16207' }, 'k=' + bd[1] + ' / ' + bd[2]));
  }
  const layers = p.layers || [];
  for (const c of p.cables || []) {
    if (c[0] < w[0] - 1 || c[0] > w[1] + 1) continue;
    for (let j = layers.length - 1; j >= 0; j--) {
      const rr = Math.max(layers[j][1] * k, j === layers.length - 1 ? 2.5 : 0.3);
      svg.appendChild(sv('circle', { cx: X(c[0]), cy: Y(c[1]), r: rr, fill: LAYER_FILL[j % LAYER_FILL.length], stroke: '#475569', 'stroke-width': 0.4 }));
    }
  }
  if (p.kind === 'annulus') {
    for (const side of ['outer', 'inner']) {
      const condition = doc.boundary_conditions && doc.boundary_conditions[side];
      svg.appendChild(sv('circle', { cx: X(0), cy: Y(0), r: p[side === 'outer' ? 'ro' : 'ri'] * k, fill: 'none',
        stroke: BC_COLOR[condition && condition.type] || '#334155', 'stroke-width': 2 }));
    }
  }
  const bcs = doc.boundary_conditions;
  const edge = (side, x1, y1, x2, y2) => {
    const t = bcs && bcs[side] ? bcs[side].type : 'dirichlet';
    svg.appendChild(sv('line', { x1, y1, x2, y2, stroke: BC_COLOR[t] || '#2563eb', 'stroke-width': 3, opacity: 0.8 }, sv('title', {}, side + ': ' + t)));
  };
  if (!p.outer_radius && p.kind !== 'annulus') {
    edge('top', X(b[0]), Y(b[3]), X(b[1]), Y(b[3]));
    edge('bottom', X(b[0]), Y(b[2]), X(b[1]), Y(b[2]));
    edge('left', X(b[0]), Y(b[3]), X(b[0]), Y(b[2]));
    edge('right', X(b[1]), Y(b[3]), X(b[1]), Y(b[2]));
  }
  if (collocation) {
    const groups = [
      ...(collocation.interior || []).map((item) => [item.xy, '#0284c7', 1.15]),
      ...(collocation.interfaces || []).map((points) => [points, '#e11d48', 1.45]),
      ...(collocation.boundary || []).map((points) => [points, '#ca8a04', 1.45])
    ];
    for (const [points, color, radius] of groups) for (const point of points) {
      if (point[0] < w[0] || point[0] > w[1] || point[1] < w[2] || point[1] > w[3]) continue;
      svg.appendChild(sv('circle', { cx: X(point[0]), cy: Y(point[1]), r: radius, fill: color,
        stroke: '#ffffff', 'stroke-width': 0.45, opacity: 0.78 }));
    }
  }
  svg.appendChild(sv('text', { x: 8, y: sy * k + 18, 'font-size': 9, fill: '#475569' }, `x ${fmt(w[0], 3)} .. ${fmt(w[1], 3)} m   y ${fmt(w[2], 3)} .. ${fmt(w[3], 3)} m`));
  return svg;
}

function drawPreview() {
  const box = $('#geoBox');
  if (!box || !S.doc) return;
  if (!S.doc.physical || (S.doc.physical.kind !== 'cable' && S.doc.physical.kind !== 'annulus' && !S.doc.physical.kind.startsWith('mms'))) { mount(box, h('div', { class: 'muted small' }, 'Sin vista previa para este tipo de caso.')); return; }
  let svg;
  try { svg = geoSvg(S.doc, CV.view); } catch (e) { mount(box, h('div', { class: 'muted small' }, 'No se puede dibujar todavia: ' + e.message)); return; }
  const legend = h('div', { class: 'row small muted', style: 'gap:10px' }, Object.entries(BC_COLOR).map(([t, c]) => h('span', {}, h('span', { style: 'display:inline-block;width:14px;height:4px;background:' + c + ';margin-right:4px;vertical-align:middle' }), t)));
  const cable = S.doc.physical.kind === 'cable';
  mount(box, h('div', { class: 'row', style: 'justify-content:space-between' }, h('h3', {}, cable ? 'Disposicion geometrica' : 'Dominio de verificacion 2D'),
    cable ? h('div', { class: 'seg' }, [['full', 'Dominio'], ['zoom', 'Cables']].map(([v, l]) => h('button', { class: CV.view === v ? 'on' : '', onclick: () => { CV.view = v; drawPreview(); } }, l))) : null), svg, legend);
}