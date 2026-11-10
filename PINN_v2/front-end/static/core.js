'use strict';
/* Utilidades comunes: constructor de nodos, llamadas a la API, avisos, dialogos. */
const S = { info: null, options: null, materials: null, cases: null, doc: null, ref: null, readonly: false, notice: null, dirty: false, check: null,
  open: {}, jobs: [], trash: [], showTrash: false, selJob: null, onTab: {} };

const PROPS = new Set(['value', 'checked', 'disabled', 'readOnly', 'type', 'placeholder', 'title', 'id', 'href', 'src', 'textContent', 'selected', 'min', 'max', 'step', 'rows', 'multiple', 'open', 'target', 'name']);

function add(parent, kids) {
  for (const k of kids) {
    if (k === null || k === undefined || k === false) continue;
    if (Array.isArray(k)) add(parent, k);
    else parent.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const key of Object.keys(attrs)) {
      const v = attrs[key];
      if (v === null || v === undefined || v === false) continue;
      if (key === 'class') e.className = v;
      else if (key === 'html') e.innerHTML = v;
      else if (key.startsWith('on')) e.addEventListener(key.slice(2), v);
      else if (PROPS.has(key)) e[key] = v;
      else e.setAttribute(key, v === true ? '' : v);
    }
  }
  add(e, kids);
  return e;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }
function mount(node, ...kids) { clear(node); add(node, kids); return node; }
const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  if (!res.ok) {
    const err = new Error((data && data.error) || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return data;
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
const qs = (obj) => Object.keys(obj).filter((k) => obj[k] !== undefined && obj[k] !== null && obj[k] !== '').map((k) => encodeURIComponent(k) + '=' + encodeURIComponent(obj[k])).join('&');

function toast(msg, kind) {
  const box = h('div', { class: kind || '' }, msg);
  $('#toast').appendChild(box);
  setTimeout(() => box.remove(), kind === 'bad' ? 7000 : 3500);
}

function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

function fmt(v, d) {
  if (v === null || v === undefined || Number.isNaN(v)) return '-';
  if (typeof v !== 'number') return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(d === undefined ? 3 : d);
  return v.toFixed(d === undefined ? 3 : d).replace(/\.?0+$/, (m) => (m.includes('.') ? '' : m));
}
function secs(s) {
  if (s === null || s === undefined) return '-';
  s = Math.round(s);
  const m = Math.floor(s / 60);
  return m ? `${m} min ${s % 60} s` : `${s} s`;
}
const clone = (o) => JSON.parse(JSON.stringify(o));

function download(name, text, mime) {
  const url = URL.createObjectURL(new Blob([text], { type: mime || 'text/plain' }));
  const a = h('a', { href: url });
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/* Dialogo sencillo con campos; devuelve una promesa con los valores o null si se cancela. */
function ask(title, fields, okText, note) {
  const dlg = $('#dlg');
  return new Promise((resolve) => {
    const inputs = {};
    const rows = fields.map((f) => {
      let inp;
      if (f.options) inp = h('select', {}, f.options.map((o) => h('option', { value: o.value === undefined ? o : o.value, selected: (o.value === undefined ? o : o.value) === f.value }, o.label || o)));
      else if (f.type === 'checkbox') inp = h('input', { type: 'checkbox', checked: !!f.value });
      else inp = h('input', { type: f.type || 'text', value: f.value === undefined ? '' : f.value, placeholder: f.placeholder || '', list: f.list || null });
      inputs[f.key] = inp;
      return f.type === 'checkbox' ? h('label', { class: 'chk' }, inp, f.label) : h('label', { class: 'f' }, h('span', {}, f.label, f.hint ? h('span', { class: 'hint' }, ' ' + f.hint) : null), inp);
    });
    let done = false;
    const finish = (v) => { if (done) return; done = true; dlg.close(); resolve(v); };
    mount(dlg, h('h2', {}, title), note ? h('p', { class: 'muted small' }, note) : null, h('div', { style: 'display:grid;gap:10px;margin:10px 0' }, rows),
      h('div', { class: 'row', style: 'justify-content:flex-end' },
        h('button', { class: 'btn ghost', onclick: () => finish(null) }, 'Cancelar'),
        h('button', { class: 'btn', onclick: () => {
          const out = {};
          for (const f of fields) out[f.key] = f.type === 'checkbox' ? inputs[f.key].checked : inputs[f.key].value;
          finish(out);
        } }, okText || 'Aceptar')));
    dlg.addEventListener('close', () => finish(null), { once: true });
    dlg.showModal();
    const first = Object.values(inputs)[0];
    if (first) first.focus();
  });
}

function confirmBox(title, text, okText) {
  const dlg = $('#dlg');
  return new Promise((resolve) => {
    let done = false;
    const finish = (v) => { if (done) return; done = true; dlg.close(); resolve(v); };
    mount(dlg, h('h2', {}, title), h('p', {}, text),
      h('div', { class: 'row', style: 'justify-content:flex-end' }, h('button', { class: 'btn ghost', onclick: () => finish(false) }, 'Cancelar'), h('button', { class: 'btn', onclick: () => finish(true) }, okText || 'Aceptar')));
    dlg.addEventListener('close', () => finish(false), { once: true });
    dlg.showModal();
  });
}

function tip(text) { return h('span', { class: 'tip', title: text }, '?'); }
function badge(text, kind) { return h('span', { class: 'badge ' + (kind || '') }, text); }
function banner(kind, ...kids) { return h('div', { class: 'banner ' + kind }, kids); }

/* Acceso por ruta a un documento: getp(doc, ['physical','cables',0,1]). */
function getp(o, path) { let cur = o; for (const k of path) { if (cur === null || cur === undefined) return undefined; cur = cur[k]; } return cur; }
function setp(o, path, v) {
  let cur = o;
  for (let i = 0; i < path.length - 1; i++) {
    if (cur[path[i]] === null || cur[path[i]] === undefined) cur[path[i]] = typeof path[i + 1] === 'number' ? [] : {};
    cur = cur[path[i]];
  }
  cur[path[path.length - 1]] = v;
}

/* Salto entre pestanas desde cualquier lugar (p. ej. "Comparar" desde Ejecucion). */
function goTab(name) {
  $$('#tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === name));
  $$('section.tab').forEach((s) => s.classList.toggle('on', s.id === 'tab-' + name));
  if (S.onTab[name]) S.onTab[name]();
  location.hash = name;
}