'use strict';
// Pestana Casos (2/6): piezas del formulario y las primeras secciones.

const HANDLED = new Set(['id', 'kind', 'bounds', 'T0', 'scale', 'source', 'k', 'power', 'cables', 'radius', 'layers', 'current', 'R20', 'alpha', 'patch', 'bands', 'band_smoothing']);
const STATUS_HELP = { publicado: 'Dato tomado tal cual de la publicacion', extraido: 'Leido de una figura o tabla', inferido: 'Deducido de otros datos', supuesto: 'Supuesto del autor, sin respaldo' };
const SIDES = { bottom: 'Borde inferior', top: 'Superficie (borde superior)', left: 'Borde izquierdo', right: 'Borde derecho', outer: 'Borde exterior (circular)' };
function field(label, input, hint) {
  return h('label', { class: 'f' }, h('span', {}, label, hint ? h('span', { class: 'hint' }, ' ' + hint) : null), input);
}
function sel(path, options, o) {
  o = o || {};
  const cur = getp(S.doc, path);
  return h('select', { onchange: (e) => {
    setp(S.doc, path, e.target.value);
    if (o.after) o.after(e.target.value);
    changed(o.structural !== false);
  } }, options.map((x) => h('option', { value: x.value === undefined ? x : x.value, selected: (x.value === undefined ? x : x.value) === cur }, x.label || x)));
}
function iconBtn(text, title, fn, cls) {
  return h('button', { class: 'mini ' + (cls || ''), title, type: 'button', onclick: fn }, text);
}
function num(path, o) {
  o = o || {};
  const cur = getp(S.doc, path);
  return h('input', { type: 'number', step: o.step || 'any', min: o.min, value: cur === undefined || cur === null ? '' : cur, onchange: (e) => {
    const v = e.target.value === '' ? null : parseFloat(e.target.value);
    setp(S.doc, path, v);
    if (o.after) o.after(v);
    changed(!!o.structural);
  } });
}
function txt(path, o) {
  o = o || {};
  const cur = getp(S.doc, path);
  const make = o.area ? 'textarea' : 'input';
  return h(make, { type: 'text', rows: o.area ? 2 : null, value: cur === undefined || cur === null ? '' : cur, placeholder: o.placeholder || '', list: o.list || null, onchange: (e) => {
    setp(S.doc, path, e.target.value);
    if (o.after) o.after(e.target.value);
    changed(!!o.structural);
  } });
}
function section(title, key, body, hint, openByDefault) {
  if (S.open[key] === undefined) S.open[key] = !!openByDefault;
  return h('details', { class: 'sec', open: S.open[key], ontoggle: (e) => { S.open[key] = e.target.open; } }, h('summary', {}, title, hint ? h('span', { class: 'hint' }, ' ' + hint) : null), h('div', { class: 'body' }, body));
}
function secGeneral() {
  const d = S.doc;
  return section('General', 'general', [
    h('div', { class: 'grid2' },
      field('Id del caso', h('input', { type: 'text', value: d.id || '', onchange: (e) => { d.id = e.target.value; d.physical.id = e.target.value; changed(false); } }), '(unico; nombra las soluciones)'),
      field('Titulo', txt(['title']))),
    h('div', { class: 'grid2' },
      field('Fuente / referencia fisica', txt(['physical', 'source']), '(texto libre)'),
      field('Tipo de caso', h('input', { type: 'text', value: d.physical.kind || '', readOnly: true }), '(solo cable se edita aqui)'))], 'Identidad del caso', true);
}
function secDomain() {
  const p = S.doc.physical;
  const labels = ['x minimo', 'x maximo', 'y minimo (fondo)', 'y maximo (superficie)'];
  const soils = (S.materials.groups.find((g) => g.id === 'soil') || { items: [] }).items;
  return section('Dominio y suelo', 'domain', [
    h('div', { class: 'grid4' }, labels.map((l, i) => field(l + ' (m)', num(['physical', 'bounds', i])))),
    h('div', { class: 'grid3' },
      field('Temperatura de referencia T0 (degC)', num(['physical', 'T0']), '(inicial / contorno por defecto)'),
      field('Escala de temperatura', num(['physical', 'scale']), '(K, normaliza la red)'),
      field('k del suelo (W/m K)', num(['physical', 'k']), '(base, bajo el parche y las bandas)')),
    h('div', { class: 'row' }, h('span', { class: 'muted small' }, 'Tipos de suelo tipicos:'), soils.map((m) =>
      h('button', { class: 'chip', type: 'button', title: m.note || '', onclick: () => { p.k = m.k; changed(true); } }, m.name + ' (' + m.k + ')'))),
    patternBox()], 'Tamano y propiedades base', true);
}
function patchBox() {
  const p = S.doc.physical;
  const bands = p.bands || [];
  const box = p.patch
    ? h('div', { class: 'card pad' }, h('div', { class: 'row' }, h('b', {}, 'Parche de conductividad'), iconBtn('Quitar', 'Quita el parche', () => { p.patch = null; changed(true); }, 'bad')),
        h('div', { class: 'grid3' }, ['centro x (m)', 'centro y (m)', 'ancho (m)', 'alto (m)', 'k dentro (W/m K)', 'suavizado eps (m)'].map((l, i) => field(l, num(['physical', 'patch', i])))))
    : h('button', { class: 'btn sec', type: 'button', onclick: () => { const b = p.bounds; p.patch = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[1] - b[0]) / 3, (b[3] - b[2]) / 3, 2 * p.k, 0.01]; changed(true); } }, '+ Parche rectangular de otro k');
  return box;
}
function bandsBox() {
  const p = S.doc.physical;
  const bands = p.bands || [];
  const bandBox = h('div', { class: 'card pad' }, h('div', { class: 'row' }, h('b', {}, 'Capas horizontales de suelo (bandas)'),
      iconBtn('+ Agregar', 'Agrega una interfaz horizontal', () => { const b = p.bounds; p.bands = (p.bands || []).concat([[(b[2] + b[3]) / 2, p.k, p.k]]); changed(true); })),
    bands.length ? h('table', { class: 't' }, h('thead', {}, h('tr', {}, ['Altura y (m)', 'k por encima', 'k por debajo', ''].map((x) => h('th', {}, x)))),
      h('tbody', {}, bands.map((row, i) => h('tr', {}, [0, 1, 2].map((j) => h('td', {}, num(['physical', 'bands', i, j]))), h('td', {}, iconBtn('x', 'Quitar', () => { p.bands = p.bands.filter((_, n) => n !== i); changed(true); }, 'bad'))))))
      : h('div', { class: 'muted small' }, 'Sin bandas: el suelo es homogeneo (salvo el parche).'),
    bands.length ? field('Suavizado de las bandas (m)', num(['physical', 'band_smoothing']), '(0 = interfaz exacta)') : null);
  return bandBox;
}
function patternBox() {
  return h('div', { style: 'display:grid;gap:10px;margin-top:10px' }, patchBox(), bandsBox());
}
