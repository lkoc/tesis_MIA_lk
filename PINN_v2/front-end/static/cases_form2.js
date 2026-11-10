'use strict';
// Pestana Casos (3/6): cables, capas y fuente de calor.

function matGroup(id) { return (S.materials.groups.find((g) => g.id === id) || { items: [] }).items; }

function syncRadius() {
  const p = S.doc.physical;
  if (!p.layers || !p.layers.length) return;
  p.layers[0][0] = 0;
  for (let i = 1; i < p.layers.length; i++) p.layers[i][0] = p.layers[i - 1][1];
  p.radius = p.layers[p.layers.length - 1][1];
}

function secCables() {
  const p = S.doc.physical;
  const cables = p.cables || [];
  const r = p.radius || 0.02;
  const b = p.bounds;
  const place = (kind) => {
    const cx = (b[0] + b[1]) / 2;
    const cy = b[3] - Math.min(1, 0.4 * (b[3] - b[2]));
    const s = 2.2 * r;
    if (kind === 'trefoil') p.cables = [[cx - s / 2, cy], [cx + s / 2, cy], [cx, cy - s * 0.866]];
    else if (kind === 'flat3') p.cables = [[cx - 2 * r * 1.1, cy], [cx, cy], [cx + 2 * r * 1.1, cy]];
    else p.cables = [[cx, cy]];
    changed(true);
  };
  const rows = cables.map((c, i) => h('tr', {}, h('td', {}, 'Cable ' + i), h('td', {}, num(['physical', 'cables', i, 0])), h('td', {}, num(['physical', 'cables', i, 1])),
    h('td', {}, iconBtn('x', 'Quitar este cable', () => { p.cables = p.cables.filter((_, n) => n !== i); changed(true); }, 'bad'))));
  return section('Cables (posicion)', 'cables', [
    h('div', { class: 'row' }, h('span', { class: 'muted small' }, 'Disposicion rapida:'),
      iconBtn('1 cable', 'Un cable centrado', () => place('one')), iconBtn('Terna en trebol', 'Tres cables en trebol con separacion de 2.2 radios', () => place('trefoil')), iconBtn('Terna plana', 'Tres cables en linea', () => place('flat3')),
      iconBtn('+ Cable', 'Agrega un cable bajo el ultimo', () => { const l = cables[cables.length - 1] || [0, b[3] - 1]; p.cables = cables.concat([[l[0] + 3 * r, l[1]]]); changed(true); })),
    cables.length ? h('table', { class: 't' }, h('thead', {}, h('tr', {}, ['', 'x (m)', 'y (m, negativo = bajo la superficie)', ''].map((x) => h('th', {}, x)))), h('tbody', {}, rows))
      : h('div', { class: 'muted small' }, 'Sin cables: no hay fuente de calor.'),
    h('div', { class: 'muted small' }, 'Radio exterior comun: ' + fmt(r, 5) + ' m (sale de la ultima capa).')], cables.length + ' cable(s)', true);
}

function matChips(group, onpick) {
  return h('div', { class: 'row' }, matGroup(group).map((m) => h('button', { class: 'chip', type: 'button', title: (m.note || '') + ' k=' + m.k, onclick: () => onpick(m) }, m.name + ' (' + m.k + ')')));
}

function secLayers() {
  const p = S.doc.physical;
  const layers = p.layers || [];
  const kinds = ['conductor', 'insulation', 'metal', 'other'];
  const rows = layers.map((l, i) => {
    const pick = h('select', { title: 'Cargar k de un material tipico', onchange: (e) => {
      const [g, n] = e.target.value.split('|');
      const m = matGroup(g).find((x) => x.name === n);
      if (m) { p.layers[i][2] = m.k; changed(true); }
    } }, h('option', { value: '' }, 'material...'), kinds.map((g) => h('optgroup', { label: g }, matGroup(g).map((m) => h('option', { value: g + '|' + m.name }, m.name + ' (' + m.k + ')')))));
    return h('tr', {}, h('td', {}, i === 0 ? 'conductor' : 'capa ' + i),
      h('td', {}, h('input', { type: 'number', value: fmt(l[0], 6), readOnly: true, title: 'Igual al radio exterior de la capa anterior' })),
      h('td', {}, num(['physical', 'layers', i, 1], { structural: true, after: syncRadius })),
      h('td', {}, num(['physical', 'layers', i, 2])), h('td', {}, pick),
      h('td', {}, layers.length > 1 ? iconBtn('x', 'Quitar capa', () => { p.layers = p.layers.filter((_, n) => n !== i); syncRadius(); changed(true); }, 'bad') : null));
  });
  return section('Capas del cable (cada capa con su propio k)', 'layers', [
    h('div', { class: 'muted small' }, 'Se resuelve -div(k grad T) = Q en cada capa; el contacto entre capas es perfecto salvo que se defina una resistencia en Contacto.'),
    h('table', { class: 't' }, h('thead', {}, h('tr', {}, ['', 'r interior (m)', 'r exterior (m)', 'k (W/m K)', 'material', ''].map((x) => h('th', {}, x)))), h('tbody', {}, rows)),
    h('div', { class: 'row' }, iconBtn('+ Capa exterior', 'Agrega una capa sobre la ultima', () => {
      const last = layers[layers.length - 1];
      p.layers = layers.concat([[last[1], last[1] + 0.002, 0.286]]);
      syncRadius();
      changed(true);
    }))], layers.length + ' region(es) por cable', true);
}