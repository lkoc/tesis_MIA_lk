'use strict';
// Pestana Casos (6/6, parte B): resumen, cabecera y montaje de la pestana.

function drawSummary() {
  const box = $('#sumBox');
  if (!box) return;
  const c = S.check;
  if (!c) { mount(box, h('div', { class: 'muted small' }, 'Validando...')); return; }
  const s = c.summary || {};
  const kv = [];
  const row = (k, v) => { if (v !== undefined && v !== null) kv.push(h('b', {}, k), h('span', {}, String(v))); };
  row('Id', s.id); row('Regiones', s.regions); row('Cables', s.cables); row('Interfaces', s.interfaces);
  row('Potencia p\'', s.power_W_m !== undefined && s.power_W_m !== null ? fmt(s.power_W_m, 4) + ' W/m' : undefined);
  row('Q en el conductor', s.Q_W_m3 !== undefined && s.Q_W_m3 !== null ? fmt(s.Q_W_m3, 4) + ' W/m3' : undefined);
  row('Ancho x profundidad', s.width_m !== undefined ? fmt(s.width_m, 3) + ' x ' + fmt(s.depth_m, 3) + ' m' : undefined);
  row('Huella (sha256)', s.sha256);
  mount(box,
    h('h3', {}, 'Validacion'),
    c.ok ? banner('ok', 'El caso es valido: se puede resolver.') : banner('bad', h('b', {}, 'No valida: '), h('ul', { style: 'margin:4px 0 0 16px;padding:0' }, (c.errors || []).map((e) => h('li', {}, e)))),
    (c.warnings || []).map((w) => banner('warn', w)),
    c.ok ? h('div', { class: 'row', style: 'margin:6px 0' }, badge('PINN: ' + (c.pinn.supported ? 'si' : 'no'), c.pinn.supported ? 'pinn' : 'bad'), badge('FEM: ' + (c.fem.supported ? 'si' : 'no'), c.fem.supported ? 'fem' : 'bad'),
      c.pinn.reason ? h('span', { class: 'muted small' }, c.pinn.reason) : null) : null,
    kv.length ? h('div', { class: 'kv' }, kv) : null);
}

function updateHead() {
  const box = $('#caseHead');
  if (!box || !S.doc) return;
  const c = S.check;
  const where = S.readonly ? badge('catalogo (solo lectura)', 'warn') : S.ref ? badge(S.ref.replace('user:', 'mis casos / '), 'ok') : badge('sin guardar', 'warn');
  mount(box,
    h('div', {}, h('h2', { style: 'margin:0' }, S.doc.title || S.doc.id || 'Caso'), h('div', { class: 'row', style: 'gap:6px;margin-top:2px' }, where, S.dirty ? badge('con cambios', 'warn') : null,
      c ? badge(c.ok ? 'valido' : 'no valido', c.ok ? 'ok' : 'bad') : null)),
    h('span', { class: 'spacer' }),
    S.readonly ? h('button', { class: 'btn', onclick: () => saveFlow(true) }, 'Duplicar a Mis casos') : h('button', { class: 'btn', onclick: () => saveFlow(false) }, 'Guardar'),
    S.readonly ? null : h('button', { class: 'btn sec', title: 'Guarda una copia con otro nombre e id', onclick: () => saveFlow(true) }, 'Guardar como...'),
    h('button', { class: 'btn sec', onclick: () => download((S.doc.id || 'caso') + '.json', JSON.stringify(S.doc, null, 2) + '\n', 'application/json') }, 'Descargar JSON'),
    h('button', { class: 'btn', style: 'background:#16a34a', disabled: !(c && c.ok), title: 'Abre la pestana Ejecucion con este caso', onclick: () => { R.src = 'current'; goTab('ejecucion'); } }, 'Ejecutar este caso'));
}

function setMode(json) {
  CV.jsonMode = json;
  $('#formBox').style.display = json ? 'none' : '';
  $('#jsonBox').style.display = json ? '' : 'none';
  $$('#modeSeg button').forEach((b) => b.classList.toggle('on', (b.dataset.m === 'json') === json));
  if (json) renderJson(); else renderForm();
}

function renderAll() {
  if (!S.doc) return;
  const notCable = !(S.doc.physical && S.doc.physical.kind === 'cable');
  if (notCable) CV.jsonMode = true;
  mount($('#caseWork'),
    h('div', { class: 'head', id: 'caseHead' }),
    h('div', { class: 'body' },
      h('div', { class: 'form' },
        S.notice ? banner('info', S.notice) : null,
        h('div', { class: 'row', style: 'margin-bottom:8px' }, h('div', { class: 'seg', id: 'modeSeg' },
          h('button', { 'data-m': 'form', onclick: () => setMode(false), disabled: notCable }, 'Formulario'),
          h('button', { 'data-m': 'json', onclick: () => setMode(true) }, 'JSON')),
          h('span', { class: 'muted small' }, 'Los dos modos editan el mismo caso.')),
        h('div', { id: 'formBox' }), h('div', { id: 'jsonBox' })),
      h('div', { class: 'prev' }, h('div', { id: 'geoBox' }), h('div', { id: 'sumBox', style: 'margin-top:14px' }))));
  setMode(CV.jsonMode);
  drawPreview();
  drawSummary();
  updateHead();
  drawTree();
}