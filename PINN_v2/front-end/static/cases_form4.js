'use strict';
// Pestana Casos (5/6): contacto, procedencia, inconsistencias, claves extra y el formulario completo.

function provIds() { return (S.doc.provenance || []).map((x) => x.id); }

function ensureContactStatus() {
  const d = S.doc;
  d.data_status = d.data_status || {};
  if ((d.contact || []).length && !d.data_status.contact) d.data_status.contact = { status: 'supuesto', source: provIds()[0] || 'autor', note: 'Resistencia de contacto definida por el usuario' };
}

function secContact() {
  const d = S.doc;
  const items = d.contact || [];
  const rows = items.map((c, i) => h('tr', {},
    h('td', {}, txt(['contact', i, 'interface'], { list: 'dl-ifaces', placeholder: 'cable_0_interface_1' })),
    h('td', {}, num(['contact', i, 'R', 'value'], { min: 0 })),
    h('td', {}, sel(['contact', i, 'status'], STATUSES, { structural: false })),
    h('td', {}, sel(['contact', i, 'source'], provIds(), { structural: false })),
    h('td', {}, iconBtn('x', 'Quitar', () => { d.contact = d.contact.filter((_, n) => n !== i); if (!d.contact.length) d.data_status = Object.fromEntries(Object.entries(d.data_status).filter(([k]) => k !== 'contact')); changed(true); }, 'bad'))));
  return section('Contacto imperfecto (opcional)', 'contact', [
    banner('warn', 'Con resistencia de contacto el caso solo lo resuelve el FEM; el PINN lo rechaza en esta version.'),
    items.length ? h('table', { class: 't' }, h('thead', {}, h('tr', {}, ['Interfaz', 'R (m2 K/W)', 'Estado', 'Fuente', ''].map((x) => h('th', {}, x)))), h('tbody', {}, rows)) : h('div', { class: 'muted small' }, 'Sin contactos: todas las interfaces son perfectas (continuidad de T y de flujo).'),
    h('datalist', { id: 'dl-ifaces' }, ((S.check && S.check.summary && S.check.summary.interface_names) || []).map((n) => h('option', { value: n }))),
    iconBtn('+ Contacto', 'Agrega una resistencia de contacto', () => {
      d.contact = (d.contact || []).concat([{ interface: ((S.check && S.check.summary && S.check.summary.interface_names) || [])[0] || '', R: { value: 1e-4, unit: 'm2 K/W' }, status: 'supuesto', source: provIds()[0] || 'autor' }]);
      ensureContactStatus();
      changed(true);
    })], items.length ? items.length + ' definido(s)' : 'ninguno', false);
}

function jsonField(label, value, onparse, rows) {
  const box = h('textarea', { rows: rows || 3, spellcheck: false, class: 'mono' }, '');
  box.value = typeof value === 'string' ? value : JSON.stringify(value);
  const msg = h('span', { class: 'hint' }, '');
  box.addEventListener('change', () => {
    try { onparse(JSON.parse(box.value)); msg.textContent = ''; } catch (e) { msg.textContent = ' JSON no valido'; }
  });
  return h('label', { class: 'f' }, h('span', {}, label, msg), box);
}

function secProv() {
  const d = S.doc;
  const prov = d.provenance || [];
  const stat = d.data_status || {};
  const groups = ['geometry', 'materials', 'source', 'boundary_conditions'].concat((d.contact || []).length ? ['contact'] : []);
  const provRows = prov.map((x, i) => h('div', { class: 'grid2' },
    field('Id de la fuente', h('input', { type: 'text', value: x.id, onchange: (e) => { const old = x.id; x.id = e.target.value; for (const g of Object.values(d.data_status || {})) if (g.source === old) g.source = x.id; changed(true); } })),
    h('div', { class: 'row', style: 'align-items:flex-end' }, field('Cita', txt(['provenance', i, 'citation'])), iconBtn('x', 'Quitar fuente', () => { d.provenance = d.provenance.filter((_, n) => n !== i); changed(true); }, 'bad'))));
  const statRows = groups.map((g) => {
    if (!stat[g]) stat[g] = { status: 'supuesto', source: provIds()[0] || 'autor', note: 'Completar' };
    d.data_status = stat;
    return h('div', { class: 'card pad' }, h('b', {}, g), h('div', { class: 'grid3' },
      field('Estado', sel(['data_status', g, 'status'], STATUSES, { structural: false }), STATUS_HELP[stat[g].status] || ''),
      field('Fuente', sel(['data_status', g, 'source'], provIds(), { structural: false })),
      field('Nota', txt(['data_status', g, 'note']), stat[g].status === 'publicado' ? '(opcional)' : '(obligatoria)')));
  });
  return section('Procedencia de los datos', 'prov', [
    h('div', { class: 'muted small' }, 'Cada grupo de datos declara si es publicado, extraido, inferido o supuesto, y de que fuente sale. No se inventan valores: lo desconocido se marca como supuesto.'),
    h('h4', {}, 'Fuentes'), provRows,
    iconBtn('+ Fuente', 'Agrega una fuente', () => { d.provenance = prov.concat([{ id: 'fuente_' + (prov.length + 1), citation: '' }]); changed(true); }),
    h('h4', {}, 'Estado de cada grupo'), statRows], prov.length + ' fuente(s)', false);
}

function secIncons() {
  const d = S.doc;
  const items = d.inconsistencies || [];
  const rows = items.map((x, i) => h('div', { class: 'card pad' }, h('div', { class: 'row' }, h('b', {}, 'Inconsistencia ' + (i + 1)), iconBtn('x', 'Quitar', () => { d.inconsistencies = d.inconsistencies.filter((_, n) => n !== i); changed(true); }, 'bad')),
    h('div', { class: 'grid3' }, field('Campo', txt(['inconsistencies', i, 'field'])), jsonField('Valores en conflicto (JSON)', x.values === undefined ? [] : x.values, (v) => { x.values = v; changed(false); }, 2), field('Resolucion adoptada', txt(['inconsistencies', i, 'resolution'])))));
  return section('Inconsistencias conocidas', 'incons', [rows, iconBtn('+ Inconsistencia', 'Registra datos que no concuerdan entre fuentes', () => { d.inconsistencies = items.concat([{ field: '', values: [], resolution: '' }]); changed(true); })], items.length ? items.length + ' registrada(s)' : 'ninguna', false);
}

function secExtra() {
  const p = S.doc.physical;
  const extra = Object.fromEntries(Object.entries(p).filter(([k]) => !HANDLED.has(k)));
  return section('Otros campos fisicos (JSON)', 'extra', [
    h('div', { class: 'muted small' }, 'Claves que el formulario no maneja (por ejemplo electrical, heat_capacity_J_m3K, adaptation). Se conservan tal cual.'),
    jsonField('Campos extra', Object.keys(extra).length ? JSON.stringify(extra, null, 1) : '{}', (obj) => {
      if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) throw new Error('objeto esperado');
      const keep = Object.fromEntries(Object.entries(S.doc.physical).filter(([k]) => HANDLED.has(k)));
      S.doc.physical = Object.assign(keep, obj);
      changed(false);
    }, 6)], Object.keys(extra).length ? Object.keys(extra).length + ' campo(s)' : 'ninguno', false);
}

const FORM_STEPS = [
  { title: 'Dominio', note: 'Identificacion del caso, dimensiones del dominio y propiedades base del suelo.', sections: () => [secGeneral(), secDomain()] },
  { title: 'Cables y capas', note: 'Ubicacion de los cables y propiedades de las capas concentricas.', sections: () => [secCables(), secLayers()] },
  { title: 'Fisica y bordes', note: 'Generacion Joule, condiciones de borde y resistencias de contacto.', sections: () => [secSource(), secBC(), secContact()] },
  { title: 'Revision', note: 'Procedencia, inconsistencias conocidas y campos adicionales del modelo.', sections: () => [secProv(), secIncons(), secExtra()] }
];

function formStepNav() {
  const step = CV.formStep || 0;
  return h('div', { class: 'form-step-nav', 'aria-label': 'Pasos de entrada de datos' },
    h('div', { class: 'form-step-tabs' }, FORM_STEPS.map((item, i) => h('button', {
      type: 'button', class: 'form-step-tab' + (step === i ? ' on' : ''), 'aria-current': step === i ? 'step' : null,
      onclick: () => { CV.formStep = i; renderForm(); }
    }, h('span', { class: 'form-step-number' }, String(i + 1)), item.title))),
    h('div', { class: 'form-step-controls' },
      h('span', { class: 'muted small' }, 'Paso ' + (step + 1) + ' de ' + FORM_STEPS.length),
      h('button', { class: 'btn sec', type: 'button', disabled: step === 0, onclick: () => { CV.formStep--; renderForm(); } }, 'Anterior'),
      h('button', { class: 'btn', type: 'button', disabled: step === FORM_STEPS.length - 1, onclick: () => { CV.formStep++; renderForm(); } }, 'Siguiente')));
}

function renderForm() {
  const box = $('#formBox');
  if (!box || !S.doc) return;
  const kind = S.doc.physical && S.doc.physical.kind;
  if (kind !== 'cable') {
    mount(box, banner('info', 'Este caso es de tipo "' + kind + '" (verificacion). El formulario cubre casos de cable; edita el JSON directamente.'));
    return;
  }
  const step = CV.formStep || 0;
  const current = FORM_STEPS[step] || FORM_STEPS[0];
  mount(box, formStepNav(), h('div', { class: 'form-step-heading' }, h('h3', {}, current.title), h('p', { class: 'sub' }, current.note)),
    h('fieldset', { class: 'form', disabled: S.readonly }, current.sections()));
}