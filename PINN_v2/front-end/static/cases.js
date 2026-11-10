'use strict';
// Pestana Casos (1/3): estado, arbol de casos, abrir, guardar, validar.

const STATUSES = ['publicado', 'extraido', 'inferido', 'supuesto'];
const CV = { jsonMode: false, view: 'full', filter: '', powerAuto: true };

async function loadCases() {
  const r = await api('/api/cases');
  S.cases = r.items;
  drawTree();
}

function detectAuto() {
  const p = S.doc.physical;
  const calc = (p.current || 0) ** 2 * (p.R20 || 0);
  CV.powerAuto = p.power === undefined || p.power === null || Math.abs(p.power - calc) <= 1e-6 * Math.max(Math.abs(p.power), 1e-12);
}

async function openCase(ref, force) {
  if (S.dirty && !force && !(await confirmBox('Cambios sin guardar', 'Hay cambios sin guardar en el caso abierto. Si continuas se pierden.', 'Descartar y abrir'))) return;
  try {
    const r = await api('/api/case?' + qs({ ref }));
    S.doc = r.doc; S.ref = ref; S.readonly = r.readonly; S.notice = r.notice; S.dirty = false; S.check = null; CV.formStep = 0;
    detectAuto();
    renderAll();
    validateNow();
  } catch (e) { toast(e.message, 'bad'); }
}

async function newCase(kind) {
  if (S.dirty && !(await confirmBox('Cambios sin guardar', 'Hay cambios sin guardar. Si continuas se pierden.', 'Descartar y crear'))) return;
  const r = await api('/api/template?' + qs({ kind }));
  S.doc = r.doc || r; S.ref = null; S.readonly = false; S.dirty = true; S.check = null; CV.formStep = 0;
  S.notice = (kind === 'single' ? 'Caso nuevo desde la plantilla de cable unico XLPE. ' : 'Caso nuevo desde cero. ') + 'Aun no esta guardado: usa Guardar para elegir carpeta y nombre.';
  detectAuto();
  renderAll();
  validateNow();
}

function changed(structural) {
  S.dirty = true;
  if (structural) renderForm();
  syncJson();
  drawPreview();
  updateHead();
  validateSoon();
}

async function validateNow() {
  if (!S.doc) return;
  try {
    S.check = await post('/api/validate', { doc: S.doc });
  } catch (e) { S.check = { ok: false, errors: [e.message], warnings: [], summary: {}, pinn: {}, fem: {} }; }
  updateHead();
  drawSummary();
  const dl = $('#dl-ifaces');
  if (dl) mount(dl, ((S.check.summary || {}).interface_names || []).map((n) => h('option', { value: n })));
  if (S.onChecked) S.onChecked();
}
const validateSoon = debounce(validateNow, 450);

function drawTree() {
  const box = $('#caseTree');
  if (!box || !S.cases) return;
  const f = CV.filter.toLowerCase();
  const match = (c) => !f || (c.name + ' ' + (c.id || '') + ' ' + (c.title || '')).toLowerCase().includes(f);
  const node = (c) => h('div', { class: 'node' + (S.ref === c.ref ? ' on' : ''), title: c.title || c.id || '', onclick: () => openCase(c.ref) }, c.broken ? '! ' : '', c.name, h('span', { class: 'kind' }, c.kind));
  const v2 = S.cases.catalog.filter((c) => c.group === 'v2' && match(c));
  const catalogFolders = Object.create(null);
  for (const c of v2) (catalogFolders[c.folder || ''] = catalogFolders[c.folder || ''] || []).push(c);
  const catalogTree = Object.keys(catalogFolders).sort().map((folder) => h('div', {}, folder ? h('div', { class: 'fold' }, folder) : null, catalogFolders[folder].map(node)));
  const byFolder = {};
  for (const fo of S.cases.folders) byFolder[fo] = [];
  for (const c of S.cases.user.filter(match)) (byFolder[c.folder || ''] = byFolder[c.folder || ''] || []).push(c);
  const mine = Object.keys(byFolder).sort().map((fo) => h('div', {}, h('div', { class: 'fold' }, fo ? '\u{1F4C1} ' + fo : '(sin carpeta)'),
    byFolder[fo].length ? byFolder[fo].map(node) : h('div', { class: 'muted small', style: 'padding:2px 14px' }, 'vacia')));
  mount(box,
    h('h4', {}, 'Mis casos (editables)'), mine.length ? mine : h('div', { class: 'muted small', style: 'padding:4px 8px' }, 'Aun no hay casos tuyos. Crea uno nuevo o duplica uno del catalogo.'),
    h('h4', {}, 'Catalogo v2 (solo lectura)'), catalogTree);
}

function drawSide() {
  mount($('#caseSide'),
    h('div', { class: 'pad', style: 'padding-bottom:6px' },
      h('h2', {}, 'Casos'),
      h('div', { class: 'row', style: 'gap:6px;margin-bottom:8px' },
        h('button', { class: 'btn', onclick: () => newCase('single') }, '+ Desde plantilla'),
        h('button', { class: 'btn sec', onclick: () => newCase('empty') }, '+ Vacio'),
        h('button', { class: 'btn sec', onclick: newFolder }, 'Carpeta'),
        h('button', { class: 'btn sec', onclick: () => $('#importFile').click() }, 'Importar')),
      h('input', { type: 'file', id: 'importFile', accept: '.json', style: 'display:none', onchange: importFile }),
      h('input', { type: 'search', placeholder: 'Buscar caso...', oninput: (e) => { CV.filter = e.target.value; drawTree(); } })),
    h('div', { class: 'tree', id: 'caseTree' }));
  drawTree();
}

async function newFolder() {
  const r = await ask('Nueva carpeta', [{ key: 'folder', label: 'Nombre de la carpeta', hint: '(letras, numeros, - y _; se puede anidar con /)', placeholder: 'mis_estudios' }], 'Crear',
    'Los casos tuyos viven siempre bajo front-end/user_cases/, separados del catalogo.');
  if (!r || !r.folder) return;
  try { await post('/api/mkdir', { folder: r.folder }); toast('Carpeta creada', 'ok'); loadCases(); } catch (e) { toast(e.message, 'bad'); }
}

async function importFile(ev) {
  const file = ev.target.files[0];
  ev.target.value = '';
  if (!file) return;
  try {
    const doc = JSON.parse(await file.text());
    if (!doc.physical) throw new Error('No es un caso v2 (falta "physical"). Los casos heredados planos se abren desde el catalogo.');
    S.doc = doc; S.ref = null; S.readonly = false; S.dirty = true; CV.formStep = 0;
    S.notice = 'Importado de ' + file.name + '. Aun no esta guardado.';
    detectAuto();
    renderAll();
    validateNow();
  } catch (e) { toast('No se pudo importar: ' + e.message, 'bad'); }
}

async function saveFlow(duplicate) {
  const folders = (S.cases && S.cases.folders) || [];
  const mineRef = S.ref && S.ref.startsWith('user:');
  const stem = mineRef ? S.ref.split('/').pop() : S.doc.id;
  const fields = [
    { key: 'folder', label: 'Carpeta', hint: folders.length ? '(existentes: ' + folders.join(', ') + ')' : '(se crea si no existe)', value: mineRef ? S.ref.slice(5).split('/').slice(0, -1).join('/') : (folders[0] || 'mis_casos'), list: 'dl-folders' },
    { key: 'name', label: 'Nombre del archivo', hint: '(sin .json)', value: duplicate ? S.doc.id + '_copia' : stem }];
  if (duplicate) fields.push({ key: 'newid', label: 'Id del caso nuevo', hint: '(distinto, para no mezclar soluciones)', value: S.doc.id + '_copia' });
  fields.push({ key: 'draft', label: 'Guardar aunque no valide (borrador)', type: 'checkbox', value: false });
  const dl = h('datalist', { id: 'dl-folders' }, folders.map((f) => h('option', { value: f })));
  document.body.appendChild(dl);
  const r = await ask(duplicate ? 'Duplicar a Mis casos' : 'Guardar caso', fields, 'Guardar', 'Siempre se guarda en front-end/user_cases/<carpeta>/, nunca en el catalogo.');
  dl.remove();
  if (!r) return;
  const doc = clone(S.doc);
  if (duplicate && r.newid) { doc.id = r.newid; doc.physical.id = r.newid; }
  const send = (overwrite) => post('/api/save', { folder: r.folder, name: r.name, doc, overwrite, allow_invalid: r.draft });
  try {
    let res;
    try { res = await send(false); } catch (e) {
      if (e.status !== 409) throw e;
      if (!(await confirmBox('Ya existe', 'Ya hay un caso ' + r.folder + '/' + r.name + '. Sobrescribirlo?', 'Sobrescribir'))) return;
      res = await send(true);
    }
    S.doc = doc; S.ref = res.ref; S.readonly = false; S.dirty = false; S.notice = null;
    toast(res.valid ? 'Guardado en ' + res.ref.slice(5) : 'Guardado como borrador (aun no valida)', res.valid ? 'ok' : '');
    await loadCases();
    renderAll();
    validateNow();
  } catch (e) { toast(e.message, 'bad'); }
}

function initCases() {
  mount($('#tab-casos'), h('div', { class: 'cases' }, h('div', { class: 'card side', id: 'caseSide' }), h('div', { class: 'card work', id: 'caseWork' }, h('div', { class: 'empty' }, 'Elige un caso del arbol o crea uno nuevo.'))));
  drawSide();
}