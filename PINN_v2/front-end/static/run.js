'use strict';
// Pestana Ejecucion (1/2): que caso, que metodo y con que parametros.

const R = { src: 'current', doc: null, check: null, kind: 'pinn', arch: 'M1', preset: 'rapido', p: null, levels: [0, 1], threads: 2, liveEvery: 4, geoView: 'full',
  view: 'zoom', cmap: 'inferno', iso: false, overlay: true, jobId: null, job: null, variant: null, timer: null, busy: false, pointMultiplier: 1 };

const POINT_MULTIPLIERS = [0.5, 1, 1.5, 2, 3, 4];

function scaledPointCounts(points, multiplier) {
  return Object.fromEntries(Object.entries(points).map(([key, count]) => [key, Math.max(1, Math.round(Number(count) * multiplier))]));
}

function presetParams(name) {
  const o = S.options.presets[name];
  return { adam_steps: o.adam_steps, lbfgs_steps: o.lbfgs_steps, lr: o.lr, width: o.width, depth: o.depth, activation: 'tanh', attention: '', init_seed: 0, sampling_seed: 11,
    log_every: 20, weights: clone(S.options.weights), points: clone(o.points) };
}

function runCaseOptions() {
  const c = S.cases || { catalog: [], user: [] };
  const cable = (x) => x.kind === 'cable';
  const verification = (x) => !cable(x);
  const opt = (x, pre) => h('option', { value: x.ref, selected: R.src === x.ref }, (pre || '') + x.name + (x.title ? '  -  ' + String(x.title).slice(0, 40) : ''));
  const groups = [];
  if (S.doc && S.doc.physical) groups.push(h('optgroup', { label: 'En el editor' }, h('option', { value: 'current', selected: R.src === 'current' }, 'Caso abierto: ' + S.doc.id + (S.dirty ? ' (con cambios sin guardar)' : ''))));
  for (const [label, filter] of [['Mis casos de cable', cable], ['Mis casos de verificacion 2D', verification]]) {
    const mine = c.user.filter(filter);
    if (mine.length) groups.push(h('optgroup', { label }, mine.map((x) => opt(x, (x.folder ? x.folder + '/' : '')))));
  }
  const catalog = Object.create(null);
  for (const item of c.catalog.filter((x) => x.group === 'v2')) {
    const prefix = cable(item) ? 'Catalogo v2' : 'Verificacion 2D';
    const group = prefix + (item.folder ? ' / ' + item.folder : '');
    (catalog[group] = catalog[group] || []).push(item);
  }
  for (const group of Object.keys(catalog).sort()) groups.push(h('optgroup', { label: group }, catalog[group].map((x) => opt(x))));
  return groups;
}
async function loadRunCase() {
  R.check = null;
  try {
    if (R.src === 'current') {
      if (!S.doc) { R.src = ''; R.doc = null; drawRunLeft(); drawRunGeo(); return; }
      R.doc = S.doc;
    } else if (R.src) {
      R.doc = (await api('/api/case?' + qs({ ref: R.src }))).doc;
    }
    R.check = R.doc ? await post('/api/validate', { doc: R.doc }) : null;
  } catch (e) { toast(e.message, 'bad'); }
  if (R.check && R.kind === 'pinn' && !R.check.pinn.supported && R.check.fem.supported) R.kind = 'fem';
  drawRunLeft();
  drawRunGeo();
}

function enterRun() {
  if (!S.options) return;
  if (R.src === 'current' && !S.doc) {
    const first = S.cases && S.cases.catalog.find((x) => x.name === 'xlpe_single');
    R.src = first ? first.ref : '';
  }
  if (!R.p) R.p = presetParams(R.preset);
  loadRunCase();
  pollNow();
}

function rnum(label, obj, key, step, hint) {
  return h('label', { class: 'f' }, h('span', {}, label, hint ? h('span', { class: 'hint' }, ' ' + hint) : null),
    h('input', { type: 'number', step: step || 'any', value: obj[key], onchange: (e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) obj[key] = v; R.preset = 'custom'; } }));
}

function archCard(a) {
  const fem = a.id === 'FEM';
  const check = R.check;
  const off = check && check.ok && (fem ? !check.fem.supported : !check.pinn.supported);
  const on = fem ? R.kind === 'fem' : R.kind === 'pinn' && R.arch === a.id;
  const why = off ? (fem ? check.fem.reason : check.pinn.reason) : '';
  return h('div', { class: 'arch' + (fem ? ' fem' : '') + (on ? ' on' : '') + (off ? ' off' : ''), title: off ? why : a.note,
    onclick: () => { if (off) { toast(why || 'No disponible para este caso', 'bad'); return; } R.kind = fem ? 'fem' : 'pinn'; if (!fem) R.arch = a.id; drawRunLeft(); } },
    h('b', {}, a.id), h('span', {}, a.title.replace(/^[A-Z0-9]+ - /, '')));
}
function pinnParams() {
  const o = S.options;
  const P = R.p;
  const chips = Object.keys(o.presets).map((k) => h('span', { class: 'chip' + (R.preset === k ? ' on' : ''), onclick: () => { R.preset = k; R.p = presetParams(k); drawRunLeft(); } }, o.presets[k].title));
  const w = o.weights;
  return [h('div', { class: 'chips' }, chips, R.preset === 'custom' ? h('span', { class: 'chip on' }, 'Personalizado') : null),
    h('div', { class: 'point-scale' },
      h('label', { class: 'f' }, h('span', {}, 'Multiplicador de puntos'), h('select', { value: String(R.pointMultiplier), onchange: (e) => { R.pointMultiplier = Number(e.target.value); drawRunLeft(); } },
        POINT_MULTIPLIERS.map((value) => h('option', { value: String(value), selected: R.pointMultiplier === value }, value + 'x')))),
      h('span', { class: 'sub' }, 'Escala los cuatro conteos al iniciar.')),
    h('div', { class: 'grid2', style: 'margin-top:8px' }, rnum('Pasos Adam', P, 'adam_steps', 100), rnum('Pasos L-BFGS', P, 'lbfgs_steps', 10, P.attention ? '(0 con atencion)' : ''),
      rnum('Ancho de capa', P, 'width', 4), rnum('Capas ocultas', P, 'depth', 1), rnum('Tasa de aprendizaje', P, 'lr', 0.0005)),
    h('details', { class: 'sec', open: !!R.advOpen, ontoggle: (e) => { R.advOpen = e.target.open; } }, h('summary', {}, 'Opciones avanzadas'),
      h('div', { class: 'in' },
        h('div', { class: 'grid2' },
          h('label', { class: 'f' }, h('span', {}, 'Activacion'), h('select', { onchange: (e) => { P.activation = e.target.value; R.preset = 'custom'; } }, o.activations.map((a) => h('option', { value: a, selected: P.activation === a }, a)))),
          h('label', { class: 'f' }, h('span', {}, 'Atencion'), h('select', { onchange: (e) => { P.attention = e.target.value; if (P.attention) P.lbfgs_steps = 0; drawRunLeft(); } }, o.attention.map((a) => h('option', { value: a.id, selected: P.attention === a.id }, a.title)))),
          rnum('Semilla de la red', P, 'init_seed', 1), rnum('Semilla del muestreo', P, 'sampling_seed', 1), rnum('Registrar cada N pasos', P, 'log_every', 5)),
        h('h4', {}, 'Pesos de la perdida'), h('div', { class: 'grid2' }, Object.keys(w).map((k) => rnum(k, P.weights, k, 'any'))),
        h('h4', {}, 'Puntos de muestreo'), h('div', { class: 'grid2' }, Object.keys(P.points).map((k) => rnum(k, P.points, k, 8))),
        h('p', { class: 'sub' }, 'Los pesos y los puntos son solo hiperparametros del entrenamiento; ninguna referencia (FEM o articulo) entra como dato de entrenamiento.')))];
}

function femParams() {
  const chips = [0, 1, 2, 3].map((l) => h('span', { class: 'chip fem' + (R.levels.includes(l) ? ' on' : ''), onclick: () => {
    R.levels = R.levels.includes(l) ? R.levels.filter((x) => x !== l) : R.levels.concat([l]).sort();
    if (!R.levels.length) R.levels = [l];
    drawRunLeft();
  } }, 'Nivel ' + l));
  return [h('p', { class: 'sub' }, 'FEniCSx (WSL2): elementos P2 sobre una malla de triangulos. Cada nivel refina mas la malla; con dos niveles se ve como converge la solucion.'), h('div', { class: 'chips' }, chips),
    h('p', { class: 'sub' }, 'Tiempo orientativo: niveles 0-1 en segundos; nivel 3 puede tardar varios minutos.')];
}
function drawRunLeft() {
  const box = $('#runLeft');
  if (!box || !S.options) return;
  if (!R.p) R.p = presetParams(R.preset);
  const c = R.check;
  const archs = S.options.architectures.map(archCard).concat([archCard({ id: 'FEM', title: 'FEM - FEniCSx', note: 'Metodo de elementos finitos: la referencia numerica.' })]);
  const femOk = S.info && S.info.fem && S.info.fem.available;
  const ready = c && c.ok && (R.kind === 'fem' ? c.fem.supported && femOk : c.pinn.supported);
  let why = '';
  if (!R.doc) why = 'Elige un caso.'; else if (c && !c.ok) why = 'El caso no valida: corrigelo en la pestana Casos.'; else if (R.kind === 'fem' && !femOk) why = 'FEniCSx no esta disponible en este equipo.';
  mount(box,
    h('h2', {}, 'Que quieres correr'),
    h('div', { class: 'run-actions' },
      h('div', { class: 'row' },
        h('button', { class: 'btn run-start', disabled: !ready || R.busy, onclick: startRun }, R.kind === 'fem' ? 'Ejecutar FEM' : 'Entrenar ' + R.arch),
        h('button', { class: 'btn sec run-reset', onclick: () => { R.preset = R.preset === 'custom' ? 'rapido' : R.preset; R.p = presetParams(R.preset); R.pointMultiplier = 1; drawRunLeft(); } }, 'Restablecer')),
      why ? h('p', { class: 'sub run-why' }, why) : null),
    h('label', { class: 'f' }, h('span', {}, '1. Caso'), h('select', { onchange: (e) => { R.src = e.target.value; loadRunCase(); } }, h('option', { value: '', selected: !R.src }, '- elige -'), runCaseOptions())),
    c ? h('div', { class: 'row', style: 'margin:6px 0' }, c.ok ? badge('valido', 'ok') : badge('no valido', 'bad'), c.pinn.supported ? badge('PINN', 'pinn') : null, c.fem.supported ? badge('FEM', 'fem') : null,
      c.summary && c.summary.verification ? h('span', { class: 'muted small' }, c.summary.verification + (c.summary.regions ? ' · ' + c.summary.regions + ' regiones' : '')) : null,
      c.summary && c.summary.regions && !c.summary.verification ? h('span', { class: 'muted small' }, c.summary.regions + ' regiones, ' + c.summary.cables + ' cable(s), P = ' + fmt(c.summary.power_W_m, 1) + ' W/m') : null) : null,
    c && !c.ok ? banner('bad', c.errors.join(' ')) : null,
    c && c.ok && !c.pinn.supported ? banner('warn', c.pinn.reason) : null,
    h('label', { class: 'f', style: 'margin-top:10px' }, h('span', {}, '2. Metodo')),
    h('div', { class: 'archs' }, archs),
    h('div', { style: 'margin-top:10px' }, h('label', { class: 'f' }, h('span', {}, '3. Parametros')), R.kind === 'fem' ? femParams() : pinnParams()),
    h('div', { class: 'grid2', style: 'margin-top:8px' }, rnum('Hilos de CPU', R, 'threads', 1, '(1-8)'), rnum('Foto en vivo cada (s)', R, 'liveEvery', 1)));
}

function drawRunGeo() {
  const box = $('#runGeo');
  if (!box) return;
  const kind = R.doc && R.doc.physical && R.doc.physical.kind;
  if (!kind || (kind !== 'cable' && kind !== 'annulus' && !kind.startsWith('mms'))) { mount(box, h('div', { class: 'muted small' }, 'Este tipo de caso no tiene vista geometrica.')); return; }
  let svg;
  try { svg = geoSvg(R.doc, R.geoView); } catch (e) { mount(box, h('div', { class: 'muted small' }, 'No se puede dibujar: ' + e.message)); return; }
  mount(box, h('div', { class: 'row', style: 'justify-content:space-between' }, h('h3', { style: 'margin:0' }, kind === 'cable' ? 'Geometria del caso' : 'Dominio de verificacion 2D'),
    kind === 'cable' ? h('div', { class: 'seg' }, [['full', 'Dominio'], ['zoom', 'Cables']].map(([v, l]) => h('button', { class: R.geoView === v ? 'on' : '', onclick: () => { R.geoView = v; drawRunGeo(); } }, l))) : null), svg);
}

async function startRun() {
  if (R.busy) return;
  R.busy = true;
  drawRunLeft();
  try {
    const params = R.kind === 'fem' ? { levels: R.levels.slice() } : Object.assign({ architecture: R.arch }, clone(R.p));
    if (R.kind === 'pinn') params.points = scaledPointCounts(params.points, R.pointMultiplier);
    const body = { kind: R.kind, threads: R.threads, live_every_s: R.liveEvery, params };
    if (R.src === 'current') body.doc = R.doc; else body.ref = R.src;
    const r = await post('/api/run', body);
    R.jobId = r.id; R.job = null; R.variant = null;
    toast('Trabajo en marcha', 'ok');
    pollNow();
  } catch (e) { toast(e.message, 'bad'); }
  R.busy = false;
  drawRunLeft();
}

function initRun() {
  mount($('#tab-ejecucion'), h('div', { class: 'run' },
    h('div', { class: 'card pad params', id: 'runLeft' }),
    h('div', { style: 'display:grid;gap:14px;min-width:0' },
      h('div', { class: 'card pad', id: 'runGeo' }),
      h('div', { class: 'card pad', id: 'runMon' }, h('div', { class: 'empty' }, 'Aqui veras el avance y el campo de temperatura mientras se calcula.')),
      h('div', { class: 'card pad jobs', id: 'runJobs' }))));
  S.onTab.ejecucion = enterRun;
}