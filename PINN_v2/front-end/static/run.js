'use strict';
// Pestana Ejecucion (1/2): que caso, que metodo y con que parametros.

const R = { src: 'current', doc: null, check: null, kind: 'pinn', arch: 'M1', preset: 'rapido', p: null, levels: [0, 1], femPreset: 'medio', nodeMultiplier: 1, threads: 2, liveEvery: 4, geoView: 'full',
  view: 'zoom', cmap: 'inferno', iso: false, overlay: true, showCollocation: false, collocationCache: null, collocationRequest: 0, geoRequest: 0,
  jobId: null, job: null, variant: null, timer: null, busy: false, pointMultiplier: 1, deepxdeModel: 'PFNN', deepxdeSampling: 'mixed', settingsKey: null, continueFrom: false, resumeJobId: null };

const POINT_MULTIPLIERS = [0.5, 1, 1.5, 2, 3, 4];
const FEM_REFINEMENT_PRESETS = { bajo: [0], medio: [0, 1], alto: [0, 1, 2] };
const FEM_NODE_MULTIPLIERS = [1, 1.5, 2, 3];

function femPresetForLevels(levels) {
  return Object.keys(FEM_REFINEMENT_PRESETS).find((name) => JSON.stringify(FEM_REFINEMENT_PRESETS[name]) === JSON.stringify(levels)) || 'manual';
}

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
  if (R.src && R.src.startsWith('job:') && R.doc) groups.push(h('optgroup', { label: 'Desde una corrida previa' },
    h('option', { value: R.src, selected: true }, R.doc.id + ' (caso guardado con la corrida)')));
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

function runSettingsKey() {
  if (R.src === 'current') return S.ref || (R.doc ? 'draft:' + R.doc.id : null);
  return R.src || null;
}

function applyPreviousRunParameters(job) {
  const params = clone(job.params || {});
  R.kind = 'pinn';
  if (S.options.architectures.some((item) => item.id === params.architecture)) R.arch = params.architecture;
  R.deepxdeModel = params.deepxde_model || 'PFNN';
  R.deepxdeSampling = params.deepxde_sampling || 'mixed';
  R.preset = 'custom';
  const base = presetParams('rapido');
  R.p = Object.assign(base, params);
  delete R.p.architecture;
  R.p.weights = Object.assign(base.weights, params.weights || {});
  R.p.points = Object.assign(base.points, params.points || {});
  R.pointMultiplier = 1;
  R.threads = Number.isFinite(job.threads) ? job.threads : 2;
  R.liveEvery = Number.isFinite(job.live_every_s) ? job.live_every_s : 4;
  R.resumeJobId = job.id;
  R.continueFrom = false;
}

function resetRunSettings() {
  R.kind = 'pinn'; R.arch = 'M1'; R.preset = 'rapido'; R.p = presetParams('rapido');
  R.levels = [0, 1]; R.femPreset = 'medio'; R.nodeMultiplier = 1; R.threads = 2; R.liveEvery = 4; R.pointMultiplier = 1; R.deepxdeModel = 'PFNN'; R.deepxdeSampling = 'mixed'; R.collocationCache = null;
}

function applyRunSettings(settings) {
  if (settings.kind === 'pinn' || settings.kind === 'fem') R.kind = settings.kind;
  if (settings.arch && S.options.architectures.some((item) => item.id === settings.arch)) R.arch = settings.arch;
  if (settings.preset && (settings.preset === 'custom' || S.options.presets[settings.preset])) R.preset = settings.preset;
  if (settings.p && typeof settings.p === 'object') {
    const base = presetParams(R.preset === 'custom' ? 'rapido' : R.preset);
    R.p = Object.assign(base, clone(settings.p));
    R.p.weights = Object.assign(base.weights, settings.p.weights || {});
    R.p.points = Object.assign(base.points, settings.p.points || {});
  }
  if (Array.isArray(settings.levels)) R.levels = settings.levels.filter((level) => [0, 1, 2, 3].includes(level));
  if (!R.levels.length) R.levels = [0, 1];
  if (Object.hasOwn(FEM_REFINEMENT_PRESETS, settings.femPreset)) R.femPreset = settings.femPreset;
  else R.femPreset = femPresetForLevels(R.levels);
  if (FEM_NODE_MULTIPLIERS.includes(Number(settings.nodeMultiplier))) R.nodeMultiplier = Number(settings.nodeMultiplier);
  if (Number.isFinite(settings.threads)) R.threads = settings.threads;
  if (Number.isFinite(settings.liveEvery)) R.liveEvery = settings.liveEvery;
  if (POINT_MULTIPLIERS.includes(Number(settings.pointMultiplier))) R.pointMultiplier = Number(settings.pointMultiplier);
  if (S.options.deepxde_models.some((item) => item.id === settings.deepxdeModel)) R.deepxdeModel = settings.deepxdeModel;
  if ((S.options.deepxde_samplers || []).some((item) => item.id === settings.deepxdeSampling)) R.deepxdeSampling = settings.deepxdeSampling;
}

function currentRunSettings() {
  return { kind: R.kind, arch: R.arch, preset: R.preset, p: clone(R.p), levels: R.levels.slice(),
    femPreset: R.femPreset, nodeMultiplier: R.nodeMultiplier,
    threads: R.threads, liveEvery: R.liveEvery, pointMultiplier: R.pointMultiplier, deepxdeModel: R.deepxdeModel,
    deepxdeSampling: R.deepxdeSampling };
}

function previousPinRunsForCase() {
  const sha = R.check && R.check.summary && R.check.summary.sha256;
  if (!sha) return [];
  const loadedJobId = R.src.startsWith('job:') ? R.src.slice(4) : null;
  return (S.jobs || []).filter((job) => job.kind === 'pinn' && job.status !== 'running' && (job.sha === sha || job.id === loadedJobId) && job.params);
}

function compatiblePreviousRun() {
  const current = R.p || {};
  const candidates = previousPinRunsForCase().filter((job) => job.has_checkpoint && (!R.resumeJobId || job.id === R.resumeJobId));
  return candidates.find((job) => {
    const previous = job.params || {};
    return previous.architecture === R.arch && Number(previous.width || 32) === Number(current.width || 32) &&
      Number(previous.depth || 3) === Number(current.depth || 3) && (previous.activation || 'tanh') === (current.activation || 'tanh') &&
      (previous.deepxde_model || 'PFNN') === (R.arch === 'DeepXDE' ? R.deepxdeModel : 'PFNN') &&
      (previous.attention || '') === (current.attention || '');
  }) || null;
}

async function loadPreviousRun(job) {
  if (!job || job.kind !== 'pinn' || job.status === 'running' || !job.params) return;
  R.src = 'job:' + job.id;
  await loadRunCase();
  const canResume = Boolean(compatiblePreviousRun());
  toast(canResume ? 'Caso y parametros cargados; puedes personalizarlos antes de continuar.' : 'Parametros cargados, pero la corrida no tiene un checkpoint compatible.', canResume ? 'ok' : 'warn');
}

async function loadRunCase() {
  R.check = null;
  R.collocationCache = null;
  try {
    if (R.src === 'current') {
      if (!S.doc) { R.src = ''; R.doc = null; drawRunLeft(); drawRunGeo(); return; }
      R.doc = S.doc;
    } else if (R.src.startsWith('job:')) {
      const id = R.src.slice(4);
      const [previous, savedCase] = await Promise.all([api('/api/job/' + encodeURIComponent(id)), api('/api/job/' + encodeURIComponent(id) + '/case')]);
      R.doc = savedCase.doc;
      R.check = await post('/api/validate', { doc: R.doc });
      if (!R.check.ok) throw new Error('El caso guardado con esta corrida ya no valida.');
      applyPreviousRunParameters(previous);
    } else if (R.src) {
      R.doc = (await api('/api/case?' + qs({ ref: R.src }))).doc;
    }
    if (!R.src.startsWith('job:')) R.check = R.doc ? await post('/api/validate', { doc: R.doc }) : null;
    const key = runSettingsKey();
    if (key && key !== R.settingsKey) {
      R.continueFrom = false;
      R.settingsKey = key;
      if (R.src.startsWith('job:')) R.settingsKey = key;
      else {
        R.resumeJobId = null;
        try {
          const saved = await api('/api/run-settings?' + qs({ ref: key }));
          R.settingsKey = key;
          if (saved.settings) applyRunSettings(saved.settings); else resetRunSettings();
        } catch (e) { R.settingsKey = null; resetRunSettings(); toast('No se pudieron cargar los parametros guardados: ' + e.message, 'bad'); }
      }
    }
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
    h('input', { type: 'number', step: step || 'any', value: obj[key], onchange: (e) => {
      const v = parseFloat(e.target.value);
      if (!Number.isNaN(v)) obj[key] = v;
      R.preset = 'custom';
      if (obj === R.p && (key === 'width' || key === 'depth')) drawRunLeft();
      if (obj === R.p.points) { R.collocationCache = null; drawRunLeft(); if (R.showCollocation) drawRunGeo(); }
    } }));
}

function pinnRunParams() {
  const params = Object.assign({ architecture: R.arch }, clone(R.p));
  params.points = scaledPointCounts(params.points, R.pointMultiplier);
  if (R.arch === 'DeepXDE') {
    params.deepxde_model = R.deepxdeModel;
    params.deepxde_sampling = R.deepxdeSampling;
    params.attention = '';
    params.weights.constitutive = 0;
  }
  return params;
}

function collocationKey(params) {
  return JSON.stringify([R.check && R.check.summary && R.check.summary.sha256, params]);
}

async function requestCollocationPreview(includePoints) {
  if (!R.doc || !R.check || !R.check.ok || R.kind !== 'pinn') return null;
  const params = pinnRunParams();
  const key = collocationKey(params);
  if (R.collocationCache && R.collocationCache.key === key && (!includePoints || R.collocationCache.points)) return R.collocationCache;
  const result = await post('/api/collocation-preview', { doc: R.doc, params, include_points: includePoints });
  if (collocationKey(pinnRunParams()) !== key) return null;
  R.collocationCache = Object.assign({ key }, result);
  return R.collocationCache;
}

async function refreshCollocationSummary() {
  const node = $('#collocationSummary');
  if (!node || R.kind !== 'pinn' || !R.check || !R.check.ok) return;
  const key = collocationKey(pinnRunParams());
  node.textContent = 'Calculando total de puntos...';
  try {
    const result = await requestCollocationPreview(false);
    if (!result || !node.isConnected || key !== collocationKey(pinnRunParams())) return;
    const summary = result.summary;
    node.textContent = `Total previsto: ${summary.total_points.toLocaleString('es-PE')} puntos (${summary.interior_points.toLocaleString('es-PE')} interiores, ${summary.interface_points.toLocaleString('es-PE')} interfaces, ${summary.boundary_points.toLocaleString('es-PE')} frontera).`;
  } catch (error) {
    if (node.isConnected) node.textContent = 'No se pudo calcular el total: ' + error.message;
  }
}

function archCard(a) {
  const fem = a.id === 'FEM';
  const check = R.check;
  const off = check && check.ok && (fem ? !check.fem.supported : !check.pinn.supported);
  const on = fem ? R.kind === 'fem' : R.kind === 'pinn' && R.arch === a.id;
  const why = off ? (fem ? check.fem.reason : check.pinn.reason) : '';
  return h('div', { class: 'arch' + (fem ? ' fem' : '') + (on ? ' on' : '') + (off ? ' off' : ''), title: off ? why : a.note,
    onclick: () => { if (off) { toast(why || 'No disponible para este caso', 'bad'); return; } R.kind = fem ? 'fem' : 'pinn'; if (!fem) { R.arch = a.id; if (a.id === 'DeepXDE' && R.p) R.p.attention = ''; } drawRunLeft(); } },
    h('b', {}, a.id), h('span', {}, a.title.replace(/^[A-Z0-9]+ - /, '')));
}
function pinnParams() {
  const o = S.options;
  const P = R.p;
  const chips = Object.keys(o.presets).map((k) => h('span', { class: 'chip' + (R.preset === k ? ' on' : ''), onclick: () => { R.preset = k; R.p = presetParams(k); drawRunLeft(); } }, o.presets[k].title));
  const w = o.weights;
  const internalModel = R.arch === 'DeepXDE' ? h('div', { class: 'grid2' },
    h('label', { class: 'f' }, h('span', {}, 'Red interna'),
      h('select', { value: R.deepxdeModel, onchange: (e) => { R.deepxdeModel = e.target.value; R.collocationCache = null; drawRunLeft(); if (R.showCollocation) drawRunGeo(); } },
        o.deepxde_models.map((model) => h('option', { value: model.id, selected: R.deepxdeModel === model.id }, model.title)))),
    h('label', { class: 'f' }, h('span', {}, 'Muestreador DeepXDE'),
      h('select', { value: R.deepxdeSampling, onchange: (e) => { R.deepxdeSampling = e.target.value; R.collocationCache = null; drawRunLeft(); if (R.showCollocation) drawRunGeo(); } },
        (o.deepxde_samplers || []).map((sampler) => h('option', { value: sampler.id, selected: R.deepxdeSampling === sampler.id }, sampler.title))))) : null;
  return [internalModel, h('div', { class: 'chips' }, chips, R.preset === 'custom' ? h('span', { class: 'chip on' }, 'Personalizado') : null),
    h('div', { class: 'point-scale' },
      h('label', { class: 'f' }, h('span', {}, 'Multiplicador de puntos'), h('select', { value: String(R.pointMultiplier), onchange: (e) => { R.pointMultiplier = Number(e.target.value); R.collocationCache = null; drawRunLeft(); if (R.showCollocation) drawRunGeo(); } },
        POINT_MULTIPLIERS.map((value) => h('option', { value: String(value), selected: R.pointMultiplier === value }, value + 'x')))),
      h('span', { class: 'sub' }, 'Escala los cuatro conteos al iniciar.')),
    h('div', { class: 'grid2', style: 'margin-top:8px' }, rnum('Pasos Adam', P, 'adam_steps', 100), rnum('Pasos L-BFGS', P, 'lbfgs_steps', 10, P.attention ? '(0 con atencion)' : ''),
      rnum('Ancho de capa', P, 'width', 4), rnum('Capas ocultas', P, 'depth', 1), rnum('Tasa de aprendizaje', P, 'lr', 0.0005)),
    h('details', { class: 'sec', open: !!R.advOpen, ontoggle: (e) => { R.advOpen = e.target.open; } }, h('summary', {}, 'Opciones avanzadas'),
      h('div', { class: 'in' },
        h('div', { class: 'grid2' },
          h('label', { class: 'f' }, h('span', {}, 'Activacion'), h('select', { onchange: (e) => { P.activation = e.target.value; R.preset = 'custom'; drawRunLeft(); } }, o.activations.map((a) => h('option', { value: a, selected: P.activation === a }, a)))),
          R.arch === 'DeepXDE' ? null : h('label', { class: 'f' }, h('span', {}, 'Atencion'), h('select', { onchange: (e) => { P.attention = e.target.value; if (P.attention) P.lbfgs_steps = 0; drawRunLeft(); } }, o.attention.map((a) => h('option', { value: a.id, selected: P.attention === a.id }, a.title)))),
          rnum('Semilla de la red', P, 'init_seed', 1), rnum('Semilla del muestreo', P, 'sampling_seed', 1), rnum('Registrar cada N pasos', P, 'log_every', 5)),
        h('h4', {}, 'Pesos de la perdida'), h('div', { class: 'grid2' }, Object.keys(w).filter((k) => R.arch !== 'DeepXDE' || k !== 'constitutive').map((k) => rnum(k, P.weights, k, 'any'))),
        h('h4', {}, 'Puntos de colocacion'), h('div', { class: 'grid2' }, Object.keys(P.points).filter((key) => {
          if (key === 'n_layer') return R.doc && R.doc.physical && R.doc.physical.kind === 'cable';
          if (key === 'n_face') return !R.check || !R.check.summary || R.check.summary.regions > 1;
          return true;
        }).map((key) => {
          const labels = {
            n_interior: ['Puntos interiores por region base', 'Por cada region fuera de las capas de cable'],
            n_layer: ['Interiores por capa de cable', 'Por cada conductor y capa'],
            n_face: ['Puntos por interfaz', 'Por cada interfaz material'],
            n_boundary: ['Puntos por tramo de frontera', 'Por cada pieza de frontera']
          };
          return rnum(labels[key][0], P.points, key, 8, labels[key][1]);
        })),
        h('p', { class: 'sub', id: 'collocationSummary' }, 'Calculando total de puntos...'),
        h('p', { class: 'sub' }, 'Los pesos y los puntos son solo hiperparametros del entrenamiento; ninguna referencia (FEM o articulo) entra como dato de entrenamiento.')))];
}

function femParams() {
  const chips = [0, 1, 2, 3].map((l) => h('span', { class: 'chip fem' + (R.levels.includes(l) ? ' on' : ''), onclick: () => {
    R.levels = R.levels.includes(l) ? R.levels.filter((x) => x !== l) : R.levels.concat([l]).sort();
    if (!R.levels.length) R.levels = [l];
    R.femPreset = femPresetForLevels(R.levels);
    drawRunLeft();
  } }, 'Nivel ' + l));
  const presets = Object.entries(FEM_REFINEMENT_PRESETS).map(([name, levels]) => h('button', {
    class: R.femPreset === name ? 'on' : '', title: 'Ejecuta los niveles ' + levels.join(', '),
    onclick: () => { R.femPreset = name; R.levels = levels.slice(); drawRunLeft(); }
  }, name[0].toUpperCase() + name.slice(1)));
  const multipliers = FEM_NODE_MULTIPLIERS.map((value) => h('button', {
    class: R.nodeMultiplier === value ? 'on' : '',
    onclick: () => { R.nodeMultiplier = value; drawRunLeft(); }
  }, value + 'x'));
  return [
    h('p', { class: 'sub' }, 'FEniCSx (WSL2), elementos P2. Los presets ejecutan una escalera de niveles para comparar refinamiento.'),
    h('label', { class: 'f' }, h('span', {}, 'Refinamiento'), h('div', { class: 'seg' }, presets,
      R.femPreset === 'manual' ? h('span', { class: 'chip fem on' }, 'Manual') : null)),
    h('div', { class: 'chips' }, chips),
    h('label', { class: 'f' }, h('span', {}, 'Densidad nodal objetivo'), h('div', { class: 'seg' }, multipliers)),
    h('p', { class: 'sub' }, '1,5x / 2x / 3x escala aproximadamente los DOFs objetivo; el FEM informa los grados de libertad reales. Nivel 3 o multiplicadores altos pueden tardar más y usar más memoria.')
  ];
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
  const priorRuns = R.kind === 'pinn' ? previousPinRunsForCase() : [];
  const previousRun = compatiblePreviousRun();
  if (!previousRun) R.continueFrom = false;
  mount(box,
    h('h2', {}, 'Que quieres correr'),
    h('div', { class: 'run-actions' },
      h('div', { class: 'row' },
        h('button', { class: 'btn run-start', disabled: !ready || R.busy, onclick: startRun }, R.kind === 'fem' ? 'Ejecutar FEM' : 'Entrenar ' + R.arch),
        R.kind === 'pinn' ? h('label', { class: 'chk', title: previousRun ? 'Usa los pesos de ' + previousRun.id + '; se reinicia el optimizador.' : (priorRuns.length ? 'Carga los parámetros y el caso original desde una corrida con checkpoint compatible.' : 'No hay un checkpoint compatible para este caso y esta arquitectura.') },
          h('input', { type: 'checkbox', checked: R.continueFrom, disabled: !previousRun || R.busy, onchange: (e) => { R.continueFrom = e.target.checked; } }),
          'Continuar desde la solución anterior') : null,
        h('button', { class: 'btn sec run-reset', onclick: () => { R.preset = R.preset === 'custom' ? 'rapido' : R.preset; R.p = presetParams(R.preset); R.pointMultiplier = 1; R.femPreset = 'medio'; R.levels = FEM_REFINEMENT_PRESETS.medio.slice(); R.nodeMultiplier = 1; drawRunLeft(); } }, 'Restablecer')),
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
  refreshCollocationSummary();
}

async function drawRunGeo() {
  const box = $('#runGeo');
  if (!box) return;
  const kind = R.doc && R.doc.physical && R.doc.physical.kind;
  if (!kind || (kind !== 'cable' && kind !== 'annulus' && !kind.startsWith('mms'))) { mount(box, h('div', { class: 'muted small' }, 'Este tipo de caso no tiene vista geometrica.')); return; }
  const request = ++R.geoRequest;
  let preview = null;
  if (R.kind === 'pinn' && R.showCollocation) {
    try { preview = await requestCollocationPreview(true); }
    catch (error) { toast('No se pudieron cargar los puntos de colocacion: ' + error.message, 'bad'); }
  }
  if (request !== R.geoRequest || !box.isConnected) return;
  let svg;
  try { svg = geoSvg(R.doc, R.geoView, preview && preview.points); } catch (e) { mount(box, h('div', { class: 'muted small' }, 'No se puede dibujar: ' + e.message)); return; }
  mount(box, h('div', { class: 'row', style: 'justify-content:space-between' }, h('h3', { style: 'margin:0' }, kind === 'cable' ? 'Geometria del caso' : 'Dominio de verificacion 2D'),
    h('div', { class: 'row' }, kind === 'cable' ? h('div', { class: 'seg' }, [['full', 'Dominio'], ['zoom', 'Cables']].map(([v, l]) => h('button', { class: R.geoView === v ? 'on' : '', onclick: () => { R.geoView = v; drawRunGeo(); } }, l))) : null,
      R.kind === 'pinn' ? h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: R.showCollocation, onchange: (e) => { R.showCollocation = e.target.checked; drawRunGeo(); if (R.job) updateMon(R.job, true); } }), 'Puntos de colocacion') : null)),
    svg, preview ? h('div', { class: 'row small muted', style: 'gap:10px;margin-top:6px' },
      h('span', {}, h('i', { style: 'display:inline-block;width:8px;height:8px;border-radius:50%;background:#0284c7;margin-right:4px' }), 'Interior'),
      h('span', {}, h('i', { style: 'display:inline-block;width:8px;height:8px;border-radius:50%;background:#e11d48;margin-right:4px' }), 'Interfaces'),
      h('span', {}, h('i', { style: 'display:inline-block;width:8px;height:8px;border-radius:50%;background:#ca8a04;margin-right:4px' }), 'Frontera'),
      h('b', {}, 'Total ' + preview.summary.total_points.toLocaleString('es-PE'))) : null);
}

async function startRun() {
  if (R.busy) return;
  R.busy = true;
  drawRunLeft();
  try {
    const params = R.kind === 'fem' ? { levels: R.levels.slice(), refinement_preset: R.femPreset, node_multiplier: R.nodeMultiplier } : pinnRunParams();
    const body = { kind: R.kind, threads: R.threads, live_every_s: R.liveEvery, params };
    const resumeJob = R.continueFrom ? compatiblePreviousRun() : null;
    if (R.continueFrom && !resumeJob) throw new Error('La solucion anterior ya no es compatible con estos parametros.');
    if (resumeJob) body.resume_from = resumeJob.id;
    if (R.src === 'current' || R.src.startsWith('job:')) body.doc = R.doc; else body.ref = R.src;
    const r = await post('/api/run', body);
    R.jobId = r.id; R.job = null; R.variant = null;
    const key = runSettingsKey();
    if (key) {
      try { await post('/api/run-settings', { ref: key, settings: currentRunSettings() }); R.settingsKey = key; }
      catch (e) { toast('El trabajo inicio, pero no se guardaron sus parametros: ' + e.message, 'bad'); }
    }
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