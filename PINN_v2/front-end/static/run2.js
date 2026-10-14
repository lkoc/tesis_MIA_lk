'use strict';
// Pestana Ejecucion (2/2): seguimiento de trabajos, campo en vivo, perdida y resultados.

const POLL = { t: null };
const M = { built: null, sig: '', imgKey: '', lossKey: '' };
const SERIES = ['#111827', '#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2', '#ca8a04', '#be185d'];
const STATUS_TXT = { running: 'en marcha', done: 'terminado', failed: 'con error', stopped: 'detenido', interrupted: 'interrumpido' };
const STATUS_KIND = { running: '', done: 'ok', failed: 'bad', stopped: 'warn', interrupted: 'bad' };

function pollNow() { clearTimeout(POLL.t); pollJobs(); }

async function pollJobs() {
  try {
    const [r, trash] = await Promise.all([api('/api/jobs'), api('/api/trash')]);
    for (const j of r.jobs) {
      const before = (S.jobs.find((x) => x.id === j.id) || {}).status;
      if (before === 'running' && j.status === 'done') toast('Termino: ' + jobName(j), 'ok');
      if (before === 'running' && j.status === 'failed') toast('Fallo: ' + jobName(j), 'bad');
    }
    S.jobs = r.jobs;
    S.trash = trash.jobs || [];
    if (R.jobId) R.job = await api('/api/job/' + R.jobId).catch(() => null);
    updatePills();
    drawJobs();
    drawMonitor();
  } catch (e) { /* servidor ocupado: se reintenta */ }
  POLL.t = setTimeout(pollJobs, S.jobs.some((j) => j.status === 'running') ? 1500 : 6000);
}

function jobName(j) {
  const p = j.params || {};
  const m = j.kind === 'fem' ? 'FEM' : (p.architecture || 'PINN') + (p.deepxde_model ? '/' + p.deepxde_model : '') + (p.attention ? '+' + p.attention : '');
  return m + ' / ' + j.case;
}

function updatePills() {
  const n = S.jobs.filter((j) => j.status === 'running').length;
  const pill = $('#jobsPill');
  pill.style.display = n ? '' : 'none';
  pill.textContent = n + (n === 1 ? ' trabajo en marcha' : ' trabajos en marcha');
}

function drawJobs() {
  const box = $('#runJobs');
  if (!box) return;
  const rows = S.jobs.slice(0, 14).map((j) => {
    const pct = j.total_steps ? Math.min(100, Math.round(100 * (j.step || 0) / j.total_steps)) : 0;
    return h('div', { class: 'j' + (R.jobId === j.id ? ' on' : ''), onclick: () => { R.jobId = j.id; R.job = null; R.variant = null; M.built = null; pollNow(); } },
      h('span', { class: 'dot ' + j.status }), h('div', {}, h('b', {}, jobName(j)), h('div', { class: 'muted small' }, (j.created || '').replace('T', ' ') + (j.title && j.title !== j.case ? '  -  ' + j.title : ''))),
      h('span', { class: 'muted small' }, j.status === 'running' ? pct + '%' : (j.summary && j.summary.Tmax_C !== undefined ? 'Tmax ' + fmt(j.summary.Tmax_C, 2) + ' C' : '')), badge(STATUS_TXT[j.status] || j.status, STATUS_KIND[j.status] || ''),
      h('div', { class: 'job-actions' },
        j.kind === 'pinn' && j.status !== 'running' && j.params ? h('button', { class: 'mini', type: 'button', title: 'Cargar parámetros de esta corrida en Ejecución', onclick: (e) => { e.stopPropagation(); loadPreviousRun(j); } }, 'Cargar parámetros') : null,
        j.status === 'running' ? null : h('button', { class: 'mini', type: 'button', title: 'Mover a la papelera', onclick: (e) => { e.stopPropagation(); archiveJob(j.id); } }, 'Papelera')));
  });
  const trashed = S.trash.map((j) => h('div', { class: 'trash-row' },
    h('div', {}, h('b', {}, jobName(j)), h('div', { class: 'muted small' }, (j.created || '').replace('T', ' '))),
    h('div', { class: 'row' },
      h('button', { class: 'mini', type: 'button', onclick: () => restoreJob(j.id) }, 'Restaurar'),
      h('button', { class: 'mini danger', type: 'button', onclick: () => deleteTrashedJob(j.id) }, 'Eliminar'))));
  mount(box,
    h('div', { class: 'row jobs-heading' }, h('h3', { style: 'margin:0' }, 'Trabajos (' + S.jobs.length + ')'), h('span', { class: 'spacer' }),
      h('button', { class: 'btn sec sm', type: 'button', onclick: () => { S.showTrash = !S.showTrash; drawJobs(); } }, (S.showTrash ? 'Ocultar' : 'Papelera') + ' (' + S.trash.length + ')')),
    rows.length ? rows : h('div', { class: 'muted small', style: 'padding:8px 0' }, 'Todavia no has lanzado nada.'),
    S.showTrash ? h('section', { class: 'trash-panel' },
      h('div', { class: 'row' }, h('h4', { style: 'margin:0' }, 'Papelera'), h('span', { class: 'spacer' }),
        S.trash.length ? h('button', { class: 'btn bad sm', type: 'button', onclick: emptyTrash }, 'Vaciar papelera') : null),
      trashed.length ? trashed : h('div', { class: 'muted small', style: 'padding:8px 0' }, 'La papelera esta vacia.')) : null);
}

async function archiveJob(id) {
  if (!(await confirmBox('Mover a la papelera', 'El trabajo se podra restaurar despues. Sus resultados no se borraran.', 'Mover'))) return;
  try { await post('/api/job/' + id + '/trash', {}); toast('Trabajo movido a la papelera', 'ok'); pollNow(); }
  catch (e) { toast(e.message, 'bad'); }
}

async function restoreJob(id) {
  try { await post('/api/job/' + id + '/restore', {}); toast('Trabajo restaurado', 'ok'); pollNow(); }
  catch (e) { toast(e.message, 'bad'); }
}

async function deleteTrashedJob(id) {
  if (!(await confirmBox('Eliminar definitivamente', 'Se borraran los archivos y resultados de este trabajo. Esta accion no se puede deshacer.', 'Eliminar'))) return;
  try { await post('/api/job/' + id + '/delete', {}); toast('Trabajo eliminado', 'ok'); pollNow(); }
  catch (e) { toast(e.message, 'bad'); }
}

async function emptyTrash() {
  if (!(await confirmBox('Vaciar papelera', 'Se eliminaran definitivamente ' + S.trash.length + ' trabajo(s). Esta accion no se puede deshacer.', 'Vaciar papelera'))) return;
  try { const result = await post('/api/trash/empty', {}); toast(result.deleted + ' trabajo(s) eliminados', 'ok'); pollNow(); }
  catch (e) { toast(e.message, 'bad'); }
}

function drawLoss(canvas, hist) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !hist.length) return;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  const skip = new Set(['step', 'phase', 'lr', 'total']);
  const keys = ['total'].concat(Object.keys(hist[hist.length - 1]).filter((k) => !skip.has(k) && typeof hist[hist.length - 1][k] === 'number')).slice(0, 8);
  const L = 46, Rr = 8, T = 8, B = 50;
  let lo = Infinity, hi = -Infinity;
  for (const row of hist) for (const k of keys) { const v = row[k]; if (v > 0) { lo = Math.min(lo, Math.log10(v)); hi = Math.max(hi, Math.log10(v)); } }
  if (!isFinite(lo)) return;
  lo = Math.floor(lo); hi = Math.ceil(hi); if (hi === lo) hi = lo + 1;
  const x0 = hist[0].step, x1 = Math.max(hist[hist.length - 1].step, x0 + 1);
  const X = (s) => L + (s - x0) / (x1 - x0) * (W - L - Rr);
  const Y = (v) => T + (hi - Math.log10(v)) / (hi - lo) * (H - T - B);
  g.font = '10px Segoe UI, sans-serif'; g.fillStyle = '#64748b'; g.strokeStyle = '#e5e9f0'; g.lineWidth = 1;
  for (let e = lo; e <= hi; e++) { const y = T + (hi - e) / (hi - lo) * (H - T - B); g.beginPath(); g.moveTo(L, y); g.lineTo(W - Rr, y); g.stroke(); g.fillText('1e' + e, 4, y + 3); }
  g.fillText('paso ' + x0, L, H - 38); g.fillText('paso ' + x1, W - Rr - 44, H - 38);
  keys.forEach((k, i) => {
    g.strokeStyle = SERIES[i % SERIES.length]; g.lineWidth = k === 'total' ? 2.2 : 1.1; g.beginPath();
    let started = false;
    for (const row of hist) { if (!(row[k] > 0)) continue; const x = X(row.step), y = Y(row[k]); if (started) g.lineTo(x, y); else { g.moveTo(x, y); started = true; } }
    g.stroke();
    const lx = L + (i % 3) * ((W - L) / 3), ly = H - 22 + 12 * Math.floor(i / 3);
    g.fillStyle = SERIES[i % SERIES.length]; g.fillRect(lx, ly - 7, 8, 8); g.fillStyle = '#334155'; g.fillText(k.slice(0, 18), lx + 11, ly + 1);
  });
}

function buildMon(job) {
  M.built = job.id; M.sig = ''; M.imgKey = ''; M.lossKey = '';
  M.title = h('h2', { style: 'margin:0' });
  M.badge = h('span');
  M.actions = h('div', { class: 'row' });
  M.phase = h('div', { class: 'muted small', style: 'margin:4px 0' });
  M.bar = h('i');
  M.err = h('div');
  M.stats = h('div', { class: 'stats', style: 'margin:8px 0' });
  M.ctrl = h('div', { class: 'row' });
  M.cap = h('div', { class: 'cap' });
  M.imgMsg = h('div', { class: 'empty' }, 'Esperando el primer campo de temperatura...');
  M.img = h('img', { alt: 'campo de temperatura', style: 'display:none', onload: () => { M.img.style.display = ''; M.imgMsg.style.display = 'none'; },
    onerror: () => { M.imgMsg.textContent = 'Todavia no hay un campo que mostrar.'; M.imgMsg.style.display = ''; M.img.style.display = 'none'; } });
  M.canvas = h('canvas', { class: 'loss' });
  M.lossBox = h('div', {}, h('h3', { style: 'margin:0 0 4px' }, 'Perdida (escala log)'), M.canvas);
  M.levels = h('div');
  mount($('#runMon'), h('div', { class: 'row', style: 'justify-content:space-between' }, h('div', { class: 'row' }, M.title, M.badge), M.actions), M.phase,
    h('div', { class: 'bar' + (job.kind === 'fem' ? ' fem' : '') }, M.bar), M.err, M.stats, M.ctrl,
    h('div', { class: 'imgs', style: 'margin-top:8px' }, h('div', { class: 'imgbox' }, M.cap, M.imgMsg, M.img), job.kind === 'pinn' ? M.lossBox : null), M.levels);
}

function statBox(label, value) { return h('div', { class: 'stat' }, h('b', {}, value), h('span', {}, label)); }

function monImage(job) {
  const sols = job.solutions || [];
  let which = null;
  if (sols.length) which = sols.some((s) => s.variant === R.variant) ? R.variant : sols[sols.length - 1].variant;
  else if (job.has_live) which = 'live';
  return which;
}
function monControls(job, which) {
  const sols = job.solutions || [];
  const seg = (opts, cur, set) => h('div', { class: 'seg' }, opts.map(([v, l]) => h('button', { class: cur === v ? 'on' : '', onclick: () => set(v) }, l)));
  const cable = (job.case_kind || 'cable') === 'cable';
  mount(M.ctrl,
    cable ? seg([['zoom', 'Cables'], ['full', 'Dominio']], R.view, (v) => { R.view = v; updateMon(R.job, true); }) : null,
    h('select', { style: 'width:auto', onchange: (e) => { R.cmap = e.target.value; updateMon(R.job, true); } }, ['inferno', 'viridis', 'magma', 'plasma', 'turbo', 'coolwarm', 'cividis'].map((c) => h('option', { value: c, selected: R.cmap === c }, c))),
    h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: R.overlay, onchange: (e) => { R.overlay = e.target.checked; updateMon(R.job, true); } }), cable ? 'Cables encima' : 'Fronteras e interfaces'),
    h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: R.iso, onchange: (e) => { R.iso = e.target.checked; updateMon(R.job, true); } }), 'Isotermas'),
    sols.length > 1 ? h('div', { class: 'chips' }, sols.map((s) => h('span', { class: 'chip fem' + (which === s.variant ? ' on' : ''), onclick: () => { R.variant = s.variant; updateMon(R.job, true); } }, s.label))) : null);
}
function updateMon(job, force) {
  if (!job) return;
  const p = job.params || {};
  const running = job.status === 'running';
  M.title.textContent = jobName(job);
  mount(M.badge, badge(STATUS_TXT[job.status] || job.status, STATUS_KIND[job.status] || ''));
  const pct = job.status === 'done' ? 100 : (job.total_steps ? Math.min(100, 100 * (job.step || 0) / job.total_steps) : 0);
  M.bar.style.width = pct + '%';
  M.phase.textContent = (job.phase || '') + (job.total_steps ? '  -  ' + (job.step || 0) + ' / ' + job.total_steps + (job.kind === 'fem' ? ' niveles' : ' pasos') : '') + '  -  ' + secs(job.elapsed_s);
  const live = running && job.live_step !== null && job.live_step !== undefined && job.kind === 'pinn';
  mount(M.actions, running ? h('button', { class: 'btn bad sm', onclick: async () => { try { await post('/api/job/' + job.id + '/stop', {}); toast('Detenido'); pollNow(); } catch (e) { toast(e.message, 'bad'); } } }, 'Detener') : null,
    job.status === 'done' ? h('button', { class: 'btn sm', onclick: () => { S.cmpCase = job.case; goTab('comparar'); } }, 'Comparar con otras soluciones') : null);
  mount(M.err, job.status === 'failed' || job.status === 'interrupted' ? banner('bad', job.message || 'El trabajo se interrumpio.', job.traceback ? h('details', {}, h('summary', {}, 'Detalle tecnico'), h('pre', { style: 'white-space:pre-wrap;font-size:11px' }, job.traceback)) : null) : null);  const last = (job.history || [])[(job.history || []).length - 1] || {};
  const sm = job.summary || {};
  const boxes = [];
  if (job.kind === 'pinn') {
    boxes.push(statBox('perdida actual', fmt(last.total, 3)), statBox('arquitectura', p.architecture + (p.deepxde_model ? '/' + p.deepxde_model : '') + (p.attention ? '+' + p.attention : '')));
    if (sm.Tmax_C !== undefined) boxes.push(statBox('T maxima (C)', fmt(sm.Tmax_C, 2)));
    if (sm.parameters) boxes.push(statBox('parametros', String(sm.parameters)));
    if (sm.power_balance_pct !== undefined && sm.power_balance_pct !== null) boxes.push(statBox('balance de potencia (%)', fmt(sm.power_balance_pct, 3)));
    if (sm.interface_T_jump_max_K !== undefined && sm.interface_T_jump_max_K !== null) boxes.push(statBox('salto T max. en interfaces (K)', fmt(sm.interface_T_jump_max_K, 4)));
  } else {
    boxes.push(statBox('niveles', (p.levels || []).join(', ')));
    if (sm.Tmax_C !== undefined) boxes.push(statBox('T maxima (C)', fmt(sm.Tmax_C, 3)), statBox('grados de libertad', String(sm.ndofs)));
    if (sm.last_change_pct_of_rise !== undefined && sm.last_change_pct_of_rise !== null) boxes.push(statBox('cambio ultimo nivel (% del salto)', fmt(sm.last_change_pct_of_rise, 3)));
  }
  mount(M.stats, boxes);  const which = monImage(job);
  const sig = [job.status, (job.solutions || []).length, which, R.view, R.cmap, R.overlay, R.iso].join('|');
  if (force || sig !== M.sig) { M.sig = sig; monControls(job, which); }
  const key = [job.id, which, R.view, R.cmap, R.overlay, R.iso, which === 'live' ? job.live_step : 'x'].join('|');
  if (which && key !== M.imgKey) {
    M.imgKey = key;
    M.img.src = '/api/job/' + job.id + '/image.png?' + qs({ which, view: R.view, cmap: R.cmap, overlay: R.overlay ? 1 : 0, iso: R.iso ? 1 : 0, k: key.length + '_' + (job.live_step || 0) });
    clear(M.cap);
    add(M.cap, [live ? h('span', { class: 'livedot' }) : null, which === 'live' ? 'Campo en vivo' + (job.live_step === undefined || job.live_step === null ? ' (primera foto en camino)' : ' (paso ' + job.live_step + ')') : ((job.solutions || []).find((s) => s.variant === which) || {}).label || which]);
  }
  const hist = job.history || [];
  const lk = hist.length + ':' + (last.step || 0);
  if (job.kind === 'pinn' && lk !== M.lossKey) { M.lossKey = lk; drawLoss(M.canvas, hist); }
  if (job.kind === 'fem') {
    mount(M.levels, (job.levels || []).length ? h('table', { class: 't num' }, h('tr', {}, ['Nivel', 'Grados de libertad', 'T max (C)', 'Tiempo (s)'].map((t) => h('th', {}, t))),
      job.levels.map((r) => h('tr', {}, h('td', {}, r.level), h('td', {}, r.ndofs), h('td', {}, fmt(r.Tmax_C, 4)), h('td', {}, fmt(r.elapsed_s, 1))))) : null);
  }
}

function drawMonitor() {
  const job = R.job;
  if (!job) return;
  if (M.built !== job.id) buildMon(job);
  updateMon(job, false);
}