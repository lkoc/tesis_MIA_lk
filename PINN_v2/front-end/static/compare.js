'use strict';
// Pestana Comparar: elige una referencia y una o varias candidatas de un mismo caso y muestra los errores.
const C = { cases: [], case: null, items: [], ref: null, cands: new Set(), view: 'full', mode: 'signed', scale: 'auto', scaleVal: 1, cmap: 'inferno', ecmap: '',
  overlay: true, detail: false, axis: 'x', at: 0, data: null, busy: false, err: null, token: 0 };

async function enterCompare() {
  try {
    const r = await api('/api/solutions');
    C.cases = r.cases || [];
  } catch (e) { C.cases = []; }
  if (S.cmpCase) { C.case = S.cmpCase; S.cmpCase = null; C.ref = null; C.cands = new Set(); }
  if (!C.case || !C.cases.some((c) => c.case === C.case)) {
    const both = C.cases.find((c) => c.fem > 0 && c.pinn > 0) || C.cases[0];
    C.case = both ? both.case : null;
  }
  await loadCmpItems();
}

async function loadCmpItems() {
  C.items = [];
  if (C.case) {
    try { C.items = (await api('/api/solutions?' + qs({ case: C.case }))).items || []; } catch (e) { toast(e.message, 'bad'); }
  }
  const ids = new Set(C.items.map((i) => i.sid));
  if (!C.ref || !ids.has(C.ref)) {
    const fem = C.items.filter((i) => i.kind === 'fem');
    const best = fem.length ? fem.slice().sort((a, b) => (b.ndofs || 0) - (a.ndofs || 0))[0] : C.items[0];
    C.ref = best ? best.sid : null;
    C.cands = new Set();
  }
  for (const c of Array.from(C.cands)) if (!ids.has(c) || c === C.ref) C.cands.delete(c);
  if (!C.cands.size) {
    const pick = C.items.filter((i) => i.sid !== C.ref && i.kind === 'pinn').slice(0, 3);
    for (const i of pick) C.cands.add(i.sid);
  }
  selectionChanged();
}

function selectionChanged() {
  drawCmpLeft();
  if (C.drawn) drawCmpRight();
  refreshCompare();
}

function solRow(it) {
  const isRef = C.ref === it.sid;
  const when = it.when ? new Date(it.when * 1000).toLocaleString() : '';
  return h('div', { class: 'sol' + (isRef ? ' hl' : '') },
    h('input', { type: 'radio', name: 'cref', checked: isRef, title: 'Usar como referencia', onchange: () => { C.ref = it.sid; C.cands.delete(it.sid); selectionChanged(); } }),
    h('input', { type: 'checkbox', checked: C.cands.has(it.sid) && !isRef, disabled: isRef, title: 'Comparar contra la referencia', onchange: (e) => { if (e.target.checked) C.cands.add(it.sid); else C.cands.delete(it.sid); selectionChanged(); } }),
    h('div', {}, h('div', {}, h('b', {}, it.label)),
      h('div', { class: 'meta' }, [it.Tmax_C !== undefined && it.Tmax_C !== null ? 'Tmax ' + fmt(it.Tmax_C, 2) + ' C' : null, it.ndofs ? it.ndofs + ' gdl' : null, it.final_loss ? 'perdida ' + Number(it.final_loss).toExponential(2) : null, it.origin === 'resultados' ? 'carpeta results' : (it.origin === 'benchmarks' ? 'Benchmarks (solo lectura)' : 'front-end'), when].filter(Boolean).join('  -  '))));
}

function drawCmpLeft() {
  const box = $('#cmpLeft');
  if (!box) return;
  const fem = C.items.filter((i) => i.kind === 'fem'), pinn = C.items.filter((i) => i.kind === 'pinn');
  const group = (title, rows) => rows.length ? h('div', {}, h('h4', { style: 'margin:10px 0 2px' }, title), rows.map(solRow)) : null;
  mount(box,
    h('h3', { style: 'margin:0 0 6px' }, 'Caso'),
    h('select', { onchange: (e) => { C.case = e.target.value; C.ref = null; C.cands = new Set(); loadCmpItems(); } },
      C.cases.length ? C.cases.map((c) => h('option', { value: c.case, selected: c.case === C.case }, c.case + '  (FEM ' + c.fem + ' / PINN ' + c.pinn + ')')) : h('option', {}, 'Sin soluciones todavia')),
    h('p', { class: 'sub' }, 'Solo se comparan soluciones de un mismo caso con la misma malla de comparacion. Referencia = circulo; candidatas = casillas.'),
    C.items.length ? [group('FEM', fem), group('PINN', pinn)] : h('div', { class: 'empty' }, 'Aun no hay soluciones de este caso. Lanza un FEM y un PINN en la pestana Ejecucion.',
      h('div', { style: 'margin-top:8px' }, h('button', { class: 'btn sm', onclick: () => goTab('ejecucion') }, 'Ir a Ejecucion'))),
    h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { class: 'btn ghost sm', onclick: () => { C.cands = new Set(C.items.filter((i) => i.sid !== C.ref).map((i) => i.sid)); selectionChanged(); } }, 'Todas'),
      h('button', { class: 'btn ghost sm', onclick: () => { C.cands = new Set(); selectionChanged(); } }, 'Ninguna'),
      h('button', { class: 'btn ghost sm', onclick: () => enterCompare() }, 'Recargar')));
}
function cmpScaleParam() { return C.scale === 'manual' ? String(C.scaleVal) : C.scale; }
function candList() { return Array.from(C.cands).filter((s) => s !== C.ref); }

function cmpUrl(kind) {
  const base = { ref: C.ref, cands: candList().join(',') };
  if (kind === 'png') return '/api/compare.png?' + qs(Object.assign(base, { view: C.view, mode: C.mode, scale: cmpScaleParam(), cmap: C.cmap, error_cmap: C.ecmap, overlay: C.overlay ? 1 : 0, detail: C.detail ? 1 : 0 }));
  if (kind === 'profile') return '/api/profile.png?' + qs(Object.assign(base, { axis: C.axis, at: C.at, mode: C.mode === 'rel' ? 'rel' : 'signed' }));
  return '/api/compare.json?' + qs(base);
}

function cmpSeg(opts, cur, set) { return h('div', { class: 'seg' }, opts.map(([v, l]) => h('button', { class: cur === v ? 'on' : '', onclick: () => set(v) }, l))); }

function drawCmpRight() {
  const box = $('#cmpRight');
  if (!box) return;
  const singleSolution = !!C.ref && !candList().length;
  const cmaps = ['inferno', 'viridis', 'magma', 'plasma', 'turbo', 'coolwarm', 'cividis'];
  const sel = (cur, set, extra) => h('select', { style: 'width:auto', onchange: (e) => { set(e.target.value); refreshCompare(); } }, (extra || []).concat(cmaps).map((c) => h('option', { value: c, selected: cur === c }, c || 'auto')));
  const ctrl = h('div', { class: 'card pad' },
    h('div', { class: 'row', style: 'gap:14px;align-items:flex-end' },
      h('div', {}, h('div', { class: 'sub' }, 'Ventana'), cmpSeg([['full', 'Dominio'], ['zoom', 'Cables']], C.view, (v) => { C.view = v; drawCmpRight(); refreshCompare(); })),
      !singleSolution ? h('div', {}, h('div', { class: 'sub' }, 'Mapa de error'), cmpSeg([['signed', 'Con signo'], ['abs', 'Absoluto'], ['rel', 'Relativo %']], C.mode, (v) => { C.mode = v; drawCmpRight(); refreshCompare(); })) : null,
      !singleSolution ? h('div', {}, h('div', { class: 'sub' }, 'Escala del color'), h('div', { class: 'row' }, cmpSeg([['auto', 'Auto'], ['max', 'Maximo'], ['manual', 'Fija']], C.scale, (v) => { C.scale = v; drawCmpRight(); refreshCompare(); }),
        C.scale === 'manual' ? h('input', { type: 'number', step: 'any', min: '0', value: C.scaleVal, style: 'width:80px', onchange: (e) => { C.scaleVal = Math.max(1e-9, Number(e.target.value) || 1); refreshCompare(); } }) : null)) : null,
      h('div', {}, h('div', { class: 'sub' }, 'Paleta T'), sel(C.cmap, (v) => { C.cmap = v; })),
      !singleSolution ? h('div', {}, h('div', { class: 'sub' }, 'Paleta error'), sel(C.ecmap, (v) => { C.ecmap = v; }, [''])) : null,
      h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: C.overlay, onchange: (e) => { C.overlay = e.target.checked; refreshCompare(); } }), 'Cables encima'),
      !singleSolution ? h('label', { class: 'chk', title: 'Con una sola candidata ya se muestra el triptico' }, h('input', { type: 'checkbox', checked: C.detail, onchange: (e) => { C.detail = e.target.checked; refreshCompare(); } }), 'Triptico (ref | cand | error)') : null),
    h('p', { class: 'sub', style: 'margin:8px 0 0' }, singleSolution ? 'Vista individual de la solucion seleccionada.' : 'Error = candidata menos referencia. Con varias candidatas se muestra una galeria con la misma escala para que se puedan comparar a simple vista.'));
  C.msg = h('div');
  C.imgBox = h('div', { class: 'card pad', style: 'margin-top:12px;text-align:center' });
  C.metBox = h('div', { class: 'card pad', style: 'margin-top:12px' });
  C.profBox = h('div', { class: 'card pad', style: 'margin-top:12px' });
  mount(box, ctrl, C.msg, C.imgBox, C.metBox, C.profBox);
  C.drawn = true;
}
const refreshCompare = debounce(() => doCompare(), 250);

function pngButton(url, name) { return h('a', { class: 'btn ghost sm', href: url, target: '_blank', onclick: (e) => { e.preventDefault(); const a = h('a', { href: url }); a.download = name; document.body.appendChild(a); a.click(); a.remove(); } }, 'Descargar PNG'); }

async function doCompare() {
  if (!$('#cmpRight')) return;
  if (!C.drawn || !C.imgBox || !C.imgBox.isConnected) drawCmpRight();
  const cands = candList();
  const token = ++C.token;
  if (!C.ref) {
    mount(C.msg);
    mount(C.imgBox, h('div', { class: 'empty' }, 'Elige un caso con soluciones y una referencia.'));
    mount(C.metBox); mount(C.profBox);
    return;
  }
  if (!cands.length) {
    const selected = C.items.find((item) => item.sid === C.ref);
    const url = '/api/solution.png?' + qs({ sid: C.ref, view: C.view, cmap: C.cmap, overlay: C.overlay ? 1 : 0 });
    const img = h('img', { alt: selected ? selected.label : 'campo de temperatura', style: 'max-width:100%', onload: () => {
      if (token === C.token) mount(C.imgBox, h('div', { class: 'row', style: 'justify-content:flex-end;margin-bottom:6px' }, pngButton(url, 'solucion_' + C.case + '.png')), img);
    }, onerror: () => { if (token === C.token) mount(C.imgBox, banner('bad', 'No se pudo dibujar esta solucion.')); } });
    img.src = url;
    mount(C.msg);
    mount(C.imgBox, h('div', { class: 'empty' }, 'Dibujando el campo de temperatura...'), img);
    mount(C.metBox, banner('info', selected ? selected.label + '. Campo FEM original, sin recalculo; aun no hay otra solucion para comparar.' : 'Vista individual; aun no hay otra solucion para comparar.'));
    mount(C.profBox);
    return;
  }
  mount(C.imgBox, h('div', { class: 'empty' }, 'Dibujando la comparacion...'));
  const url = cmpUrl('png');
  const img = h('img', { alt: 'comparacion', style: 'max-width:100%', onload: () => { if (token === C.token) mount(C.imgBox, h('div', { class: 'row', style: 'justify-content:flex-end;margin-bottom:6px' }, pngButton(url, 'comparacion_' + C.case + '.png')), img); },
    onerror: () => { if (token === C.token) mount(C.imgBox, banner('bad', 'No se pudo dibujar la comparacion (revisa que las soluciones tengan la misma malla).')); } });
  img.src = url;
  let data = null;
  try { data = await api(cmpUrl('json')); } catch (e) { mount(C.metBox, banner('bad', e.message)); return; }
  if (token !== C.token) return;
  C.data = data;
  drawMetrics(data);
  drawProfile();
}

function drawMetrics(data) {
  const rows = data.rows || [];
  const warn = [];
  if ((data.skipped || []).length) warn.push(banner('info', 'Se omitieron por tener otra malla o dominio: ' + data.skipped.join(', ') + '.'));
  if (rows.some((r) => r.same_physics === false)) warn.push(banner('bad', 'Algunas candidatas se calcularon con una fisica distinta (hash de caso diferente): su diferencia mezcla error numerico y cambio de datos.'));
  const ok = rows.filter((r) => !r.error);
  const top = Math.max(1e-12, ...ok.map((r) => r.rmse_K));
  const cell = (v, d) => h('td', {}, v === undefined || v === null ? '-' : fmt(v, d));
  const body = rows.map((r) => r.error ? h('tr', {}, h('td', {}, r.label), h('td', { colspan: 8 }, r.error)) :
    h('tr', {}, h('td', {}, h('b', {}, r.label), r.same_physics === false ? ' ' : '', r.same_physics === false ? badge('otra fisica', 'bad') : null),
      h('td', { class: 'bar' }, h('span', { class: 'b', style: 'width:' + Math.max(2, 100 * r.rmse_K / top) + '%' })),
      cell(r.rmse_K, 4), cell(r.mae_K, 4), cell(r.max_abs_K, 4), cell(r.bias_K, 4), cell(r.nrmse_pct, 3), cell(r.conductor_tmax_diff_K, 4), h('td', {}, r.max_at ? '(' + fmt(r.max_at[0], 3) + ', ' + fmt(r.max_at[1], 3) + ')' : '-')));
  const regions = ok.filter((r) => (r.regions || []).length).map((r) => h('details', { class: 'sec', style: 'margin-top:6px' }, h('summary', {}, 'Error por region: ' + r.label),
    h('div', { class: 'in' }, h('table', { class: 't num m' }, h('tr', {}, ['Region', 'RMSE (K)', 'Max (K)', 'Celdas'].map((t) => h('th', {}, t))),
      r.regions.map((g, i) => h('tr', { class: i === 0 ? 'hl' : '' }, h('td', {}, g.region), cell(g.rmse_K, 4), cell(g.max_abs_K, 4), h('td', {}, g.cells))),
      h('tr', {}, h('td', { colspan: 4, class: 'muted small' }, 'Se muestran las ' + r.regions.length + ' regiones con mayor RMSE de ' + r.regions_total + '.'))))));
  mount(C.metBox, h('div', { class: 'row', style: 'justify-content:space-between' }, h('h3', { style: 'margin:0' }, 'Metricas frente a ' + data.reference), h('button', { class: 'btn ghost sm', onclick: () => csvOut(data) }, 'Descargar CSV')),
    h('p', { class: 'sub' }, 'Calculadas sobre la malla comun de todo el dominio (' + (ok[0] ? ok[0].cells : 0) + ' celdas); salto termico de la referencia ' + (ok[0] ? fmt(ok[0].rise_K, 2) : '-') + ' K. dTmax cond. = T max de la candidata menos la de la referencia en el conductor.'),
    warn,
    h('div', { style: 'overflow:auto' }, h('table', { class: 't num m' }, h('tr', {}, ['Candidata', '', 'RMSE (K)', 'MAE (K)', 'Max |e| (K)', 'Sesgo (K)', 'NRMSE (%)', 'dTmax cond. (K)', 'Donde esta el maximo'].map((t) => h('th', {}, t))), body)),
    regions);
}

function csvOut(data) {
  const cols = ['label', 'rmse_K', 'mae_K', 'max_abs_K', 'bias_K', 'nrmse_pct', 'conductor_tmax_ref_C', 'conductor_tmax_cand_C', 'conductor_tmax_diff_K', 'same_physics'];
  const lines = [['referencia'].concat(cols).join(',')];
  for (const r of data.rows) lines.push([JSON.stringify(data.reference)].concat(cols.map((c) => (r[c] === undefined || r[c] === null ? '' : c === 'label' ? JSON.stringify(r[c]) : r[c]))).join(','));
  download('metricas_' + C.case + '.csv', lines.join('\n') + '\n', 'text/csv');
}

function drawProfile() {
  const url = cmpUrl('profile');
  const img = h('img', { alt: 'perfil', style: 'max-width:100%' });
  img.src = url;
  mount(C.profBox, h('div', { class: 'row', style: 'justify-content:space-between' }, h('h3', { style: 'margin:0' }, 'Perfil de temperatura'),
    h('div', { class: 'row' }, cmpSeg([['x', 'Vertical (x fijo)'], ['y', 'Horizontal (y fijo)']], C.axis, (v) => { C.axis = v; drawProfile(); }),
      h('label', { class: 'f', style: 'flex-direction:row;align-items:center;gap:6px' }, h('span', {}, C.axis === 'x' ? 'x =' : 'y ='), h('input', { type: 'number', step: 'any', value: C.at, style: 'width:90px', onchange: (e) => { C.at = Number(e.target.value) || 0; drawProfile(); } })),
      pngButton(url, 'perfil_' + C.case + '.png'))),
    h('div', { style: 'text-align:center;margin-top:6px' }, img));
}

function initCompare() {
  mount($('#tab-comparar'), h('div', { class: 'cmp' }, h('div', { class: 'card pad', id: 'cmpLeft' }), h('div', { id: 'cmpRight' })));
  S.onTab.comparar = enterCompare;
}