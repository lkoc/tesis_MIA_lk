'use strict';
// Arranque: pide los datos basicos al servidor, cablea las pestanas e inicia el sondeo de trabajos.

function setFemPill() {
  const pill = $('#femPill');
  const ok = !!(S.info && S.info.fem && S.info.fem.available);
  pill.className = 'pill' + (ok ? ' ok' : '');
  pill.textContent = ok ? 'FEM listo (WSL2)' : 'FEM no disponible';
  pill.title = ok ? 'FEniCSx responde en ' + (S.info.fem.distro || 'WSL2') : 'Sin FEniCSx solo se pueden lanzar PINN y editar casos';
}

async function boot() {
  $$('#tabs button').forEach((b) => b.addEventListener('click', () => goTab(b.dataset.tab)));
  try {
    const [info, options, materials] = await Promise.all([api('/api/info'), api('/api/options'), api('/api/materials')]);
    S.info = info; S.options = options; S.materials = materials;
  } catch (e) {
    toast('No se pudo hablar con el servidor: ' + e.message, 'bad');
  }
  setFemPill();
  try { await loadCases(); } catch (e) { toast(e.message, 'bad'); }
  initCases(); initRun(); initCompare(); initHelp();
  if (typeof drawRunLeft === 'function') drawRunLeft();
  pollNow();
  const want = (location.hash || '').replace('#', '');
  if (['casos', 'ejecucion', 'comparar', 'ayuda'].includes(want)) goTab(want);
  window.addEventListener('beforeunload', (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } });
}

boot();