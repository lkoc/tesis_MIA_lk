'use strict';
// Pestana Ayuda: guia rapida, la documentacion HTML local, la ayuda de la linea de comandos y datos del entorno.
const HP = { view: 'guia', page: 'index.html', topic: null, docs: null, topics: null };
const GUIDE = [
  ['1 - Casos', 'Elige un caso de catalogo (solo lectura) o de Mis casos. Duplicalo para editarlo: geometria, capas, materiales, fuente, bordes y contacto. Los datos tipicos de materiales estan en la lista desplegable. Guarda siempre en una carpeta tuya (user_cases), nunca sobre el catalogo.'],  ['2 - Ejecucion', 'Elige FEM (FEniCSx en WSL2) o una arquitectura PINN (M0, M1, M2, CUSP, DCSNN, FBPINN, con atencion RBA o SA). Los preajustes rapido, estandar y detallado fijan pasos y tamano de red. Se muestra la geometria y el campo de temperatura se refresca durante el calculo.'],
  ['3 - Comparar', 'Elige un caso con varias soluciones, marca una referencia (normalmente el FEM mas fino) y una o varias candidatas. Veras mapas de error con signo, absoluto o relativo, perfiles y una tabla de metricas con el error por region. El FEM nunca se usa para entrenar: solo sirve de referencia al comparar.'],
  ['4 - Ayuda', 'Aqui: la documentacion completa (la misma que en docs/_build/html) y la ayuda de la linea de comandos.'],
];
const TIPS = [
  'Los trabajos son procesos aparte: puedes cerrar la pestana y volver; el progreso se conserva en front-end/runs.',
  'Se ejecutan como maximo 2 trabajos a la vez; el resto espera tu siguiente intento.',
  'Un PINN no admite contacto imperfecto en esta version; el FEM si.',
  'Las soluciones guardadas en PINN_v2/results tambien aparecen en Comparar, junto a las nuevas.',
  'Un caso de cable tiene su potencia como p = I^2 R; la fuente es Q = p/A en el conductor y se comprueba su integral.',
];

function helpMenu() {
  const btn = (label, on, fn) => h('button', { class: on ? 'on' : '', onclick: fn }, label);
  const docs = (HP.docs && HP.docs.pages) || [];
  return [
    h('h4', {}, 'Empezar'),
    btn('Guia rapida', HP.view === 'guia', () => { HP.view = 'guia'; drawHelp(); }),
    btn('Entorno', HP.view === 'env', () => { HP.view = 'env'; drawHelp(); }),
    h('h4', {}, 'Documentacion'),
    docs.length ? docs.map((pg) => btn(pg.replace('.html', ''), HP.view === 'doc' && HP.page === pg, () => { HP.view = 'doc'; HP.page = pg; drawHelp(); })) : h('div', { class: 'muted small', style: 'padding:4px 12px' }, 'Sin compilar: ejecuta docs/build_docs.py'),
    h('h4', {}, 'Linea de comandos'),
    (HP.topics || []).map((t) => btn(t, HP.view === 'cli' && HP.topic === t, () => { HP.view = 'cli'; HP.topic = t; drawHelp(); })),
  ];
}
function helpGuide() {
  return h('div', { class: 'doc card pad' }, h('h2', {}, 'Guia rapida'),
    GUIDE.map(([t, x]) => h('div', {}, h('h3', {}, t), h('p', {}, x))),
    h('h3', {}, 'Consejos'), h('ul', {}, TIPS.map((t) => h('li', {}, t))),
    h('p', {}, h('button', { class: 'btn sm', onclick: () => { HP.view = 'doc'; HP.page = 'index.html'; drawHelp(); } }, 'Abrir la documentacion completa')));
}

function helpEnv() {
  const i = S.info || {};
  const fem = i.fem || {};
  return h('div', { class: 'doc card pad' }, h('h2', {}, 'Entorno'),
    h('div', { class: 'kv' }, h('b', {}, 'FEM (FEniCSx)'), h('span', {}, fem.available ? 'listo' : 'no disponible'),
      h('b', {}, 'Distribucion WSL2'), h('span', {}, fem.distro || '-'), h('b', {}, 'Python de FEniCSx'), h('span', { class: 'mono' }, fem.python || '-'),
      h('b', {}, 'Python (servidor)'), h('span', { class: 'mono' }, i.python || '-'), h('b', {}, 'Carpeta raiz'), h('span', { class: 'mono' }, i.root || '-'),
      h('b', {}, 'Mis casos'), h('span', { class: 'mono' }, i.user_cases || '-'), h('b', {}, 'Trabajos'), h('span', { class: 'mono' }, i.runs || '-'),
      h('b', {}, 'Trabajos simultaneos'), h('span', {}, String(i.max_running || '-'))));
}
async function drawHelp() {
  const root = $('#helpRoot');
  if (!root) return;
  let body;
  if (HP.view === 'guia') body = helpGuide();
  else if (HP.view === 'env') body = helpEnv();
  else if (HP.view === 'doc') {
    const src = '/docs/' + HP.page;
    body = h('div', { style: 'height:100%;display:flex;flex-direction:column;gap:6px' },
      h('div', { class: 'row', style: 'justify-content:flex-end' }, h('a', { class: 'btn ghost sm', href: src, target: '_blank' }, 'Abrir en otra pestana')),
      h('iframe', { class: 'docframe', src, style: 'flex:1' }));
  } else if (HP.view === 'cli') {
    body = h('div', { class: 'doc' }, h('h2', {}, 'pinn_v2 ' + HP.topic), h('pre', {}, 'Cargando...'));
    api('/api/help?' + qs({ topic: HP.topic })).then((r) => { const pre = $('pre', body); if (pre) pre.textContent = r.text; }).catch((e) => { const pre = $('pre', body); if (pre) pre.textContent = e.message; });
  }
  mount(root, h('div', { class: 'menu card pad', style: 'overflow:auto' }, helpMenu()), h('div', { style: 'min-height:0;overflow:auto' }, body));
}

async function enterHelp() {
  if (!HP.docs) {
    try { HP.docs = await api('/api/docs'); } catch (e) { HP.docs = { available: false, pages: [] }; }
    try { HP.topics = (await api('/api/help')).topics; } catch (e) { HP.topics = []; }
  }
  drawHelp();
}

// Para abrir una pagina concreta desde otras pestanas.
S.openDoc = async (page) => { HP.view = 'doc'; HP.page = page || 'index.html'; goTab('ayuda'); };

function initHelp() {
  mount($('#tab-ayuda'), h('div', { class: 'help', id: 'helpRoot', style: 'height:calc(100vh - 96px)' }));
  S.onTab.ayuda = enterHelp;
}