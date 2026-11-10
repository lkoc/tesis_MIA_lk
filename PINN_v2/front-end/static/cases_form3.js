'use strict';
// Pestana Casos (4/6): fuente electrica y condiciones de borde.

function autoPower() {
  const p = S.doc.physical;
  if (CV.powerAuto) p.power = (p.current || 0) ** 2 * (p.R20 || 0);
}

function secSource() {
  const p = S.doc.physical;
  const r0 = p.layers && p.layers[0] ? p.layers[0][1] : 0;
  const area = Math.PI * r0 * r0;
  const conductors = matGroup('conductor');
  return section('Fuente de calor (efecto Joule)', 'source', [
    h('div', { class: 'grid4' },
      field('Corriente I (A)', num(['physical', 'current'], { structural: true, after: autoPower })),
      field('R20 (ohm/m a 20 degC)', num(['physical', 'R20'], { structural: true, after: autoPower })),
      field('alpha (1/K)', num(['physical', 'alpha']), '(coef. de temperatura)'),
      field('Potencia p\' (W/m)', num(['physical', 'power'], { after: () => { CV.powerAuto = false; } }), CV.powerAuto ? '(= I^2 R20)' : '(manual)')),
    h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: CV.powerAuto, onchange: (e) => { CV.powerAuto = e.target.checked; autoPower(); changed(true); } }), 'Calcular la potencia como I^2 por R20'),
    h('div', { class: 'row' }, h('span', { class: 'muted small' }, 'Conductor tipico (fija alpha y R20 con la seccion actual):'), conductors.map((m) =>
      h('button', { class: 'chip', type: 'button', title: m.note || '', onclick: () => {
        p.alpha = Number(m.alpha);
        if (area > 0) p.R20 = Number(m.rho20) / area;
        if (p.layers && p.layers[0]) p.layers[0][2] = m.k;
        autoPower();
        changed(true);
      } }, m.name))),
    h('div', { class: 'muted small' }, 'Q = potencia / area del conductor se reparte de forma uniforme en la capa 0; la integral de Q sobre el area se comprueba al validar.')], 'Potencia por unidad de longitud', true);
}

const BC_DEFAULT = {
  dirichlet: () => ({ type: 'dirichlet', temperature: { value: S.doc.physical.T0, unit: 'degC' } }),
  neumann: () => ({ type: 'neumann', heat_flux_out: { value: 0, unit: 'W/m2' } }),
  robin: () => ({ type: 'robin', h: { value: 14.605, unit: 'W/(m2 K)' }, T_inf: { value: S.doc.physical.T0, unit: 'degC' } })
};

function bcSides() { return S.doc.physical.outer_radius ? ['outer'] : ['bottom', 'top', 'left', 'right']; }

function bcPreset(name) {
  const bc = {};
  for (const s of bcSides()) bc[s] = BC_DEFAULT.dirichlet();
  if (name === 'robin_top' && bc.top) bc.top = BC_DEFAULT.robin();
  if (name === 'adiabatic_sides') { if (bc.left) bc.left = BC_DEFAULT.neumann(); if (bc.right) bc.right = BC_DEFAULT.neumann(); }
  S.doc.boundary_conditions = bc;
  changed(true);
}

function bcSide(side) {
  const bc = S.doc.boundary_conditions[side] || BC_DEFAULT.dirichlet();
  S.doc.boundary_conditions[side] = bc;
  const base = ['boundary_conditions', side];
  const kind = h('select', { onchange: (e) => { S.doc.boundary_conditions[side] = BC_DEFAULT[e.target.value](); changed(true); } },
    [['dirichlet', 'Dirichlet (temperatura fija)'], ['neumann', 'Neumann (flujo prescrito)'], ['robin', 'Robin (conveccion)']].map(([v, l]) => h('option', { value: v, selected: bc.type === v }, l)));
  let body;
  if (bc.type === 'dirichlet') body = field('Temperatura (degC)', num(base.concat(['temperature', 'value'])));
  else if (bc.type === 'neumann') body = field('Flujo de calor saliente (W/m2)', num(base.concat(['heat_flux_out', 'value'])), '(0 = adiabatico)');
  else body = [
    h('div', { class: 'grid2' }, field('h (W/m2 K)', num(base.concat(['h', 'value']), { min: 0 })), field('T ambiente (degC)', num(base.concat(['T_inf', 'value'])))),
    h('div', { class: 'row' }, h('span', { class: 'muted small' }, 'h tipicos:'), (S.materials.convection.items || []).map((c) =>
      h('button', { class: 'chip', type: 'button', title: c.note || '', onclick: () => { S.doc.boundary_conditions[side].h.value = c.h; changed(true); } }, c.name + ' (' + c.h + ')')))];
  return h('div', { class: 'card pad' }, h('div', { class: 'row' }, h('b', {}, SIDES[side] || side), kind), body);
}

function secBC() {
  const bcs = S.doc.boundary_conditions;
  const body = [];
  if (!bcs) {
    body.push(banner('info', 'Este caso usa la condicion de borde heredada (Dirichlet T0 implicito). Escribela de forma explicita para poder editarla.'),
      h('button', { class: 'btn', type: 'button', onclick: () => bcPreset('dirichlet') }, 'Escribir Dirichlet T0 explicito'));
  } else {
    body.push(h('div', { class: 'row' }, h('span', { class: 'muted small' }, 'Atajos:'),
      iconBtn('Todo Dirichlet T0', 'Temperatura fija en todos los bordes', () => bcPreset('dirichlet')),
      iconBtn('Superficie con conveccion', 'Robin arriba, Dirichlet en el resto', () => bcPreset('robin_top')),
      iconBtn('Laterales adiabaticos', 'Neumann 0 a los lados, Dirichlet arriba y abajo', () => bcPreset('adiabatic_sides'))),
      h('div', { class: 'grid2' }, bcSides().map(bcSide)),
      h('div', { class: 'muted small' }, 'Se necesita al menos un borde Dirichlet o Robin. Robin nunca se convierte a Dirichlet.'));
  }
  return section('Condiciones de borde', 'bc', body, bcs ? Object.values(bcs).map((v) => v.type).join(' / ') : 'heredado', true);
}