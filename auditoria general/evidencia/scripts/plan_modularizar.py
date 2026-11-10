#!/usr/bin/env python3
"""Divide el Plan monolitico en archivo principal + capitulos/ + capitulos/anexos/ (mismo esquema que la tesis).

Lee la copia congelada Plan/old/plan_tesis_cables_pinn_monolitico_2026-09-29.tex, escribe los archivos nuevos y
verifica que, al expandir los \\input/\\include, se recupera exactamente la secuencia de lineas no vacias y sin
comentarios del monolito. No volver a ejecutar sobre un Plan ya dividido.
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PLAN = os.path.join(ROOT, 'Plan')
MONO = os.path.join(PLAN, 'old', 'plan_tesis_cables_pinn_monolitico_2026-09-29.tex')
CAP = os.path.join(PLAN, 'capitulos')
ANX = os.path.join(CAP, 'anexos')

raw = open(MONO, 'rb').read().decode('utf-8')
NL = '\r\n' if '\r\n' in raw else '\n'
L = raw.splitlines()  # sin terminadores; L[i-1] es la linea i del monolito


def seg(a, b):
    """Lineas a..b del monolito (1-based, inclusive)."""
    return L[a - 1:b]


def esperar(n, prefijo):
    assert L[n - 1].startswith(prefijo), f'linea {n}: se esperaba {prefijo!r}, hay {L[n - 1]!r}'


def escribir(ruta, lineas):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'wb') as f:
        f.write((NL.join(lineas) + NL).encode('utf-8'))


# --- verificacion de limites en el monolito ---
esperar(196, r'\begin{document}')
esperar(245, r'\chapter*{DATOS GENERALES}')
esperar(290, r'\endgroup')
esperar(292, r'% \chapter*{Resumen}')
esperar(303, r'\begingroup')
esperar(314, r'\pagenumbering{arabic}')
capitulos = [
    (317, 825, '01_planteamiento_del_problema', r'\chapter{PLANTEAMIENTO DEL PROBLEMA}'),
    (827, 1019, '02_objetivos', r'\chapter{OBJETIVOS DE LA INVESTIGACIÓN}'),
    (1021, 1482, '03_marco_teorico', r'\chapter{MARCO TEÓRICO}'),
    (1484, 1808, '04_metodologia', r'\chapter{METODOLOGÍA DE LA INVESTIGACIÓN}'),
    (1810, 1878, '05_administracion', r'\chapter{ADMINISTRACIÓN DEL PLAN DE TESIS}'),
]
for a, b, _, pref in capitulos:
    esperar(a, pref)
    assert L[a - 2].startswith('% ====='), f'falta separador antes de {a}'
    assert L[b].startswith('% ====='), f'falta separador despues de {b}'
esperar(1880, r'\begin{singlespace}')
esperar(1883, r'\end{singlespace}')
esperar(1886, r'\renewcommand{\thetable}')
esperar(1901, r'\begin{landscape}')
esperar(1902, r'\chapter*{ANEXOS}')
esperar(1908, r'\section{MATRIZ DE CONSISTENCIA}')
esperar(2071, r'\end{longtable}')
esperar(2072, r'\end{landscape}')
esperar(2077, r'\begin{landscape}')
esperar(2078, r'\section{MATRIZ DE OPERACIONALIZACIÓN')
esperar(2150, r'\end{longtable}')
esperar(2151, r'\end{landscape}')
esperar(2156, r'\begin{landscape}')
esperar(2157, r'\section{DISEÑO DISCRETO')
esperar(2239, r'\end{landscape}')
esperar(2243, r'\section{ESTRATEGIA DE REVISIÓN DOCUMENTAL}')
esperar(2316, r'\end{landscape}')
esperar(2319, r'\end{singlespace}')
esperar(2322, r'% Requiere en el pre')
esperar(2335, r'\section{REPOSITORIO DIGITAL DEL INFORME}')
esperar(2372, r'\end{document}')

# --- capitulos ---
for a, b, nombre, _ in capitulos:
    escribir(os.path.join(CAP, nombre + '.tex'), seg(a, b))

# --- datos generales ---
escribir(os.path.join(CAP, '00_datos_generales.tex'), seg(245, 290))

# --- anexos ---
anexos = [
    ('anexo_1_matriz_consistencia', 1908, 2071),
    ('anexo_2_operacionalizacion_variables', 2078, 2150),
    ('anexo_3_diseno_casos', 2157, 2238),
    ('anexo_4_revision_documental', 2243, 2316),
    ('anexo_5_repositorio_digital', 2335, 2371),
]
for nombre, a, b in anexos:
    lineas = seg(a, b)
    if nombre.startswith('anexo_5'):
        lineas = seg(2322, 2324) + [''] + lineas
    escribir(os.path.join(ANX, nombre + '.tex'), lineas)

I = lambda n: r'\input{capitulos/anexos/%s}' % n
anexos_tex = (
    ['% Anexos del Plan. Cada anexo vive en capitulos/anexos/; aqui solo se controla orientacion y numeración.']
    + seg(1886, 1907)
    + [I(anexos[0][0])]
    + seg(2072, 2077)
    + [I(anexos[1][0])]
    + seg(2151, 2156)
    + [I(anexos[2][0])]
    + seg(2239, 2242)
    + ['% El anexo 4 alterna vertical y horizontal internamente.', I(anexos[3][0])]
    + seg(2317, 2321)
    + seg(2325, 2333)
    + [I(anexos[4][0])]
)
escribir(os.path.join(CAP, '06_anexos.tex'), anexos_tex)

# --- comentario historico del resumen (no es parte del documento) ---
escribir(os.path.join(PLAN, 'old', 'resumen_ejecutivo_comentado.tex'), seg(292, 301))

# --- archivo principal ---
principal = (
    seg(1, 244)
    + [r'\input{capitulos/00_datos_generales}', '']
    + seg(303, 315)
    + ['% Capítulos del Plan: un archivo por capítulo en capitulos/.']
)
for a, b, nombre, _ in capitulos:
    principal.append(r'\include{capitulos/%s}' % nombre)
principal += [''] + seg(1879, 1884) + [
    r'\include{capitulos/06_anexos}',
    '',
    r'\end{document}',
]
escribir(os.path.join(PLAN, 'plan_tesis_cables_pinn.tex'), principal)


# --- verificacion: expandir \input/\include y comparar con el monolito ---
def expandir(ruta):
    out = []
    for ln in open(ruta, 'rb').read().decode('utf-8').splitlines():
        m = re.match(r'\\(?:input|include)\{(capitulos/[^}]+)\}\s*$', ln)
        out.extend(expandir(os.path.join(PLAN, m.group(1) + '.tex')) if m else [ln])
    return out


def norm(lineas):
    return [x.rstrip() for x in lineas if x.strip() and not x.lstrip().startswith('%')]


a = norm(L)
b = norm(expandir(os.path.join(PLAN, 'plan_tesis_cables_pinn.tex')))
if a != b:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print('primera diferencia en la posicion', i, repr(x[:120]), repr(y[:120]))
            break
    print('longitudes', len(a), len(b))
    sys.exit(1)
print('OK: la expansion coincide con el monolito;', len(a), 'lineas de contenido')
