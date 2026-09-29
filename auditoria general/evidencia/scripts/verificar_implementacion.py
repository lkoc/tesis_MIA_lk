# -*- coding: utf-8 -*-
"""Verifica de forma independiente que las recomendaciones R-01..R-40 de la auditoria general se aplicaron.

Solo lee archivos y los PDF ya compilados en la copia temporal; no compila ni modifica nada.
Estados: OK, PARCIAL, FALLA, PEND (pendiente por diseno o por decision del usuario).
Uso: python verificar_implementacion.py
"""
import os, re, sys, json, glob, subprocess, hashlib
from urllib.parse import unquote
import fitz, openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bibcheck as b
import plan_fuentes as pf

ROOT = r'c:\usr\ths_mia_fiis\tesis_MIA_lk'
TMP = os.path.join(os.environ.get('TEMP', ''), 'ths_build')
PLAN_DIR = os.path.join(ROOT, 'Plan')
TES_DIR = os.path.join(ROOT, 'Tesis_LaTeX_Borrador_UNI')
PLAN = os.path.join(PLAN_DIR, 'plan_tesis_cables_pinn.tex')
TES = os.path.join(TES_DIR, 'tesis.tex')
CAPS = os.path.join(TES_DIR, 'capitulos')
TABS = os.path.join(TES_DIR, 'tablas')


def rd(p):
    return open(p, encoding='utf-8', errors='replace').read().replace('\r\n', '\n')


def nocom(t):
    return re.sub(r'(?m)(?<!\\)%.*$', '', t)


def cap(n):
    return rd(os.path.join(CAPS, n))


plan, tes = pf.expandido(PLAN), rd(TES)
c00i, c00r = cap('00_introduccion.tex'), cap('00_resumen_abstract.tex')
c01, c02, c03, c04, c05 = (cap(f) for f in ('01_planteamiento_del_problema.tex', '02_marco_teorico.tex',
                                              '03_propuesta_y_desarrollo.tex', '04_conclusiones_y_recomendaciones.tex', '05_anexos.tex'))
tabs = {os.path.basename(p): rd(p) for p in glob.glob(os.path.join(TABS, '*.tex'))}
R = []


def add(i, estado, det):
    R.append((i, estado, det))


def has(t, *ss):
    return [s for s in ss if s not in t]


def captions(text):
    text = nocom(text)
    for m in re.finditer(r'\\caption(?:\[[^\]]*\])?\{', text):
        i, d = m.end(), 1
        while d and i < len(text):
            d += (text[i] == '{') - (text[i] == '}')
            i += 1
        yield text[m.end():i - 1]


def sin_fuente(*texts):
    return sum(1 for t in texts for c in captions(t) if 'Fuente' not in c)


# ---- PDF compilados en la copia temporal y PDF rev2
pdf_tes = os.path.join(TMP, 'Tesis_LaTeX_Borrador_UNI', 'tesis.pdf')
pdf_plan = os.path.join(TMP, 'Plan', 'plan_tesis_cables_pinn.pdf')
pdf_rev2 = os.path.join(PLAN_DIR, 'Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev2.pdf')
n_tes = len(fitz.open(pdf_tes)) if os.path.exists(pdf_tes) else None
n_plan = len(fitz.open(pdf_plan)) if os.path.exists(pdf_plan) else None
toc = fitz.open(pdf_rev2).get_toc() if os.path.exists(pdf_rev2) else []
toc_t = [t.strip().upper() for _, t, _ in toc]


def newest(paths):
    return max(os.path.getmtime(p) for p in paths)


src_tes = [TES] + glob.glob(os.path.join(CAPS, '*.tex')) + glob.glob(os.path.join(TABS, '*.tex')) + [os.path.join(TES_DIR, 'referencias.bib')]


def _h(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.exists(p) else None


# el PDF corresponde a las fuentes si son identicas por hash a la copia temporal que se compilo
fuentes_mas_nuevas = [os.path.basename(p) for p in src_tes
                      if _h(p) != _h(os.path.join(TMP, 'Tesis_LaTeX_Borrador_UNI', os.path.relpath(p, TES_DIR)))]
frescura_tes = os.path.exists(pdf_tes) and not fuentes_mas_nuevas

# ---- R-01
pen = rd(os.path.join(TES_DIR, 'PENDIENTES.md'))
falt = [d for d in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6') if d not in pen]
add('R-01', 'PARCIAL' if not falt else 'FALLA', 'decisiones D1-D6 registradas en PENDIENTES.md; falta el acta con el asesor' if not falt else f'faltan {falt}')

# ---- R-02
falt = has(plan, '\\chapter{ADMINISTRACIÓN DEL PLAN DE TESIS}', '\\section{CRONOGRAMA}', '\\section{PRESUPUESTO}', '\\section{FINANCIAMIENTO}',
           'autofinanciad', 'a) Equipos', 'b) Materiales e insumos', 'c) Servicios', 'd) Viáticos y publicación', 'Jul--', '2027')
gantt = re.findall(r'\\gc|\\gp', plan)
pc = plan.count('Por confirmar')
add('R-02', 'OK' if not falt else 'FALLA', f'cap. V con Gantt ({len(gantt)} celdas marcadas), rubros a-d y financiamiento; montos "Por confirmar": {pc}' if not falt else f'faltan {falt}')

# ---- R-03
def idx(s):
    return plan.find(s)
orden = [idx('\\chapter{PLANTEAMIENTO DEL PROBLEMA}'), idx('\\chapter{OBJETIVOS DE LA INVESTIGACIÓN}'), idx('\\chapter{MARCO TEÓRICO}'),
         idx('\\chapter{METODOLOGÍA DE LA INVESTIGACIÓN}'), idx('\\chapter{ADMINISTRACIÓN DEL PLAN DE TESIS}'), idx('\\printbibliography'), idx('\\chapter*{ANEXOS}')]
ok = all(o >= 0 for o in orden) and orden == sorted(orden) and 'heading=bibintoc' in plan and 'ANEXO \\arabic{section}' in plan
toc_ok = [t for t in ('DATOS GENERALES', 'PLANTEAMIENTO DEL PROBLEMA', 'OBJETIVOS DE LA INVESTIGACIÓN', 'MARCO TEÓRICO', 'METODOLOGÍA DE LA INVESTIGACIÓN',
                      'ADMINISTRACIÓN DEL PLAN DE TESIS', 'REFERENCIAS', 'ANEXOS') if t in toc_t]
add('R-03', 'OK' if ok and len(toc_ok) == 8 else 'FALLA', f'orden de capitulos correcto en .tex: {ok}; entradas del indice del PDF rev2 en el orden de la Guia: {len(toc_ok)}/8')

# ---- R-04
falt = has(plan, '\\section{DIAGNÓSTICO}', '\\section{IDENTIFICACIÓN Y DESCRIPCIÓN DEL PROBLEMA DE ESTUDIO}', '\\subsection{ANTECEDENTES BIBLIOGRÁFICOS}',
           '\\subsection{FORMULACIÓN DEL PROBLEMA}', '\\subsection{JUSTIFICACIÓN Y ALCANCES}')
add('R-04', 'OK' if not falt else 'FALLA', 'Cap. I = 1.1 Diagnostico; 1.2 Identificacion (1.2.1, 1.2.2, 1.2.3)' if not falt else f'faltan {falt}')

# ---- R-05
tit = ['TIPO DE INVESTIGACIÓN', 'NIVEL DE INVESTIGACIÓN', 'MÉTODOS DE TRABAJO', 'POBLACIÓN Y MUESTRA', 'TIPO DE DISEÑO',
       'TÉCNICAS E INSTRUMENTOS DE RECOLECCIÓN DE DATOS', 'TÉCNICAS E INSTRUMENTOS DE ANÁLISIS Y PROCESAMIENTO DE DATOS', 'ETAPAS DE INTERVENCIÓN DEL ESTUDIO']
falt = [t for t in tit if '\\section{%s}' % t not in plan]
add('R-05', 'OK' if not falt else 'FALLA', '8 rotulos 4.1-4.8 presentes' if not falt else f'faltan {falt}')

# ---- R-06
f1 = has(plan, 'Asesor & Samuel Oporto', 'Fecha de inicio', 'Fecha de culminación prevista')
f2 = has(tes, 'Asesor & Samuel Oporto', 'Área involucrada', 'Fecha de inicio', 'Fecha de finalización prevista', 'Duración')
add('R-06', 'OK' if not f1 and not f2 else 'FALLA', f'Plan faltan {f1}; tesis faltan {f2}' if (f1 or f2) else 'Datos generales completos en Plan y tesis')

# ---- R-07
def years(t):
    m = re.search(r'tab:antecedentes-cronologia.*?\\end\{longtable\}', t, re.S)
    return [int(y) for y in re.findall(r'(?m)^(\d{4}) &', m.group(0))] if m else []
yp, yt = years(plan), years(c02)
ok = yp == sorted(yp) and yt == sorted(yt) and len(yp) > 10 and len(yt) > 10 and 'UBICACIÓN DEL PROYECTO' in plan and 'UBICACIÓN DEL PROYECTO' in c02
add('R-07', 'OK' if ok else 'FALLA', f'tabla cronologica ordenada: Plan {len(yp)} fuentes ({min(yp) if yp else "-"}-{max(yp) if yp else "-"}), tesis {len(yt)}; subseccion Ubicacion en ambos: {"UBICACIÓN DEL PROYECTO" in plan and "UBICACIÓN DEL PROYECTO" in c02}')

# ---- R-08
hs = {}
for ln in rd(os.path.join(PLAN_DIR, 'rev2_SHA256.txt')).splitlines():
    m = re.match(r'([0-9a-f]{64})\s+(\S+)', ln)
    if m:
        hs[m.group(2)] = m.group(1)
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
desc = [k for k, v in hs.items() if sha(os.path.join(PLAN_DIR, k)) != v]
arch = os.path.exists(os.path.join(PLAN_DIR, 'old', 'versiones_pdf', 'Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev0.pdf')) and os.path.isdir(os.path.join(PLAN_DIR, 'old', 'compilacion_2026-08-18'))
add('R-08', ('OK' if arch else 'PARCIAL') if not desc else 'FALLA', ('rev2.pdf y sus fuentes coinciden con los hashes registrados; ' + ('PDF previos archivados en Plan/old/' if arch else 'faltan archivar los PDF previos') if not desc else f'hash distinto: {desc}') + f'; PDF de {len(fitz.open(pdf_rev2))} paginas')

# ---- R-09
add('R-09', 'OK' if ('\\poppins\\bfseries\\fontsize{22}' in plan and 'Poppins-Bold.ttf' in plan and os.path.exists(os.path.join(PLAN_DIR, 'fonts', 'Poppins-Bold.ttf'))) else 'FALLA', 'caratula del Plan con Poppins 22 y Arial 16/14/12')

# ---- R-10
ent, _ = b.parse_bib(os.path.join(PLAN_DIR, 'referencias_plan_tesis.bib'))
cited = set(b.cited_keys(pf.archivos()))
sin_citar = sorted(k for k in ent if k not in cited)
drive_bib = [p for p in (os.path.join(PLAN_DIR, 'referencias_plan_tesis.bib'), os.path.join(TES_DIR, 'referencias.bib')) if 'drive.google' in rd(p)]
drive_tex = plan.count('drive.google')
lab = ['fig:modelo-conceptual', 'fig:sintesis-metodologica-dsr', 'tab:dimensiones-operativas', 'tab:operacionalizacion-variables', 'tab:diseno-casos']
sin_llamada = [l for l in lab if '\\ref{%s}' % l not in plan]
nf = sin_fuente(plan)
ok = not sin_citar and not drive_bib and not sin_llamada and nf == 0
det = f'entradas sin citar: {len(sin_citar)}; .bib con URL de Drive: {len(drive_bib)}; etiquetas sin llamada: {len(sin_llamada)}; leyendas sin Fuente: {nf}; URL de Drive en el anexo del repositorio: {drive_tex}'
add('R-10', ('PARCIAL' if ok and drive_tex else 'OK' if ok else 'FALLA'), det)

# ---- R-11
wb = openpyxl.load_workbook(os.path.join(PLAN_DIR, 'evaluacion_formato_8_2_plan_tesis.xlsx'))
ws = wb['Formato 8.2']
marcas = sum(1 for r in range(1, ws.max_row + 1) if ws.cell(r, 4).value == 'X' or ws.cell(r, 5).value == 'X')
mojibake = sum(1 for w in wb.worksheets for row in w.iter_rows() for c in row if isinstance(c.value, str) and re.search(r'[A-Za-zÁ-ú]\?[A-Za-zÁ-ú]', c.value))
res = wb['Resumen']
add('R-11', 'OK' if marcas == 30 and mojibake == 0 and str(res['B7'].value).startswith('2026-09-29') else 'FALLA', f'criterios marcados: {marcas}/30; celdas con "?" por acento perdido: {mojibake}; Cumple={res["B5"].value}, No cumple={res["B6"].value}, fecha={res["B7"].value}')

# ---- R-12
prot = os.path.exists(os.path.join(ROOT, 'docs', 'PROTOCOLO_RONDA_E.md'))
falt = has(c03, 'sec:dominio-validez', 'tab:dominio-aceptacion')
add('R-12', 'PARCIAL' if not falt and prot else 'FALLA', 'dominio aceptado fijado en la tesis y protocolo E redactado; la ronda E no esta firmada ni ejecutada')

# ---- R-13
falt = has(c03, 'tab:explicit-D-diagnosis', 'tab:explicit-C-diagnosis')
add('R-13', 'PARCIAL' if not falt else 'FALLA', 'tablas de restriccion dominante de D y C/C2 en la tesis; el generador del informe aun no las produce')

# ---- R-14
add('R-14', 'PARCIAL' if ('no alcanzado' in c04 and 'OE4' in c04) else 'FALLA', 'OE4 declarado no alcanzado; las 9 corridas de ampacidad siguen pendientes')

# ---- R-15
sm = json.load(open(os.path.join(ROOT, 'Benchmarks', 'explicit_study', 'report', 'summary.json'), encoding='utf-8'))
man = rd(os.path.join(ROOT, 'Benchmarks', 'MANIFEST.json')) if os.path.exists(os.path.join(ROOT, 'Benchmarks', 'MANIFEST.json')) else ''
ok_sum = sm.get('total') == 101 and sm.get('production_total') == 21 and sm.get('production_accepted') == 7
add('R-15', 'PARCIAL' if ok_sum and 'explicit_study' not in man else 'OK' if ok_sum else 'FALLA', f'summary.json: total={sm.get("total")}, produccion {sm.get("production_accepted")}/{sm.get("production_total")}; MANIFEST.json cubre explicit_study: {"explicit_study" in man}')

# ---- R-16..R-21 (codigo/configuracion)
py = rd(os.path.join(ROOT, 'pyproject.toml'))
add('R-16', 'PEND' if 'Benchmarks/tests' not in py else 'OK', f'pyproject testpaths incluye Benchmarks/tests: {"Benchmarks/tests" in py}')
add('R-17', 'PEND' if not os.path.isdir(os.path.join(ROOT, '.github', 'workflows')) else 'OK', f'.github/workflows existe: {os.path.isdir(os.path.join(ROOT, ".github", "workflows"))}')
c_ru = 0
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in ('.git', 'explicit_study', 'auditoria general', 'estado_inicial', 'node_modules', '__pycache__')]
    for f in files:
        if f.endswith('.py'):
            try:
                if 'C:\\Users\\QU1267' in open(os.path.join(base, f), encoding='utf-8', errors='ignore').read():
                    c_ru += 1
            except OSError:
                pass
add('R-19', 'PEND' if c_ru else 'OK', f'archivos .py fuera de explicit_study con C:\\Users\\QU1267 fijo: {c_ru}')
add('R-20', 'PEND', 'no verificable sin medir el repositorio; no se toco')
linea_torch = re.search(r'torch[^\n]*', py)
linea_torch = linea_torch.group(0) if linea_torch else ''
add('R-21', 'PEND' if ('torch' in linea_torch and '<' not in linea_torch) else 'OK', 'rango de torch en pyproject: ' + (linea_torch.strip() or 'no aparece'))

# ---- R-18
falt = has(c03, 'tab:matriz-validez')
add('R-18', 'PARCIAL' if not falt else 'FALLA', 'matriz de validez en el cap. III; el contraste numerico con Aras/Kim sigue pendiente (exige FEM nuevo)')

# ---- R-22
def enlaces_rotos(files):
    rotos = []
    for f in files:
        base = os.path.dirname(f)
        for m in re.finditer(r'\]\(([^)\s]+)\)', rd(f)):
            u = m.group(1)
            if re.match(r'^(https?:|mailto:|#)', u):
                continue
            p = os.path.normpath(os.path.join(base, unquote(u.split('#')[0])))
            if not os.path.exists(p):
                rotos.append((os.path.relpath(f, ROOT), u))
    return rotos
docs_md = [os.path.join(ROOT, 'docs', n) for n in os.listdir(os.path.join(ROOT, 'docs')) if n.endswith('.md')]
md_all = docs_md + [os.path.join(ROOT, n) for n in ('README.md', 'ESTADO.md', 'LICENSE.md')] + [os.path.join(TES_DIR, 'PENDIENTES.md')]
rotos = enlaces_rotos(md_all)
bann = all('Documento histórico' in rd(os.path.join(ROOT, 'docs', n)) for n in ('AUDITORIA_TESIS.md', 'PLAN_AUDITORIA_TESIS.md'))
add('R-22', 'OK' if not rotos and bann else 'FALLA', f'banners historicos: {bann}; enlaces relativos rotos en docs/, README, ESTADO, LICENSE y PENDIENTES: {len(rotos)} {rotos[:3]}')

# ---- R-23
ej = sum(len(re.findall(r'en ejecuci[oó]n|in progress', t)) for t in [c00i, c00r, c01, c02, c03, c04, c05] + list(tabs.values()))
add('R-23', 'PARCIAL' if ej == 0 and 'tab:explicit-D-diagnosis' in c03 else 'FALLA', f'apariciones de "en ejecucion": {ej}; el generador sigue pudiendo sobrescribir 04 y las tablas')

# ---- R-24 / R-25
k = [i for i in range(1, 5) if f'Conclusión {i}' in c04 and f'Recomendación {i}' in c04]
add('R-24', 'OK' if k == [1, 2, 3, 4] else 'FALLA', f'pares conclusion-recomendacion presentes: {k}')
ok = '7 de 21' in c00r and 'tuvo como objetivo' in c00r and '\\input{tablas/explicit_abstract' not in c00r and 'Se concluyó' in c00r
add('R-25', 'OK' if ok else 'FALLA', 'resumen en pasado con resultados de D (7 de 21) y sin insertos generados')

# ---- R-26
ch = {'00_introduccion': c00i, '01': c01, '02': c02, '03': c03, '04': c04, '05': c05}
sin = [k2 for k2, t in ch.items() if '\\thispagestyle{empty}' not in t]
arab = tes.find('\\pagenumbering{arabic}')
introp = tes.find('\\input{capitulos/00_introduccion}')
add('R-26', 'OK' if not sin and 0 < arab < introp else 'FALLA', f'capitulos sin thispagestyle{{empty}}: {sin}; arabica antes de la introduccion: {0 < arab < introp}')

# ---- R-27
r = subprocess.run([sys.executable, '-X', 'utf8', os.path.join(HERE, 'etiquetas_sin_referencia.py')], capture_output=True, text=True, encoding='utf-8', errors='replace').stdout
det_t = re.search(r'TESIS (\{.*\})', r).group(1) if re.search(r'TESIS (\{.*\})', r) else '?'
det_p = re.search(r'PLAN (\{.*\})', r).group(1) if re.search(r'PLAN (\{.*\})', r) else '?'
bad = [k2 for k2 in ('tab', 'fig', 'for', 'anx') if (f"'{k2}'" in det_t) or (f"'{k2}'" in det_p)]
nf_t = sin_fuente(c01, c02, c03, c05, *tabs.values())
add('R-27', 'OK' if not bad and nf_t == 0 and nf == 0 else 'FALLA', f'etiquetas sin llamada (tab/fig/for/anx): {bad or 0}; leyendas sin Fuente: tesis {nf_t}, Plan {nf}')

# ---- R-28 / R-29 / R-30 / R-31
falt = has(c03, 'VIABILIDAD TÉCNICA Y COSTO COMPUTACIONAL', 'tab:costo-computacional', 'tab:matriz-riesgos', 'SOSTENIBILIDAD')
add('R-28', 'OK' if not falt else 'FALLA', 'costo computacional, matriz de riesgos, beneficio-costo y sostenibilidad (falta visto bueno del asesor)')
add('R-29', 'OK' if 'CRITERIOS DE ACEPTACIÓN Y PROPOSICIONES DE DISEÑO' in c01 and 'HIPÓTESIS Y CRITERIOS DE CONTRASTE' not in c01 else 'FALLA', 'seccion renombrada y justificada')
add('R-30', 'OK' if re.search(r'\\begin\{singlespace\}\s*\\printbibliography', tes) else 'FALLA', 'rechazada: referencias siguen en interlineado simple')
add('R-31', 'OK' if '\\setstretch{1.4}' in nocom(tes) and '\\onehalfspacing' not in nocom(tes) else 'FALLA', 'interlineado 1,4 adoptado en tesis.tex')

# ---- R-32
orden_txt = nocom(''.join([c00r, c00i, c01, c02, c03, c04]))
orden_txt = re.sub(r'\\(?:parencites?|textcites?|cites?)\*?(?:\[[^\]]*\])*\{[^}]*\}', '', orden_txt)
sig = {}
for a in ('FEM', 'XLPE', 'IEC', 'CIGRÉ', 'PDE', 'NRMSE', 'RMSE', 'SEIN'):
    m = re.search(r'(?<![\\\w{])%s(?![\w}])' % a, orden_txt)
    if not m:
        sig[a] = 'no aparece'
        continue
    v = orden_txt[max(0, m.start() - 160):m.end() + 12]
    sig[a] = 'definida' if re.search(r'\(\s*(?:\\textit\{[^}]*\},\s*)?%s\)|\(%s\)|,\s*%s\)' % (a, a, a), v) else 'SIN DEFINIR'
mal = [a for a, s in sig.items() if s == 'SIN DEFINIR']
frases = sum(len(re.findall(r'esta tesis|presente estudio|la presente tesis', t)) for t in [c00i, c01, c02, c03, c04])
add('R-32', 'PARCIAL' if not mal and frases == 0 else 'FALLA', f'siglas en primera mencion: {sig}; "esta tesis/presente estudio": {frases}; los 38 parrafos de una sola oracion no se revisaron')

# ---- R-33 .. R-40
pdte = sum(t.count('\\pendiente{') for t in (nocom(cap('00_dedicatoria.tex')), nocom(cap('00_agradecimientos.tex')), nocom(cap('00_copia_documentos.tex')), c01, c02, c03, c04, c05))
add('R-33', 'PEND', f'marcas \\pendiente activas: {pdte} (dedicatoria, agradecimientos y copia de documentos: datos de los tesistas)')
lic = rd(os.path.join(ROOT, 'LICENSE.md'))
add('R-34', 'OK' if lic.strip() != 'tbd' and 'MIT' in lic and 'CC BY' in lic else 'FALLA', 'LICENSE.md con MIT (codigo) y CC BY 4.0 (texto) e inventario de terceros por carpeta')
gi = open(os.path.join(ROOT, '.gitignore'), 'rb').read()
add('R-35', 'PEND' if (gi.startswith(b'\xef\xbb\xbf') or 'Ã'.encode('utf-8') in gi) else 'OK', f'.gitignore con BOM: {gi.startswith(b"\xef\xbb\xbf")}; mojibake: {"Ã".encode("utf-8") in gi}')
add('R-36', 'PEND' if os.path.exists(os.path.join(ROOT, '__cmd_probe.txt')) else 'OK', 'raiz con sondas y scripts sin ordenar: ' + str(os.path.exists(os.path.join(ROOT, '__cmd_probe.txt'))))
add('R-37', 'PEND', 'Plan y tesis siguen con .bib separados (referencias_plan_tesis.bib y referencias.bib)')
add('R-38', 'PEND', 'generadores de la matriz sin actualizar (scripts/generar_matriz_consistencia.py)')
est = os.path.exists(os.path.join(ROOT, 'ESTADO.md'))
enl = 'ESTADO.md' in rd(os.path.join(ROOT, 'README.md')) and 'ESTADO.md' in pen
add('R-39', 'PARCIAL' if est and enl else 'FALLA', 'ESTADO.md creado a mano y enlazado desde README y PENDIENTES; falta el generador')
add('R-40', 'PEND', 'informe de similitud: trámite del usuario')

# ---- Consistencia de cifras entre ESTADO.md y los PDF
estado = rd(os.path.join(ROOT, 'ESTADO.md'))
extra = []
extra.append(('Paginas de la tesis (PDF temporal)', n_tes, f'{n_tes} páginas con interlineado' in estado or '88 páginas' in estado and n_tes == 88))
extra.append(('Paginas del Plan (PDF rev2)', len(fitz.open(pdf_rev2)), '76 páginas' in estado and len(fitz.open(pdf_rev2)) == 76))
extra.append(('Entrenamientos y aceptacion (summary.json)', f'{sm.get("total")} / {sm.get("production_accepted")} de {sm.get("production_total")}', '101' in estado and '7 de 21' in estado))
extra.append(('Fuentes de la tesis identicas a las compiladas (hash)', frescura_tes, frescura_tes))

out = ['{:<6} {:<8} {}'.format(i, e, d) for i, e, d in sorted(R, key=lambda x: int(x[0][2:]))]
cont = {}
for _, e, _ in R:
    cont[e] = cont.get(e, 0) + 1
out.append('')
out.append('Resumen: ' + ', '.join(f'{k}={v}' for k, v in sorted(cont.items())) + f' de {len(R)} recomendaciones comprobadas (R-20 sin medir)')
out.append('')
out.append('Consistencia de cifras (ESTADO.md frente a artefactos):')
for n, v, ok in extra:
    out.append(f'  {"OK" if ok else "REVISAR":8} {n}: {v}')
if fuentes_mas_nuevas:
    out.append(f'  Fuentes de la tesis mas nuevas que el PDF compilado: {fuentes_mas_nuevas}')
txt = '\n'.join(out)
open(os.path.join(HERE, '..', 'salidas', 'verificacion_implementacion.txt'), 'w', encoding='utf-8').write(txt + '\n')
print(txt)
