import re, glob, os, sys, collections
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')

def clean(t):
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\\begin\{(table|figure|longtable|tabularx|tikzpicture|formula|equation|align|align\*|equation\*|lstlisting|landscape)\*?\}.*?\\end\{\1\*?\}', ' ', t, flags=re.S)
    t = re.sub(r'\$[^$]*\$', ' X ', t)
    t = re.sub(r'\\(?:parencites?|textcites?|cites?|autocite|nocite)\*?(?:\s*\[[^\]]*\])*\s*\{[^}]*\}(?:\s*(?:\[[^\]]*\])*\s*\{[^}]*\})*', ' ', t)
    t = re.sub(r'\\(?:label|ref|autoref|input|include|includegraphics|caption|hypersetup|newcommand|renewcommand|setlength|SI|si|num|pct)\*?(?:\[[^\]]*\])?\{[^}]*\}(?:\{[^}]*\})?', ' ', t)
    t = re.sub(r'\\(?:chapter|section|subsection|subsubsection|paragraph)\*?\{[^}]*\}', '\n\n', t)
    t = re.sub(r'\\(?:textbf|textit|emph|textcolor|texttt|underline)\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\item', '\n\n', t)
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t)
    t = re.sub(r'[{}~\\]', ' ', t)
    return t

def sentences(par):
    par = re.sub(r'\s+', ' ', par).strip()
    ss = re.split(r'(?<=[.!?;:])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])', par)
    return [s for s in ss if len(s.split()) > 3]

EVAL = r'\b(importante|importantes|fundamental|fundamentales|crítico|crítica|críticos|novedoso|novedosa|robusto|robusta|robustos|eficiente|eficientes|óptimo|óptima|significativo|significativa|significativos|relevante|relevantes|clave|excelente|adecuado|adecuada|adecuados|notable|notables|sólido|sólida|potente|poderoso|innovador|innovadora|drástic[oa]s?|enorme|gran)\b'
FIRST = r'\b(nosotros|nuestro|nuestra|nuestros|nuestras|estamos|hemos|podemos|vamos|veremos|analizamos|presentamos|proponemos|desarrollamos|consideramos|utilizamos|usamos|yo)\b'
SELF = r'(en esta tesis|esta tesis|el presente estudio|la presente investigación|el presente trabajo|en este trabajo|este trabajo)'
CONV = r'\b(básicamente|obviamente|en realidad|de alguna manera|por supuesto|claramente|simplemente|muy|bastante|cosas|etc\.)\b'

def analyze(files, label):
    print(f'===== {label} =====')
    tot_words = tot_s = long45 = long90 = 0
    allpar = []
    ev = collections.Counter(); fp = collections.Counter(); sf = collections.Counter(); cv = collections.Counter()
    rows = []
    for f in files:
        raw = open(f, encoding='utf-8', errors='replace').read()
        t = clean(raw)
        pars = [p for p in re.split(r'\n\s*\n', t) if len(p.split()) > 8]
        w = s45 = s90 = ns = 0
        oneS = 0
        for p in pars:
            ss = sentences(p)
            if len(ss) == 1: oneS += 1
            for s in ss:
                n = len(s.split()); w += n; ns += 1
                if n > 45: s45 += 1
                if n > 90: s90 += 1
            allpar.append((f, len(ss), len(p.split())))
        low = t.lower()
        for m in re.finditer(EVAL, low): ev[m.group(1)] += 1
        for m in re.finditer(FIRST, low): fp[m.group(1)] += 1
        for m in re.finditer(SELF, low): sf[m.group(1)] += 1
        for m in re.finditer(CONV, low): cv[m.group(1)] += 1
        rows.append((os.path.relpath(f), len(pars), ns, w, s45, s90, oneS))
        tot_words += w; tot_s += ns; long45 += s45; long90 += s90
    print(f'{"archivo":70s} par  ora  pal  >45  >90  parr1ora')
    for r in rows: print(f'{r[0]:70s} {r[1]:4d} {r[2]:4d} {r[3]:5d} {r[4]:4d} {r[5]:4d} {r[6]:5d}')
    print(f'TOTAL oraciones {tot_s}, palabras {tot_words}, prom {tot_words/max(tot_s,1):.1f} pal/oración; >45: {long45} ({100*long45/max(tot_s,1):.1f}%), >90: {long90}')
    print('párrafos (n):', len(allpar), '| con >4 oraciones:', sum(1 for _, n, _ in allpar if n > 4), '| con 1 oración:', sum(1 for _, n, _ in allpar if n == 1), '| >180 palabras:', sum(1 for _, _, w in allpar if w > 180))
    print('adjetivos valorativos:', dict(ev.most_common(15)))
    print('primera persona:', dict(fp))
    print('autorreferencia:', dict(sf))
    print('coloquial:', dict(cv))

def find_long(files, top=8):
    out = []
    for f in files:
        t = clean(open(f, encoding='utf-8', errors='replace').read())
        for p in re.split(r'\n\s*\n', t):
            for s in sentences(p):
                n = len(s.split())
                if n > 60: out.append((n, os.path.relpath(f), s[:230]))
    out.sort(reverse=True)
    for n, f, s in out[:top]: print(f'  [{n} pal] {f}: {s}...')

if __name__ == '__main__':
    th = sorted(glob.glob('Tesis_LaTeX_Borrador_UNI/capitulos/0*.tex'))
    analyze(th, 'TESIS (capitulos)')
    find_long(th)
    from plan_fuentes import archivos as plan_tex
    pl = plan_tex(relativos=True)
    analyze(pl, 'PLAN')
    find_long(pl)
    # siglas
    for label, files in [('TESIS', th), ('PLAN', pl)]:
        txt = '\n'.join(clean(open(f, encoding='utf-8', errors='replace').read()) for f in files)
        raw = '\n'.join(open(f, encoding='utf-8', errors='replace').read() for f in files)
        acr = ['PINN', 'FEM', 'XLPE', 'IEC', 'CIGRÉ', 'DSR', 'PDE', 'NRMSE', 'SEIN', 'PAC', 'MMS', 'FEniCSx', 'RMSE', 'DTR', 'IEEE', 'COES', 'MINEM', 'AC', 'DC', 'FNO', 'PDE']
        print(f'--- siglas {label}: primera aparición / definición (patrón "(SIGLA)" o "SIGLA (") ---')
        for a in dict.fromkeys(acr):
            m = re.search(r'\b' + re.escape(a) + r'\b', txt)
            d = re.search(r'\(\s*' + re.escape(a) + r'\s*\)', txt)
            print(f'  {a:8s} primera pos {m.start() if m else None} | definición "(sigla)" pos {d.start() if d else None}')
        it = re.findall(r'\\textit\{([^}]*)\}', raw)
        print('  \\textit{...} :', collections.Counter(it).most_common(25))
