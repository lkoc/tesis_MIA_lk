import re, glob, os
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
def load(files):
    t = ''
    for f in files:
        t += open(f, encoding='utf-8', errors='replace').read() + '\n'
    return re.sub(r'(?<!\\)%.*', '', t)
def cnt(name, files):
    t = load(files)
    pats = {'figure': r'\\begin\{figure\*?\}', 'table/longtable': r'\\begin\{(?:table\*?|longtable)\}', 'formula': r'\\begin\{formula\}', 'procedure': r'\\begin\{procedure\}',
            'tikzpicture': r'\\begin\{tikzpicture\}', 'includegraphics': r'\\includegraphics', 'footnote': r'\\footnote\{'}
    print(name, {k: len(re.findall(p, t)) for k, p in pats.items()})
    labels = set(re.findall(r'\\label\{([^}]+)\}', t))
    refs = set()
    for m in re.finditer(r'\\(?:auto|c|v|n)?ref\{([^}]+)\}', t):
        refs.update(x.strip() for x in m.group(1).split(','))
    print('  labels:', len(labels), '| refs únicas:', len(refs), '| refs sin label:', sorted(refs - labels), '| labels sin ref:', len(labels - refs))
    caps = re.findall(r'\\caption(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}', t)
    nofuente = [c[:60] for c in caps if 'Fuente' not in c and 'fuente' not in c]
    print('  captions:', len(caps), '| sin la palabra "Fuente":', len(nofuente))
    for c in nofuente[:12]: print('     -', c)
from plan_fuentes import archivos as plan_tex
cnt('PLAN', plan_tex(relativos=True))
cnt('TESIS', glob.glob('Tesis_LaTeX_Borrador_UNI/capitulos/*.tex') + glob.glob('Tesis_LaTeX_Borrador_UNI/tablas/*.tex'))
