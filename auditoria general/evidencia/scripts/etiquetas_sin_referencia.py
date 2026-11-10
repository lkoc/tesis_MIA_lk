import re, glob, os
from plan_fuentes import archivos as plan_tex
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
def load(files):
    return re.sub(r'(?<!\\)%.*', '', '\n'.join(open(f, encoding='utf-8', errors='replace').read() for f in files))
for name, files in [('PLAN', plan_tex(relativos=True)), ('TESIS', glob.glob('Tesis_LaTeX_Borrador_UNI/capitulos/*.tex') + glob.glob('Tesis_LaTeX_Borrador_UNI/tablas/*.tex'))]:
    t = load(files)
    labels = re.findall(r'\\label\{([^}]+)\}', t)
    refs = set()
    for m in re.finditer(r'\\(?:auto|c|v|n)?ref\{([^}]+)\}', t):
        refs.update(x.strip() for x in m.group(1).split(','))
    un = [l for l in labels if l not in refs]
    grp = {}
    for l in un:
        grp.setdefault(l.split(':')[0], []).append(l)
    print(name, {k: len(v) for k, v in grp.items()})
    for k in ('fig', 'tab', 'for', 'proc', 'anx'):
        if k in grp: print('  ', k, grp[k])
