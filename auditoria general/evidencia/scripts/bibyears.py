import re, os, sys, collections
sys.path.insert(0, os.path.join(os.environ['TEMP'], 'audit_gen'))
import bibcheck as b
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
def yr(e):
    f = e['fields']
    y = f.get('year') or f.get('date') or ''
    m = re.search(r'(\d{4})', y)
    return int(m.group(1)) if m else None
def stats(name, bibs, texs):
    ent = {}
    for x in bibs: ent.update(b.parse_bib(x)[0])
    cited = b.cited_keys(texs)
    ys = {k: yr(ent[k]) for k in cited if k in ent}
    types = {k: ent[k]['type'] for k in cited if k in ent}
    n = len(ys)
    recent = [k for k, y in ys.items() if y and y >= 2021]
    old = sorted((y, k) for k, y in ys.items() if y and y < 2021)
    print(f'== {name}: citadas {n}; >=2021: {len(recent)} ({100*len(recent)/n:.0f}%); 2016-2020: {sum(1 for y in ys.values() if y and 2016<=y<=2020)}; <2016: {sum(1 for y in ys.values() if y and y<2016)}')
    print('   tipos citados:', dict(collections.Counter(types.values())))
    print('   <2021:', old)
from plan_fuentes import archivos as plan_tex
stats('PLAN', ['Plan/referencias_plan_tesis.bib'], plan_tex(relativos=True))
import glob
stats('TESIS', ['Tesis_LaTeX_Borrador_UNI/referencias.bib', 'Benchmarks/references_selection.bib'], glob.glob('Tesis_LaTeX_Borrador_UNI/**/*.tex', recursive=True))
