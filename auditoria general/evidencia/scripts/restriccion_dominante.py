"""Restricción dominante por corrida y por caso (etapas C, C2 y D). Solo lectura de resultados existentes."""
import json, sys, os, collections
ROOT = r'c:\usr\ths_mia_fiis\tesis_MIA_lk'
os.chdir(ROOT); sys.path.insert(0, ROOT)
from Benchmarks import full_study_report as R
BASE = R.BASE
def rows(stage):
    return json.loads((BASE / stage / 'summary.json').read_text(encoding='utf-8'))
out = {}
for stage in ('C', 'C2', 'D'):
    rs = rows(stage)
    for r in rs:
        r['components'] = R.failure_components(r)
    out[stage] = rs
lines = []
def dom(c):
    k = max(c, key=c.get); return k, c[k]
print('== D por corrida')
for r in out['D']:
    k, v = dom(r['components'])
    print(f"{r['case']:22s} s{r['seed']:3d} acc={str(r['accepted']):5s} dom={k:11s} {v:7.2f} | " + ' '.join(f"{a}={b:.2f}" for a, b in r['components'].items()))
print('== D por caso (máximo por componente entre semillas)')
cases = sorted({r['case'] for r in out['D']})
tab = []
for c in cases:
    g = [r for r in out['D'] if r['case'] == c]
    worst = {k: max(r['components'][k] for r in g) for k in g[0]['components']}
    k, v = dom(worst)
    acc = sum(r['accepted'] for r in g)
    tab.append((c, acc, len(g), k, v, worst))
    print(c, f'{acc}/{len(g)}', k, round(v, 2), {a: round(b, 2) for a, b in worst.items()})
print('== C y C2 por candidato')
for stage in ('C', 'C2'):
    cands = sorted({r['candidate'] for r in out[stage]})
    for cnd in cands:
        g = [r for r in out[stage] if r['candidate'] == cnd]
        worst = {k: max(r['components'][k] for r in g) for k in g[0]['components']}
        k, v = dom(worst)
        print(stage, cnd, f"{sum(r['accepted'] for r in g)}/{len(g)}", k, round(v, 2), {a: round(b, 2) for a, b in worst.items()})
json.dump({'D': [(c, a, n, k, v, w) for c, a, n, k, v, w in tab]}, open(os.path.join(os.path.dirname(__file__), '..', 'salidas', 'restriccion_dominante.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
