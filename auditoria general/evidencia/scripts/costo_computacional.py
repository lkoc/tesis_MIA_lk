import json, statistics, os
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk\Benchmarks\explicit_study')
d = json.load(open('D/summary.json', encoding='utf-8'))
cases = sorted({r['case'] for r in d})
print('caso | parámetros | segundos min-med-máx | aceptadas')
for c in cases:
    g = [r for r in d if r['case'] == c]
    s = [r['seconds'] for r in g]
    print(c, max(r['parameters'] for r in g), round(min(s)), round(statistics.median(s)), round(max(s)), sum(r['accepted'] for r in g), '/', len(g))
tot = sum(r['seconds'] for r in d)
print('total D horas (suma de corridas, concurrentes):', round(tot / 3600, 1))
allrows = []
for s in ['A', 'B', 'C', 'C2', 'D']:
    allrows += json.load(open(f'{s}/summary.json', encoding='utf-8'))
print('total corridas', len(allrows), '| horas suma', round(sum(r['seconds'] for r in allrows) / 3600, 1))
