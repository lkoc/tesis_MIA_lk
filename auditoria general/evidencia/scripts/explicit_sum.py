import csv, json, collections, os
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
rows = list(csv.DictReader(open('Benchmarks/explicit_study/report/runs.csv', encoding='utf-8')))
print('columns:', list(rows[0].keys()))
print('rows:', len(rows))
by = collections.Counter((r.get('stage'), r.get('accepted') or r.get('pass') or r.get('thermal_pass')) for r in rows)
print(by)
for r in rows[:2]: print(r)
d = json.load(open('Benchmarks/explicit_study/report/summary.json', encoding='utf-8'))
print('production_accepted', d['production_accepted'], '/', d['production_total'])
print(json.dumps(d.get('production'), ensure_ascii=False)[:3000])
print(json.dumps(d.get('case_choices'), ensure_ascii=False)[:1500])
for k in ('effects', 'convergence', 'hotspots'):
    print(k, json.dumps(d.get(k), ensure_ascii=False)[:1200])
