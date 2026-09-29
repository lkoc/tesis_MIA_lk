import json, os, glob, collections, datetime
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk\Benchmarks\explicit_study')
for s in ['A', 'B', 'C', 'C2', 'D']:
    p = f'{s}/summary.json'
    if os.path.exists(p):
        d = json.load(open(p, encoding='utf-8'))
        print(s, 'mtime', datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%m-%d %H:%M'), 'keys', list(d)[:8] if isinstance(d, dict) else len(d))
    m = f'{s}/manifest.json'
    if os.path.exists(m):
        d = json.load(open(m, encoding='utf-8'))
        runs = d.get('runs') if isinstance(d, dict) else d
        print('   manifest runs:', len(runs) if runs is not None else None, 'keys', list(d)[:8] if isinstance(d, dict) else '')
d = json.load(open('D/summary.json', encoding='utf-8'))
print(json.dumps(d, ensure_ascii=False)[:2500])
print('--- per-run D')
files = glob.glob('D/production/**/*.json', recursive=True)
print(len(files), 'json files in D/production')
acc = collections.Counter()
rows = []
for f in files:
    try:
        j = json.load(open(f, encoding='utf-8'))
    except Exception:
        continue
    if isinstance(j, dict) and ('accepted' in j or 'thermal_criteria_pass' in j):
        rows.append((f.replace('\\', '/').split('D/production/')[1], j.get('accepted', j.get('thermal_criteria_pass')), j.get('rmse_K'), j.get('Tmax_error_K')))
for r in sorted(rows)[:40]: print(r)
print('status.json', open('status.json', encoding='utf-8').read())
