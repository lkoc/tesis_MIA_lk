import os, re
root = r'c:\usr\ths_mia_fiis\tesis_MIA_lk'
pat = re.compile(r'C:\\\\?Users\\\\?QU1267|C:/Users/QU1267', re.I)
skip = {'.git', '.venv-benchmarks-repro', '.venv-engie', '__pycache__', 'node_modules', 'estado_inicial', '.engie', 'zotero_offline'}
hits = []
for d, dirs, files in os.walk(root):
    dirs[:] = [x for x in dirs if x not in skip and not x.startswith('.venv')]
    for f in files:
        if f.endswith(('.py', '.ps1', '.js', '.md', '.txt', '.json')) and 'explicit_study' not in d:
            p = os.path.join(d, f)
            try:
                if os.path.getsize(p) > 3_000_000: continue
                t = open(p, encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            n = len(pat.findall(t))
            if n: hits.append((os.path.relpath(p, root), n))
py = [h for h in hits if h[0].endswith(('.py', '.ps1', '.js'))]
print('archivos de código (.py/.ps1/.js) con C:\\Users\\QU1267 fuera de explicit_study:', len(py))
for h in sorted(py): print('  ', h)
print('otros (md/txt/json):', [h for h in hits if h not in py][:15])
