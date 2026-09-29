import os, re, glob, subprocess, json
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
print('LICENSE.md:', repr(open('LICENSE.md', encoding='utf-8', errors='replace').read()[:200]))
raw = open('.gitignore', 'rb').read()
try:
    raw.decode('utf-8'); print('.gitignore utf-8 ok, lines', len(raw.splitlines()))
except Exception as e:
    print('.gitignore NOT utf-8:', e)
print('gitignore mojibake lines:', [l for l in raw.decode('utf-8', 'replace').splitlines() if 'Ã' in l or '\ufffd' in l][:10])
root = [f for f in os.listdir('.') if os.path.isfile(f)]
print('archivos en raíz:', len(root))
print(' .py en raíz:', len([f for f in root if f.endswith('.py')]), '| _*.py:', len([f for f in root if f.startswith('_') and f.endswith('.py')]))
print(' otros:', [f for f in root if not f.endswith('.py')])
# rutas absolutas del usuario
pat = re.compile(r'C:\\\\?Users\\\\?QU1267|C:/Users/QU1267|/home/lkoc|C:\\\\?usr\\\\?ths_mia', re.I)
hits = {}
skip = ('.venv', '.git', 'node_modules', '__pycache__', 'docs/auditoria/estado_inicial', 'zotero_offline')
for ext in ('py', 'ps1', 'json', 'md', 'tex', 'txt', 'yml', 'yaml', 'toml', 'js'):
    for f in glob.glob(f'**/*.{ext}', recursive=True):
        n = f.replace('\\', '/')
        if any(s in n for s in skip): continue
        try:
            if os.path.getsize(f) > 3_000_000: continue
            t = open(f, encoding='utf-8', errors='replace').read()
        except Exception: continue
        c = len(pat.findall(t))
        if c: hits[n] = c
print('archivos con rutas absolutas (usuario/PC):', len(hits))
for k, v in sorted(hits.items(), key=lambda kv: -kv[1])[:40]: print('  ', v, k)
# secretos
sec = re.compile(r'(api[_-]?key|secret|token|passw(or)?d|bearer|authorization)\s*[:=]\s*[\'"][A-Za-z0-9_\-\.]{12,}', re.I)
sh = []
for ext in ('py', 'ps1', 'json', 'md', 'yml', 'yaml', 'toml', 'js', 'txt', 'cfg', 'ini', 'env'):
    for f in glob.glob(f'**/*.{ext}', recursive=True):
        n = f.replace('\\', '/')
        if any(s in n for s in skip + ('.engie/',)): continue
        try:
            if os.path.getsize(f) > 2_000_000: continue
            t = open(f, encoding='utf-8', errors='replace').read()
        except Exception: continue
        if sec.search(t): sh.append(n)
print('posibles secretos (patrón):', sh[:20])
print('.engie existe:', os.path.isdir('.engie'), '| en gitignore:', any(l.strip().startswith('.engie') for l in raw.decode('utf-8','replace').splitlines()))
for f in ('.github/workflows',):
    print(f, os.path.exists(f))
print('pyproject:'); print(open('pyproject.toml', encoding='utf-8').read()[:1500])
print('req files:', glob.glob('**/requirements*.txt', recursive=True)[:10], glob.glob('**/conda-explicit*.txt', recursive=True)[:5], glob.glob('**/environment*.yml', recursive=True)[:5])
