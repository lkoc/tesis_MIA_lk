"""Valida el documento de auditoría: enlaces relativos, tablas y registro de recomendaciones."""
import re, os, sys, urllib.parse, collections
doc = os.path.join(os.path.dirname(__file__), '..', '..', 'AUDITORIA_GENERAL_2026-09-29.md')
doc = os.path.abspath(doc)
base = os.path.dirname(doc)
txt = open(doc, encoding='utf-8').read()

# 1. Enlaces relativos
bad = []
for m in re.finditer(r'\]\(([^)#\s]+)(#[^)]*)?\)', txt):
    tgt = urllib.parse.unquote(m.group(1))
    if re.match(r'^(https?:|mailto:)', tgt):
        continue
    p = os.path.normpath(os.path.join(base, tgt))
    if not os.path.exists(p):
        bad.append(tgt)
print('enlaces relativos rotos:', sorted(set(bad)))

# 2. Consistencia de columnas por tabla
tables_bad = []
lines = txt.split('\n')
i = 0
while i < len(lines):
    if lines[i].startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|$', lines[i + 1]):
        n = len(re.findall(r'(?<!\\)\|', lines[i])) - 1
        j = i + 2
        while j < len(lines) and lines[j].startswith('|'):
            k = len(re.findall(r'(?<!\\)\|', lines[j])) - 1
            if k != n:
                tables_bad.append((j + 1, n, k))
            j += 1
        i = j
    else:
        i += 1
print('filas con distinto número de columnas:', tables_bad)

# 3. Registro de recomendaciones
rec = re.findall(r'^\| (R-\d\d) \| (Crítico|Mayor|Menor|Sugerencia) \|', txt, flags=re.M)
ids = [r[0] for r in rec]
print('recomendaciones:', len(rec), '| duplicadas:', [k for k, v in collections.Counter(ids).items() if v > 1])
print('por severidad:', dict(collections.Counter(r[1] for r in rec)))
print('faltan IDs:', [f'R-{n:02d}' for n in range(1, 41) if f'R-{n:02d}' not in ids])

# 4. IDs de hallazgos citados y definidos
defined = set(re.findall(r'^\| ((?:PL|TS|CT|MD|BI|RP)-\d\d) \|', txt, flags=re.M))
cited = set(re.findall(r'\b((?:PL|TS|CT|MD|BI|RP)-\d\d)\b', txt))
print('IDs citados sin definir en tabla:', sorted(cited - defined))
print('marcas de casilla sin normalizar:', len(re.findall(r'☐ ☐ ☐', txt)))
print('caracteres de reemplazo:', txt.count('\ufffd'), '| longitud (líneas):', len(lines))
