# -*- coding: utf-8 -*-
"""R-10: mueve del .bib del Plan las entradas no citadas a un archivo de reserva (no se pierde informacion).

Uso: python bib_podar_plan.py
"""
import re, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import bibcheck as b

ROOT = r'c:\usr\ths_mia_fiis\tesis_MIA_lk'
BIB = os.path.join(ROOT, 'Plan', 'referencias_plan_tesis.bib')
RESERVA = os.path.join(ROOT, 'Plan', 'referencias_plan_tesis_no_usadas.bib')
TEX = os.path.join(ROOT, 'Plan', 'plan_tesis_cables_pinn.tex')

raw = open(BIB, encoding='utf-8', newline='').read()
crlf = '\r\n' in raw
txt = raw.replace('\r\n', '\n')
from plan_fuentes import archivos as plan_tex
cited = set(b.cited_keys(plan_tex()))
# separar entradas: cada una empieza en '@tipo{clave,' al inicio de linea
starts = [m.start() for m in re.finditer(r'(?m)^@\w+\s*\{', txt)]
starts.append(len(txt))
head = txt[:starts[0]]
entries = []
for a, z in zip(starts, starts[1:]):
    blk = txt[a:z]
    key = re.match(r'@\w+\s*\{\s*([^,\s]+)\s*,', blk).group(1)
    entries.append((key, blk))
keep = [blk for k, blk in entries if k in cited]
move = [(k, blk) for k, blk in entries if k not in cited]
missing = sorted(k for k in cited if k not in {k2 for k2, _ in entries})
assert not missing, missing
print('total', len(entries), 'usadas', len(keep), 'a reserva', len(move))
print([k for k, _ in move])
nuevo = head + ''.join(keep).rstrip('\n') + '\n'
res = ('% Entradas del Plan de tesis que no se citan en el texto vigente (R-10, auditoria general 2026-09-29).\n'
       '% Se conservan aqui para no perder los datos; pueden volver a referencias_plan_tesis.bib si se citan.\n\n'
       + ''.join(blk for _, blk in move).rstrip('\n') + '\n')
if crlf:
    nuevo = nuevo.replace('\n', '\r\n'); res = res.replace('\n', '\r\n')
open(BIB, 'w', encoding='utf-8', newline='').write(nuevo)
open(RESERVA, 'w', encoding='utf-8', newline='').write(res)
print('ok')
