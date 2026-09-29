"""Cuenta palabras de prosa del Plan: total actual y las lineas nuevas o modificadas respecto de HEAD (moverlas no cuenta)."""
import re, subprocess

RUTA = 'Plan/plan_tesis_cables_pinn.tex'
ROOT = r'c:\usr\ths_mia_fiis\tesis_MIA_lk'
head = subprocess.run(['git', '-C', ROOT, 'show', f'HEAD:{RUTA}'], capture_output=True, text=True, encoding='utf-8').stdout
from plan_fuentes import expandido
actual = expandido()


def norm(l):
    return re.sub(r'\s+', ' ', l.strip())


def prosa(l):
    l = re.sub(r'(?<!\\)%.*$', '', l)
    l = re.sub(r'\$[^$]*\$', ' ', l)
    l = re.sub(r'\\(?:parencites?|textcites?|cites?)\*?(?:\[[^\]]*\])*\{[^}]*\}(?:\[[^\]]*\]\{[^}]*\})*', ' ', l)
    l = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?', ' ', l)
    l = re.sub(r'[{}&\\]', ' ', l)
    return len(re.findall(r'[A-Za-zÁ-úñÑ]{2,}', l))


lh = {norm(l) for l in head.splitlines() if l.strip()}
total = nuevas = 0
por_seccion = {}
sec = 'Preliminares'
for l in actual.splitlines():
    m = re.match(r'\\(chapter|section)\*?\{([^}]*)\}', l)
    if m:
        sec = m.group(2)
    w = prosa(l)
    total += w
    if l.strip() and norm(l) not in lh:
        nuevas += w
        por_seccion[sec] = por_seccion.get(sec, 0) + w
print(f'palabras de prosa actuales: {total}')
print(f'palabras en lineas nuevas o modificadas respecto de HEAD: {nuevas} ({100 * nuevas / total:.1f} %)')
for s, w in sorted(por_seccion.items(), key=lambda x: -x[1])[:12]:
    print(f'  {w:6d}  {s}')
