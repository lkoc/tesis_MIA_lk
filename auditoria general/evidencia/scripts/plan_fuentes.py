"""Fuentes LaTeX del Plan: archivo principal + capitulos/ + capitulos/anexos/ (el Plan esta dividido igual que la tesis)."""
import glob
import os
import re

ROOT = r'c:\usr\ths_mia_fiis\tesis_MIA_lk'
PLAN_DIR = os.path.join(ROOT, 'Plan')
PRINCIPAL = os.path.join(PLAN_DIR, 'plan_tesis_cables_pinn.tex')


def archivos(relativos=False):
    """Principal, capitulos y anexos; con relativos=True devuelve rutas 'Plan/...' respecto de la raiz del repositorio."""
    rutas = [PRINCIPAL] + sorted(glob.glob(os.path.join(PLAN_DIR, 'capitulos', '*.tex'))) \
        + sorted(glob.glob(os.path.join(PLAN_DIR, 'capitulos', 'anexos', '*.tex')))
    return [os.path.relpath(p, ROOT).replace('\\', '/') for p in rutas] if relativos else rutas


def expandido(ruta=PRINCIPAL):
    """Texto del principal con cada \\input/\\include sustituido por el contenido de su archivo, en orden."""
    salida = []
    for ln in open(ruta, encoding='utf-8', errors='replace').read().replace('\r\n', '\n').split('\n'):
        m = re.match(r'\\(?:input|include)\{(capitulos/[^}]+)\}\s*$', ln)
        salida.append(expandido(os.path.join(PLAN_DIR, m.group(1) + '.tex')) if m else ln)
    return '\n'.join(salida)
