"""Conserva fuentes previas y extrae documentos para la auditoría, sin modificarlos."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'auditoria'
OUT.mkdir(parents=True, exist_ok=True)
snapshot = OUT / 'estado_inicial'
if not snapshot.exists():
    snapshot.mkdir()
    for source in [ROOT / 'Tesis_LaTeX_Borrador_UNI', ROOT / 'pinn_cables', ROOT / 'README.md']:
        if source.is_dir():
            shutil.copytree(source, snapshot / source.name, ignore=shutil.ignore_patterns('__pycache__', '*.pt', '*.png'))
        else:
            shutil.copy2(source, snapshot / source.name)
    (snapshot / 'git_status.txt').write_text(subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True), encoding='utf-8')
paths = []
for dirname in ['Plan', 'matriz_consistencia', 'temas_cursos_Tesis_1']:
    paths.extend(p for p in (ROOT / dirname).rglob('*.pdf') if any(t in p.name.lower() for t in ['guía','guia','formato','criterio','desarrollo']))
for path in paths:
    with fitz.open(path) as doc:
        text = '\n'.join(f'\n--- Página {i+1} ---\n'+p.get_text() for i,p in enumerate(doc))
    (OUT / (path.stem + '.txt')).write_text(text, encoding='utf-8')
inventory = []
for directory in ['pinn_cables', 'examples', 'scripts']:
    for p in (ROOT / directory).rglob('*.py'):
        b = p.read_bytes()
        inventory.append(dict(path=p.relative_to(ROOT).as_posix(), lines=len(b.splitlines()), sha256=hashlib.sha256(b).hexdigest()))
(OUT/'inventario_codigo.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
print(json.dumps({'python_files':len(inventory),'lines':sum(p['lines'] for p in inventory),'guides':[str(p.relative_to(ROOT)) for p in paths]},ensure_ascii=False))
