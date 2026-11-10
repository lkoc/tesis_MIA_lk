"""
Reclasificación completa de paper_type basada en revisión del contenido de cada paper.
Cambios:
  - 29 papers sin tipo → asignados por primera vez
  - 5 papers mal clasificados → reclasificados
Criterios:
  estado_arte   : surveys, revisiones sistemáticas, artículos de revisión/comparación
  aporte_teorico: contribuciones metodológicas nuevas (PINN, modelos, algoritmos)
  aplicacion    : aplicación de métodos existentes a casos concretos
  normativa     : estándares IEC/IEEE/ASTM oficiales
  fuente_clasica: obras fundacionales históricas (>40 años, altamente citadas)
  guia_tecnica  : manuales, handbooks, catálogos técnicos, reportes CIGRE/guías
  apoyo         : material didáctico de soporte
"""
import pathlib, json, re

HTML = pathlib.Path(r'c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html')
html = HTML.read_text(encoding='utf-8')

# ── Cambios de clasificación ──────────────────────────────────────────────────
CHANGES = {
    # Papers que NO tenían tipo (sin tipo → asignar)
    1:  'aporte_teorico',   # Al-Dulaimi 2024: nuevo modelo híbrido FEM+BPNN para temperatura de cables
    2:  'aporte_teorico',   # Sergio 2025: nuevo método ML para thermal ratings climáticos de largo plazo
    3:  'aplicacion',       # Möbius 2025: estudio aplicado DTR bajo cambio climático en Hamburg (IEC 60287)
    4:  'aplicacion',       # Kolawole 2024: estudio aplicado de conductividad térmica de suelos
    5:  'aporte_teorico',   # Li 2024: desarrolla modelo térmico mejorado para cables en ductos con relleno mixto
    6:  'aporte_teorico',   # Nadjem 2026: desarrollo de nuevos nanocompuestos XLPE para cables
    7:  'aplicacion',       # Khumalo 2025: estudio aplicado de ampacidad bajo condiciones sudafricanas
    56: 'aplicacion',       # Oladunjoye 2012: medición in-situ de resistividad térmica de suelos
    57: 'guia_tecnica',     # Anónimo 2017: reporte CIGRE sobre desempeño de largo plazo de suelos y backfill
    58: 'aporte_teorico',   # Raissi 2019: artículo fundacional de PINNs (Phys-Informed Neural Networks)
    59: 'estado_arte',      # Lawal 2022: revisión sistemática + análisis bibliométrico de PINNs
    60: 'estado_arte',      # Ren 2025: revisión de la evolución metodológica de PINNs
    61: 'aporte_teorico',   # Jagtap 2020: propone XPINNs (descomposición de dominio espacio-tiempo)
    62: 'aporte_teorico',   # Shukla 2021: PINNs paralelos via descomposición de dominio
    63: 'aporte_teorico',   # Billah 2023: PINN para transferencia de calor inversa con datos ruidosos
    64: 'aporte_teorico',   # Chen 2024: PINN para problema inverso no-lineal de conducción
    65: 'aporte_teorico',   # Anónimo 2024: PINN para problema inverso con datos limitados y ruidosos
    66: 'aporte_teorico',   # Anónimo 2023: PINN para conductividad efectiva en medios porosos
    67: 'aporte_teorico',   # Xing 2023: PINN para conducción anisótropa 3D en estado estacionario
    68: 'aporte_teorico',   # Pan 2025: PINN para reconstrucción de campo térmico 2D inverso
    69: 'aporte_teorico',   # Liu 2026: PINN para conducción inversa 3D en estado estacionario
    70: 'aporte_teorico',   # Palar 2023: PINN para conducción inversa con sensores limitados
    71: 'aporte_teorico',   # Anónimo 2023: PINN para modelado geotérmico inverso
    72: 'aporte_teorico',   # Anónimo 2026: PINN para modelado inverso en ingeniería civil
    73: 'estado_arte',      # Enescu 2020: overview/revisión de evaluación térmica de cables
    74: 'aplicacion',       # Atoccsa 2024: optimización de ampacidad con PSO aplicado a backfill concreto
    75: 'aporte_teorico',   # Moutassem 2025: nuevo método CNN-LSTM physics-enhanced para monitoreo de cables
    76: 'normativa',        # Anónimo 2021: análisis térmico dinámico IEC 60853 (documento normativo)
    77: 'normativa',        # Commission 2002: IEC 60853 Parte 3 — rating cíclico con secado de suelo
    # Papers mal clasificados (reclasificar)
    8:  'guia_tecnica',     # Anders 1997: libro de referencia para cálculo de corriente admisible (no investigación)
    9:  'guia_tecnica',     # Anders 2005: libro guía para cables en entornos adversos (no survey de investigación)
    30: 'aplicacion',       # Heggås 2019: tesis que APLICA DTR a casos concretos (no es revisión)
    45: 'guia_tecnica',     # Minkowycz 2006: "Handbook of Numerical Heat Transfer" (manual de referencia)
    49: 'fuente_clasica',   # Patankar 1980: obra clásica seminal del método de volúmenes finitos (>40 años)
}

# ── Aplicar cambios a NODES ───────────────────────────────────────────────────
m = re.search(r'const NODES=(\[.*?\]);', html, re.DOTALL)
if not m:
    print("ERROR: no se encontraron NODES"); exit(1)

nodes = json.loads(m.group(1))
changed = 0
for n in nodes:
    if n['id'] in CHANGES:
        old = n.get('paper_type', '')
        n['paper_type'] = CHANGES[n['id']]
        print(f"  ID {n['id']:2d}  {old or '(sin tipo)':22s}  →  {n['paper_type']}")
        changed += 1

print(f"\nPapers modificados: {changed}")

# Calcular distribución
from collections import Counter
dist = Counter(n.get('paper_type', '') for n in nodes)
print("\nDistribución final:")
for k, v in sorted(dist.items(), key=lambda x: -x[1]):
    label = k if k else '(sin tipo)'
    print(f"  {v:3d}  {label}")

# ── Serializar NODES de vuelta al HTML ───────────────────────────────────────
nodes_json = json.dumps(nodes, ensure_ascii=False)
html = html[:m.start(1)] + nodes_json + html[m.end(1):]
print("\n✓ NODES actualizados")

# ── Actualizar PAPER_TYPES ────────────────────────────────────────────────────
# No cambia la estructura de tipos (los tipos existentes se mantienen)
# Solo el stipo div necesita reflejar los nuevos conteos

# Obtener el orden y colores del PAPER_TYPES actual
pt_m = re.search(r'const PAPER_TYPES=(\{.*?\});', html, re.DOTALL)
pt_data = json.loads(pt_m.group(1))

# Reconstruir el stipo div con nuevos conteos
stipo_items = []
for pt_key, info in pt_data.items():
    cnt = dist.get(pt_key, 0)
    if cnt == 0:
        continue
    stipo_items.append(
        f'<div class="li" id="ltipo_{pt_key}" onclick="toggleTipoF(\'{pt_key}\')">'
        f'<span class="ld" style="background:{info["color"]}"></span>'
        f'<span class="lt">{info["label"]}</span>'
        f'<span class="lc">{cnt}</span></div>'
    )

new_stipo = '<div class="ls" id="stipo" style="display:none">' + ''.join(stipo_items) + '</div>'

# Reemplazar stipo con contador de profundidad (evita bug del regex anterior)
idx_open = html.find('<div class="ls" id="stipo"')
idx_content = html.find('>', idx_open) + 1
depth, i = 1, idx_content
while depth > 0 and i < len(html):
    if html[i:i+5] == '<div ':
        depth += 1; i += 5
    elif html[i:i+6] == '</div>':
        depth -= 1; i += 6
    else:
        i += 1
html = html[:idx_open] + new_stipo + html[i:]
print("✓ stipo div reconstruido")

# ── Guardar ───────────────────────────────────────────────────────────────────
HTML.write_text(html, encoding='utf-8')
print("\nHTML guardado OK")
