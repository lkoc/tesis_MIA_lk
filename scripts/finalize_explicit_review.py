"""Publish the explicit-physics review from retained machine-readable evidence."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'docs/auditoria/metodologia_2026-09-14'


def banner(relative,text):
    path=ROOT/relative;content=path.read_text(encoding='utf-8')
    start='<!-- explicit-heat-2d-v1 -->';end='<!-- /explicit-heat-2d-v1 -->'
    if start in content:content=content.split(end,1)[1].lstrip()
    path.write_text(start+'\n'+text+'\n'+end+'\n\n'+content,encoding='utf-8')


def main():
    path=ROOT/'README.md';content=path.read_text(encoding='utf-8')
    first,rest=content.split('\n## ',1)
    title=first.splitlines()[0]
    path.write_text(title+'\n\nLa formulación activa resuelve la ecuación de calor en conductor, todas las capas y suelo 2D, con interfaces explícitas. '+
        'La metodología y el flujo vigente están en [FULL_DOMAIN.md](Benchmarks/FULL_DOMAIN.md). '+
        'La verificación nueva, el efecto piel y sus límites se documentan en [la revisión de física explícita](docs/REVISION_FISICA_EXPLICITA.md). '+
        'Los PDF, tablas y campañas anteriores conservan la formulación reducida histórica; no constituyen resultados finales de la nueva formulación.\n\n## '+rest,encoding='utf-8')
    banner('Benchmarks/README.md','**Flujo vigente:** [calor explícito en todas las capas](FULL_DOMAIN.md). Las instrucciones que siguen documentan la campaña reducida histórica. Para cables, las entradas antiguas requieren `--legacy-reduced`; el flujo activo es `run_case.py`.')
    banner('Benchmarks/FORMAT.md','**Contrato vigente:** [FULL_DOMAIN.md](FULL_DOMAIN.md). `explicit_results` usa identidad de física, etiquetas de material y temperatura interior explícita; este contrato histórico no debe aplicarse automáticamente a esos archivos.')
    banner('docs/README.md','La metodología vigente es [METODOLOGIA_PROPUESTA_2026-09-14.md](METODOLOGIA_PROPUESTA_2026-09-14.md), desarrollada en [FULL_DOMAIN.md](../Benchmarks/FULL_DOMAIN.md). Véase [la revisión de física explícita](REVISION_FISICA_EXPLICITA.md) para el estado de verificación y los resultados sobre efecto piel.')
    banner('docs/AUDITORIA_POST_METODOLOGIA_2026-09-14.md','Esta auditoría conserva el diagnóstico de la formulación reducida anterior. La implementación posterior con ecuación de calor interior y su evidencia se describen en [REVISION_FISICA_EXPLICITA.md](REVISION_FISICA_EXPLICITA.md).')
    path=ROOT/'docs/METODOLOGIA_PROPUESTA_2026-09-14.md';text=path.read_text(encoding='utf-8')
    text=text.replace('regiones explícitas, transmisión, escalas locales y opción reducida condicionada a verificación','regiones explícitas, transmisión y escalas locales; equivalentes térmicos retirados de todas las variantes activas')
    marker='\n## 11. Precisamiento eléctrico y cierre de la implementación'
    text=(text.split(marker)[0]+marker+'\n\nSe aprueba técnicamente el uso de la ecuación de calor interior en todas las variantes. '+
        'El contrato identifica resistencia DC/AC, frecuencia, temperatura y procedencia; los datos DC a 60 Hz necesitan corrección de pérdidas y los datos AC no se corrigen dos veces. '+
        'La distribución de calor es una fuente volumétrica de la PDE, nunca una sustitución del interior térmico. '+
        'El perfil cilíndrico de piel es una estimación condicionada a la construcción y no incluye automáticamente proximidad. '+
        'La fuente uniforme debe justificarse mediante sensibilidad a igual pérdida total. '+
        'Las especificaciones, cifras reproducibles y límites están en [FULL_DOMAIN.md](../Benchmarks/FULL_DOMAIN.md); '+
        'la verificación posterior se conserva en [REVISION_FISICA_EXPLICITA.md](REVISION_FISICA_EXPLICITA.md). '+
        'La aprobación del método no equivale a aceptación de toda la campaña ni a aprobación institucional de la tesis.\n')
    path.write_text(text,encoding='utf-8')
    runs=[]
    for path in sorted((ROOT/'Benchmarks/explicit_results').rglob('pinn_seed*.json')):
        data=json.loads(path.read_text());entry=dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        entry.update({k:data.get(k) for k in ['variant','mode','accepted','Tmax_C','error_Tmax_K','rmse_K','balance_pct','derived_balance_pct']});runs.append(entry)
    suite=ET.parse(AUDIT/'explicit_tests.xml').getroot().find('testsuite')
    skin_path=ROOT/'Benchmarks/explicit_results/skin_aras_60Hz/equal_power/comparison.json'
    skin=json.loads(skin_path.read_text()) if skin_path.exists() else None
    cleanup=json.loads((AUDIT/'cleanup.json').read_text(encoding='utf-8-sig'))
    manifest=dict(runs=runs,tests=dict(suite.attrib),cleanup=dict(files=len(cleanup['files']),bytes=cleanup['total_bytes']),
        skin_2d_comparison=str(skin_path.relative_to(ROOT)) if skin else None)
    current_sources=['Benchmarks/full_domain.py','Benchmarks/full_physics.py','Benchmarks/full_pinn.py','Benchmarks/full_fem.py',
        'Benchmarks/skin_effect.py','Benchmarks/skin_study.py','Benchmarks/skin_fem_study.py','Benchmarks/run_case.py',
        'Benchmarks/FULL_DOMAIN.md','docs/METODOLOGIA_PROPUESTA_2026-09-14.md','Plan/plan_tesis_cables_pinn.tex']
    manifest['current_source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in current_sources}
    (AUDIT/'explicit_review.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=['# Revisión de física explícita y efecto piel','',
        'La revisión metodológica iniciada el 14 de septiembre de 2026 precede a estos cambios. Se implementó la ecuación de calor en todas las capas para todas las variantes activas; la verificación posterior no cambia retroactivamente los resultados reducidos.',
        '', '## Evidencia nueva','',
        '| Corrida | Variante | Tmax, °C | Error Tmax, K | Balance, % | Estado |',
        '|---|---|---:|---:|---:|---|']
    for r in runs:
        lines.append(f"| [{r['path']}](../{r['path']}) | {r['variant']} | {r['Tmax_C']:.6f} | {r['error_Tmax_K']:.6f} | {r['balance_pct']:.6f} | {'Aceptada' if r['accepted'] else 'Rechazada'} |")
    lines+=['','Los pilotos aceptados cumplen también los saltos de temperatura/flujo y el balance por material registrados en sus JSON. El piloto mixto rechazado se conserva. No hay aún confirmación con semillas independientes ni campaña completa de ampacidad con esta física.',
        '',f"Pruebas automatizadas: {suite.attrib['tests']} ejecutadas; {suite.attrib['failures']} fallos y {suite.attrib['errors']} errores. Tres pruebas lentas fueron excluidas por la configuración existente. La primera ejecución conjunta detectó contaminación de la precisión global de PyTorch entre pruebas; se aisló el estado de cada prueba y se repitió la suite completa.",
        '', '## Efecto piel','',
        'Los datos del catálogo se tratan como DC para la estimación; su etiqueta R20 no prueba la naturaleza del dato original. Antes de concluir sobre un cable real, documentar construcción, resistencia, frecuencia y temperatura de la fuente primaria. La corrección no debe repetirse si el dato ya es AC a 60 Hz.',
        '', 'La estimación de conductor circular macizo aislado arroja aumento de pérdida de 0,195–0,317 % para XLPE y 25,578–37,136 % para los casos grandes, al comparar 90 y 20 °C. A igual pérdida, el diagnóstico radial cambia Tmax en aproximadamente 0,00001 K, 0,004 K y 0,0016 K para XLPE, Aras y Kim. Son resultados condicionales calculados, no valores certificados de esos cables. [Datos y supuestos](auditoria/metodologia_2026-09-14/skin_effect.json).']
    if skin:
        lines+=['',f"El contraste FEM 2D adicional del caso Aras, con cuatro mallas por fuente y potencia AC igualada por integración, dio cambio de Tmax de **{skin['Tmax_change_K']:.6f} K**, máximo cambio de campo muestreado {skin['max_sample_field_difference_K']:.6f} K y diferencia relativa de potencia {skin['source_difference_pct']:.3g} %. [Comparación completa](../Benchmarks/explicit_results/skin_aras_60Hz/equal_power/comparison.json). La normalización elimina la pequeña diferencia de potencia debida a interpolar el perfil eléctrico; sus factores están registrados. Las mallas se refinan para evaluar la estabilidad de la diferencia, no solo la temperatura absoluta.",
            '', 'Este ensayo apoya que la redistribución radial es secundaria para la temperatura estacionaria de ese caso, conservando el aumento de pérdida AC. No demuestra lo mismo para proximidad, conductores segmentados, pantallas con pérdidas o transitorios.']
    lines+=['', 'La ecuación de calor y el perfil volumétrico se conservan en el código incluso cuando la sensibilidad sea pequeña. El acoplamiento DC usa conductividad eléctrica local; el acoplamiento electromagnético AC con temperatura no uniforme sigue pendiente. [Especificación, fuentes primarias y comandos](../Benchmarks/FULL_DOMAIN.md).',
        '', '## Depuración y trazabilidad','',
        f"Se retiraron {len(cleanup['files'])} archivos obsoletos ({cleanup['total_bytes']:,} bytes). Los dos respaldos LaTeX únicos y la copia PDF del plan están en un [archivo verificado](historico/plan_respaldos_previos_2026-09-14.zip). [Manifiesto de limpieza](auditoria/metodologia_2026-09-14/cleanup.json).",
        '', 'La referencia coaxial inicial tenía un defecto de interpolación al seleccionar una celda del material vecino; se retiraron sus archivos numéricos y se preservaron los metadatos del diagnóstico. `reference_verified` contiene la referencia corregida: error de Tmax de 0,00000368 K frente a la solución analítica. No se borraron campañas históricas ni cambios previos del usuario.',
        '', 'El plan fuente incluye el precisamiento térmico y eléctrico. Los PDF y tablas anteriores aún representan la formulación histórica y requieren recompilación y actualización de resultados cuando se cierre la campaña nueva.']
    (ROOT/'docs/REVISION_FISICA_EXPLICITA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
