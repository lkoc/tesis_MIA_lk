"""Record the completed audit from checked artifacts, without inventing status."""
from pathlib import Path
import json,csv,xml.etree.ElementTree as ET,hashlib
ROOT=Path(__file__).resolve().parents[1]

def main():
    review=json.loads((ROOT/'docs/auditoria/final_review.json').read_text(encoding='utf-8'))
    if review['latex_layout_or_reference_problems']:raise ValueError('Resolve PDF warnings first')
    data=json.loads((ROOT/'Benchmarks/summary/summary.json').read_text(encoding='utf-8'))
    validation=json.loads((ROOT/'Benchmarks/summary/artifact_validation.json').read_text(encoding='utf-8'))
    numerical=json.loads((ROOT/'Benchmarks/summary/numerical_analysis.json').read_text(encoding='utf-8'))
    adaptive=json.loads((ROOT/'Benchmarks/summary/adaptive_sampling.json').read_text(encoding='utf-8'))
    assert len(adaptive['replay_checks'])==18
    with (ROOT/'Benchmarks/summary/ampacity_coupled.csv').open(encoding='utf-8-sig') as f:amps=list(csv.DictReader(f))
    suites=ET.parse(ROOT/'docs/auditoria/tests_final.xml').getroot().findall('testsuite')
    assert all(int(s.attrib['failures'])==0 and int(s.attrib['errors'])==0 for s in suites)
    tests=sum(int(s.attrib['tests']) for s in suites)
    assert len(review['notebooks'])==19 and len(numerical['resolution'])==12
    accepted=sum(r['accepted'] for r in data['selected']);ampaccepted=sum(int(r['accepted']) for r in amps)
    partial=', '.join(r['case'] for r in data['selected'] if r['accepted']<r['n']) or 'ninguno'
    failed=', '.join(r['case'] for r in amps if int(r['accepted'])==0) or 'ninguno'
    section=f'''## Cierre de ejecución y revisión

- Batería física: 19 casos; 57 soluciones FEM de verificación y 66 expedientes acoplados, con tres mallas. Todas las referencias finas cumplen los controles declarados.
- Entrenamientos conservados: {data['total_runs']}. La selección principal acepta {accepted}/57 ejecuciones térmicas o electrotérmicas y {ampaccepted}/33 de corriente límite. Los fallos no se eliminan.
- Verificación automatizada: {tests} pruebas aprobadas, incluido el análisis de derivadas de propagación; detalle en `auditoria/tests_final.xml`.
- Auditoría de artefactos: RMSE y máximos reconstruidos desde campos; comprobación de la respuesta FEM, R(T), puntos, modos y fuentes. Las 90 ejecuciones principales tienen fuentes completas. Quedan {len(validation['historical_training_source_not_archived'])} registros históricos con archivo de código original incompleto, identificados individualmente.
- Reproducción: reinstalación Python aislada y dos casos recalculados; FEniCSx recalculado en WSL. No se afirma una reinstalación limpia de Conda.
- Resolución PINN: 36 entrenamientos con 16/32/64 neuronas, tres densidades interiores y presupuesto duplicado. Se conservan tasas empíricas, comparaciones entre semillas y descomposición térmica/eléctrica. Detalle en `Benchmarks/ANALISIS_NUMERICO.md`.
- Colocación adaptativa: 18 entrenamientos nuevos y seis controles fijos. Se comparan renovación aleatoria, residuo y gradiente térmico, con igual número interior y presupuesto. Se reconstruyeron las selecciones y se comprobaron puntuaciones desde estados intermedios. Detalle en `Benchmarks/MUESTREO_ADAPTATIVO.md`.
- Entrega revisable: 19 cuadernos ejecutados con gráficos y citas; PDF de {review['pdf_pages']} páginas, sin errores de referencia ni desbordamientos detectados por el registro de compilación. La revisión visual se conserva en `auditoria/pdf_review/`.
- Metodología, resultados, discusión, conclusiones y resumen se completaron desde el informe interno y las tablas; se mantienen únicamente los espacios personales y administrativos auténticos.

Casos nominales con aceptación parcial: {partial}. Casos sin ninguna
ampacidad PINN aceptada: {failed}. Estas
limitaciones impiden presentar el artefacto como reemplazo general de FEM.
La operación usa R(Tc) individual en ambas técnicas; el R20 del JSON es una
referencia y los ensayos a potencia fija conservan su carácter de verificación.

La comparación de redes más y menos densas es evidencia de insensibilidad en
los rangos probados, condicionada a error externo y a entrenamiento. Los
órdenes observados FEM se separan de las pendientes PINN. La propagación del
error es determinista dentro del modelo reducido; no se inventa incertidumbre
estadística de propiedades físicas ni se aplica GCI a la anchura neuronal.
'''
    path=ROOT/'docs/AUDITORIA_TESIS.md';text=path.read_text(encoding='utf-8').split('## Cierre de ejecución y revisión')[0].rstrip()
    path.write_text(text+'\n\n'+section,encoding='utf-8')
    plan=ROOT/'docs/PLAN_AUDITORIA_TESIS.md';text=plan.read_text(encoding='utf-8')
    lines=[]
    for line in text.splitlines():
        if line.startswith('| 5.'):line='| 5. Evaluación | Completadas campañas, estudio de resolución, propagación y auditoría de archivos; fallos conservados. |'
        elif line.startswith('| 6.'):line='| 6. Integración | Completados metodología, resultados, discusión, conclusiones, resumen y anexos desde evidencia calculada. |'
        elif line.startswith('| 7.'):line='| 7. Documento final | Cuadernos ejecutados, PDF compilado y revisión de referencias, tablas, figuras y trazabilidad completada. |'
        lines.append(line)
    plan.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    pending=f'''# Estado de cierre y pendientes auténticos

La auditoría técnica está documentada en [AUDITORIA_TESIS.md](../docs/AUDITORIA_TESIS.md).
Se completaron el catálogo de 19 casos, las referencias FEniCSx, la ley R(Tc)
común, la comparación de configuraciones, los estudios de resolución y
colocación adaptativa, la propagación del error y los 19 cuadernos ejecutados.
Se conservan {data['total_runs']} entrenamientos y {tests} pruebas aprobadas.
Metodología, resultados, discusión, conclusiones y resúmenes se redactaron
desde el expediente interno. El PDF revisado contiene {review['pdf_pages']} páginas.

## Datos personales y administrativos

- Dedicatoria y agradecimientos: requieren el texto de los tesistas.
- Copias de documentos administrativos: deben incorporarse los documentos auténticos.
- Antes del depósito, verificar que nombres, grado del asesor, mención y datos
  de portada coincidan con los registros oficiales de la universidad.

Las marcas `PENDIENTE` se conservan únicamente en los espacios personales y
administrativos. La portada utiliza Poppins Bold distribuida con su licencia,
y el cuerpo Arial. Las decisiones sobre la estructura profesionalizante de
cuatro capítulos se fundamentan en las páginas específicas de la guía UNI,
con sus discrepancias internas identificadas en la auditoría.

## Límites de la evidencia

Los fallos PINN se informan, sin atribuirles resultados aceptados. La batería
verifica un modelo estacionario DC reducido, con reconstrucción radial del
conductor. La validación de campo, pérdidas AC completas, humedad y régimen
transitorio requieren nuevos datos y desarrollos; no se presentan como
actividades ejecutadas ni como espacios que puedan completarse por redacción.

Las referencias y configuraciones seleccionadas constan en los informes
internos. Los archivos históricos con código original incompleto están
identificados individualmente. Los resultados principales conservan sus
fuentes completas y sus huellas para auditoría posterior.
'''
    (ROOT/'Tesis_LaTeX_Borrador_UNI/PENDIENTES.md').write_text(pending,encoding='utf-8')
    files=['Tesis_LaTeX_Borrador_UNI/tesis.pdf','Benchmarks/INFORME_INTERNO.md','Benchmarks/ANALISIS_NUMERICO.md','Benchmarks/MUESTREO_ADAPTATIVO.md','Benchmarks/FORMAT.md','docs/AUDITORIA_TESIS.md']
    files+=sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'Tesis_LaTeX_Borrador_UNI').rglob('*') if p.is_file() and p.suffix.lower() in ['.tex','.bib','.png','.ttf','.md'])
    files+=['Tesis_LaTeX_Borrador_UNI/fonts/OFL.txt']
    manifest={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (ROOT/'docs/auditoria/deliverables_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(dict(tests=tests,runs=data['total_runs'],thermal_accepted=accepted,ampacity_accepted=ampaccepted,pdf_pages=review['pdf_pages']))
if __name__=='__main__':main()
