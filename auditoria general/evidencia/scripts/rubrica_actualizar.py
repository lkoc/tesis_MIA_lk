# -*- coding: utf-8 -*-
"""R-11: repite la rubrica 8.2 sobre el Plan vigente y restituye los acentos del Excel.

Solo escribe valores de celdas; conserva formato y celdas combinadas. Uso:
    python rubrica_actualizar.py <plan.pdf compilado en copia temporal>
"""
import sys, os, re
import fitz, openpyxl
sys.path.insert(0, os.path.dirname(__file__))
import bibcheck as b

ROOT = r'c:\usr\ths_mia_fiis\tesis_MIA_lk'
XLSX = os.path.join(ROOT, 'Plan', 'evaluacion_formato_8_2_plan_tesis.xlsx')
BIB = os.path.join(ROOT, 'Plan', 'referencias_plan_tesis.bib')
pdf = fitz.open(sys.argv[1])
toc = pdf.get_toc()
NPAG = len(pdf)


def pag(titulo):
    for lev, t, p in toc:
        if t.strip().upper() == titulo.upper():
            return p
    raise KeyError(titulo)


P = {k: pag(k) for k in ['MARCO TEÓRICO', 'METODOLOGÍA DE LA INVESTIGACIÓN', 'ADMINISTRACIÓN DEL PLAN DE TESIS', 'REFERENCIAS', 'ANEXOS',
                        'ANTECEDENTES BIBLIOGRÁFICOS', 'CRONOGRAMA', 'PRESUPUESTO', 'FINANCIAMIENTO', 'DIAGNÓSTICO', 'UBICACIÓN DEL PROYECTO',
                        'PLANTEAMIENTO DEL PROBLEMA', 'OBJETIVOS DE LA INVESTIGACIÓN']}

ent, _ = b.parse_bib(BIB)
n_ent = len(ent)
anios = []
for k, v in ent.items():
    y = re.sub(r'[^0-9]', '', v['fields'].get('year', v['fields'].get('date', ''))[:6])[:4]
    if y:
        anios.append(int(y))
n_rec = sum(1 for y in anios if y >= 2021)
n_doi = sum(1 for v in ent.values() if 'doi' in v['fields'])
n_isbn = sum(1 for v in ent.values() if 'isbn' in v['fields'])
n_url = sum(1 for v in ent.values() if 'url' in v['fields'])

wb = openpyxl.load_workbook(XLSX)
ws = wb['Formato 8.2']

TITULO = ('DISEÑO Y EVALUACIÓN DE UNA RED NEURONAL INFORMADA POR FÍSICA PARA ESTIMAR LA TEMPERATURA Y LA AMPACIDAD DE CABLES '
          'ELÉCTRICOS ENTERRADOS EN ENTORNOS HETEROGÉNEOS')
ws['A1'] = '8.2 FORMATO DE EVALUACIÓN DEL PLAN DE TESIS VÁLIDO PARA MAESTRÍA DE ESPECIALIZACIÓN'
ws['A3'] = 'Título de la Tesis'
ws['C3'] = TITULO
ws['A6'] = 'Correo del tesista'; ws['C6'] = 'No consignado en el plan revisado'
ws['E6'] = 'Celular del tesista'; ws['F6'] = 'No consignado'
ws['A8'] = 'ASPECTOS DE EVALUACIÓN'; ws['D8'] = 'SÍ'; ws['E8'] = 'NO'; ws['F8'] = 'OBSERVACIONES'

SECCIONES = {9: '1. TÍTULO', 12: '2. ÍNDICE DE CONTENIDO', 14: '3. INTRODUCCIÓN', 17: '4. CONTENIDOS', 20: '5. EL PROBLEMA: FORMULACIÓN DEL PROBLEMA',
             24: '7. ALCANCES DE LA INVESTIGACIÓN', 26: '8. JUSTIFICACIÓN', 29: '9. METODOLOGÍA', 32: '10. MARCO TEÓRICO',
             37: '11. UBICACIÓN DEL PROYECTO DE INVESTIGACIÓN', 39: '12. ESQUEMA DE CONTENIDO', 41: '13. CRONOGRAMA', 43: '14. PRESUPUESTO',
             45: '15. FINANCIAMIENTO', 47: '16. FUENTES DE INFORMACIÓN (Bibliografía)', 50: '17. OTROS', 54: '18. ANEXOS'}
for r, t in SECCIONES.items():
    ws.cell(r, 1).value = t

# (fila, criterio, cumple, observacion)
C = [
 (10, '¿El enunciado del título de la tesis es claro y preciso, orientado a dar respuesta al problema?', True,
  '[Menor] El título identifica artefacto, método, variables de salida y objeto de estudio. Observación menor: supera la recomendación institucional de brevedad (24 palabras). '
  'Se mantiene el título más reciente; aún no se ha enviado a la Unidad de Posgrado. Evidencia: Carátula; PDF p. 1.'),
 (11, '¿Refleja los objetivos el tema abordado?', True,
  '[Sin observación crítica] Refleja el objetivo general: diseñar y evaluar una PINN 2D para estimar temperatura y ampacidad en entornos heterogéneos. Evidencia: Capítulo II, Objetivo general; Anexo 1.'),
 (13, '¿El índice está estructurado de acuerdo a la Guía?', True,
  f'[Sin observación crítica] El índice sigue la Guía N.° 01: Capítulo I Planteamiento del problema (1.1 Diagnóstico; 1.2 Identificación y descripción del problema, con antecedentes, formulación, justificación y alcances), II Objetivos, '
  f'III Marco teórico, IV Metodología (4.1 a 4.8), V Administración del plan; Referencias y Anexos como bloques propios. Evidencia: índice del PDF ({NPAG} páginas): Marco teórico p. {P["MARCO TEÓRICO"]}, '
  f'Metodología p. {P["METODOLOGÍA DE LA INVESTIGACIÓN"]}, Capítulo V p. {P["ADMINISTRACIÓN DEL PLAN DE TESIS"]}, Referencias p. {P["REFERENCIAS"]}, Anexos p. {P["ANEXOS"]}.'),
 (15, '¿Explica el interés del proyecto, las razones de la elección del tema, destaca los problemas que motivaron la elección del proyecto, su ubicación en contexto y describe qué es lo que busca lograr con los objetivos de la tesis?', True,
  f'[Menor] La sección 1.1 Diagnóstico (p. {P["DIAGNÓSTICO"]}) explica el problema térmico, la brecha entre entorno heterogéneo y representación homogénea, el uso de PINN y el vínculo con DSR. Evidencia: Capítulo I, 1.1.'),
 (16, '¿Indica las fuentes de investigación utilizadas para obtener el tema de tesis?', True,
  '[Sin observación crítica] El texto cita fuentes clásicas, normas, estudios recientes y metodología DSR desde el diagnóstico. La trazabilidad bibliográfica es suficiente. Evidencia: citas a Neher, IEC, CIGRÉ, Enescu, Raissi y Gregor, entre otras.'),
 (18, '¿El Plan de Tesis describe la justificación y referencias bibliográficas?', True,
  '[Sin observación crítica] La justificación se divide en práctica, tecnológica, metodológica y científica y académica, con citas pertinentes. Evidencia: Capítulo I, 1.2.3.1 Justificación.'),
 (19, '¿En el Plan de tesis se especifica la evolución de los antecedentes de la variable dependiente ordenados desde el más antiguo hasta el más reciente?', True,
  f'[Sin observación crítica] La sección 1.2.1 (p. {P["ANTECEDENTES BIBLIOGRÁFICOS"]}) incluye una tabla de evolución cronológica de 1957 a 2025 (19 fuentes) y agrupa los antecedentes por tema en orden cronológico; '
  'declara qué fuentes son clásicas, normas o fundamentos metodológicos. Evidencia: Tabla de evolución cronológica de los antecedentes.'),
 (21, '¿El Diagnóstico describe la problemática indicando el problema principal y los problemas secundarios?', True,
  '[Sin observación crítica] La sección 1.1 Diagnóstico y su subsección Problemática sustentan la brecha; el problema general y los cuatro problemas específicos están formulados en 1.2.2. Evidencia: Capítulo I, 1.1 y 1.2.2.'),
 (22, '¿En la formulación del problema, el Investigador ha propuesto la variable independiente de la solución del problema?', True,
  '[Sin observación crítica] La conductividad térmica espacialmente variable k(x,y), sus factores y el artefacto PINN están definidos como elementos independientes o de intervención. Evidencia: Dimensiones operativas; matriz de operacionalización; diseño discreto de casos.'),
 (23, '¿Formula correctamente el objetivo principal y objetivos secundarios de la tesis?', True,
  '[Sin observación crítica] El objetivo general y los cuatro objetivos específicos son coherentes con el problema, la matriz de consistencia y la metodología DSR. Evidencia: Capítulo II, 2.1 y 2.2; Anexo 1.'),
 (25, '¿Describe con claridad los alcances de la tesis?', True,
  '[Sin observación crítica] Los alcances temático, metodológico, espacial, temporal y de producto están definidos y las limitaciones declaradas. Evidencia: Capítulo I, 1.2.3.2 Alcances y limitaciones.'),
 (27, '¿Presenta justificación con sustento de forma clara y precisa?', True,
  '[Sin observación crítica] La justificación tiene sustento bibliográfico y conecta la necesidad práctica con el aporte tecnológico y metodológico. Evidencia: Capítulo I, 1.2.3.1.'),
 (28, '¿El tipo de justificación presentada está acorde a su tesis?', True,
  '[Sin observación crítica] Los tipos de justificación son acordes con una tesis de especialización o profesionalizante: práctica, tecnológica, metodológica y académica. Evidencia: Capítulo I, 1.2.3.1.'),
 (30, '¿Describe de manera clara y concreta la metodología que aplica, para la solución del problema?', True,
  '[Sin observación crítica] La metodología DSR se explica con tipo, nivel, métodos de trabajo, población y muestra, tipo de diseño, técnicas e instrumentos y etapas (4.1 a 4.8). No afirma mediciones de campo directas. Evidencia: Capítulo IV.'),
 (31, '¿Presenta los instrumentos de recolección de datos y las técnicas de análisis de la información?', True,
  '[Sin observación crítica] Las secciones 4.6 (recolección) y 4.7 (análisis y procesamiento) usan los títulos de la Guía y reúnen evidencia, instrumentos, trazabilidad, métricas y criterios de verificación. Evidencia: Capítulo IV, 4.6 y 4.7.'),
 (33, '¿Describe y justifica los antecedentes necesarios relacionados con el problema y el tema?', True,
  '[Sin observación crítica] Los antecedentes cubren ampacidad, heterogeneidad térmica, PINN y DSR, y se ubican en la sección 1.2.1 según la Guía. Evidencia: Capítulo I, 1.2.1.'),
 (34, '¿Presenta y describe las Bases teóricas necesarias que fundamenten su relación con la propuesta de tesis?', True,
  '[Sin observación crítica] Las bases explican conducción, PDE, ampacidad, heterogeneidad, métodos de cálculo y PINN, conectadas con la metodología y los anexos. Evidencia: Capítulo III, 3.1 Bases teóricas.'),
 (35, '¿Presenta claramente las teorías y conocimiento que sirven para base de estudio?', True,
  '[Sin observación crítica] Se explican la teoría de conducción, la formulación estacionaria y transitoria, los métodos FEM, FDM y FVM y la estructura de PINN. Evidencia: Capítulo III; fórmulas y figuras.'),
 (36, '¿Son relevantes los conceptos y definiciones de la definición de términos?', True,
  '[Sin observación crítica] La definición de términos es amplia, técnica y relacionada con variables, método y alcance. Evidencia: Capítulo III, 3.2.'),
 (38, '¿Justifica la localización, ubicación donde se desarrollará el proyecto de investigación y describe brevemente las características del contexto y entorno?', True,
  f'[Menor] Existe la subsección «Ubicación del proyecto» (1.2.1, p. {P["UBICACIÓN DEL PROYECTO"]}), que sitúa el trabajo entre el cálculo térmico de cables, las PINN y la ciencia del diseño y en el programa de la FIIS-UNI; el lugar de desarrollo consta en Datos generales. '
  'Observación menor: el contexto es institucional y computacional, sin una instalación real. Evidencia: 1.2.1; Datos generales.'),
 (40, '¿Presenta adecuadamente la estructura de su tesis de acuerdo a la Guía?', True,
  '[Sin observación crítica] Orden: I Problema, II Objetivos, III Marco teórico, IV Metodología, V Administración; Referencias y Anexos separados de la numeración de capítulos. Las hipótesis y las dimensiones operativas se conservan en el Capítulo II como adaptación por trazabilidad. Evidencia: índice del PDF.'),
 (42, '¿Presenta correctamente el cronograma de actividades?', True,
  f'[Menor] El Capítulo V, 5.1 (p. {P["CRONOGRAMA"]}) presenta un diagrama de Gantt de octubre de 2026 a junio de 2027 con un bloque para julio a septiembre de 2026 (cumplido) y los hitos H1 a H6. Observación: las fechas posteriores a septiembre de 2026 son una programación que el asesor debe validar. Evidencia: Tabla del cronograma.'),
 (44, '¿La administración del Plan de tesis, cuenta con el Presupuesto?', True,
  f'[Mayor] El Capítulo V, 5.2 (p. {P["PRESUPUESTO"]}) presenta el presupuesto por los rubros a) equipos, b) materiales e insumos, c) servicios y d) viáticos y publicación. Observación: varios montos figuran «Por confirmar» y deben completarse con tarifas y cotizaciones antes de la entrega. Evidencia: Tabla del presupuesto.'),
 (46, '¿La administración del Plan de tesis, considera financiamiento e indica el tipo?', True,
  f'[Sin observación crítica] El Capítulo V, 5.3 (p. {P["FINANCIAMIENTO"]}) declara autofinanciamiento por los tesistas y que no se han solicitado fondos concursables ni de terceros. Evidencia: 5.3.'),
 (48, '¿Cumple las normas APA? Versión vigente.', True,
  '[Menor] El documento usa biblatex-apa con citas autor-año y las Referencias figuran como entrada del índice. Observación: conviene una revisión manual final del estilo APA institucional. Evidencia: preámbulo LaTeX (style=apa); bibliografía generada.'),
 (49, '¿Las Referencias está actualizada en un máximo de 5 años de (salvo los textos clásicos)?', True,
  f'[Menor] La base citada tiene {n_ent} entradas y {n_rec} son de 2021 o posteriores; las anteriores son clásicos, normas o fundamentos metodológicos, lo que se declara en 1.2.1. Evidencia: BibTeX del Plan ({n_ent} entradas; {n_doi} DOI, {n_isbn} ISBN, {n_url} URL; años {min(anios)}-{max(anios)}).'),
 (51, '¿La redacción gramatical del Plan de tesis es correcta, coherente y cumple las normas APA? Versión vigente', True,
  '[Menor] La redacción es académica, formal, técnica y en tercera persona, con hilo conductor y citas. Observación: conviene una revisión final de estilo tras cerrar los montos del presupuesto y el cronograma. Evidencia: documento completo.'),
 (52, '¿El documento está presentado con numeración correcta?', True,
  f'[Sin observación crítica] Numeración romana en preliminares y arábiga en el cuerpo; sin número visible en el primer folio de cada capítulo; niveles de índice hasta el cuarto nivel; anexos numerados «Anexo 1» a «Anexo 5» y tablas de anexos «A.n». Evidencia: PDF de {NPAG} páginas.'),
 (53, '¿El documento de tesis cumple con los estándares de presentación del proyecto de tesis?', True,
  '[Menor] Cumple los criterios formales principales: A4, márgenes de 3 cm y 2,5 cm, Arial 11, numeración superior derecha, sin sangría y carátula con Poppins 22 y Arial 16, 14 y 12. Observación: el interlineado medido con \\onehalfspacing es 16,4 pt; en la tesis se adoptó 19,0 pt (equivalente a 1,5 líneas de Word) y podría aplicarse también al Plan. Evidencia: preámbulo LaTeX; carátula.'),
 (55, '¿Incluye Anexos y los menciona?', True,
  '[Sin observación crítica] Incluye Anexo 1 a Anexo 5 (matriz de consistencia, operacionalización, diseño discreto de casos, revisión documental y repositorio digital) y los menciona en el texto; los anexos de matriz y diseño están en hoja apaisada. Observación menor: el repositorio digital conserva una dirección de Google Drive por sustituir por una estable. Evidencia: Anexos.'),
]
for fila, crit, ok, obs in C:
    ws.cell(fila, 1).value = crit
    ws.cell(fila, 4).value = 'X' if ok else None
    ws.cell(fila, 5).value = None if ok else 'X'
    ws.cell(fila, 6).value = obs

n = len(C)
cumple = sum(1 for _, _, ok, _ in C if ok)

# ---- Resumen
rs = wb['Resumen']
rs['A1'] = 'DICTAMEN DEL REVISOR'
rs['A3'] = 'Resultado global'
rs['B3'] = 'CUMPLE LA ESTRUCTURA REQUERIDA; PENDIENTES MENORES Y UN PENDIENTE MAYOR (MONTOS DEL PRESUPUESTO) ANTES DE LA ENTREGA'
rs['A4'] = 'Criterios evaluados'; rs['B4'] = n
rs['A5'] = 'Cumple'; rs['B5'] = cumple
rs['A6'] = 'No cumple'; rs['B6'] = n - cumple
rs['A7'] = 'Fecha de evaluación'; rs['B7'] = '2026-09-29'
rs['A8'] = 'Documento evaluado'; rs['B8'] = 'Plan/plan_tesis_cables_pinn.tex compilado en copia temporal (PDF de ' + str(NPAG) + ' páginas)'
rs['A9'] = 'Formato usado'; rs['B9'] = 'GUÍA N.° 01 Plan.pdf, sección 8.2, páginas impresas 54-56 del formato.'
rs['A11'] = 'Observaciones abiertas antes de la entrega'
ABIERTAS = [
 ('¿La administración del Plan de tesis, cuenta con el Presupuesto?', 'Mayor',
  'Los montos marcados «Por confirmar» del presupuesto deben completarse con tarifas y cotizaciones reales.', 'Capítulo V, 5.2; Tabla del presupuesto.'),
 ('¿Presenta correctamente el cronograma de actividades?', 'Menor',
  'La programación posterior a septiembre de 2026 debe validarse con el asesor.', 'Capítulo V, 5.1; Tabla del cronograma.'),
 ('¿El enunciado del título de la tesis es claro y preciso, orientado a dar respuesta al problema?', 'Menor',
  'Se mantiene el título más reciente; no se ha enviado a la Unidad de Posgrado.', 'Carátula; PENDIENTES.md (D3).'),
 ('¿El documento de tesis cumple con los estándares de presentación del proyecto de tesis?', 'Menor',
  'El interlineado medido es 16,4 pt; la tesis adoptó 19,0 pt (1,5 líneas de Word con Arial 11) y el Plan podría hacerlo.', 'Preámbulo LaTeX; PENDIENTES.md.'),
 ('¿Incluye Anexos y los menciona?', 'Menor',
  'El anexo del repositorio digital conserva una dirección de Google Drive por sustituir por una estable.', 'Anexo 5.'),
]
for i, (a, sev, c, d) in enumerate(ABIERTAS):
    r = 12 + i
    rs.cell(r, 1).value = a; rs.cell(r, 2).value = sev; rs.cell(r, 3).value = c; rs.cell(r, 4).value = d
for r in range(12 + len(ABIERTAS), 19):
    for c in range(1, 5):
        rs.cell(r, c).value = None
rs['A20'] = 'Conclusión'
rs['B20'] = ('El Plan cumple la estructura institucional de la Guía N.° 01: Capítulos I a V, Referencias y Anexos, con antecedentes cronológicos, ubicación del proyecto, '
             'cronograma, presupuesto y financiamiento. Antes de la entrega deben completarse los montos del presupuesto, validarse el cronograma con el asesor, '
             'decidir el interlineado del Plan y sustituir la dirección de Google Drive del repositorio digital.')

# ---- Evidencia
ev = wb['Evidencia']
FILAS = [
 ('Elemento verificado', 'Resultado'),
 ('Páginas del PDF del plan', str(NPAG)),
 ('Orden del índice observado', 'DATOS GENERALES; PLANTEAMIENTO DEL PROBLEMA; OBJETIVOS; MARCO TEÓRICO; METODOLOGÍA; ADMINISTRACIÓN DEL PLAN DE TESIS; REFERENCIAS; ANEXOS'),
 ('Capítulo V Administración', f'Encontrado (p. {P["ADMINISTRACIÓN DEL PLAN DE TESIS"]})'),
 ('Cronograma', f'Encontrado (p. {P["CRONOGRAMA"]}); Gantt de octubre de 2026 a junio de 2027'),
 ('Presupuesto', f'Encontrado (p. {P["PRESUPUESTO"]}); rubros a a d; montos por confirmar'),
 ('Financiamiento', f'Encontrado (p. {P["FINANCIAMIENTO"]}); autofinanciamiento'),
 ('Referencias en índice', f'Aparecen como entrada del índice (p. {P["REFERENCIAS"]})'),
 ('Referencias BibTeX', f'{n_ent} entradas; {n_doi} DOI; {n_isbn} ISBN; {n_url} URL; años {min(anios)}-{max(anios)}'),
 ('Formato físico', 'A4; márgenes 3 cm superior e izquierdo y 2,5 cm derecho e inferior; Arial 11; interlineado \\onehalfspacing (16,4 pt); sin sangría'),
 ('Numeración e índice', 'tocdepth=4 y secnumdepth=4; sin número visible en el primer folio de cada capítulo'),
 ('Anexos', 'Matriz de consistencia, matriz de operacionalización, diseño discreto de casos, estrategia de revisión documental y repositorio digital (Anexo 1 a Anexo 5)'),
]
for i, (a, bb) in enumerate(FILAS):
    ev.cell(1 + i, 1).value = a
    ev.cell(1 + i, 2).value = bb
wb.save(XLSX)
print('ok', n, cumple, NPAG, n_ent, n_rec)
