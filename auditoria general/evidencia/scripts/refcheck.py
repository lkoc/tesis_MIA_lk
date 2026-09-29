import fitz, collections, os
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
d = fitz.open('Tesis_LaTeX_Borrador_UNI/tesis.pdf')
CM = 72/2.54
pg = d[54]
xs = collections.Counter(); ys = []
for b in pg.get_text('dict')['blocks']:
    for l in b.get('lines', []):
        xs[round(l['bbox'][0]/CM, 2)] += 1; ys.append(l['bbox'][1])
print('pdf 55 (referencias): distribución x0 de líneas (cm):', xs.most_common(6))
ys = sorted(ys); dif = [round(b-a, 1) for a, b in zip(ys, ys[1:]) if 8 < b-a < 25]
print('interlineado típico (pt):', collections.Counter(dif).most_common(4), '| cuerpo 1.5x11pt≈19.8pt; simple≈13.2pt')
pg = d[19]
ys = sorted(l['bbox'][1] for b in pg.get_text('dict')['blocks'] for l in b.get('lines', []))
dif = [round(b-a, 1) for a, b in zip(ys, ys[1:]) if 8 < b-a < 30]
print('cuerpo pdf 20 interlineado (pt):', collections.Counter(dif).most_common(4))
t = d[16].get_text()[:400].replace('\n', ' | ')
print('pdf 17:', t)
# resumen / abstract palabras
import re
for n in (12, 13):
    txt = d[n].get_text()
    print('pdf', n+1, 'palabras', len(txt.split()), '|', txt[:200].replace('\n', ' '))
