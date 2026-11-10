import fitz, re, os
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
CM = 72/2.54
def pages(path, label, upto=None):
    d = fitz.open(path)
    print('=====', label, len(d), 'pages')
    for pg in d:
        blocks = pg.get_text('blocks')
        num = None
        for b in blocks:
            if b[1] < 2.6*CM and b[0] > 15*CM and re.fullmatch(r'\s*[ivxlcdm\d]+\s*', b[4], flags=re.I):
                num = b[4].strip()
        body = [b for b in blocks if b[1] >= 2.6*CM and b[4].strip()]
        first = body[0][4].strip().replace('\n', ' ')[:70] if body else ''
        flag = ''
        if re.match(r'(CAP[IÍ]TULO|INTRODUCCI|RESUMEN|ABSTRACT|REFERENCIAS|ANEXO|DATOS GENERALES|DEDICATORIA|AGRADECIMIENTOS|COPIA DE|[ÍI]NDICE|LISTA)', first, flags=re.I):
            flag = '<<'
        if flag or pg.number < 12 or (upto and pg.number < upto):
            print(f'pdf {pg.number+1:3d} | impreso {str(num):5s} | {first} {flag}')
pages('Tesis_LaTeX_Borrador_UNI/tesis.pdf', 'TESIS')
pages('Plan/Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev2.pdf', 'PLAN')
# PENDIENTE en el Plan
d = fitz.open('Plan/Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev2.pdf')
for i in (3, 24, 25):
    t = d[i].get_text()
    for m in re.finditer(r'PENDIENTE', t):
        print('Plan pdf', i+1, '...', t[max(0, m.start()-120):m.end()+120].replace('\n', ' '))
d = fitz.open('Tesis_LaTeX_Borrador_UNI/tesis.pdf')
for i in (47,):
    t = d[i].get_text()
    for m in re.finditer(r'PENDIENTE', t):
        print('Tesis pdf', i+1, '...', t[max(0, m.start()-200):m.end()+160].replace('\n', ' '))
