import fitz, os
out = os.path.join(os.environ['TEMP'], 'audit_gen', 'pages'); os.makedirs(out, exist_ok=True)
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
for pdf, tag in [('Plan/guias/GUÍA N°01 Plan.pdf', 'g01'), ('matriz_consistencia/GUIA No.2 Desarrollo.pdf', 'g02')]:
    d = fitz.open(pdf)
    for i, pg in enumerate(d):
        z = 2200 / pg.rect.width
        pg.get_pixmap(matrix=fitz.Matrix(z, z)).save(os.path.join(out, f'{tag}_p{i+1:02d}.png'))
    print(tag, len(d))
