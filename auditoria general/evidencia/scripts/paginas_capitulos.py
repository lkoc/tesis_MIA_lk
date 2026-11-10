import fitz, sys, re
p = sys.argv[1]
d = fitz.open(p)
print('paginas', len(d))
# ultima pagina y paginas con encabezado de capitulo
for i, pg in enumerate(d):
    t = pg.get_text().strip().splitlines()
    if not t: continue
    head = ' | '.join(t[:3])[:110]
    if re.match(r'^(CAP[IÍ]TULO|INTRODUCCI|REFERENCIAS|ANEXOS|RESUMEN|ABSTRACT|DATOS GENERALES|[ÍI]ndice|Lista de)', t[0].strip(), re.I) or re.match(r'^(CAP[IÍ]TULO)', ' '.join(t[:2]).strip(), re.I):
        print(i + 1, '::', head)
