import fitz, os, glob
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
for p in ['Plan/old/versiones_pdf/Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev0.pdf'] + glob.glob('Plan/env*rev*/02_Informe*.pdf'):
    d = fitz.open(p)
    md = d.metadata
    t = d[0].get_text().replace('\n', ' ')[:520]
    print(p, '| pp', len(d), '| creado', md.get('creationDate'), '| título meta:', md.get('title')[:140])
    print('    portada:', t[:400])
