import fitz, sys
for p in ['Plan/guias/GUÍA N°01 Plan.pdf','matriz_consistencia/GUIA No.2 Desarrollo.pdf','Plan/guias/class 08. Marco de Trabajo.pdf','Plan/Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev2.pdf','Tesis_LaTeX_Borrador_UNI/tesis.pdf','Plan/old/versiones_pdf/Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev0.pdf']:
    try:
        d = fitz.open(p)
    except Exception as e:
        print(p, 'ERR', e); continue
    txt = sum(len(pg.get_text()) for pg in d)
    print(p, 'pages', len(d), 'chars', txt, 'imgs', sum(len(pg.get_images()) for pg in d))
