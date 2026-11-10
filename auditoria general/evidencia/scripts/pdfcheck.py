import fitz, re, sys, os, collections
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
CM = 72/2.54
def analyze(path, label):
    d = fitz.open(path)
    print(f'===== {label}: {path} =====')
    print('pages:', len(d), '| size pt:', d[0].rect.width, d[0].rect.height, '(A4=595.3x841.9)')
    print('metadata:', {k: v for k, v in d.metadata.items() if v})
    fonts = collections.Counter()
    for pg in d:
        for f in pg.get_fonts():
            fonts[f[3]] += 1
    print('fonts (name: pages):', dict(fonts.most_common(15)))
    # margins on body pages (skip first 3 pages)
    xs0, xs1, ys0, ys1 = [], [], [], []
    for i, pg in enumerate(d):
        if i < 3: continue
        for b in pg.get_text('blocks'):
            x0, y0, x1, y1, t = b[0], b[1], b[2], b[3], b[4]
            if not t.strip(): continue
            # ignore the page-number block near top
            if y1 < 3*CM - 2 and len(t.strip()) < 8: continue
            xs0.append(x0); xs1.append(x1); ys0.append(y0); ys1.append(y1)
    W, H = d[0].rect.width, d[0].rect.height
    import statistics
    print('text bbox (cm): left min %.2f | right margin min %.2f | top min %.2f | bottom margin min %.2f' % (min(xs0)/CM, (W-max(xs1))/CM, min(ys0)/CM, (H-max(ys1))/CM))
    # page-number position
    pos = []
    for i, pg in enumerate(d[:12] + d[30:32] if len(d) > 32 else d):
        blocks = [b for b in pg.get_text('blocks') if re.fullmatch(r'\s*[ivxlc\d]+\s*', b[4])]
        for b in blocks:
            pos.append((pg.number+1, round(b[0]/CM, 2), round(b[1]/CM, 2), b[4].strip()))
    print('page-number blocks (page, x cm, y cm, text):', pos[:14])
    # pendiente
    pend = [pg.number+1 for pg in d if 'PENDIENTE' in pg.get_text()]
    print('pages containing PENDIENTE:', pend)
    words_total = sum(len(pg.get_text().split()) for pg in d)
    print('total words:', words_total)
    # search 'Página' before number
    pagw = [pg.number+1 for pg in d if re.search(r'P[áa]gina\s+\d', pg.get_text())]
    print("pages with 'Página <n>':", pagw[:10])
    return d
def section_words(d, start_pat, end_pat):
    txt = '\n'.join(pg.get_text() for pg in d)
    m = re.search(start_pat, txt)
    if not m: return None
    rest = txt[m.end():]
    e = re.search(end_pat, rest)
    seg = rest[:e.start()] if e else rest[:6000]
    return seg
if __name__ == '__main__':
    t = analyze('Tesis_LaTeX_Borrador_UNI/tesis.pdf', 'TESIS')
    for name, sp, ep in [('RESUMEN', r'\nRESUMEN\s*\n', r'\nABSTRACT\s*\n'), ('ABSTRACT', r'\nABSTRACT\s*\n', r'\n(ÍNDICE|INDICE|CONTENIDO|INTRODUCCI)')]:
        seg = section_words(t, sp, ep)
        if seg:
            print(name, 'words:', len(seg.split()))
            print(seg[:2500].replace('\n', ' '))
    p = analyze('Plan/Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev2.pdf', 'PLAN')
    r0 = analyze('Plan/old/versiones_pdf/Plan_Tesis_Grupo91_Koc_Melendez_cables_pinn_rev0.pdf', 'PLAN_REV0 (entregado)')
