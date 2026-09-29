import fitz, sys, os
from PIL import Image
out = os.path.join(os.environ['TEMP'], 'audit_gen')
def sheets(pdf, tag, cols=4, rows=3, w=520):
    d = fitz.open(pdf)
    per = cols*rows
    for s in range(0, len(d), per):
        ims = []
        for i in range(s, min(s+per, len(d))):
            pg = d[i]; z = w / pg.rect.width
            pix = pg.get_pixmap(matrix=fitz.Matrix(z, z))
            ims.append(Image.frombytes('RGB', (pix.width, pix.height), pix.samples))
        h = max(im.height for im in ims)
        sheet = Image.new('RGB', (cols*w, rows*h), 'white')
        for k, im in enumerate(ims):
            sheet.paste(im, ((k % cols)*w, (k//cols)*h))
        fn = os.path.join(out, f'{tag}_sheet_{s+1:02d}.png'); sheet.save(fn); print(fn)
def page(pdf, tag, n, w=1100):
    d = fitz.open(pdf); pg = d[n-1]; z = w / pg.rect.width
    pix = pg.get_pixmap(matrix=fitz.Matrix(z, z))
    fn = os.path.join(out, f'{tag}_p{n:02d}.png'); pix.save(fn); print(fn)
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'sheets': sheets(sys.argv[2], sys.argv[3])
    else:
        for n in sys.argv[4:]: page(sys.argv[2], sys.argv[3], int(n))
