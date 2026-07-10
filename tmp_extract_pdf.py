from pathlib import Path
from pypdf import PdfReader
p = Path(r'Plan/GUÍA N°01 Plan.pdf')
r = PdfReader(str(p))
print('pages', len(r.pages))
for i in range(min(20, len(r.pages))):
    text = r.pages[i].extract_text() or ''
    print(f'--- PAGE {i+1} ---')
    print(text[:7000])
    print()
