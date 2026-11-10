"""Check executed notebooks and the compiled thesis; render review pages."""
from pathlib import Path
import sys,json,re,hashlib,base64
import fitz
import nbformat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases
ROOT=Path(__file__).resolve().parents[1]

def main():
    notebooks=[]
    for name in cases():
        p=ROOT/'Benchmarks/notebooks'/f'{name}.ipynb';n=nbformat.read(p,as_version=4)
        errors=[o for c in n.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
        images=sum('image/png' in o.get('data',{}) for c in n.cells if c.cell_type=='code' for o in c.get('outputs',[]))
        assert not errors and images>=5,p
        assert n.cells[-1].cell_type=='markdown' and n.cells[-1].source.startswith('## Referencias'),p
        assert all(c.execution_count is not None for c in n.cells if c.cell_type=='code'),p
        text='\n'.join(c.source for c in n.cells if c.cell_type=='markdown')
        assert re.search(r'\(20\d\d: [\d–, ]+\)',text),p
        if name in ['kim_layered','xlpe_backfill']:
            embedded={hashlib.sha256(base64.b64decode(o['data']['image/png'])).hexdigest() for c in n.cells if c.cell_type=='code' for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})}
            for suffix in ['adaptive','collocation']:
                plot=ROOT/'Tesis_LaTeX_Borrador_UNI/imagenes/benchmarks'/f'{name}_{suffix}.png'
                assert hashlib.sha256(plot.read_bytes()).hexdigest() in embedded,('Stale embedded figure',name,suffix)
        notebooks.append(dict(case=name,cells=len(n.cells),images=images,errors=len(errors)))
    thesis=ROOT/'Tesis_LaTeX_Borrador_UNI';pdf=fitz.open(thesis/'tesis.pdf')
    texts=[p.get_text() for p in pdf];all_text='\n'.join(texts)
    for phrase in ['Completar el resumen','Redactar las conclusiones','Traducir la versión final','no se formulan conclusiones anticipadas']:
        assert phrase not in all_text,phrase
    log=(thesis/'tesis.log').read_text(encoding='utf-8',errors='replace')
    problems=[line for line in log.splitlines() if ('Overfull' in line or 'undefined' in line.lower() or 'LaTeX Error' in line or 'Missing character:' in line)]
    markers=['RESUMEN','SELECCIÓN DE ESQUEMA','Selección nominal acoplada','Comparación de configuraciones para corriente','COLOCACIÓN ADAPTATIVA','COMPARACIÓN DEL MUESTREO ADAPTATIVO','Distribución inicial y final','CONCLUSIONES Y RECOMENDACIONES','REFERENCIAS BIBLIOGRÁFICAS','PROTOCOLO DE REPRODUCCIÓN']
    pages={0}
    for phrase in markers:
        candidates=[i for i,t in enumerate(texts) if phrase in t and (i>15 or phrase=='RESUMEN')]
        if candidates:pages.add(candidates[-1] if phrase=='RESUMEN' else candidates[0])
    folder=ROOT/'docs/auditoria/pdf_review';folder.mkdir(exist_ok=True)
    for i in sorted(pages):pdf[i].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(folder/f'final_{i+1:03d}.png')
    from PIL import Image,ImageDraw
    contacts=[]
    for first in range(0,len(pdf),12):
        sheet=Image.new('RGB',(1440,1650),'#dddddd');draw=ImageDraw.Draw(sheet)
        for j in range(min(12,len(pdf)-first)):
            page=pdf[first+j];pix=page.get_pixmap(matrix=fitz.Matrix(.57,.57))
            tile=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
            tile.thumbnail((340,495))
            x=(j%4)*360+10;y=(j//4)*550+25
            sheet.paste(tile,(x,y));draw.text((x,y-18),str(first+j+1),fill='black')
        target=folder/f'final_contact_{first//12+1:02d}.png';sheet.save(target);contacts.append(target.name)
    report=dict(pdf_pages=len(pdf),pdf_sha256=hashlib.sha256((thesis/'tesis.pdf').read_bytes()).hexdigest(),notebooks=notebooks,
        rendered_pages=[i+1 for i in sorted(pages)],contact_sheets=contacts,latex_layout_or_reference_problems=problems,
        administrative_pending=['dedicatoria','agradecimientos','copia de documentos auténticos'])
    (ROOT/'docs/auditoria/final_review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if problems:raise SystemExit('Inspect layout/reference warnings before finalizing')
if __name__=='__main__':main()
