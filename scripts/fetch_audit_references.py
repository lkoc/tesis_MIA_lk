from pathlib import Path
import hashlib
import json
import requests
import truststore
truststore.inject_into_ssl()
items={
 'benchmark_papers/Raissi_2019_JCP_verificado.pdf':'https://www.aer.com/siteassets/21data/raissietal2019.pdf',
 'benchmark_papers/Wang_2020_gradientes.pdf':'https://arxiv.org/pdf/2001.04536',
 'Tesis_LaTeX_Borrador_UNI/fonts/Poppins-Bold.ttf':'https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Bold.ttf',
 'Tesis_LaTeX_Borrador_UNI/fonts/OFL.txt':'https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/OFL.txt',
}
rows=[]
for dest,url in items.items():
    p=Path(dest);p.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not p.exists():
            r=requests.get(url,timeout=60);r.raise_for_status()
            if p.suffix=='.pdf' and not r.content.startswith(b'%PDF'):raise ValueError('Not a PDF')
            p.write_bytes(r.content)
        rows.append(dict(path=dest,url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        print(dest,p.stat().st_size,flush=True)
    except Exception as e:rows.append(dict(path=dest,url=url,error=str(e)));print(dest,str(e),flush=True)
Path('docs/auditoria/fuentes_descargadas.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
