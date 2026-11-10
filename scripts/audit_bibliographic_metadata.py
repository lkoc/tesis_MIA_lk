"""Read-only DOI metadata audit against Crossref; no automatic title rewriting."""
from pathlib import Path
import concurrent.futures
import difflib
import json
import re
import requests
import truststore
truststore.inject_into_ssl()

def check(entry):
    key,body=entry
    doi=re.search(r'\bdoi\s*=\s*\{([^}]+)\}',body)
    title=re.search(r'\btitle\s*=\s*\{(.+)\}',body)
    if not doi:return dict(key=key,status='no_doi')
    value=doi[1]
    try:
        response=requests.get('https://api.crossref.org/works/'+value,timeout=35)
        response.raise_for_status();m=response.json()['message']
        actual=m.get('title',[''])[0]
        clean=lambda x:re.sub(r'[^a-z0-9 ]','',x.lower())
        score=difflib.SequenceMatcher(None,clean(title[1] if title else ''),clean(actual)).ratio()
        return dict(key=key,doi=value,status='resolved',title_local=title[1] if title else '',title_crossref=actual,similarity=score,authors=m.get('author',[]),published=m.get('published'),url=m.get('URL'))
    except Exception as e:return dict(key=key,doi=value,status='unresolved',error=str(e))

if __name__=='__main__':
    text=Path('Tesis_LaTeX_Borrador_UNI/referencias.bib').read_text(encoding='utf-8')
    entries=re.findall(r'@\w+\{([^,]+),\s*(.*?)(?=\n@|\Z)',text,re.S)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(check,entries))
    Path('docs/auditoria/doi_metadata.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Entries:',len(rows),'Resolved:',sum(r['status']=='resolved' for r in rows))
    print(json.dumps([r for r in rows if r['status']=='unresolved' or r.get('similarity',1)<.8],ensure_ascii=False,indent=2))
