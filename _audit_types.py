"""Lista todos los papers con tipo actual, tags, abstract y título para revisión."""
import pathlib, json, re

html = pathlib.Path(r'c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html').read_text(encoding='utf-8')
m = re.search(r'const NODES=(\[.*?\]);', html, re.DOTALL)
nodes = json.loads(m.group(1))

for n in sorted(nodes, key=lambda x: (x.get('paper_type',''), x.get('label',''))):
    pt   = n.get('paper_type', '') or '(sin tipo)'
    tags = ' | '.join(n.get('tags', []))
    abst = (n.get('abstract') or '')[:120].replace('\n', ' ')
    print(f"[{pt}]")
    print(f"  ID    : {n['id']}")
    print(f"  Título: {n['label']}")
    print(f"  Tags  : {tags}")
    print(f"  Abst  : {abst}")
    print()
