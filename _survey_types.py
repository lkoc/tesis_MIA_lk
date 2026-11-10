import re, json, pathlib, collections

html = pathlib.Path(r'c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html').read_text(encoding='utf-8')
m = re.search(r'const NODES=(\[.*?\]);', html, re.DOTALL)
nodes = json.loads(m.group(1))

tipos = collections.Counter()
for n in nodes:
    t = n.get('tipo') or n.get('type') or ''
    if t:
        tipos[('campo:tipo', t)] += 1
    tags = n.get('tags', []) or []
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        if 'tipo' in str(tag).lower():
            tipos[('tag', str(tag))] += 1

print('Tipos encontrados:')
for (src,k),v in sorted(tipos.items(), key=lambda x:-x[1]):
    print(f'  {v:>3}  [{src}]  {k}')
print(f'\nTotal nodos: {len(nodes)}')
print('\nCampos nodo 0:', list(nodes[0].keys()))
print('\nSample tags nodo 0:', nodes[0].get('tags'))
