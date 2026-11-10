import re
from collections import Counter

html = open(r'c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html', encoding='utf-8').read()
lines = html.splitlines()
print('Total lines:', len(lines))

# Check for duplicate node ids in NODES section
nodes_end = html.find('const EDGES')
nodes_section = html[:nodes_end]
all_ids = re.findall(r'"id":\s*(\d+),\s*"label"', nodes_section)
cnt = Counter(all_ids)
dups = {k: v for k, v in cnt.items() if v > 1}
print('Duplicate node ids:', dups if dups else 'none')
print('Total node entries:', len(all_ids))

# Find vis.Network / DataSet and showD
keywords = ['vis.Network', 'vis.DataSet', 'const nodesDS', 'const edgesDS',
            'function showD', 'const cld', 'zotero_cloud']
for i, l in enumerate(lines):
    for kw in keywords:
        if kw in l:
            print(f'L{i+1} [{kw}]: {l[:150]}')
            break
