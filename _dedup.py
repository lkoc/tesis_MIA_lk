"""
Remove duplicate nodes 16-31 from zotero_network.html
(they were inserted twice by the fix script running twice)
"""
import re
from pathlib import Path

html_path = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html")
html = html_path.read_text(encoding='utf-8')

# The NODES are on one line. Find the section up to const EDGES
edges_pos = html.find('const EDGES')
nodes_line = html[:edges_pos]

# Find positions of first and second occurrence of {"id": 16
# The pattern we want to remove is the second block from ,{"id":16 up to ,{"id":32
# Both blocks are `, {"id": 16, ...}` chunks

# Strategy: find ALL occurrences of '"id": 16,' in the nodes section
occurrences = [m.start() for m in re.finditer(r',\s*\{"id":\s*16\b', nodes_line)]
print(f'Occurrences of id=16: {len(occurrences)} at positions {occurrences}')

if len(occurrences) < 2:
    print("No duplicates found, nothing to do")
    exit()

# The second occurrence starts at occurrences[1]
# It ends right before the NEXT occurrence of ,{"id": 32
dup_start = occurrences[1]

# Find ,{"id": 32 AFTER dup_start
m32 = re.search(r',\s*\{"id":\s*32\b', nodes_line[dup_start+1:])
if not m32:
    print("ERROR: cannot find id=32 after second id=16")
    exit()

dup_end = dup_start + 1 + m32.start()  # position of the comma before {"id":32
print(f'Removing duplicate chunk from pos {dup_start} to {dup_end}')
print(f'Duplicate chunk length: {dup_end - dup_start} chars')

# Remove the duplicate chunk
html = html[:dup_start] + html[dup_end:]

# Verify
nodes_end2 = html.find('const EDGES')
all_ids = re.findall(r'"id":\s*(\d+),\s*"label"', html[:nodes_end2])
from collections import Counter
cnt = Counter(all_ids)
dups = {k: v for k, v in cnt.items() if v > 1}
print(f'Remaining duplicates: {dups if dups else "none"}')
print(f'Total nodes: {len(all_ids)}')
missing = sorted(set(range(78)) - set(int(x) for x in all_ids))
print(f'Missing ids: {missing if missing else "none"}')

html_path.write_text(html, encoding='utf-8')
print(f'DONE: {html_path}')
