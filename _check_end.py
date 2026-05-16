"""
Show the last ~500 chars of the NODES array to see the extra }
"""
from pathlib import Path

html = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html").read_text(encoding='utf-8')
lines = html.splitlines()
nodes_line = lines[134]  # line 135

end_nodes = nodes_line.rfind('];')
print(f"NODES line ends with ...{repr(nodes_line[end_nodes-200:end_nodes+10])}")
print()
# Find the last node object
last_node_start = nodes_line.rfind(', {"id":')
last_node_end = nodes_line.rfind('};')  # this would be wrong, should be }]
print(f"Last }, ] section: ...{repr(nodes_line[end_nodes-400:end_nodes+3])}")
print()
# Count depth near end
chunk_end = nodes_line[end_nodes-300:end_nodes+3]
depth = 0
for i, c in enumerate(chunk_end):
    if c == '{': depth += 1
    elif c == '}': depth -= 1
    elif c == '[': depth -= 1  # we're already inside [, so this would increase
    elif c == ']': depth += 1
    # Just track braces
print(f"Brace check in last 300 chars: print last chars and their running brace depth")
depth = 0
in_str = False
for i, c in enumerate(chunk_end):
    if in_str:
        if c == '\\': continue
        elif c == '"': in_str = False
    else:
        if c == '"': in_str = True
        elif c == '{': depth += 1
        elif c == '}': 
            depth -= 1
            # show status at each }
            ctx_i = end_nodes - 300 + i
            # print(f"  }} at relative pos {i}, depth→{depth}: {repr(chunk_end[max(0,i-30):i+5])}")

print(f"\nFinal brace depth in last 300 chars: {depth}")
print(f"\nLast 100 chars before ]\\;: {repr(nodes_line[end_nodes-100:end_nodes+3])}")
