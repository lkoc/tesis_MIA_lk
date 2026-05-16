"""
Find the exact character position of the JS syntax error in zotero_network.html
by scanning the NODES section for unescaped quotes or other problems.
"""
import re
from pathlib import Path

html = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html").read_text(encoding='utf-8')

# Get NODES line (line 135)
lines = html.splitlines()
print(f"Total lines: {len(lines)}")
for i, l in enumerate(lines):
    if l.startswith('const NODES=['):
        nodes_line_idx = i
        print(f"NODES on line {i+1}, length={len(l)}")
        break

nodes_line = lines[nodes_line_idx]

# Try to find malformed JSON strings by scanning for unescaped double-quotes
# inside string values. We look for patterns like: "key": "...something"..."}  
# Actually let's use the node 16-31 range specifically.
# Find where id:16 starts in the nodes line
pos16 = nodes_line.find('"id": 16,')
pos32 = nodes_line.find('"id": 32,')
print(f"id=16 at char {pos16}, id=32 at char {pos32}")

chunk = nodes_line[pos16:pos32]
print(f"\nChunk 16-31, length={len(chunk)}")

# Print aporte fields - they are the most likely to contain problematic chars
# Find all aporte values
aporte_matches = list(re.finditer(r'"aporte":\s*"(.*?)"(?:,\s*"(?:pub|color|comm_color|size)"|(?=\s*\}))', chunk, re.DOTALL))
print(f"Found {len(aporte_matches)} aporte values")
for m in aporte_matches:
    val = m.group(1)
    if '"' in val:
        print(f"  UNESCAPED QUOTE in aporte: ...{val[:100]}...")

# Also scan for any double-quote preceded by non-backslash inside a string
# Count open/close braces to check balance
depth = 0
for i, c in enumerate(chunk):
    if c == '{': depth += 1
    elif c == '}': depth -= 1
    if depth < 0:
        print(f"Unmatched '}}' at position {i} in chunk")
        print(f"Context: {chunk[max(0,i-50):i+50]}")
        break
print(f"Final brace depth in chunk: {depth}")

# Check the nodes 16-31 by simple split on node boundaries
node_strs = re.split(r',\s*(?=\{"id":\s*\d)', chunk)
for ns in node_strs:
    m = re.search(r'"id":\s*(\d+)', ns)
    nid = m.group(1) if m else '?'
    # Count quotes (unescaped) - odd number means imbalanced string
    unesc_quotes = len(re.findall(r'(?<!\\)"', ns))
    if unesc_quotes % 2 != 0:
        print(f"  Node {nid}: ODD number of unescaped quotes ({unesc_quotes}) - MALFORMED!")
        # Find the approximate position
        for qi, qm in enumerate(re.finditer(r'(?<!\\)"', ns)):
            if qi == unesc_quotes - 1:
                print(f"    Last quote at pos {qm.start()}: ...{ns[max(0,qm.start()-30):qm.start()+30]}...")
    else:
        print(f"  Node {nid}: OK (quotes={unesc_quotes})")
