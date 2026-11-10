"""
Deep inspect node 16 content to find the exact syntax problem.
Also try json.loads on each node.
"""
import re, json
from pathlib import Path

html = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html").read_text(encoding='utf-8')
lines = html.splitlines()
nodes_line = lines[134]  # line 135, 0-indexed

pos16 = nodes_line.find('"id": 16,')
pos32 = nodes_line.find('"id": 32,')
# The opening { of node 16 is 1 char before pos16
chunk = nodes_line[pos16-1:pos32-1]  # includes opening { and closing }

print(f"Chunk len: {len(chunk)}")
print(f"First 80 chars: {repr(chunk[:80])}")
print(f"Last 80 chars: {repr(chunk[-80:])}")

# Find the exact breaking point by tracking string context
depth = 0
in_str = False
i = 0
while i < len(chunk):
    c = chunk[i]
    if in_str:
        if c == '\\':
            i += 2  # skip escaped char
            continue
        elif c == '"':
            in_str = False
    else:
        if c == '"':
            in_str = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth < 0:
                print(f"\n*** Unmatched '}}' at position {i} ***")
                print(f"Context: ...{repr(chunk[max(0,i-100):i+100])}...")
                break
    i += 1

print(f"Final depth: {depth}")

# Also try to parse each node as JSON
# Split nodes by finding {id: N} boundaries
node_starts = [m.start()-1 for m in re.finditer(r'"\s*id"\s*:\s*\d+', chunk)]
print(f"\nNode starts: {node_starts[:5]}...")

# Try node 16 alone
# Find its extent properly
def extract_obj(s, start):
    depth = 0
    in_str = False
    i = start
    while i < len(s):
        c = s[i]
        if in_str:
            if c == '\\': i += 2; continue
            elif c == '"': in_str = False
        else:
            if c == '"': in_str = True
            elif c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        i += 1
    return None

obj16 = extract_obj(chunk, 0)
if obj16:
    print(f"\nNode 16 object (length={len(obj16)}):")
    print(obj16[:200], "...")
    print("...", obj16[-200:])
    try:
        parsed = json.loads(obj16)
        print("json.loads: OK")
    except json.JSONDecodeError as e:
        print(f"json.loads ERROR: {e}")
        # Show surrounding context
        err_pos = e.pos
        print(f"  at pos {err_pos}: {repr(obj16[max(0,err_pos-50):err_pos+50])}")
