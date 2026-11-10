"""
Extract the <script> block from zotero_network.html and save as .js to parse with Node.js
"""
import re
from pathlib import Path

html = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\zotero_network.html").read_text(encoding='utf-8')
# Find the script block
m = re.search(r'<script>(.*?)</script>', html, re.DOTALL)
if m:
    script = m.group(1)
    Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\_script_extract.js").write_text(script, encoding='utf-8')
    print(f"Script extracted: {len(script)} chars, {script.count(chr(10))} lines")
    print("Saved to _script_extract.js")
else:
    print("No <script> block found!")
