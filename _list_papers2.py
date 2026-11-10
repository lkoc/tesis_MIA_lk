import json, sys
with open('zotero_library.json', encoding='utf-8') as f:
    items = json.load(f)
for i, it in enumerate(items):
    yr = (it.get('date') or '')[:4]
    title = it.get('title') or ''
    authors = it.get('authors') or []
    first_author = authors[0].split()[-1] if authors else '?'
    tags = ', '.join(it.get('tags') or [])
    colls = ', '.join(it.get('collections') or [])
    doi = it.get('DOI') or it.get('url') or ''
    print(f"{i:3d}. [{yr}] {first_author}: {title[:100]}")
    print(f"      col: {colls[:80]}")
    if tags:
        print(f"      tags: {tags[:80]}")
print(f"\nTOTAL: {len(items)}")
