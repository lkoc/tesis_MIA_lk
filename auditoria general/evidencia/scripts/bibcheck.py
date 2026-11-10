import re, sys, glob, os, json
from collections import Counter
from plan_fuentes import archivos as plan_tex

def parse_bib(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    entries = {}
    dups = []
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', txt):
        typ, key = m.group(1).lower(), m.group(2)
        if typ in ('comment', 'string', 'preamble'):
            continue
        # body until next top-level @entry
        start = m.end()
        nxt = re.search(r'\n@\w+\s*\{', txt[start:])
        body = txt[start:start + nxt.start()] if nxt else txt[start:]
        if key in entries:
            dups.append(key)
        fields = {f.lower(): v.strip() for f, v in re.findall(r'(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|"[^"]*"|\d+)', body)}
        entries[key] = {'type': typ, 'fields': fields}
    return entries, dups

CITE = re.compile(r'\\(?:[Pp]arencites?|[Tt]extcites?|[Cc]ites?|[Aa]utocites?|[Ff]ootcites?|[Cc]iteauthor|[Cc]iteyear|[Nn]ocite|[Ss]upercites?|[Ff]ullcite|[Cc]itetitle|[Cc]iteurl)\*?((?:\s*\[[^\]]*\]){0,2})\s*\{([^}]*)\}')
CITES_MULTI = re.compile(r'\\(?:[Pp]arencites|[Tt]extcites|[Cc]ites|[Aa]utocites|[Ff]ootcites)((?:\s*(?:\[[^\]]*\]){0,2}\s*\{[^}]*\})+)')

def strip_comments(t):
    return re.sub(r'(?<!\\)%.*', '', t)

def cited_keys(tex_files):
    keys = Counter()
    for f in tex_files:
        t = strip_comments(open(f, encoding='utf-8', errors='replace').read())
        for m in CITES_MULTI.finditer(t):
            for k in re.findall(r'\{([^}]*)\}', m.group(1)):
                for kk in k.split(','):
                    if kk.strip():
                        keys[kk.strip()] += 1
        for m in CITE.finditer(t):
            for kk in m.group(2).split(','):
                if kk.strip():
                    keys[kk.strip()] += 0  # ensure presence; count handled by multi where applicable
    # single-command counts
    for f in tex_files:
        t = strip_comments(open(f, encoding='utf-8', errors='replace').read())
        for m in re.finditer(r'\\(?:[Pp]arencite|[Tt]extcite|[Cc]ite|[Aa]utocite|[Nn]ocite|[Ff]ootcite)\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}', t):
            for kk in m.group(1).split(','):
                if kk.strip():
                    keys[kk.strip()] += 1
    return keys

def report(name, bibs, texs):
    entries, dups = {}, []
    for b in bibs:
        e, d = parse_bib(b)
        for k in e:
            if k in entries:
                d.append(k + ' (multi-bib)')
        entries.update(e); dups += d
    cited = cited_keys(texs)
    missing = sorted(k for k in cited if k not in entries)
    uncited = sorted(k for k in entries if k not in cited)
    types = Counter(v['type'] for v in entries.values())
    nodoi = sorted(k for k, v in entries.items() if 'doi' not in v['fields'] and 'url' not in v['fields'] and v['type'] not in ('book',))
    noyear = sorted(k for k, v in entries.items() if 'year' not in v['fields'] and 'date' not in v['fields'])
    gdrive = sorted(k for k, v in entries.items() if 'drive.google' in json.dumps(v['fields']))
    noauthor = sorted(k for k, v in entries.items() if 'author' not in v['fields'] and 'editor' not in v['fields'])
    online_no_urldate = sorted(k for k, v in entries.items() if v['type'] in ('online', 'misc') and 'url' in v['fields'] and 'urldate' not in v['fields'])
    art_no_doi = sorted(k for k, v in entries.items() if v['type'] == 'article' and 'doi' not in v['fields'])
    print(f'=== {name} ===')
    print('bib files:', [os.path.relpath(b) for b in bibs])
    print('entries:', len(entries), dict(types))
    print('tex files:', len(texs), '| unique cited keys:', len(cited), '| total cite occurrences:', sum(cited.values()))
    print('duplicate keys:', dups)
    print('MISSING (cited, not in bib):', missing)
    print(f'UNCITED entries: {len(uncited)}', uncited)
    print('entries without doi and url (non-book):', len(nodoi), nodoi)
    print('articles without doi:', len(art_no_doi), art_no_doi)
    print('entries without year/date:', noyear)
    print('entries without author/editor:', noauthor)
    print('google drive urls:', gdrive)
    print('online/misc with url but no urldate:', online_no_urldate)
    return entries

if __name__ == '__main__':
    os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
    plan = report('PLAN', ['Plan/referencias_plan_tesis.bib'], plan_tex(relativos=True))
    th_tex = [f for f in glob.glob('Tesis_LaTeX_Borrador_UNI/**/*.tex', recursive=True)]
    th = report('TESIS', ['Tesis_LaTeX_Borrador_UNI/referencias.bib', 'Benchmarks/references_selection.bib'], th_tex)
    a, _ = parse_bib('Plan/referencias_plan_tesis.bib')
    b, _ = parse_bib('Tesis_LaTeX_Borrador_UNI/referencias.bib')
    c, _ = parse_bib('Benchmarks/references_selection.bib')
    print('=== COMPARACION ===')
    print('plan entries', len(a), '| tesis referencias.bib', len(b), '| references_selection.bib', len(c))
    print('solo en plan:', sorted(set(a) - set(b)))
    print('solo en tesis(referencias.bib):', sorted(set(b) - set(a)))
    diff = [k for k in set(a) & set(b) if a[k]['fields'] != b[k]['fields']]
    print('claves comunes con campos distintos:', len(diff), sorted(diff)[:40])
    print('selection ∩ referencias:', sorted(set(c) & set(b)))
