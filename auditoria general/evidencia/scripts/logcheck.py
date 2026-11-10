import re, os, collections
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
for p in [os.path.join(os.environ.get('TEMP', ''), 'ths_build', 'Plan', 'plan_tesis_cables_pinn.log'), 'Tesis_LaTeX_Borrador_UNI/tesis.log']:
    t = open(p, encoding='utf-8', errors='replace').read()
    print('=====', p, len(t.splitlines()), 'lines')
    c = collections.Counter()
    for pat in ['Overfull \\\\hbox', 'Underfull \\\\hbox', 'Overfull \\\\vbox', 'Underfull \\\\vbox', 'LaTeX Warning', 'Package \\w+ Warning', 'undefined', 'Missing character', 'multiply defined', 'Citation .* undefined', 'Reference .* undefined', 'Font shape', 'Float too large', "Token not allowed", 'Rerun']:
        c[pat] = len(re.findall(pat, t))
    print(dict(c))
    for m in re.finditer(r'(Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*)', t):
        if float(m.group(2)) > 10: print('  ', m.group(1)[:150])
    for m in re.finditer(r'((?:LaTeX|Package \w+) Warning:[^\n]*(?:\n\([^\n]*\))?)', t):
        s = m.group(1)
        if 'Rerun' in s or 'Label(s) may have changed' in s: continue
        print('  W:', s[:220].replace('\n', ' '))
    for m in re.finditer(r'(Missing character:[^\n]*)', t): print('  MC:', m.group(1)[:120])
    tail = t[-1500:]
    print('  TAIL:', tail.replace('\n', ' | ')[-600:])
