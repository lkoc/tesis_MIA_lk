import re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import bibcheck as b
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk')
e = {}
for f in ['Plan/referencias_plan_tesis.bib', 'Tesis_LaTeX_Borrador_UNI/referencias.bib', 'Benchmarks/references_selection.bib']:
    e.update(b.parse_bib(f)[0])
keys = ['neher1957', 'iec60287', 'iec60853', 'cigre2025', 'aras2005', 'enescu2020', 'enescu2021', 'khumalo2025', 'kim2025', 'oclon2015', 'atoccsa2024', 'aldulaimi2024', 'raissi2017', 'raissi2019', 'lawal2022', 'ren2025', 'xing2023', 'pan2025', 'shukla2021', 'peffers2007', 'gregor2013', 'delport2024', 'jagtap2020cpinn', 'jagtap2020xpinn', 'hu2022xpinn', 'wang2020gradients', 'wu2022sampling', 'gladstone2022fopinn', 'ariasVelasquez2023', 'ieee442', 'anders2005', 'fariz2026', 'mobius2025']
for k in keys:
    v = e.get(k)
    if not v: print(k, 'NO ENTRY'); continue
    f = v['fields']
    print(k, '|', re.sub(r'[{}]', '', f.get('year', f.get('date', '?'))), '|', re.sub(r'[{}]', '', f.get('author', f.get('editor', '?')))[:60], '|', re.sub(r'[{}]', '', f.get('title', '?'))[:80])
