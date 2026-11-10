"""Audit active thesis inputs against the completed explicit experiments."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_paths import artifact_path
BASE=ROOT/'Benchmarks/explicit_study';THESIS=ROOT/'Tesis_LaTeX_Borrador_UNI';OUT=ROOT/'docs/auditoria/revision_integral_2026-09-15'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def active_inputs():
    found={}
    def visit(path):
        path=path.resolve()
        if path in found:return
        text=path.read_text(encoding='utf-8-sig');found[path]=text
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}',text):
            child=THESIS/name
            if not child.suffix:child=child.with_suffix('.tex')
            visit(child)
    visit(THESIS/'tesis.tex');return found


def verify_run(folder,seed):
    report=read(folder/f'pinn_seed{seed}.json');training=read(folder/f'training_seed{seed}.json')
    assert training['fem_labels_used']==0
    reference=artifact_path(report['reference']);assert sha(reference)==report['reference_sha256']
    assert read(reference.with_suffix('.json'))['physics_sha256']==report['physics_sha256']
    for name,digest in training['source_sha256'].items():
        basename=Path(name.replace('\\','/')).name
        assert sha(folder/'source'/(digest+'_'+basename))==digest
    with np.load(folder/f'pinn_seed{seed}.npz') as data,np.load(reference) as ref:
        assert np.array_equal(data['region'],ref['region'])
        assert np.allclose(data['xy'],ref['xy'],rtol=0,atol=1e-14)
        assert np.isfinite(data['T']).all()
        rmse=float(np.sqrt(np.average((data['T']-ref['T'])**2,weights=data['weights'])))
        assert abs(rmse-report['rmse_K'])<1e-10
    assert (folder/f'pinn_seed{seed}.pt').exists()
    return dict(path=str(folder.relative_to(ROOT)),seed=seed,accepted=report['accepted'],reference_sha256=report['reference_sha256'],source_files=len(training['source_sha256']))


def review(pdf=False):
    OUT.mkdir(parents=True,exist_ok=True);inputs=active_inputs();text='\n'.join(inputs.values())
    assert not re.search(r'\\input\{[^}]*(?:benchmark_|03_analisis_numerico|03_muestreo_adaptativo)',text)
    for obsolete in ['53/57','30/33','83/90','19 casos','90 entrenamientos']:
        assert obsolete not in text,obsolete
    assert 'en ejecución en esta compilación' not in text
    assert 'La verificación de corriente límite explícita está en ejecución' not in text
    main=[];seeds={}
    for stage,expected in [('A',36),('B',24),('C',12),('C2',8),('D',21)]:
        rows=read(BASE/stage/'summary.json');assert len(rows)==expected
        jobs=read(BASE/stage/'manifest.json')['jobs'];assert len(jobs)==expected
        assert len({(r['candidate'],r['case'],r['seed']) for r in rows})==expected
        seeds[stage]=sorted({r['seed'] for r in rows})
        main += [verify_run(ROOT/r['path'],r['seed']) for r in rows]
    assert not set(seeds['A']+seeds['B']+seeds['C']+seeds['C2'])&set(seeds['D'])
    transition=json.loads((OUT/'c2_transition.json').read_text(encoding='utf-8-sig'))
    assert transition['utc']<read(BASE/'C/manifest.json')['created_utc']
    assert (BASE/'protocol_C2.md').read_bytes()==(ROOT/'docs/PROTOCOLO_CALIBRACION_INTERFACES.md').read_bytes()
    frozen=read(BASE/'production_configuration.json')
    assert frozen['sha256']==hashlib.sha256(json.dumps(frozen['configuration'],sort_keys=True).encode()).hexdigest()
    manifest=read(BASE/'D/manifest.json')
    assert frozen['frozen_before_confirmation_utc']<=manifest['created_utc']
    for job in manifest['jobs']:
        assert all(job['config'][key]==value for key,value in frozen['configuration'].items())
    amp_rows=read(BASE/'ampacity/summary.json');assert len(amp_rows)==9
    amp=[verify_run(ROOT/r['path'],r['seed']) for r in amp_rows]
    amp_config=read(BASE/'ampacity/configuration.json')['configuration']
    assert amp_config['adam']==2*frozen['configuration']['adam'] and amp_config['lbfgs']==2*frozen['configuration']['lbfgs']
    for key in ['variant','width','depth','n','n_layer','n_interface','sampling','lr','temperature_weight']:assert amp_config[key]==frozen['configuration'][key]
    references=[]
    for parent in [BASE/'references',BASE/'ampacity/references']:
        for path in parent.rglob('gate.json'):
            gate=read(path);assert gate['passed'];assert len(gate['levels'])>=3
            references.append(dict(path=str(path.relative_to(ROOT)),levels=len(gate['levels'])))
    repeats=[];candidate=read(BASE/'A/selection.json')['winner']
    for seed in [11,23]:
        first=BASE/'A'/candidate/'xlpe_single'/f'seed{seed}';second=BASE/'B/mixed_1x/xlpe_single'/f'seed{seed}'
        with np.load(first/f'pinn_seed{seed}.npz') as a,np.load(second/f'pinn_seed{seed}.npz') as b:
            repeats.append(dict(seed=seed,field_bitwise_identical=bool(np.array_equal(a['T'],b['T'])),maximum_difference_K=float(np.max(abs(a['T']-b['T'])))))
    bibliographies=[THESIS/'referencias.bib',ROOT/'Benchmarks/references_selection.bib']
    keys=set().union(*(set(re.findall(r'@\w+\s*\{\s*([^,]+),',p.read_text(encoding='utf-8'))) for p in bibliographies))
    citation_keys=set()
    for match in re.finditer(r'\\(?:paren|text|auto)?cite\w*\s*((?:\[[^\]]*\]\s*|\{[^}]*\}\s*)+)',text):
        for group in re.findall(r'\{([^}]+)\}',match[1]):citation_keys.update(k.strip() for k in group.split(','))
    assert not citation_keys-keys,sorted(citation_keys-keys)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),active_inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},main_runs=len(main),ampacity_runs=len(amp),main_accepted=sum(r['accepted'] for r in main),ampacity_accepted=sum(r['accepted'] for r in amp),reference_gates=references,seeds=seeds,repeated_A_B_baseline=repeats,citation_keys_checked=len(citation_keys),runs=main+amp)
    if pdf:
        import fitz
        path=THESIS/'tesis.pdf';doc=fitz.open(path);pages=[p.get_text() for p in doc]
        log=(THESIS/'tesis.log').read_text(encoding='utf-8',errors='replace')
        errors=[line for line in log.splitlines() if any(s in line for s in ['Overfull','undefined','Missing character','LaTeX Error'])]
        assert not errors,errors
        report['pdf']=dict(sha256=sha(path),pages=len(doc),build_errors=errors)
        (OUT/'tesis_texto.txt').write_text('\n\f\n'.join(pages),encoding='utf-8')
        words={}
        for heading,end in [('RESUMEN','Palabras clave:'),('ABSTRACT','Keywords:')]:
            selected=next(p for p in pages if p.lstrip().startswith(heading) or re.match(r'^\s*[ivxlcdm]+\s+'+heading,p))
            content=selected.split(heading,1)[1].split(end,1)[0];words[heading]=len(content.split())
        report['abstract_words_approx']=words
        report['administrative_pending']=any('PENDIENTE' in p for p in pages[:8])
    (OUT/'review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['main_runs','ampacity_runs','citation_keys_checked','repeated_A_B_baseline']}));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pdf',action='store_true');review(parser.parse_args().pdf)
