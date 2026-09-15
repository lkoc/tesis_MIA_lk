"""Independently check stored fields, metadata, electrical identities and sources."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases,fingerprint
BASE=Path(__file__).resolve().parent

def reference_path(evaluation):
    """Resolve archived provenance after moving the repository to another OS/path."""
    recorded=evaluation['fem_reference'].replace('\\','/')
    marker='/Benchmarks/'
    if marker in recorded:
        relative=recorded.rsplit(marker,1)[1]
        candidate=(BASE/relative).resolve()
        if candidate.is_relative_to(BASE.resolve()) and candidate.exists():return candidate
    path=Path(evaluation['fem_reference'])
    if path.exists():return path
    raise FileNotFoundError('FEM reference missing in this checkout: '+recorded)

def close(a,b,label,rtol=1e-8,atol=1e-9):
    if not np.allclose(a,b,rtol=rtol,atol=atol):raise AssertionError(label)

def sources(directory,mapping,allow_missing=False):
    missing=[]
    for name,digest in mapping.items():
        basename=name.replace('\\','/').split('/')[-1]
        p=directory/'source'/(digest+'_'+basename)
        if not p.exists() and allow_missing:missing.append(p.relative_to(BASE).as_posix());continue
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise AssertionError(f'Source archive: {p}')
    return missing

def main():
    catalog=cases();historical=[];counts=dict(pinn=0,fem_verification=0,fem_coupled=0)
    for p in sorted(BASE.rglob('pinn_seed*.json')):
        r=json.loads(p.read_text(encoding='utf-8'));m=r['metadata'];c=catalog[m['case']['id']]
        assert m['case_sha256']==fingerprint(c),p
        f=reference_path(r['evaluation']);fm=json.loads(f.with_suffix('.json').read_text())
        assert hashlib.sha256(f.with_suffix('.json').read_bytes()).hexdigest()==r['evaluation']['fem_json_sha256'],p
        assert bool(m.get('coupled'))==bool(fm.get('coupled')) and bool(m.get('ampacity'))==bool(fm.get('ampacity')),p
        d=np.load(p.with_suffix('.npz'));fd=np.load(f)
        assert np.isfinite(d['T']).all(),p
        close(d['xy'],fd['xy'],str(p)+' points',rtol=0,atol=1e-14)
        close(np.sqrt(np.mean((d['T']-fd['T'])**2)),r['rmse_fem_K'],str(p)+' RMSE')
        close(np.max(abs(d['T']-fd['T'])),r['max_error_fem_K'],str(p)+' maximum')
        if m.get('coupled'):
            powers=np.array(r['powers_W_m']);target=r['current_A']**2*c['R20']*(1+c['alpha']*(np.array(r['conductor_C'])-20))
            close(100*max(abs(powers-target)/target),r['electrical_residual_pct'],str(p)+' R(T)')
            close(powers/r['current_A']**2,r['resistance_ohm_m'],str(p)+' resistance')
            for b in r['boundary_diagnostics']:
                if 'relative_power_error_pct' in b:
                    close(100*abs(b['computed_heat_into_soil_W_m']/b['prescribed_heat_into_soil_W_m']-1),b['relative_power_error_pct'],str(p)+' boundary')
        sources(p.parent,r['evaluation']['source_sha256'])
        if m.get('source_sha256'):
            missing=sources(p.parent,m['source_sha256'],allow_missing=True)
            if missing:historical.append(dict(record=p.relative_to(BASE).as_posix(),missing_archives=missing))
        else:historical.append(dict(record=p.relative_to(BASE).as_posix(),missing_archives='No source hash recorded'))
        counts['pinn']+=1
    for name,c in catalog.items():
        families=[('results','fem')]+([('coupled_results','fem'),('coupled_results','fem_ampacity')] if c['kind']=='cable' else [])
        for family,stem in families:
            fine=None;previous=None
            for level in range(3):
                p=BASE/family/name/f'{stem}_l{level}.json';r=json.loads(p.read_text());d=np.load(p.with_suffix('.npz'))
                assert r['case_sha256']==fingerprint(c) and r['method']=='FEniCSx',p
                assert np.isfinite(d['T']).all(),p
                sources(p.parent,r['source_sha256'])
                if r.get('coupled'):
                    powers=np.array(r['powers_W_m']);a=np.array(r['response_K_m_W'])
                    close(c['T0']+a@powers,r['conductor_C'],str(p)+' thermal response',atol=1e-6)
                    close(c['T0']+powers@d['field_response'],d['T'],str(p)+' field response',atol=1e-6)
                    close(r['current_A']**2*c['R20']*(1+c['alpha']*(np.array(r['conductor_C'])-20)),powers,str(p)+' electrical identity',atol=1e-6)
                    if r['ampacity']:assert abs(r['Tmax_C']-90)<=1e-3,p
                previous=fine;fine=r
                counts['fem_coupled' if r.get('coupled') else 'fem_verification']+=1
            assert abs(fine['Tmax_C']-previous['Tmax_C'])/(fine['Tmax_C']-c['T0'])<=.005,(name,stem)
            assert fine['balance_pct']<=2,(name,stem)
    selection=json.loads((BASE/'selection.json').read_text(encoding='utf-8'))
    selected_paths=[]
    for directory in list(selection['cases'].values())+list(selection['ampacity_cases'].values()):
        paths=sorted((BASE/directory).glob('pinn_seed*.json'))
        assert len(paths)==3,directory
        for path in paths:
            m=json.loads(path.read_text())['metadata']
            assert m.get('source_sha256'),path
            sources(path.parent,m['source_sha256'])
        selected_paths.extend(paths)
    assert len(selected_paths)==90
    counts['selected_with_complete_training_source']=len(selected_paths)
    output=dict(completed_utc=datetime.now(timezone.utc).isoformat(),status='passed',counts=counts,
        checks=['common case hashes','stored-field RMSE and maximum','FEM reference hash and electrical mode','per-conductor R(T)','FEM response superposition','fine-mesh convergence and balance','source archive hashes'],
        historical_training_source_not_archived=historical,
        limitation='Historical training sources without a recorded hash remain unidentified; all current evaluation sources are archived. This checks numerical artifacts, not field validation.')
    (BASE/'summary/artifact_validation.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
