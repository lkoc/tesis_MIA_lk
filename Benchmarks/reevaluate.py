"""Re-evaluate saved weights with the current audited evaluator, preserving prior reports."""
from pathlib import Path
import sys,json,hashlib,shutil
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases,fingerprint
from Benchmarks.pinn import Network,evaluate_model
from Benchmarks.electrothermal import LOSS_MODEL
BASE=Path(__file__).resolve().parent

def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    catalog=cases();paths=sorted(BASE.rglob('pinn_seed*.pt'))
    for p in paths:
        saved=torch.load(p,weights_only=False);m=saved['metadata'];c=catalog[m['case']['id']]
        if m['case_sha256']!=fingerprint(c):raise ValueError(f'Stale checkpoint: {p}')
        if m.get('coupled') and 'loss_model' not in m:
            # The three early pilots predate the metadata field. Verify their
            # archived implementation before adding an explicit migration note.
            digest=m.get('source_sha256',{}).get('Benchmarks\\pinn.py')
            if digest!='12a2286edf4ccfd540c133d741d00d79a3e24a3caf35669efb76431893feea66':
                raise ValueError(f'Unknown historical loss law: {p}')
            source=p.parent/'source'/(digest+'_pinn.py')
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError('Historical source mismatch')
            m['loss_model']=dict(LOSS_MODEL)
            m['reevaluation_metadata_note']='Ley DC individual verificada en la fuente archivada del piloto; campo loss_model añadido durante la reevaluación. El checkpoint original se conserva.'
        old=p.with_suffix('.json')
        if old.exists():
            data=old.read_bytes();archive=p.parent/'evaluation_history'/(hashlib.sha256(data).hexdigest()+'.json')
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():archive.write_bytes(data)
            if p.with_suffix('.npz').exists() and not archive.with_suffix('.npz').exists():shutil.copy2(p.with_suffix('.npz'),archive.with_suffix('.npz'))
        model=Network(c,m['width'],m['depth'],m.get('variant','enriched'),m.get('coupled',False),m.get('ampacity',False))
        model.load_state_dict(saved['state_dict']);evaluate_model(model,m,p.parent)
    print(f'Reevaluated {len(paths)} saved checkpoints',flush=True)

if __name__=='__main__':main()
