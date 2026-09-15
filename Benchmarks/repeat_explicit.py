"""Repeat a saved run using its archived solver sources and an unchanged FEM field."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_paths import artifact_path


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run,output):
    run=run.resolve();output=output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new empty reproduction directory')
    training_files=list(run.glob('training_seed*.json'))
    if len(training_files)!=1:raise ValueError('Point --run to exactly one completed seed directory')
    training=read(training_files[0]);configuration=read(run/'configuration.json');seed=configuration['seed'];result=read(run/f'pinn_seed{seed}.json')
    reference=artifact_path(result['reference'])
    if sha(reference)!=result['reference_sha256']:raise ValueError('Archived FEM field differs')
    output.mkdir(parents=True,exist_ok=True);snapshot=output/'source_environment'
    copied={}
    for name,digest in training['source_sha256'].items():
        name=name.replace('\\','/');basename=Path(name).name;source=run/'source'/(digest+'_'+basename)
        if sha(source)!=digest:raise ValueError('Archived solver source differs')
        relative=name if '/' in name else ('pinn_cables/pinn/pde.py' if name=='pde.py' else 'Benchmarks/'+name)
        target=(snapshot/relative).resolve()
        if not target.is_relative_to(snapshot):raise ValueError('Invalid archived source location')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes());copied[relative]=digest
    for package in ['Benchmarks','pinn_cables','pinn_cables/pinn']:
        (snapshot/package/'__init__.py').write_text('',encoding='utf-8')
    physics=configuration['physics'];ampacity=physics.get('problem','').startswith('explicit DC ampacity')
    config=dict(training['configuration']);config['reference']=str(reference)
    job=dict(case=physics['case'],mode=physics['source_mode'],configuration=config,ampacity=ampacity,limit=physics.get('temperature_limit_C',90.),reference=str(reference),output=str(output/'result'))
    (snapshot/'job.json').write_text(json.dumps(job,indent=2),encoding='utf-8')
    bootstrap="""import json,sys
from pathlib import Path
job=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if job['ampacity']:
    from Benchmarks.full_ampacity import train_ampacity
    train_ampacity(job['case'],job['configuration'],job['reference'],job['output'],job['limit'])
else:
    from Benchmarks.full_domain import FullDomain
    from Benchmarks.full_pinn import train
    train(FullDomain(job['case'],job['mode']),job['configuration'],job['output'])
"""
    (snapshot/'replay.py').write_text(bootstrap,encoding='utf-8')
    manifest=dict(original_run=str(run),seed=seed,source_sha256=copied,reference_sha256=sha(reference),original_environment={k:training.get(k) for k in ['python','torch','platform','device','dtype']},executed=False)
    (output/'reproduction.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');return snapshot,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    snapshot,manifest=prepare(args.run,args.output)
    if args.prepare_only:print('Archived sources verified and reproduction prepared; no training executed.');return
    print('Repeating the archived configuration and solver sources',flush=True)
    with (args.output/'reproduction.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(snapshot/'replay.py'),str(snapshot/'job.json')],cwd=snapshot,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    seed=manifest['seed']
    with np.load(args.run/f'pinn_seed{seed}.npz') as a,np.load(args.output/'result'/f'pinn_seed{seed}.npz') as b:
        manifest.update(executed=True,field_bitwise_identical=bool(np.array_equal(a['T'],b['T'])),maximum_field_difference_K=float(np.max(abs(a['T']-b['T']))))
    original=read(args.run/f'pinn_seed{seed}.json');repeated=read(args.output/'result'/f'pinn_seed{seed}.json')
    if original['physics_sha256']!=repeated['physics_sha256']:raise ValueError('Repeated physical identity differs')
    manifest.update(original_accepted=original['accepted'],repeated_accepted=repeated['accepted'])
    (args.output/'reproduction.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest))


if __name__=='__main__':main()
