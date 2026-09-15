"""Create or verify the final portable artifact inventory."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,argparse
BASE=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    path=BASE/'MANIFEST.json'
    if a.verify:
        m=json.loads(path.read_text(encoding='utf-8'));bad=[]
        for name,r in m['files'].items():
            p=BASE/name
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:bad.append(name)
        if bad:raise SystemExit('Changed or missing artifacts: '+', '.join(bad))
        print(f"Verified {len(m['files'])} artifact hashes")
    else:
        files={}
        for p in sorted(BASE.rglob('*')):
            if not p.is_file() or p==path or '__pycache__' in p.parts or '.ipynb_checkpoints' in p.parts or '.pytest_cache' in p.parts:continue
            data=p.read_bytes();files[p.relative_to(BASE).as_posix()]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        path.write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'files':files},indent=2),encoding='utf-8')
        print(f'Inventoried {len(files)} artifacts')
if __name__=='__main__':main()
