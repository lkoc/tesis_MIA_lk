"""Run every physics and benchmark test, including the opt-in slow tests."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
log=ROOT/'docs/auditoria/tests_final.log'
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with log.open('w',encoding='utf-8') as f:
    result=subprocess.run([sys.executable,'-X','utf8','-m','pytest','pinn_cables/tests','Benchmarks/tests','tests',
        '-q','-o','addopts=','--junitxml=docs/auditoria/tests_final.xml'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
print(log.read_text(encoding='utf-8')[-5000:])
raise SystemExit(result.returncode)
