"""Regenerate the review artifacts in dependency order after solvers finish."""
from pathlib import Path
import subprocess,sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'docs/auditoria/final_build.log'

def main():
    parser=argparse.ArgumentParser(description='Build only the active explicit-physics thesis after all experiments finish.');parser.parse_args()
    stages=[['Benchmarks/full_case_report.py'],['Benchmarks/full_study_report.py'],['Benchmarks/full_ampacity_report.py'],['scripts/report_integral_review.py'],['scripts/review_explicit_thesis.py']]
    results=[]
    with LOG.open('w',encoding='utf-8') as stream:
        for args in stages:
            print('Building',args[0],flush=True);start=time.perf_counter()
            p=subprocess.run([sys.executable,'-X','utf8',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            results.append(dict(stage=args[0],returncode=p.returncode,seconds=time.perf_counter()-start))
            if p.returncode:raise SystemExit(f'Build failed; inspect {LOG}')
        print('Compiling final PDF',flush=True);start=time.perf_counter()
        p=subprocess.run(['latexmk','-lualatex','-interaction=nonstopmode','-file-line-error','tesis.tex'],cwd=ROOT/'Tesis_LaTeX_Borrador_UNI',stdout=stream,stderr=subprocess.STDOUT)
        results.append(dict(stage='latexmk',returncode=p.returncode,seconds=time.perf_counter()-start))
        (ROOT/'docs/auditoria/final_build.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        if p.returncode:raise SystemExit(f'PDF build failed; inspect {LOG}')
        p=subprocess.run([sys.executable,'-X','utf8','scripts/review_explicit_thesis.py','--pdf'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if p.returncode:raise SystemExit(f'PDF audit failed; inspect {LOG}')
    print('Final artifact build completed',flush=True)
if __name__=='__main__':main()
