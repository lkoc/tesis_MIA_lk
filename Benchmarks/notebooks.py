"""Build and execute review notebooks; optional cells rerun both solvers."""
from pathlib import Path
import argparse
import sys
import nbformat as nb
from nbclient import NotebookClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Benchmarks.cases import cases
ROOT=Path(__file__).resolve().parents[1]

def build(name,execute=True):
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md(f'''# {name}: expediente PINN–FEniCSx

Este cuaderno revisa resultados calculados de un problema físico común. Las
opciones `RUN_FEM` y `RUN_PINN` permiten recalcularlo. La ejecución guardada del
cuaderno carga los artefactos existentes; los entrenamientos y las soluciones
FEM se realizaron con los scripts indicados en sus metadatos.

Las comparaciones de cables usan un dominio de suelo con flujo circular
uniforme y reconstrucción radial de la temperatura del conductor. Los datos
de Aras y Kim se adaptan a condiciones controladas; no se reproduce exactamente
el experimento publicado ni se demuestra validación de campo.

## 1. Datos y trazabilidad'''),code(f'''from pathlib import Path
import sys, json, hashlib, subprocess, platform
import numpy as np
import pandas as pd
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'Benchmarks/cases.py').exists())
sys.path.insert(0, str(ROOT))
from Benchmarks.cases import cases, fingerprint, field_k
from Benchmarks.report import records, selected_directory, plot_case
CASE = {name!r}
c = cases()[CASE]
print(json.dumps(c, ensure_ascii=False, indent=2))
print('SHA-256 de la especificación:', fingerprint(c))'''),md('''## 2. Ecuación, fuentes y fronteras

Se resuelve $-\nabla\cdot(k\nabla T)=Q$. En el suelo de los casos de cables,
$Q=0$ y cada frontera circular recibe $P=I^2R_{20}$ W/m. Las normales apuntan
hacia fuera del suelo; por ello su flujo impuesto es $-P/(2\pi r)$.
En las soluciones manufacturadas, `exact` y `source` definen una identidad
analítica comprobada por pruebas independientes.

## 3. Recalcular ambos métodos

FEM emplea tres mallas P2. PINN nunca utiliza temperaturas FEM para entrenar.
Se recomienda asignar otro directorio de salida para una nueva campaña.'''),code('''RUN_FEM = False
RUN_PINN = False
FENICS_PYTHON = '/home/lkoc/miniforge3/envs/fenicsx/bin/python'
if RUN_FEM:
    if platform.system() == 'Windows':
        linux_script = '/mnt/' + ROOT.drive[0].lower() + str(ROOT/'Benchmarks/fem.py')[2:].replace('\\\\','/')
        cmd = ['wsl','-d','Ubuntu','--',FENICS_PYTHON,linux_script,'--cases',CASE,'--levels','0','1','2']
    else:
        cmd = [FENICS_PYTHON,str(ROOT/'Benchmarks/fem.py'),'--cases',CASE,'--levels','0','1','2']
    subprocess.run(cmd,cwd=ROOT,check=True)
if RUN_PINN:
    saved = sorted(selected_directory(CASE).glob('pinn_seed*.json'))[0]
    m = json.loads(saved.read_text())['metadata']
    cfg = dict(cases=[CASE],seeds=[11,23,37],variant=m.get('variant','enriched'),
               width=m['width'],depth=m['depth'],adam=m['adam'],lbfgs=m['lbfgs_max_iter'],
               output='Benchmarks/reproduced',threads=2)
    for key in ['n','lr','pde_weight','bc_weight','flux_weight','energy_weight']:
        if key in m.get('configuration',{}):cfg[key]=m['configuration'][key]
    path=ROOT/'Benchmarks/notebooks'/f'{CASE}_rerun.json'
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'Benchmarks/pinn.py'),'--config',str(path)],cwd=ROOT,check=True)
print('Consulta de artefactos existentes; activar las opciones anteriores para recalcular.')'''),md('## 4. Convergencia FEM'),code('''mesh=[]
for level in [0,1,2]:
    r=json.loads((ROOT/'Benchmarks/results'/CASE/f'fem_l{level}.json').read_text())
    assert r['case_sha256']==fingerprint(c)
    mesh.append({k:r[k] for k in ['level','ndofs','Tmax_C','balance_pct','elapsed_s','dolfinx']})
display(pd.DataFrame(mesh))'''),md('''## 5. Comparación de todas las variantes y semillas

NRMSE ≤ 5 %, error del incremento máximo ≤ 5 % y balance ≤ 2 % constituyen
la puerta térmica. Los residuos locales y errores máximos se reportan
adicionalmente; cumplir el balance global no garantiza la solución local.'''),code('''df=pd.DataFrame([r for r in records() if r['case']==CASE])
display(df[['run','variant','width','depth','seed','rmse_fem_K','nrmse_fem_pct','error_Tmax_rise_pct','balance_pct','thermal_criteria_pass']])'''),md('## 6. Campos sobre los mismos puntos de evaluación'),code('''import matplotlib.pyplot as plt
fig=plot_case(CASE)
display(fig)
plt.close(fig)'''),md('## 7. Entrenamiento y procedencia'),code('''directory=selected_directory(CASE)
for path in sorted(directory.glob('training_seed*.json')):
    r=json.loads(path.read_text())
    h=pd.DataFrame(r['history'])
    display(pd.DataFrame([r['metadata']]).drop(columns=['case'],errors='ignore'))
    if not h.empty:
        fig,ax=plt.subplots(figsize=(7,3))
        ax.semilogy(np.arange(len(h)),h['loss'],'o-')
        ax.set(xlabel='Registro secuencial Adam / L-BFGS',ylabel='Pérdida',title=path.name)
        display(fig);plt.close(fig)'''),md('''## 8. Alcance de la interpretación

Los fallos permanecen en la tabla. El uso de una configuración seleccionada en
el mismo caso es evidencia de ajuste, no una prueba independiente de
generalización. Los tiempos incluyen procesos concurrentes y dos entornos;
no sustentan una ventaja de velocidad frente a FEM. Las corrientes derivadas
por escalamiento térmico son índices DC condicionados a los supuestos
documentados en `ampacity.py`, no capacidades normativas de una instalación.''')]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    folder=ROOT/'Benchmarks/notebooks';folder.mkdir(exist_ok=True)
    path=folder/f'{name}.ipynb'
    if execute:NotebookClient(book,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nb.write(book,path)
    print(name,'executed' if execute else 'written',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+',default=['all']);ap.add_argument('--write-only',action='store_true');args=ap.parse_args()
    for name in (list(cases()) if args.cases==['all'] else args.cases):build(name,not args.write_only)
