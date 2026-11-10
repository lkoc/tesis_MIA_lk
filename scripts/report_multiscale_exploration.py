"""Build an exploratory report and figures without editing thesis inputs."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Benchmarks.full_paths import artifact_path

BASE = ROOT/'Benchmarks/multiscale_exploration'
def read(path): return json.loads(path.read_text(encoding='utf-8'))


def transfer_report(out, patch=False):
    rows=[]
    for case in (['xlpe_dry_near'] if patch else ['coaxial_angular','xlpe_dry_near']):
        for seed in [11,23]:
            folder=BASE/'patch'/f'seed{seed}' if patch else BASE/'transfer'/case/f'seed{seed}'
            path=folder/f'pinn_seed{seed}.json'
            if not path.exists():continue
            result=read(path);meta=read(folder/f'training_seed{seed}.json')
            ref=artifact_path(result['reference'])
            assert hashlib.sha256(ref.read_bytes()).hexdigest()==result['reference_sha256']
            assert read(ref.with_suffix('.json'))['physics_sha256']==result['physics_sha256']
            assert read(ref.parent/'gate.json')['passed']
            actual=np.load(path.with_suffix('.npz'));expected=np.load(ref)
            assert np.allclose(actual['xy'],expected['xy'],rtol=0,atol=1e-14)
            assert np.array_equal(actual['region'],expected['region'])
            rmse=float(np.sqrt(np.average((actual['T']-expected['T'])**2,weights=actual['weights'])))
            assert abs(rmse-result['rmse_K'])<1e-10
            for sha in meta['source_sha256'].values():
                archived=list((folder/'source').glob(sha+'_*'))
                assert len(archived)==1 and hashlib.sha256(archived[0].read_bytes()).hexdigest()==sha
            rows.append(dict(case=case,seed=seed,accepted=result['accepted'],Tmax_C=result['Tmax_C'],
                             Tmax_error_K=result['error_Tmax_K'],rmse_K=rmse,
                             energy_max_pct=max(result['balance_pct'],*result['region_energy_error_pct'].values()),
                             seconds=meta['training_seconds'],adjusted_parameters=meta['parameters'],fem_labels_used=meta['fem_labels_used']))
    stem='patch' if patch else 'transfer'
    (out/f'{stem}.json').write_text(json.dumps(dict(rows=rows,completed=len(rows)==(2 if patch else 4),artifact_checks_passed=True),indent=2),encoding='utf-8')
    lines=['# Desarrollo con experto local de suelo' if patch else '# Transferencia exploratoria sin reajustar la receta','','32×3, seis armónicos, SVD, mismos criterios y FEM de cada caso. Estas semillas/casos no son una confirmación ciega.','',
           '| Caso | Semilla | Aceptado | Error Tmax (K) | RMSE (K) | Energía máx. (%) | s |','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['seed']} | {'sí' if r['accepted'] else 'no'} | {r['Tmax_error_K']:.4f} | {r['rmse_K']:.4f} | {r['energy_max_pct']:.4f} | {r['seconds']:.2f} |")
    if patch:lines += ['', 'El experto local 16×2 añade 17 coeficientes ajustados (286 en total). Es una iteración de desarrollo posterior al fallo de transferencia y conserva los mismos puntos. No reemplaza ni oculta esos fallos.']
    (out/('EXPERTO_LOCAL.md' if patch else 'TRANSFERENCIA.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows


def main():
    rows = []
    references = set()
    for method, root in [('PINN anterior', ROOT/'Benchmarks/explicit_study/C/lr_0.0005/xlpe_single'),
                         ('Trazas + Adam/L-BFGS', BASE/'pilot'), ('Trazas + SVD 32x3 / N', BASE/'linear'),
                         ('Trazas + SVD 32x3 / N/2', BASE/'points_half'),
                         ('Trazas + SVD 32x3 / 2N', BASE/'points_double'),
                         ('Trazas + SVD 16x2 / N', BASE/'small')]:
        for seed in [11,23]:
            folder = root/f'seed{seed}'
            result = folder/f'pinn_seed{seed}.json'
            if not result.exists():
                continue
            r = read(result)
            t = read(folder/f'training_seed{seed}.json')
            ref = artifact_path(r['reference'])
            assert hashlib.sha256(ref.read_bytes()).hexdigest() == r['reference_sha256']
            references.add(r['reference_sha256'])
            reference = np.load(ref)
            current = np.load(folder/f'pinn_seed{seed}.npz')
            assert np.allclose(current['xy'], reference['xy'], rtol=0, atol=1e-14)
            assert np.array_equal(current['region'], reference['region'])
            assert r['physics_sha256'] == read(ref.with_suffix('.json'))['physics_sha256']
            for source, sha in t['source_sha256'].items():
                copies = list((folder/'source').glob(sha+'_*'))
                assert len(copies)==1 and hashlib.sha256(copies[0].read_bytes()).hexdigest()==sha
            extra = folder/'experimental_source_sha256.json'
            if extra.exists():
                for sha in read(extra).values():
                    copies = list((folder/'source').glob(sha+'_*'))
                    assert len(copies)==1 and hashlib.sha256(copies[0].read_bytes()).hexdigest()==sha
            error = current['T']-reference['T']
            rmse = float(np.sqrt(np.average(error**2, weights=current['weights'])))
            assert abs(rmse-r['rmse_K'])<1e-10
            rows.append(dict(method=method,seed=seed,accepted=r['accepted'],
                             soil_points=t['configuration']['n'],layer_points=t['configuration']['n_layer'],
                             adjusted_parameters=t['parameters'], total_parameters=t.get('parameters_total'),
                             seconds=t['training_seconds'],Tmax_C=r['Tmax_C'],
                             Tmax_error_K=r['error_Tmax_K'],rmse_K=rmse,nrmse_pct=r['nrmse_pct'],
                             energy_max_pct=max(r['balance_pct'],*r['region_energy_error_pct'].values()),
                             temperature_jump_max_K=max(i['T_jump_max_K'] for i in r['interfaces']),
                             flux_jump_max_rms_pct=max(i['flux_jump_rms_pct'] for i in r['interfaces']),
                             path=str(folder.relative_to(ROOT))))
    assert len(references)==1
    out = BASE/'report'
    out.mkdir(exist_ok=True)
    transfer=transfer_report(out)
    patch=transfer_report(out,patch=True)
    convergence=[]
    fem=read(ROOT/'Benchmarks/explicit_study/references/fixed/xlpe_single/fem_l2.json')
    rise=fem['Tmax_C']-fem['specification']['case']['T0']
    for seed in [11,23]:
        for low,high in [('points_half','linear'),('linear','points_double')]:
            paths=[BASE/group/f'seed{seed}'/f'pinn_seed{seed}.npz' for group in [low,high]]
            if not all(path.exists() for path in paths):continue
            first,last=[np.load(path) for path in paths]
            region=first['region']
            field=max(float(np.sqrt(np.mean((first['T'][region==r]-last['T'][region==r])**2))) for r in np.unique(region))
            reports=[read(path.with_suffix('.json')) for path in paths]
            hot=abs(reports[0]['Tmax_C']-reports[1]['Tmax_C'])
            convergence.append(dict(seed=seed,low=low,high=high,field_change_pct=100*field/rise,
                                    Tmax_change_pct=100*hot/rise,insensitive=field/rise<=.005 and hot/rise<=.005,
                                    both_accepted=all(r['accepted'] for r in reports)))
    summary = dict(rows=rows,all_twelve_complete=len(rows)==12,new_runs=sum(r['method']!='PINN anterior' for r in rows),
                   reused_baselines=sum(r['method']=='PINN anterior' for r in rows),
                   transfer_new_runs=len(transfer),transfer_complete=len(transfer)==4,
                   patch_new_runs=len(patch),patch_complete=len(patch)==2,
                   convergence=convergence,reference_sha256=next(iter(references)),
                   scope='Two development seeds, one fixed-source case; no production qualification',
                   artifact_checks_passed=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Resultados exploratorios multiescala', '',
           'Mismo problema físico, mismos puntos de evaluación y mismo FEM convergido. Sin etiquetas FEM en la pérdida.', '',
           '| Método | Semilla | Aceptado | Coef. ajustados | s | Error Tmax (K) | RMSE (K) | Energía máx. (%) | Salto T (K) | Salto q RMS (%) |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['seed']} | {'sí' if r['accepted'] else 'no'} | {r['adjusted_parameters']} | {r['seconds']:.2f} | {r['Tmax_error_K']:.4f} | {r['rmse_K']:.4f} | {r['energy_max_pct']:.4f} | {r['temperature_jump_max_K']:.3g} | {r['flux_jump_max_rms_pct']:.3g} |")
    lines += ['', 'Los tiempos incluyen construcción/resolución para SVD y optimización para Adam/L-BFGS; excluyen la evaluación FEM posterior. Se ejecutaron ensayos concurrentes: estas medidas no prueban una aceleración aislada del hardware.',
              '', 'SVD congela las capas ocultas y ajusta sólo salidas y trazas; no son 269 pesos totales ni una PINN entrenada de extremo a extremo. Las cantidades de parámetros fijos y ajustados constan en JSON.',
              '', 'Las semillas son de desarrollo. No hay evidencia todavía para múltiples cables, estratos, transitorio ni acoplamiento DC no lineal de esta nueva construcción.',
              '', '![Escalas geométricas](escalas.png)', '', '![Comparación exploratoria](comparacion.png)',
              '', '[Campos FEM/PINN/error en los mismos puntos](campos_semilla11.png)']
    lines += ['', '## Sensibilidad a colocaciones', '', '| Semilla | Cambio | Campo máx. regional / ΔT FEM (%) | Tmax / ΔT FEM (%) | Ambos ≤0,5% | Ambos aceptados |', '|---:|---|---:|---:|---|---|']
    for r in convergence:
        lines.append(f"| {r['seed']} | {r['low']} → {r['high']} | {r['field_change_pct']:.4f} | {r['Tmax_change_pct']:.4f} | {'sí' if r['insensitive'] else 'no'} | {'sí' if r['both_accepted'] else 'no'} |")
    lines += ['', 'El refinamiento corresponde a la arquitectura 32×3. No certifica independencia de puntos para 16×2 ni para otros casos. N=512 puntos de suelo, 256 por capa y 192 por interfaz. También cambia la cuadratura de balances al cambiar puntos de interfaz/frontera; es una sensibilidad conjunta, no un aislamiento del efecto de colocaciones interiores.']
    (out/'RESULTADOS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Actual geometry, local layer chart and nondimensional thickness charts.
    c=read(ROOT/'Benchmarks/explicit_study/cases/xlpe_single.json')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    ax=axes[0];ax.set(xlim=(-4,4),ylim=(-4,0),xlabel='x (m)',ylabel='y (m)',title='Dominio: 8 × 4 m')
    ax.add_patch(Circle((0,-.7),.015,color='#d97706'));ax.annotate('Cable: radio 15 mm',(0,-.7),(-3.5,-2),arrowprops=dict(arrowstyle='->'))
    ax.set_aspect('equal');ax.grid(alpha=.2)
    colors=['#d97706','#2563eb','#9ca3af','#059669']
    ax=axes[1]
    for layer,color in reversed(list(zip(c['layers'],colors))):
        ax.add_patch(Circle((0,0),layer[1]*1000,color=color))
    ax.set(xlim=(-17,17),ylim=(-17,17),xlabel='x local (mm)',ylabel='y local (mm)',title='Cuatro materiales explícitos')
    ax.set_aspect('equal')
    ax=axes[2]
    for j,((a,b,k),color) in enumerate(zip(c['layers'],colors)):
        ax.barh(j,1,color=color)
        ax.text(.02,j,f'{(b-a)*1000:g} mm; k={k:g}',va='center',color='black' if j==2 else 'white')
    ax.set(xlim=(0,1),yticks=range(4),yticklabels=['Núcleo','XLPE','Pantalla','Cubierta'],xlabel='Coordenada local normalizada',title='Escala propia; k en W/(m K)')
    fig.savefig(out/'escalas.png',dpi=170);fig.savefig(out/'escalas.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for r in rows:
        label=f"{r['method']} / {r['seed']}"
        axes[0].scatter(r['seconds'],r['Tmax_error_K'],marker='o' if r['accepted'] else 'x',s=55,label=label)
        axes[1].scatter(r['seconds'],r['energy_max_pct'],marker='o' if r['accepted'] else 'x',s=55)
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('Tiempo medido (s)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Error Tmax frente a FEM (K)')
    axes[1].set_ylabel('Error energético máximo (%)');axes[1].axhline(2,color='gray',linestyle='--')
    axes[0].legend(fontsize=7)
    fig.savefig(out/'comparacion.png',dpi=170);fig.savefig(out/'comparacion.pdf');plt.close(fig)
    # Fixed display seed 11. Scatter avoids interpolation across material jumps.
    path=BASE/'linear/seed11/pinn_seed11.npz'
    if path.exists():
        data=np.load(path);meta=read(path.with_suffix('.json'));femdata=np.load(artifact_path(meta['reference']))
        xy=data['xy'];labels=data['region'];error=data['T']-femdata['T']
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for i,(mask,unit) in enumerate([(labels==0,'m'),(labels>0,'mm')]):
            points=xy[mask].copy()
            if i:points=(points-np.array(c['cables'][0]))*1000
            values=[femdata['T'][mask],data['T'][mask],error[mask]]
            lo=min(values[0].min(),values[1].min());hi=max(values[0].max(),values[1].max());em=max(abs(values[2]).max(),1e-10)
            for j,value in enumerate(values):
                dots=axes[i,j].scatter(points[:,0],points[:,1],c=value,s=7 if i else 10,
                                       cmap='coolwarm' if j==2 else 'viridis',vmin=-em if j==2 else lo,vmax=em if j==2 else hi)
                axes[i,j].set(xlabel=f'x local ({unit})' if i else f'x ({unit})',ylabel=f'y local ({unit})' if i else f'y ({unit})',
                              title=['FEM (°C)','Trazas + SVD (°C)','PINN − FEM (K)'][j])
                axes[i,j].set_aspect('equal');fig.colorbar(dots,ax=axes[i,j],shrink=.8)
        fig.savefig(out/'campos_semilla11.png',dpi=170);fig.savefig(out/'campos_semilla11.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
