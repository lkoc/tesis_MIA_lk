"""Current-limit thesis evidence; never substitutes reduced-model results."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_study_report import read,table,esc,fmt
from Benchmarks.full_ampacity import ampacity_cases
BASE=ROOT/'Benchmarks/explicit_study/ampacity';THESIS=ROOT/'Tesis_LaTeX_Borrador_UNI'


def build():
    if not read(BASE/'status.json')['completed']:raise RuntimeError('Finish explicit current-limit confirmation before publication')
    rows=read(BASE/'summary.json');catalog=ampacity_cases();fem={};meshes=[]
    for name in catalog:
        gate=read(BASE/'references'/name/'gate.json')
        if not gate['passed']:raise ValueError('Rejected current-limit reference')
        fem[name]=read(BASE/'references'/name/'fem_ampacity_l2.json')
        meshes += [[esc(name),r['level'],fmt(r['current_A'],3),fmt(r['Tmax_C'],5),fmt(r['balance_pct'],5)] for r in gate['levels']]
    output=BASE/'report';output.mkdir(exist_ok=True)
    with (output/'runs.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[r'\subsection{CORRIENTE LÍMITE CON ACOPLAMIENTO DC LOCAL}',
        'Esta extensión resuelve la misma PDE térmica en todos los materiales, con '+r'$Q=\sigma(T)E_z^2$, $E_z=I/\int_{A_c}\sigma(T)\,dA$'+'. FEM determina la raíz de corriente mediante acotación y bisección; la PINN incorpora '+r'$I=I_0\exp(\eta)$'+' como incógnita positiva y una penalización de temperatura límite de 90 °C. No recibe la corriente FEM como etiqueta. La máxima PINN se evalúa en una nube independiente con el centro incluido; se trata de una máxima muestreada, no de una cota analítica del continuo.',
        'El protocolo adicional, fijado antes de los entrenamientos, conserva arquitectura, colocaciones y tasa de C y el peso de continuidad elegido en C2; duplica los límites Adam y L-BFGS por la incógnita y el acoplamiento no lineal adicionales. Las semillas 71, 83 y 97 se incluyen completas. Se exige, además de las puertas térmicas y de interfaces, error de corriente no mayor de 2 '+r'\pct{}'+' y desviación del límite térmico no mayor de 0,1 K.',
        table('Refinamiento FEM de la corriente límite DC. Fuente: soluciones térmicas y trazas de raíz.','tab:explicit-amp-fem',['Caso','Nivel','Corriente, A',r'$T_{\max}$, °C','Balance, '+r'\pct{}'],meshes,columns='L{5.8cm}rrrr')]
    lines.append('Las tres referencias cumplen variación de corriente entre las dos mallas finales menor de 0,5 '+r'\pct{}'+' y balance global menor de 0,2 '+r'\pct{}'+'. El cierre de la raíz exige error térmico de 0,01 K e intervalo relativo de 0,05 '+r'\pct{}'+'. Esta puerta controla corriente y energía; no equivale a una estimación rigurosa del error de discretización en cualquier norma del campo.')
    lines.append(table('Todas las semillas PINN de corriente límite; se incluyen rechazos. Fuente: registros de confirmación.','tab:explicit-amp-pinn',
        ['Caso','Semilla','Corriente, A','Error, '+r'\pct{}',r'$|T_{\max}-90|$, K','Aceptada'],
        [[esc(r['case']),r['seed'],fmt(r['current_A']),fmt(r['current_error_pct']),fmt(r['temperature_limit_error_K']), 'Sí' if r['accepted'] else 'No'] for r in rows],columns='L{5.4cm}rrrrr'))
    base=fem['xlpe_single']['solution_current_A'];dry=fem['xlpe_dry_near']['solution_current_A'];mean=fem['xlpe_dry_near_hom_arithmetic']['solution_current_A']
    reduction=100*(1-dry/base);overestimate=100*(mean/dry-1);kmean=catalog['xlpe_dry_near_hom_arithmetic']['k']
    accepted=sum(r['accepted'] for r in rows);paired=[]
    for seed in [71,83,97]:
        a=next(r for r in rows if r['case']=='xlpe_dry_near' and r['seed']==seed);b=next(r for r in rows if r['case']=='xlpe_dry_near_hom_arithmetic' and r['seed']==seed)
        paired.append(dict(seed=seed,difference_A=b['current_A']-a['current_A'],accepted=a['accepted'] and b['accepted']))
    pairs=sum(r['accepted'] for r in paired)
    lines.append(f'El control homogéneo alcanza {base:.3f} A y la zona seca cercana {dry:.3f} A: una reducción de {reduction:.3f} '+r'\pct{}'+'. El promedio aritmético espacial del suelo, excluidos los discos del cable, es '+f'{kmean:.6f} '+r'W/(m\,K)'+f' y produce {mean:.3f} A. Por tanto, ese promedio sobreestima la corriente del mapa seco en {mean-dry:.3f} A ({overestimate:.3f} '+r'\pct{}'+'). La diferencia no procede de omitir la PDE del conductor: las tres referencias resuelven su interior explícitamente.')
    lines.append(f'La confirmación PINN acepta {accepted} de {len(rows)} ejecuciones; {pairs} de tres pares zona seca/promedio cumplen conjuntamente. '+
        ('Los pares aceptados permiten contrastar HE4 mediante la PINN dentro del control especificado.' if pairs else 'El contraste FEM caracteriza el efecto físico, pero esta campaña PINN no confirma HE4 con un par completamente aceptado.')+
        ' Los rechazos limitan la conclusión de OE4 aunque el error de corriente aislado resulte pequeño. El resultado corresponde a DC y a un promedio definido: no certifica ampacidad AC ni demuestra que todo promedio produzca el mismo sesgo.')
    (THESIS/'tablas/explicit_ampacity.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    conclusion=(f'OE4 cuenta con raíces FEM del modelo DC explícito: {base:.3f} A para XLPE homogéneo, {dry:.3f} A con zona seca cercana y {mean:.3f} A al sustituir ese mapa por su conductividad media del suelo. Esta última sustitución sobreestima la corriente en {overestimate:.3f} '+r'\pct{}'+f'. La confirmación PINN satisface todos los criterios en {accepted} de {len(rows)} ejecuciones y en {pairs} de tres pares seco/promedio. '+
        ('HE4 obtiene soporte en los pares aceptados para este promedio y esta geometría, sin universalizar el signo de cualquier homogenización.' if pairs else 'HE4 obtiene evidencia física FEM, pero no queda confirmada mediante pares PINN aceptados en esta campaña; el objetivo conserva esa limitación.')+
        ' La corriente límite calculada es DC, no una ampacidad AC normativa.\n')
    (THESIS/'tablas/explicit_ampacity_conclusion.tex').write_text(conclusion,encoding='utf-8')
    spanish=f'La extensión DC acepta {accepted} de {len(rows)} PINN; FEM estima {dry:.2f} A con zona seca cercana y {mean:.2f} A al reemplazarla por la conductividad media del suelo.'
    english=f'The DC extension accepts {accepted} of {len(rows)} PINN runs; FEM estimates {dry:.2f} A with a nearby dry zone and {mean:.2f} A when replacing it with the mean soil conductivity.'
    for lang,value in [('es',spanish),('en',english)]:
        (THESIS/f'tablas/explicit_ampacity_abstract_{lang}.tex').write_text(value+'\n',encoding='utf-8')
    summary=dict(total=len(rows),accepted=accepted,paired=paired,accepted_pairs=pairs,FEM_current_A={k:v['solution_current_A'] for k,v in fem.items()},dry_reduction_pct=reduction,arithmetic_overestimate_pct=overestimate,arithmetic_soil_k_W_mK=kmean)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');return summary


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False))
