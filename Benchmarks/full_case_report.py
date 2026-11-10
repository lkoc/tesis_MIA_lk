"""Publish the complete engineering inputs of the selected explicit problems."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_study import catalogue
from Benchmarks.full_study_report import table,esc,read
BASE=ROOT/'Benchmarks/explicit_study';THESIS=ROOT/'Tesis_LaTeX_Borrador_UNI'


def build():
    names=['coaxial_angular','xlpe_single','xlpe_discrete_layers','xlpe_backfill','xlpe_dry_near','xlpe_dry_far','aras_flat','kim_sand']
    catalog=catalogue();selected={name:read(BASE/'cases'/f'{name}.json') if (BASE/'cases'/f'{name}.json').exists() else catalog[name] for name in names}
    lines=[table('Entradas eléctricas y del suelo base de la campaña a fuente prescrita. Fuente: contratos de caso documentados.','tab:explicit-case-inputs',
        ['Caso','Cables','I, A',r'$R_{20}$, $\Omega$/km','P por cable, W/m','k base, W/(m K)'],
        [[esc(name),len(c['cables']),f"{c['current']:.0f}",f"{c['R20']*1000:.5f}",f"{c['power']:.5f}",f"{c['k']:.3f}"] for name,c in selected.items()])]
    geometry=[]
    for name in ['xlpe_single','aras_flat','kim_sand']:
        c=selected[name]
        for index,(ri,ro,k) in enumerate(c['layers']):geometry.append([esc(name),index,f'{ri*1000:.2f}',f'{ro*1000:.2f}',f'{(ro-ri)*1000:.2f}',f'{k:.4g}'])
    lines.append(table('Capas explícitas: índice cero conductor, los restantes materiales pasivos. Fuente: fichas de entrada auditadas.','tab:explicit-layers',
        ['Familia','Capa',r'$r_i$, mm',r'$r_o$, mm','Espesor, mm','k, W/(m K)'],geometry))
    lines.append('Los casos coaxial y XLPE comparten la secuencia radial de XLPE. El centro coaxial es (0,0); XLPE está en (0,−0,7) m. Aras utiliza centros (−0,33;−1,2), (0;−1,2) y (0,33;−1,2) m. Kim utiliza x=−0,4, 0 y 0,4 m en dos filas y=−1,6 y −1,2 m. Las capas de Kim se conservan individualmente aunque algunas tengan igual conductividad. No se homogeneiza su difusión térmica.')
    lines.append(r'Las zonas suaves emplean $k(x,y)=k_0+(k_p-k_0)S_\varepsilon(x;c_x,w)S_\varepsilon(y;c_y,h)$, donde $S_\varepsilon(s;c,w)=\tfrac12[\tanh((s-c+w/2)/\varepsilon)-\tanh((s-c-w/2)/\varepsilon)]$. El mapa se evalúa únicamente en suelo; las capas conservan su k.')
    lines.append(table('Definición de heterogeneidades suaves del suelo. Fuente: casos registrados.','tab:explicit-patches',
        ['Caso',r'$c_x$, m',r'$c_y$, m','w, m','h, m',r'$k_p$',r'$\varepsilon$, m'],
        [[esc(name),*[f'{value:g}' for value in selected[name]['patch']]] for name in ['xlpe_backfill','xlpe_dry_near','xlpe_dry_far']]))
    lines.append('Los estratos emplean k=0,5 W/(m K) bajo y=−1 m y k=1 W/(m K) sobre esa interfaz, con salto exacto. Los demás casos de esta tabla no tienen bandas adicionales. Las resistencias de los controles se interpretan como DC a 20 °C; el coeficiente térmico es 0,00393 '+r'$\mathrm{K}^{-1}$'+'. La frase histórica «flujo circular uniforme» que subsiste en la procedencia del catálogo describe su uso anterior; el contrato explícito y la pérdida activa utilizan generación volumétrica y transmisión entre materiales.')
    (THESIS/'tablas/explicit_cases.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':build()
