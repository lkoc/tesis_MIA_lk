"""Apply the documented R(T) scope consistently to active audit documentation."""
from pathlib import Path
import json,ast,shutil
ROOT=Path(__file__).resolve().parents[1]

def edit(name,old,new):
    p=ROOT/name;s=p.read_text(encoding='utf-8')
    if old not in s:raise ValueError(f'Missing passage in {name}: {old[:60]}')
    p.write_text(s.replace(old,new),encoding='utf-8')

edit('Benchmarks/README.md','Se estudian quince problemas estacionarios: cuatro soluciones manufacturadas,\nun anillo y diez escenarios de cables.','Se estudian diecinueve problemas estacionarios: siete soluciones manufacturadas,\nun anillo y once escenarios de cables.')
edit('Benchmarks/README.md','Las pérdidas son `power = current**2 * R20`, en W/m, fijadas a 20 °C. No incluyen\npérdidas dieléctricas, pantallas, proximidad ni efecto pelicular. `alpha` queda\nregistrado para extensiones electrotérmicas; no interviene en el campo base.',
'''**Los resultados de operación usan la temperatura actual de cada conductor:**
`P_j = current**2 * R20 * (1 + alpha * (T_conductor_j - 20))`, en W/m.
La ley y las tolerancias comunes están en `electrical_model.json`. FEniCSx
calcula una matriz de respuesta térmica, converge la actualización individual
y verifica el resultado mediante una solución matricial independiente. La
PINN aprende las potencias junto con el campo, imponiendo la misma ley R(T).

El campo `power = current**2 * R20` del JSON es la **potencia de referencia**
y la escala del entrenamiento. Solo `results/` y `comparisons/` la mantienen
fija para aislar la verificación térmica y comparar arquitecturas. Las salidas
`coupled_results/` (FEM), `coupled_final/` (PINN nominal) y `ampacity_final/`
(PINN a temperatura límite) contienen la física electrotérmica acoplada.
No se incluyen pérdidas dieléctricas, pantallas, proximidad ni efecto pelicular.
La ley DC y α = 0,00393 K⁻¹ se contrastan con CIGRÉ WG B1.56 (2022: 132,
*Power cable rating examples for calculation tool verification*, TB 880).''')
edit('Benchmarks/README.md','La versión actual admite un tipo de cable y una potencia común por escenario.','La versión actual admite un tipo de cable y una corriente común por escenario;\nlas potencias acopladas son distintas si las temperaturas lo son.')
edit('Benchmarks/README.md','dos hilos. Los cuadernos','uno o dos hilos, según el manifiesto. Los cuadernos')
p=ROOT/'Benchmarks/README.md'
p.write_text(p.read_text(encoding='utf-8')+'''
## Reproducir el acoplamiento y la corriente límite

```powershell
wsl -d Ubuntu -- /home/lkoc/miniforge3/envs/fenicsx/bin/python /mnt/c/usr/ths_mia_fiis/tesis_MIA_lk/Benchmarks/fem_coupled.py --cases all --levels 0 1 2
python Benchmarks/coupled_campaign.py
python Benchmarks/report.py
python Benchmarks/internal_report.py
python Benchmarks/notebooks.py
```

La campaña conserva tres semillas por caso y modo. El modo `--ampacity`
añade una corriente desconocida y exige Tmax = 90 °C; el modo `--coupled`
conserva la corriente nominal del JSON. Se exige residuo eléctrico ≤ 0,1 %,
además de la puerta térmica. Para corriente límite se verifica error frente
a FEM ≤ 5 % y distancia a 90 °C ≤ 0,1 K. `ampacity.py` conserva únicamente
el índice histórico con resistencia uniforme a 90 °C; no genera la tabla
principal de ampacidad acoplada.

El [informe interno](INFORME_INTERNO.md) presenta todos los intentos, ventajas,
limitaciones y gráficos. La [especificación](FORMAT.md) explica fórmulas y
estratos. Los [cuadernos](notebooks/) incluyen citas y referencias; sus salidas
guardadas revisan ejecuciones realizadas previamente por los solucionadores.
''',encoding='utf-8')
edit('Benchmarks/FORMAT.md','situado sobre una interfaz de suelo.','situado por encima de una interfaz de suelo.')
p=ROOT/'Benchmarks/FORMAT.md';p.write_text(p.read_text(encoding='utf-8')+'''
## Pérdidas eléctricas dependientes de la temperatura

`current` (A), `R20` (Ω/m) y `alpha` (K⁻¹) describen el conductor común del
escenario. `power` es la referencia I²R20, no la potencia final de operación.
El archivo común `electrical_model.json` declara la ley DC lineal, temperatura
de referencia, tolerancias y límite térmico. Ambos solucionadores archivan
ese archivo y su SHA-256 junto con los datos del caso.

En modo acoplado cada conductor tiene su propia temperatura y potencia:
`P[j] = I**2 * R20 * (1 + alpha * (Tc[j] - 20))`. Los resultados guardan
`conductor_C`, `powers_W_m`, `resistance_ohm_m`, `current_A` y
`electrical_residual_pct`. La hipótesis de flujo circular uniforme y la
reconstrucción radial son comunes a FEM y PINN. Un modelo que resuelva el
interior conductor debe usar generación volumétrica local J·E en W/m³;
no puede introducir directamente una potencia lineal W/m en esa ecuación.

La conductividad térmica espacial k(x,y) y la resistencia eléctrica R(Tc)
son propiedades distintas. La primera admite las fórmulas anteriores; la
segunda usa la ley explícita común. No se implementa aún k(x,y,T), ni cables
con diferentes secciones o corrientes dentro de un mismo escenario.
''',encoding='utf-8')
edit('docs/AUDITORIA_TESIS.md','para el índice DC','para el acoplamiento DC R(T)')
edit('docs/AUDITORIA_TESIS.md','Las tablas de la tesis proceden de `report.py` y `ampacity.py`.','Las tablas de la tesis proceden de `report.py`; `ampacity.py` conserva el\níndice histórico con resistencia común a 90 °C.')
edit('docs/AUDITORIA_TESIS.md','''El índice de corriente utiliza pérdidas DC y resistencia común a 90 °C.
Se verifica por bisección y expresión cerrada aprovechando linealidad térmica.
No equivale a ampacidad IEC completa, actualización individual R(T), validación
experimental, acoplamiento humedad–calor ni autorización para una instalación.
El OE4 del Plan queda cubierto parcialmente en su alcance operativo.''','''La corriente límite se calcula con actualización individual R(T), tanto en
FEniCSx como en PINN. La primera técnica usa respuestas térmicas unitarias,
iteración y bisección; la segunda incorpora potencia y corriente como
incógnitas y conserva sus residuos eléctricos. Se ejecutan tres mallas FEM
y tres semillas PINN por escenario y modo. Los resultados a R20 fija se
conservan como pruebas térmicas controladas, identificadas explícitamente.
La ampacidad corresponde al modelo DC reducido; no incluye pérdidas AC,
validación experimental ni acoplamiento humedad–calor. OE4 se concreta en
ese dominio y no certifica una instalación.''')
p=ROOT/'docs/AUDITORIA_TESIS.md';p.write_text(p.read_text(encoding='utf-8')+'''
## Errata de adjunto bibliográfico

El PDF local identificado como Raissi (2019) contiene realmente la versión 1
de *Physics informed deep learning (Part I)*, publicada en arXiv en 2017.
Se conserva la referencia de 2019 como artículo con metadatos editoriales
verificados, pero las citas de páginas de los cuadernos se atribuyen a la
prepublicación efectivamente consultada. No se fabrican páginas del artículo.
La ley R(T) y α del cobre se contrastan con TB 880, p. 132, del PDF local.
''',encoding='utf-8')
p=ROOT/'Benchmarks/citations.py';s=p.read_text(encoding='utf-8');s=s.replace('REFERENCES={','REFERENCES={\n \'cigre2022\':\'CIGRÉ Working Group B1.56. (2022). *Power cable rating examples for calculation tool verification* (Technical Brochure No. 880). CIGRÉ.\',')
s=s.replace("    if 'interface' in name", "    if name.startswith(('aras','kim','xlpe')):\n        keys.append('cigre2022');text+=' La resistencia DC se actualiza con la temperatura individual del conductor, siguiendo la ley lineal y el coeficiente del cobre documentados en CIGRÉ Working Group B1.56 (2022: 132). Las pérdidas AC adicionales de esa fuente quedan fuera del modelo DC ejecutado.'\n    if 'interface' in name")
p.write_text(s,encoding='utf-8')
