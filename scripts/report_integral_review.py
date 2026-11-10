"""Keep the project entry points consistent with the completed thesis campaign."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from Benchmarks.full_study_report import read


def build():
    base=ROOT/'Benchmarks/explicit_study';main=read(base/'report/summary.json');amp=read(base/'ampacity/report/summary.json');choice=main['choices']['C2'];config=choice['configuration']
    good=main['production_accepted'];total=main['production_total'];robust=sum(r['accepted']==r['total'] for r in main['production'])
    status='aceptada en desarrollo' if choice['qualified'] else 'provisional, con aceptación incompleta en desarrollo'
    smallest='\n'.join(f"| {r['case']} | {r['winner']['candidate'] if r['winner'] else 'Ninguna aceptada en ambas semillas'} | {r['winner']['parameters'] if r['winner'] else '—'} |" for r in main['case_choices'])
    review=f'''# Revisión integral de la tesis: física explícita

## Dictamen técnico

La metodología es coherente con el plan DSR y conserva la ecuación de calor dentro del conductor y de todas las capas. Su aprobación técnica habilita una comparación rigurosa; no equivale a aprobación institucional de la tesis ni a aceptación automática del artefacto. La evidencia delimita el alcance: **{good}/{total} ejecuciones de confirmación térmica aceptadas, {robust}/7 casos aceptados en las tres semillas y {amp['accepted']}/{amp['total']} ejecuciones de corriente límite aceptadas**. La receta final está {status}. Los incumplimientos permanecen en resultados y conclusiones.

## Revisión del documento completo

Se actualizaron resumen, abstract, introducción, problema e hipótesis HE1–HE4, teoría, desarrollo, resultados, discusión, conclusiones y anexos. El documento utiliza referencias FEM de la misma física, separa fuente prescrita de acoplamiento DC y trata AC y evolución temporal con su alcance real. Se eliminó la atribución de resultados reducidos a la nueva PDE interior. Se conservaron los objetivos del plan, haciendo explícitos los productos restringidos o no concluyentes. Los apartados personales y las constancias académicas siguen requiriendo contenido auténtico de los autores.

El marco teórico diferencia PINN global, descomposición por materiales, cPINN/XPINN, formulación mixta, Fourier/enriquecimientos y alternativas VPINN/FBPINN. Solo se atribuyen resultados a las implementaciones efectivamente ensayadas. Se documentan la difusión angular, regularidad del centro, escalas locales, Joule volumétrico, base DC/AC y estimación cilíndrica de piel. El catálogo contiene radios, espesores, conductividades, posiciones, corrientes y mapas de suelo, con sus unidades y adaptaciones.

## Selección de red y entrenamiento

Se completaron {main['total']} entrenamientos principales: A=36, B=24, C=12, C2=8 adicionales y D=21. A comparó nueve configuraciones sobre coaxial angular y XLPE; B aisló presupuesto y política de muestreo sobre XLPE y estratos; C comparó tres tasas con presupuesto común; C2 calibró el peso de continuidad térmica; D utilizó tres semillas nuevas y siete problemas. C2 se añadió durante B, antes de C y de toda confirmación, como respuesta a saltos térmicos persistentes. La enmienda se conserva y no se hace pasar por una decisión original de A. Las referencias FEM intervienen en evaluación y selección de hiperparámetros, nunca como temperaturas etiquetadas en la pérdida.

| Problema de desarrollo | Menor candidata aceptada en A | Parámetros |
|---|---|---:|
{smallest}

Estas son decisiones preliminares dentro de A, no certificaciones de tamaño mínimo para cualquier problema. La receta congelada es `{config['variant']}`, {config['width']}×{config['depth']} por material, muestreo `{config['sampling']}`, puntos {config['n']} por suelo / {config['n_layer']} por capa / {config['n_interface']} por interfaz, Adam {config['adam']} a tasa {config['lr']}, L-BFGS hasta {config['lbfgs']} y peso de continuidad térmica {config['temperature_weight']}. La búsqueda secuencial puede omitir interacciones; los rechazos bajo presupuesto finito no prueban incapacidad matemática de una familia.

## Convergencia y ubicación de puntos

La comprobación de insensibilidad se cumple en {sum(r['insensitive'] for r in main['convergence'])}/{len(main['convergence'])} pares. Se exigen dos soluciones aceptadas y cambios de campo y Tmax menores de 0,5 % de la elevación térmica. Se comparan 1×, 2× y 4×, y con igual presupuesto 2× las políticas uniforme, cercana, de interfaces y residual. La adaptación conserva cobertura uniforme y no usa errores FEM para elegir puntos. No se demuestra que multiplicar puntos por diez sea necesario, suficiente o siempre beneficioso.

Los errores y balances se controlan por material para que el suelo de 32 m² no oculte capas milimétricas. La concentración de colocaciones cambia la medida efectiva de la pérdida; las integrales de balance conservan cuadraturas uniformes. La posición del máximo se informa sobre muestras independientes y no se presenta como una ubicación continua certificada.

## Corriente límite y efecto piel

La extensión resuelve `Q=sigma(T) Ez²`, con `Ez=I/integral(sigma dA)`, y aprende una corriente positiva. FEM determina la raíz sobre tres mallas, conservando trazas y campos finales. Para el suelo seco cercano obtiene {amp['FEM_current_A']['xlpe_dry_near']:.3f} A; el promedio aritmético del suelo produce {amp['FEM_current_A']['xlpe_dry_near_hom_arithmetic']:.3f} A, sobreestimación de {amp['arithmetic_overestimate_pct']:.3f} %. Se aceptan {amp['accepted_pairs']}/3 pares PINN seco/promedio. Es ampacidad DC del modelo declarado, no una capacidad normativa AC.

Una resistencia ya AC a 60 Hz no recibe otra corrección por piel. Si es DC, la estimación de conductor macizo permite cuantificar el aumento de pérdida, separándolo de la redistribución espacial de la misma potencia. El contraste Aras FEM 2D de cuatro mallas estima esta última diferencia en −0,004045 K; no respalda omitir aumentos de pérdidas de hasta decenas de puntos porcentuales en los conductores grandes. Construcción, proximidad y pérdidas de pantallas requieren información adicional.

## Verificación, reproducción y limpieza

Las pruebas automáticas verifican operadores, interfaces, fuentes, regularidad, muestreo e identidad física. El acta exacta está en `auditoria/revision_integral_2026-09-15/tests.xml`; la auditoría documental y de huellas, en `review.json` de esa carpeta. Las ejecuciones A y B con receta 1× idéntica permiten comprobar repetición sin otra selección de semillas.

Se conservan JSON físicos y numéricos separados, NPZ de nubes y campos, pesos, estados de optimizadores, fuentes archivadas, versiones y huellas. Un resultado completo se reutiliza solo con la misma identidad; no se promete reanudación exacta a mitad de L-BFGS. Los tiempos de tres procesos concurrentes con un hilo por entrenamiento no son una prueba de aceleración aislada. El ensayo previo de CPU/XPU motivó CPU float64; no se afirma aprovechamiento óptimo de todo el hardware ni paralelismo distribuido de subredes.

Se retiraron 21 entradas LaTeX obsoletas después de verificar el respaldo `historico/tesis_antes_fisica_explicita_2026-09-15.zip`. Se preservó la evidencia científica anterior como histórica. El constructor final ya no llama los generadores de resultados reducidos.

## Reproducción

```powershell
python -X utf8 -m pytest Benchmarks/tests pinn_cables/tests tests -q
python -X utf8 Benchmarks/full_study.py
python -X utf8 Benchmarks/full_ampacity_study.py
python -X utf8 scripts/build_final_thesis.py
```

`full_study.py` reusa artefactos completos. Para una réplica nueva, usar una copia aislada del proyecto y un directorio `explicit_study` inicialmente vacío, conservando fuera la evidencia original; regenerar también las referencias de ampacidad siguiendo FULL_DOMAIN.md. No borrar los resultados aceptados para fingir una ejecución nueva. Los tres primeros protocolos registrados y los manifiestos fijan la secuencia de decisiones.
'''
    (ROOT/'docs/REVISION_INTEGRAL_TESIS.md').write_text(review,encoding='utf-8')
    readme=f'''# Tesis PINN para cálculo térmico de cables enterrados

La tesis activa resuelve la ecuación de calor 2D en conductor, todas las capas y suelo. El documento completo incorpora la selección preliminar de red, colocaciones y entrenamiento, la confirmación con semillas nuevas y la corriente límite DC; todas las comparaciones mantienen FEM explícito.

- [Tesis PDF](Tesis_LaTeX_Borrador_UNI/tesis.pdf) y [fuente LaTeX](Tesis_LaTeX_Borrador_UNI/tesis.tex).
- [Revisión integral y dictamen metodológico](docs/REVISION_INTEGRAL_TESIS.md).
- [Formulación, formatos y ejecución](Benchmarks/FULL_DOMAIN.md).
- [Protocolo A–D](docs/PROTOCOLO_EXPERIMENTAL_EXPLICITO.md), [enmienda C2](docs/PROTOCOLO_CALIBRACION_INTERFACES.md) y [protocolo de ampacidad](docs/PROTOCOLO_AMPACIDAD_EXPLICITA.md).
- [Resultados de todos los entrenamientos](Benchmarks/explicit_study/report/runs.csv), [selección y convergencia](Benchmarks/explicit_study/report/summary.json) y [corriente límite](Benchmarks/explicit_study/ampacity/report/summary.json).
- [Auditoría y pruebas](docs/auditoria/revision_integral_2026-09-15/).
- [Entornos FEM/PINN](Benchmarks/environment/README.md).

La confirmación térmica acepta {good}/{total} ejecuciones y la de corriente límite {amp['accepted']}/{amp['total']}. Los rechazos se conservan y delimitan las conclusiones. La etiqueta producción identifica una receta congelada; no certifica uso operativo, ampacidad AC ni validación de campo.

```powershell
python -X utf8 -m pytest Benchmarks/tests pinn_cables/tests tests -q
python -X utf8 Benchmarks/full_study.py
python -X utf8 Benchmarks/full_ampacity_study.py
python -X utf8 scripts/build_final_thesis.py
```

`Benchmarks/explicit_study` contiene la campaña principal y `explicit_results` los controles y pilotos. `Plan` conserva el plan de tesis; `pinn_cables` y `examples` incluyen bibliotecas y ejemplos previos. Las carpetas `results`, `comparisons`, `coupled_*`, `ampacity_final`, los informes anteriores y sus cuadernos corresponden al modelo reducido histórico. No alimentan la tesis activa. Las entradas reducidas exigen `--legacy-reduced` cuando calculan cables.

El respaldo verificado de la tesis anterior está en `docs/historico`; los auxiliares y tablas obsoletos retirados tienen manifiesto. Dedicatoria, agradecimientos y constancias académicas requieren información auténtica de los autores.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')


if __name__=='__main__':build()
