from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
for name in ['README.md','docs/teoria_mejoras_pinn.md']:
    p=ROOT/name;b=ROOT/'docs/auditoria/estado_inicial'/name
    if p.exists() and not b.exists():b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
(ROOT/'README.md').write_text('''# Tesis PINN para cálculo térmico de cables enterrados

Proyecto de tesis profesionalizante de maestría, FIIS–UNI. La versión activa
integra un modelo estacionario de suelo 2D, reconstrucción radial del cable,
pérdidas DC actualizadas con la temperatura individual y comparación con
FEniCSx. La selección de PINN se apoya en casos manufacturados, ablaciones,
estudios de malla y varias semillas; las ejecuciones rechazadas se conservan.

## Documentos y evidencia activa

- [Tesis PDF](Tesis_LaTeX_Borrador_UNI/tesis.pdf) y [fuente LaTeX](Tesis_LaTeX_Borrador_UNI/tesis.tex).
- [Informe interno de todos los casos](Benchmarks/INFORME_INTERNO.md): datos, resultados, ventajas, limitaciones y gráficos.
- [Auditoría de tesis y código](docs/AUDITORIA_TESIS.md) y [plan de trabajo](docs/PLAN_AUDITORIA_TESIS.md).
- [Formato común](Benchmarks/FORMAT.md), [catálogo JSON](Benchmarks/cases/) y [configuraciones](Benchmarks/configurations/).
- [Cuadernos Jupyter](Benchmarks/notebooks/), con citas y referencias; [instrucciones de ejecución](Benchmarks/README.md).
- [Entornos y reproducción](Benchmarks/environment/README.md) y [tablas completas](Benchmarks/summary/).
- [Fundamento de las variantes implementadas](docs/teoria_mejoras_pinn.md).

## Organización

| Carpeta | Función |
|---|---|
| `Benchmarks` | Batería común y evidencia numérica actual: 19 casos, ambos métodos y comparación de configuraciones. |
| `Tesis_LaTeX_Borrador_UNI` | Documento académico, referencias y figuras derivadas de la evidencia. |
| `pinn_cables` | Biblioteca original revisada; pruebas de física y utilidades. La tesis usa la ruta auditada de `Benchmarks`. |
| `Plan` | Plan, guías y entregas históricas; se conserva su identidad documental. |
| `examples` | Ejemplos previos y salidas históricas; sus comparaciones no sustituyen la batería común. |
| `docs` | Auditoría, criterios, registro de cambios y antecedentes. |
| `benchmark_papers`, `zotero_offline` | Fuentes bibliográficas y adjuntos; las erratas de metadatos están documentadas. |

`results/` y `comparisons/` dentro de Benchmarks identifican verificaciones
a fuente fija y campañas comparativas. `coupled_results/` contiene FEM con
R(T); `coupled_final/` y `ampacity_final/` contienen las campañas PINN nominal
y de corriente límite. El selector explícito `selection.json` identifica la
configuración principal de cada caso, conservando las demás ejecuciones.

## Revisión mínima

```powershell
python -m pytest Benchmarks/tests -q
python Benchmarks/notebooks.py --cases mms_constant xlpe_single
```

Para recalcular FEM, PINN o toda la campaña, seguir las instrucciones de
Benchmarks. La ejecución guardada de los cuadernos carga artefactos producidos
por los scripts; sus celdas opcionales permiten volver a calcularlos.

La evidencia corresponde a un modelo DC reducido. No demuestra validación
de campo, generalización a cualquier geometría, pérdidas AC completas ni
una ventaja de tiempo frente a FEM. Dedicatoria, agradecimientos y constancias
académicas requieren los datos auténticos de los tesistas.
''',encoding='utf-8')
(ROOT/'docs/teoria_mejoras_pinn.md').write_text('''# Formulaciones PINN implementadas y evidencia de selección

Este documento describe el código activo en `Benchmarks/pinn.py`. Los informes
previos de 64×4, 128×5 y destilación son antecedentes; no constituyen una
comparación común cuando difieren los datos, las pérdidas o la referencia FEM.
Los resultados completos se consultan en el [informe interno](../Benchmarks/INFORME_INTERNO.md).

## Ecuación y representación

Se resuelve −div(k grad T) = Q. Para k espacial variable se incluye grad(k)·grad(T).
En interfaces exactas se exige continuidad de temperatura y de flujo normal.
En el suelo exterior a los cables Q = 0; cada contorno circular recibe una
potencia lineal, cuya unidad W/m se convierte a flujo W/m² dividiendo por 2πr.

| Variante | Implementación | Ventaja que se evalúa | Limitación |
|---|---|---|---|
| Directa | MLP tanh para el campo completo. | Base simple para ablación. | Puede fallar al representar fuentes pequeñas en un dominio grande. |
| Enriquecida | Fondo de fuentes e imágenes más corrección MLP. | Incorpora la estructura de una solución homogénea. | El fondo no satisface la PDE completa si k varía; deben conservarse sus derivadas. |
| Conservación integral | Añade residuo del balance de energía. | Detecta y penaliza déficit global. | Un balance correcto no garantiza un campo correcto. No es por sí sola cPINN. |
| Redes por material | Una MLP por estrato; continuidad de T y k∂T/∂n. | Permite derivadas laterales distintas en saltos exactos. | Requiere declarar la geometría de interfaz y comprobar ambas trazas. |
| Multipolar | Tres órdenes armónicos por cable más MLP. | Representa variaciones angulares e interacción próxima. | Más incógnitas; sensibilidad persistente en heterogeneidad compleja. |
| Electrotérmica | Potencias individuales aprendidas con R(Tc); corriente aprendida en ampacidad. | Consistencia entre generación y temperatura actual. | Residuos eléctricos y térmicos deben aprobarse conjuntamente. |

La base de residuos y diferenciación automática se relaciona con Raissi et al.
(2017: 4–5). La descomposición por material se contrasta con Shukla et al.
(2021: 1, 5–6, versión arXiv). El fondo y el enriquecimiento multipolar son
adaptaciones del artefacto; no se atribuye su mejora a un artículo que no
evaluó estos casos.

## Selección finita y controles

C01–C08 varían balance, peso de PDE, anchura, profundidad, tasa de Adam y
densidad de puntos. Se mantiene un caso y la semilla 5 durante la exploración.
C05 obtiene el menor error de esa campaña; C04 ofrece un compromiso compacto
con 609 parámetros, frente a 8577 de C05. La transferencia y la sensibilidad
se observan con semillas 11, 23 y 37; no prueban un óptimo universal.

El control multipolar conserva arquitectura y presupuesto para distinguir
su efecto del aumento de iteraciones. La red global de interfaz se compara
con subredes. Los pesos fijos se motivan por la competencia entre gradientes
documentada por Wang et al. (2020: 9–10), pero no implementan su algoritmo
adaptativo. Las variantes débiles, mixtas, RAD y fronteras exactas están en
la discusión bibliográfica de la tesis y no se presentan como ejecutadas.

## Acoplamiento eléctrico

Cada conductor satisface Pj = I²R20[1 + α(Tcj − 20)], con α = 0,00393 K⁻¹.
La ley y el coeficiente del cobre se contrastan con CIGRÉ WG B1.56 (2022: 132).
La PINN aprende potencias individuales; FEniCSx resuelve su actualización
mediante respuesta térmica, punto fijo y comprobación matricial independiente.
En ampacidad solo el conductor más caliente se fija a 90 °C.

La configuración inicial acoplada usa 32×3, 1200 pasos Adam, hasta 1600
iteraciones L-BFGS y peso eléctrico 100. Los fallos heterogéneos motivan
una campaña explícita con peso 1000 y mayor presupuesto; se conserva un
control de presupuesto. Toda elección posterior debe mostrar sus resultados
por semilla y distinguir ajuste sobre el propio caso de generalización.

La aceptación exige NRMSE ≤ 5 %, error del incremento térmico ≤ 5 %, balance
≤ 2 % y residuo eléctrico ≤ 0,1 %. En ampacidad se añade error de corriente
≤ 5 % y distancia a la temperatura límite ≤ 0,1 K. No se eliminan semillas
fallidas para aparentar robustez; solo se filtran al informar una corriente
aceptada, junto con el denominador total.

## Reproducibilidad y alcance

Los puntos de evaluación son comunes e independientes del entrenamiento.
Se conservan JSON físicos, configuraciones, pesos, historiales, campos y
fuentes por huella. La revisión distingue el código de entrenamiento del
código de reevaluación, y registra las limitaciones de procedencia de los
primeros pilotos. Los tiempos con procesos concurrentes no prueban aceleración.

La verificación sigue la distinción de CIGRÉ WG B1.87 (2025: 93–99). El
modelo resuelve suelo 2D y reconstruye radialmente el conductor; no verifica
un campo interno 2D de cada capa ni pérdidas AC completas. FNO y PINO no
forman parte de los resultados PINN de la tesis.

## Referencias

Las referencias completas y las versiones consultadas se encuentran al final
de cada cuaderno y del informe interno. Las nuevas fuentes primarias sobre
formulaciones están en `Benchmarks/references_selection.bib`; las fuentes de
cables y CIGRÉ están en `Tesis_LaTeX_Borrador_UNI/referencias.bib`.
''',encoding='utf-8')
