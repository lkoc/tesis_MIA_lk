> **Alcance actualizado:** las reinstalaciones y campañas reducidas descritas abajo son históricas. El flujo activo utiliza `full_study.py`, `full_ampacity.py` y los comandos de [FULL_DOMAIN.md](../FULL_DOMAIN.md). La nueva campaña registra versiones y fuentes por corrida; no se atribuye a ella la reinstalación previa.

Para el modelo explícito se conservan las dependencias fijadas de `requirements-review.txt`, incluida SciPy para el efecto piel. La distribución ejecutada es PyTorch 2.9.0+xpu con tensores CPU float64. La reproducción con PyTorch CPU necesita verificación de tolerancias, sin prometer identidad binaria entre distribuciones. FEniCSx usa 0.10.0, Gmsh 4.15.2 y un proceso MPI. El constructor documental necesita además LuaLaTeX, Biber, latexmk y PyMuPDF para auditar el PDF.
# Entornos y reproducción

La campaña se ejecutó con Python 3.12.9 en Windows y Python 3.12.13 en
Ubuntu/WSL. Los archivos de esta carpeta registran versiones instaladas;
no constituyen por sí solos una prueba de reinstalación desde cero.

Se ejecutó además una reinstalación aislada de Python para los solucionadores
PINN, con PyTorch 2.9.0+cpu y NumPy 2.3.4. Pasaron las pruebas de Benchmarks y
se entrenaron de nuevo `mms_constant` y `xlpe_single` con R(T). Ambos resultados
cumplieron los criterios; la diferencia máxima de campo respecto al registro
original fue 0,0041602 K y 0 K, respectivamente. El primer caso también cambió
de dos hilos a uno, por lo que no se atribuye su diferencia solo a la distribución
de PyTorch. Los comandos, versiones y métricas están en
`clean_reproduction.log`, `clean_reproduction.json`, `pip-freeze-clean-cpu.txt`
y `reproduction_comparison.json`. FEniCSx se recalculó en el entorno WSL
declarado; no se afirma haber reinstalado Conda desde cero.

## Python y PINN

Desde la raíz del proyecto, crear un entorno Python 3.12 e instalar:

```powershell
python -m venv .venv-benchmarks
.venv-benchmarks/Scripts/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.venv-benchmarks/Scripts/python -m pip install -r Benchmarks/environment/requirements-review.txt
```

Este comando usa la distribución CPU. El entorno de cálculo original tenía
la distribución `2.9.0+xpu`, pero ejecutó todos los tensores en CPU. Una
reproducción con otra distribución debe comparar tolerancias y registrar
su versión, sin prometer identidad binaria. Las instrucciones del índice CPU
proceden de [PyTorch, versiones anteriores](https://pytorch.org/get-started/previous-versions/).

`pip-freeze-windows.txt` conserva el inventario completo del entorno original;
`python_packages.json` identifica las dependencias directas de revisión.

## FEniCSx

El archivo `conda-explicit-linux-64.txt` fija los paquetes instalados en WSL.
Desde Linux, con Conda disponible:

```bash
conda create --name tesis-fenics --file Benchmarks/environment/conda-explicit-linux-64.txt
conda run -n tesis-fenics python Benchmarks/fem.py --cases mms_constant --levels 0 1 2 --output Benchmarks/reproduced_fem
conda run -n tesis-fenics python Benchmarks/fem_coupled.py --cases xlpe_single --levels 0 1 2 --output Benchmarks/reproduced_coupled_fem
```

Las opciones de instalación oficiales están en la
[documentación de DOLFINx](https://docs.fenicsproject.org/dolfinx/main/python/installation.html).
El manifiesto fija **0.10.0**, sin sustituirlo silenciosamente por la versión
que presente esa página en el futuro. El solucionador usa un proceso MPI.

## Prueba mínima y revisión

```powershell
python -m pytest Benchmarks/tests -q
python scripts/verify_suite.py
python Benchmarks/pinn.py --cases mms_constant --seeds 11 --threads 1 --output Benchmarks/reproduced
python Benchmarks/pinn.py --cases xlpe_single --seeds 11 --coupled --variant multipole --lbfgs 1600 --threads 1 --output Benchmarks/reproduced_coupled
python Benchmarks/notebooks.py --cases mms_constant xlpe_single
```

La evaluación PINN busca la referencia FEM del mismo modo eléctrico y
comprueba la huella física y la nube de puntos. Para una campaña nueva se
debe proporcionar o recalcular una referencia compatible; un valor ausente
no se interpreta como error nulo. Los cuadernos distinguen lectura de campos
guardados de recálculo de los solucionadores.

Para repetir **exactamente una configuración declarada**, utilizar su JSON
en `configurations/`, cambiando el directorio de salida para preservar la
campaña anterior. La semilla fija inicialización y muestreo; las bibliotecas,
la plataforma y el paralelismo también forman parte de la procedencia.

Las rutas absolutas guardadas identifican el entorno original. El auditor
`validate_artifacts.py` resuelve sus referencias dentro de `Benchmarks` del
checkout actual antes de recurrir a esa ruta original; comprueba igualmente
la huella del archivo FEM. Se puede trasladar la carpeta sin alterar los
registros históricos. Las rutas al intérprete FEniCS deben ajustarse al
entorno donde se repita el cálculo.

