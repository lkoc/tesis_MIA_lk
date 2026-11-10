# Tesis UNI y fuentes de la versión revisable

El archivo principal es `tesis.tex`. Los preliminares, los cuatro capítulos y los anexos están separados en `capitulos/`.

La estructura sigue la **Guía N.° 02 para elaboración y presentación de tesis de grado (UNI, 2023)** en su modalidad de maestría de especialización o profesionalizante. El formato de página y redacción toma también la Guía N.° 01 y las guías locales del proyecto.

## Compilación

Desde esta carpeta:

```powershell
latexmk -lualatex tesis.tex
```

La compilación usa LuaLaTeX porque el documento exige Arial mediante `fontspec`. La bibliografía se procesa con Biber y estilo APA.

La lámina de portada de la Guía N.° 02 indica Poppins para el nombre de la Universidad. La fuente Poppins Bold y su licencia OFL se distribuyen en `fonts/`; la portada la carga desde esa carpeta. El cuerpo utiliza Arial.

Las tablas, figuras y conclusiones proceden de la batería de 19 casos de
[`Benchmarks`](../Benchmarks/README.md), con referencias FEniCSx de tres
mallas y pérdidas R(Tc) individuales en operación. El [informe interno](../Benchmarks/INFORME_INTERNO.md),
el [análisis numérico](../Benchmarks/ANALISIS_NUMERICO.md) y el [estudio de colocación](../Benchmarks/MUESTREO_ADAPTATIVO.md)
conservan resultados, fallos, criterios y gráficos completos.

Desde la raíz, `python scripts/build_final_thesis.py` regenera los informes,
cuadernos y PDF después de las campañas. `python scripts/review_final_artifacts.py`
comprueba los artefactos y prepara la revisión visual. La auditoría de guías,
Plan, entrega Drive, fuentes y código está en [`docs/AUDITORIA_TESIS.md`](../docs/AUDITORIA_TESIS.md).

## Estructura

```text
tesis.tex
capitulos/
  00_dedicatoria.tex
  00_agradecimientos.tex
  00_copia_documentos.tex
  00_resumen_abstract.tex
  00_introduccion.tex
  01_planteamiento_del_problema.tex
  02_marco_teorico.tex
  03_propuesta_y_desarrollo.tex
  04_conclusiones_y_recomendaciones.tex
  05_anexos.tex
imagenes/
referencias.bib
PENDIENTES.md
```

El comando `\pendiente{...}` imprime `PENDIENTE` en rojo a 18 pt. No debe quedar ninguna marca en la versión destinada a depósito.
