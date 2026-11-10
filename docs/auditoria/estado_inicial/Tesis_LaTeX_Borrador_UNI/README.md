# Borrador estructurado de tesis UNI

El archivo principal es `tesis.tex`. Los preliminares, los cuatro capítulos y los anexos están separados en `capitulos/`.

La estructura sigue la **Guía N.° 02 para elaboración y presentación de tesis de grado (UNI, 2023)** en su modalidad de maestría de especialización o profesionalizante. El formato de página y redacción toma también la Guía N.° 01 y las guías locales del proyecto.

## Compilación

Desde esta carpeta:

```powershell
latexmk -lualatex tesis.tex
```

La compilación usa LuaLaTeX porque el documento exige Arial mediante `fontspec`. La bibliografía se procesa con Biber y estilo APA.

La lámina de portada de la Guía N.° 02 indica Poppins para el nombre de la Universidad. Como esa fuente no está instalada en el entorno actual, el borrador usa Arial como sustituto; este punto queda anotado en `PENDIENTES.md` para la revisión formal final.

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
