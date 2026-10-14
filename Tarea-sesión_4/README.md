# Entregables: revisión del plan y datos

Paquete preparado con corte documental al 10 de octubre de 2026. Los nombres del plan de tesis y la portada institucional se conservan en los documentos LaTeX y Excel.

## Archivos

- `Plan_trabajo_12_semanas.tex`: plan para revisar/completar el plan profesionalizante, 15-sep a 7-dic-2026. No sustituye el cronograma de investigación de la tesis.
- `Cronograma_12_semanas.xlsx`: Gantt independiente con avance documental al corte.
- `Informe_datos_proyecto.tex`: inventario previo de 16 instancias, casos CIGRE, evidencia COES/MINEM, criterios y faltantes.
- `Inventario_datos.xlsx`: los 16 registros previos, tres entradas COES, 11 ejemplos MINEM normalizados, las 139 menciones textuales MINEM sin deduplicar, diccionario de campos, calidad y estados PINN_v2.
- `Propuesta_revistas.tex`: tres revistas con artículos temáticamente comparables y cautelas de verificación editorial.
- `Datos_casos_PINN_v2_20261010.zip`: 21 casos v2 activos y 154 JSON candidatos no ejecutables, con índice y manifiesto.
- `generar_entregables.py`: generador reproducible de los dos libros Excel y del ZIP JSON.
- `estilo_institucional.tex`: estilo compartido por los informes LaTeX.

Los PDF compilados y archivos auxiliares LaTeX se guardan en `build/`.

## Generación y validación

Desde esta carpeta:

```powershell
python generar_entregables.py
lualatex -output-directory=build -interaction=nonstopmode -halt-on-error Plan_trabajo_12_semanas.tex
lualatex -output-directory=build -interaction=nonstopmode -halt-on-error Informe_datos_proyecto.tex
lualatex -output-directory=build -interaction=nonstopmode -halt-on-error Informe_datos_proyecto.tex
lualatex -output-directory=build -interaction=nonstopmode -halt-on-error Propuesta_revistas.tex
lualatex -output-directory=build -interaction=nonstopmode -halt-on-error Propuesta_revistas.tex
```

El generador carga cada JSON activo con `CaseDefinition.load`, comprueba las filas/hojas de los Excel y verifica que los candidatos con grupos requeridos faltantes sigan rechazados por el cargador.

## Límites de interpretación

Las 139 filas MINEM son menciones extraídas, no un conteo deduplicado de instalaciones. Los candidatos mantienen datos publicados disponibles, pero no se presentan como geometrías completas. El proyecto COES Chavarría--San Martín aparece como planificado en el diagnóstico 2023; el enlace separado de la propuesta definitiva COES 2023--2032 devolvió un archivo vacío en esta ejecución. Los JSON bajo `candidatos_no_ejecutables/` reutilizan la versión y envoltura PINN_v2, pero deliberadamente no son entradas del solver.