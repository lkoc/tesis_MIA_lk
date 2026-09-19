# Reproducción de la presentación

Requiere una distribución LaTeX con LuaLaTeX, `latexmk`, Biber y la fuente Arial instalada. Ejecute `03_compilar.bat`. La carpeta contiene solo el `.tex`, la bibliografía y las dos imágenes usadas; no incluye auxiliares.

La presentación tiene 35 láminas. La lámina 13 reutiliza directamente el TikZ de la figura 4.4 del informe `02_plan_tesis_cables_pinn.tex` (etiqueta `fig:arquitectura-pinn-lawal`), basado en Lawal et al. (2022, fig. 1). La explicación procede de la sección de PINN del informe y de Raissi et al. (2019, pp. 686–690).

La franja sobre desempeño multiescala resume el capítulo 3 de `Tesis_LaTeX_Borrador_UNI/capitulos/03_propuesta_y_desarrollo.tex`: selección de arquitectura y tamaño, distribución de colocaciones y entrenamiento. Describe alternativas y criterios de evaluación, sin atribuir superioridad universal a una arquitectura. Las notas de exposición contienen el detalle y distinguen el esquema general del problema térmico estacionario de la tesis.

La lámina 14 resume el marco térmico: ecuación de calor 2D estacionaria con conductividad variable, contornos e interfaces, y máxima corriente compatible con el límite de temperatura del conductor. Se basa en las fórmulas `for:calor-estacionario-marco`, `for:interfaz` y `for:ampacidad` del informe.
