# Presentación breve del plan de tesis · 2026-2

Presentación Beamer de **5 diapositivas**, formato 16:9, para una exposición de aproximadamente **4 minutos**.

1. Título oficial, autores y asesor.
2. Problema, explicación física y brecha de investigación.
3. Objetivo general y esquema de la PINN.
4. Los cuatro objetivos específicos y sus productos.
5. Resultado esperado, utilidad y alcance.

## Archivos

- `presentacion_plan_2026_2.tex`: fuente editable y diagramas vectoriales TikZ.
- `presentacion_plan_2026_2.pdf`: presentación compilada.

## Compilación

Desde esta carpeta, ejecutar dos veces:

```powershell
lualatex -interaction=nonstopmode -halt-on-error presentacion_plan_2026_2.tex
```

Requiere una distribución LaTeX con Beamer, TikZ, babel y fontspec, y la fuente Arial. No requiere imágenes externas ni Biber. Las notas breves para exposición están en cada comando `\note{...}` y permanecen ocultas en el PDF.

## Referencias de elaboración

Se tomó como base `../Plan/envío_rev1/`:

- `02_Informe_latex/02_Fuentes_LaTeX/02_plan_tesis_cables_pinn.tex`: planteamiento del problema, objetivo general, OE1–OE4, justificación, alcance y síntesis DSR.
- `03_Presentacion_latex/03_Fuentes_LaTeX/03_presentacion_plan_tesis_cables_pinn.tex`: título oficial, autores, asesor y contexto visual.

Los objetivos se sintetizan para exposición. Los gráficos son esquemas conceptuales originales, no resultados de simulación. El cierre comunica productos esperados; no presupone superioridad de la PINN ni resultados ya obtenidos.
