# Interfaz web (front-end)

Una página local para editar casos, correr un PINN o el FEM y comparar los resultados sin escribir
comandos. Funciona sin internet y no necesita instalar nada más que lo que ya usa el proyecto.

## Cómo abrirla

```powershell
$env:PYTHONPATH = (Get-Location).Path
python front-end/server.py --open
```

Se abre en `http://127.0.0.1:8765`. Opciones: `--port`, `--host` y `--open` (abre el navegador).
El servidor escucha solo en tu equipo. Para detenerlo, cierra la terminal o pulsa Ctrl+C.

## Las cuatro pestañas

### 1. Casos

- A la izquierda, el árbol: **Mis casos** (editables), el **catálogo v2** y el **catálogo heredado** (estos dos son de solo lectura).
- Botones: **Desde plantilla** (un cable XLPE listo para cambiar), **Vacío** (un caso mínimo), **Carpeta** (crea una carpeta nueva) e **Importar** (carga un JSON de tu disco).
- El editor tiene dos modos que trabajan sobre el mismo caso: **Formulario** y **JSON**.
- El formulario cubre todo lo configurable: dominio y suelo, parche y bandas de conductividad, posición de los cables, capas de cada cable (con una lista de materiales típicos), fuente de calor, condiciones de borde, contacto imperfecto, procedencia de los datos e inconsistencias conocidas.
- Los atajos cargan valores típicos: tipos de suelo, conductores (Cu, Al), disposiciones (un cable, trébol, plana) y condiciones de borde frecuentes.
- La geometría se dibuja al lado y se actualiza al escribir; debajo aparece la validación (con la misma comprobación que usan los solvers) y un resumen: regiones, interfaces, potencia, Q en el conductor y huella sha256.
- Un caso del catálogo no se modifica: se **duplica a Mis casos** y se edita la copia.

Guardar siempre escribe en `front-end/user_cases/<carpeta>/`, nunca en el catálogo, así los casos propios no se mezclan con los del proyecto. Un caso que no valida solo se guarda como borrador si lo marcas de forma expresa.

### 2. Ejecución

1. Elige el caso (el abierto en el editor, uno tuyo o uno del catálogo).
2. Elige el método: M0, M1, M2, CUSP, DCSNN, FBPINN o FEM (FEniCSx). Los métodos que no aceptan el caso aparecen apagados con el motivo (por ejemplo, el PINN no resuelve contacto imperfecto).
3. Ajusta los parámetros: perfil rápido, estándar o personalizado; pasos de Adam y L-BFGS, ancho y profundidad de la red, tasa de aprendizaje, activación, atención (RBA o SA), semillas, pesos de la pérdida y puntos de muestreo. Para FEM, los niveles de malla.
4. Pulsa **Entrenar** o **Ejecutar FEM**.

Cada ejecución es un proceso aparte (como máximo dos a la vez). La página muestra la barra de avance, la curva de pérdida y el **campo de temperatura en vivo** sobre la geometría del cable, y al terminar un resumen con la temperatura máxima, el balance de potencia y el salto de T en las interfaces. Se puede detener una ejecución con **Stop**.

Las carpetas de cada ejecución quedan en `front-end/runs/`.

### 3. Comparar

Compara soluciones de **un mismo caso**: FEM contra PINN, o un PINN contra otro. Incluye también los resultados de las carpetas `results/`.

- **Referencia**: el círculo; **candidatas**: las casillas.
- Mapa de error: con signo, absoluto o relativo (%), con escala automática, máxima o fija, y paletas a elección.
- Ventana: todo el dominio o un detalle de los cables. Vista de **tríptico** (referencia, candidata, error) o galería con la misma escala.
- Tabla de métricas (RMSE, MAE, error máximo, sesgo, NRMSE, diferencia de T en el conductor y dónde está el máximo), con detalle por región, perfiles de temperatura y descarga en PNG y CSV.

Todas las soluciones se remuestrean en la misma malla regular, así que se pueden restar punto a punto. Si una candidata se calculó con datos distintos (otra huella del caso), la etiqueta **otra física** avisa que la diferencia mezcla error numérico y cambio de datos.

### 4. Ayuda y docs

Los temas de ayuda y esta documentación se leen sin salir de la página.

## Notas importantes

- La referencia FEM y los datos de artículos no entran nunca como etiquetas de entrenamiento; la interfaz solo los usa para dibujar y comparar.
- Un caso heredado se convierte al formato v2 al abrirlo y se escribe de forma explícita la condición Dirichlet T0; por eso su huella puede diferir de la de resultados históricos.
- Las imágenes se generan en el servidor; la página es JavaScript sin dependencias externas.
- El FEM corre en WSL2 (Ubuntu con FEniCSx). Si no responde, el encabezado lo indica y el botón de FEM queda apagado.

Ver también: [formato de caso](cases.md), [modelos](models.md), [entrenamiento](training.md) y [referencia FEM](fem.md).