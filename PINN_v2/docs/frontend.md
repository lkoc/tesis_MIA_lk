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
2. Elige el metodo: M0, M1, M1_EXACT, M2, CUSP, DCSNN, FBPINN, DeepXDE, DeepXDE_EXACT o FEM (FEniCSx). `M1_EXACT` y `DeepXDE_EXACT` imponen por construccion las condiciones compatibles y continuidad de T/flujo; optimizan solo PDE. El primero usa el entrenador nativo; el segundo, DeepXDE con PyTorch. Comparten las capacidades del levantamiento exacto y se validan antes de crear el trabajo. PFNN/FNN pertenecen al DeepXDE historico por penalizacion.
3. Ajusta los parámetros: perfil rápido, estándar o personalizado; pasos de Adam y L-BFGS, ancho y profundidad de la red, tasa de aprendizaje, activación, atención (RBA o SA), semillas, pesos de la pérdida y puntos de muestreo. Para FEM, los niveles de malla.
4. Pulsa **Entrenar** o **Ejecutar FEM**.

Cada ejecución es un proceso aparte (como máximo dos a la vez). La página muestra la barra de avance, la curva de pérdida y el **campo de temperatura en vivo** sobre la geometría del cable, y al terminar un resumen con la temperatura máxima, el balance de potencia y el salto de T en las interfaces. Se puede detener una ejecución con **Stop**.

Las carpetas de cada ejecución quedan en `front-end/runs/`.

La opción exacta guarda su identificador en la misma configuración de corrida y usa los mismos
`result.json`, `manifest.json`, checkpoint y grillas. Los historiales de M1 siguen cargándose como
M1; no son intercambiables para reanudar con M1_EXACT. En cables se requiere fuente fija, suelo
homogéneo, Dirichlet constante y contacto perfecto; admite cualquier cantidad/posición de cables
siempre que no se solapen. MMS permite Dirichlet/Neumann/Robin con esquinas compatibles; annulus
impone sus condiciones en ambos radios.

### Modelos portatiles y controles numericos

`DeepXDE_EXACT` aparece separado de `DeepXDE`. El primero usa el campo exacto compartido y
solo PDE, por lo que deshabilita los pesos de penalizacion y permite seleccionar float32/64.
PFNN/FNN corresponden al DeepXDE historico, no al nuevo campo exacto. El modo nuevo acepta
importacion/exportacion e inferencia con el par portable, pero muestra inicio desde pesos y
no reanudacion completa. Atencion dinamica/remuestreo no soportados se rechazan explicitamente.

En Ejecucion, **Importar modelo PT + JSON** acepta un par exportado sin su directorio original.
**Descargar PT + JSON** genera un ZIP con exactamente esos dos archivos. El caso y parametros
se recuperan de la especificacion; **Inferencia del modelo** evalua T y q en coordenadas fisicas.
La API de inferencia es `POST /api/job/<id>/infer` con `xy` y `flux`.

Los controles avanzados incluyen dtype, dispositivo, normalizacion, muestreo equilibrado,
remuestreo, validacion y las parametrizaciones experimentales de trazas. En `M1_EXACT` los
pesos de penalizacion estan deshabilitados; se distinguen interiores de entrenamiento y
cuadratura independiente de diagnostico. Los parametros efectivos se restauran del modelo.

**Reanudar checkpoint completo** conserva optimizador, scheduler, RNG, atencion y puntos.
**Iniciar desde pesos anteriores** reinicia el optimizador y preserva el encoder del origen.
Cambiar estructura, dtype, fisica, pesos, nubes u horizonte no se presenta como reanudacion exacta.
Los checkpoints historicos sin estados completos solo permiten la segunda opcion.

El monitor separa ejecucion terminada de auditoria fisica aceptada/rechazada/sin referencia.
Tener una grilla o una temperatura finita no certifica el resultado.

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