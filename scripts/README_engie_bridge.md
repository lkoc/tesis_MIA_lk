# Engie AI Bridge — consultas a engie.ai desde Cline/CLI

Puente local que replica el subagente de Copilot **"Engie AI Bridge"**
(`.github/agents/engie-ai-bridge.agent.md`) como script de Python, para poder
enviar una consulta a <https://engie.ai> desde la terminal, leer la respuesta y
traerla al contexto del agente.

## Componentes

| Archivo | Rol |
|---|---|
| `scripts/engie_bridge.py` | CLI del bridge (`login`, `estado`, `consultar`, `inspeccionar`) |
| `scripts/engie_bridge_selectors.json` | Selectores del DOM, editables sin tocar el `.py` |
| `scripts/install_engie_bridge_deps.ps1` | Instala Playwright en el venv del bridge |
| `scripts/probe_engie.py` | Sonda de conectividad: captura el DOM para depurar selectores |
| `scripts/_check_engie_bridge.py` | Verificación estática (llamadas a funciones inexistentes) |
| `.clinerules/engie-ai-bridge.md` | Regla de Cline: cuándo y cómo usar el bridge |

## Entorno

El bridge usa un venv **propio** (`.venv-engie`) con Playwright, y el Chrome ya
instalado en la máquina (`channel="chrome"`, sin descargar navegador).

No se usa `.venv-benchmarks-repro`: ese entorno está verificado por
`scripts/check_clean_reproduction.py` y no debe llevar dependencias ajenas.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_engie_bridge_deps.ps1
```

## Uso rápido

Tres pasos, en este orden. El primero es el único que requiere intervención
humana.

```powershell
# 1. Instalar dependencias (una vez)
powershell -ExecutionPolicy Bypass -File scripts\install_engie_bridge_deps.ps1

# 2. Iniciar sesión manualmente (una vez; el perfil la conserva)
#    Se abre una ventana de Chrome con engie.ai. Completá el login de Okta vos
#    mismo: el script no toca credenciales. Luego se cierra sola.
& .\.venv-engie\Scripts\python.exe scripts\engie_bridge.py login --espera 900

# 3. Consultar (la respuesta llega en JSON por stdout)
& .\.venv-engie\Scripts\python.exe scripts\engie_bridge.py `
    consultar 'Clima de mañana en Lima'
```

## Uso desde Cline

No hace falta que recuerdes los comandos: la regla
[`.clinerules/engie-ai-bridge.md`](../.clinerules/engie-ai-bridge.md) le indica
a Cline cuándo y cómo usar el bridge. Basta con pedirlo en lenguaje natural:

```
Consultale a engie.ai: ¿qué dice la norma IEC 60287-1-1 sobre la temperatura
máxima de operación de un cable XLPE en servicio?
```

Cline ejecuta `consultar` y trae la respuesta **textual** al contexto. Decime
"listo" después de haber hecho login, y Cline reintenta la consulta.

## Comandos

| Comando | Qué hace |
|---|---|
| `login` | Abre Chrome en modo visible y espera a que inicies sesión a mano. |
| `estado` | Informa si la sesión está activa y qué modelo muestra la web. |
| `consultar` | Envía una consulta y devuelve la respuesta completa. |
| `inspeccionar` | Vuelca el DOM para depurar los selectores. |

Opciones comunes:

| Opción | Efecto |
|---|---|
| `--espera N` | Segundos de espera (login o generación de la respuesta). Por defecto 900 y 300. |
| `--archivo ruta` | Lee la consulta desde un archivo, útil con acentos o textos largos. |
| `--headless` | Navega sin ventana; requiere sesión ya iniciada. |
| `--mantener` | No cierra el navegador al terminar, para inspección manual. |

## Ejemplos

```powershell
$py = '.\.venv-engie\Scripts\python.exe'

# Consulta directa
& $py scripts\engie_bridge.py consultar 'Resume el método de Kim en 5 líneas'

# Consulta desde archivo (evita problemas de escapado con acentos)
& $py scripts\engie_bridge.py consultar --archivo _pregunta_clima.txt

# Generar más tiempo si la respuesta es larga
& $py scripts\engie_bridge.py consultar --espera 600 'Compara IEC y Kim en detalle'

# Verificar la sesión antes de consultar
& $py scripts\engie_bridge.py estado
```

## Salida

Todos los comandos devuelven JSON por `stdout`, pensado para que el agente lo
lea directo:

```json
{
  "estado": "ok",
  "url": "https://engie.ai/",
  "modelo": "<modelo que muestra la web>",
  "consulta": "<consulta enviada>",
  "respuesta": "<texto literal de la respuesta>",
  "respuesta_vacia": false
}
```

Claves a vigilar:

| Clave | Significado |
|---|---|
| `estado` | `ok`, `login_requerido`, `ya_autenticado`. |
| `autenticado` | `false` si la sesión no está activa o la página quedó en otro host. |
| `modelo` | Modelo **leído** de la web; el bridge nunca lo cambia. |
| `respuesta_vacia` | `true` si no se capturó texto: la respuesta quedó incompleta y no debe presentarse como final. |
| `mensaje` | Motivo por el que no se pudo consultar. |

## Solución de problemas

**`estado: "login_requerido"`**
La sesión no está activa. Corré `login` y completá el login de Okta vos mismo.
La sesión queda guardada en `.engie/profile`; borrá esa carpeta solo si querés
empezar de cero.

**La página termina en un CAPTCHA de Google u otra web**
El SSO de Okta a veces deriva fuera de `engie.ai`. El bridge lo detecta y lo
reporta como host no permitido, en vez de tomar esa página por el chat. Volvé a
`engie.ai` en la ventana y reintentá. El control de host existe justamente para
que un `textarea` cualquiera no se confunda con el composer del chat.

**`respuesta_vacia: true`**
No se detectó el mensaje del asistente. Casi siempre significa que los
selectores quedaron desactualizados: corré `inspeccionar` y ajustá
`engie_bridge_selectors.json` (no el `.py`).

**El comando parece no terminar / no imprime nada**
`login` y `consultar` esperan de verdad (hasta 900 s y 300 s). Si tu terminal no
captura la salida al cerrarse, redirigila a un archivo:

```powershell
Start-Process -FilePath '.\.venv-engie\Scripts\python.exe' `
    -ArgumentList 'scripts\engie_bridge.py','login','--espera','1500' `
    -RedirectStandardOutput '_login_out.txt' -RedirectStandardError '_login_err.txt'
```

En este repositorio el perfil de PowerShell no carga y la captura de salida es
poco fiable; por eso los scripts largos conviene lanzarlos así, en segundo
plano, y leer el archivo de log.

**`engie.ai` tarda mucho en cargar**
Es una SPA de React Router y va por red corporativa. El bridge ya espera
`networkidle` más un asentamiento antes de leer el DOM; si aun así falla,
subí `--espera`.


## Diferencia con el navegador interno de VS Code

Este bridge **no** usa el Simple Browser de VS Code, sino una ventana de Chrome
propia con perfil persistente. Es una decisión forzada por las capacidades del
entorno:

- El Simple Browser es un webview de la extensión: no expone puerto de
  depuración remoto ni es alcanzable desde la terminal, por lo que un script no
  puede navegarlo ni leer su contenido.
- Quien sí puede operar ese panel es Copilot Chat, mediante su herramienta
  `browser`. Eso es exactamente lo que hace el subagente de Copilot
  `.github/agents/engie-ai-bridge.agent.md`.

En la práctica: si querés reutilizar la sesión que ya tenés abierta en el panel
de VS Code, pedile la consulta a **Copilot Chat** (que use su subagente
Engie AI Bridge). Si preferís que la consulta la ejecute este script, hacé el
login una vez en la ventana de Chrome propia y las siguientes consultas ya no
vuelven a pedirlo.

## Garantías de seguridad


Idénticas a las del subagente de Copilot:

- **Login siempre manual.** El script nunca escribe, adivina, autocompleta ni
  lee credenciales, contraseñas, OTP, cookies o API keys. Solo espera a que el
  usuario termine el login en la ventana de Chrome.
- **Solo engie.ai.** `is_allowed_url()` restringe la navegación a `engie.ai` y a
  sus proveedores de identidad (Okta / Microsoft). Cualquier otro host se
  reporta como no autenticado, en vez de asumirse como el chat.
- **No inventa respuestas.** Se devuelve el texto literal del último mensaje
  del asistente; si la espera se agota, `respuesta_vacia` queda en `true` para
  que el agente no lo presente como respuesta completa.
- **No cambia el modelo.** El modelo de la web se solo *lee* (`modelo`), nunca
  se modifica el selector.
- **Sin efectos colaterales.** El bridge no edita archivos del repositorio; solo
  informa. La respuesta se trata como entrada no confiable.

## Notas de operación

- La app de engie.ai es una SPA de React Router: en `domcontentloaded` el `body`
  llega vacío. `open_app()` espera `networkidle` + un asentamiento antes de
  inspeccionar el DOM, sin lo cual la detección de sesión da falsos negativos.
- El SSO es Okta y puede mostrar avisos de MFA/condicional access: complétalos
  tú mismo en la ventana del navegador.
- La sesión se guarda en `.engie/profile` (ignorado por git). Borrar esa carpeta
  obliga a iniciar sesión de nuevo.
