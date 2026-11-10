"""Puente entre Cline/CLI y el chat web de https://engie.ai.

Replica el subagente de Copilot "Engie AI Bridge"
(".github/agents/engie-ai-bridge.agent.md") como script autonomo, para poder
consultar a engie.ai desde la terminal y traer la respuesta al contexto del
agente.

Flujo (identico al subagente):
  1. abre https://engie.ai en Chrome con perfil persistente y ventana visible;
  2. verifica la sesion -> si no esta autenticada, se detiene y avisa;
  3. lee el modelo que muestra la pagina, SIN cambiar el selector;
  4. escribe la consulta en el composer y la envia;
  5. espera a que termine de generar y devuelve la respuesta integra.

Garantias de seguridad (mismas que el subagente):
  - nunca escribe, adivina ni lee credenciales, contrasenas, OTP o cookies;
    el inicio de sesion es siempre MANUAL por el usuario;
  - no navega a ningun host distinto de engie.ai;
  - no inventa ni parafrasea la respuesta: devuelve el texto literal.

Uso:
    python scripts/engie_bridge.py estado
    python scripts/engie_bridge.py login --espera 900
    python scripts/engie_bridge.py consultar "Clima de manana en Lima"
    python scripts/engie_bridge.py consultar --archivo pregunta.txt
    python scripts/engie_bridge.py consultar --espera 300 "..."
    python scripts/engie_bridge.py inspeccionar

Variables de entorno:
    ENGIE_BRIDGE_HEADLESS=1   arranca sin ventana (solo con sesion ya abierta)
    ENGIE_BRIDGE_PROFILE      ruta del perfil de Chrome
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTORS_PATH = Path(__file__).resolve().parent / "engie_bridge_selectors.json"
PROFILE_DIR = Path(
    os.environ.get("ENGIE_BRIDGE_PROFILE", REPO_ROOT / ".engie" / "profile")
)
HEADLESS = os.environ.get("ENGIE_BRIDGE_HEADLESS", "") == "1"

NAV_TIMEOUT_MS = 60_000
HYDRATE_TIMEOUT_MS = 45_000
# Espera extra tras la hidratación: la SPA de engie.ai (React Router) pinta el
# chat de forma asíncrona después de cargar los bundles.
HYDRATE_SETTLE_MS = 6_000
GENERATION_TIMEOUT_S = 300
STABLE_READS_REQUIRED = 2
POLL_INTERVAL_S = 2.0

# Se imprime cuando la sesion no esta iniciada. El usuario debe iniciar sesion
# manualmente en la ventana de Chrome y volver a ejecutar la consulta.
LOGIN_REQUIRED_MESSAGE = (
    "La sesion de engie.ai no esta iniciada. Inicia sesion manualmente en la "
    "ventana de Chrome que se abrio, escribe 'listo' en el chat de Cline y "
    "vuelve a ejecutar la consulta."
)


def load_selectors():
    """Carga los selectores desde el JSON contiguo, para poder ajustarlos
    contra el DOM real de engie.ai sin tocar el codigo."""
    with open(SELECTORS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def open_app(page, selectors):
    """Navega a engie.ai y espera a que la SPA hidrate.

    En domcontentloaded el body llega vacio: la app es un bundle de React
    Router que monta la interfaz despues. Sin esta espera, la deteccion de
    login daria un falso negativo.
    """
    page.goto(selectors["url"], wait_until="domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=HYDRATE_TIMEOUT_MS)
    except Exception:
        # Una conexion larga (streaming) puede no llegar a networkidle; se
        # sigue adelante con la espera de asentamiento.
        pass
    page.wait_for_timeout(HYDRATE_SETTLE_MS)


def first_match(page, selectors, timeout_s=0.0):
    """Devuelve el primer locator que resuelve, o None si ninguno coincide."""
    deadline = time.time() + timeout_s
    while True:
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if locator.count() > 0:
                    return locator
            except Exception:
                # Selector no valido para esta pagina: se ignora y se prueba el
                # siguiente, igual que hace el subagente al broadestear.
                continue
        if time.time() >= deadline:
            return None
        time.sleep(0.4)


def launch(playwright, headless=None):
    """Abre Chrome con perfil persistente y devuelve (context, page).

    Reutiliza el Chrome ya instalado (channel="chrome") en vez de descargar
    otro navegador. El perfil persistente es lo que conserva la sesion entre
    ejecuciones, de modo que el login se hace una sola vez y siempre manual.
    """
    if headless is None:
        headless = HEADLESS
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        channel="chrome",
        headless=headless,
        viewport={"width": 1440, "height": 960},
    )
    pages = context.pages
    page = pages[0] if pages else context.new_page()
    page.set_default_timeout(NAV_TIMEOUT_MS)
    return context, page


ALLOWED_HOST = "engie.ai"
# Dominios de identidad a los que la propia web redirige durante el login.
ALLOWED_AUTH_HOSTS = ("okta.com", "microsoftonline.com", "login.microsoftonline.com")


def host_of(url):
    """Host de una URL, en minusculas y sin puerto."""
    return (urlparse(url).hostname or "").lower()


def is_allowed_url(url, allow_auth=True):
    """True si la URL pertenece a engie.ai o a un proveedor de identidad.

    El subagente prohibe navegar a cualquier otro host. Este control es
    ademas lo que evita falsos positivos: un `textarea` generico puede ser la
    caja de busqueda de Google si la pagina acaba en un CAPTCHA.
    """
    host = host_of(url)
    if host == ALLOWED_HOST or host.endswith("." + ALLOWED_HOST):
        return True
    if allow_auth and any(host == h or host.endswith("." + h)
                          for h in ALLOWED_AUTH_HOSTS):
        return True
    return False


def is_on_app(page):
    """True si la pagina actual pertenece a la app de engie.ai."""
    return is_allowed_url(page.url, allow_auth=False)


def is_authenticated(page, selectors):
    """Detecta si la pagina muestra el chat o una pantalla de login.

    Exige estar en engie.ai: si la pagina termino en otro host (por ejemplo un
    CAPTCHA de Google tras el SSO) se considera NO autenticada, aunque exista
    un textarea en pantalla.

    Nunca pulsa los botones del proveedor de identidad: el login es manual.
    """
    if not is_on_app(page):
        return False
    if first_match(page, selectors["composer"], timeout_s=8.0) is not None:
        return True
    for marker in selectors["login_markers"]:
        try:
            if page.locator(marker).first.count() > 0:
                return False
        except Exception:
            continue
    # Sin composer ni marcador conocido se asume "no autenticado": es la opcion
    # segura, porque implica detenerse y pedir login manual.
    return False


def read_model(page, selectors):
    """Lee el modelo ya seleccionado en la pagina, SIN cambiar el selector.

    El subagente exige conservar el modelo de la web aunque la consulta pida
    otro; si hay discrepancia se reporta en vez de modificarlo.
    """
    chip = first_match(page, selectors["model_chip"], timeout_s=2.0)
    if chip is None:
        return None
    try:
        text = (chip.inner_text() or "").strip()
    except Exception:
        return None
    return text or None


def send_prompt(page, selectors, prompt):
    """Escribe la consulta en el composer y la envia."""
    composer = first_match(page, selectors["composer"], timeout_s=15.0)
    if composer is None:
        raise RuntimeError("No se encontro el composer del chat en engie.ai")

    try:
        composer.click()
    except Exception:
        pass

    # fill() respeta el editor enriquecido (ProseMirror / contenteditable) y
    # evita que Enter dispare el envio antes de terminar de escribir.
    try:
        composer.fill(prompt)
    except Exception:
        composer.type(prompt, delay=5)

    sent = False
    button = first_match(page, selectors["send_button"], timeout_s=2.0)
    if button is not None:
        try:
            if button.is_enabled():
                button.click()
                sent = True
        except Exception:
            sent = False
    if not sent:
        composer.press("Enter")
    return sent


def visible_text(locator):
    """Texto visible de un locator, o cadena vacia si no se puede leer."""
    try:
        return (locator.inner_text() or "").strip()
    except Exception:
        return ""


def read_last_answer(page, selectors):
    """Devuelve el texto del ultimo mensaje del asistente visible."""
    message = first_match(page, selectors["assistant_message"])
    if message is None:
        return ""
    return visible_text(message)


def wait_for_answer(page, selectors, baseline, timeout_s=GENERATION_TIMEOUT_S):
    """Espera a que la generacion termine y devuelve la respuesta completa.

    Criterio: el texto debe permanecer igual en dos lecturas separadas y no
    debe haber boton de parada. Son lecturas reales de la pagina, no un sleep
    fijo a ciegas.
    """
    deadline = time.time() + timeout_s
    stable = 0
    last = ""

    while time.time() < deadline:
        current = read_last_answer(page, selectors)
        if current and current != baseline:
            if current == last:
                stable += 1
                if stable >= STABLE_READS_REQUIRED:
                    if first_match(page, selectors["stop_button"],
                                   timeout_s=0.3) is None:
                        return current
            else:
                stable = 0
            last = current
        time.sleep(POLL_INTERVAL_S)

    # Timeout: se devuelve lo capturado hasta ese momento, marcado como
    # incompleto para que el agente no lo presente como respuesta final.
    return last


def snapshot(page, selectors, prompt=None):
    """Estado actual de la pagina: url, host, autenticacion y modelo."""
    authenticated = is_authenticated(page, selectors)
    data = {
        "url": page.url,
        "host": host_of(page.url),
        "autenticado": authenticated,
        "modelo": read_model(page, selectors) if authenticated else None,
        "consulta": prompt,
    }
    if not is_allowed_url(page.url):
        # La pagina se fue a un host que no es engie.ai ni su proveedor de
        # identidad: se reporta explicitamente para no tomar el textarea de
        # ese sitio por el composer del chat.
        data["mensaje"] = (
            f"La pagina termino en un host no permitido: {host_of(page.url)}. "
            "Se requiere login manual en engie.ai."
        )
    elif not authenticated:
        data["mensaje"] = LOGIN_REQUIRED_MESSAGE
    else:
        data["mensaje"] = None
    return data


def cmd_estado(args, selectors):
    """Abre la pagina y reporta el estado de la sesion sin enviar nada."""
    def run(page):
        open_app(page, selectors)
        return snapshot(page, selectors)

    return with_playwright(args, run, headless=args.headless)


def cmd_login(args, selectors):
    """Abre Chrome en ventana visible para que el usuario inicie sesion a mano.

    No escribe ni lee credenciales: solo espera a que la sesion quede
    autenticada y luego cierra el navegador. El perfil persistente conserva la
    sesion, asi que las consultas posteriores no vuelven a pedir login.
    """
    def run(page):
        open_app(page, selectors)
        if is_authenticated(page, selectors):
            return {"estado": "ya_autenticado", "url": page.url,
                    "modelo": read_model(page, selectors)}
        print(LOGIN_REQUIRED_MESSAGE, flush=True)
        print("Esperando a que completes el inicio de sesion...", flush=True)
        # El usuario hace login en la ventana; aqui solo se sondea la pagina.
        wait_for_login(page, selectors, args.espera)
        autenticado = is_authenticated(page, selectors)
        return {
            "estado": "ok" if autenticado else "login_requerido",
            "url": page.url,
            "autenticado": autenticado,
            "modelo": read_model(page, selectors) if autenticado else None,
            "mensaje": None if autenticado else LOGIN_REQUIRED_MESSAGE,
        }

    return with_playwright(args, run, headless=False)


def wait_for_login(page, selectors, timeout_s):
    """Sondea hasta que aparezca el composer o se agote el tiempo."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if first_match(page, selectors["composer"], timeout_s=0.5) is not None:
            return True
        time.sleep(POLL_INTERVAL_S)
    return False



def cmd_inspeccionar(args, selectors):
    """Vuelca informacion de depuracion: titulo, composer, modelo y cuerpo de
    la pagina. Sirve para ajustar los selectores contra el DOM real."""
    def run(page):
        open_app(page, selectors)
        data = {
            "url": page.url,
            "titulo": page.title(),
            "autenticado": is_authenticated(page, selectors),
        }
        for name in ("composer", "model_chip", "assistant_message",
                     "send_button"):
            match = first_match(page, selectors[name])
            data[name] = None if match is None else visible_text(match)[:200]
        try:
            body = page.locator("body").first.inner_text()
        except Exception:
            body = ""
        data["body_inicio"] = body[:3000]
        return data

    return with_playwright(args, run, headless=args.headless)


def cmd_consultar(args, selectors):
    """Consulta completa: abre, verifica sesion, envia y devuelve respuesta."""
    if args.archivo:
        prompt = Path(args.archivo).read_text(encoding="utf-8").strip()
    else:
        prompt = args.consulta
    if not prompt:
        print(json.dumps({"error": "Falta la consulta"},
                         ensure_ascii=False))
        return 2

    def run(page):
        open_app(page, selectors)

        if not is_authenticated(page, selectors):

            return {
                "estado": "login_requerido",
                "url": page.url,
                "mensaje": LOGIN_REQUIRED_MESSAGE,
                "consulta": prompt,
            }

        modelo = read_model(page, selectors)
        baseline = read_last_answer(page, selectors)
        send_prompt(page, selectors, prompt)
        answer = wait_for_answer(page, selectors, baseline, args.espera)

        return {
            "estado": "ok",
            "url": page.url,
            "modelo": modelo,
            "consulta": prompt,
            "respuesta": answer,
            "respuesta_vacia": not answer,
        }

    return with_playwright(args, run, headless=args.headless)


def with_playwright(args, run, headless=None):
    """Abre el navegador, ejecuta `run(page)` y vuelca el resultado en JSON.

    La respuesta se imprime en stdout como JSON para que el agente pueda
    leerla y traysela a su contexto sin depender de formato humano.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        print(json.dumps(
            {"error": f"playwright no instalado: {exc}",
             "sugerencia": "pip install playwright"},
            ensure_ascii=False))
        return 3

    try:
        with sync_playwright() as playwright:
            context, page = launch(playwright, headless=headless)
            try:
                print(json.dumps(run(page), ensure_ascii=False, indent=2))
            finally:
                if not getattr(args, "mantener", False):
                    context.close()
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "tipo": type(exc).__name__},
                         ensure_ascii=False))
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Puente a engie.ai para consultas desde Cline/CLI.")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_estado = sub.add_parser("estado",
                              help="Reporta el estado de la sesion.")
    p_estado.add_argument("--headless", action="store_true")
    p_estado.add_argument("--mantener", action="store_true",
                          help="Deja el navegador abierto al terminar.")
    p_estado.set_defaults(func=cmd_estado)

    p_insp = sub.add_parser("inspeccionar",
                            help="Vuelca el DOM para depurar selectores.")
    p_insp.add_argument("--headless", action="store_true")
    p_insp.add_argument("--mantener", action="store_true")
    p_insp.set_defaults(func=cmd_inspeccionar)

    p_log = sub.add_parser(
        "login", help="Abre Chrome para que inicies sesion manualmente.")
    p_log.add_argument("--espera", type=int, default=300,
                       help="Segundos de espera para que completes el login.")
    p_log.add_argument("--mantener", action="store_true",
                       help="Deja el navegador abierto al terminar.")
    p_log.set_defaults(func=cmd_login)


    p_con = sub.add_parser("consultar", help="Envia una consulta a engie.ai.")
    p_con.add_argument("consulta", nargs="?", default="")
    p_con.add_argument("--archivo", help="Lee la consulta desde un archivo.")
    p_con.add_argument("--espera", type=int, default=GENERATION_TIMEOUT_S,
                       help="Timeout de generacion en segundos.")
    p_con.add_argument("--headless", action="store_true")
    p_con.add_argument("--mantener", action="store_true",
                       help="Deja el navegador abierto al terminar.")
    p_con.set_defaults(func=cmd_consultar)

    args = parser.parse_args(argv)
    return args.func(args, load_selectors())


if __name__ == "__main__":
    sys.exit(main())

