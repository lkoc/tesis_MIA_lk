"""Sonda de conectividad y captura del DOM real de engie.ai.

Guarda el HTML y un JSON de diagnostico para ajustar los selectores de
engie_bridge.py. Se ejecuta en segundo plano porque la red corporativa puede
tardar mas que el timeout de la terminal que lo invoca.
"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\_probe_result.json")
HTML = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_lk\_engie.html")
URL = "https://engie.ai"


def main():
    out = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page()
            try:
                response = page.goto(URL, wait_until="domcontentloaded",
                                     timeout=60000)
                out["status"] = response.status if response else None
                out["url"] = page.url
                out["title"] = page.title()

                # La app es una SPA de React Router: el DOM llega vacio en
                # domcontentloaded, hay que esperar a que hidrate.
                page.wait_for_load_state("networkidle", timeout=30000)
                page.wait_for_timeout(6000)

                HTML.write_text(page.content(), encoding="utf-8")
                out["body_inicio"] = page.locator("body").first.inner_text(
                    timeout=15000)[:3000]
                out["textareas"] = page.locator("textarea").count()
                out["contenteditables"] = page.locator(
                    '[contenteditable="true"]').count()
                out["buttons"] = page.locator("button").count()
                out["inputs"] = page.locator("input").count()

                # Inventario de controles, para deducir los selectores reales.
                out["button_labels"] = page.locator("button").evaluate_all(
                    "els => els.slice(0,40).map(e => "
                    "((e.getAttribute('aria-label')||'') + ' | ' + "
                    "(e.innerText||'').trim()).slice(0,80))")
                out["textarea_info"] = page.locator(
                    "textarea").evaluate_all(
                    "els => els.slice(0,10).map(e => "
                    "({ph: e.placeholder, testid: e.getAttribute('data-testid')}))")
                out["links"] = page.locator("a").evaluate_all(
                    "els => els.slice(0,25).map(e => "
                    "((e.innerText||'').trim() + ' -> ' + e.getAttribute('href')"
                    ").slice(0,90))")
            finally:
                browser.close()
    except Exception as exc:  # pragma: no cover - sondeo de diagnostico
        out["error"] = f"{type(exc).__name__}: {exc}"

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("PROBE_WRITTEN", OUT)


if __name__ == "__main__":
    main()
