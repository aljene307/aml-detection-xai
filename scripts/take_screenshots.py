"""Captures d'écran de l'application et de l'API, dans captures_brutes/ (ignoré par Git).

Chrome déjà installé (channel="chrome"), sans interface, dans un contexte neuf
et vide : aucun profil, cookie ni onglet de l'utilisateur. L'application est
lancée en local, reliée à l'API Render, réveillée avant les captures. Chaque
capture attend son texte, ses graphiques et la fin des calculs de Streamlit.

Usage : python scripts/take_screenshots.py
"""

import json
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

from aml_xai.config import ROOT

API = "https://aml-detection-xai-api.onrender.com"
PORT = 8502
APP = f"http://127.0.0.1:{PORT}"
OUT = ROOT / "captures_brutes"
WIDTH, SCALE = 1400, 2


def wait_http(url: str, limit_s: int = 300) -> None:
    start = time.perf_counter()
    while True:
        try:
            urllib.request.urlopen(url, timeout=90)
            return
        except OSError:
            if time.perf_counter() - start > limit_s:
                raise RuntimeError(f"{url} ne répond pas.")
            time.sleep(3)


def settle(page, texte: str, graphiques: int) -> None:
    """Attend le texte, les graphiques Plotly dessinés et la fin des calculs de Streamlit."""
    page.get_by_text(texte, exact=False).first.wait_for(timeout=180_000)
    if graphiques:
        page.wait_for_function(f"document.querySelectorAll('.js-plotly-plot .main-svg').length >= {graphiques}",
                               timeout=120_000)
    page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=180_000)
    page.wait_for_timeout(1500)


def capture_app(page, nom: str) -> None:
    """Toute la page : la fenêtre prend la hauteur du contenu (Streamlit défile dans un cadre)."""
    hauteur = page.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
    page.set_viewport_size({"width": WIDTH, "height": max(900, int(hauteur) + 40)})
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / nom))
    page.set_viewport_size({"width": WIDTH, "height": 900})


def main() -> None:
    OUT.mkdir(exist_ok=True)
    wait_http(API + "/health")  # réveil de l'API avant tout
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
                               "--server.headless", "true", "--server.port", str(PORT),
                               "--client.toolbarMode", "minimal", "--browser.gatherUsageStats", "false"],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              env={**__import__("os").environ, "API_URL": API})
    erreurs = {}
    try:
        wait_http(APP + "/_stcore/health")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(viewport={"width": WIDTH, "height": 900}, device_scale_factor=SCALE,
                                          locale="fr-FR")
            page = context.new_page()
            courant = {"nom": "démarrage"}
            page.on("console", lambda m: m.type == "error" and erreurs.setdefault(courant["nom"], []).append(m.text))
            page.on("pageerror", lambda e: erreurs.setdefault(courant["nom"], []).append(str(e)))

            for nom, modele in [("01_analyser_lof.png", None), ("02_analyser_lightgbm.png", "LightGBM")]:
                courant["nom"] = nom
                page.goto(APP + "/")  # la page par défaut est servie à la racine
                settle(page, "Exemple de départ", 0)
                if modele:
                    page.get_by_text(modele, exact=False).first.click()
                page.get_by_role("button", name="Analyser").click()
                settle(page, "Codes motifs", 1)
                capture_app(page, nom)

            for nom, chemin, texte, graphiques in [("03_budget.png", "/budget", "Au budget de", 1),
                                                    ("04_resultats.png", "/resultats", "Au seuil figé", 3)]:
                courant["nom"] = nom
                page.goto(APP + chemin)
                settle(page, texte, graphiques)
                capture_app(page, nom)

            courant["nom"] = "05_api_docs.png"
            page.goto(API + "/docs")
            page.wait_for_selector(".swagger-ui .opblock", timeout=120_000)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / "05_api_docs.png"), full_page=True)
            context.close()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)
    print(json.dumps({"captures": sorted(f.name for f in OUT.glob("*.png")), "erreurs_console": erreurs},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
