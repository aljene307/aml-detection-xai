"""Mémoire de l'application Streamlit, en local.

1. Le serveur `streamlit run app/streamlit_app.py`, lancé comme en production,
   au repos (Streamlit n'exécute les pages qu'à la connexion d'un navigateur).
2. Un processus qui exécute les 5 pages avec l'outil de test de Streamlit,
   dont « Analyser » avec un vrai appel à l'API : pic de mémoire de ce processus.

Usage : python scripts/measure_app_memory.py
"""

import json
import subprocess
import sys
import time
import urllib.request

import psutil

from aml_xai.config import MEASURES_DIR, ROOT

PORT = 8601
PAGES = ["page_analyser", "page_exemples", "page_budget", "page_resultats", "page_methode"]


def run_pages() -> None:
    """Exécute les 5 pages dans ce processus et affiche sa mémoire (appelé avec --pages)."""
    from streamlit.testing.v1 import AppTest

    app_dir = str(ROOT / "app")
    for module in PAGES:
        script = "\n".join(["import sys", f"sys.path.insert(0, {app_dir!r})", f"import {module}",
                             f"{module}.afficher()"])
        app = AppTest.from_string(script, default_timeout=200).run()
        if module == "page_analyser":
            app.button[0].click().run()
        if app.exception:
            raise RuntimeError(f"{module} : {app.exception}")
    info = psutil.Process().memory_info()
    print(json.dumps({"actuelle": info.rss / 2**20, "pic": getattr(info, "peak_wset", info.rss) / 2**20}))


def main() -> None:
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
                               "--server.headless", "true", "--server.port", str(PORT)],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        start = time.perf_counter()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=5)
                break
            except OSError:
                if time.perf_counter() - start > 120:
                    raise RuntimeError("Streamlit n'a pas démarré en 2 minutes.")
                time.sleep(0.5)
        time.sleep(3)
        process = psutil.Process(server.pid)
        idle = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)]) / 2**20
    finally:
        server.terminate()
        server.wait(timeout=30)
    pages = subprocess.run([sys.executable, "-W", "ignore", __file__, "--pages"], cwd=ROOT,
                           capture_output=True, text=True, check=True)
    session = json.loads(pages.stdout.strip().splitlines()[-1])
    result = {
        "serveur_au_repos_mo": round(idle, 1),
        "cinq_pages_executees_mo": round(session["actuelle"], 1),
        "cinq_pages_pic_mo": round(session["pic"], 1),
        "note": "mesure sous Windows ; la page Analyser appelle la vraie API",
    }
    (MEASURES_DIR / "app_memoire.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_pages() if "--pages" in sys.argv else main()
