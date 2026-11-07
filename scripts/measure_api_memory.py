"""Mémoire de l'API dans un vrai serveur (Render : 512 Mo).

Lance `uvicorn api.main:app` dans un processus séparé, mesure sa mémoire au
démarrage, puis après 200 appels (deux modèles, /score et /explain, les 30
exemples), et la durée médiane des réponses.

Usage : python scripts/measure_api_memory.py
"""

import json
import subprocess
import sys
import time
import urllib.request

import numpy as np
import psutil

from aml_xai.config import MEASURES_DIR, ROOT

PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
CALLS = 200
LIMIT_MB = 512


def get(path: str):
    with urllib.request.urlopen(URL + path, timeout=60) as r:
        return json.loads(r.read())


def post(path: str, body: dict):
    request = urllib.request.Request(URL + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as r:
        return json.loads(r.read())


def memory_mb(process: psutil.Process) -> tuple[float, float]:
    """(mémoire actuelle, pic) du serveur et de ses processus enfants, en Mo."""
    procs = [process, *process.children(recursive=True)]
    infos = [p.memory_info() for p in procs]
    return (sum(i.rss for i in infos) / 2**20, sum(getattr(i, "peak_wset", i.rss) for i in infos) / 2**20)


def main() -> None:
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(PORT)],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process = psutil.Process(server.pid)
    try:
        start = time.perf_counter()
        while True:
            try:
                get("/health")
                break
            except OSError:
                if time.perf_counter() - start > 120:
                    raise RuntimeError("L'API n'a pas démarré en 2 minutes.")
                time.sleep(0.5)
        startup_s = time.perf_counter() - start
        at_start, _ = memory_mb(process)

        examples = [get(f"/examples/{e['id']}")["entree"] for e in get("/examples")]
        durations = {"lof": [], "lightgbm": []}
        for i in range(CALLS):
            model = "lof" if i % 2 == 0 else "lightgbm"
            path = "/explain" if i % 4 < 2 else "/score"
            t0 = time.perf_counter()
            post(f"{path}?modele={model}", examples[i % len(examples)])
            durations[model].append(time.perf_counter() - t0)
        after, peak = memory_mb(process)
        result = {
            "demarrage_s": round(startup_s, 1),
            "memoire_demarrage_mo": round(at_start, 1),
            "memoire_apres_appels_mo": round(after, 1),
            "pic_mo": round(peak, 1),
            "appels": CALLS,
            "limite_render_mo": LIMIT_MB,
            "marge_mo": round(LIMIT_MB - peak, 1),
            "duree_mediane_ms": {m: round(1000 * float(np.median(d)), 1) for m, d in durations.items()},
            "memoire_vue_par_health_mo": get("/health")["memoire_mo"],
        }
    finally:
        server.terminate()
        server.wait(timeout=30)
    (MEASURES_DIR / "api_memoire.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
