"""Version réglée des méthodes non supervisées : grilles sur la validation.

- grilles : apprend chaque réglage sans étiquettes, puis mesure sa PR-AUC et
  sa ROC-AUC sur la validation (seul usage des étiquettes). Chaque réglage est
  écrit dès qu'il est terminé (une ligne JSON) ; une relance reprend après le
  dernier réglage écrit. Ordre : autoencodeur, Isolation Forest, contrôle des
  graines, puis LOF (le plus long) ;
- tableau : versions par défaut et réglées, écart, variation due aux graines.

Le test n'est jamais lu.

Usage : python scripts/tune_unsupervised.py --cas grilles
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_xai.config import MEASURES_DIR, PARQUET_PATH, PROCESSED_DIR, SEED, TARGET, TIMESTAMP
from aml_xai.matrix import load_matrix
from aml_xai.measure import available_ram_mb, peak_memory_mb
from aml_xai.split import model_periods
from aml_xai.unsupervised import (
    CONTROL_SEEDS,
    DEFAULT_SETTINGS,
    GRIDS,
    anomaly_scores,
    fit,
    model_from_settings,
    training_rows,
)

MATRIX_PATHS = {p: PROCESSED_DIR / f"matrice_{p}.npy" for p in ["entrainement", "validation"]}
RESULTS_PATH = MEASURES_DIR / "grilles.jsonl"
SCORES_DIR = PROCESSED_DIR / "scores_grilles"
MIN_FREE_RAM_MB = 1500
ABORT_FREE_RAM_MB = 500
ORDER = ["autoencodeur", "isolation_forest", "lof"]


def key(method: str, settings: dict, seed: int) -> str:
    parts = [f"{k}={v}" for k, v in settings.items()]
    return "|".join([method, *parts, f"graine={seed}"])


def done_results() -> dict[str, dict]:
    if not RESULTS_PATH.exists():
        return {}
    lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    return {r["cle"]: r for r in map(json.loads, filter(None, lines))}


def append_result(result: dict) -> None:
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def validation_labels(n_rows: int) -> np.ndarray:
    periods = model_periods()
    table = pq.read_table(
        PARQUET_PATH,
        columns=[TARGET],
        filters=[(TIMESTAMP, ">=", periods["fin_entrainement"]), (TIMESTAMP, "<", periods["fin_validation"])],
    )
    labels = table.column(TARGET).to_numpy().astype(np.int8)
    if len(labels) != n_rows:
        raise RuntimeError("Les étiquettes ne correspondent pas à la matrice de validation.")
    return labels


def run_setting(method: str, settings: dict, seed: int, labels: np.ndarray) -> dict:
    free = available_ram_mb()
    if free < ABORT_FREE_RAM_MB:
        sys.exit(f"Arrêt : {free:.0f} Mo disponibles. Relancer la même commande pour reprendre.")
    train = load_matrix(MATRIX_PATHS["entrainement"])
    rows = training_rows(method, len(train), settings)
    data = np.asarray(train if rows is None else train[rows], dtype=np.float32)
    model = model_from_settings(method, settings, seed)
    start = time.perf_counter()
    fit(model, data)  # sans étiquettes
    fit_s = time.perf_counter() - start
    del data, train
    gc.collect()
    start = time.perf_counter()
    scores = anomaly_scores(model, load_matrix(MATRIX_PATHS["validation"]))
    score_s = time.perf_counter() - start
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    k = key(method, settings, seed)
    np.save(SCORES_DIR / f"{k.replace('|', '__').replace('=', '-')}.npy", scores)
    result = {
        "cle": k,
        "methode": method,
        "reglage": settings,
        "graine": seed,
        "par_defaut": settings == DEFAULT_SETTINGS[method],
        "pr_auc": round(float(average_precision_score(labels, scores)), 6),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "apprentissage_s": round(fit_s, 1),
        "notation_s": round(score_s, 1),
        "duree_s": round(fit_s + score_s, 1),
    }
    if hasattr(model, "n_iter_"):
        result["iterations"] = int(model.n_iter_)
    return result


def best(results: dict, method: str) -> dict:
    candidates = [r for r in results.values() if r["methode"] == method and r["graine"] == SEED]
    return max(candidates, key=lambda r: r["pr_auc"])


def jobs(results: dict):
    """Réglages à calculer, dans l'ordre ; le contrôle des graines vient après la grille d'Isolation Forest."""
    for method in ORDER:
        for settings in GRIDS[method]:
            yield method, settings, SEED
        if method == "isolation_forest":
            tuned = best(results, method)["reglage"]
            for settings in [DEFAULT_SETTINGS[method]] + ([tuned] if tuned != DEFAULT_SETTINGS[method] else []):
                for seed in CONTROL_SEEDS[1:]:
                    yield method, settings, seed


def case_grids() -> None:
    free = available_ram_mb()
    if free < MIN_FREE_RAM_MB:
        sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles, {MIN_FREE_RAM_MB} requis.")
    labels = validation_labels(len(load_matrix(MATRIX_PATHS["validation"])))
    results = done_results()
    print(f"{len(results)} réglages déjà terminés, reprise après eux.", flush=True)
    for method, settings, seed in jobs(results):
        k = key(method, settings, seed)
        if k in results:
            continue
        result = run_setting(method, settings, seed, labels)
        append_result(result)
        results[k] = result
        print(f"{k} : PR-AUC {result['pr_auc']}, {result['duree_s']} s, pic {peak_memory_mb():.0f} Mo", flush=True)
    print("Grilles terminées.", flush=True)


def case_table() -> None:
    results = done_results()
    table = {}
    for method in ORDER:
        default = results[key(method, DEFAULT_SETTINGS[method], SEED)]
        tuned = best(results, method)
        table[method] = {
            "sans_etiquette": {k: default[k] for k in ["reglage", "pr_auc", "roc_auc", "duree_s"]},
            "reglee": {k: tuned[k] for k in ["reglage", "pr_auc", "roc_auc", "duree_s"]},
            "ecart_pr_auc": round(tuned["pr_auc"] - default["pr_auc"], 6),
            "reglages_essayes": sum(r["methode"] == method and r["graine"] == SEED for r in results.values()),
        }
    seeds = {}
    for name, settings in [("par_defaut", DEFAULT_SETTINGS["isolation_forest"]),
                           ("reglee", table["isolation_forest"]["reglee"]["reglage"])]:
        values = [results[key("isolation_forest", settings, s)]["pr_auc"] for s in CONTROL_SEEDS]
        seeds[name] = {"pr_auc": values, "min": min(values), "max": max(values), "ecart_type": round(float(np.std(values)), 6)}
    second = max(["lof", "autoencodeur"], key=lambda m: table[m]["reglee"]["pr_auc"])
    labels_rate = float(validation_labels(len(load_matrix(MATRIX_PATHS["validation"]))).mean())
    out = {
        "taux_validation": round(labels_rate, 6),
        "methodes": table,
        "graines_isolation_forest": seeds,
        "deuxieme_methode_sans_etiquette": "lof",
        "deuxieme_methode_reglee": second,
    }
    (MEASURES_DIR / "grilles_tableau.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cas", choices=["grilles", "tableau"], required=True)
    args = parser.parse_args()
    case_grids() if args.cas == "grilles" else case_table()


if __name__ == "__main__":
    main()
