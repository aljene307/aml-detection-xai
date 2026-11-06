"""Version réglée des méthodes non supervisées : grilles sur la validation.

- grilles : apprend chaque réglage sans étiquettes, puis mesure sa PR-AUC et
  sa ROC-AUC sur la validation (seul usage des étiquettes). Chaque réglage est
  écrit dès qu'il est terminé (une ligne JSON) ; une relance reprend après le
  dernier réglage écrit. Ordre : autoencodeur, Isolation Forest, contrôle des
  graines, puis LOF (le plus long) ;
- tableau : versions par défaut et réglées, écart, variation due aux graines ;
- petit_echantillon : contrôle, règle fixée avant la mesure. 20 fois, on tire
  10 % des comptes émetteurs de la validation, on choisit le réglage avec leurs
  seules transactions, puis on le mesure sur les 90 % restantes. On compte les
  fois où le choix est celui de la grille complète ;
- one_hot : contrôle (docs/CHOIX.md, section 19), codage one-hot contre codage
  par fréquence, versions par défaut, mêmes lignes d'apprentissage pour les
  deux codages, mesurés sur les mêmes 20 % des comptes émetteurs de la
  validation ;
- graines_lof : contrôle, 4 autres tirages de l'échantillon de LOF pour les
  versions par défaut et réglée (environ 45 minutes, reprise possible).

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

import pandas as pd

from aml_xai.config import (
    ACCOUNT_COLUMNS,
    FREQUENCY_MAPS_PATH,
    MEASURES_DIR,
    PARQUET_PATH,
    PROCESSED_DIR,
    SEED,
    TARGET,
    TIMESTAMP,
)
from aml_xai.features import FREQUENCY_COLUMNS, frequency_feature, model_feature_names
from aml_xai.matrix import load_matrix, one_hot
from aml_xai.measure import available_ram_mb, peak_memory_mb
from aml_xai.split import model_periods
from aml_xai.unsupervised import (
    CONTROL_SEEDS,
    DEFAULT_SETTINGS,
    GRIDS,
    anomaly_scores,
    fit,
    default_model,
    model_from_settings,
    sample_rows,
    training_rows,
)

MATRIX_PATHS = {p: PROCESSED_DIR / f"matrice_{p}.npy" for p in ["entrainement", "validation"]}
RESULTS_PATH = MEASURES_DIR / "grilles.jsonl"
SCORES_DIR = PROCESSED_DIR / "scores_grilles"
MIN_FREE_RAM_MB = 1500
ABORT_FREE_RAM_MB = 500
ORDER = ["autoencodeur", "isolation_forest", "lof"]
# Contrôle « petit échantillon d'étiquettes »
SMALL_SHARE = 0.10
SMALL_REPEATS = 20
# Contrôle one-hot : Isolation Forest apprend sur un échantillon (87 colonnes
# sur tout l'entraînement pèseraient 1,7 Go) ; mesure sur 20 % des comptes
ONE_HOT_FOREST_ROWS = 1_000_000
ONE_HOT_VALID_SHARE = 0.20


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


def validation_column(name: str, n_rows: int) -> np.ndarray:
    periods = model_periods()
    table = pq.read_table(
        PARQUET_PATH,
        columns=[name],
        filters=[(TIMESTAMP, ">=", periods["fin_entrainement"]), (TIMESTAMP, "<", periods["fin_validation"])],
    )
    values = table.column(name).to_numpy()
    if len(values) != n_rows:
        raise RuntimeError("La colonne ne correspond pas à la matrice de validation.")
    return values


def validation_labels(n_rows: int) -> np.ndarray:
    return validation_column(TARGET, n_rows).astype(np.int8)


def scores_path(k: str):
    return SCORES_DIR / f"{k.replace('|', '__').replace('=', '-')}.npy"


def run_setting(method: str, settings: dict, seed: int, labels: np.ndarray) -> dict:
    free = available_ram_mb()
    if free < ABORT_FREE_RAM_MB:
        sys.exit(f"Arrêt : {free:.0f} Mo disponibles. Relancer la même commande pour reprendre.")
    train = load_matrix(MATRIX_PATHS["entrainement"])
    rows = training_rows(method, len(train), settings, seed)
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
    np.save(scores_path(k), scores)
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


def case_grids(job_list=None) -> None:
    free = available_ram_mb()
    if free < MIN_FREE_RAM_MB:
        sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles, {MIN_FREE_RAM_MB} requis.")
    labels = validation_labels(len(load_matrix(MATRIX_PATHS["validation"])))
    results = done_results()
    print(f"{len(results)} réglages déjà terminés, reprise après eux.", flush=True)
    for method, settings, seed in (job_list or jobs)(results):
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
    for method in ["isolation_forest", "lof"]:
        if not all(key(method, DEFAULT_SETTINGS[method], s) in results for s in CONTROL_SEEDS):
            continue
        seeds[method] = {}
        for name, settings in [("par_defaut", DEFAULT_SETTINGS[method]), ("reglee", table[method]["reglee"]["reglage"])]:
            values = [results[key(method, settings, s)]["pr_auc"] for s in CONTROL_SEEDS]
            seeds[method][name] = {
                "pr_auc": values,
                "moyenne": round(float(np.mean(values)), 6),
                "min": min(values),
                "max": max(values),
                "ecart_type": round(float(np.std(values)), 6),
            }
        seeds[method]["ecart_des_moyennes"] = round(seeds[method]["reglee"]["moyenne"] - seeds[method]["par_defaut"]["moyenne"], 6)
    second = max(["lof", "autoencodeur"], key=lambda m: table[m]["reglee"]["pr_auc"])
    labels_rate = float(validation_labels(len(load_matrix(MATRIX_PATHS["validation"]))).mean())
    out = {
        "taux_validation": round(labels_rate, 6),
        "methodes": table,
        "graines": seeds,
        "deuxieme_methode_sans_etiquette": "lof",
        "deuxieme_methode_reglee": second,
    }
    (MEASURES_DIR / "grilles_tableau.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


def case_small_sample() -> None:
    n_rows = len(load_matrix(MATRIX_PATHS["validation"]))
    labels = validation_labels(n_rows)
    senders = validation_column(ACCOUNT_COLUMNS[0], n_rows)
    accounts = np.unique(senders)
    results = done_results()
    out = {"part_des_comptes": SMALL_SHARE, "repetitions": SMALL_REPEATS, "methodes": {}}
    for method in ORDER:
        grid = [key(method, s, SEED) for s in GRIDS[method]]
        scores = {k: np.load(scores_path(k)) for k in grid}
        full_choice = best(results, method)["cle"]
        default = key(method, DEFAULT_SETTINGS[method], SEED)
        draws = []
        for repeat in range(SMALL_REPEATS):
            rng = np.random.default_rng(repeat)
            chosen_accounts = rng.choice(accounts, size=round(SMALL_SHARE * len(accounts)), replace=False)
            small = np.isin(senders, chosen_accounts)
            rest = ~small
            choice = max(grid, key=lambda k: average_precision_score(labels[small], scores[k][small]))
            rest_ap = {name: float(average_precision_score(labels[rest], scores[k][rest]))
                       for name, k in [("choisi", choice), ("grille_complete", full_choice), ("par_defaut", default)]}
            draws.append({"cas_petit_echantillon": int(labels[small].sum()), "choix": choice,
                          **{f"pr_auc_reste_{n}": round(v, 6) for n, v in rest_ap.items()}})
        del scores
        gc.collect()
        mean = {n: round(float(np.mean([d[f"pr_auc_reste_{n}"] for d in draws])), 6)
                for n in ["choisi", "grille_complete", "par_defaut"]}
        out["methodes"][method] = {
            "choix_grille_complete": full_choice,
            "fois_meme_choix": sum(d["choix"] == full_choice for d in draws),
            "pr_auc_moyenne_reste": mean,
            "choix": {c: sum(d["choix"] == c for d in draws) for c in sorted({d["choix"] for d in draws})},
            "cas_petit_echantillon_min_max": [min(d["cas_petit_echantillon"] for d in draws),
                                              max(d["cas_petit_echantillon"] for d in draws)],
            "tirages": draws,
        }
    out["comptes_emetteurs"] = len(accounts)
    (MEASURES_DIR / "petit_echantillon.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {m: {k: v for k, v in r.items() if k != "tirages"} for m, r in out["methodes"].items()}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def case_one_hot() -> None:
    free = available_ram_mb()
    if free < MIN_FREE_RAM_MB:
        sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles, {MIN_FREE_RAM_MB} requis.")
    maps = json.loads(FREQUENCY_MAPS_PATH.read_text(encoding="utf-8"))
    periods = model_periods()
    train, valid = load_matrix(MATRIX_PATHS["entrainement"]), load_matrix(MATRIX_PATHS["validation"])
    frequency_names = {frequency_feature(c) for c in FREQUENCY_COLUMNS}
    keep = [j for j, name in enumerate(model_feature_names()) if name not in frequency_names]

    def raw(start, end):
        return pd.read_parquet(PARQUET_PATH, columns=FREQUENCY_COLUMNS, filters=[(TIMESTAMP, ">=", start), (TIMESTAMP, "<", end)])

    raw_train = raw(periods["debut"], periods["fin_entrainement"])
    raw_valid = raw(periods["fin_entrainement"], periods["fin_validation"])
    if len(raw_train) != len(train) or len(raw_valid) != len(valid):
        raise RuntimeError("Les colonnes catégorielles ne correspondent pas aux matrices.")
    labels = validation_labels(len(valid))
    senders = validation_column(ACCOUNT_COLUMNS[0], len(valid))
    accounts = np.unique(senders)
    chosen = np.random.default_rng(SEED).choice(accounts, size=round(ONE_HOT_VALID_SHARE * len(accounts)), replace=False)
    subset = np.flatnonzero(np.isin(senders, chosen))
    y = labels[subset]

    def matrices(source, raw_part, rows):
        base = np.asarray(source[rows], dtype=np.float32)
        dummies, _ = one_hot(raw_part.iloc[rows], maps)
        return {"frequence": base, "one_hot": np.hstack([base[:, keep], dummies])}

    valid_versions = matrices(valid, raw_valid, subset)
    out = {
        "lignes_mesure": len(subset),
        "cas_mesure": int(y.sum()),
        "colonnes": {k: v.shape[1] for k, v in valid_versions.items()},
        "methodes": {},
    }
    for method in ORDER:
        rows = sample_rows(len(train), ONE_HOT_FOREST_ROWS) if method == "isolation_forest" else training_rows(
            method, len(train), DEFAULT_SETTINGS[method])
        train_versions = matrices(train, raw_train, rows)
        out["methodes"][method] = {"lignes_apprentissage": len(rows)}
        for name in ["frequence", "one_hot"]:
            start = time.perf_counter()
            model = fit(default_model(method), train_versions[name])  # sans étiquettes
            scores = anomaly_scores(model, valid_versions[name])
            out["methodes"][method][name] = {
                "pr_auc": round(float(average_precision_score(y, scores)), 6),
                "roc_auc": round(float(roc_auc_score(y, scores)), 6),
                "duree_s": round(time.perf_counter() - start, 1),
            }
            print(method, name, out["methodes"][method][name], flush=True)
        del train_versions
        gc.collect()
    out["pic_memoire_mo"] = round(peak_memory_mb())
    (MEASURES_DIR / "one_hot.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


def lof_seed_jobs(results: dict):
    """Autres tirages de l'échantillon de LOF, versions par défaut et réglée."""
    tuned = best(results, "lof")["reglage"]
    for settings in [DEFAULT_SETTINGS["lof"], tuned]:
        for seed in CONTROL_SEEDS[1:]:
            yield "lof", settings, seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cas", choices=["grilles", "tableau", "petit_echantillon", "one_hot", "graines_lof"], required=True)
    args = parser.parse_args()
    if args.cas == "grilles":
        case_grids()
    elif args.cas == "graines_lof":
        case_grids(lof_seed_jobs)
    elif args.cas == "tableau":
        case_table()
    elif args.cas == "one_hot":
        case_one_hot()
    else:
        case_small_sample()


if __name__ == "__main__":
    main()
