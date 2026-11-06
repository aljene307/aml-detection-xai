"""Méthodes non supervisées : matrice, faisabilité, puis règle de choix.

- matrice : apprend la transformation (sans étiquettes) et écrit les matrices
  d'entraînement et de validation. Le test n'est jamais construit ici ;
- faisabilite --methode X : apprend la version par défaut de X, note toute la
  validation, mesure durée, mémoire et taille du modèle (un processus par
  méthode, pour un pic de mémoire propre) ;
- regle : applique la règle de choix de la deuxième méthode, fixée avant les
  mesures (docs/CHOIX.md, section 6).

Aucun de ces cas ne lit les étiquettes.

Usage : python scripts/train_unsupervised.py --cas faisabilite --methode lof
"""

import argparse
import gc
import json
import sys
import time

import joblib
import numpy as np
from scipy.stats import skew

from aml_xai.config import FREQUENCY_MAPS_PATH, MEASURES_DIR, MODELS_DIR, PROCESSED_DIR
from aml_xai.matrix import LOG_FEATURES, build_matrix, load_matrix, period_columns, period_size, save_matrix
from aml_xai.measure import available_ram_mb, measure, peak_memory_mb
from aml_xai.split import model_periods
from aml_xai.unsupervised import (
    METHODS,
    SAMPLE_ROWS,
    SCORE_CHUNK_ROWS,
    chunk_scores,
    default_model,
    fit,
    needs_sample,
    sample_rows,
)

TRANSFORM_PATH = MODELS_DIR / "transformation.json"
MATRIX_PATHS = {p: PROCESSED_DIR / f"matrice_{p}.npy" for p in ["entrainement", "validation"]}
TRIAL_MODELS_DIR = PROCESSED_DIR / "modeles_essai"
MIN_FREE_RAM_MB = 1500

# Règle de faisabilité, fixée avant les mesures (docs/CHOIX.md, section 6)
MAX_FIT_S = 15 * 60
MAX_SCORE_S = 15 * 60
MAX_PEAK_MB = 1536
MAX_MODEL_MB = 100
# Si le premier bloc annonce plus que ce multiple de la limite, on arrête la
# notation et on garde l'estimation (la méthode est de toute façon infaisable)
STOP_FACTOR = 2


def with_raw_skewness(columns, out: dict):
    """Laisse passer les colonnes en notant l'asymétrie brute des variables à longue traîne."""
    for name, values in columns:
        if name in LOG_FEATURES:
            out[name] = round(float(skew(values.astype(np.float64))), 2)
        yield name, values


def case_matrix(r: dict) -> None:
    bounds = model_periods()
    maps = json.loads(FREQUENCY_MAPS_PATH.read_text(encoding="utf-8"))
    start, end = bounds["debut"], bounds["fin_entrainement"]
    raw_skew = {}
    columns = with_raw_skewness(period_columns(start, end, maps), raw_skew)
    matrix, params = build_matrix(columns, period_size(start, end))
    TRANSFORM_PATH.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    r["lignes"] = {"entrainement": len(matrix)}
    r["asymetrie"] = {
        name: {"avant": raw_skew[name], "apres_log": round(float(skew(matrix[:, j].astype(np.float64))), 2)}
        for j, name in enumerate(params["variables"])
        if name in LOG_FEATURES
    }
    # Contrôle colonne par colonne (une copie float64 de toute la matrice coûterait 900 Mo)
    moments = [(matrix[:, j].mean(dtype=np.float64), matrix[:, j].std(dtype=np.float64)) for j in range(matrix.shape[1])]
    r["moyenne_max_abs"] = round(max(abs(float(m)) for m, _ in moments), 6)
    r["ecart_type_min_max"] = [round(float(min(s for _, s in moments)), 4), round(float(max(s for _, s in moments)), 4)]
    save_matrix(matrix, MATRIX_PATHS["entrainement"])
    del matrix
    gc.collect()

    start, end = bounds["fin_entrainement"], bounds["fin_validation"]
    valid, _ = build_matrix(period_columns(start, end, maps), period_size(start, end), params)
    r["lignes"]["validation"] = len(valid)
    save_matrix(valid, MATRIX_PATHS["validation"])
    r["periodes"] = {k: str(v) for k, v in bounds.items()}


def score_with_limit(model, matrix: np.ndarray, r: dict) -> np.ndarray | None:
    """Note toute la matrice ; s'arrête si le premier bloc annonce une durée infaisable."""
    out = np.empty(len(matrix), dtype=np.float64)
    start_time = time.perf_counter()
    for start in range(0, len(matrix), SCORE_CHUNK_ROWS):
        chunk = np.asarray(matrix[start : start + SCORE_CHUNK_ROWS], dtype=np.float32)
        out[start : start + SCORE_CHUNK_ROWS] = chunk_scores(model, chunk)
        if start == 0:
            projected = (time.perf_counter() - start_time) * len(matrix) / len(chunk)
            r["notation_estimee_s"] = round(projected, 1)
            if projected > STOP_FACTOR * MAX_SCORE_S:
                r["notation_arretee"] = True
                return None
    r["notation_s"] = round(time.perf_counter() - start_time, 1)
    return out


def case_feasibility(r: dict, method: str) -> None:
    train = load_matrix(MATRIX_PATHS["entrainement"])
    rows = sample_rows(len(train)) if needs_sample(method) else np.arange(len(train))
    r["lignes_apprentissage"] = len(rows)
    data = np.asarray(train[rows] if needs_sample(method) else train, dtype=np.float32)
    model = default_model(method)
    start = time.perf_counter()
    fit(model, data)
    r["apprentissage_s"] = round(time.perf_counter() - start, 1)
    if hasattr(model, "n_iter_"):
        r["iterations"] = int(model.n_iter_)
        r["converge"] = bool(model.n_iter_ < model.max_iter)
    del data, train
    gc.collect()

    TRIAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRIAL_MODELS_DIR / f"{method}.joblib"
    joblib.dump(model, path)
    r["taille_modele_mo"] = round(path.stat().st_size / 2**20, 1)

    valid = load_matrix(MATRIX_PATHS["validation"])
    scores = score_with_limit(model, valid, r)
    if scores is not None:
        r["lignes_notees"] = len(scores)
        r["scores_finis"] = bool(np.isfinite(scores).all())
        np.save(TRIAL_MODELS_DIR / f"scores_validation_{method}.npy", scores)


def feasible(m: dict) -> dict:
    checks = {
        "apprentissage": m["apprentissage_s"] <= MAX_FIT_S,
        "notation": not m.get("notation_arretee") and m.get("notation_s", np.inf) <= MAX_SCORE_S,
        "memoire": m["pic_memoire_mo"] <= MAX_PEAK_MB,
        "taille": m["taille_modele_mo"] <= MAX_MODEL_MB,
    }
    return {**checks, "faisable": all(checks.values())}


def case_rule(r: dict) -> None:
    measures = {m: json.loads((MEASURES_DIR / f"faisabilite_{m}.json").read_text(encoding="utf-8")) for m in METHODS}
    r["limites"] = {
        "apprentissage_s": MAX_FIT_S,
        "notation_s": MAX_SCORE_S,
        "pic_memoire_mo": MAX_PEAK_MB,
        "taille_modele_mo": MAX_MODEL_MB,
    }
    r["verdicts"] = {m: feasible(measures[m]) for m in METHODS}
    lof, ae = r["verdicts"]["lof"]["faisable"], r["verdicts"]["autoencodeur"]["faisable"]
    # LOF si les deux passent (un seul réglage principal), sinon celle qui passe
    r["deuxieme_methode"] = "lof" if lof else ("autoencodeur" if ae else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cas", choices=["matrice", "faisabilite", "regle"], required=True)
    parser.add_argument("--methode", choices=METHODS)
    args = parser.parse_args()
    if args.cas != "regle":
        free = available_ram_mb()
        if free < MIN_FREE_RAM_MB:
            sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles, {MIN_FREE_RAM_MB} requis.")
    result = {}
    with measure(result):
        if args.cas == "matrice":
            case_matrix(result)
            path = MEASURES_DIR / "matrice.json"
        elif args.cas == "faisabilite":
            if args.methode is None:
                sys.exit("--methode est requis pour la faisabilité.")
            result["methode"] = args.methode
            result["echantillon"] = SAMPLE_ROWS if needs_sample(args.methode) else None
            case_feasibility(result, args.methode)
            path = MEASURES_DIR / f"faisabilite_{args.methode}.json"
        else:
            case_rule(result)
            path = MEASURES_DIR / "faisabilite.json"
    if args.cas == "faisabilite":
        result["pic_memoire_mo"] = round(peak_memory_mb())
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
