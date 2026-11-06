"""Modèle supervisé : déséquilibre, version réglée, courbe d'apprentissage, contrôle.

- tout : enchaîne les quatre phases, chaque calcul écrit dès qu'il est terminé
  (une ligne JSON) ; une relance reprend après le dernier calcul écrit :
  1. desequilibre : LightGBM par défaut, 3 traitements (graine 42), puis
     choix par bootstrap apparié sur la validation (docs/CHOIX.md, section 25) ;
  2. grille : traitement retenu par la règle,
     6 réglages avec arrêt précoce sur la PR-AUC de validation ;
  3. courbe : LightGBM par défaut avec 1, 5, 10, 25 et 100 % des comptes
     émetteurs de l'entraînement (5 tirages sous 100 %), nombre de cas noté ;
  4. logistique : régression logistique, contrôle ;
- tableau : tableau comparatif avec les méthodes non supervisées.

Le test n'est jamais lu.

Usage : python scripts/train_supervised.py --cas tout
"""

import argparse
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_xai.config import ACCOUNT_COLUMNS, MEASURES_DIR, PARQUET_PATH, PROCESSED_DIR, SEED, TARGET, TIMESTAMP
from aml_xai.matrix import load_matrix
from aml_xai.measure import available_ram_mb, peak_memory_mb
from aml_xai.split import model_periods
from aml_xai.supervised import (
    CURVE_SHARES,
    EARLY_STOPPING_ROUNDS,
    IMBALANCE_OPTIONS,
    MAX_TREES,
    TUNED_GRID,
    account_subset,
    choose_imbalance,
    lightgbm_model,
    logistic_model,
    ranking_scores,
    training_set,
)
from aml_xai.unsupervised import CONTROL_SEEDS

MATRIX_PATHS = {p: PROCESSED_DIR / f"matrice_{p}.npy" for p in ["entrainement", "validation"]}
RESULTS_PATH = MEASURES_DIR / "supervise.jsonl"
SCORES_DIR = PROCESSED_DIR / "scores_supervise"
DECISION_PATH = MEASURES_DIR / "desequilibre.json"
MIN_FREE_RAM_MB = 1500
ABORT_FREE_RAM_MB = 500


def key(phase: str, **parts) -> str:
    return "|".join([phase, *(f"{k}={v}" for k, v in parts.items())])


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


def period_column(name: str, start_key: str, end_key: str, n_rows: int) -> np.ndarray:
    periods = model_periods()
    table = pq.read_table(
        PARQUET_PATH, columns=[name], filters=[(TIMESTAMP, ">=", periods[start_key]), (TIMESTAMP, "<", periods[end_key])]
    )
    values = table.column(name).to_numpy()
    if len(values) != n_rows:
        raise RuntimeError(f"{name} ne correspond pas à la matrice.")
    return values


def imbalance_decision(data: "Data") -> dict:
    """Décision de la règle, calculée une fois à partir des scores de validation enregistrés."""
    if DECISION_PATH.exists():
        return json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    scores = {o: np.load(scores_path(key("desequilibre", option=o, graine=SEED))) for o in IMBALANCE_OPTIONS}
    decision = choose_imbalance(scores, data.y_valid, data.senders_valid)
    DECISION_PATH.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    return decision


def jobs(results: dict, chooser):
    """(clé, phase, réglages) dans l'ordre ; `chooser()` donne l'option retenue après la phase 1."""
    for option in IMBALANCE_OPTIONS:
        yield key("desequilibre", option=option, graine=SEED), "desequilibre", {"option": option, "graine": SEED}
    chosen = chooser()
    for settings in TUNED_GRID:
        yield key("grille", option=chosen, **settings), "grille", {"option": chosen, "graine": SEED, "reglage": settings}
    for share in CURVE_SHARES:
        for seed in CONTROL_SEEDS if share < 1 else [SEED]:
            yield key("courbe", option=chosen, part=share, graine=seed), "courbe", {
                "option": chosen, "graine": seed, "part": share}
    yield key("logistique", option=chosen), "logistique", {"option": chosen, "graine": SEED}


class Data:
    """Matrices et étiquettes, chargées une fois."""

    def __init__(self):
        self.train = np.asarray(load_matrix(MATRIX_PATHS["entrainement"]), dtype=np.float32)
        self.valid = np.asarray(load_matrix(MATRIX_PATHS["validation"]), dtype=np.float32)
        self.y = period_column(TARGET, "debut", "fin_entrainement", len(self.train)).astype(np.int8)
        self.y_valid = period_column(TARGET, "fin_entrainement", "fin_validation", len(self.valid)).astype(np.int8)
        self.senders = period_column(ACCOUNT_COLUMNS[0], "debut", "fin_entrainement", len(self.train))
        self.senders_valid = period_column(ACCOUNT_COLUMNS[0], "fin_entrainement", "fin_validation", len(self.valid))


def run_job(phase: str, job: dict, data: Data) -> dict:
    free = available_ram_mb()
    if free < ABORT_FREE_RAM_MB:
        sys.exit(f"Arrêt : {free:.0f} Mo disponibles. Relancer la même commande pour reprendre.")
    seed = job["graine"]
    rows = account_subset(data.senders, job["part"], seed) if phase == "courbe" else np.arange(len(data.y))
    labels = data.y[rows]
    kept, extra = training_set(job["option"], labels, seed)
    if kept is not None:
        rows, labels = rows[kept], labels[kept]
    x = data.train if len(rows) == len(data.train) else data.train[rows]

    fit_options = {}
    if phase == "logistique":
        model = logistic_model(job["option"], labels)
    elif phase == "grille":
        model = lightgbm_model(job["reglage"], seed, n_estimators=MAX_TREES, metric="average_precision", **extra)
        fit_options = {
            "eval_X": (data.valid,),
            "eval_y": (data.y_valid,),
            "callbacks": [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        }
    else:
        model = lightgbm_model(None, seed, **extra)
    start = time.perf_counter()
    model.fit(x, labels, **fit_options)
    fit_s = time.perf_counter() - start
    del x
    gc.collect()
    start = time.perf_counter()
    scores = ranking_scores(model, data.valid)
    score_s = time.perf_counter() - start

    ties = check_no_saturation(scores)
    result = {
        "phase": phase,
        **{k: v for k, v in job.items()},
        "lignes": int(len(rows)),
        "cas": int(labels.sum()),
        "pr_auc": round(float(average_precision_score(data.y_valid, scores)), 6),
        "roc_auc": round(float(roc_auc_score(data.y_valid, scores)), 6),
        "apprentissage_s": round(fit_s, 1),
        "notation_s": round(score_s, 1),
        "ex_aequo": ties,
    }
    if phase != "logistique":
        result["arbres"] = int(model.best_iteration_ or model.n_estimators)
        result["taille_modele_mo"] = round(len(model.booster_.model_to_string()) / 2**20, 2)
    return result, scores


# Contrôle de saturation (docs/CHOIX.md, section 25) : au plus 1 % des premiers
# ex æquo au score maximal
TOP_ROWS = 5000
MAX_TIED_SHARE = 0.01


def check_no_saturation(scores: np.ndarray) -> dict:
    top = np.sort(scores)[::-1][:TOP_ROWS]
    ties = {"au_maximum": int((scores == scores.max()).sum()), "distincts_en_tete": len(np.unique(top))}
    if ties["au_maximum"] > MAX_TIED_SHARE * TOP_ROWS:
        raise RuntimeError(f"Scores saturés : {ties['au_maximum']} lignes ex æquo au maximum.")
    return ties


def scores_path(k: str):
    return SCORES_DIR / f"{k.replace('|', '__').replace('=', '-')}.npy"


def case_all() -> None:
    free = available_ram_mb()
    if free < MIN_FREE_RAM_MB:
        sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles, {MIN_FREE_RAM_MB} requis.")
    data = Data()
    results = done_results()
    print(f"{len(results)} calculs déjà terminés, reprise après eux.", flush=True)
    for k, phase, job in jobs(results, lambda: imbalance_decision(data)["retenue"]):
        if k in results:
            continue
        result, scores = run_job(phase, job, data)
        result = {"cle": k, **result}
        if phase != "courbe" and job["graine"] == SEED:
            SCORES_DIR.mkdir(parents=True, exist_ok=True)
            np.save(scores_path(k), scores)
        append_result(result)
        results[k] = result
        print(f"{k} : PR-AUC {result['pr_auc']}, {result['cas']} cas, "
              f"{result['apprentissage_s'] + result['notation_s']:.1f} s, pic {peak_memory_mb():.0f} Mo", flush=True)
    print("Calculs terminés.", flush=True)


def case_table() -> None:
    """Tableau comparatif : méthodes non supervisées (étape 3) et supervisées."""
    results = done_results()
    unsupervised = json.loads((MEASURES_DIR / "grilles_tableau.json").read_text(encoding="utf-8"))
    rate = unsupervised["taux_validation"]
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    chosen = decision["retenue"]
    rows = []
    for method, name in [("isolation_forest", "Isolation Forest"), ("lof", "LOF"), ("autoencodeur", "Autoencodeur")]:
        for version, label in [("sans_etiquette", "aucune"), ("reglee", "validation (réglage)")]:
            r = unsupervised["methodes"][method][version]
            rows.append({"methode": name, "version": version, "etiquettes": label,
                         "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"]})
    default = results[key("desequilibre", option=chosen, graine=SEED)]
    tuned = max((r for r in results.values() if r["phase"] == "grille"), key=lambda r: r["pr_auc"])
    logistic = results[key("logistique", option=chosen)]
    for r, version, label in [
        (default, "par_defaut", "entraînement (apprentissage)"),
        (tuned, "reglee", "entraînement + validation (arrêt précoce)"),
        (logistic, "controle", "entraînement (apprentissage)"),
    ]:
        rows.append({"methode": "Régression logistique" if r["phase"] == "logistique" else "LightGBM",
                     "version": version, "etiquettes": label, "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
                     "reglage": r.get("reglage"), "arbres": r.get("arbres"),
                     "apprentissage_s": r["apprentissage_s"], "taille_modele_mo": r.get("taille_modele_mo")})
    for row in rows:
        row["fois_le_hasard"] = round(row["pr_auc"] / rate, 1)
    curve = {}
    for share in CURVE_SHARES:
        runs = [r for r in results.values() if r["phase"] == "courbe" and r["part"] == share]
        curve[str(share)] = {
            "cas": [r["cas"] for r in runs],
            "pr_auc": [r["pr_auc"] for r in runs],
            "pr_auc_moyenne": round(float(np.mean([r["pr_auc"] for r in runs])), 6),
        }
    lof = unsupervised["methodes"]["lof"]
    out = {
        "taux_validation": rate,
        "desequilibre": {
            "pr_auc": {o: results[key("desequilibre", option=o, graine=SEED)]["pr_auc"] for o in IMBALANCE_OPTIONS},
            **decision,
        },
        "lignes": rows,
        "courbe_apprentissage": curve,
        "reference_lof": {"sans_etiquette": lof["sans_etiquette"]["pr_auc"], "reglee": lof["reglee"]["pr_auc"]},
        "part_lof_sans_etiquette_sur_lightgbm_par_defaut": round(lof["sans_etiquette"]["pr_auc"] / default["pr_auc"], 4),
    }
    (MEASURES_DIR / "supervise_tableau.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cas", choices=["tout", "tableau"], required=True)
    args = parser.parse_args()
    case_all() if args.cas == "tout" else case_table()


if __name__ == "__main__":
    main()
