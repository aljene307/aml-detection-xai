"""Explications SHAP et codes motifs sur la validation (docs/CHOIX.md, section 28).

- LightGBM réglé : réappris avec le réglage et l'arrêt précoce de l'étape 4 ;
  ses scores de validation doivent être identiques à ceux enregistrés.
  TreeSHAP sur toutes ses alertes au seuil retenu, test de suppression sur 500.
- LOF sans étiquette : modèle de l'essai de faisabilité (scores identiques à
  la grille, vérifié). KernelSHAP sur 100 alertes, stabilité sur 50.

Test de suppression : la règle fixée avant les mesures (médiane, groupes au
hasard parmi tous), et un second test conçu après les résultats, à titre
d'information (référence de SHAP, groupes au hasard parmi les autres).

Les étiquettes et `Laundering_type` servent seulement à décrire les alertes.
Le test n'est pas lu. Seules les lignes des alertes expliquées sont lues.

Deux cas, un processus chacun (pic de mémoire plus bas) :
- modele : réapprend LightGBM réglé, vérifie ses scores et l'enregistre ;
- explications : relit les modèles et calcule les explications.

Usage : python scripts/explain.py --cas modele, puis --cas explications
"""

import argparse
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from aml_xai.config import (
    FEATURES_PATH,
    MEASURES_DIR,
    MODELS_DIR,
    PARQUET_PATH,
    PROCESSED_DIR,
    SEED,
    TARGET,
    TIMESTAMP,
    TRANSACTION_ID,
    TYPOLOGY,
)
from aml_xai.explain import (
    both_tests,
    deletion_drop,
    group_matrix,
    kernel_background,
    kernel_contributions,
    lightgbm_contributions,
    load_codes,
    lof_score,
    phrase,
    reason_codes,
)
from aml_xai.features import model_feature_names
from aml_xai.matrix import load_matrix
from aml_xai.measure import measure
from aml_xai.split import model_periods
from aml_xai.supervised import EARLY_STOPPING_ROUNDS, MAX_TREES, lightgbm_model, ranking_scores

MATRIX_PATHS = {p: PROCESSED_DIR / f"matrice_{p}.npy" for p in ["entrainement", "validation"]}
TRIAL_MODELS_DIR = PROCESSED_DIR / "modeles_essai"
LGBM_PATH = TRIAL_MODELS_DIR / "lightgbm_regle.joblib"
LGBM_DELETION_ALERTS = 500
LOF_ALERTS = 100
LOF_STABILITY_ALERTS = 50
USUAL_VALUES_PATH = MODELS_DIR / "valeurs_habituelles.json"
# Variables citées dans les phrases : leur médiane d'entraînement sert de valeur habituelle
USUAL_FEATURES = ["em_ratio_montant_30j", "re_ratio_montant_30j", "em_nb_1h", "em_nb_24h", "re_nb_1h", "re_nb_24h",
                  "em_nb_distincts_24h", "em_nb_distincts_30j", "re_nb_distincts_30j", "em_ecart_precedent_s",
                  "re_ecart_precedent_s"]
RAW_COLUMNS = [TIMESTAMP, "Amount", "Payment_currency", "Received_currency", "Sender_bank_location",
               "Receiver_bank_location", "Payment_type", TARGET, TYPOLOGY]


def period_filter(start, end) -> list:
    return [(TIMESTAMP, ">=", start), (TIMESTAMP, "<", end)]


def alert_rows(periods: dict, ids: np.ndarray) -> pd.DataFrame:
    """Vraies valeurs des seules transactions expliquées, dans l'ordre de `ids`.

    Lit seulement les colonnes utiles de la validation, puis garde les lignes voulues.
    """
    validation = period_filter(periods["fin_entrainement"], periods["fin_validation"])
    raw = pd.read_parquet(PARQUET_PATH, columns=[TRANSACTION_ID, *RAW_COLUMNS], filters=validation)
    raw = raw[raw[TRANSACTION_ID].isin(ids)]
    columns = [TRANSACTION_ID, *USUAL_FEATURES, "em_sans_historique_30j", "re_sans_historique_30j"]
    features = pd.read_parquet(FEATURES_PATH, columns=columns, filters=validation)
    features = features[features[TRANSACTION_ID].isin(ids)]
    rows = raw.merge(features, on=TRANSACTION_ID, validate="one_to_one").set_index(TRANSACTION_ID)
    if len(rows) != len(ids):
        raise RuntimeError("Des transactions expliquées manquent.")
    return rows.loc[ids].reset_index()


def usual_values(periods: dict) -> dict:
    """Médianes de l'entraînement retenu, une colonne à la fois (valeurs habituelles des phrases)."""
    period = period_filter(periods["debut"], periods["fin_entrainement"])
    usual = {name: float(np.median(pq.read_table(FEATURES_PATH, columns=[name], filters=period)
                                   .column(name).to_numpy())) for name in USUAL_FEATURES}
    usual["Amount"] = float(np.median(pq.read_table(PARQUET_PATH, columns=["Amount"], filters=period)
                                      .column("Amount").to_numpy()))
    USUAL_VALUES_PATH.write_text(json.dumps(usual, indent=2), encoding="utf-8")
    return usual


def refit_lightgbm(train, y, valid, y_valid) -> lgb.LGBMClassifier:
    table = json.loads((MEASURES_DIR / "supervise_tableau.json").read_text(encoding="utf-8"))
    settings = next(r for r in table["lignes"] if r["methode"] == "LightGBM" and r["version"] == "reglee")["reglage"]
    model = lightgbm_model(settings, SEED, n_estimators=MAX_TREES, metric="average_precision")
    model.fit(train, y, eval_X=(valid,), eval_y=(y_valid,),
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return model


def describe(contributions: np.ndarray, base: np.ndarray, scores: np.ndarray, gmatrix: np.ndarray,
             names: list[str], rows: pd.DataFrame, usual: dict) -> list[dict]:
    """Codes motifs, contributions de groupe et phrases de chaque alerte."""
    groups = contributions @ gmatrix
    out = []
    for i in range(len(contributions)):
        row = rows.iloc[i].to_dict()
        codes = reason_codes(groups[i], names)
        out.append({
            "transaction_id": int(row[TRANSACTION_ID]),
            "blanchiment": int(row[TARGET]),
            "typologie": str(row[TYPOLOGY]),
            "score_de_risque": round(float(scores[i]), 4),
            "valeur_de_base": round(float(base[i]), 4),
            "codes": [{"code": c, "contribution": round(float(groups[i, names.index(c)]), 4),
                       "phrase": phrase(c, row, usual)} for c in codes],
        })
    return out


def codes_of(explained: dict) -> list[str]:
    return [c["code"] for c in explained["codes"]]


def code_frequencies(explained: list[dict], by_typology: bool = False) -> dict:
    df = pd.DataFrame([{"typologie": e["typologie"], "code": c["code"]} for e in explained for c in e["codes"]])
    if not by_typology:
        return (df["code"].value_counts() / len(explained)).round(4).to_dict()
    counts = pd.Series([e["typologie"] for e in explained]).value_counts()
    table = df.groupby("typologie")["code"].value_counts().unstack(fill_value=0)
    return {t: (table.loc[t] / counts[t]).sort_values(ascending=False).head(3).round(3).to_dict()
            for t in table.index if not t.startswith("Normal_")}


def case_model() -> None:
    """Réapprend LightGBM réglé ; ses scores doivent être identiques à ceux de l'étape 4."""
    result = {}
    with measure(result):
        periods = model_periods()
        evaluation = json.loads((MEASURES_DIR / "evaluation.json").read_text(encoding="utf-8"))
        train = np.asarray(load_matrix(MATRIX_PATHS["entrainement"]), dtype=np.float32)
        valid = np.asarray(load_matrix(MATRIX_PATHS["validation"]), dtype=np.float32)
        read = lambda start, end: pq.read_table(PARQUET_PATH, columns=[TARGET], filters=period_filter(  # noqa: E731
            periods[start], periods[end])).column(TARGET).to_numpy().astype(np.int8)
        model = refit_lightgbm(train, read("debut", "fin_entrainement"), valid,
                               read("fin_entrainement", "fin_validation"))
        info = evaluation["methodes"]["LightGBM, réglé"]
        saved = np.load(PROCESSED_DIR / "scores_supervise" / info["fichier_scores"])
        if not np.array_equal(ranking_scores(model, valid), saved):
            raise RuntimeError("LightGBM réappris ne redonne pas les scores de l'étape 4.")
        joblib.dump(model, LGBM_PATH)
        result["arbres"] = int(model.best_iteration_)
    print(f"Modèle LightGBM réglé vérifié et enregistré ({result['arbres']} arbres, "
          f"{result['duree_s']} s, pic {result['pic_memoire_mo']} Mo)")


def case_explanations() -> None:
    result = {}
    with measure(result):
        codes = load_codes()
        names = list(codes)
        gmatrix = group_matrix(codes)
        periods = model_periods()
        evaluation = json.loads((MEASURES_DIR / "evaluation.json").read_text(encoding="utf-8"))
        train = load_matrix(MATRIX_PATHS["entrainement"])
        valid = np.asarray(load_matrix(MATRIX_PATHS["validation"]), dtype=np.float32)
        validation = period_filter(periods["fin_entrainement"], periods["fin_validation"])
        labels = pq.read_table(PARQUET_PATH, columns=[TRANSACTION_ID, TARGET], filters=validation)
        ids_valid = labels.column(TRANSACTION_ID).to_numpy()
        y_valid = labels.column(TARGET).to_numpy().astype(np.int8)
        if len(y_valid) != len(valid):
            raise RuntimeError("Les étiquettes ne correspondent pas à la matrice de validation.")
        usual = usual_values(periods)
        # Remplacement du test de suppression : médiane (règle) ou moyenne (référence de SHAP)
        medians = np.array([np.median(train[:, j]) for j in range(train.shape[1])], dtype=np.float32)
        train_mean = np.array([train[:, j].mean(dtype=np.float64) for j in range(train.shape[1])], dtype=np.float32)

        # LightGBM réglé (réappris et vérifié par le cas « modele »)
        model = joblib.load(LGBM_PATH)
        scores = ranking_scores(model, valid)
        info = evaluation["methodes"]["LightGBM, réglé"]
        saved = np.load(PROCESSED_DIR / "scores_supervise" / info["fichier_scores"])
        if not np.array_equal(scores, saved):
            raise RuntimeError("Le modèle LightGBM enregistré ne redonne pas les scores de l'étape 4.")
        alerts = np.flatnonzero(scores >= info["seuils"]["retenu"]["seuil"])
        start = time.perf_counter()
        contributions, base = lightgbm_contributions(model, valid[alerts])
        tree_s = time.perf_counter() - start
        explained = describe(contributions, base, scores[alerts], gmatrix, names, alert_rows(periods, ids_valid[alerts]),
                             usual)
        sample = np.sort(np.random.default_rng(SEED).choice(len(alerts), LGBM_DELETION_ALERTS, replace=False))
        lgbm_score = lambda x: ranking_scores(model, np.asarray(x, dtype=np.float32))  # noqa: E731
        result["lightgbm"] = {
            "scores_identiques_etape_4": True,
            "arbres": int(model.best_iteration_),
            "alertes": len(alerts),
            "duree_par_alerte_ms": round(1000 * tree_s / len(alerts), 3),
            "additivite_ecart_max": float(np.abs(contributions.sum(axis=1) + base - scores[alerts]).max()),
            "importance_globale_groupes": dict(sorted(
                zip(names, np.abs(contributions @ gmatrix).mean(axis=0).round(4).tolist()), key=lambda kv: -kv[1])),
            "nombre_de_codes": pd.Series([len(e["codes"]) for e in explained]).value_counts().sort_index().to_dict(),
            "frequence_codes": code_frequencies(explained),
            "codes_par_typologie": code_frequencies([e for e in explained if e["blanchiment"]], by_typology=True),
            "suppression": both_tests(lgbm_score, valid[alerts[sample]], [codes_of(explained[i]) for i in sample],
                                      medians, train_mean, gmatrix, names),
        }

        # LOF sans étiquette
        lof = joblib.load(TRIAL_MODELS_DIR / "lof.joblib")
        info = evaluation["methodes"]["LOF, sans étiquette"]
        saved = np.load(PROCESSED_DIR / "scores_grilles" / info["fichier_scores"])
        score = lof_score(lof)
        if not np.allclose(score(valid[:2_000]), saved[:2_000], rtol=0, atol=1e-9):
            raise RuntimeError("Le modèle LOF ne redonne pas les scores de la grille.")
        lof_alerts = np.flatnonzero(saved >= info["seuils"]["retenu"]["seuil"])
        chosen = np.sort(np.random.default_rng(SEED).choice(lof_alerts, LOF_ALERTS, replace=False))
        background = kernel_background(train, SEED)
        start = time.perf_counter()
        values, kernel_base = kernel_contributions(score, valid[chosen], background)
        kernel_s = time.perf_counter() - start
        lof_scores = saved[chosen]
        lof_explained = describe(values, np.full(len(chosen), kernel_base), lof_scores, gmatrix, names,
                                 alert_rows(periods, ids_valid[chosen]), usual)
        background_mean = np.average(background.data, axis=0, weights=background.weights).astype(np.float32)
        other = kernel_background(train, 1)
        stable_values, _ = kernel_contributions(score, valid[chosen[:LOF_STABILITY_ALERTS]], other)
        same_list = same_set = 0
        for i in range(LOF_STABILITY_ALERTS):
            first = [c["code"] for c in lof_explained[i]["codes"]]
            second = reason_codes((stable_values @ gmatrix)[i], names)
            same_list += first == second
            same_set += set(first) == set(second)
        result["lof"] = {
            "alertes_expliquees": len(chosen),
            "duree_par_alerte_s": round(kernel_s / len(chosen), 2),
            "additivite_ecart_max": float(np.abs(values.sum(axis=1) + kernel_base - lof_scores).max()),
            "additivite_ecart_relatif_max": float(
                np.abs(values.sum(axis=1) + kernel_base - lof_scores).max() / np.abs(lof_scores - kernel_base).min()),
            "importance_globale_groupes": dict(sorted(
                zip(names, np.abs(values @ gmatrix).mean(axis=0).round(4).tolist()), key=lambda kv: -kv[1])),
            "nombre_de_codes": pd.Series([len(e["codes"]) for e in lof_explained]).value_counts().sort_index().to_dict(),
            "frequence_codes": code_frequencies(lof_explained),
            "suppression": both_tests(score, valid[chosen], [codes_of(e) for e in lof_explained], medians,
                                      background_mean, gmatrix, names),
            "stabilite": {"alertes": LOF_STABILITY_ALERTS,
                          "memes_codes_meme_ordre": round(same_list / LOF_STABILITY_ALERTS, 3),
                          "memes_codes": round(same_set / LOF_STABILITY_ALERTS, 3)},
        }
        result["exemples"] = {
            "lightgbm": [next(e for e in explained if e["typologie"] == t) for t in ["Structuring", "Fan_Out"]]
            + [next(e for e in explained if not e["blanchiment"])],
            "lof": [next((e for e in lof_explained if e["blanchiment"]), lof_explained[0]),
                    next(e for e in lof_explained if not e["blanchiment"])],
        }
    (MEASURES_DIR / "explications.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                                                    encoding="utf-8")
    print(f"Écrit : explications.json ({result['duree_s']} s, pic {result['pic_memoire_mo']} Mo)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cas", choices=["modele", "explications"], required=True)
    args = parser.parse_args()
    case_model() if args.cas == "modele" else case_explanations()


if __name__ == "__main__":
    main()
