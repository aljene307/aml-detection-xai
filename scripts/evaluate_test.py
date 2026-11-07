"""Étape 7b : ouvrir le test une seule fois et calculer les mesures figées.

Choix et mesures : docs/CHOIX.md, section 29 (config/fige_test.json). Les
modèles sont ceux vérifiés en 7a (reports/mesures/test_preparation.json),
contrôlés par leur empreinte. Les seuils sont appliqués tels quels.

- `--periode test` (par défaut) : écrit reports/mesures/test.json à la toute
  fin, en une fois ; refuse de tourner s'il existe déjà. `--correction "raison"`
  écrit test_correction_N.json et garde les fichiers précédents (section 7.3).
- `--periode validation` : répétition sur la validation, sans rien lire du
  test ; doit retrouver les chiffres des étapes 5 et 6. Écrit
  reports/mesures/repetition_validation.json.

Usage : python scripts/evaluate_test.py
"""

import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_xai.config import FREQUENCY_MAPS_PATH, MEASURES_DIR, MODELS_DIR, PARQUET_PATH, PROCESSED_DIR, SEED
from aml_xai.config import TARGET, TIMESTAMP, TYPOLOGY
from aml_xai.evaluation import (
    choose_thresholds,
    confusion_metrics,
    cost_curve,
    daily_top_k,
    false_alerts_by_type,
    load_hypotheses,
    novelty_shares,
    period_frame,
    typology_recall,
)
from aml_xai.explain import (
    both_tests,
    group_matrix,
    kernel_background,
    kernel_contributions,
    lightgbm_contributions,
    load_codes,
    lof_score,
    reason_codes,
)
from aml_xai.frozen import check_files, load_frozen, output_path, score, sha256_file
from aml_xai.matrix import build_matrix, load_matrix, period_columns, period_size
from aml_xai.measure import available_ram_mb, measure
from aml_xai.split import model_periods
from aml_xai.supervised import (
    BOOTSTRAP_DRAWS,
    account_bootstrap_weights,
    ranking,
    ranking_scores,
    weighted_average_precision,
)

MODELS_TEST_DIR = PROCESSED_DIR / "modeles_test"
MIN_FREE_RAM_MB = 1500
LGBM_DELETION_ALERTS = 500
LOF_ALERTS = 100
WITH_THRESHOLD = "seuil"


def load_models(frozen: dict) -> dict:
    """Modèles de 7a, contrôlés par leur empreinte."""
    preparation = json.loads((MEASURES_DIR / "test_preparation.json").read_text(encoding="utf-8"))
    models = {}
    for name in frozen["modeles"]:
        info = preparation["modeles"][name]
        path = MODELS_TEST_DIR / info["fichier"]
        if sha256_file(path) != info["empreinte_modele"]:
            raise RuntimeError(f"Le modèle {name} a changé depuis la préparation.")
        models[name] = joblib.load(path)
    return models


def period_bounds(periode: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    periods = model_periods()
    if periode == "validation":
        return periods["fin_entrainement"], periods["fin_validation"]
    last = pd.read_parquet(PARQUET_PATH, columns=[TIMESTAMP], filters=[
        (TIMESTAMP, ">=", periods["fin_validation"])])[TIMESTAMP].max()
    return periods["fin_validation"], last + pd.Timedelta(seconds=1)


def period_matrix(periode: str, start, end) -> np.ndarray:
    """Matrice de la période, avec les paramètres figés (rien n'est réappris)."""
    if periode == "validation":
        return np.asarray(load_matrix(PROCESSED_DIR / "matrice_validation.npy"), dtype=np.float32)
    maps = json.loads(FREQUENCY_MAPS_PATH.read_text(encoding="utf-8"))
    params = json.loads((MODELS_DIR / "transformation.json").read_text(encoding="utf-8"))
    matrix, _ = build_matrix(period_columns(start, end, maps), period_size(start, end), params)
    return matrix


def bootstrap_interval(labels, scores, senders) -> list[float]:
    rank = ranking(scores)
    values = [weighted_average_precision(labels, w, rank) for w in account_bootstrap_weights(senders)]
    return [round(float(np.percentile(values, 2.5)), 6), round(float(np.percentile(values, 97.5)), 6)]


def threshold_measures(scores, threshold, df, labels, amounts, hypotheses, days, capacity) -> dict:
    alerted = scores >= threshold
    over = df[TYPOLOGY].astype(str).to_numpy() == "Over-Invoicing"
    alert_cost = hypotheses["cout_alerte_livres"]
    total = float((amounts * labels).sum())
    missed = float((amounts * labels)[~alerted].sum())
    test_best = choose_thresholds(cost_curve(scores, labels, amounts, alert_cost), capacity)["retenu"]
    daily_k = int(round(hypotheses["alertes_par_jour"]))
    metrics = confusion_metrics(alerted, labels)
    return {
        "alertes": int(alerted.sum()),
        "alertes_par_jour": round(alerted.sum() / days, 1),
        **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
        "cout_total": round(missed + alerted.sum() * alert_cost, 2),
        "cout_sans_over_invoicing": round(float((amounts * labels)[~alerted & ~over].sum())
                                          + alerted.sum() * alert_cost, 2),
        "part_montant_detecte": round(1 - missed / total, 4),
        "k_premieres_par_jour": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in confusion_metrics(
            daily_top_k(scores, df[TIMESTAMP].dt.floor("D").to_numpy(), daily_k), labels).items()},
        "information_seuil_cout_minimal_sur_la_periode": {
            "alertes_par_jour": round(test_best["alertes"] / days, 1),
            "cout_total": round(float(test_best["cout_total"]), 2),
        },
        "typologies": typology_recall(alerted, labels, df[TYPOLOGY].astype(str).to_numpy()).round(4).to_dict("index"),
        "fausses_alertes_par_type": false_alerts_by_type(
            alerted, labels, df[TYPOLOGY].astype(str).to_numpy()).round(4).head(6).to_dict(),
        "nouveaute": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in novelty_shares(alerted, labels, {
            "emetteur_sans_historique_30j": df["em_sans_historique"].to_numpy(),
            "emetteur_jamais_vu_en_entrainement": df["em_jamais_vu"].to_numpy(),
            "destinataire_sans_historique_30j": df["re_sans_historique"].to_numpy(),
            "destinataire_jamais_vu_en_entrainement": df["re_jamais_vu"].to_numpy(),
        }).items()},
    }


def explanation_measures(models, frozen, scores, matrix, labels, typologies) -> dict:
    """Mêmes règles qu'en section 28 : additivité, suppression (règle et second test)."""
    codes = load_codes()
    names = list(codes)
    gmatrix = group_matrix(codes)
    train = load_matrix(PROCESSED_DIR / "matrice_entrainement.npy")
    medians = np.array([np.median(train[:, j]) for j in range(train.shape[1])], dtype=np.float32)
    train_mean = np.array([train[:, j].mean(dtype=np.float64) for j in range(train.shape[1])], dtype=np.float32)
    out = {}

    name = "LightGBM, réglé"
    model = models[name]
    alerts = np.flatnonzero(scores[name] >= frozen["modeles"][name]["seuil"])
    contributions, base = lightgbm_contributions(model, matrix[alerts])
    groups = contributions @ gmatrix
    alert_codes = [reason_codes(g, names) for g in groups]
    sample = np.sort(np.random.default_rng(SEED).choice(len(alerts), min(LGBM_DELETION_ALERTS, len(alerts)),
                                                         replace=False))
    lgbm = lambda x: ranking_scores(model, np.asarray(x, dtype=np.float32))  # noqa: E731
    detected = labels[alerts] == 1
    frequencies = pd.DataFrame([{"typologie": typologies[alerts][i], "code": c}
                                for i in np.flatnonzero(detected) for c in alert_codes[i]])
    out[name] = {
        "alertes": len(alerts),
        "additivite_ecart_max": float(np.abs(contributions.sum(axis=1) + base - scores[name][alerts]).max()),
        "importance_globale_groupes": dict(zip(names, np.abs(groups).mean(axis=0).round(4).tolist())),
        "codes_par_typologie": {t: g["code"].value_counts().head(3).to_dict() for t, g in frequencies.groupby("typologie")},
        "suppression": both_tests(lgbm, matrix[alerts[sample]], [alert_codes[i] for i in sample], medians,
                                  train_mean, gmatrix, names),
    }

    name = "LOF, sans étiquette"
    lof = lof_score(models[name])
    alerts = np.flatnonzero(scores[name] >= frozen["modeles"][name]["seuil"])
    chosen = np.sort(np.random.default_rng(SEED).choice(alerts, min(LOF_ALERTS, len(alerts)), replace=False))
    background = kernel_background(train, SEED)
    start = time.perf_counter()
    values, kernel_base = kernel_contributions(lof, matrix[chosen], background)
    reference = np.average(background.data, axis=0, weights=background.weights).astype(np.float32)
    out[name] = {
        "alertes_expliquees": len(chosen),
        "duree_par_alerte_s": round((time.perf_counter() - start) / len(chosen), 2),
        "additivite_ecart_max": float(np.abs(values.sum(axis=1) + kernel_base - scores[name][chosen]).max()),
        "importance_globale_groupes": dict(zip(names, np.abs(values @ gmatrix).mean(axis=0).round(4).tolist())),
        "suppression": both_tests(lof, matrix[chosen], [reason_codes(g, names) for g in values @ gmatrix],
                                  medians, reference, gmatrix, names),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periode", choices=["test", "validation"], default="test")
    parser.add_argument("--correction", help="raison écrite d'une correction (section 7.3 du plan)")
    args = parser.parse_args()
    if args.periode == "test":
        destination = output_path(MEASURES_DIR, args.correction)  # refus avant toute lecture du test
    else:
        destination = MEASURES_DIR / "repetition_validation.json"
    free = available_ram_mb()
    if free < MIN_FREE_RAM_MB:
        sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles, {MIN_FREE_RAM_MB} requis.")

    result = {"periode": args.periode, "correction": args.correction}
    with measure(result):
        frozen = load_frozen()
        result["configuration_identique"] = all(check_files(frozen["configuration"]).values())
        result["matrices_identiques"] = all(check_files(frozen["matrices"], PROCESSED_DIR).values())
        models = load_models(frozen)
        print("Choix figés et modèles vérifiés.", flush=True)

        start, end = period_bounds(args.periode)
        periods = model_periods()
        matrix = period_matrix(args.periode, start, end)
        df = period_frame(periods["fin_entrainement"], start, end)
        if len(df) != len(matrix):
            raise RuntimeError("La matrice et les transactions n'ont pas le même nombre de lignes.")
        labels = df[TARGET].to_numpy().astype(np.int8)
        amounts = df["Amount"].to_numpy()
        senders = df["Sender_account"].to_numpy()
        days = (end - start).total_seconds() / 86_400
        hypotheses = load_hypotheses()
        capacity = hypotheses["alertes_par_jour"] * days
        result.update({"debut": str(start), "fin_exclue": str(end), "jours": round(days, 2),
                       "lignes": len(df), "cas": int(labels.sum()), "taux": round(float(labels.mean()), 6),
                       "capacite_alertes": round(capacity, 1)})
        print(f"Période lue : {len(df):,} transactions.".replace(",", " "), flush=True)

        scores, result["modeles"] = {}, {}
        for name, spec in frozen["modeles"].items():
            t0 = time.perf_counter()
            scores[name] = score(spec, models[name], matrix)
            pr = float(average_precision_score(labels, scores[name]))
            entry = {
                "pr_auc": round(pr, 6),
                "pr_auc_intervalle_95": bootstrap_interval(labels, scores[name], senders),
                "roc_auc": round(float(roc_auc_score(labels, scores[name])), 6),
                "fois_le_hasard": round(pr / labels.mean(), 1),
            }
            if spec["seuil"] is not None:
                entry[WITH_THRESHOLD] = threshold_measures(scores[name], spec["seuil"], df, labels, amounts,
                                                           hypotheses, days, capacity)
            result["modeles"][name] = entry
            print(f"{name} : PR-AUC {pr:.4f} ({time.perf_counter() - t0:.0f} s)", flush=True)

        lgbm, lof = scores["LightGBM, par défaut"], scores["LOF, sans étiquette"]
        over = df[TYPOLOGY].astype(str).to_numpy() == "Over-Invoicing"
        decimals = df["nombreuses_decimales"].to_numpy()
        result["artefacts"] = {}
        for label, drop in [("tous_les_cas", np.zeros(len(df), bool)), ("sans_over_invoicing", over & (labels == 1)),
                            ("sans_montants_a_nombreuses_decimales", decimals & (labels == 1)),
                            ("sans_les_deux", (over | decimals) & (labels == 1))]:
            keep = ~drop
            a = float(average_precision_score(labels[keep], lgbm[keep]))
            b = float(average_precision_score(labels[keep], lof[keep]))
            result["artefacts"][label] = {"cas": int(labels[keep].sum()), "pr_auc_lightgbm": round(a, 4),
                                          "pr_auc_lof": round(b, 4), "rapport": round(a / b, 1)}
        print("Explications (KernelSHAP sur LOF : quelques minutes)...", flush=True)
        result["explications"] = explanation_measures(models, frozen, scores, matrix, labels,
                                                      df[TYPOLOGY].astype(str).to_numpy())
        result["tirages_bootstrap"] = BOOTSTRAP_DRAWS
    # Écrit à la toute fin, en une fois : un arrêt en cours de route n'écrit rien
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"Écrit : {destination.name} ({result['duree_s']} s, pic {result['pic_memoire_mo']} Mo). "
          "Mesures terminées.", flush=True)


if __name__ == "__main__":
    main()
