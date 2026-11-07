"""Évaluation sur la validation : seuil de coût, métriques, typologies, nouveauté, artefacts.

Lit les scores de validation déjà enregistrés (étapes 3 et 4) : aucun modèle
n'est réentraîné, le test n'est pas lu. Les étiquettes, les montants et
`Laundering_type` servent ici à évaluer seulement.

Usage : python scripts/evaluate.py
"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from aml_xai.config import (
    ACCOUNT_COLUMNS,
    FEATURES_PATH,
    MEASURES_DIR,
    PARQUET_PATH,
    PROCESSED_DIR,
    TARGET,
    TIMESTAMP,
    TYPOLOGY,
)
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
from aml_xai.measure import measure
from aml_xai.split import model_periods

# Courbe de sensibilité : coût et rappel pour ces budgets (alertes par jour)
SENSITIVITY_PER_DAY = [25, 50, 100, 140, 200, 300, 500, 1000, 2000, 5000]


def file_name(key: str) -> str:
    return key.replace("|", "__").replace("=", "-") + ".npy"


def method_files() -> dict[str, str]:
    """Les cinq versions évaluées (docs/CHOIX.md, section 9), d'après les tableaux des étapes 3 et 4."""
    unsup = json.loads((MEASURES_DIR / "grilles_tableau.json").read_text(encoding="utf-8"))["methodes"]
    sup = json.loads((MEASURES_DIR / "supervise_tableau.json").read_text(encoding="utf-8"))

    def unsup_key(method, version):
        settings = unsup[method][version]["reglage"]
        return "|".join([method, *(f"{k}={v}" for k, v in settings.items()), "graine=42"])

    option = sup["desequilibre"]["retenue"]
    tuned = next(r for r in sup["lignes"] if r["methode"] == "LightGBM" and r["version"] == "reglee")["reglage"]
    grids, supervised = PROCESSED_DIR / "scores_grilles", PROCESSED_DIR / "scores_supervise"
    return {
        "Isolation Forest, sans étiquette": grids / file_name(unsup_key("isolation_forest", "sans_etiquette")),
        "LOF, sans étiquette": grids / file_name(unsup_key("lof", "sans_etiquette")),
        "LOF, réglé": grids / file_name(unsup_key("lof", "reglee")),
        "LightGBM, par défaut": supervised / file_name(f"desequilibre|option={option}|graine=42"),
        "LightGBM, réglé": supervised / file_name(
            "|".join(["grille", f"option={option}", *(f"{k}={v}" for k, v in tuned.items())])),
    }


def summary(row: dict, days: float) -> dict:
    return {
        "seuil": float(row["seuil"]),
        "alertes": int(row["alertes"]),
        "alertes_par_jour": round(row["alertes"] / days, 1),
        "cas_detectes": int(row["cas_detectes"]),
        "montant_manque": round(float(row["montant_manque"]), 2),
        "cout_alertes": round(float(row["cout_alertes"]), 2),
        "cout_total": round(float(row["cout_total"]), 2),
    }


def rounded(metrics: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}


def main() -> None:
    result = {}
    with measure(result):
        hypotheses = load_hypotheses()
        periods = model_periods()
        days = (periods["fin_validation"] - periods["fin_entrainement"]).total_seconds() / 86_400
        capacity = hypotheses["alertes_par_jour"] * days
        df = period_frame(periods["fin_entrainement"], periods["fin_entrainement"], periods["fin_validation"])
        labels = df[TARGET].to_numpy().astype(np.int8)
        amounts = df["Amount"].to_numpy()
        day_keys = df[TIMESTAMP].dt.floor("D").to_numpy()
        typologies = df[TYPOLOGY].astype(str).to_numpy()
        groups = {
            "emetteur_sans_historique_30j": df["em_sans_historique"].to_numpy(),
            "emetteur_jamais_vu_en_entrainement": df["em_jamais_vu"].to_numpy(),
            "destinataire_sans_historique_30j": df["re_sans_historique"].to_numpy(),
            "destinataire_jamais_vu_en_entrainement": df["re_jamais_vu"].to_numpy(),
        }
        total = float((amounts * labels).sum())
        over = typologies == "Over-Invoicing"
        result.update({
            "hypotheses": hypotheses,
            "jours_validation": round(days, 2),
            "capacite_alertes": round(capacity, 1),
            "cas": int(labels.sum()),
            "montant_blanchi_total": round(total, 2),
            "part_montant_over_invoicing": round(float((amounts * labels)[over].sum() / total), 4),
            "cout_sans_alerte": round(total, 2),
            "methodes": {},
        })
        daily_k = int(round(hypotheses["alertes_par_jour"]))
        for name, path in method_files().items():
            scores = np.load(path)
            if len(scores) != len(df):
                raise RuntimeError(f"{path.name} n'a pas le nombre de lignes de la validation.")
            curve = cost_curve(scores, labels, amounts, hypotheses["cout_alerte_livres"])
            chosen = choose_thresholds(curve, capacity)
            alerted = scores >= chosen["retenu"]["seuil"]
            daily = daily_top_k(scores, day_keys, daily_k)
            sensitivity = []
            for per_day in SENSITIVITY_PER_DAY:
                row = curve[curve["alertes"] <= per_day * days].iloc[-1]
                sensitivity.append({"alertes_par_jour": per_day, **summary(row, days),
                                    "rappel": round(row["cas_detectes"] / labels.sum(), 4)})
            result["methodes"][name] = {
                "fichier_scores": path.name,
                "pr_auc": round(float(average_precision_score(labels, scores)), 6),
                "seuils": {k: summary(v, days) for k, v in chosen.items() if k != "cout_minimal"}
                | {"cout_minimal": summary(chosen["cout_minimal"], days)
                   | {"respecte_capacite": chosen["cout_minimal"]["respecte_capacite"]}},
                "metriques_seuil_retenu": rounded(confusion_metrics(alerted, labels))
                | {"part_montant_detecte": round(float((amounts * labels)[alerted].sum() / total), 4)},
                # Sensibilité : coût au seuil retenu si les cas Over-Invoicing (75 % du montant) sont retirés
                "cout_retenu_sans_over_invoicing": round(
                    float((amounts * labels)[~alerted & ~over].sum() + alerted.sum() * hypotheses["cout_alerte_livres"]), 2),
                "k_premieres_par_jour": rounded(confusion_metrics(daily, labels)) | {"k_par_jour": daily_k},
                "sensibilite": sensitivity,
                "typologies": typology_recall(alerted, labels, typologies).round(4).to_dict("index"),
                "fausses_alertes_par_type": false_alerts_by_type(alerted, labels, typologies).round(4).head(6).to_dict(),
                "nouveaute": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                              for k, v in novelty_shares(alerted, labels, groups).items()},
            }
        # Section 26 : l'écart LightGBM / LOF tient-il sans les cas liés aux artefacts ?
        lof = np.load(method_files()["LOF, sans étiquette"])
        lgbm = np.load(method_files()["LightGBM, par défaut"])
        decimals = df["nombreuses_decimales"].to_numpy()
        artefacts = {}
        for name, drop in [("tous_les_cas", np.zeros(len(df), bool)),
                           ("sans_over_invoicing", over & (labels == 1)),
                           ("sans_montants_a_nombreuses_decimales", decimals & (labels == 1)),
                           ("sans_les_deux", (over | decimals) & (labels == 1))]:
            keep = ~drop
            a = float(average_precision_score(labels[keep], lgbm[keep]))
            b = float(average_precision_score(labels[keep], lof[keep]))
            artefacts[name] = {"cas": int(labels[keep].sum()), "pr_auc_lightgbm": round(a, 4),
                               "pr_auc_lof": round(b, 4), "rapport": round(a / b, 1)}
        result["artefacts"] = artefacts
    (MEASURES_DIR / "evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float),
                                                  encoding="utf-8")
    print(f"Écrit : evaluation.json ({result['duree_s']} s, pic {result['pic_memoire_mo']} Mo)")


if __name__ == "__main__":
    main()
