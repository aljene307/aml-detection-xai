"""Fichiers lus par l'application sans appeler l'API (docs/CHOIX.md, section 33).

- data/demo/exemples_app.json : pour les 30 exemples, score, rang et alerte des
  deux modèles, explication de LightGBM (même code que l'API) et explication
  de LOF (calculée à l'avance, reprise des exemples) ;
- reports/mesures/budget_validation.json : règle « les k transactions les plus
  suspectes de chaque jour » sur la validation, pour k de 1 à 2 000 alertes par
  jour et les 5 modèles à seuil. Vérifiée à k = 140 contre l'étape 5.

Rien n'est calculé sur le test.

Usage : python scripts/build_app_data.py
"""

import json

import numpy as np
import pandas as pd

from aml_xai.api_service import MODELS, Service, rank_text
from aml_xai.config import MEASURES_DIR, PROCESSED_DIR, ROOT, TIMESTAMP, TYPOLOGY
from aml_xai.evaluation import period_frame
from aml_xai.frozen import load_frozen
from aml_xai.split import model_periods

BUDGETS = np.r_[np.arange(1, 301), np.arange(310, 1001, 10), np.arange(1050, 2001, 50)]
BUDGET_MODELS = ["Isolation Forest, sans étiquette", "LOF, sans étiquette", "LOF, réglé",
                 "LightGBM, par défaut", "LightGBM, réglé"]


def daily_curves(scores: np.ndarray, days: np.ndarray, labels: np.ndarray, amounts: np.ndarray,
                 over: np.ndarray) -> dict:
    """Pour chaque budget k : les k plus suspectes de chaque jour (ex æquo dans l'ordre des lignes)."""
    caught = labels * amounts
    caught_no_over = caught * ~over
    alerts, tp, detected, detected_no_over = (np.zeros(len(BUDGETS)) for _ in range(4))
    for day in np.unique(days):
        rows = np.flatnonzero(days == day)
        order = rows[np.lexsort((rows, -scores[rows]))]
        take = np.minimum(BUDGETS, len(order))
        for total, values in [(tp, labels), (detected, caught), (detected_no_over, caught_no_over)]:
            cumulative = np.r_[0, np.cumsum(values[order])]
            total += cumulative[take]
        alerts += take
    return {"alertes": alerts.astype(int).tolist(), "vp": tp.astype(int).tolist(),
            "montant_detecte": detected.round(2).tolist(),
            "montant_detecte_sans_over_invoicing": detected_no_over.round(2).tolist()}


def main() -> None:
    service = Service()
    examples = []
    for e in service.examples:
        x, row, _ = service.prepare(e["entree"])
        out = {k: e[k] for k in ("id", "categorie", "transaction_id", "blanchiment", "typologie", "entree")}
        for code in MODELS:
            value = service.score(code, x)
            if value != e["scores_attendus"][code]:
                raise RuntimeError(f"Exemple {e['id']} : score {code} différent du score attendu.")
            share = service.rank(code, value)
            out[code] = {"score": value, "seuil": service.threshold(code), "alerte": value >= service.threshold(code),
                         "rang": share, "rang_texte": rank_text(share)}
        out["lof"]["explication"] = e["explication_lof"]
        out["lightgbm"]["explication"] = service.explain_lightgbm(x, row)
        examples.append(out)
    (ROOT / "data" / "demo" / "exemples_app.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")

    periods = model_periods()
    df = period_frame(periods["fin_entrainement"], periods["fin_entrainement"], periods["fin_validation"])
    labels = df["Is_laundering"].to_numpy().astype(np.int64)
    amounts = df["Amount"].to_numpy()
    over = df[TYPOLOGY].astype(str).to_numpy() == "Over-Invoicing"
    days = df[TIMESTAMP].dt.floor("D").to_numpy()
    frozen = load_frozen()["modeles"]
    evaluation = json.loads((MEASURES_DIR / "evaluation.json").read_text(encoding="utf-8"))["methodes"]
    budget = {
        "periode": "validation",
        "regle": "les k transactions les plus suspectes de chaque jour",
        "jours_calendrier": int(len(np.unique(days))),
        "cas": int(labels.sum()),
        "montant_blanchi": round(float((labels * amounts).sum()), 2),
        "montant_blanchi_sans_over_invoicing": round(float((labels * amounts * ~over).sum()), 2),
        "alertes_par_jour": BUDGETS.tolist(),
        "modeles": {},
    }
    k140 = int(np.flatnonzero(BUDGETS == 140)[0])
    for name in BUDGET_MODELS:
        scores = np.load(PROCESSED_DIR / frozen[name]["scores_validation"])
        curves = daily_curves(scores, days, labels, amounts, over)
        reference = evaluation[name]["k_premieres_par_jour"]
        if curves["vp"][k140] != reference["vp"] or curves["alertes"][k140] != reference["vp"] + reference["fp"]:
            raise RuntimeError(f"{name} : la règle des 140 par jour ne retrouve pas l'étape 5.")
        budget["modeles"][name] = curves
    (MEASURES_DIR / "budget_validation.json").write_text(json.dumps(budget, ensure_ascii=False), encoding="utf-8")
    print(f"Écrits : exemples_app.json ({len(examples)} exemples), budget_validation.json "
          f"({len(BUDGETS)} budgets, vérifié à 140 par jour contre l'étape 5).")


if __name__ == "__main__":
    main()
