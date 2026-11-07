"""Évaluation au seuil : coût, capacité, métriques, typologies et nouveauté.

Coût défini par le sujet (module 4), en livres sterling (monnaie des montants) :
somme des montants des blanchiments non détectés + nombre d'alertes × coût
d'une alerte. Une alerte = une transaction dont le score atteint le seuil.

Règle fixée avant les mesures (docs/CHOIX.md, section 9) : le seuil retenu est
celui de coût minimal parmi ceux qui respectent la capacité de l'équipe. Le
seuil de coût minimal sans contrainte et le seuil au budget sont montrés pour
comparaison.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aml_xai.config import (
    ACCOUNT_COLUMNS,
    FEATURES_PATH,
    PARQUET_PATH,
    ROOT,
    TARGET,
    TIMESTAMP,
    TYPOLOGY,
)
from aml_xai.eda import wilson_interval
from aml_xai.supervised import ranking

HYPOTHESES_PATH = ROOT / "config" / "hypotheses.json"


def load_hypotheses(path: Path = HYPOTHESES_PATH) -> dict:
    """Hypothèses écrites, et les deux valeurs qui en découlent par formule."""
    h = json.loads(Path(path).read_text(encoding="utf-8"))
    h["alertes_par_jour"] = h["analystes"] * h["heures_par_jour"] * 60 / h["minutes_par_alerte"]
    h["cout_alerte_livres"] = h["minutes_par_alerte"] / 60 * h["cout_horaire_livres"]
    return h


def cost_curve(scores: np.ndarray, labels: np.ndarray, amounts: np.ndarray, alert_cost: float) -> pd.DataFrame:
    """Coût de chaque seuil possible (une valeur distincte de score), plus « aucune alerte ».

    Alerte si score >= seuil ; les ex æquo sont toujours alertés ensemble.
    """
    order, ends = ranking(scores)
    caught = labels[order].astype(np.int64)
    detected = np.cumsum(amounts[order] * caught)[ends]
    total = float((amounts * labels).sum())
    alerts = ends + 1
    curve = pd.DataFrame({
        "seuil": scores[order][ends],
        "alertes": alerts,
        "cas_detectes": np.cumsum(caught)[ends],
        "montant_manque": total - detected,
    })
    none = pd.DataFrame({"seuil": [np.inf], "alertes": [0], "cas_detectes": [0], "montant_manque": [total]})
    curve = pd.concat([none, curve], ignore_index=True)
    curve["cout_alertes"] = curve["alertes"] * alert_cost
    curve["cout_total"] = curve["montant_manque"] + curve["cout_alertes"]
    return curve


def choose_thresholds(curve: pd.DataFrame, capacity: float) -> dict:
    """Seuil de coût minimal, seuil au budget, et seuil retenu (coût minimal sous la capacité)."""
    free = curve.loc[curve["cout_total"].idxmin()]
    within = curve[curve["alertes"] <= capacity]
    retained = within.loc[within["cout_total"].idxmin()]
    budget = within.loc[within["alertes"].idxmax()]
    return {
        "cout_minimal": free.to_dict() | {"respecte_capacite": bool(free["alertes"] <= capacity)},
        "au_budget": budget.to_dict(),
        "retenu": retained.to_dict(),
    }


def confusion_metrics(alerted: np.ndarray, labels: np.ndarray) -> dict:
    """Précision, rappel, F2 (le rappel compte deux fois plus) et MCC."""
    labels = labels.astype(bool)
    tp = int((alerted & labels).sum())
    fp = int((alerted & ~labels).sum())
    fn = int((~alerted & labels).sum())
    tn = int((~alerted & ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {"vp": tp, "fp": fp, "fn": fn, "vn": tn, "precision": precision, "rappel": recall,
            "f2": f2, "mcc": float(mcc)}


def daily_top_k(scores: np.ndarray, days: np.ndarray, k: int) -> np.ndarray:
    """Complément : chaque jour, les k transactions aux scores les plus hauts."""
    rank = pd.Series(scores).groupby(days).rank(method="first", ascending=False)
    return (rank <= k).to_numpy()


def typology_recall(alerted: np.ndarray, labels: np.ndarray, typologies: np.ndarray) -> pd.DataFrame:
    """Rappel par typologie de blanchiment, avec l'intervalle de Wilson à 95 %."""
    cases = pd.DataFrame({"typologie": typologies[labels == 1], "detecte": alerted[labels == 1]})
    table = cases.groupby("typologie")["detecte"].agg(cas="size", detectes="sum")
    low, high = wilson_interval(table["detectes"], table["cas"])
    table["rappel"] = table["detectes"] / table["cas"]
    table["ic_bas"], table["ic_haut"] = low, high
    return table.sort_values("rappel")


def false_alerts_by_type(alerted: np.ndarray, labels: np.ndarray, typologies: np.ndarray) -> pd.Series:
    """Part des fausses alertes par type de transaction normale."""
    false = typologies[alerted & (labels == 0)]
    return pd.Series(false).value_counts(normalize=True)


def novelty_shares(alerted: np.ndarray, labels: np.ndarray, groups: dict[str, np.ndarray]) -> dict:
    """Pour chaque groupe (par exemple « émetteur sans historique ») : sa part des
    transactions, sa part des alertes, et la précision des alertes dans et hors du groupe."""
    out = {}
    labels = labels.astype(bool)
    for name, member in groups.items():
        inside, outside = alerted & member, alerted & ~member
        out[name] = {
            "part_transactions": float(member.mean()),
            "part_alertes": float(inside.sum() / max(alerted.sum(), 1)),
            "precision_dans_groupe": float(labels[inside].mean()) if inside.any() else 0.0,
            "precision_hors_groupe": float(labels[outside].mean()) if outside.any() else 0.0,
            "taux_blanchiment_groupe": float(labels[member].mean()) if member.any() else 0.0,
        }
    return out


def period_frame(train_end: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Transactions de `[start ; end[` pour l'évaluation : montants, étiquettes, typologies,
    absence d'historique, comptes jamais vus avant `train_end`, montants à nombreuses décimales."""
    period = [(TIMESTAMP, ">=", start), (TIMESTAMP, "<", end)]
    columns = [TIMESTAMP, *ACCOUNT_COLUMNS, "Amount", TARGET, TYPOLOGY]
    df = pd.read_parquet(PARQUET_PATH, columns=["transaction_id", *columns], filters=period)
    history = pd.read_parquet(FEATURES_PATH, columns=["transaction_id", "em_sans_historique_30j",
                                                      "re_sans_historique_30j"], filters=period)
    if not np.array_equal(df["transaction_id"].to_numpy(), history["transaction_id"].to_numpy()):
        raise RuntimeError("Les deux fichiers ne donnent pas les mêmes lignes.")
    df["em_sans_historique"] = history["em_sans_historique_30j"].to_numpy() == 1
    df["re_sans_historique"] = history["re_sans_historique_30j"].to_numpy() == 1
    train = pd.read_parquet(PARQUET_PATH, columns=ACCOUNT_COLUMNS, filters=[(TIMESTAMP, "<", train_end)])
    for side, column in zip(["em", "re"], ACCOUNT_COLUMNS):
        df[f"{side}_jamais_vu"] = ~df[column].isin(set(train[column].unique()))
    cents = df["Amount"].to_numpy() * 100
    df["nombreuses_decimales"] = np.abs(cents - np.rint(cents)) > 1e-6
    return df
