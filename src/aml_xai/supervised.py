"""Modèle supervisé (comparaison) : LightGBM, et la régression logistique comme contrôle.

Même matrice que les méthodes non supervisées (docs/CHOIX.md, section 22). Les
étiquettes de l'entraînement servent à apprendre ; celles de la validation à
l'arrêt précoce de la version réglée et aux choix (section 24).
"""

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

from aml_xai.config import SEED

# Traitements du déséquilibre, du plus simple au plus complexe (docs/CHOIX.md, section 25)
IMBALANCE_OPTIONS = ["aucun", "poids", "sous_echantillonnage"]
UNDERSAMPLE_SHARE = 0.10
# Règle (docs/CHOIX.md, section 25), fixée avant toute mesure : bootstrap
# apparié sur la validation, par comptes émetteurs entiers tirés avec remise
BOOTSTRAP_DRAWS = 200
REQUIRED_WIN_SHARE = 0.95

# Version réglée : petite grille, arrêt précoce sur la PR-AUC de validation
TUNED_GRID = [{"num_leaves": n, "learning_rate": r} for n in (15, 31, 63) for r in (0.05, 0.1)]
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 50

# Courbe d'apprentissage : part des comptes émetteurs de l'entraînement
CURVE_SHARES = [0.01, 0.05, 0.10, 0.25, 1.0]


def undersample(labels: np.ndarray, share: float = UNDERSAMPLE_SHARE, seed: int = SEED) -> np.ndarray:
    """Positions gardées : tous les cas, et une part des transactions normales tirée au hasard."""
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    kept = rng.choice(negatives, size=round(share * len(negatives)), replace=False)
    return np.sort(np.concatenate([np.flatnonzero(labels == 1), kept]))


def positive_weight(labels: np.ndarray) -> float:
    """Poids d'un cas : nombre de transactions normales par cas."""
    return float((labels == 0).sum() / max(int(labels.sum()), 1))


def training_set(option: str, labels: np.ndarray, seed: int = SEED):
    """(positions gardées ou None pour toutes, réglages ajoutés au modèle)."""
    if option == "aucun":
        return None, {}
    if option == "poids":
        return None, {"scale_pos_weight": positive_weight(labels)}
    if option == "sous_echantillonnage":
        return undersample(labels, seed=seed), {}
    raise ValueError(f"Option inconnue : {option}")


def lightgbm_model(settings: dict | None = None, seed: int = SEED, **extra) -> lgb.LGBMClassifier:
    """Réglages par défaut de LightGBM (100 arbres, 31 feuilles, vitesse 0,1), sauf `settings`."""
    return lgb.LGBMClassifier(random_state=seed, verbose=-1, **(settings or {}), **extra)


def logistic_model(option: str, labels: np.ndarray) -> LogisticRegression:
    """Contrôle : somme pondérée des variables ; même traitement du déséquilibre."""
    weight = {0: 1.0, 1: positive_weight(labels)} if option == "poids" else None
    return LogisticRegression(max_iter=1000, class_weight=weight)


def ranking_scores(model, x: np.ndarray) -> np.ndarray:
    """Score de classement : le score brut (log-odds), pas la probabilité.

    La probabilité arrondie à 1,0 rend ex æquo les transactions dont le modèle
    est très sûr, et l'ordre entre elles est perdu (docs/CHOIX.md, section 25).
    Le score brut donne le même ordre, sans cet arrondi.
    """
    if isinstance(model, LogisticRegression):
        return model.decision_function(x)
    return model.predict(x, raw_score=True)


def account_subset(senders: np.ndarray, share: float, seed: int = SEED) -> np.ndarray:
    """Positions des transactions d'une part des comptes émetteurs tirée au hasard."""
    if share >= 1:
        return np.arange(len(senders))
    accounts = np.unique(senders)
    chosen = np.random.default_rng(seed).choice(accounts, size=round(share * len(accounts)), replace=False)
    return np.flatnonzero(np.isin(senders, chosen))


def ranking(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordre décroissant des scores et fin de chaque groupe d'ex æquo, calculés une fois."""
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    return order, ends


def weighted_average_precision(labels: np.ndarray, weights: np.ndarray, rank: tuple) -> float:
    """PR-AUC (précision moyenne) pondérée, même définition que scikit-learn."""
    order, ends = rank
    w = weights[order].astype(np.float64)
    positives = labels[order] * w
    tp = np.cumsum(positives)[ends]
    seen = np.cumsum(w)[ends]
    precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def account_bootstrap_weights(senders: np.ndarray, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED):
    """Pour chaque tirage, le poids de chaque ligne : nombre de fois où son compte est tiré."""
    accounts, inverse = np.unique(senders, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        counts = np.bincount(rng.integers(0, len(accounts), len(accounts)), minlength=len(accounts))
        yield counts[inverse]


def choose_imbalance(scores: dict[str, np.ndarray], labels: np.ndarray, senders: np.ndarray,
                     draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> dict:
    """Règle fixée avant toute mesure : on part de « aucun » ; une option plus complexe
    remplace l'option retenue seulement si elle est meilleure dans au moins 95 %
    des tirages du bootstrap apparié (mêmes lignes pour les deux options)."""
    ranks = {option: ranking(scores[option]) for option in IMBALANCE_OPTIONS}
    values = {option: [] for option in IMBALANCE_OPTIONS}
    for weights in account_bootstrap_weights(senders, draws, seed):
        for option in IMBALANCE_OPTIONS:
            values[option].append(weighted_average_precision(labels, weights, ranks[option]))
    values = {option: np.array(v) for option, v in values.items()}
    chosen, comparisons = IMBALANCE_OPTIONS[0], []
    for option in IMBALANCE_OPTIONS[1:]:
        share = float((values[option] > values[chosen]).mean())
        comparisons.append({"option": option, "contre": chosen, "part_des_tirages_meilleure": round(share, 3),
                            "ecart_median": round(float(np.median(values[option] - values[chosen])), 6)})
        if share >= REQUIRED_WIN_SHARE:
            chosen = option
    return {
        "retenue": chosen,
        "tirages": draws,
        "comparaisons": comparisons,
        "pr_auc_intervalle_95": {o: [round(float(np.percentile(v, 2.5)), 6), round(float(np.percentile(v, 97.5)), 6)]
                                 for o, v in values.items()},
    }
