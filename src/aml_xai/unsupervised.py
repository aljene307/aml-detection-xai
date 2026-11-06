"""Méthodes non supervisées : apprentissage sans étiquettes et score d'anomalie.

Les réglages « sans étiquette » sont fixés ici avant toute mesure
(docs/CHOIX.md, sections 5, 6 et 22). Tous les scores sont orientés de la même
façon : plus le score est haut, plus la transaction est suspecte.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor

from aml_xai.config import SEED

METHODS = ["isolation_forest", "lof", "autoencodeur"]
# LOF garde ses lignes d'apprentissage en mémoire : il apprend sur un
# échantillon tiré au hasard. L'autoencodeur apprend sur le même échantillon.
SAMPLE_ROWS = 100_000
# Goulot de l'autoencodeur : 23 variables → 8 → 23
BOTTLENECK = 8
SCORE_CHUNK_ROWS = 200_000


def default_model(method: str, seed: int = SEED):
    """Modèle avec les réglages par défaut, choisis sans étiquettes."""
    if method == "isolation_forest":
        # 100 arbres, 256 lignes par arbre, toutes les variables
        return IsolationForest(random_state=seed, n_jobs=-1)
    if method == "lof":
        # 20 voisines ; novelty=True permet de noter de nouvelles lignes
        return LocalOutlierFactor(novelty=True, n_jobs=-1)
    if method == "autoencodeur":
        return MLPRegressor(hidden_layer_sizes=(BOTTLENECK,), random_state=seed)
    raise ValueError(f"Méthode inconnue : {method}")


def needs_sample(method: str) -> bool:
    return method in ("lof", "autoencodeur")


def sample_rows(n_rows: int, size: int = SAMPLE_ROWS, seed: int = SEED) -> np.ndarray:
    """Positions tirées au hasard sans remise, triées pour une lecture dans l'ordre."""
    if size >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=size, replace=False))


def fit(model, matrix: np.ndarray):
    """Apprentissage sans étiquettes ; l'autoencodeur apprend à recopier son entrée."""
    if isinstance(model, MLPRegressor):
        return model.fit(matrix, matrix)
    return model.fit(matrix)


def chunk_scores(model, chunk: np.ndarray) -> np.ndarray:
    if isinstance(model, MLPRegressor):
        return ((model.predict(chunk) - chunk) ** 2).mean(axis=1)
    # score_samples : plus haut = plus normal, d'où le signe moins
    return -model.score_samples(chunk)


def anomaly_scores(model, matrix: np.ndarray, chunk_rows: int = SCORE_CHUNK_ROWS) -> np.ndarray:
    """Score d'anomalie de chaque ligne, calculé par blocs pour borner la mémoire."""
    out = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), chunk_rows):
        chunk = np.asarray(matrix[start : start + chunk_rows], dtype=np.float32)
        out[start : start + chunk_rows] = chunk_scores(model, chunk)
    return out
