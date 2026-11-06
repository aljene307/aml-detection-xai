"""Matrice d'entrée des modèles : variables, codage par fréquence et transformation.

Même matrice pour toutes les méthodes (docs/CHOIX.md, section 22) :
`log(1 + x)` sur les variables à longue traîne, puis standardisation. Les
paramètres sont appris sur les lignes d'apprentissage retenues, sans étiquettes.

La construction traite une colonne à la fois (lire, transformer, écrire dans
la matrice) : le pic de mémoire reste proche de la taille de la matrice.
"""

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from aml_xai.config import FEATURES_PATH, PARQUET_PATH, TIMESTAMP, TRANSACTION_ID
from aml_xai.features import (
    FREQUENCY_COLUMNS,
    SIDES,
    apply_frequency_maps,
    frequency_feature,
    model_feature_names,
)

# Compteurs, rapports de montant et écarts : positifs, avec quelques valeurs
# énormes. Les autres variables sont bornées (sinus, jour, indicateurs, parts)
# ou déjà en logarithme (log_montant).
_LONG_TAIL = ["nb_1h", "nb_24h", "nb_distincts_24h", "nb_distincts_30j", "ratio_montant_30j", "ecart_precedent_s"]
LOG_FEATURES = [f"{side}_{name}" for side in SIDES for name in _LONG_TAIL if f"{side}_{name}" in model_feature_names()]


def _period_filter(start: pd.Timestamp, end: pd.Timestamp) -> list:
    return [(TIMESTAMP, ">=", start), (TIMESTAMP, "<", end)]


def _read_column(path: Path, name: str, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    table = pq.read_table(path, columns=[name], filters=_period_filter(start, end))
    return table.column(name).to_numpy()


def period_size(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return len(_read_column(FEATURES_PATH, TRANSACTION_ID, start, end))


def period_columns(start: pd.Timestamp, end: pd.Timestamp, maps: dict) -> Iterator[tuple[str, np.ndarray]]:
    """Variables des transactions de `[start ; end[`, une à une, dans l'ordre de model_feature_names()."""
    ids = _read_column(FEATURES_PATH, TRANSACTION_ID, start, end)
    if not np.array_equal(ids, _read_column(PARQUET_PATH, TRANSACTION_ID, start, end)):
        raise RuntimeError("Les deux fichiers ne donnent pas les mêmes lignes.")
    del ids
    raw = pd.read_parquet(PARQUET_PATH, columns=FREQUENCY_COLUMNS, filters=_period_filter(start, end))
    encoded = apply_frequency_maps(raw, maps)
    del raw
    frequency_names = {frequency_feature(c) for c in FREQUENCY_COLUMNS}
    for name in model_feature_names():
        if name in frequency_names:
            yield name, encoded.pop(name)
        else:
            yield name, _read_column(FEATURES_PATH, name, start, end)


def _logged(name: str, values: np.ndarray) -> np.ndarray:
    """Une colonne en float64, après le logarithme si elle est à longue traîne."""
    values = np.asarray(values, dtype=np.float64)
    return np.log1p(values) if name in LOG_FEATURES else values


def _moments(values: np.ndarray) -> tuple[float, float]:
    spread = float(values.std())
    return float(values.mean()), spread if spread > 0 else 1.0


def fit_transform_params(data: Mapping[str, np.ndarray]) -> dict:
    """Moyenne et écart type de chaque variable après le logarithme."""
    moments = [_moments(_logged(name, data[name])) for name in data.keys()]
    return {
        "variables": list(data.keys()),
        "log": LOG_FEATURES,
        "moyenne": [m for m, _ in moments],
        "ecart_type": [s for _, s in moments],
    }


def build_matrix(columns: Iterable[tuple[str, np.ndarray]], n_rows: int, params: dict | None = None):
    """Matrice float32 transformée, colonne par colonne.

    Sans `params`, les paramètres sont appris sur ces mêmes colonnes (lignes
    d'apprentissage). Renvoie (matrice, paramètres).
    """
    names = model_feature_names() if params is None else params["variables"]
    if params is not None and params["log"] != LOG_FEATURES:
        raise ValueError("Les paramètres ont été appris avec une autre liste de logarithmes.")
    out = np.empty((n_rows, len(names)), dtype=np.float32)
    mean, std = [], []
    for j, (name, values) in enumerate(columns):
        if j >= len(names) or name != names[j]:
            raise ValueError("Les colonnes ne correspondent pas aux paramètres appris.")
        logged = _logged(name, values)
        if params is None:
            m, s = _moments(logged)
        else:
            m, s = params["moyenne"][j], params["ecart_type"][j]
        mean.append(m)
        std.append(s)
        out[:, j] = (logged - m) / s
    if len(mean) != len(names):
        raise ValueError("Il manque des colonnes.")
    learned = {"variables": names, "log": LOG_FEATURES, "moyenne": mean, "ecart_type": std}
    return out, (learned if params is None else params)


def transform(data: Mapping[str, np.ndarray], params: dict) -> np.ndarray:
    """Matrice float32 transformée, colonnes dans l'ordre des paramètres."""
    if list(data.keys()) != params["variables"]:
        raise ValueError("Les colonnes ne correspondent pas aux paramètres appris.")
    n_rows = len(data[params["variables"][0]])
    return build_matrix(((name, data[name]) for name in params["variables"]), n_rows, params)[0]


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)


def load_matrix(path: Path) -> np.ndarray:
    """Lecture sans tout charger en mémoire (seules les pages lues sont chargées)."""
    return np.load(path, mmap_mode="r")
