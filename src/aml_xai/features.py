"""Variables de la transaction, variables d'historique par compte et codage.

Règle unique de l'historique (docs/CHOIX.md, section 18) : on ne regarde que
`[t − fenêtre ; t[`. La transaction actuelle et ses ex æquo à la même seconde
ne comptent jamais, ce qui garantit qu'aucune variable ne voit le futur.
"""

import gc
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from aml_xai.config import (
    ACCOUNT_COLUMNS,
    CATEGORICAL_COLUMNS,
    FORBIDDEN_FEATURES,
    TIMESTAMP,
    TRANSACTION_ID,
)
from aml_xai.eda import currency_differs

HOUR_S = 3_600
DAY_S = 86_400
WINDOW_30D_S = 30 * DAY_S

# Valeurs quand l'historique est vide (docs/CHOIX.md, section 18)
NEUTRAL_RATIO = 1.0
# Plafond de l'écart, valeur aussi sans transaction précédente (section 21) :
# au-delà de 30 jours, l'écart observable dépendrait de la date
GAP_CAP_S = WINDOW_30D_S

TRANSACTION_FEATURES = [
    "log_montant",
    "heure_sin",
    "heure_cos",
    "jour_semaine",
    "devises_differentes",
]
HISTORY_FEATURES = [
    "nb_1h",
    "nb_24h",
    "ratio_montant_30j",
    "ecart_precedent_s",
    "sans_historique_30j",
]
# Candidates mesurées avant de décider (docs/CHOIX.md, section 20). `nb_30j` ne
# sert qu'à vérifier que `nb_distincts_30j` ne répète pas le compteur.
COUNTERPARTY_CANDIDATES = ["nb_distincts_24h", "nb_distincts_30j", "nb_30j"]
# Décision de la section 20 : re_nb_distincts_24h répète le compteur (Spearman 0,974)
KEPT_COUNTERPARTY_FEATURES = {
    "em": ["nb_distincts_24h", "nb_distincts_30j"],
    "re": ["nb_distincts_30j"],
}

# em = historique des envois de l'émetteur, re = réceptions du destinataire
SIDES = {"em": ACCOUNT_COLUMNS[0], "re": ACCOUNT_COLUMNS[1]}
FREQUENCY_COLUMNS = [c for c in CATEGORICAL_COLUMNS if c not in FORBIDDEN_FEATURES]


def _counterparty_names(counterparty_features: list[str] | None) -> dict[str, list[str]]:
    """Contreparties par côté : celles retenues, ou une même liste pour les deux côtés."""
    if counterparty_features is None:
        return KEPT_COUNTERPARTY_FEATURES
    return {side: list(counterparty_features) for side in SIDES}


def history_columns(counterparty_features: list[str] | None = None) -> list[str]:
    """Noms des colonnes d'historique, pour les deux côtés de la transaction."""
    extra = _counterparty_names(counterparty_features)
    return [f"{side}_{name}" for side in SIDES for name in [*HISTORY_FEATURES, *extra[side]]]


def frequency_feature(column: str) -> str:
    return f"freq_{column}"


def model_feature_names() -> list[str]:
    """Variables d'entrée des modèles."""
    return [
        *TRANSACTION_FEATURES,
        *[frequency_feature(c) for c in FREQUENCY_COLUMNS],
        *history_columns(),
    ]


def transaction_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Variables calculées à partir de la seule ligne de la transaction."""
    ts = df[TIMESTAMP].dt
    angle = 2 * np.pi * (ts.hour * HOUR_S + ts.minute * 60 + ts.second).to_numpy() / DAY_S
    return {
        "log_montant": np.log10(df["Amount"].to_numpy()).astype(np.float32),
        "heure_sin": np.sin(angle).astype(np.float32),
        "heure_cos": np.cos(angle).astype(np.float32),
        "jour_semaine": ts.day_of_week.to_numpy().astype(np.float32),
        "devises_differentes": currency_differs(df).to_numpy().astype(np.float32),
    }


def _to_cents(amount: np.ndarray) -> np.ndarray:
    """Montants arrondis au centime entier : les sommes cumulées restent exactes.

    Avec des nombres à virgule, les arrondis d'une somme cumulée dépendraient
    des autres comptes ; ici, le résultat d'un compte ne dépend que de lui.
    """
    return np.rint(amount * 100).astype(np.int64)


@lru_cache(maxsize=1)
def _distinct_kernel():
    """Fonction compilée à la demande : numba est lourd à importer."""
    import numba

    @numba.njit
    def kernel(other, start, end, n_other):
        # Deux curseurs avancent : `end` fait entrer des lignes, `start` en fait sortir.
        counts = np.zeros(n_other, dtype=np.int32)
        out = np.zeros(len(start), dtype=np.int32)
        low = high = distinct = 0
        for i in range(len(start)):
            while high < end[i]:
                if counts[other[high]] == 0:
                    distinct += 1
                counts[other[high]] += 1
                high += 1
            while low < start[i]:
                counts[other[low]] -= 1
                if counts[other[low]] == 0:
                    distinct -= 1
                low += 1
            out[i] = distinct
        return out

    return kernel


def account_history(
    account: np.ndarray,
    t: np.ndarray,
    cents: np.ndarray,
    other: np.ndarray | None = None,
    n_other: int = 0,
    counterparty_features: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Variables d'historique d'un côté, renvoyées dans l'ordre des lignes reçues.

    `account` et `other` sont des codes entiers positifs, `t` des secondes
    positives. Une ligne n'a besoin que de l'historique de son propre compte.
    """
    order = np.lexsort((t, account))
    account_s, t_s, cents_s = account[order], t[order], cents[order]
    # Clé croissante : compte, puis temps. L'écart entre deux comptes dépasse la
    # plus longue fenêtre, donc une fenêtre ne déborde jamais sur un autre compte.
    span = int(t.max()) + WINDOW_30D_S + 1
    key = account_s.astype(np.int64) * span + t_s

    # Lignes triées avant la première ligne de la même seconde : borne « t[ »
    end = np.searchsorted(key, key, side="left")

    def window_start(window_s: int) -> np.ndarray:
        return np.searchsorted(key, key - window_s, side="left")

    start_30d = window_start(WINDOW_30D_S)
    nb_30d = end - start_30d
    cumulative = np.concatenate([[0], np.cumsum(cents_s)])
    total_30d = cumulative[end] - cumulative[start_30d]
    empty = nb_30d == 0
    # Le dénominateur vaut 1 pour une fenêtre vide : la valeur est remplacée juste après.
    ratio = cents_s * nb_30d.astype(np.float64) / np.where(empty, 1, total_30d)
    ratio[empty] = NEUTRAL_RATIO

    has_previous = np.zeros(len(key), dtype=bool)
    has_previous[end > 0] = account_s[end[end > 0] - 1] == account_s[end > 0]
    gap = np.full(len(key), float(GAP_CAP_S))
    gap[has_previous] = np.minimum(t_s[has_previous] - t_s[end[has_previous] - 1], GAP_CAP_S)

    sorted_values = {
        "nb_1h": end - window_start(HOUR_S),
        "nb_24h": end - window_start(DAY_S),
        "ratio_montant_30j": ratio,
        "ecart_precedent_s": gap,
        "sans_historique_30j": empty,
    }
    if "nb_30j" in counterparty_features:
        sorted_values["nb_30j"] = nb_30d
    if "nb_distincts_24h" in counterparty_features or "nb_distincts_30j" in counterparty_features:
        kernel = _distinct_kernel()
        other_s = other[order]
        if "nb_distincts_24h" in counterparty_features:
            sorted_values["nb_distincts_24h"] = kernel(other_s, window_start(DAY_S), end, n_other)
        if "nb_distincts_30j" in counterparty_features:
            sorted_values["nb_distincts_30j"] = kernel(other_s, start_30d, end, n_other)

    output = {}
    for name, values in sorted_values.items():
        restored = np.empty(len(key), dtype=np.float32)
        restored[order] = values
        output[name] = restored
    return output


def history_features(
    df: pd.DataFrame,
    n_chunks: int = 1,
    counterparty_features: list[str] | None = None,
    on_chunk=None,
) -> dict[str, np.ndarray]:
    """Variables d'historique des deux côtés, sur toutes les lignes de `df`.

    Avec `n_chunks` > 1, les comptes sont répartis en groupes et chaque groupe
    est traité avec tout son historique : un compte n'est jamais coupé, donc le
    résultat est identique au calcul en un seul morceau.
    """
    names = _counterparty_names(counterparty_features)
    n = len(df)
    t = df[TIMESTAMP].to_numpy().astype("datetime64[s]").astype(np.int64)
    t = t - t.min()
    cents = _to_cents(df["Amount"].to_numpy())
    columns = history_columns(counterparty_features)
    output = {c: np.empty(n, dtype=np.float32) for c in columns}
    for prefix, account_column in SIDES.items():
        other_column = next(c for c in ACCOUNT_COLUMNS if c != account_column)
        account = pd.factorize(df[account_column])[0].astype(np.int32)
        other, other_values = pd.factorize(df[other_column])
        other = other.astype(np.int32)
        for chunk in range(n_chunks):
            if on_chunk is not None:
                on_chunk(prefix, chunk)
            rows = np.flatnonzero(account % n_chunks == chunk)
            if len(rows) == 0:
                continue
            values = account_history(
                account[rows],
                t[rows],
                cents[rows],
                other[rows],
                len(other_values),
                tuple(names[prefix]),
            )
            for name, column in values.items():
                output[f"{prefix}_{name}"][rows] = column
    return output


def history_features_from_parquet(
    path: Path,
    n_rows: int,
    n_groups: int = 8,
    counterparty_features: list[str] | None = None,
    extra_filter=None,
    on_chunk=None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Même calcul que `history_features`, en lisant un groupe de comptes à la fois.

    Seules les lignes du groupe en cours sont lues (filtre à la lecture du
    Parquet). Le groupe d'un compte est donné par les derniers bits de son
    numéro : un compte n'est jamais coupé. Les résultats sont rangés à la
    position `transaction_id`. Renvoie aussi les lignes effectivement calculées.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    if n_groups < 1 or n_groups & (n_groups - 1):
        raise ValueError("n_groups doit être une puissance de 2.")
    names = _counterparty_names(counterparty_features)
    dataset = ds.dataset(path)
    output = {c: np.zeros(n_rows, dtype=np.float32) for c in history_columns(counterparty_features)}
    seen = {}
    for prefix, account_column in SIDES.items():
        other_column = next(c for c in ACCOUNT_COLUMNS if c != account_column)
        written = np.zeros(n_rows, dtype=np.uint8)
        for chunk in range(n_groups):
            if on_chunk is not None:
                on_chunk(prefix, chunk)
            condition = pc.equal(pc.bit_wise_and(ds.field(account_column), n_groups - 1), chunk)
            if extra_filter is not None:
                condition = condition & extra_filter
            table = dataset.to_table(
                columns=[TRANSACTION_ID, TIMESTAMP, account_column, other_column, "Amount"],
                filter=condition,
            )
            if table.num_rows:
                rows = table[TRANSACTION_ID].to_numpy()
                t = table[TIMESTAMP].to_numpy().astype("datetime64[s]").astype(np.int64)
                account = pd.factorize(table[account_column].to_numpy())[0].astype(np.int32)
                other, other_values = pd.factorize(table[other_column].to_numpy())
                cents = _to_cents(table["Amount"].to_numpy())
                del table
                values = account_history(
                    account,
                    t - t.min(),
                    cents,
                    other.astype(np.int32),
                    len(other_values),
                    tuple(names[prefix]),
                )
                for name, column in values.items():
                    output[f"{prefix}_{name}"][rows] = column
                written[rows] += 1
                del rows, t, account, other, other_values, cents, values
            else:
                del table
            # Libère la mémoire du morceau avant de lire le suivant
            gc.collect()
            pa.default_memory_pool().release_unused()
        if written.max() > 1:
            raise RuntimeError("Une ligne a été calculée deux fois.")
        seen[prefix] = written == 1
    if not np.array_equal(seen["em"], seen["re"]):
        raise RuntimeError("Les deux côtés n'ont pas calculé les mêmes lignes.")
    return output, seen["em"]


def fit_frequency_maps(df: pd.DataFrame, columns: list[str] = FREQUENCY_COLUMNS) -> dict:
    """Part de chaque valeur dans `df` (les lignes d'apprentissage, sans étiquettes)."""
    return {
        column: {
            str(value): float(share)
            for value, share in df[column].value_counts(normalize=True, sort=False).items()
            if share > 0
        }
        for column in columns
    }


def apply_frequency_maps(df: pd.DataFrame, maps: dict) -> dict[str, np.ndarray]:
    """Applique les fréquences apprises ; une valeur inconnue reçoit 0."""
    output = {}
    for column, shares in maps.items():
        values = df[column].astype("category")
        table = np.array([shares.get(str(c), 0.0) for c in values.cat.categories], dtype=np.float32)
        output[frequency_feature(column)] = table[values.cat.codes.to_numpy()]
    return output
