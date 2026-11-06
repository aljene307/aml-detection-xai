"""Calculs de l'exploration, partagés par le notebook et l'application."""

import numpy as np
import pandas as pd

from aml_xai.config import TARGET, TIMESTAMP, TYPOLOGY

DAY_NAMES = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def wilson_interval(cases, total, z: float = 1.96):
    """Intervalle de confiance de Wilson (95 % par défaut) d'une proportion.

    Reste fiable quand la proportion est très faible, contrairement à la
    formule usuelle p ± z·√(p(1-p)/n).
    """
    p = cases / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return center - half, center + half


def rate_by(df: pd.DataFrame, key: str | pd.Series) -> pd.DataFrame:
    """Transactions, cas de blanchiment, taux (%) et son intervalle pour chaque valeur de `key`."""
    grouped = df.groupby(key, observed=True)[TARGET]
    table = grouped.agg(transactions="size", blanchiment="sum")
    table["taux_pct"] = 100 * table["blanchiment"] / table["transactions"]
    low, high = wilson_interval(table["blanchiment"], table["transactions"])
    table["ic_bas_pct"] = 100 * low
    table["ic_haut_pct"] = 100 * high
    return table


def rate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return rate_by(df, df[TIMESTAMP].dt.hour.rename("heure"))


def rate_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    table = rate_by(df, df[TIMESTAMP].dt.day_of_week.rename("jour"))
    table.index = [DAY_NAMES[i] for i in table.index]
    return table


def rate_by_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
    return rate_by(df, df[TIMESTAMP].dt.day.rename("jour_du_mois"))


def rate_by_week(df: pd.DataFrame) -> pd.DataFrame:
    """Taux par semaine complète : les semaines coupées au début et à la fin sont retirées."""
    weeks = df[TIMESTAMP].dt.to_period("W").rename("semaine")
    table = rate_by(df, weeks).iloc[1:-1]
    table.index = table.index.start_time
    return table


def currency_differs(df: pd.DataFrame) -> pd.Series:
    """Vrai si la devise envoyée diffère de la devise reçue."""
    # Comparaison des codes de catégorie : bien moins de mémoire que du texte
    sent, received = shared_categories(df["Payment_currency"], df["Received_currency"])
    differs = sent.cat.codes.to_numpy() != received.cat.codes.to_numpy()
    return pd.Series(differs, index=df.index, name="devises_differentes")


def shared_categories(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Les deux colonnes avec la même liste de catégories, dans le même ordre.

    Un même code désigne alors la même valeur dans les deux colonnes, même si
    leurs listes d'origine diffèrent (contenu ou ordre).
    """
    a, b = a.astype("category"), b.astype("category")
    categories = pd.Index(sorted(set(a.cat.categories) | set(b.cat.categories)))
    return a.cat.set_categories(categories), b.cat.set_categories(categories)


def location_differs(df: pd.DataFrame) -> pd.Series:
    """Vrai si les banques émettrice et destinataire sont dans deux pays différents."""
    sender = df["Sender_bank_location"].astype(str)
    receiver = df["Receiver_bank_location"].astype(str)
    return (sender != receiver).rename("pays_differents")


def amount_shares(df: pd.DataFrame, bins_per_decade: int = 5) -> pd.DataFrame:
    """Part (%) des transactions de chaque classe par tranche de montant (échelle log)."""
    amounts = df["Amount"].to_numpy()
    low = np.floor(np.log10(amounts.min()))
    high = np.ceil(np.log10(amounts.max()))
    edges = np.logspace(low, high, int(high - low) * bins_per_decade + 1)
    labels = df[TARGET].to_numpy() == 1
    normal, _ = np.histogram(amounts[~labels], bins=edges)
    laundering, _ = np.histogram(amounts[labels], bins=edges)
    return pd.DataFrame(
        {
            "borne_basse": edges[:-1],
            "borne_haute": edges[1:],
            "normal_pct": 100 * normal / max(normal.sum(), 1),
            "blanchiment_pct": 100 * laundering / max(laundering.sum(), 1),
        }
    )


def typology_table(df: pd.DataFrame) -> pd.DataFrame:
    """Effectifs et montants de chaque typologie, du plus fréquent au plus rare."""
    table = df.groupby(TYPOLOGY, observed=True).agg(
        transactions=(TARGET, "size"),
        blanchiment=(TARGET, "max"),
        montant_median=("Amount", "median"),
    )
    return table.sort_values("transactions", ascending=False)
