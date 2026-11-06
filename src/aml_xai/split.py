"""Séparation chronologique 60 / 20 / 20 et effectifs par période."""

import json

import numpy as np
import pandas as pd

from aml_xai.config import (
    ACCOUNT_COLUMNS,
    MEASURES_DIR,
    PARQUET_PATH,
    TARGET,
    TIMESTAMP,
    TRAIN_FRAC,
    TYPOLOGY,
    VALID_FRAC,
)
from aml_xai.data import train_cutoff

PERIODS = ["entrainement", "validation", "test"]


def split_cutoffs(timestamps: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Début de la validation et début du test."""
    return (
        train_cutoff(timestamps, TRAIN_FRAC),
        train_cutoff(timestamps, TRAIN_FRAC + VALID_FRAC),
    )


def assign_periods(timestamps: pd.Series, cutoffs: tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
    """Numéro de période de chaque ligne ; une même seconde n'est jamais coupée."""
    values = timestamps.to_numpy()
    return (values >= cutoffs[0].to_datetime64()).astype(np.int8) + (
        values >= cutoffs[1].to_datetime64()
    ).astype(np.int8)


def period_report(df: pd.DataFrame) -> dict:
    """Effectifs par période : lignes, cas, comptes et cas par typologie."""
    cutoffs = split_cutoffs(df[TIMESTAMP])
    periods = assign_periods(df[TIMESTAMP], cutoffs)
    laundering = df[TARGET].to_numpy() == 1
    sender, receiver = (df[c].to_numpy() for c in ACCOUNT_COLUMNS)
    seen_sender = np.unique(sender[periods == 0])
    seen_receiver = np.unique(receiver[periods == 0])
    report = {
        "debut_validation": str(cutoffs[0]),
        "debut_test": str(cutoffs[1]),
        "periodes": {},
    }
    for number, name in enumerate(PERIODS):
        mask = periods == number
        report["periodes"][name] = {
            "lignes": int(mask.sum()),
            "cas": int(laundering[mask].sum()),
            "taux_pct": round(100 * float(laundering[mask].mean()), 4),
            "debut": str(df.loc[mask, TIMESTAMP].min()),
            "fin": str(df.loc[mask, TIMESTAMP].max()),
            "comptes_emetteurs": int(len(np.unique(sender[mask]))),
            "comptes_destinataires": int(len(np.unique(receiver[mask]))),
            "lignes_emetteur_connu_entrainement_pct": round(
                100 * float(np.isin(sender[mask], seen_sender).mean()), 2
            ),
            "lignes_destinataire_connu_entrainement_pct": round(
                100 * float(np.isin(receiver[mask], seen_receiver).mean()), 2
            ),
        }
    suspects = df.loc[laundering, TYPOLOGY].astype(str).to_numpy()
    table = pd.crosstab(suspects, periods[laundering]).reindex(
        columns=range(len(PERIODS)), fill_value=0
    )
    table.columns = PERIODS
    table = table.sort_values("test")
    report["cas_par_typologie"] = {
        str(typology): {period: int(row[period]) for period in PERIODS}
        for typology, row in table.iterrows()
    }
    return report


def model_periods() -> dict[str, pd.Timestamp]:
    """Bornes des lignes des modèles : apprentissage à partir du jour retenu
    (docs/CHOIX.md, section 20), fin de l'entraînement, fin de la validation."""
    startup = json.loads((MEASURES_DIR / "demarrage.json").read_text(encoding="utf-8"))
    timestamps = pd.read_parquet(PARQUET_PATH, columns=[TIMESTAMP])[TIMESTAMP]
    train_end, valid_end = split_cutoffs(timestamps)
    fit_start = timestamps.min() + pd.Timedelta(days=startup["decision"]["exclure_avant_jour"])
    return {"debut": fit_start, "fin_entrainement": train_end, "fin_validation": valid_end}
