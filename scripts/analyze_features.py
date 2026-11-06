"""Mesures sur les variables calculées. Chaque cas tourne séparément.

- morceaux : le calcul par morceaux comparé au calcul complet ;
- contreparties : corrélation de rangs avec les compteurs (entraînement) ;
- demarrage : PSI et décision, règle de docs/CHOIX.md, section 20 ;
- periodes : effectifs par période et par typologie ;
- frequences : apprentissage du codage, valeurs inconnues ensuite ;
- fuite : recalcul sur la seule période d'entraînement, comparé au calcul complet.

Seul le cas « periodes » lit les étiquettes, pour compter les cas.

Usage : python scripts/analyze_features.py --cas demarrage
"""

import argparse
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from aml_xai.config import (
    ACCOUNT_COLUMNS,
    CANDIDATES_PATH,
    FEATURES_PATH,
    FREQUENCY_MAPS_PATH,
    MEASURES_DIR,
    PARQUET_PATH,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEED,
    TARGET,
    TIMESTAMP,
    TYPOLOGY,
)
from aml_xai.features import (
    FREQUENCY_COLUMNS,
    SIDES,
    apply_frequency_maps,
    fit_frequency_maps,
    history_columns,
    history_features_from_parquet,
)
from aml_xai.measure import measure
from aml_xai.split import assign_periods, period_report, split_cutoffs
from aml_xai.startup import STARTUP_SLICES, startup_decision, startup_report, with_missing

CHUNKS_PATH = PROCESSED_DIR / "features_morceaux.parquet"
SPEARMAN_SAMPLE = 1_000_000
DAY_S = 86_400


def cutoffs_of_dataset() -> tuple[pd.Timestamp, pd.Timestamp]:
    return split_cutoffs(pd.read_parquet(PARQUET_PATH, columns=[TIMESTAMP])[TIMESTAMP])


def case_counterparties(r: dict) -> None:
    """Le nombre de contreparties distinctes répète-t-il le compteur ?"""
    columns = [
        f"{side}_{name}"
        for side in SIDES
        for name in ["nb_24h", "nb_distincts_24h", "nb_30j", "nb_distincts_30j"]
    ]
    train_end = cutoffs_of_dataset()[0]
    df = pd.read_parquet(CANDIDATES_PATH, columns=columns, filters=[(TIMESTAMP, "<", train_end)])
    r["lignes_entrainement"] = len(df)
    sample = np.random.default_rng(SEED).choice(len(df), SPEARMAN_SAMPLE, replace=False)
    r["lignes_echantillon_correlation"] = SPEARMAN_SAMPLE
    r["variables"] = {}
    for side in SIDES:
        for counter, distinct in [("nb_24h", "nb_distincts_24h"), ("nb_30j", "nb_distincts_30j")]:
            a = df[f"{side}_{counter}"].to_numpy()
            b = df[f"{side}_{distinct}"].to_numpy()
            r["variables"][f"{side}_{distinct}"] = {
                "spearman_avec_compteur": round(float(spearmanr(a[sample], b[sample]).statistic), 4),
                "lignes_ou_distinct_differe_pct": round(100 * float((a != b).mean()), 2),
                "moyenne_compteur": round(float(a.mean()), 2),
                "moyenne_distincts": round(float(b.mean()), 2),
            }


def case_startup(r: dict) -> None:
    """Effet de démarrage, sur les distributions seules."""
    train_end = cutoffs_of_dataset()[0]
    columns = [c for c in history_columns()]
    df = pd.read_parquet(FEATURES_PATH, columns=[TIMESTAMP, *columns], filters=[(TIMESTAMP, "<", train_end)])
    if {TARGET, TYPOLOGY} & set(df.columns):
        raise RuntimeError("Les étiquettes ne doivent pas entrer dans cette mesure.")
    t = df[TIMESTAMP].to_numpy().astype("datetime64[s]").astype(np.int64)
    days = (t - t.min()) / DAY_S
    r["premier_jour"] = str(df[TIMESTAMP].min())
    no_history = {
        side: df[f"{side}_sans_historique_30j"].to_numpy() == 1 for side in SIDES
    }
    slices = [*STARTUP_SLICES, (30, 60), (60, 90), (90, 120), (120, 150), (150, int(days.max()) + 1)]
    r["sans_historique_30j_pct"] = {
        side: {
            f"{a}-{b}": round(100 * float(flags[(days >= a) & (days < b)].mean()), 2)
            for a, b in slices
        }
        for side, flags in no_history.items()
    }
    features = with_missing({c: df[c].to_numpy() for c in columns})
    # L'indicateur répète l'absence de valeur du rapport : il n'est pas compté deux fois.
    features = {name: v for name, v in features.items() if "sans_historique" not in name}
    r.update(startup_report(days, features))
    r["decision"] = startup_decision(r)


def case_periods(r: dict) -> None:
    columns = [TIMESTAMP, *ACCOUNT_COLUMNS, TARGET, TYPOLOGY]
    r.update(period_report(pd.read_parquet(PARQUET_PATH, columns=columns)))


def case_frequencies(r: dict) -> None:
    """Fréquences apprises sur les lignes d'apprentissage retenues."""
    startup = json.loads((MEASURES_DIR / "demarrage.json").read_text(encoding="utf-8"))
    excluded_days = startup["decision"]["exclure_avant_jour"]
    timestamps = pd.read_parquet(PARQUET_PATH, columns=[TIMESTAMP])[TIMESTAMP]
    cutoffs = split_cutoffs(timestamps)
    fit_start = timestamps.min() + pd.Timedelta(days=excluded_days)
    del timestamps
    train = pd.read_parquet(
        PARQUET_PATH,
        columns=FREQUENCY_COLUMNS,
        filters=[(TIMESTAMP, ">=", fit_start), (TIMESTAMP, "<", cutoffs[0])],
    )
    maps = fit_frequency_maps(train)
    r["lignes_apprentissage"] = len(train)
    r["debut_apprentissage"] = str(fit_start)
    r["nb_valeurs"] = {column: len(shares) for column, shares in maps.items()}
    del train
    FREQUENCY_MAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREQUENCY_MAPS_PATH.write_text(json.dumps(maps, indent=2, ensure_ascii=False), encoding="utf-8")

    df = pd.read_parquet(PARQUET_PATH, columns=[TIMESTAMP, *FREQUENCY_COLUMNS])
    periods = assign_periods(df[TIMESTAMP], cutoffs)
    encoded = apply_frequency_maps(df, maps)
    r["lignes_valeur_inconnue"] = {
        name: {
            period: int((values[periods == number] == 0).sum())
            for number, period in enumerate(["entrainement", "validation", "test"])
        }
        for name, values in encoded.items()
    }


def case_leak(r: dict) -> None:
    """Essai réel : calculer sans le futur doit donner exactement les mêmes valeurs."""
    train_end = cutoffs_of_dataset()[0]
    n_rows = pq.ParquetFile(PARQUET_PATH).metadata.num_rows
    # Le futur n'est même pas lu : seules les lignes d'entraînement entrent dans le calcul
    recomputed, past = history_features_from_parquet(
        PARQUET_PATH,
        n_rows,
        extra_filter=ds.field(TIMESTAMP) < pa.scalar(train_end, pa.timestamp("s")),
    )
    r["lignes_comparees"] = int(past.sum())
    r["colonnes_comparees"] = len(recomputed)
    r["lignes_differentes_par_colonne"] = {}
    for name, values in recomputed.items():
        stored = pd.read_parquet(FEATURES_PATH, columns=[name])[name].to_numpy()
        r["lignes_differentes_par_colonne"][name] = int((values[past] != stored[past]).sum())
    r["toutes_identiques"] = not any(r["lignes_differentes_par_colonne"].values())


def case_chunks(r: dict) -> None:
    """Essai réel : le calcul par morceaux donne-t-il exactement le calcul complet ?"""
    names = pq.ParquetFile(FEATURES_PATH).schema_arrow.names
    if names != pq.ParquetFile(CHUNKS_PATH).schema_arrow.names:
        raise RuntimeError("Les deux fichiers n'ont pas les mêmes colonnes.")
    r["lignes_differentes_par_colonne"] = {}
    for name in names:
        full = pd.read_parquet(FEATURES_PATH, columns=[name])[name].to_numpy()
        chunked = pd.read_parquet(CHUNKS_PATH, columns=[name])[name].to_numpy()
        r["lignes_comparees"] = len(full)
        r["lignes_differentes_par_colonne"][name] = int((full != chunked).sum())
    r["toutes_identiques"] = not any(r["lignes_differentes_par_colonne"].values())


CASES = {
    "morceaux": (case_chunks, MEASURES_DIR / "morceaux_reel.json"),
    "contreparties": (case_counterparties, MEASURES_DIR / "contreparties.json"),
    "demarrage": (case_startup, MEASURES_DIR / "demarrage.json"),
    "periodes": (case_periods, REPORTS_DIR / "effectifs_periodes.json"),
    "frequences": (case_frequencies, MEASURES_DIR / "frequences.json"),
    "fuite": (case_leak, MEASURES_DIR / "fuite_reel.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cas", choices=CASES, required=True)
    args = parser.parse_args()
    run, path = CASES[args.cas]
    result = {}
    with measure(result):
        run(result)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
