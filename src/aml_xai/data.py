"""Téléchargement, conversion en Parquet, chargement et contrôle de qualité."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from aml_xai.config import (
    CATEGORICAL_COLUMNS,
    CSV_PATH,
    EXPECTED_CSV_BYTES,
    KAGGLE_DOWNLOAD_URL,
    KAGGLE_HANDLE,
    PARQUET_PATH,
    RAW_DIR,
    TARGET,
    TIMESTAMP,
    TRAIN_FRAC,
    TRANSACTION_ID,
    TYPOLOGY,
)


def _csv_is_complete(csv_path: Path) -> bool:
    return csv_path.exists() and csv_path.stat().st_size == EXPECTED_CSV_BYTES


def _download_direct(raw_dir: Path) -> None:
    """Téléchargement HTTP anonyme de l'archive, puis extraction."""
    import requests

    archive = raw_dir / "saml_d.zip"
    with requests.get(KAGGLE_DOWNLOAD_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in response.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    with zipfile.ZipFile(archive) as zf:
        zf.extract(CSV_PATH.name, raw_dir)
    archive.unlink()


def download_raw(force: bool = False) -> str:
    """Télécharge SAML-D.csv dans data/raw/ et renvoie la méthode utilisée.

    Le jeu est public : aucun jeton Kaggle n'est nécessaire. kagglehub est
    essayé d'abord ; en cas d'échec, on télécharge l'archive directement.
    """
    if _csv_is_complete(CSV_PATH) and not force:
        return "déjà présent"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub

        kagglehub.dataset_download(KAGGLE_HANDLE, output_dir=str(RAW_DIR), force_download=force)
        method = "kagglehub"
    except Exception as exc:  # noqa: BLE001 - toute erreur bascule sur le secours
        print(f"kagglehub a échoué ({type(exc).__name__}: {exc}) : téléchargement direct.")
        _download_direct(RAW_DIR)
        method = "HTTP direct"
    if not _csv_is_complete(CSV_PATH):
        raise RuntimeError(f"{CSV_PATH.name} absent ou de taille inattendue après téléchargement.")
    return method


def convert_to_parquet(
    csv_path: Path = CSV_PATH,
    parquet_path: Path = PARQUET_PATH,
    block_size: int = 64 << 20,
) -> Path:
    """Convertit le CSV en Parquet, morceau par morceau (mémoire limitée).

    - Date et Time sont fusionnées en un horodatage `timestamp` ;
    - les colonnes texte répétitives deviennent des catégories ;
    - `transaction_id` garde l'ordre d'origine des lignes.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    column_types = {
        "Time": pa.string(),
        "Date": pa.string(),
        "Sender_account": pa.int64(),
        "Receiver_account": pa.int64(),
        "Amount": pa.float64(),
        TARGET: pa.int8(),
        **{c: pa.dictionary(pa.int32(), pa.string()) for c in CATEGORICAL_COLUMNS},
    }
    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=block_size),
        convert_options=pacsv.ConvertOptions(column_types=column_types),
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parquet_path.with_suffix(".tmp")
    writer = None
    offset = 0
    try:
        for batch in reader:
            n = batch.num_rows
            text_ts = pc.binary_join_element_wise(batch["Date"], batch["Time"], " ")
            columns = {
                TRANSACTION_ID: pa.array(np.arange(offset, offset + n, dtype=np.int64)),
                TIMESTAMP: pc.cast(text_ts, pa.timestamp("s")),
            }
            for name in batch.schema.names:
                if name not in ("Date", "Time"):
                    columns[name] = batch[name]
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
            writer.write_table(table)
            offset += n
    finally:
        if writer is not None:
            writer.close()
    tmp_path.replace(parquet_path)
    return parquet_path


def load_transactions(columns: list[str] | None = None, path: Path = PARQUET_PATH) -> pd.DataFrame:
    """Charge le Parquet dans pandas (les catégories restent des catégories)."""
    return pd.read_parquet(path, columns=columns)


def train_cutoff(timestamps: pd.Series, train_frac: float = TRAIN_FRAC) -> pd.Timestamp:
    """Horodatage qui sépare les premiers `train_frac` des lignes (ordre du temps)."""
    ordered = np.sort(timestamps.to_numpy())
    return pd.Timestamp(ordered[int(len(ordered) * train_frac)])


def load_training_period(
    columns: list[str] | None = None, path: Path = PARQUET_PATH
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Charge les lignes strictement antérieures à la coupure des 60 %.

    Le filtre est appliqué à la lecture pour limiter la mémoire. Une même
    seconde n'est jamais coupée en deux : la coupure porte sur l'horodatage.
    """
    cutoff = train_cutoff(load_transactions([TIMESTAMP], path)[TIMESTAMP])
    df = pd.read_parquet(path, columns=columns, filters=[(TIMESTAMP, "<", cutoff)])
    return df, cutoff


def quality_report(df: pd.DataFrame) -> dict:
    """Contrôles de qualité du jeu complet (chiffres descriptifs globaux)."""
    typologies = df.groupby(TYPOLOGY, observed=True)[TARGET].agg(["min", "max", "size"])
    normal_names = typologies.index.str.startswith("Normal_")
    cutoff = train_cutoff(df[TIMESTAMP])
    return {
        "lignes": int(len(df)),
        "colonnes": list(df.columns),
        "valeurs_manquantes": {c: int(v) for c, v in df.isna().sum().items() if v},
        "doublons_hors_identifiant": int(df.drop(columns=TRANSACTION_ID).duplicated().sum()),
        "periode": [str(df[TIMESTAMP].min()), str(df[TIMESTAMP].max())],
        "ordre_chronologique_respecte": bool(df[TIMESTAMP].is_monotonic_increasing),
        "fin_periode_entrainement_60pct": str(cutoff),
        "nb_blanchiment": int(df[TARGET].sum()),
        "taux_blanchiment_pct": round(100 * float(df[TARGET].mean()), 4),
        "nb_typologies": int(len(typologies)),
        "nb_typologies_normales": int(normal_names.sum()),
        "nb_typologies_suspectes": int((~normal_names).sum()),
        # Fuite : une typologie « Normal_… » doit toujours valoir 0, les autres 1
        "typologie_determine_etiquette": bool(
            (typologies.loc[normal_names, "max"] == 0).all()
            and (typologies.loc[~normal_names, "min"] == 1).all()
        ),
        "nb_comptes_emetteurs": int(df["Sender_account"].nunique()),
        "nb_comptes_destinataires": int(df["Receiver_account"].nunique()),
        "montant_min_max": [float(df["Amount"].min()), float(df["Amount"].max())],
        "transactions_par_typologie": {
            str(k): int(v) for k, v in typologies["size"].sort_values(ascending=False).items()
        },
    }
