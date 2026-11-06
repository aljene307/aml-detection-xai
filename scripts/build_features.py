"""Calcule les variables sur tout le jeu et enregistre le temps et la mémoire.

Chaque calcul ne lit que ses colonnes. L'historique est calculé par groupes
de comptes, en ne lisant que les lignes du groupe en cours ; les variables de
la transaction sont calculées par lots. Les étiquettes ne sont jamais lues.

Usage : python scripts/build_features.py [--candidats] [--morceaux N] [--sortie FICHIER]
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aml_xai.config import (
    CANDIDATES_PATH,
    FEATURES_PATH,
    MEASURES_DIR,
    PARQUET_PATH,
    TIMESTAMP,
    TRANSACTION_ID,
)
from aml_xai.features import (
    COUNTERPARTY_CANDIDATES,
    TRANSACTION_FEATURES,
    history_features_from_parquet,
    transaction_features,
)
from aml_xai.measure import available_ram_mb, measure, peak_memory_mb

TRANSACTION_COLUMNS = [TRANSACTION_ID, TIMESTAMP, "Amount", "Payment_currency", "Received_currency"]
DEFAULT_GROUPS = 8
MIN_FREE_RAM_MB = 1500
ABORT_FREE_RAM_MB = 500
BATCH_ROWS = 1_000_000


def stop_if_ram_is_low(side: str, chunk: int) -> None:
    free = available_ram_mb()
    print(f"  {side}, morceau {chunk} : {free:.0f} Mo disponibles, pic {peak_memory_mb():.0f} Mo", flush=True)
    if free < ABORT_FREE_RAM_MB:
        sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles.")


def transaction_columns(n_rows: int) -> dict[str, np.ndarray]:
    """Variables de la transaction, lot par lot, rangées à la position `transaction_id`."""
    output = {name: np.zeros(n_rows, dtype=np.float32) for name in TRANSACTION_FEATURES}
    output[TIMESTAMP] = np.zeros(n_rows, dtype="datetime64[s]")
    for batch in pq.ParquetFile(PARQUET_PATH).iter_batches(BATCH_ROWS, columns=TRANSACTION_COLUMNS):
        df = batch.to_pandas()
        rows = df[TRANSACTION_ID].to_numpy()
        output[TIMESTAMP][rows] = df[TIMESTAMP].to_numpy().astype("datetime64[s]")
        for name, values in transaction_features(df).items():
            output[name][rows] = values
        del df, batch
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidats", action="store_true", help="ajoute les contreparties à évaluer")
    parser.add_argument("--morceaux", type=int, default=DEFAULT_GROUPS, help="puissance de 2")
    parser.add_argument("--sortie", type=Path, help="autre fichier de sortie")
    args = parser.parse_args()

    free = available_ram_mb()
    if free < MIN_FREE_RAM_MB:
        sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles, {MIN_FREE_RAM_MB} requis.")
    label = "candidats" if args.candidats else "variables"
    print(f"{free:.0f} Mo disponibles : {label}, {args.morceaux} morceau(x).", flush=True)

    n_rows = pq.ParquetFile(PARQUET_PATH).metadata.num_rows
    result = {"ram_disponible_avant_mo": round(free), "morceaux": args.morceaux, "lignes": n_rows}
    with measure(result):
        start = time.perf_counter()
        history, seen = history_features_from_parquet(
            PARQUET_PATH,
            n_rows,
            args.morceaux,
            counterparty_features=COUNTERPARTY_CANDIDATES if args.candidats else None,
            on_chunk=stop_if_ram_is_low,
        )
        if not seen.all():
            raise RuntimeError("Des lignes n'ont pas été calculées.")
        del seen
        result["duree_historique_s"] = round(time.perf_counter() - start, 2)
        result["pic_apres_historique_mo"] = round(peak_memory_mb())

        start = time.perf_counter()
        gc.collect()
        transaction = transaction_columns(n_rows)
        result["duree_transaction_s"] = round(time.perf_counter() - start, 2)
        result["pic_apres_transaction_mo"] = round(peak_memory_mb())

        columns = {
            TRANSACTION_ID: np.arange(n_rows, dtype=np.int64),
            TIMESTAMP: transaction.pop(TIMESTAMP),
            **transaction,
            **history,
        }
        output = args.sortie or (CANDIDATES_PATH if args.candidats else FEATURES_PATH)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(".tmp")
        table = pa.table({name: pa.array(values) for name, values in columns.items()})
        # Écriture groupe de lignes par groupe de lignes
        with pq.ParquetWriter(tmp, table.schema, compression="zstd") as writer:
            for begin in range(0, n_rows, BATCH_ROWS):
                writer.write_table(table.slice(begin, BATCH_ROWS))
        del table
        tmp.replace(output)
    result["taille_fichier_mo"] = round(output.stat().st_size / 2**20)
    result["nb_colonnes"] = len(columns)
    result["valeurs_non_finies"] = {
        name: int((~np.isfinite(values)).sum())
        for name, values in columns.items()
        if values.dtype == np.float32 and (~np.isfinite(values)).any()
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    path = MEASURES_DIR / "variables.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    saved[f"{label}_{args.morceaux}_morceaux"] = result
    path.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
