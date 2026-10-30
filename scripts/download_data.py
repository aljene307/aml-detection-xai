"""Télécharge SAML-D puis le convertit en Parquet, en mesurant chaque phase.

Usage : python scripts/download_data.py [--force]
"""

import argparse
import json

import pyarrow.parquet as pq

from aml_xai.config import CSV_PATH, MEASURES_DIR, PARQUET_PATH
from aml_xai.data import convert_to_parquet, download_raw
from aml_xai.measure import available_ram_mb, measure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retélécharger et reconvertir")
    args = parser.parse_args()

    results = {"ram_disponible_au_depart_mo": round(available_ram_mb())}

    download = {}
    with measure(download):
        download["methode"] = download_raw(force=args.force)
    download["taille_csv_octets"] = CSV_PATH.stat().st_size
    results["telechargement"] = download
    print("Téléchargement :", download)

    conversion = {}
    if PARQUET_PATH.exists() and not args.force:
        conversion["statut"] = "déjà présent"
    else:
        # Le pic de mémoire est celui du processus depuis son démarrage :
        # le téléchargement en consomme peu, ce pic reflète donc la conversion.
        with measure(conversion):
            convert_to_parquet()
    meta = pq.ParquetFile(PARQUET_PATH).metadata
    conversion["taille_parquet_octets"] = PARQUET_PATH.stat().st_size
    conversion["lignes"] = meta.num_rows
    conversion["groupes_de_lignes"] = meta.num_row_groups
    results["conversion_parquet"] = conversion
    print("Conversion :", conversion)

    MEASURES_DIR.mkdir(parents=True, exist_ok=True)
    out = MEASURES_DIR / "donnees.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mesures écrites dans {out.relative_to(MEASURES_DIR.parents[1])}")


if __name__ == "__main__":
    main()
