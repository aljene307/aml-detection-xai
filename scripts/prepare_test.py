"""Étape 7a : préparer et vérifier les modèles, sans rien lire du test.

1. Les fichiers de configuration et les matrices ont les empreintes de la
   section 29 de docs/CHOIX.md.
2. Chaque modèle est appris avec son réglage figé, sur l'entraînement retenu,
   puis note la validation : l'empreinte de ces scores doit être celle de la
   section 29. Au premier écart, arrêt.
3. Les modèles sont enregistrés (data/processed/modeles_test/) avec leur
   empreinte, dans reports/mesures/test_preparation.json.

Usage : python scripts/prepare_test.py
"""

import gc
import json
import sys
import time

import joblib
import numpy as np
import pyarrow.parquet as pq

from aml_xai.config import MEASURES_DIR, PARQUET_PATH, PROCESSED_DIR, TARGET, TIMESTAMP
from aml_xai.frozen import build_and_fit, check_files, load_frozen, score, sha256_file, sha256_npy
from aml_xai.matrix import load_matrix
from aml_xai.measure import available_ram_mb, measure
from aml_xai.split import model_periods

MODELS_TEST_DIR = PROCESSED_DIR / "modeles_test"
MIN_FREE_RAM_MB = 1500


def slug(name: str) -> str:
    return (name.lower().replace(", ", "_").replace(" ", "_").replace("é", "e").replace("è", "e"))


def main() -> None:
    free = available_ram_mb()
    if free < MIN_FREE_RAM_MB:
        sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles, {MIN_FREE_RAM_MB} requis.")
    result = {}
    with measure(result):
        frozen = load_frozen()
        result["configuration"] = check_files(frozen["configuration"])
        result["matrices"] = check_files(frozen["matrices"], PROCESSED_DIR)
        periods = model_periods()
        labels = pq.read_table(PARQUET_PATH, columns=[TARGET], filters=[
            (TIMESTAMP, ">=", periods["debut"]), (TIMESTAMP, "<", periods["fin_entrainement"])
        ]).column(TARGET).to_numpy().astype(np.int8)
        train = load_matrix(PROCESSED_DIR / "matrice_entrainement.npy")
        valid = load_matrix(PROCESSED_DIR / "matrice_validation.npy")
        if len(labels) != len(train):
            raise RuntimeError("Les étiquettes ne correspondent pas à la matrice d'entraînement.")
        MODELS_TEST_DIR.mkdir(parents=True, exist_ok=True)
        result["modeles"] = {}
        for name, spec in frozen["modeles"].items():
            start = time.perf_counter()
            model = build_and_fit(spec, train, labels)
            scores = score(spec, model, valid)
            same = sha256_npy(scores) == spec["empreinte_validation"]
            print(f"{name} : scores de validation {'identiques' if same else 'DIFFÉRENTS'} "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
            if not same:
                raise RuntimeError(f"{name} ne redonne pas ses scores de validation : arrêt, test non ouvert.")
            path = MODELS_TEST_DIR / f"{slug(name)}.joblib"
            joblib.dump(model, path)
            result["modeles"][name] = {"scores_validation_identiques": True, "fichier": path.name,
                                       "empreinte_modele": sha256_file(path),
                                       "duree_s": round(time.perf_counter() - start, 1)}
            del model, scores
            gc.collect()
    (MEASURES_DIR / "test_preparation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(f"Préparation terminée : 8 modèles vérifiés ({result['duree_s']} s, pic {result['pic_memoire_mo']} Mo)")


if __name__ == "__main__":
    main()
