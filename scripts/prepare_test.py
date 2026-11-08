"""Étape 7a : préparer et vérifier les modèles, sans rien lire du test.

1. Les fichiers de configuration et les matrices ont les empreintes de la
   section 29 de docs/CHOIX.md.
2. Chaque modèle est appris dans son propre processus (pic de mémoire plus
   bas), avec son réglage et sa graine figés : le résultat ne dépend pas de
   l'ordre. Il note la validation, et l'empreinte de ces scores doit être celle
   de la section 29. Au premier écart, arrêt.
3. Les modèles sont enregistrés (data/processed/modeles_test/) avec leur
   empreinte et le pic de mémoire de leur processus, dans
   reports/mesures/test_preparation.json.

Usage : python scripts/prepare_test.py
(un modèle seul : python scripts/prepare_test.py --modele "LOF, réglé")
"""

import json
import subprocess
import sys
import time

import joblib
import numpy as np
import pyarrow.parquet as pq

from aml_xai.config import MEASURES_DIR, PARQUET_PATH, PROCESSED_DIR, ROOT, TARGET, TIMESTAMP
from aml_xai.frozen import build_and_fit, check_files, load_frozen, score, sha256_file, sha256_npy
from aml_xai.matrix import load_matrix
from aml_xai.measure import available_ram_mb, measure, peak_memory_mb
from aml_xai.split import model_periods

MODELS_TEST_DIR = PROCESSED_DIR / "modeles_test"
MIN_FREE_RAM_MB = 1500


def slug(name: str) -> str:
    return (name.lower().replace(", ", "_").replace(" ", "_").replace("é", "e").replace("è", "e"))


def prepare_one(name: str) -> dict:
    """Un modèle, dans ce processus : apprentissage, contrôle des scores de validation, enregistrement."""
    spec = load_frozen()["modeles"][name]
    start = time.perf_counter()
    periods = model_periods()
    labels = pq.read_table(PARQUET_PATH, columns=[TARGET], filters=[
        (TIMESTAMP, ">=", periods["debut"]), (TIMESTAMP, "<", periods["fin_entrainement"])
    ]).column(TARGET).to_numpy().astype(np.int8)
    train = load_matrix(PROCESSED_DIR / "matrice_entrainement.npy")
    if len(labels) != len(train):
        raise RuntimeError("Les étiquettes ne correspondent pas à la matrice d'entraînement.")
    model = build_and_fit(spec, train, labels)
    del train, labels
    scores = score(spec, model, load_matrix(PROCESSED_DIR / "matrice_validation.npy"))
    if sha256_npy(scores) != spec["empreinte_validation"]:
        raise RuntimeError(f"{name} ne redonne pas ses scores de validation : arrêt, test non ouvert.")
    MODELS_TEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_TEST_DIR / f"{slug(name)}.joblib"
    joblib.dump(model, path)
    return {"scores_validation_identiques": True, "fichier": path.name, "empreinte_modele": sha256_file(path),
            "duree_s": round(time.perf_counter() - start, 1), "pic_memoire_mo": round(peak_memory_mb())}


def main() -> None:
    free = available_ram_mb()
    if free < MIN_FREE_RAM_MB:
        sys.exit(f"Arrêt : seulement {free:.0f} Mo disponibles, {MIN_FREE_RAM_MB} requis.")
    result = {}
    with measure(result):
        frozen = load_frozen()
        result["configuration"] = check_files(frozen["configuration"])
        result["matrices"] = check_files(frozen["matrices"], PROCESSED_DIR)
        result["modeles"] = {}
        for name in frozen["modeles"]:
            free = available_ram_mb()
            if free < MIN_FREE_RAM_MB:
                sys.exit(f"Arrêt avant {name} : {free:.0f} Mo disponibles. Relancer la même commande.")
            child = subprocess.run([sys.executable, __file__, "--modele", name], cwd=ROOT,
                                   capture_output=True, text=True, encoding="utf-8")
            if child.returncode != 0:
                raise RuntimeError(f"{name} : échec du processus.\n{child.stderr[-2000:]}")
            entry = json.loads(child.stdout.strip().splitlines()[-1])
            result["modeles"][name] = entry
            print(f"{name} : scores de validation identiques ({entry['duree_s']:.0f} s, "
                  f"pic {entry['pic_memoire_mo']} Mo)", flush=True)
    result["pic_memoire_max_modele_mo"] = max(m["pic_memoire_mo"] for m in result["modeles"].values())
    (MEASURES_DIR / "test_preparation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(f"Préparation terminée : 8 modèles vérifiés ({result['duree_s']} s ; pic du processus principal "
          f"{result['pic_memoire_mo']} Mo, pic maximal d'un modèle {result['pic_memoire_max_modele_mo']} Mo)")


if __name__ == "__main__":
    if "--modele" in sys.argv:
        print(json.dumps(prepare_one(sys.argv[sys.argv.index("--modele") + 1]), ensure_ascii=False))
    else:
        main()
