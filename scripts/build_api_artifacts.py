"""Fichiers lus par l'API au démarrage (docs/CHOIX.md, section 32).

- models/lof_sans_etiquette.joblib : copie du modèle vérifié en 7a ;
- models/lightgbm_regle.txt : LightGBM réglé au format texte, qui doit redonner
  exactement les scores de validation figés ;
- models/quantiles_validation.json : distribution des scores de validation des
  deux modèles, pour le rang ;
- data/demo/exemples.json : 30 transactions du test (règle fixe, graine 42),
  entrées prêtes à envoyer, explications de LOF calculées à l'avance ;
- models/manifeste.json : empreinte SHA-256 de chaque fichier lu par l'API.

Le test est déjà ouvert et figé (sections 29 à 31) : ces fichiers ne changent
aucun résultat.

Usage : python scripts/build_api_artifacts.py
"""

import json
import shutil

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from aml_xai.api_service import (
    DEMO_PATH,
    HISTORY_FEATURES,
    LGBM_PATH,
    LOF_PATH,
    MANIFEST_PATH,
    MANIFEST_FILES,
    QUANTILES_PATH,
    manifest_hash,
    RAW_FIELDS,
)
from aml_xai.config import (
    FEATURES_PATH,
    FREQUENCY_MAPS_PATH,
    MODELS_DIR,
    PARQUET_PATH,
    PROCESSED_DIR,
    ROOT,
    SEED,
    TARGET,
    TIMESTAMP,
    TRANSACTION_ID,
    TYPOLOGY,
)
from aml_xai.explain import group_matrix, kernel_background, kernel_contributions, load_codes, lof_score, phrase
from aml_xai.explain import reason_codes
from aml_xai.frozen import load_frozen, sha256_npy
from aml_xai.matrix import build_matrix, load_matrix, period_columns, period_size
from aml_xai.split import model_periods
from aml_xai.unsupervised import anomaly_scores

QUANTILE_POINTS = 2001
EXAMPLES = {"vrai_cas_alerte": 10, "fausse_alerte": 10, "cas_manque": 5, "normale": 5}


def tail_quantiles(scores: np.ndarray) -> dict:
    """Score atteint par les parts 1/n à 1 des plus suspectes (échelle logarithmique)."""
    ordered = np.sort(scores)[::-1]
    shares = np.unique(np.r_[np.geomspace(1 / len(ordered), 1, QUANTILE_POINTS), 1.0])
    positions = np.minimum(np.ceil(shares * len(ordered)).astype(int) - 1, len(ordered) - 1)
    return {"parts": shares.tolist(), "scores": ordered[positions].tolist()}


def choose_examples(labels, alerted, typologies) -> list[tuple[int, str]]:
    """30 exemples, règle validée (8.6) : positions dans le test et catégorie."""
    rng = np.random.default_rng(SEED)
    true_alerts = pd.DataFrame({"pos": np.flatnonzero(alerted & (labels == 1))})
    true_alerts["typologie"] = typologies[true_alerts["pos"]]
    per_type = true_alerts.groupby("typologie")["pos"].apply(lambda s: int(rng.choice(s.to_numpy())))
    chosen_types = rng.choice(per_type.index.to_numpy(), size=EXAMPLES["vrai_cas_alerte"], replace=False)
    out = [(int(per_type[t]), "vrai_cas_alerte") for t in sorted(chosen_types)]
    for name, mask in [("fausse_alerte", alerted & (labels == 0)), ("cas_manque", ~alerted & (labels == 1)),
                       ("normale", ~alerted & (labels == 0))]:
        picked = rng.choice(np.flatnonzero(mask), size=EXAMPLES[name], replace=False)
        out += [(int(p), name) for p in np.sort(picked)]
    return out


def main() -> None:
    frozen = load_frozen()["modeles"]
    periods = model_periods()
    MODELS_DIR.mkdir(exist_ok=True)

    # Modèles
    shutil.copyfile(PROCESSED_DIR / "modeles_test" / "lof_sans_etiquette.joblib", LOF_PATH)
    lgbm = joblib.load(PROCESSED_DIR / "modeles_test" / "lightgbm_regle.joblib")
    lgbm.booster_.save_model(str(LGBM_PATH))
    booster = lgb.Booster(model_file=str(LGBM_PATH))
    valid = np.asarray(load_matrix(PROCESSED_DIR / "matrice_validation.npy"), dtype=np.float32)
    lgbm_valid = booster.predict(valid, raw_score=True)
    if sha256_npy(lgbm_valid) != frozen["LightGBM, réglé"]["empreinte_validation"]:
        raise RuntimeError("Le modèle LightGBM au format texte ne redonne pas les scores figés.")

    # Rang : distribution des scores de validation (figés)
    lof_valid = np.load(PROCESSED_DIR / frozen["LOF, sans étiquette"]["scores_validation"])
    QUANTILES_PATH.write_text(json.dumps({"lof": tail_quantiles(lof_valid),
                                          "lightgbm": tail_quantiles(lgbm_valid)}), encoding="utf-8")
    del valid

    # Exemples du test
    last = pd.read_parquet(PARQUET_PATH, columns=[TIMESTAMP], filters=[
        (TIMESTAMP, ">=", periods["fin_validation"])])[TIMESTAMP].max()
    start, end = periods["fin_validation"], last + pd.Timedelta(seconds=1)
    maps = json.loads(FREQUENCY_MAPS_PATH.read_text(encoding="utf-8"))
    params = json.loads((MODELS_DIR / "transformation.json").read_text(encoding="utf-8"))
    matrix, _ = build_matrix(period_columns(start, end, maps), period_size(start, end), params)
    period = [(TIMESTAMP, ">=", start), (TIMESTAMP, "<", end)]
    raw = pd.read_parquet(PARQUET_PATH, columns=[TRANSACTION_ID, *RAW_FIELDS.values(), TARGET, TYPOLOGY],
                          filters=period)
    history = pd.read_parquet(FEATURES_PATH, columns=[TRANSACTION_ID, *HISTORY_FEATURES], filters=period)
    if not np.array_equal(raw[TRANSACTION_ID].to_numpy(), history[TRANSACTION_ID].to_numpy()):
        raise RuntimeError("Les deux fichiers ne donnent pas les mêmes lignes.")
    lgbm_test = booster.predict(matrix, raw_score=True)
    labels = raw[TARGET].to_numpy()
    typologies = raw[TYPOLOGY].astype(str).to_numpy()
    examples = choose_examples(labels, lgbm_test >= frozen["LightGBM, réglé"]["seuil"], typologies)
    positions = np.array([p for p, _ in examples])

    lof = joblib.load(LOF_PATH)
    lof_scores = anomaly_scores(lof, matrix[positions])
    codes = load_codes()
    names, gmatrix = list(codes), group_matrix(codes)
    train = load_matrix(PROCESSED_DIR / "matrice_entrainement.npy")
    values, base = kernel_contributions(lof_score(lof), matrix[positions], kernel_background(train, SEED))
    usual = json.loads((MODELS_DIR / "valeurs_habituelles.json").read_text(encoding="utf-8"))

    demo = []
    for i, (position, category) in enumerate(examples):
        r, h = raw.iloc[position], history.iloc[position]
        entree = {field: (r[column].isoformat() if field == "horodatage" else
                          float(r[column]) if field == "montant" else str(r[column]))
                  for field, column in RAW_FIELDS.items()}
        entree["historique"] = {name: float(h[name]) if "ratio" in name else int(h[name]) for name in HISTORY_FEATURES}
        row = {**{column: r[column] for column in RAW_FIELDS.values()}, **entree["historique"]}
        groups = values[i] @ gmatrix
        demo.append({
            "id": i + 1,
            "categorie": category,
            "transaction_id": int(r[TRANSACTION_ID]),
            "blanchiment": int(r[TARGET]),
            "typologie": typologies[position],
            "entree": entree,
            "scores_attendus": {"lof": float(lof_scores[i]), "lightgbm": float(lgbm_test[position])},
            "explication_lof": {
                "valeur_de_base": base,
                "contributions_groupes": dict(zip(names, groups.round(6).tolist())),
                "codes_motifs": [{"code": c, "libelle": codes[c]["libelle"],
                                  "contribution": round(float(groups[names.index(c)]), 6),
                                  "phrase": phrase(c, row, usual)} for c in reason_codes(groups, names)],
            },
        })
    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_PATH.write_text(json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {name: manifest_hash(name) for name in MANIFEST_FILES}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Fichiers de l'API écrits ; {len(demo)} exemples ; LightGBM au format texte vérifié.")


if __name__ == "__main__":
    main()
