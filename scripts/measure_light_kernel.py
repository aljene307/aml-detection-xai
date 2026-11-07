"""Version allégée de KernelSHAP pour expliquer LOF en direct (docs/CHOIX.md, section 32).

Sur les 100 alertes de LOF sans étiquette de la section 28 (validation), compare
les codes motifs de chaque candidate à ceux de la version complète, et mesure la
durée de chaque explication, une alerte à la fois comme dans l'API. Le test
n'est pas lu.

Usage : python scripts/measure_light_kernel.py
"""

import json
import time

import joblib
import numpy as np

from aml_xai.config import MEASURES_DIR, PROCESSED_DIR, SEED
from aml_xai.explain import (
    KERNEL_BACKGROUND_CENTERS,
    KERNEL_SAMPLES,
    group_matrix,
    kernel_background,
    kernel_contributions,
    load_codes,
    lof_score,
    reason_codes,
)
from aml_xai.frozen import load_frozen
from aml_xai.matrix import load_matrix
from aml_xai.measure import measure

ALERTS = 100
CANDIDATES = {"L1": {"references": 10, "tirages": 200}, "L2": {"references": 5, "tirages": 100}}
REQUIRED_AGREEMENT = 0.90
MAX_MEDIAN_S = 0.5


def explain_each(score, x, background, samples) -> tuple[list[np.ndarray], list[float]]:
    """Une alerte à la fois, comme dans l'API : contributions et durée de chacune."""
    values, durations = [], []
    for row in x:
        start = time.perf_counter()
        v, _ = kernel_contributions(score, row[None, :], background, samples=samples)
        durations.append(time.perf_counter() - start)
        values.append(v[0])
    return values, durations


def main() -> None:
    result = {"regle": {"accord_minimal": REQUIRED_AGREEMENT, "duree_mediane_max_s": MAX_MEDIAN_S}}
    with measure(result):
        frozen = load_frozen()["modeles"]["LOF, sans étiquette"]
        lof = joblib.load(PROCESSED_DIR / "modeles_test" / "lof_sans_etiquette.joblib")
        score = lof_score(lof)
        saved = np.load(PROCESSED_DIR / frozen["scores_validation"])
        alerts = np.flatnonzero(saved >= frozen["seuil"])
        chosen = np.sort(np.random.default_rng(SEED).choice(alerts, ALERTS, replace=False))
        train = load_matrix(PROCESSED_DIR / "matrice_entrainement.npy")
        valid = np.asarray(load_matrix(PROCESSED_DIR / "matrice_validation.npy")[chosen], dtype=np.float32)
        codes = load_codes()
        names, gmatrix = list(codes), group_matrix(codes)

        full_values, full_durations = explain_each(score, valid, kernel_background(train, SEED), KERNEL_SAMPLES)
        full_codes = [reason_codes(v @ gmatrix, names) for v in full_values]
        result["complete"] = {"references": KERNEL_BACKGROUND_CENTERS, "tirages": KERNEL_SAMPLES,
                              "duree_mediane_s": round(float(np.median(full_durations)), 3)}
        result["candidates"] = {}
        for name, c in CANDIDATES.items():
            values, durations = explain_each(score, valid, kernel_background(train, SEED, c["references"]),
                                             c["tirages"])
            light = [reason_codes(v @ gmatrix, names) for v in values]
            same_set = float(np.mean([set(a) == set(b) for a, b in zip(full_codes, light)]))
            same_order = float(np.mean([a == b for a, b in zip(full_codes, light)]))
            first = float(np.mean([a[:1] == b[:1] for a, b in zip(full_codes, light)]))
            median = float(np.median(durations))
            result["candidates"][name] = {
                **c, "memes_codes": round(same_set, 3), "memes_codes_meme_ordre": round(same_order, 3),
                "meme_premier_code": round(first, 3), "duree_mediane_s": round(median, 3),
                "duree_max_s": round(float(np.max(durations)), 3),
                "acceptee": same_set >= REQUIRED_AGREEMENT and median <= MAX_MEDIAN_S,
            }
            print(name, result["candidates"][name], flush=True)
        accepted = [n for n, c in result["candidates"].items() if c["acceptee"]]
        result["retenue"] = accepted[-1] if accepted else None
    (MEASURES_DIR / "kernel_allege.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(f"Retenue : {result['retenue']} ({result['duree_s']} s, pic {result['pic_memoire_mo']} Mo)")


if __name__ == "__main__":
    main()
