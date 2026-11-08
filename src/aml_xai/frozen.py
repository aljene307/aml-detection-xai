"""Choix figés avant l'ouverture du test (docs/CHOIX.md, section 29).

Vérification des empreintes SHA-256, construction des modèles à partir de leur
réglage figé, notation, et fichier de sortie du test (écrit une seule fois).
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from aml_xai.config import ROOT, SEED
from aml_xai.supervised import lightgbm_model, logistic_model, ranking_scores
from aml_xai.unsupervised import SAMPLE_ROWS, anomaly_scores, fit, model_from_settings, sample_rows

FROZEN_PATH = ROOT / "config" / "fige_test.json"
TEST_RESULT = "test.json"
# Isolation Forest avec une part des variables : cœurs utilisés pour apprendre (mémoire)
FOREST_FIT_JOBS = 2


def load_frozen(path: Path = FROZEN_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path, normalize_eol: bool = False) -> str:
    """Empreinte d'un fichier. Avec `normalize_eol`, les fins de ligne CRLF comptent comme LF :
    l'empreinte d'un fichier texte est alors la même sous Windows et sous Linux."""
    if normalize_eol:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_npy(array: np.ndarray) -> str:
    """Empreinte du fichier .npy qu'écrirait `np.save(array)`."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def matches_frozen(path: Path, digest: str) -> bool:
    """Le fichier a-t-il l'empreinte figée ? Binaire : octet par octet. Texte : le même contenu
    en LF ou en CRLF (Git convertit les fins de ligne selon le système ; les empreintes de la
    section 29 ont été prises sur le PC, certaines en LF, d'autres en CRLF)."""
    if Path(path).suffix == ".npy":
        return sha256_file(path) == digest
    lf = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return digest in (hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest())


def check_files(expected: dict[str, str], base: Path = ROOT) -> dict[str, bool]:
    """Compare les empreintes ; lève une erreur au premier écart."""
    result = {}
    for name, digest in expected.items():
        result[name] = matches_frozen(base / name, digest)
        if not result[name]:
            raise RuntimeError(f"Empreinte différente de la section 29 : {name}")
    return result


def training_rows(spec: dict, n_rows: int) -> np.ndarray | None:
    """Lignes d'apprentissage : tout l'entraînement, ou l'échantillon figé (graine 42)."""
    if spec["type"] == "lof":
        return sample_rows(n_rows, spec["reglage"]["echantillon"], SEED)
    if spec["type"] == "autoencodeur":
        return sample_rows(n_rows, SAMPLE_ROWS, SEED)
    return None


def build_and_fit(spec: dict, train: np.ndarray, labels: np.ndarray):
    """Modèle appris avec son réglage figé ; les étiquettes ne servent qu'aux modèles supervisés."""
    kind, settings = spec["type"], spec["reglage"]
    rows = training_rows(spec, len(train))
    x = np.asarray(train if rows is None else train[rows], dtype=np.float32)
    if kind == "isolation_forest" and settings.get("max_features", 1.0) < 1.0:
        # Avec une part des variables, chaque arbre copie ses colonnes de tout l'entraînement :
        # sur 12 cœurs en parallèle, le pic dépassait 5 Go. Deux cœurs pour apprendre, puis le
        # réglage d'origine. Résultat identique : les graines des arbres sont tirées avant.
        model = model_from_settings(kind, settings, SEED).set_params(n_jobs=FOREST_FIT_JOBS)
        return fit(model, x).set_params(n_jobs=-1)
    if kind in ("isolation_forest", "lof", "autoencodeur"):
        return fit(model_from_settings(kind, settings, SEED), x)
    if kind == "lightgbm":
        return lightgbm_model(settings or None, SEED).fit(x, labels)
    if kind == "logistique":
        return logistic_model("aucun", labels).fit(x, labels)
    raise ValueError(f"Type inconnu : {kind}")


def score(spec: dict, model, x: np.ndarray) -> np.ndarray:
    """Score orienté : plus haut = plus suspect (section 29)."""
    if spec["type"] in ("lightgbm", "logistique"):
        return ranking_scores(model, np.asarray(x, dtype=np.float32))
    return anomaly_scores(model, x)


def output_path(directory: Path, correction: str | None) -> Path:
    """Fichier de résultats du test.

    Sans correction : `test.json`, refusé s'il existe déjà (le test n'est ouvert
    qu'une fois). Avec une correction (raison écrite obligatoire) : le premier
    `test_correction_N.json` libre ; les fichiers précédents sont gardés.
    """
    first = directory / TEST_RESULT
    if correction is None:
        if first.exists():
            raise RuntimeError(f"{first.name} existe déjà : le test a été ouvert. "
                               "Une correction exige --correction avec une raison écrite.")
        return first
    if not correction.strip():
        raise ValueError("Une correction exige une raison écrite.")
    if not first.exists():
        raise RuntimeError("Aucun résultat à corriger : lancer d'abord sans --correction.")
    n = 1
    while (directory / f"test_correction_{n}.json").exists():
        n += 1
    return directory / f"test_correction_{n}.json"
