"""Cœur de l'API (docs/CHOIX.md, section 32), indépendant du web.

Au démarrage : vérifie l'empreinte de chaque fichier, charge les modèles une
seule fois. Ensuite : transforme une transaction reçue en ligne de la matrice
(même code que le notebook), la note, donne son rang par rapport à la
validation et explique LightGBM en direct.
"""

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from aml_xai.config import FREQUENCY_MAPS_PATH, MODELS_DIR, ROOT, TIMESTAMP
from aml_xai.explain import group_matrix, load_codes, phrase, reason_codes
from aml_xai.features import (
    GAP_CAP_S,
    NEUTRAL_RATIO,
    apply_frequency_maps,
    history_columns,
    transaction_features,
)
from aml_xai.frozen import load_frozen, sha256_file
from aml_xai.matrix import transform

LOF_PATH = MODELS_DIR / "lof_sans_etiquette.joblib"
LGBM_PATH = MODELS_DIR / "lightgbm_regle.txt"
QUANTILES_PATH = MODELS_DIR / "quantiles_validation.json"
MANIFEST_PATH = MODELS_DIR / "manifeste.json"
DEMO_PATH = ROOT / "data" / "demo" / "exemples.json"
MANIFEST_FILES = [
    "models/lof_sans_etiquette.joblib",
    "models/lightgbm_regle.txt",
    "models/quantiles_validation.json",
    "models/frequences.json",
    "models/transformation.json",
    "models/valeurs_habituelles.json",
    "config/codes_motifs.json",
    "config/hypotheses.json",
    "config/fige_test.json",
    "data/demo/exemples.json",
]

# Champ de l'API → colonne du jeu
RAW_FIELDS = {
    "montant": "Amount",
    "horodatage": TIMESTAMP,
    "devise_paiement": "Payment_currency",
    "devise_reception": "Received_currency",
    "pays_emetteur": "Sender_bank_location",
    "pays_destinataire": "Receiver_bank_location",
    "type_paiement": "Payment_type",
}
HISTORY_FEATURES = history_columns()
MODELS = {"lof": "LOF, sans étiquette", "lightgbm": "LightGBM, réglé"}
DEFAULT_MODEL = "lof"


def manifest_hash(name: str) -> str:
    """Empreinte d'un fichier du manifeste : octets exacts pour le binaire, fins de ligne
    normalisées pour le texte (Git les convertit selon le système)."""
    return sha256_file(ROOT / name, normalize_eol=not name.endswith(".joblib"))


def history_errors(h: dict) -> list[str]:
    """Contrôle de cohérence des variables d'historique (définitions de la section 18)."""
    errors = []
    counts = [n for n in HISTORY_FEATURES if "_nb_" in n]
    for name in counts:
        if h[name] < 0 or h[name] != int(h[name]):
            errors.append(f"{name} doit être un entier positif ou nul")
    for side in ("em", "re"):
        if not 0 <= h[f"{side}_ecart_precedent_s"] <= GAP_CAP_S:
            errors.append(f"{side}_ecart_precedent_s doit être compris entre 0 et {GAP_CAP_S} s (30 jours)")
        if h[f"{side}_ratio_montant_30j"] <= 0:
            errors.append(f"{side}_ratio_montant_30j doit être strictement positif")
        if h[f"{side}_nb_1h"] > h[f"{side}_nb_24h"]:
            errors.append(f"{side}_nb_1h ne peut pas dépasser {side}_nb_24h")
        if h[f"{side}_sans_historique_30j"] not in (0, 1):
            errors.append(f"{side}_sans_historique_30j vaut 0 ou 1")
        elif h[f"{side}_sans_historique_30j"] == 1:
            empty = [n for n in counts if n.startswith(side) and h[n] != 0]
            if empty or h[f"{side}_ratio_montant_30j"] != NEUTRAL_RATIO or h[f"{side}_ecart_precedent_s"] != GAP_CAP_S:
                errors.append(f"sans historique sur 30 jours ({side}) : compteurs nuls, rapport de 1 et écart de "
                              f"{GAP_CAP_S} s attendus")
    if h["em_nb_distincts_24h"] > h["em_nb_24h"]:
        errors.append("em_nb_distincts_24h ne peut pas dépasser em_nb_24h")
    if h["em_nb_distincts_24h"] > h["em_nb_distincts_30j"]:
        errors.append("em_nb_distincts_24h ne peut pas dépasser em_nb_distincts_30j")
    return errors


def rank_text(share: float) -> str:
    """« parmi les 0,4 % les plus suspectes » (deux chiffres significatifs)."""
    percent = 100 * share
    digits = max(0, 1 - int(np.floor(np.log10(percent)))) if percent < 10 else 0
    return f"parmi les {f'{percent:.{digits}f}'.replace('.', ',')} % les plus suspectes de la validation"


class Service:
    """Fichiers vérifiés et modèles chargés une seule fois."""

    def __init__(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for name in MANIFEST_FILES:
            if manifest_hash(name) != manifest[name]:
                raise RuntimeError(f"Empreinte différente du manifeste : {name}")
        self.empreintes = manifest
        self.maps = json.loads(FREQUENCY_MAPS_PATH.read_text(encoding="utf-8"))
        self.params = json.loads((MODELS_DIR / "transformation.json").read_text(encoding="utf-8"))
        self.usual = json.loads((MODELS_DIR / "valeurs_habituelles.json").read_text(encoding="utf-8"))
        self.quantiles = json.loads(QUANTILES_PATH.read_text(encoding="utf-8"))
        self.codes = load_codes()
        self.code_names = list(self.codes)
        self.gmatrix = group_matrix(self.codes)
        self.frozen = load_frozen()["modeles"]
        self.examples = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
        self.lof = joblib.load(LOF_PATH)
        self.lgbm = lgb.Booster(model_file=str(LGBM_PATH))

    def threshold(self, model: str) -> float:
        return self.frozen[MODELS[model]]["seuil"]

    def prepare(self, transaction: dict) -> tuple[np.ndarray, dict, list[str]]:
        """Ligne de la matrice, valeurs pour les phrases, valeurs inconnues de l'entraînement."""
        raw = {column: transaction[field] for field, column in RAW_FIELDS.items()}
        raw[TIMESTAMP] = pd.Timestamp(raw[TIMESTAMP]).tz_localize(None)
        df = pd.DataFrame([raw])
        df[TIMESTAMP] = df[TIMESTAMP].astype("datetime64[s]")
        unknown = [f for f, c in RAW_FIELDS.items() if c in self.maps and str(raw[c]) not in self.maps[c]]
        data = {**transaction_features(df), **apply_frequency_maps(df, self.maps),
                **{n: np.array([transaction["historique"][n]], dtype=np.float32) for n in HISTORY_FEATURES}}
        ordered = {name: data[name] for name in self.params["variables"]}
        return transform(ordered, self.params), {**raw, **transaction["historique"]}, unknown

    def score(self, model: str, x: np.ndarray) -> float:
        if model == "lof":
            return float(-self.lof.score_samples(x)[0])
        return float(self.lgbm.predict(x, raw_score=True)[0])

    def rank(self, model: str, value: float) -> float:
        """Part des transactions de validation au moins aussi suspectes."""
        q = self.quantiles[model]
        return float(np.interp(-value, -np.array(q["scores"]), q["parts"]))

    def explain_lightgbm(self, x: np.ndarray, row: dict) -> dict:
        contributions = self.lgbm.predict(x, pred_contrib=True)[0]
        groups = contributions[:-1] @ self.gmatrix
        return {
            "valeur_de_base": float(contributions[-1]),
            "contributions_groupes": dict(zip(self.code_names, np.round(groups, 6).tolist())),
            "codes_motifs": [{"code": c, "libelle": self.codes[c]["libelle"],
                              "contribution": round(float(groups[self.code_names.index(c)]), 6),
                              "phrase": phrase(c, row, self.usual)}
                             for c in reason_codes(groups, self.code_names)],
        }
