"""Explications SHAP et codes motifs (docs/CHOIX.md, section 28).

Sens unique pour tous les modèles : une contribution positive pousse vers
« suspect ». LightGBM est expliqué sur son score brut (log-odds) par TreeSHAP,
calculé par LightGBM lui-même ; LOF sur son score inversé (−score_samples) par
KernelSHAP.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aml_xai.config import ROOT, SEED
from aml_xai.features import model_feature_names

CODES_PATH = ROOT / "config" / "codes_motifs.json"
MAX_CODES = 3
# KernelSHAP sur LOF (réglages fixés avant les mesures)
KERNEL_BACKGROUND_ROWS = 10_000
KERNEL_BACKGROUND_CENTERS = 20
KERNEL_SAMPLES = 500

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def load_codes(path: Path = CODES_PATH) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["codes"]


def group_matrix(codes: dict, names: list[str] | None = None) -> np.ndarray:
    """Matrice (variables × codes) de 0 et 1 : contributions de groupe = contributions @ matrice."""
    names = names or model_feature_names()
    out = np.zeros((len(names), len(codes)))
    for j, code in enumerate(codes.values()):
        for variable in code["variables"]:
            out[names.index(variable), j] = 1.0
    return out


def lightgbm_contributions(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TreeSHAP exact sur le score brut : (contributions par variable, valeur de base)."""
    raw = model.booster_.predict(x, pred_contrib=True, num_iteration=model.best_iteration_ or None)
    return raw[:, :-1], raw[:, -1]


def lof_score(model):
    """Score de LOF orienté comme les autres : plus haut = plus suspect."""
    return lambda x: -model.score_samples(np.asarray(x, dtype=np.float32))


def kernel_background(train: np.ndarray, seed: int = SEED, centers: int = KERNEL_BACKGROUND_CENTERS):
    """Référence de KernelSHAP : centres de k-moyennes d'un échantillon de l'entraînement."""
    import shap

    rows = np.sort(np.random.default_rng(seed).choice(len(train), KERNEL_BACKGROUND_ROWS, replace=False))
    return shap.kmeans(np.asarray(train[rows], dtype=np.float64), centers)


def kernel_contributions(score, x: np.ndarray, background, seed: int = SEED,
                         samples: int = KERNEL_SAMPLES) -> tuple[np.ndarray, float]:
    """KernelSHAP (approché) : (contributions par variable, valeur de base)."""
    import shap

    explainer = shap.KernelExplainer(score, background)
    # KernelSHAP tire ses échantillons avec le générateur global de numpy (il ignore
    # `random_state`) : on le fixe ici pour que les explications soient reproductibles
    np.random.seed(seed)
    values = explainer.shap_values(np.asarray(x, dtype=np.float64), nsamples=samples, silent=True)
    return np.asarray(values), float(explainer.expected_value)


def reason_codes(group_contributions: np.ndarray, code_names: list[str], top: int = MAX_CODES) -> list[str]:
    """Au plus `top` codes, ceux de plus forte contribution, seulement s'ils sont positifs."""
    order = np.argsort(-group_contributions, kind="stable")[:top]
    return [code_names[j] for j in order if group_contributions[j] > 0]


def deletion_drop(score, x: np.ndarray, groups: list[int], reference: np.ndarray, gmatrix: np.ndarray) -> float:
    """Baisse du score quand les variables des groupes donnés prennent la valeur de référence
    (médiane de l'entraînement pour la règle, référence de SHAP pour le second test)."""
    changed = x.copy()
    columns = np.flatnonzero(gmatrix[:, groups].sum(axis=1) > 0)
    changed[columns] = reference[columns]
    return float(score(x[None, :])[0] - score(changed[None, :])[0])


def random_groups(rng, n_groups: int, chosen: list[int], among_others: bool) -> list[int]:
    """Autant de groupes que `chosen`, tirés au hasard ; parmi les autres groupes seulement
    pour le second test (conçu après les résultats)."""
    pool = [j for j in range(n_groups) if not among_others or j not in chosen]
    return list(rng.choice(pool, size=len(chosen), replace=False))


DUREE_UNITES = [(86_400, "jours"), (3_600, "h"), (60, "min"), (1, "s")]


REQUIRED_DELETION_SHARE = 0.90


def deletion_test(score, x: np.ndarray, codes: list[list[str]], reference: np.ndarray, gmatrix: np.ndarray,
                  names: list[str], among_others: bool, seed: int = SEED) -> dict:
    """Baisse du score en retirant les codes motifs de chaque alerte, contre autant de groupes au hasard."""
    rng = np.random.default_rng(seed)
    wins, drops_codes, drops_random = 0, [], []
    for i, alert_codes in enumerate(codes):
        chosen = [names.index(c) for c in alert_codes]
        if not chosen:
            continue
        a = deletion_drop(score, x[i], chosen, reference, gmatrix)
        b = deletion_drop(score, x[i], random_groups(rng, len(names), chosen, among_others), reference, gmatrix)
        drops_codes.append(a)
        drops_random.append(b)
        wins += a > b
    n = len(drops_codes)
    return {"alertes": n, "part_codes_plus_forts": round(wins / n, 4),
            "part_codes_ne_baissent_pas": round(float(np.mean(np.array(drops_codes) <= 0)), 4),
            "baisse_mediane_codes": round(float(np.median(drops_codes)), 4),
            "baisse_mediane_hasard": round(float(np.median(drops_random)), 4)}


def both_tests(score, x, codes, medians, shap_reference, gmatrix, names) -> dict:
    """La règle fixée avant les mesures, et le second test conçu après les résultats (section 28)."""
    rule = deletion_test(score, x, codes, medians, gmatrix, names, among_others=False)
    rule["fidele"] = rule["part_codes_plus_forts"] >= REQUIRED_DELETION_SHARE
    after = deletion_test(score, x, codes, shap_reference, gmatrix, names, among_others=True)
    after["statut"] = "conçu après les résultats, à titre d'information"
    return {"regle": rule, "apres_resultats": after}


def _durees(*seconds: float) -> list[str]:
    """Durées écrites dans la même unité, celle de la plus courte (« 29 h », « 14 h »)."""
    size, unit = next((s, u) for s, u in DUREE_UNITES if min(seconds) >= s or s == 1)
    return [_nombre(v / size, 0 if v / size >= 10 else 1) + f" {unit}" for v in seconds]


def _nombre(v: float, decimales: int = 0) -> str:
    return f"{v:,.{decimales}f}".replace(",", " ").replace(".", ",")


def _compare(value: float, usual: float, text: str, shown: str, usual_shown: str) -> str:
    """« seulement » si la valeur est sous l'habitude, et la valeur habituelle entre parenthèses."""
    if value == usual:
        return f"{shown} {text} (valeur habituelle)"
    return f"{'seulement ' if value < usual else ''}{shown} {text} (habituellement {usual_shown})"


def _count(row: dict, usual: dict, name: str, singular: str, plural: str, feminine: bool = False) -> str:
    """Compte accordé : « aucun envoi », « 1 envoi », « 13 envois », avec le sens et l'habitude."""
    value, habit = row[name], usual[name]
    if value == 0:
        text = f"{'aucune' if feminine else 'aucun'} {singular}"
    else:
        text = f"{'seulement ' if value < habit else ''}{_nombre(value)} {singular if value == 1 else plural}"
    return text + (" (valeur habituelle)" if value == habit else f" (habituellement {_nombre(habit)})")


def phrase(code: str, row: dict, usual: dict) -> str:
    """Phrase d'un code motif : vraies valeurs, sens et valeur habituelle (médiane de l'entraînement)."""
    if code == "MONTANT":
        return "Montant de " + _compare(row["Amount"], usual["Amount"], "£", _nombre(row["Amount"], 2),
                                        f"{_nombre(usual['Amount'], 2)} £")
    if code in ("MONTANT_INHABITUEL_EMETTEUR", "MONTANT_INHABITUEL_DESTINATAIRE"):
        side, verb = ("em", "envoyé") if code.endswith("EMETTEUR") else ("re", "reçu")
        if row[f"{side}_sans_historique_30j"]:
            return f"Aucun montant {verb} sur les 30 jours précédents pour comparer"
        name = f"{side}_ratio_montant_30j"
        return _compare(row[name], usual[name], f"fois le montant moyen {verb} sur 30 jours",
                        _nombre(row[name], 1), _nombre(usual[name], 1))
    if code == "RAFALE_EMETTEUR":
        return (_count(row, usual, "em_nb_1h", "envoi dans l'heure précédente", "envois dans l'heure précédente")
                + ", " + _count(row, usual, "em_nb_24h", "envoi sur 24 h", "envois sur 24 h")).capitalize()
    if code == "RAFALE_DESTINATAIRE":
        return (_count(row, usual, "re_nb_1h", "réception dans l'heure précédente",
                       "réceptions dans l'heure précédente", feminine=True)
                + ", " + _count(row, usual, "re_nb_24h", "réception sur 24 h", "réceptions sur 24 h",
                                feminine=True)).capitalize()
    if code == "CONTREPARTIES_MULTIPLES_EMETTEUR":
        return (_count(row, usual, "em_nb_distincts_24h", "destinataire différent en 24 h",
                       "destinataires différents en 24 h")
                + ", " + _count(row, usual, "em_nb_distincts_30j", "destinataire différent en 30 jours",
                                "destinataires différents en 30 jours")).capitalize()
    if code == "SOURCES_MULTIPLES_DESTINATAIRE":
        return _count(row, usual, "re_nb_distincts_30j", "émetteur différent en 30 jours",
                      "émetteurs différents en 30 jours").capitalize()
    if code in ("REPRISE_OU_NOUVEAU_EMETTEUR", "REPRISE_OU_NOUVEAU_DESTINATAIRE"):
        side = "em" if code.endswith("EMETTEUR") else "re"
        if row[f"{side}_sans_historique_30j"]:
            return "Aucun envoi dans les 30 jours précédents" if side == "em" else "Aucune réception dans les 30 jours précédents"
        name = f"{side}_ecart_precedent_s"
        start = "Envoi précédent il y a" if side == "em" else "Réception précédente il y a"
        value, habit = _durees(row[name], usual[name])
        if row[name] == usual[name]:
            return f"{start} {value} (valeur habituelle)"
        word = "seulement " if row[name] < usual[name] else ""
        return f"{start} {word}{value} (habituellement {habit})"
    if code == "HORAIRE":
        t = pd.Timestamp(row["timestamp"])
        return f"Le {JOURS[t.day_of_week]} à {t.hour} h {t.minute:02d}"
    if code == "DEVISE":
        if row["Payment_currency"] == row["Received_currency"]:
            return f"Même devise : payé et reçu en {row['Payment_currency']}"
        return f"Devise différente : payé en {row['Payment_currency']}, reçu en {row['Received_currency']}"
    if code == "PAYS":
        return f"De {row['Sender_bank_location']} vers {row['Receiver_bank_location']}"
    if code == "TYPE_PAIEMENT":
        return f"Type de paiement : {row['Payment_type']}"
    raise ValueError(f"Code inconnu : {code}")
