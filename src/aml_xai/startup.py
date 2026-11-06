"""Effet de démarrage : règle fixée dans docs/CHOIX.md, section 20.

Ce module reçoit seulement des jours et des variables : il n'a jamais accès
aux étiquettes.
"""

import numpy as np

from aml_xai.features import DAY_S, SIDES, WINDOW_30D_S

REFERENCE_START_DAY = 30
# Plus longue fenêtre d'historique (moyenne, contreparties, écart plafonné) :
# avant ce jour, elles sont incomplètes pour tous les comptes, par construction
LONGEST_WINDOW_DAYS = WINDOW_30D_S // DAY_S
STARTUP_SLICES = [(0, 7), (7, 14), (14, 21), (21, 30)]
CONTROL_DAYS = 7
PSI_STRONG_CHANGE = 0.25
# Évite une division par zéro quand une classe est vide d'un côté
PSI_EPSILON = 1e-4


def with_missing(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remet une absence de valeur là où une valeur neutre a été placée.

    L'écart plafonné n'a pas de valeur neutre : le plafond est une vraie
    mesure (« 30 jours ou plus »), il reste donc tel quel.
    """
    output = dict(features)
    for side in SIDES:
        ratio = f"{side}_ratio_montant_30j"
        if ratio in output:
            output[ratio] = np.where(output[f"{side}_sans_historique_30j"] == 1, np.nan, output[ratio])
    return output


def decile_edges(reference: np.ndarray) -> np.ndarray:
    """Déciles de la référence ; les bornes répétées sont fusionnées."""
    present = reference[~np.isnan(reference)]
    if len(present) == 0:
        return np.array([])
    return np.unique(np.quantile(present, np.arange(0.1, 1.0, 0.1)))


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Part des valeurs par classe ; la dernière classe reçoit les absences."""
    classes = np.searchsorted(edges, values, side="right")
    classes[np.isnan(values)] = len(edges) + 1
    return np.bincount(classes, minlength=len(edges) + 2) / len(values)


def psi(reference: np.ndarray, sample: np.ndarray, edges: np.ndarray | None = None) -> float:
    """Indice de stabilité de population : somme de (p − q) × ln(p / q)."""
    edges = decile_edges(reference) if edges is None else edges
    p = np.maximum(_shares(reference, edges), PSI_EPSILON)
    q = np.maximum(_shares(sample, edges), PSI_EPSILON)
    return float(np.sum((p - q) * np.log(p / q)))


def _control_slices(last_day: float) -> list[tuple[int, int]]:
    """Tranches témoins complètes de la référence."""
    starts = range(REFERENCE_START_DAY, int(last_day) - CONTROL_DAYS + 1, CONTROL_DAYS)
    return [(start, start + CONTROL_DAYS) for start in starts]


def startup_report(days: np.ndarray, features: dict[str, np.ndarray]) -> dict:
    """PSI des tranches de démarrage et des témoins, par variable.

    `days` : jours écoulés depuis le début du jeu, pour les lignes d'entraînement.
    """
    is_reference = days >= REFERENCE_START_DAY
    controls = _control_slices(float(days.max()))
    report = {
        "lignes": int(len(days)),
        "lignes_reference": int(is_reference.sum()),
        "part_premiers_30_jours_pct": round(100 * float((~is_reference).mean()), 2),
        "nb_temoins": len(controls),
        "psi": {},
        "psi_temoins_max": {},
    }
    for name, values in features.items():
        reference = values[is_reference]
        edges = decile_edges(reference)
        report["psi"][name] = {
            f"{start}-{stop}": round(psi(reference, values[(days >= start) & (days < stop)], edges), 4)
            for start, stop in STARTUP_SLICES
        }
        report["psi_temoins_max"][name] = round(
            max(psi(reference, values[(days >= a) & (days < b)], edges) for a, b in controls), 4
        )
    return report


def startup_decision(report: dict) -> dict:
    """Applique la règle : quelles tranches diffèrent, et à partir de quel jour apprendre.

    La décision retenue n'est jamais avant la fin de la plus longue fenêtre
    (docs/CHOIX.md, section 20), quel que soit le résultat de la règle PSI.
    """
    different = {}
    for name, slices in report["psi"].items():
        limit = max(PSI_STRONG_CHANGE, report["psi_temoins_max"][name])
        for label, value in slices.items():
            if value > limit:
                different.setdefault(label, []).append(name)
    last_stop = max((int(label.split("-")[1]) for label in different), default=0)
    return {
        "tranches_differentes": different,
        "jour_regle_psi": last_stop,
        "exclure_avant_jour": max(last_stop, LONGEST_WINDOW_DAYS),
    }
