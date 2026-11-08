"""Fichiers JSON commités, lus sans l'API, et mise en forme commune."""

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
NOTE_DONNEES = ("Données **synthétiques** : jeu SAML-D (Oztas et al., IEEE ICEBE 2023), licence "
                "CC BY-NC-SA 4.0. Aucune donnée réelle.")
DEPOT = "https://github.com/aljene307/aml-detection-xai"

HISTORIQUE = {
    "em_nb_1h": "Envois dans l'heure précédente",
    "em_nb_24h": "Envois dans les 24 h précédentes",
    "em_ratio_montant_30j": "Montant / montant moyen envoyé sur 30 jours",
    "em_ecart_precedent_s": "Secondes depuis l'envoi précédent (max. 30 jours)",
    "em_sans_historique_30j": "Aucun envoi sur 30 jours (0 ou 1)",
    "em_nb_distincts_24h": "Destinataires différents sur 24 h",
    "em_nb_distincts_30j": "Destinataires différents sur 30 jours",
    "re_nb_1h": "Réceptions dans l'heure précédente",
    "re_nb_24h": "Réceptions dans les 24 h précédentes",
    "re_ratio_montant_30j": "Montant / montant moyen reçu sur 30 jours",
    "re_ecart_precedent_s": "Secondes depuis la réception précédente (max. 30 jours)",
    "re_sans_historique_30j": "Aucune réception sur 30 jours (0 ou 1)",
    "re_nb_distincts_30j": "Émetteurs différents sur 30 jours",
}
MODELES = {"lof": "LOF, sans étiquette", "lightgbm": "LightGBM, réglé"}
# Catégories des exemples : réponse de LightGBM réglé au seuil figé, comparée à l'étiquette.
# LOF peut répondre autrement sur la même transaction.
CATEGORIES = {"vrai_cas_alerte": "vrai cas alerté par LightGBM", "fausse_alerte": "fausse alerte de LightGBM",
              "cas_manque": "cas manqué par LightGBM", "normale": "normale, non alertée par LightGBM"}


@st.cache_data(show_spinner=False)
def lire(chemin: str):
    return json.loads((ROOT / chemin).read_text(encoding="utf-8"))


def exemples() -> list[dict]:
    return lire("data/demo/exemples_app.json")


def fr(x: float, decimales: int = 2) -> str:
    """Nombre au format français : virgule décimale, espace pour les milliers."""
    return f"{x:,.{decimales}f}".replace(",", " ").replace(".", ",")


def budget_le_plus_proche(alertes_par_jour: list[int], k: float) -> int:
    """Position du plus grand budget calculé qui ne dépasse pas k (au moins le premier)."""
    positions = [i for i, b in enumerate(alertes_par_jour) if b <= k]
    return positions[-1] if positions else 0


def hypotheses_derivees(analystes: int, heures: float, minutes: float, cout_horaire: float) -> tuple[float, float]:
    """Formules de config/hypotheses.json : alertes par jour et coût d'une alerte (£)."""
    return analystes * heures * 60 / minutes, minutes / 60 * cout_horaire


def pied_de_page() -> None:
    st.divider()
    st.caption(NOTE_DONNEES + f" Code : [{DEPOT}]({DEPOT}).")
