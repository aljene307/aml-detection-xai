"""Page « Résultats » : lue dans reports/mesures/, sans appeler l'API."""

import pandas as pd
import streamlit as st

from donnees import fr, lire, pied_de_page
from graphiques import comparaison_modeles, courbe_apprentissage, rappel_typologies


def afficher() -> None:
    st.title("Résultats")
    test = lire("reports/mesures/test.json")
    validation = lire("reports/mesures/repetition_validation.json")
    st.write(f"Test ouvert **une seule fois** après avoir figé tous les choix : {fr(test['lignes'], 0)} transactions, "
             f"{fr(test['cas'], 0)} cas de blanchiment ({fr(100 * test['taux'], 3)} %).")

    st.plotly_chart(comparaison_modeles(test, validation), width="stretch")
    st.dataframe(pd.DataFrame({
        nom: {"PR-AUC validation": fr(validation["modeles"][nom]["pr_auc"], 4), "PR-AUC test": fr(t["pr_auc"], 4),
              "Intervalle à 95 % (test)": " à ".join(fr(v, 4) for v in t["pr_auc_intervalle_95"]),
              "Fois le hasard (test)": fr(t["fois_le_hasard"], 0)}
        for nom, t in test["modeles"].items()}).T, width="stretch")
    st.caption("Pour les 8 modèles, la PR-AUC de validation est dans l'intervalle du test : aucun écart ne dépasse "
               "le hasard.")

    st.subheader("Au seuil figé, sur le test (capacité : 140 alertes par jour)")
    st.dataframe(pd.DataFrame({
        nom: {"Alertes par jour": fr(s["alertes_par_jour"], 1), "Précision": fr(s["precision"], 3),
              "Rappel": fr(s["rappel"], 3), "F2": fr(s["f2"], 3), "MCC": fr(s["mcc"], 3),
              "Coût (M£)": fr(s["cout_total"] / 1e6, 1),
              "Coût sans Over-Invoicing (M£)": fr(s["cout_sans_over_invoicing"] / 1e6, 1)}
        for nom, t in test["modeles"].items() if "seuil" in t for s in [t["seuil"]]}).T, width="stretch")
    st.caption("Recommandation : en production, garder les 140 transactions les plus suspectes de chaque jour, "
               "qui respecte exactement la capacité, avec presque les mêmes résultats.")

    supervise = lire("reports/mesures/supervise_tableau.json")
    st.plotly_chart(courbe_apprentissage(supervise["courbe_apprentissage"],
                                         supervise["reference_lof"]["sans_etiquette"]), width="stretch")
    st.caption("Validation ; chaque point est un tirage de comptes de l'entraînement.")
    st.plotly_chart(rappel_typologies(test), width="stretch")
    pied_de_page()
