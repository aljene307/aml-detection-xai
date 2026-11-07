"""Application Streamlit : détection explicable du blanchiment (SAML-D).

Lancement local : streamlit run app/streamlit_app.py
Tous les imports sont faits au démarrage, avant st.navigation.
"""

import streamlit as st

import page_analyser
import page_budget
import page_exemples
import page_methode
import page_resultats
from client import warm_up
from donnees import NOTE_DONNEES

st.set_page_config(page_title="Détection explicable du blanchiment", layout="wide")
warm_up()  # réveille l'API à l'avance, sans bloquer l'affichage
st.sidebar.caption(NOTE_DONNEES)

navigation = st.navigation([
    st.Page(page_analyser.afficher, title="Analyser une transaction", url_path="analyser", default=True),
    st.Page(page_exemples.afficher, title="Exemples", url_path="exemples"),
    st.Page(page_budget.afficher, title="Budget et coût", url_path="budget"),
    st.Page(page_resultats.afficher, title="Résultats", url_path="resultats"),
    st.Page(page_methode.afficher, title="Méthode et limites", url_path="methode"),
])
navigation.run()
