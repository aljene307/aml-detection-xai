"""Page « Analyser une transaction » : la seule qui appelle l'API."""

from datetime import datetime

import streamlit as st

from client import ApiUnavailable, call
from donnees import CATEGORIES, HISTORIQUE, MODELES, exemples, fr, lire, pied_de_page
from graphiques import contributions

SENS_SCORE = {
    "lof": "Score d'anomalie de LOF : plus il est haut, plus la transaction s'écarte de ses voisines.",
    "lightgbm": "Score de risque de LightGBM, en log-odds : il peut être négatif ; seul le rang dit "
                "si la transaction est parmi les plus suspectes.",
}


def formulaire(exemple: dict) -> tuple[dict, str] | None:
    """Champs préremplis à partir de l'exemple ; renvoie la transaction et le modèle à l'envoi."""
    frequences = lire("models/frequences.json")
    entree = exemple["entree"]
    with st.form("transaction"):
        modele = st.radio("Modèle", list(MODELES), format_func=lambda m: MODELES[m] + (
            " (par défaut, sans étiquette)" if m == "lof" else " (comparaison)"), horizontal=True)
        a, b = st.columns(2)
        montant = a.number_input("Montant (£)", min_value=0.01, value=float(entree["montant"]), step=100.0)
        # Un seul champ texte : le champ « heure » de Streamlit perd les secondes, ce qui changerait la transaction
        moment = b.text_input("Date et heure (AAAA-MM-JJ HH:MM:SS)",
                              value=entree["horodatage"].replace("T", " "), key=f"moment_{exemple['id']}")
        champs = {}
        for colonne, (champ, source, libelle) in zip(
                st.columns(5),
                [("devise_paiement", "Payment_currency", "Devise payée"),
                 ("devise_reception", "Received_currency", "Devise reçue"),
                 ("pays_emetteur", "Sender_bank_location", "Pays de l'émetteur"),
                 ("pays_destinataire", "Receiver_bank_location", "Pays du destinataire"),
                 ("type_paiement", "Payment_type", "Type de paiement")]):
            valeurs = sorted(frequences[source])
            champs[champ] = colonne.selectbox(libelle, valeurs, index=valeurs.index(entree[champ]))
        historique = {}
        with st.expander("Variables d'historique (calculées par le système de la banque)", expanded=False):
            emetteur, destinataire = st.columns(2)
            for nom, libelle in HISTORIQUE.items():
                colonne = emetteur if nom.startswith("em_") else destinataire
                valeur = entree["historique"][nom]
                if "ratio" in nom:
                    historique[nom] = colonne.number_input(libelle, min_value=0.0001, value=float(valeur),
                                                           format="%.4f", key=f"{nom}_{exemple['id']}")
                else:
                    historique[nom] = int(colonne.number_input(libelle, min_value=0, value=int(valeur), step=1,
                                                               key=f"{nom}_{exemple['id']}"))
        if not st.form_submit_button("Analyser", type="primary"):
            return None
    try:
        horodatage = datetime.strptime(moment.strip(), "%Y-%m-%d %H:%M:%S").isoformat()
    except ValueError:
        st.error("Date et heure attendues au format AAAA-MM-JJ HH:MM:SS, par exemple 2023-07-30 21:58:04.")
        return None
    transaction = {"montant": montant, "horodatage": horodatage, **champs, "historique": historique}
    return transaction, modele


def resultat(r: dict) -> None:
    alerte, rang, score = st.columns([1, 2, 1])
    alerte.metric("Alerte", "Oui" if r["alerte"] else "Non")
    rang.metric("Rang", r["rang_texte"].capitalize())
    score.metric("Score", fr(r["score"], 3), help=f"Seuil figé : {fr(r['seuil'], 3)}")
    st.caption(SENS_SCORE[r["modele"]])
    if r["valeurs_inconnues"]:
        st.info("Valeurs jamais vues à l'entraînement (codées comme rares) : " + ", ".join(r["valeurs_inconnues"]))
    if r["explication"] is None:
        st.warning(r["message"])
        return
    if r.get("message"):
        st.caption(r["message"])
    st.subheader("Codes motifs")
    if not r["explication"]["codes_motifs"]:
        st.write("Aucun groupe ne pousse vers « suspect ».")
    for code in r["explication"]["codes_motifs"]:
        st.markdown(f"- **{code['libelle']}** (+{fr(code['contribution'], 3)}) : {code['phrase']}")
    libelles = {c: v["libelle"] for c, v in lire("config/codes_motifs.json")["codes"].items()}
    st.plotly_chart(contributions(r["explication"]["contributions_groupes"], libelles,
                                  "Contribution de chaque groupe de variables"), width="stretch")


def afficher() -> None:
    st.title("Analyser une transaction")
    st.write("Choisissez un exemple pour préremplir le formulaire, modifiez les valeurs si vous le souhaitez, "
             "puis lancez l'analyse. Cette page appelle l'API ; les autres pages fonctionnent sans elle.")
    liste = exemples()
    choix = st.selectbox("Exemple de départ", liste, format_func=lambda e: (
        f"{e['id']} · {CATEGORIES[e['categorie']]} · {e['typologie']}"))
    envoi = formulaire(choix)
    if envoi is not None:
        transaction, modele = envoi
        try:
            with st.spinner("Analyse en cours (si l'API dort, son réveil prend environ une minute)…"):
                reponse = call("POST", "/explain", transaction, {"modele": modele})
        except ApiUnavailable as erreur:
            st.warning(str(erreur))
        else:
            if reponse.status_code == 422:
                details = reponse.json().get("detail", [])
                messages = [d.get("msg", str(d)) for d in details] if isinstance(details, list) else [str(details)]
                st.error("Entrée refusée par l'API : " + " ; ".join(messages))
            elif reponse.status_code != 200:
                st.error(f"Réponse inattendue de l'API (code {reponse.status_code}).")
            else:
                resultat(reponse.json())
    pied_de_page()
