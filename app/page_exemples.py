"""Page « Exemples » : les 30 transactions du test, sans appeler l'API."""

import pandas as pd
import streamlit as st

from donnees import CATEGORIES, MODELES, exemples, fr, lire, pied_de_page
from graphiques import contributions


def colonne_modele(colonne, code: str, e: dict, libelles: dict) -> None:
    r = e[code]
    colonne.subheader(MODELES[code])
    a, b = colonne.columns(2)
    a.metric("Alerte", "Oui" if r["alerte"] else "Non")
    b.metric("Score", fr(r["score"], 3), help=f"Seuil figé : {fr(r['seuil'], 3)}")
    colonne.markdown(f"**{r['rang_texte'].capitalize()}**")
    codes = r["explication"]["codes_motifs"]
    if not codes:
        colonne.write("Aucun groupe ne pousse vers « suspect ».")
    for c in codes:
        colonne.markdown(f"- **{c['libelle']}** (+{fr(c['contribution'], 3)}) : {c['phrase']}")
    colonne.plotly_chart(contributions(r["explication"]["contributions_groupes"], libelles,
                                       "Contribution de chaque groupe"), width="stretch",
                         key=f"graphique_{code}")


def afficher() -> None:
    st.title("Exemples")
    st.write("Trente transactions du **test**, tirées par une règle fixe. Les catégories viennent de "
             "**LightGBM réglé** (avec étiquettes) au seuil figé : 10 vrais cas alertés (typologies différentes), "
             "10 fausses alertes, 5 cas manqués, 5 transactions normales non alertées. LOF peut répondre "
             "autrement sur la même transaction. Scores et explications sont calculés à l'avance : cette page "
             "n'appelle pas l'API.")
    liste = exemples()
    tableau = pd.DataFrame([{
        "N°": e["id"], "Catégorie": CATEGORIES[e["categorie"]], "Typologie": e["typologie"],
        "Blanchiment": "oui" if e["blanchiment"] else "non",
        "Alerte LOF": "oui" if e["lof"]["alerte"] else "non",
        "Alerte LightGBM": "oui" if e["lightgbm"]["alerte"] else "non",
    } for e in liste])
    a, b = st.columns(2)
    categories = a.multiselect("Catégorie", list(CATEGORIES.values()), default=list(CATEGORIES.values()))
    typologies = b.multiselect("Typologie", sorted(tableau["Typologie"].unique()))
    filtre = tableau["Catégorie"].isin(categories)
    if typologies:
        filtre &= tableau["Typologie"].isin(typologies)
    st.dataframe(tableau[filtre], hide_index=True, width="stretch")

    choix = st.selectbox("Exemple à détailler", tableau[filtre]["N°"].tolist())
    if choix is None:
        st.info("Aucun exemple ne correspond aux filtres.")
    else:
        e = next(x for x in liste if x["id"] == choix)
        st.markdown(f"**Exemple {e['id']}** · {CATEGORIES[e['categorie']]} · typologie `{e['typologie']}` · "
                    f"{fr(e['entree']['montant'], 2)} £ de {e['entree']['pays_emetteur']} vers "
                    f"{e['entree']['pays_destinataire']}")
        libelles = {c: v["libelle"] for c, v in lire("config/codes_motifs.json")["codes"].items()}
        gauche, droite = st.columns(2)
        colonne_modele(gauche, "lof", e, libelles)
        colonne_modele(droite, "lightgbm", e, libelles)
        st.caption("LOF : explication KernelSHAP complète, calculée à l'avance. LightGBM : TreeSHAP exact.")
    pied_de_page()
