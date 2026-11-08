"""Page « Méthode et limites »."""

import streamlit as st

from donnees import DEPOT, NOTE_DONNEES, pied_de_page

TEXTE = f"""
### Données
{NOTE_DONNEES} Le jeu compte 9,5 millions de transactions, dont environ 0,1 % de blanchiment, réparties en
typologies (`Structuring`, `Fan_Out`, `Over-Invoicing`…). Séparation chronologique : entraînement 60 %,
validation 20 %, test 20 %.

### Démarche
- **Cœur sans étiquettes** : LOF (densité locale) et Isolation Forest apprennent sans jamais voir les
  étiquettes ; elles ne servent qu'à évaluer et à fixer le seuil.
- **Comparaison supervisée** : LightGBM, pour mesurer ce que l'on perd sans étiquettes.
- **Variables d'historique** calculées sur l'historique complet de chaque compte, en ne regardant que le
  passé `[t − fenêtre ; t[`.
- **Seuil** choisi sur la validation : coût minimal sous la capacité de l'équipe (140 alertes par jour).
- **Explications** : SHAP regroupé en 13 codes motifs, au plus 3 par alerte, avec une phrase.
- **Test ouvert une seule fois**, après avoir figé et poussé tous les choix sur GitHub.

### Limites
- Données **synthétiques** : les résultats ne disent pas comment les méthodes se comporteraient sur de vraies
  données bancaires.
- La **validation** a servi à régler et à choisir : ses scores sont optimistes ; le test les a confirmés.
- Les **explications ne sont pas fidèles** selon la règle fixée avant les mesures : retirer les codes motifs
  fait plus baisser le score que retirer d'autres variables dans environ trois alertes sur quatre, au lieu
  des neuf sur dix exigées.
- Le **coût** est dominé par quelques très gros montants, surtout `Over-Invoicing` : il se lit avec le rappel et la précision.
- Le **montant** n'est qu'une approximation du coût réel d'un blanchiment manqué (amendes, risque d'image).
- Les **changements de comportement** échappent à LOF, faute de variables comparant un compte à son passé.

### Pour aller plus loin
Tous les choix, leurs raisons et les mesures sont dans le journal du dépôt :
[{DEPOT}]({DEPOT}) (`docs/CHOIX.md`).
"""


def afficher() -> None:
    st.title("Méthode et limites")
    st.markdown(TEXTE)
    pied_de_page()
