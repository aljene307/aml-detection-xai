"""Page « Budget et coût » (précision 3) : hypothèses réglables, courbes de validation."""

import pandas as pd
import streamlit as st

from donnees import budget_le_plus_proche, fr, hypotheses_derivees, lire, pied_de_page
from graphiques import courbes_budget, mise_en_page

MESURES = {"rappel": "Rappel (part des cas trouvés)", "precision": "Précision (part de vrais cas parmi les alertes)",
           "cout": "Coût total (M£)", "cout_sans_over_invoicing": "Coût sans Over-Invoicing (M£)"}
PAR_DEFAUT = ["LOF, sans étiquette", "LightGBM, réglé"]


def series(budget: dict, cout_alerte: float) -> dict:
    """Rappel, précision et coûts pour chaque budget, à partir des comptes enregistrés."""
    out = {}
    for nom, m in budget["modeles"].items():
        alertes = m["alertes"]
        out[nom] = {
            "rappel": [vp / budget["cas"] for vp in m["vp"]],
            "precision": [vp / a if a else 0.0 for vp, a in zip(m["vp"], alertes)],
            "cout": [(budget["montant_blanchi"] - d + a * cout_alerte) / 1e6 for d, a in zip(m["montant_detecte"], alertes)],
            "cout_sans_over_invoicing": [
                (budget["montant_blanchi_sans_over_invoicing"] - d + a * cout_alerte) / 1e6
                for d, a in zip(m["montant_detecte_sans_over_invoicing"], alertes)],
        }
    return out


def afficher() -> None:
    st.title("Budget et coût")
    hyp = lire("config/hypotheses.json")
    st.write("Les hypothèses sont **écrites et réglables** (`config/hypotheses.json`). Le coût suit le sujet : "
             "montants des blanchiments non détectés + nombre d'alertes × coût d'une alerte, en livres sterling.")
    a, b, c, d = st.columns(4)
    analystes = a.slider("Analystes", 1, 20, hyp["analystes"])
    heures = b.slider("Heures par jour", 1, 10, int(hyp["heures_par_jour"]))
    minutes = c.slider("Minutes par alerte", 5, 60, hyp["minutes_par_alerte"])
    cout_horaire = d.slider("Coût horaire (£)", 10, 100, hyp["cout_horaire_livres"])
    k, cout_alerte = hypotheses_derivees(analystes, heures, minutes, cout_horaire)
    x, y = st.columns(2)
    x.metric("Alertes par jour", fr(k, 0), help=hyp["formules"]["alertes_par_jour"])
    y.metric("Coût d'une alerte", f"{fr(cout_alerte, 2)} £", help=hyp["formules"]["cout_alerte_livres"])

    budget = lire("reports/mesures/budget_validation.json")
    test = lire("reports/mesures/test.json")["modeles"]
    modeles = st.multiselect("Modèles", list(budget["modeles"]), default=PAR_DEFAUT)
    mesure = st.radio("Mesure", list(MESURES), format_func=MESURES.get, horizontal=True)
    donnees = {"alertes_par_jour": budget["alertes_par_jour"], "series": series(budget, cout_alerte)}
    points_test = {}
    if mesure in ("rappel", "precision"):
        points_test = {n: test[n]["seuil"]["k_premieres_par_jour"][mesure] for n in modeles}
    fig = courbes_budget(donnees, modeles, mesure, k, points_test)
    fig.update_yaxes(title=MESURES[mesure], rangemode="tozero")
    st.plotly_chart(mise_en_page(fig, f"{MESURES[mesure]} selon le budget d'alertes"), width="stretch")
    st.caption(f"Courbes calculées sur la **validation** ({budget['jours_calendrier']} jours), règle : "
               f"{budget['regle']}. Losanges : **test, ouvert une seule fois**, à 140 alertes par jour "
               "(lus dans `reports/mesures/test.json`, rien n'est recalculé sur le test ; le coût de cette règle "
               "n'y est pas enregistré).")

    i = budget_le_plus_proche(budget["alertes_par_jour"], k)
    st.subheader(f"Au budget de {budget['alertes_par_jour'][i]} alertes par jour (validation)")
    st.dataframe(pd.DataFrame({
        nom: {"Rappel": fr(s["rappel"][i], 3), "Précision": fr(s["precision"][i], 3),
              "Coût total (M£)": fr(s["cout"][i], 1), "Coût sans Over-Invoicing (M£)": fr(s["cout_sans_over_invoicing"][i], 1)}
        for nom, s in donnees["series"].items() if nom in modeles}).T, width="stretch")
    st.info("Le coût est dominé par quelques très gros montants (`Over-Invoicing`) : il se lit avec le rappel, la "
            "précision et le coût sans cette typologie.")
    pied_de_page()
