"""Graphiques Plotly de l'application (mêmes couleurs que le notebook)."""

import math

import plotly.graph_objects as go

BLEU, ORANGE, VERT, GRIS = "#2a78d6", "#eb6834", "#008300", "#52514e"
COULEUR_METHODE = {"Isolation Forest": GRIS, "LOF": BLEU, "LightGBM": VERT, "Autoencodeur": "#7a5af5",
                   "Régression logistique": "#b0632a"}


def style(nom_modele: str) -> dict:
    methode = next(m for m in COULEUR_METHODE if nom_modele.startswith(m))
    return {"color": COULEUR_METHODE[methode], "dash": "dash" if "réglé" in nom_modele else "solid"}


def trait_vertical_log(fig: go.Figure, x: float, texte: str, tirets: str, cote: str) -> None:
    """Trait vertical sur un axe x logarithmique : le trait se place à la valeur réelle,
    mais son étiquette au logarithme de la valeur (fonctionnement de Plotly)."""
    fig.add_vline(x=x, line_color=GRIS, line_width=1, line_dash=tirets)
    fig.add_annotation(x=math.log10(x), y=1, yref="paper", text=texte, showarrow=False,
                       xanchor=cote, yanchor="bottom", font={"color": GRIS, "size": 11})


def mise_en_page(fig: go.Figure, titre: str, hauteur: int = 420) -> go.Figure:
    fig.update_layout(title={"text": titre, "x": 0, "font": {"size": 16}}, height=hauteur,
                      separators=", ", margin={"l": 10, "r": 10, "t": 60, "b": 10},
                      legend={"orientation": "h", "y": -0.2}, template="plotly_white")
    return fig


def contributions(groupes: dict[str, float], libelles: dict[str, str], titre: str) -> go.Figure:
    """Contribution de chaque groupe : orange vers « suspect », bleu vers « normal »."""
    ordre = sorted(groupes, key=lambda c: abs(groupes[c]))
    valeurs = [groupes[c] for c in ordre]
    fig = go.Figure(go.Bar(
        x=valeurs, y=[libelles[c] for c in ordre], orientation="h",
        marker_color=[ORANGE if v > 0 else BLEU for v in valeurs],
        hovertemplate="%{y} : %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=GRIS, line_width=1)
    fig.update_xaxes(title="Contribution au score (positive = pousse vers suspect)")
    return mise_en_page(fig, titre, 460)


def courbes_budget(budget: dict, modeles: list[str], mesure: str, k: float, points_test: dict) -> go.Figure:
    """Mesure (rappel, précision ou coût) selon le nombre d'alertes par jour, sur la validation."""
    x = budget["alertes_par_jour"]
    fig = go.Figure()
    for nom in modeles:
        fig.add_trace(go.Scatter(x=x, y=budget["series"][nom][mesure], mode="lines", name=nom,
                                 line={**style(nom), "width": 2}))
        if nom in points_test:
            fig.add_trace(go.Scatter(x=[140], y=[points_test[nom]], mode="markers", showlegend=False,
                                     name="test, ouvert une seule fois",
                                     marker={"symbol": "diamond", "size": 11, "color": style(nom)["color"],
                                             "line": {"color": "white", "width": 1.5}},
                                     hovertemplate=f"{nom}<br>test, ouvert une seule fois : %{{y:.3f}}<extra></extra>"))
    trait_vertical_log(fig, k, f"budget choisi : {k:.0f} par jour", "dot", "left")
    fig.update_xaxes(type="log", title="Alertes par jour (échelle logarithmique)")
    return fig


def comparaison_modeles(test: dict, validation: dict) -> go.Figure:
    """PR-AUC de validation et de test, avec l'intervalle à 95 % du test."""
    noms = sorted(test["modeles"], key=lambda n: test["modeles"][n]["pr_auc"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[validation["modeles"][n]["pr_auc"] for n in noms], y=noms, mode="markers",
                             name="validation", marker={"color": GRIS, "size": 9}))
    bas = [test["modeles"][n]["pr_auc"] - test["modeles"][n]["pr_auc_intervalle_95"][0] for n in noms]
    haut = [test["modeles"][n]["pr_auc_intervalle_95"][1] - test["modeles"][n]["pr_auc"] for n in noms]
    fig.add_trace(go.Scatter(x=[test["modeles"][n]["pr_auc"] for n in noms], y=noms, mode="markers",
                             name="test (intervalle à 95 %)", marker={"color": BLEU, "size": 10, "symbol": "diamond"},
                             error_x={"type": "data", "array": haut, "arrayminus": bas, "color": BLEU}))
    trait_vertical_log(fig, test["taux"], "hasard", "dash", "left")
    fig.update_xaxes(type="log", title="PR-AUC (échelle logarithmique)")
    return mise_en_page(fig, "Chaque modèle garde son niveau de validation sur le test", 460)


def courbe_apprentissage(courbe: dict, lof: float) -> go.Figure:
    """PR-AUC de LightGBM par défaut selon le nombre de cas étiquetés utilisés."""
    fig = go.Figure()
    cas, pr = [], []
    for part in courbe.values():
        cas += part["cas"]
        pr += part["pr_auc"]
    fig.add_trace(go.Scatter(x=cas, y=pr, mode="markers", name="LightGBM par défaut (un tirage)",
                             marker={"color": VERT, "size": 8}))
    fig.add_hline(y=lof, line_color=BLEU, annotation_text="LOF sans étiquette", annotation_position="top left")
    fig.update_xaxes(type="log", title="Cas de blanchiment étiquetés utilisés (échelle logarithmique)")
    fig.update_yaxes(title="PR-AUC de validation", rangemode="tozero")
    return mise_en_page(fig, "Avec environ 500 cas étiquetés, LightGBM dépasse LOF sans étiquette")


def rappel_typologies(test: dict) -> go.Figure:
    """Rappel par typologie au seuil figé, sur le test, pour LOF et LightGBM réglé."""
    lgbm = test["modeles"]["LightGBM, réglé"]["seuil"]["typologies"]
    lof = test["modeles"]["LOF, sans étiquette"]["seuil"]["typologies"]
    ordre = sorted(lgbm, key=lambda t: lgbm[t]["rappel"])
    fig = go.Figure()
    for nom, table, couleur in [("LOF, sans étiquette", lof, BLEU), ("LightGBM, réglé", lgbm, VERT)]:
        fig.add_trace(go.Bar(y=[f"{t} ({table[t]['cas']} cas)" for t in ordre], x=[table[t]["rappel"] for t in ordre],
                             orientation="h", name=nom, marker_color=couleur))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Rappel au seuil figé (test)", range=[0, 1])
    return mise_en_page(fig, "Rappel par typologie sur le test", 640)
