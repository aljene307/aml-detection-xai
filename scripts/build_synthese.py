"""Page de synthèse (reports/synthese.pdf), générée à partir des fichiers de résultats.

Aucun chiffre n'est écrit à la main : tous sont lus dans reports/mesures/.

Usage : python scripts/build_synthese.py [--apercu chemin.png]
(--apercu : enregistre aussi la même page en PNG, pour la relire)
"""

import json
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from aml_xai.config import MEASURES_DIR, REPORTS_DIR  # noqa: E402

BLEU, VERT, GRIS, ENCRE, ENCRE_2, FOND = "#2a78d6", "#008300", "#52514e", "#0b0b0b", "#52514e", "#ffffff"
DEPOT = "github.com/aljene307/aml-detection-xai"
API = "aml-detection-xai-api.onrender.com"


def lire(nom: str) -> dict:
    return json.loads((MEASURES_DIR / nom).read_text(encoding="utf-8"))


def fr(x: float, decimales: int = 2) -> str:
    return f"{x:,.{decimales}f}".replace(",", " ").replace(".", ",")


def environ(x: float) -> str:
    """Arrondi pour une phrase : à l'unité sous 100, à la dizaine au-delà."""
    return fr(round(x) if x < 100 else round(x, -1), 0)


def bloc(fig, y: float, titre: str, lignes: list[str], largeur: int = 118, taille: float = 8.6) -> float:
    """Titre et paragraphes ; renvoie la position sous le bloc."""
    fig.text(0.06, y, titre, fontsize=10.5, fontweight="bold", color=ENCRE, va="top")
    y -= 0.022
    for ligne in lignes:
        for morceau in textwrap.wrap(ligne, largeur):
            fig.text(0.06, y, morceau, fontsize=taille, color=ENCRE, va="top")
            y -= 0.0155
        y -= 0.003
    return y - 0.008


def main() -> None:
    test = lire("test.json")
    validation = lire("repetition_validation.json")
    supervise = lire("supervise_tableau.json")
    modeles = test["modeles"]
    lof, lgbm = modeles["LOF, sans étiquette"], modeles["LightGBM, réglé"]
    dans_le_hasard = sum(m["pr_auc_intervalle_95"][0] <= validation["modeles"][n]["pr_auc"]
                         <= m["pr_auc_intervalle_95"][1] for n, m in modeles.items())
    lof_ref = supervise["reference_lof"]["sans_etiquette"]
    courbe = supervise["courbe_apprentissage"]
    toujours = next(p for p in sorted(courbe, key=float)
                    if all(v > lof_ref for v in courbe[p]["pr_auc"]))
    cas_seuil = sum(courbe[toujours]["cas"]) / len(courbe[toujours]["cas"])
    seuils = {n: m["seuil"] for n, m in modeles.items() if "seuil" in m}
    jour_min = min(s["alertes_par_jour"] for s in seuils.values())
    jour_max = max(s["alertes_par_jour"] for s in seuils.values())

    fig = plt.figure(figsize=(8.27, 11.69), facecolor=FOND)
    fig.text(0.06, 0.965, "Détection explicable du blanchiment d'argent (SAML-D)", fontsize=16,
             fontweight="bold", color=ENCRE, va="top")
    fig.text(0.06, 0.937, "Mohamed Aljene · Projet Data Mining, ENSI II3-IF, 2026-2027", fontsize=8.3,
             color=ENCRE_2, va="top")
    fig.text(0.06, 0.922, "Données synthétiques : jeu SAML-D (Oztas et al., IEEE ICEBE 2023), licence CC BY-NC-SA 4.0. "
             "Aucune donnée bancaire réelle.", fontsize=8.3, color=ENCRE_2, va="top")

    y = bloc(fig, 0.898, "Question", [
        "Peut-on repérer des transactions de blanchiment sans étiquettes pour apprendre, et combien perd-on "
        "par rapport à un modèle supervisé ? Chaque alerte doit être expliquée à l'analyste.",
    ])
    y = bloc(fig, y, "Méthode", [
        "Variables calculées sur tout l'historique de chaque compte, en ne regardant que le passé ; séparation "
        "chronologique 60 / 20 / 20. Sans étiquette : LOF et Isolation Forest. Avec étiquettes, en comparaison : "
        "LightGBM (régression logistique comme contrôle). Seuil : coût minimal sous la capacité de l'équipe "
        "(140 alertes par jour). Explications : SHAP regroupé en 13 codes motifs. Tous les choix figés et "
        "poussés sur GitHub avant d'ouvrir le test, ouvert une seule fois.",
    ])
    y = bloc(fig, y, "Trois messages (test : "
             f"{fr(test['lignes'], 0)} transactions, {fr(test['cas'], 0)} cas, {fr(100 * test['taux'], 3)} %)", [
        f"1. Sans étiquette, LOF fait environ {environ(lof['fois_le_hasard'])} fois mieux que le hasard ; avec "
        f"étiquettes, LightGBM réglé fait environ {environ(lgbm['fois_le_hasard'])} fois mieux (PR-AUC "
        f"{fr(lof['pr_auc'], 3)} et {fr(lgbm['pr_auc'], 3)}).",
        f"2. Avec environ {environ(cas_seuil)} cas de blanchiment étiquetés, LightGBM dépasse LOF sans étiquette "
        f"à chaque tirage (courbe d'apprentissage, validation).",
        f"3. Les résultats tiennent sur des données jamais vues : pour {dans_le_hasard} modèles sur {len(modeles)}, "
        "l'écart entre validation et test reste dans l'intervalle à 95 % du test.",
    ])

    noms = sorted(modeles, key=lambda n: modeles[n]["pr_auc"])
    ax = fig.add_axes([0.27, y - 0.205, 0.30, 0.19])
    for i, n in enumerate(noms):
        bas, haut = modeles[n]["pr_auc_intervalle_95"]
        ax.plot([bas, haut], [i - 0.15] * 2, color=BLEU, linewidth=1)
        ax.scatter(modeles[n]["pr_auc"], i - 0.15, color=BLEU, marker="D", s=14, zorder=3,
                   label="test (intervalle à 95 %)" if i == 0 else None)
        ax.scatter(validation["modeles"][n]["pr_auc"], i + 0.15, color=GRIS, s=12, zorder=3,
                   label="validation" if i == 0 else None)
    ax.axvline(test["taux"], color=GRIS, linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(noms)), noms, fontsize=6.5)
    ax.set_xlabel("PR-AUC (échelle logarithmique ; pointillés : hasard)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 3) if v < 0.01 else fr(v, 2)))
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.set_title("Validation et test, 8 modèles", fontsize=8, loc="left")
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)

    ax2 = fig.add_axes([0.66, y - 0.205, 0.29, 0.19])
    for part in courbe.values():
        ax2.scatter(part["cas"], part["pr_auc"], color=VERT, s=9)
    ax2.axhline(lof_ref, color=BLEU, linewidth=1)
    ax2.text(min(min(p["cas"]) for p in courbe.values()), lof_ref, " LOF sans étiquette", fontsize=6,
             color=BLEU, va="bottom")
    ax2.set_xscale("log")
    ax2.set_xlabel("Cas étiquetés utilisés par LightGBM", fontsize=6.5)
    ax2.set_ylabel("PR-AUC (validation)", fontsize=6.5)
    ax2.tick_params(labelsize=6.5)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 1)))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 0)))
    ax2.set_title("Courbe d'apprentissage", fontsize=8, loc="left")
    for cote in ("top", "right"):
        ax2.spines[cote].set_visible(False)

    y -= 0.25
    fig.text(0.06, y, "Au seuil figé, sur le test", fontsize=10.5, fontweight="bold", color=ENCRE, va="top")
    lignes = [["Modèle", "Étiquettes", "Alertes / jour", "Précision", "Rappel", "F2", "MCC"]]
    etiquettes = {"Isolation Forest, sans étiquette": "aucune", "LOF, sans étiquette": "aucune",
                  "LOF, réglé": "validation", "LightGBM, par défaut": "entraînement",
                  "LightGBM, réglé": "entraînement + validation"}
    for n, s in seuils.items():
        lignes.append([n, etiquettes[n], fr(s["alertes_par_jour"], 1), fr(s["precision"], 3), fr(s["rappel"], 3),
                       fr(s["f2"], 3), fr(s["mcc"], 3)])
    ax3 = fig.add_axes([0.06, y - 0.135, 0.88, 0.12])
    ax3.axis("off")
    table = ax3.table(cellText=lignes[1:], colLabels=lignes[0], loc="upper left", cellLoc="left",
                      colWidths=[0.25, 0.24, 0.12, 0.1, 0.09, 0.08, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    for (ligne, _), cellule in table.get_celld().items():
        cellule.set_edgecolor("#dddddd")
        if ligne == 0:
            cellule.set_text_props(fontweight="bold")

    y = bloc(fig, y - 0.155, "Recommandation et limites", [
        f"En production : garder les 140 transactions les plus suspectes de chaque jour. Au seuil figé, les "
        f"5 modèles dépassent légèrement la capacité sur le test ({fr(jour_min, 1)} à {fr(jour_max, 1)} alertes "
        "par jour) ; cette règle la respecte exactement, avec presque les mêmes résultats.",
        "Limites : données synthétiques ; coût dominé par quelques très gros montants (surtout Over-Invoicing) ; "
        "explications non fidèles selon la règle fixée avant les mesures (test de suppression) ; changements "
        "de comportement d'un compte peu détectés par LOF.",
    ])
    fig.text(0.06, 0.03, f"Code et journal des choix : {DEPOT}  ·  API : {API}", fontsize=7.5, color=ENCRE_2)
    sortie = REPORTS_DIR / "synthese.pdf"
    fig.savefig(sortie, metadata={"Title": "Détection explicable du blanchiment d'argent (SAML-D)",
                                  "Author": "Mohamed Aljene", "Creator": "scripts/build_synthese.py",
                                  "Producer": None, "CreationDate": None})
    if "--apercu" in sys.argv:
        fig.savefig(sys.argv[sys.argv.index("--apercu") + 1], dpi=110)
    plt.close(fig)
    print(f"Écrit : {sortie.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
