# Détection explicable du blanchiment d'argent (SAML-D)

Projet Challenge 2 — Data Mining, ENSI, filière II3-IF, 2026-2027.

Détection de transactions de blanchiment par des méthodes **non supervisées**
(Isolation Forest et une deuxième méthode), comparées à un modèle supervisé
de référence, avec seuil fixé par un coût métier, explications SHAP,
application Streamlit et API FastAPI.

> Projet en cours de construction. Les décisions et leurs justifications sont
> dans [docs/CHOIX.md](docs/CHOIX.md).

## Installation (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Données

- Jeu : *Anti Money Laundering Transaction Data (SAML-D)*, Berkan Oztas,
  [Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml).
- Licence : [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Tout extrait commité dans `data/demo/` reste sous cette licence.
- Référence : B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan et G. Aksu,
  « Enhancing Anti-Money Laundering: Development of a Synthetic Transaction
  Monitoring Dataset », IEEE ICEBE 2023, pp. 47-54,
  doi:[10.1109/ICEBE59045.2023.00028](https://doi.org/10.1109/ICEBE59045.2023.00028).

## Licence

Code sous licence MIT (voir [LICENSE](LICENSE)).
