# Journal des choix

Chaque décision indique : le **choix**, la **raison**, les **mesures** qui la
justifient et l'**alternative écartée**. Une décision « à mesurer » sera
complétée quand l'étape correspondante aura produit ses chiffres.

---

## 1. Jeu de données : SAML-D (n° 11 du catalogue)

- **Choix** : *Anti Money Laundering Transaction Data (SAML-D)*, Kaggle
  `berkanoztas/synthetic-transaction-monitoring-dataset-aml`, version 2.
- **Raison** : la colonne `Laundering_type` donne la typologie de chaque
  transaction, ce qui permet de mesurer directement quelles typologies les
  méthodes non supervisées détectent ou manquent (analyse demandée).
- **Mesures** (vérification du 30/10/2026) :
  - un fichier `SAML-D.csv`, 996 168 850 octets décompressé, archive ZIP de
    202 528 964 octets ;
  - 9 504 852 transactions, 0,1039 % suspectes (description officielle,
    à recompter à l'étape 1) ;
  - 12 colonnes : `Time, Date, Sender_account, Receiver_account, Amount,
    Payment_currency, Received_currency, Sender_bank_location,
    Receiver_bank_location, Payment_type, Is_laundering, Laundering_type` ;
  - 28 typologies (11 normales, 17 suspectes) ;
  - licence CC BY-NC-SA 4.0 ;
  - téléchargement anonyme testé : la requête de téléchargement Kaggle
    renvoie une redirection 302 vers l'archive, sans jeton.
- **Alternative écartée** : n° 12 IBM AML. Aucun problème bloquant n'a été
  trouvé sur SAML-D, et IBM AML ne fournit pas la typologie ligne par ligne
  dans le même fichier.

## 2. `Laundering_type` exclue des variables d'entrée

- **Choix** : cette colonne ne sert qu'à l'analyse par typologie, jamais à
  l'apprentissage ni au score.
- **Raison** : ses valeurs révèlent l'étiquette (`Normal_Cash_Deposits`,
  `Normal_Fan_Out`, … pour les transactions normales). L'utiliser serait une
  fuite de données.
- **Mesures** : lecture des premières lignes du fichier (toutes les
  transactions normales ont un type préfixé `Normal_`). Un test automatique
  vérifiera qu'elle n'entre dans aucun modèle.
- **Alternative écartée** : aucune ; c'est une contrainte de validité.

## 3. Licences

- **Choix** : code sous licence MIT ; données et extraits dérivés
  (`data/demo/`) sous CC BY-NC-SA 4.0, avec citation de l'article
  d'Oztas et al. (IEEE ICEBE 2023).
- **Raison** : la clause *ShareAlike* impose de garder la licence d'origine
  pour tout extrait redistribué ; la clause *NonCommercial* est compatible
  avec un projet d'école.
- **Alternative écartée** : une seule licence pour tout le dépôt (impossible
  à cause de la clause *ShareAlike*).

## 4. Approche centrale non supervisée

- **Choix** : Isolation Forest et une deuxième méthode apprennent **sans
  étiquettes**. Les étiquettes servent uniquement à évaluer et à choisir le
  seuil sur la validation.
- **Raison** : consigne de l'enseignante ; dans la réalité, les cas de
  blanchiment avérés sont rares, tardifs et incomplets.
- **Alternative écartée** : entraîner seulement sur les transactions
  « normales » (détection de nouveauté) : il faudrait les étiquettes pour les
  trier.

## 5. Deux versions des méthodes non supervisées (précision 1)

- **Choix** : présenter deux versions de chaque méthode non supervisée :
  - **version « sans étiquette »** : réglages par défaut fixés à l'avance
    (par exemple Isolation Forest de scikit-learn avec ses paramètres par
    défaut), deuxième méthode choisie sans regarder les étiquettes ;
  - **version « réglée »** : réglages et deuxième méthode choisis par la
    PR-AUC de validation.
- **Raison** : choisir des réglages avec la PR-AUC de validation utilise déjà
  les étiquettes. Mesurer l'écart montre ce que ce réglage apporte, et donc
  ce qu'on perdrait dans une situation réellement sans étiquette.
- **Mesures** : à compléter (PR-AUC, précision@k et rappel@k des deux
  versions, sur la validation puis sur le test).
- **Alternative écartée** : présenter seulement la version réglée, ce qui
  surestimerait la performance d'une approche « non supervisée ».

## 6. Deuxième méthode non supervisée

- **Choix** : à mesurer entre LOF et un autoencodeur (MLP de scikit-learn).
- **Raison** : ces deux méthodes donnent un score continu et restent légères.
  Un autoencodeur scikit-learn évite PyTorch, trop lourd pour les 512 Mo de
  Render.
- **Mesures** : à compléter (temps d'entraînement, mémoire, PR-AUC de
  validation).
- **Alternatives écartées** : One-Class SVM (son coût de calcul grandit très
  vite avec le nombre de lignes) ; DBSCAN (pas de score continu, seulement
  une étiquette « bruit »).

## 7. Calcul des variables avant l'échantillonnage (précision 2)

- **Choix** : les variables comportementales (nombre de transactions sur 1 h
  et 24 h, montant rapporté à la moyenne des 30 derniers jours, temps depuis
  la transaction précédente) sont calculées sur **l'historique complet** de
  chaque compte, en n'utilisant que le passé. L'échantillonnage vient
  **après**, ou se fait par comptes entiers ou par période, jamais ligne par
  ligne.
- **Raison** : un échantillon ligne par ligne retirerait des transactions de
  l'historique des comptes, ce qui fausserait les compteurs et les moyennes.
- **Mesures** : à compléter (mémoire nécessaire mesurée **avant** de
  choisir : pic de mémoire et temps, pandas comparé à polars).
- **Alternative écartée** : échantillonnage aléatoire des lignes avant le
  calcul des variables.

## 8. Séparation chronologique 60 / 20 / 20

- **Choix** : entraînement sur les 60 % les plus anciens, validation sur les
  20 % suivants, test sur les 20 % les plus récents. Le test ne sert qu'une
  fois, à la fin.
- **Raison** : consigne du sujet ; reproduit l'usage réel (on apprend sur le
  passé, on détecte dans le futur).
- **Alternative écartée** : séparation aléatoire stratifiée (réservée par le
  sujet aux jeux sans information temporelle).

## 9. Coût d'analyse et budget k (précision 3)

- **Choix** : le coût d'une alerte et le budget k sont calculés à partir
  d'**hypothèses écrites et réglables** (dans le code et par curseurs dans
  l'application) :
  - coût d'une alerte = temps d'analyse d'une alerte × tarif horaire d'un
    analyste ;
  - budget k = nombre d'analystes × alertes traitées par analyste et par
    jour, rapporté à la période évaluée.
- **Raison** : le seuil retenu dépend directement de ces valeurs ; elles
  doivent être visibles et discutables plutôt que cachées.
- **Mesures** : à compléter (valeurs retenues, sources, et sensibilité du
  seuil à ces hypothèses).
- **Alternative écartée** : un coût fixe arbitraire sans justification.

## 10. Environnement Python et versions figées

- **Choix** : Python 3.11 dans `.venv` ; toutes les versions figées, y compris
  les dépendances indirectes importantes (scipy, pyarrow 24.0.0, numba…).
  Trois fichiers :
  - `requirements.txt` : application (Streamlit Cloud) ;
  - `requirements-api.txt` : API (Render), sans Streamlit ni Plotly ;
  - `requirements-dev.txt` : les deux, plus Jupyter, pytest, polars, psutil.
- **Raison** : leçon du projet 1 (une version non figée a fait planter
  l'application sur Streamlit Cloud) ; un fichier léger pour l'API économise
  de la mémoire sur Render.
- **Mesures** : `pip check` → « No broken requirements found » (30/10/2026).
  Mémoire de l'API à mesurer à l'étape 8.
- **Alternative écartée** : un seul fichier de dépendances pour tout (plus
  lourd sur Render et Streamlit Cloud).

## 11. Organisation du code et du notebook

- **Choix** : code partagé dans `src/aml_xai/`, installé par `-e .` et utilisé
  par le notebook, l'application et l'API. Un seul notebook, découpé en
  sections, une par module du sujet.
- **Raison** : un seul endroit pour chaque calcul, donc des chiffres
  identiques partout ; un seul notebook est plus simple à présenter.
- **Alternative écartée** : un notebook par module.
