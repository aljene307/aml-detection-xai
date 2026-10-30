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
  - 9 504 852 transactions, dont 9 873 suspectes (0,1039 %), recomptées
    sur le fichier téléchargé (`reports/qualite_donnees.json`) ;
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
- **Mesures** (étape 1, voir section 14) : trois variables par compte
  émetteur calculées sur les 9,5 M de lignes en 5 s, avec un pic de 1,8 Go.
  L'historique complet tient donc en mémoire sur le PC de travail : pas
  d'échantillonnage nécessaire pour le calcul des variables. Pic avec toutes
  les variables de l'étape 2 à mesurer.
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

## 12. Téléchargement avec kagglehub, sans jeton

- **Choix** : `kagglehub.dataset_download`, sans identifiants ; requête HTTP
  directe en secours (`aml_xai.data.download_raw`).
- **Raison** : bibliothèque officielle de Kaggle ; le jeu est public, donc
  aucun secret à gérer.
- **Mesures** (`reports/mesures/donnees.json`) : téléchargement anonyme
  réussi en 286 s (débit réseau d'environ 0,7 Mo/s), pic de mémoire 117 Mo,
  taille du CSV conforme (996 168 850 octets).
- **Alternative écartée** : CLI `kaggle`, qui exige un jeton.

## 13. Format Parquet et types de colonnes

- **Choix** : conversion unique du CSV en Parquet (compression zstd), par
  morceaux de 64 Mo : `Date` et `Time` fusionnées en `timestamp`, comptes en
  entiers, colonnes texte en catégories, `transaction_id` pour garder l'ordre
  d'origine. `Is_laundering` sur 1 octet.
- **Raison** : fichier 6,5 fois plus petit et relu en moins d'une seconde ;
  la conversion par morceaux reste possible avec peu de RAM libre (3,3 Go
  libres sur 15,7 Go au moment des mesures).
- **Mesures** (`reports/mesures/donnees.json`, `pandas_polars.json`) :

  | | CSV | Parquet |
  |---|---|---|
  | Taille sur disque | 996 Mo | 154 Mo |
  | Tableau pandas en mémoire | ≈ 1 460 Mo (154 Mo par million de lignes, extrapolé) | 426 Mo |
  | Lecture complète pandas | non mesurée (risque pour la RAM) | 0,45 s, pic 1,6 Go |

  Conversion : 19 s, pic 1,3 Go.
- **Alternative écartée** : relire le CSV à chaque fois.

## 14. Calcul par compte : numpy (`searchsorted`) sur des tableaux pandas

- **Choix** : les fenêtres glissantes par compte (nombre de transactions sur
  1 h et 24 h, moyenne sur 30 jours, temps depuis la précédente) seront
  calculées avec numpy : tri par compte puis par temps, puis recherche
  dichotomique (`np.searchsorted`) du début de chaque fenêtre et sommes
  cumulées. Polars **reste dans `requirements-dev.txt`** : il ne sert qu'aux
  mesures, pas au code de `src/`.
- **Raison** : sept fois plus rapide que pandas, avec des résultats
  **identiques** à pandas ; ordre des lignes maîtrisé (voir ci-dessous) ; pas
  de dépendance supplémentaire pour l'application et l'API.
- **Mesures** (`reports/mesures/pandas_polars.json`, 9,5 M de lignes,
  compte émetteur, 3 variables) :

  | Méthode | Calcul | Pic mémoire | Somme des compteurs 24 h |
  |---|---|---|---|
  | pandas `groupby().rolling()` | 38,4 s | 2,2 Go | 249 826 357 |
  | polars `rolling()` | 1,4 s | 1,1 Go | 249 829 747 |
  | numpy `searchsorted` | 5,1 s | 1,8 Go | 249 826 357 |

  **Écart de polars expliqué** : 3 384 groupes de transactions d'un même
  émetteur à la même seconde (6 771 lignes). Polars compte, pour chaque
  ligne, toutes les transactions de la même seconde, y compris celles qui la
  suivent dans le fichier ; pandas et numpy ne comptent que celles qui la
  précèdent. L'écart attendu (somme de k(k−1)/2 sur les groupes) vaut
  exactement 3 390. Le traitement de ces ex æquo, sans regarder le futur,
  sera fixé à l'étape 2.
- **Alternatives écartées** : polars (le plus rapide et le plus léger, mais
  une dépendance de plus et des ex æquo comptés autrement ; à 5 s contre
  1,4 s, le gain ne compte pas à cette échelle) ; pandas `rolling` (sept fois
  plus lent).

## 15. Contrôles de qualité du jeu complet

- **Mesures** (`reports/qualite_donnees.json`) :
  - aucune valeur manquante, aucun doublon (hors identifiant) ;
  - fichier déjà trié dans l'ordre du temps, du 07/10/2022 10:35 au
    23/08/2023 10:57 (environ 10,5 mois) ;
  - 28 typologies (11 normales, 17 suspectes) ; la typologie détermine
    exactement l'étiquette (fuite confirmée, voir section 2) ;
  - 292 715 comptes émetteurs, 652 266 comptes destinataires ;
  - montants de 3,73 à 12 618 498,40 ;
  - typologies suspectes très inégales : de 1 870 transactions
    (`Structuring`) à 54 (`Over-Invoicing`).
  - Le contrôle des doublons atteint un pic de 3,1 Go : à ne pas relancer
    quand la RAM libre est faible.
- **Conséquence** : le rappel par typologie sera instable pour les plus
  petites typologies (environ 11 cas dans le test pour `Over-Invoicing` si
  elle est répartie uniformément dans le temps, à vérifier à l'étape 2) ;
  il sera présenté avec les effectifs.

## 16. Période utilisée par l'exploration

- **Choix** :
  - chiffres descriptifs globaux (taille, période, taux global) : jeu
    complet ;
  - graphiques qui guident le choix des variables (taux par heure, par jour,
    par type de paiement, par pays, par devise) : **période d'entraînement
    seulement**, c'est-à-dire les premiers 60 % des lignes dans l'ordre du
    temps, jusqu'au 16/04/2023 21:10:37 ;
  - chaque graphique du notebook indique la période qu'il utilise.
- **Raison** : choisir des variables en regardant la validation ou le test
  reviendrait à regarder le futur ; les résultats finaux seraient
  optimistes.
- **Alternative écartée** : explorer tout le jeu.
