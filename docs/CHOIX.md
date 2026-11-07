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
- **Mesures sur la validation** (`reports/mesures/grilles.jsonl`, un réglage
  par ligne ; `grilles_tableau.json`). Hasard = taux de la validation,
  0,00105 :

  | Méthode | Version | Réglage | PR-AUC | ROC-AUC |
  |---|---|---|---|---|
  | Isolation Forest | sans étiquette | 100 arbres, 256 lignes, 100 % des variables | 0,0042 | 0,771 |
  | Isolation Forest | réglée (12 essais) | 100 arbres, 4 096 lignes, 50 % des variables | 0,0052 | 0,808 |
  | LOF | sans étiquette | 20 voisines, 100 000 lignes | 0,0330 | 0,828 |
  | LOF | réglée (8 essais) | 10 voisines, 100 000 lignes | 0,0444 | 0,834 |
  | Autoencodeur | sans étiquette | goulot de 8 | 0,0151 | 0,847 |
  | Autoencodeur | réglée (3 essais) | goulot de 8 (déjà le meilleur) | 0,0151 | 0,847 |

  Graines d'Isolation Forest (5 graines) : sans étiquette 0,0028 à 0,0042
  (moyenne 0,0034, écart type 0,0005) ; réglée 0,0041 à 0,0052 (moyenne
  0,0049, écart type 0,0004). Les plages se recouvrent à peine : gain réel
  mais minuscule. La graine 42 est par hasard la meilleure des cinq pour la
  version par défaut.
- **Constats** : les meilleurs réglages sont au bord des grilles (10 voisines,
  4 096 lignes par arbre) ; la grille n'est pas agrandie après coup, pour ne
  pas chercher un bon score par chance sur la validation. LOF fait mieux avec
  100 000 lignes qu'avec 50 000 pour les 4 nombres de voisines ; 200 000
  dépasserait la limite de notation. Les durées varient du simple au double
  avec la charge du PC (même réglage de LOF : 300 s dans la grille, 613 s
  pendant la faisabilité).
- **Vérification** : les réglages par défaut recalculés dans la grille
  redonnent exactement les scores de l'essai de faisabilité (0 ligne
  différente pour les trois méthodes).
- **Contrôles du réglage** : section 23.
- **À compléter** : précision@k et rappel@k (étape 5), test (étape 7).
- **Alternative écartée** : présenter seulement la version réglée, ce qui
  surestimerait la performance d'une approche « non supervisée ».

## 6. Deuxième méthode non supervisée

- **Choix** : **LOF** (Local Outlier Factor, 20 voisines), choisi par une règle
  de coût sans étiquettes, entre LOF et un autoencodeur (MLP de scikit-learn,
  23 → 8 → 23).
- **Raison** : ces deux méthodes donnent un score continu et restent légères.
  Un autoencodeur scikit-learn évite PyTorch, trop lourd pour les 512 Mo de
  Render. Choisir avec la PR-AUC de validation utiliserait les étiquettes :
  pour la version « sans étiquette », le choix repose donc sur le coût.
- **Règle fixée avant les mesures** (`scripts/train_unsupervised.py`) : une
  méthode est faisable si elle apprend en moins de 15 min, note toute la
  validation (1,9 M de lignes) en moins de 15 min, reste sous 1,5 Go de pic
  et donne un modèle de moins de 100 Mo. Si les deux sont faisables, LOF (un
  seul réglage principal, alors que l'autoencodeur oblige à inventer une
  architecture) ; sinon, celle qui est faisable.
- **Protocole** : versions par défaut ; LOF et l'autoencodeur apprennent sur
  le même échantillon de 100 000 lignes tirées au hasard (graine 42), ce qui
  ne fausse aucune variable puisqu'elles sont calculées sur l'historique
  complet (section 7) ; Isolation Forest apprend sur tout l'entraînement
  retenu. Un processus par méthode, pour un pic de mémoire propre.
- **Mesures** (`reports/mesures/faisabilite*.json`) :

  | Méthode | Lignes apprises | Apprentissage | Notation validation | Pic | Modèle |
  |---|---|---|---|---|---|
  | Isolation Forest | 4 836 832 | 3,7 s | 13,9 s | 1 201 Mo | 1,3 Mo |
  | LOF | 100 000 | 23,4 s | **612,9 s** | 501 Mo | 17,2 Mo |
  | Autoencodeur | 100 000 | 9,0 s | 0,8 s | 435 Mo | < 0,1 Mo |

  Les trois sont faisables ; la règle donne **LOF**. LOF passe de justesse sur
  la notation et est environ 750 fois plus lent que l'autoencodeur : la règle
  ne départageait pas sur la vitesse, et elle n'est pas modifiée après coup.
  Pour l'API, noter une transaction reste instantané.
- **Version réglée** (section 5) : petites grilles sur la PR-AUC de
  validation, pour LOF (10, 20, 50 ou 100 voisines ; échantillon de 50 000
  ou 100 000 lignes, 200 000 dépasserait la limite de notation) et pour
  l'autoencodeur (goulot de 4, 8 ou 12). Résultat : la version réglée
  choisit aussi LOF (PR-AUC 0,0444 contre 0,0151 pour le meilleur
  autoencodeur), avec 10 voisines (section 5).
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
  d'échantillonnage nécessaire pour le calcul des variables.
- **Mesures de l'étape 2** (`reports/mesures/variables.json`, 9,5 M de lignes,
  toutes les variables) :

  | Calcul | Durée | Pic mémoire |
  |---|---|---|
  | Tout le jeu chargé, un seul morceau | 48 s | 3 291 Mo |
  | Tout le jeu chargé, découpé en 4 morceaux | 32 s | 2 242 Mo |
  | 2 groupes de comptes lus un par un | 28 s | 2 107 Mo |
  | **8 groupes de comptes lus un par un (retenu)** | 47 s | **1 384 Mo** |

  Pour tenir dans environ 1,5 Go pendant que le navigateur reste ouvert, le
  calcul retenu ne lit que les lignes du groupe de comptes en cours (filtre à
  la lecture du Parquet sur les 3 derniers bits du numéro de compte) et libère
  la mémoire entre deux groupes. Les variables de la transaction sont
  calculées par lots d'un million de lignes, sur les seules colonnes utiles.
  **Résultat identique au calcul complet** : 0 ligne différente sur les
  17 colonnes et les 9 504 852 lignes (`reports/mesures/morceaux_reel.json`
  dans le commit `b2fc38f`). Après le plafond (section 21), ce fichier contient
  la comparaison du calcul en 2 groupes avec le calcul en 8 groupes : 0 ligne
  différente sur les 20 colonnes.
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

## 9. Coût d'analyse, seuil et budget k (précision 3)

- **Coût (sujet, module 4)**, en livres sterling, monnaie des montants du jeu :
  somme des montants des blanchiments non détectés + nombre d'alertes × coût
  d'une alerte. Une alerte est une transaction dont le score atteint le seuil.
- **Hypothèses écrites et réglables** (`config/hypotheses.json`, lues par
  `src/aml_xai/evaluation.py`, puis par curseurs dans l'application) :
  5 analystes, 7 h par jour, 15 min par alerte, 30 £ de l'heure.
  - alertes par jour = analystes × heures par jour × 60 ÷ minutes par alerte
    = 5 × 7 × 60 ÷ 15 = **140** ;
  - coût d'une alerte = 15 ÷ 60 × 30 = **7,50 £**.
  Une première proposition (40 alertes de 20 min par analyste) dépassait
  13 h de travail par jour : k est maintenant déduit de la formule.
- **Règle fixée avant les mesures** : seuil choisi sur la validation. Le seuil
  **retenu** est celui de coût minimal parmi ceux qui respectent la capacité
  (140 × 64,04 jours = 8 966 alertes) ; le seuil de coût minimal sans
  contrainte et le seuil au budget sont montrés pour comparaison. Un seul
  seuil (valeur de score), fixé sur la validation et confirmé sur le test ;
  « les k premières par jour » en complément seulement. Au seuil retenu :
  précision, rappel, F2 (le rappel compte deux fois plus que la précision)
  et MCC (corrélation entre alertes et vrais cas, de −1 à 1).
- **Mesures** (validation, `reports/mesures/evaluation.json`) : 1 988 cas,
  102,8 M£ blanchis, soit 51 735 £ par cas en moyenne.

  | Méthode | Coût minimal sans contrainte : alertes par jour | Seuil retenu : coût total | Précision | Rappel | F2 | MCC | Coût sans Over-Invoicing |
  |---|---|---|---|---|---|---|---|
  | Isolation Forest, sans étiquette | 9 453 | 98,5 M£ | 0,008 | 0,035 | 0,020 | 0,014 | 24,5 M£ |
  | LOF, sans étiquette | 7 303 | **20,2 M£** | 0,067 | 0,302 | 0,178 | 0,141 | 15,4 M£ |
  | LOF, réglé | 4 466 | 20,3 M£ | 0,076 | 0,342 | 0,201 | 0,160 | 13,6 M£ |
  | LightGBM, par défaut | 16 699 | 37,4 M£ | 0,128 | 0,566 | 0,336 | 0,267 | 11,5 M£ |
  | LightGBM, réglé | 3 882 | 23,1 M£ | **0,150** | **0,677** | **0,398** | **0,318** | **10,3 M£** |

  - **Aucun seuil de coût minimal ne respecte la capacité** : il faudrait de
    3 900 à 16 700 alertes par jour, 28 à 120 fois les 140 possibles. Un cas
    manqué coûte en moyenne 51 735 £ contre 7,50 £ par alerte : alerter reste
    rentable tant qu'environ une alerte sur 7 000 est un vrai cas. Le seuil
    retenu est donc fixé par la capacité (environ 140 alertes par jour pour
    les cinq méthodes).
  - **Le coût est dominé par 18 cas** : `Over-Invoicing` représente 75 % du
    montant blanchi de la validation. LOF en trouve 16 sur 18 (des montants
    au-delà du plafond des transactions normales sont rares, donc
    anormaux) ; LightGBM par défaut 9, réglé 12. D'où le paradoxe : au seuil
    retenu, LOF sans étiquette a le coût le plus bas, avec deux fois moins de
    cas trouvés que LightGBM. Sans ces 18 cas, le classement par le coût
    rejoint celui de la PR-AUC : LightGBM réglé 10,3 M£, LOF 15,4 M£.
  - **Les k premières par jour** donnent presque les mêmes résultats que le
    seuil unique (LightGBM réglé : précision 0,148, rappel 0,676).
- **Limites** : le montant d'une transaction est une approximation du coût
  réel d'un blanchiment manqué, qui comprend aussi des amendes et un risque
  d'image. La monnaie des montants est supposée être la livre sterling. Le
  coût est très sensible à une poignée de gros montants : il faut toujours le
  lire avec la PR-AUC, le F2, le MCC et le coût sans `Over-Invoicing`.
- **Seuil retenu : coût minimal sous la contrainte de capacité, et la
  capacité est toujours atteinte** : pour les cinq méthodes, la contrainte
  est active. Le seuil retenu se place au plus près des 140 alertes par jour
  (de 137,8 à 139,8) : il s'arrête sur le dernier cas trouvé avant la limite,
  car les alertes suivantes ne seraient que des fausses alertes à 7,50 £.
- **Décision : le coût du sujet reste la mesure principale**, avec le coût
  sans `Over-Invoicing` comme sensibilité. **Pas de coût plafonné par cas** :
  il serait choisi après avoir vu son effet sur le classement, ce que la
  méthode interdit.
- **Alternatives écartées** : un coût fixe par cas manqué sans source ; un
  coût plafonné par cas (voir ci-dessus).

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
  exactement 3 390. Le traitement de ces ex æquo est fixé en section 18.
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
  petites typologies ; il sera présenté avec les effectifs. Vérification de
  l'étape 2 (`reports/effectifs_periodes.json`) : `Over-Invoicing` compte
  26 cas à l'entraînement, 18 à la validation et **10 au test** ; la plus
  petite typologie suivante, `Bipartite`, en compte 43 au test.

## 16. Période utilisée par l'exploration

- **Choix** :
  - chiffres descriptifs globaux (taille, période, taux global) : jeu
    complet ;
  - graphiques qui guident le choix des variables (taux par heure, par jour,
    par type de paiement, par pays, par devise), ainsi que les montants et
    les typologies : **période d'entraînement seulement**, c'est-à-dire les
    lignes strictement antérieures au 16/04/2023 21:10:37 (5 702 911 lignes,
    60,0 %) ;
  - exception validée, le graphique du taux par semaine, demandé par le
    module 1 : jeu complet, comme simple description de la stabilité entre les
    trois périodes. **Aucune variable n'en est tirée**, et il ne sert à choisir
    aucun réglage. Le rythme qu'il montre est mesuré à part sur la période
    d'entraînement (section 17) ;
  - chaque graphique du notebook indique la période qu'il utilise.
- **Raison** : choisir des variables en regardant la validation ou le test
  reviendrait à regarder le futur ; les résultats finaux seraient
  optimistes.
- **Mesures** : charger seulement cette période et les colonnes utiles
  atteint un pic de 831 Mo (essai du 02/11/2026), contre 1,6 Go pour le jeu
  complet.
- **Alternative écartée** : explorer tout le jeu.

## 17. Constats de l'exploration (période d'entraînement)

Les chiffres sont calculés dans la section 1 du notebook. Ce sont des
hypothèses pour le choix des variables, pas des résultats de modèle.

- **Plafond des montants** : aucune transaction normale ne dépasse
  999 946,47, alors que 26 cas de blanchiment vont au-delà. Ce sont tous les
  cas `Over-Invoicing` de la période, tous au-dessus de 1,26 M. **Noté comme
  artefact du générateur**, à rappeler dans l'analyse par typologie : une bonne
  détection de cette typologie viendrait du montant seul et ne prouverait pas
  la qualité de la méthode.
- **Rythme mensuel** : le taux par jour du mois va de 0,053 % (jour 31) à
  0,138 % (jour 17), avec des intervalles à 95 % qui ne se recouvrent pas.
  Aucune raison métier évidente, donc probable effet du générateur.
  **Décision validée : pas de variable « jour du mois »**, car elle apprendrait
  le générateur plutôt qu'un comportement de blanchiment.
- **Montants à nombreuses décimales** (constat de l'étape 2, validé) :
  1 071 montants du jeu ont plus de deux décimales, par exemple
  `10753.26726981541`. Sur la période d'entraînement, ces 660 lignes sont
  **toutes du blanchiment** (11,5 % des cas), surtout `Layered_Fan_Out`,
  `Cycle`, `Layered_Fan_In` et `Scatter-Gather`. C'est un **artefact du
  générateur** : **aucune variable n'est tirée de la précision des montants**.
  Dans les sommes de l'historique, les montants sont arrondis au centime, avec
  un écart relatif d'au plus 2,3 millionièmes. À rappeler dans l'analyse par
  typologie.
- **Variables utiles attendues** : type de paiement (taux de 0,05 % à 0,61 %),
  pays et devise du destinataire, devises différentes (0,322 % contre 0,072 %).
- **Redondance** : 99,95 % des transactions entre deux pays sont de type
  `Cross-border`.
- **Signal faible** : heure (de 0,086 % à 0,135 %) ; jour de la semaine (de
  0,095 % à 0,105 %, intervalles qui se recouvrent).
- **Montant seul insuffisant** : 13 typologies suspectes sur 17 ont un montant
  médian à moins d'un facteur 3 de la médiane normale.
- **Pays émetteurs peu fiables** : hors Royaume-Uni (96,7 % des transactions),
  chacun ne compte que 14 à 37 cas. D'où l'intervalle de confiance de Wilson
  sur les graphiques, plus fiable que la formule usuelle pour un taux très
  faible.

## 18. Définition des fenêtres par compte

- **Choix validé** : pour les transactions d'un même compte à la même seconde,
  seules les transactions **strictement antérieures** comptent. Leur ordre dans
  le fichier est arbitraire ; compter l'une d'elles reviendrait à regarder une
  transaction simultanée.
- **Règle unique validée** : toute variable d'historique ne regarde que
  `[t − fenêtre ; t[`, donc ce qui est strictement antérieur à la seconde de la
  transaction. La transaction actuelle et ses ex æquo sont exclus partout, y
  compris de la moyenne des 30 derniers jours.
- **Historique vide, choix validé** : rapport au montant moyen = 1 (valeur
  neutre) et indicateur « sans historique » = 1 quand la fenêtre de 30 jours est
  vide. Le temps depuis la précédente vaut 365 jours (plus que toute la durée du
  jeu) s'il n'existe aucune transaction antérieure ; un compte inactif depuis
  plus de 30 jours garde son vrai écart.
- **Si la mémoire ne suffit pas** : calcul par morceaux de comptes, chaque
  compte gardant tout son historique, et non échantillonnage.

## 19. Codage des colonnes catégorielles

- **Choix validé** : codage par fréquence, appris sur les lignes d'apprentissage
  seulement, sans étiquettes, puis appliqué tel quel à la validation et au test.
  Une valeur jamais vue à l'apprentissage reçoit la fréquence 0.
- **Raison** : aucune étiquette n'est utilisée, et une valeur rare devient un
  petit nombre, ce qui convient à la recherche d'anomalies.
- **Limite** : deux valeurs de même fréquence deviennent presque identiques pour
  le modèle.
- **Contrôle** : la comparaison avec le codage one-hot utilise les étiquettes.
  Elle ne sert donc que de contrôle, comme les réglages de la section 5. **La
  version principale reste le codage par fréquence, quel que soit le résultat
  de cette comparaison.**
- **Résultat du contrôle** (`reports/mesures/one_hot.json`, versions par
  défaut) : one-hot = 69 colonnes 0/1 non standardisées à la place des
  5 fréquences (87 colonnes au lieu de 23). Mêmes lignes d'apprentissage pour
  les deux codages (100 000 pour LOF et l'autoencodeur, 1 million pour
  Isolation Forest, car 87 colonnes sur tout l'entraînement pèseraient
  1,7 Go), mesure sur les mêmes 20 % des comptes émetteurs de la validation
  (379 578 lignes, 387 cas) :

  | Méthode | PR-AUC fréquence | PR-AUC one-hot |
  |---|---|---|
  | LOF | 0,0399 | 0,0303 |
  | Isolation Forest | 0,0038 | 0,0027 |
  | Autoencodeur | 0,0174 | 0,0173 |

  Le codage par fréquence fait au moins aussi bien partout : la limite notée
  plus haut ne coûte rien de mesurable ici. Pic de mémoire du contrôle :
  2 149 Mo, au-dessus de l'objectif de 1,5 Go (calcul allé au bout, sans effet
  sur les résultats ; à réduire s'il doit être relancé).
- **Alternatives écartées** : one-hot comme version principale (environ
  69 colonnes de plus) ; codage par taux de blanchiment (utilise les étiquettes).

## 20. Règles de décision fixées avant les mesures

Ces deux règles ont été écrites avant de calculer les variables de l'étape 2.
Aucune n'utilise les étiquettes.

### Contreparties distinctes (24 h et 30 jours)

Une contrepartie est l'autre compte de la transaction. Chaque variable candidate
est gardée si les trois conditions sont vraies :

1. le calcul complet avec les contreparties tient dans la RAM disponible, en
   morceaux de comptes s'il le faut ;
2. il ajoute moins de 5 minutes au calcul ;
3. sa corrélation de Spearman (corrélation des rangs) avec le compteur de la même
   fenêtre reste sous 0,95 en valeur absolue, sur la période d'entraînement. Au
   delà, elle répéterait le compteur.

### Effet de démarrage

L'indice de stabilité de population (PSI) mesure l'écart entre deux distributions
d'une même variable : 0 si elles sont identiques, et plus de 0,25 est le seuil
usuel d'un changement fort.

1. **Référence** : lignes d'entraînement à partir du 30e jour.
2. **Tranches testées** : jours 0-7, 7-14, 14-21 et 21-30.
3. **Témoins** : tranches de 7 jours prises dans la référence, comparées à la
   référence. Elles donnent, pour chaque variable, la variation normale.
4. **Mesure** : PSI de chaque variable d'historique, sur les déciles de la
   référence, l'absence de valeur formant sa propre classe.
5. **Décision** : une tranche est **différente** si au moins une variable a un
   PSI supérieur à 0,25 **et** supérieur au plus grand PSI de ses témoins. Toutes
   les lignes jusqu'à la fin de la dernière tranche différente sont **exclues de
   l'apprentissage**, mais continuent de nourrir l'historique. Si aucune tranche
   n'est différente, rien n'est exclu et l'indicateur « sans historique » suffit.
6. **Sans retour en arrière** : une comparaison de la détection avec et sans ces
   lignes pourra être affichée comme contrôle. Elle ne change pas la décision.

### Résultats de l'application des deux règles

**Contreparties** (`reports/mesures/contreparties.json`, entraînement,
corrélation sur 1 M de lignes tirées au hasard). Le calcul avec les
contreparties atteint un pic de 1 476 Mo et ajoute 5,5 s.

| Variable | Spearman avec le compteur | Décision |
|---|---|---|
| `em_nb_distincts_24h` | 0,910 | gardée |
| `em_nb_distincts_30j` | 0,927 | gardée |
| `re_nb_distincts_24h` | 0,974 | **écartée** (répète le compteur) |
| `re_nb_distincts_30j` | 0,886 | gardée |

**Démarrage** (`reports/mesures/demarrage.json`) : les quatre tranches du
premier mois sont différentes. Pour les tranches 0-7 et 7-14, plusieurs
variables dépassent le seuil (par exemple `em_nb_distincts_30j` : PSI de 1,24
puis 0,69). Jusqu'au jour 30, `re_ecart_precedent_s` le dépasse aussi (PSI de
0,82 pour la tranche 21-30, contre 0,30 au plus pour les témoins).
**Décision retenue (validée) : les 30 premiers jours sont exclus de
l'apprentissage** (avant le 06/11/2022 10:35:19, soit 866 079 lignes, 15,19 % de
l'entraînement). Ces lignes continuent de nourrir l'historique. Les fréquences
du codage sont apprises sur les 4 836 832 lignes restantes ; aucune valeur de la
validation ou du test n'est inconnue.

- **Ce qu'aurait donné la règle initiale** : appliquée de nouveau après le
  plafond des écarts (section 21), la règle PSI ne trouve plus la tranche 21-30
  différente (plus grand PSI : 0,030, pour `em_nb_distincts_30j`). Elle aurait
  donc exclu **21 jours** seulement.
- **Pourquoi on retient 30 jours** : toutes les fenêtres d'historique durent au
  plus 30 jours (moyenne, contreparties, écart plafonné). Avant le 30e jour,
  elles sont donc incomplètes pour tous les comptes, **par construction**. Ce
  choix découle de la décision du plafond. Il n'utilise ni les étiquettes ni une
  mesure de performance. La mesure va dans le même sens : pour l'écart
  plafonné du destinataire, le plus grand PSI des témoins vaut 0,003 avec une
  référence à partir du jour 30, contre 0,043 à partir du jour 21.
- **Dans le code** : `startup_decision` renvoie le résultat de la règle PSI
  (`jour_regle_psi` = 21) et la décision retenue (`exclure_avant_jour` =
  max(règle, 30) = 30).

## 21. Plafond de 30 jours sur le temps depuis la transaction précédente

- **Problème** : le jeu commence le 07/10/2022, donc au jour 50 aucun écart ne
  peut dépasser 50 jours (censure). La distribution des écarts glisse avec la
  date, et donc entre l'entraînement, la validation et le test, sans aucun
  changement de comportement.
- **Proposition** : écart = min(écart, 30 jours), et 30 jours quand il n'existe
  aucune transaction précédente (au lieu de 365 jours).
- **Raison de la valeur** : 30 jours est la plus longue fenêtre déjà utilisée.
  À partir du jour 30, un écart de 30 jours au plus est observable pour toutes
  les transactions, et l'apprentissage commence justement au jour 30
  (section 20) : l'écart plafonné ne dépend plus de la date. Au-delà de
  30 jours, l'indicateur « sans historique » donne déjà l'information.
- **Règle fixée avant les mesures** : le plafond est gardé si, pour l'émetteur
  et le destinataire, le PSI de l'écart plafonné reste sous 0,10 (seuil usuel
  de stabilité) pour chaque tranche témoin de 7 jours de l'entraînement retenu,
  pour la validation entière et pour chacune de ses semaines. Le test n'est pas
  lu. Les variables ayant changé, la règle de démarrage (section 20) est
  ensuite appliquée de nouveau, telle quelle.
- **Alternatives écartées** : 7 jours (perd l'information entre 7 et
  30 jours) ; logarithme (change l'échelle sans supprimer la censure) ; plafond
  plus long (encore censuré pour une partie de l'apprentissage).
- **Mesures** (`reports/mesures/ecarts.json`, référence = entraînement retenu à
  partir du jour 30, PSI le plus élevé) :

  | Variable | | Témoins | Validation | Semaines de validation |
  |---|---|---|---|---|
  | `re_ecart_precedent_s` | sans plafond | **0,054** | 0,001 | 0,002 |
  | `re_ecart_precedent_s` | plafond 30 j | **0,003** | 0,0003 | 0,001 |
  | `em_ecart_precedent_s` | sans plafond | 0,002 | 0,001 | 0,003 |
  | `em_ecart_precedent_s` | plafond 30 j | 0,002 | 0,001 | 0,003 |

  Avec une référence à partir du jour 21 (règle initiale, section 20), les
  témoins du destinataire valaient 0,273 sans plafond et 0,043 avec.

  **Plafond gardé (validé)** : tous les PSI plafonnés sont sous 0,10. La dérive
  venait des écarts de plus de 30 jours du destinataire. Côté émetteur, rien ne
  change : tous ses déciles sont sous 30 jours. Limite : le PSI par déciles
  voit peu la queue de la distribution.
- **Vérifications** : 53 tests réussis ; sur le jeu réel, le nouveau fichier ne
  diffère de l'ancien que sur les écarts, qui valent exactement min(ancien,
  30 jours) (488 176 lignes plafonnées pour l'émetteur, 1 637 336 pour le
  destinataire, `reports/mesures/plafond_reel.json`) ; calcul en 2 groupes
  identique au calcul en 8 groupes ; recalcul sans lire le futur identique sur
  les 5 702 911 lignes d'entraînement.

**Constat à l'origine de la section 21** : même dans la référence, le temps depuis la
transaction précédente du destinataire varie plus que les autres variables
(PSI des témoins jusqu'à 0,30). Cette variable dépend aussi du temps écoulé
depuis le début du jeu, car un écart ne peut pas remonter avant le 07/10/2022.

## 22. Matrice d'entrée : logarithme puis standardisation

- **Choix** : une même matrice pour toutes les méthodes. `log(1 + x)` sur les
  11 variables à longue traîne (compteurs, contreparties distinctes, rapports
  de montant, écarts), puis standardisation (moyenne 0, écart type 1) de
  toutes les variables. Paramètres appris sur l'entraînement retenu
  (4 836 832 lignes, à partir du 06/11/2022), sans étiquettes, enregistrés dans
  `models/transformation.json`.
- **Raison** : LOF et l'autoencodeur reposent sur des distances et des
  erreurs, donc sur l'échelle des variables. Isolation Forest aussi est
  concerné par la longue traîne : il coupe au hasard entre le minimum et le
  maximum, et presque toutes ses coupes tomberaient dans la zone vide des
  valeurs énormes. Une matrice commune garde la comparaison équitable.
- **Mesures** (`reports/mesures/matrice.json`, asymétrie sur l'entraînement
  retenu) : `re_ratio_montant_30j` 1 178,9 → 3,7 ; `em_ratio_montant_30j`
  513,6 → 2,8 ; `em_ecart_precedent_s` 3,6 → −0,15 ; les compteurs passent de
  1,5 à 2,5 environ à moins de 1,4 (sauf `em_nb_distincts_30j`, déjà peu
  asymétrique : 0,4 → −0,6).
- **Mémoire** : une première version lisait toute la table puis la recopiait
  (pic de 2 690 Mo). La matrice est maintenant construite colonne par colonne
  (lire, transformer, écrire) : pic de 1 320 Mo, matrices identiques à la
  première version (0 valeur différente sur l'entraînement et la validation).
  Les matrices (`data/processed/`, non commitées) sont lues sans tout charger
  en mémoire. La matrice du test n'est pas construite avant l'étape 7.
- **Alternatives écartées** : standardiser sans logarithme (quelques valeurs
  énormes écraseraient les autres) ; une transformation par rangs (plus
  coûteuse à appliquer dans l'API, et elle efface l'écart entre une valeur
  forte et une valeur extrême) ; une matrice différente par méthode
  (comparaison moins équitable).

## 23. Contrôles du réglage et messages pour l'oral

Deux contrôles, sans effet sur les choix, avec des règles fixées avant les
mesures. Le test n'est pas lu.

- **Contrôle 1 : le gain du réglage de LOF dépasse-t-il le hasard du
  tirage ?** LOF n'a pas d'autre hasard que le tirage de son échantillon de
  100 000 lignes. Quatre autres tirages (graines 1 à 4) pour la version par
  défaut (20 voisines) et la version réglée (10 voisines)
  (`reports/mesures/grilles.jsonl`, `grilles_tableau.json`) :

  | LOF | PR-AUC des 5 tirages | Moyenne | Écart type |
  |---|---|---|---|
  | Sans étiquette (20 voisines) | 0,0330 à 0,0369 | 0,0347 | 0,0013 |
  | Réglée (10 voisines) | 0,0444 à 0,0538 | 0,0477 | 0,0032 |

  Les plages ne se recouvrent pas : le plus mauvais tirage réglé (0,0444)
  dépasse le meilleur tirage par défaut (0,0369). Pour chaque tirage, le
  réglage gagne entre + 0,011 et + 0,017. Gain moyen + 0,013 (+ 37 %),
  environ dix fois l'écart type des tirages par défaut : **le gain est réel**.
  La graine 42 retenue était le tirage le moins favorable pour les deux
  versions. Correction faite avant ce contrôle : la graine n'était pas
  transmise au tirage de l'échantillon, qui aurait donné cinq fois le même
  résultat ; les résultats de la graine 42 ne changent pas (test ajouté).
- **Contrôle 2 : un petit échantillon d'étiquettes suffit-il pour régler ?**
  20 fois, on tire 10 % des comptes émetteurs de la validation (155 à 235 cas
  selon le tirage), on choisit le réglage avec leurs seules transactions, puis
  on le mesure sur les 90 % restantes (`reports/mesures/petit_echantillon.json`) :

  | Méthode | Même choix que la grille complète | PR-AUC sur les 90 % : choix du petit échantillon / grille complète / par défaut |
  |---|---|---|
  | LOF | 20 fois sur 20 | 0,0443 / 0,0443 / 0,0330 |
  | Autoencodeur | 20 fois sur 20 | 0,0151 / 0,0151 / 0,0151 |
  | Isolation Forest | 14 fois sur 20 | 0,0051 / 0,0053 / 0,0042 |

  Environ 200 cas étiquetés suffisent pour retrouver le réglage. Quand Isolation
  Forest choisit autrement, c'est entre réglages presque équivalents.
- **Limite** : ces gains sont mesurés sur la validation qui a servi à choisir ;
  le test (étape 7) dira s'ils tiennent.

**Trois messages pour l'oral** (méthodes non supervisées, validation) :

1. **Sans étiquette, LOF fait environ 30 fois mieux que le hasard** : PR-AUC
   0,033 (0,035 en moyenne sur 5 tirages) contre 0,00105 pour un score au
   hasard.
2. **Régler avec les étiquettes rapporte peu** : + 0,013 de PR-AUC pour LOF
   (réel, au-delà du hasard du tirage, et retrouvé avec environ 200 cas),
   + 0,0015 pour Isolation Forest, rien pour l'autoencodeur. En valeur
   relative, + 37 % pour LOF : « peu » s'entend par rapport au point 3.
3. **Le choix de la méthode compte bien plus que son réglage** : en moyenne
   sur 5 graines, LOF fait environ 10 fois mieux qu'Isolation Forest, aussi
   bien en version par défaut qu'en version réglée. Le meilleur Isolation
   Forest réglé (0,0052) reste 6 fois sous LOF sans étiquette (0,033). La
   règle sans étiquette a choisi LOF sur le seul coût : bon choix, mais par
   coïncidence, puisqu'elle ne regardait pas la performance.

## 24. Limite : les scores de validation sont optimistes

- **Constat** : la validation sert à plusieurs choix qui utilisent ses
  étiquettes : le réglage des méthodes non supervisées (grilles de LOF,
  d'Isolation Forest et de l'autoencodeur), l'arrêt précoce de LightGBM, le
  choix du traitement du déséquilibre et le seuil de l'étape 5.
- **Conséquence** : chaque choix retient ce qui marche le mieux sur ces lignes
  précises, y compris par chance. Les scores de validation des versions
  réglées sont donc **optimistes** : ils mesurent la qualité du choix sur les
  données qui ont servi à le faire.
- **Ce qui limite l'effet** : des grilles petites (3 à 12 réglages), non
  agrandies après coup ; les contrôles de la section 23 (gain de LOF au-delà
  du hasard du tirage, réglage retrouvé avec 10 % des comptes).
- **Mesure honnête** : seul le test (étape 7), lu une seule fois après tous
  les choix, donne une mesure non biaisée. Les versions « sans étiquette »
  ne sont pas réglées sur la validation : leurs scores de validation sont
  déjà honnêtes, à la dérive dans le temps près.

## 25. Modèle supervisé : LightGBM, déséquilibre et courbe d'apprentissage

- **Choix** : LightGBM (gradient boosting : des arbres construits l'un après
  l'autre, chacun corrigeant les erreurs des précédents), sur la même matrice
  de 23 variables que les méthodes non supervisées ; régression logistique
  comme contrôle. Rôle : mesurer ce que l'on perd sans étiquettes.
- **Raison** : référence sur les données en tableau, rapide (un modèle par
  défaut apprend en une dizaine de secondes sur 4,8 M de lignes), léger pour
  Render, expliqué exactement par SHAP (étape 6), déjà dans les versions
  figées. Le logarithme et la standardisation ne changent rien pour des
  arbres, qui ne regardent que l'ordre des valeurs.
- **Données** : entraînement retenu (4 836 832 lignes, 4 954 cas) pour
  apprendre ; validation (1 900 970 lignes, 1 988 cas) pour mesurer, pour
  l'arrêt précoce et pour les choix (section 24). Test non lu.
- **Déséquilibre (environ 1 cas pour 1 000)** : trois options, du plus simple
  au plus complexe — aucun traitement ; poids des cas (`scale_pos_weight` =
  nombre de transactions normales par cas) ; sous-échantillonnage (tous les
  cas, 10 % des transactions normales tirées au hasard). LightGBM par défaut,
  graine 42, une fois par option.
- **Règle de choix du déséquilibre, version définitive, fixée avant toute
  mesure** (`choose_imbalance` dans `src/aml_xai/supervised.py`) : bootstrap
  apparié sur la validation. 200 tirages (graine 42) de comptes émetteurs
  entiers, avec remise ; un compte tiré deux fois compte deux fois (poids des
  lignes). Dans chaque tirage, les PR-AUC des options sont calculées sur les
  mêmes lignes. On part de « aucun » ; « poids » le remplace seulement s'il est
  strictement meilleur dans au moins 95 % des tirages ; puis le
  sous-échantillonnage est comparé de la même façon à l'option retenue à ce
  moment. Sinon, on garde la plus simple.
- **Pourquoi cette règle remplace la première** : la première version
  comparait l'écart des moyennes à 2 écarts types sur 5 graines. Or LightGBM
  par défaut ne tire au hasard ni lignes ni variables : il ne dépend presque
  pas de la graine, et cet écart type ne mesurait pas la vraie incertitude.
  Le bootstrap mesure l'incertitude qui compte : celle due aux comptes qui
  composent la validation. **Changement décidé et codé avant toute mesure de
  l'étape 4**, à une exception près : un essai de durée de LightGBM par
  défaut, sans traitement, avait affiché une PR-AUC de validation de 0,38.
  Aucune autre option n'avait été mesurée, donc rien ne permettait de
  comparer les options. Tests : PR-AUC pondérée identique à celle de
  scikit-learn, poids par comptes entiers, option nettement meilleure
  retenue, options identiques → la plus simple, ordre des comparaisons.
- **SMOTE écarté** : il fabriquerait des transactions impossibles (types de
  paiement ou compteurs « entre deux ») et n'est pas dans les versions
  figées.
- **Effet des poids et du sous-échantillonnage** : ils faussent la
  probabilité mais pas l'ordre des transactions ; la PR-AUC et le budget k
  (étape 5) ne regardent que l'ordre.
- **Deux versions** : par défaut (100 arbres, 31 feuilles, vitesse 0,1) ;
  réglée (feuilles 15, 31 ou 63 × vitesse 0,05 ou 0,1, jusqu'à 2 000 arbres,
  arrêt précoce après 50 arbres sans progrès de la PR-AUC de validation).
- **Courbe d'apprentissage** : LightGBM par défaut avec le traitement retenu,
  appris sur 1, 5, 10, 25 et 100 % des comptes émetteurs de l'entraînement
  (5 tirages sous 100 %), avec le nombre de cas utilisés à chaque point, pour
  dire « avec N cas, le modèle supervisé dépasse LOF sans étiquette ».
- **Point à 10 % ajouté avant la relance** : les premiers résultats (voir
  ci-dessous, archivés) laissaient le seuil entre environ 280 et 1 200 cas ;
  un point à 10 % des comptes (environ 500 cas) le situe mieux. C'est une
  mesure de plus, pas un réglage ; il a été ajouté avant de relancer, sans
  connaître ses résultats.
- **Erreur de mesure, trouvée et corrigée** : la première exécution notait
  les transactions avec la probabilité de LightGBM et de la régression
  logistique. Quand le modèle est très sûr, la probabilité est arrondie à
  1,0 exactement : les transactions concernées deviennent ex æquo et leur
  ordre est perdu. Avec les poids, 208 476 transactions de validation
  valaient exactement 1,0 (les 5 000 premières toutes ex æquo) ; sans
  traitement, 1 118. Sur le modèle par défaut sans traitement, la PR-AUC
  passe de 0,3785 (probabilité) à 0,4110 (score brut).
  - **Correction** : classement par le score brut (log-odds de LightGBM,
    `decision_function` de la régression logistique), qui donne le même ordre
    sans arrondi (`ranking_scores`). Chaque calcul enregistre les ex æquo en
    tête et s'arrête si plus de 1 % des 5 000 premières transactions sont ex
    æquo au maximum. Tests : mécanisme de la saturation, score brut renvoyé,
    arrêt en cas d'ex æquo.
  - **Arrêt précoce non touché** : la PR-AUC interne de LightGBM est calculée
    sur le score brut (0,410958 à 100 arbres, identique à scikit-learn sur le
    score brut, contre 0,378463 sur la probabilité).
  - **Méthodes non supervisées non touchées** : dans les 39 fichiers de
    scores des grilles, au plus une ligne au score maximal, et au moins 4 793
    valeurs distinctes parmi les 5 000 premières
    (`reports/mesures/ex_aequo_non_supervise.json`).
  - **Résultats de la première exécution** : déplacés hors du dépôt, non
    supprimés ; toute l'étape 4 est recalculée.
- **Mesures** (validation, scores bruts ; `reports/mesures/supervise.jsonl`,
  `desequilibre.json`, `supervise_tableau.json`) :
  - **Déséquilibre : « aucun » retenu.** Ni les poids ni le
    sous-échantillonnage ne battent « aucun » dans un seul des 200 tirages
    (écart médian − 0,399 et − 0,081). PR-AUC (intervalle à 95 % du
    bootstrap) : aucun 0,411 (0,372 à 0,448), sous-échantillonnage 0,331
    (0,294 à 0,362), poids 0,012 (0,009 à 0,014). Avec un poids d'environ
    976 par cas, le modèle par défaut se trompe massivement en tête de
    classement, tout en gardant une ROC-AUC de 0,886.
  - **Tableau comparatif** (hasard = 0,00105) :

    | Méthode | Étiquettes utilisées | PR-AUC | Fois le hasard | ROC-AUC |
    |---|---|---|---|---|
    | Isolation Forest, sans étiquette | aucune | 0,0042 | 4 | 0,771 |
    | Isolation Forest, réglée | validation (réglage) | 0,0052 | 5 | 0,808 |
    | Autoencodeur | aucune | 0,0151 | 14 | 0,847 |
    | **LOF, sans étiquette** | aucune | **0,0330** | 32 | 0,828 |
    | LOF, réglé | validation (réglage) | 0,0444 | 42 | 0,834 |
    | Régression logistique (contrôle) | entraînement | 0,0273 | 26 | 0,912 |
    | **LightGBM par défaut** | entraînement | **0,4110** | 393 | 0,901 |
    | LightGBM réglé (63 feuilles, vitesse 0,05, 29 arbres) | entraînement + validation (arrêt précoce) | 0,5756 | 550 | 0,948 |

    LOF sans étiquette atteint 8 % de la PR-AUC de LightGBM par défaut. La
    régression logistique, malgré les étiquettes, fait moins bien que LOF
    sans étiquette : sans combinaison non linéaire des variables, les
    étiquettes ne suffisent pas. L'arrêt précoce s'arrête entre 14 et
    94 arbres : au-delà, la PR-AUC de validation baisse.
  - **Courbe d'apprentissage** (LightGBM par défaut, sans traitement ;
    5 tirages de comptes sous 100 %) :

    | Comptes | Cas utilisés | PR-AUC (min à max) | Dépasse LOF sans étiquette (0,0330) | Dépasse LOF réglé (0,0444) |
    |---|---|---|---|---|
    | 1 % | 30 à 79 | 0,002 à 0,050 | 1 sur 5 | 1 sur 5 |
    | 5 % | 240 à 280 | 0,014 à 0,059 | 2 sur 5 | 1 sur 5 |
    | 10 % | 431 à 538 | 0,044 à 0,120 | **5 sur 5** | 4 sur 5 |
    | 25 % | 1 211 à 1 305 | 0,182 à 0,264 | 5 sur 5 | 5 sur 5 |
    | 100 % | 4 954 | 0,411 | oui | oui |

    **Avec environ 450 à 550 cas étiquetés, LightGBM dépasse LOF sans
    étiquette à chaque tirage** ; avec environ 250 cas, seulement 2 fois
    sur 5. Le point à 1 % est très instable (un tirage de 79 cas atteint
    0,050 par chance).
  - **Contrôle de saturation** : dans les 31 calculs, une seule ligne au
    score maximal. Les modèles à peu d'arbres ont moins de valeurs
    distinctes en tête (1 356 parmi les 5 000 premières pour 14 arbres) :
    c'est le nombre limité de combinaisons de feuilles, pas un arrondi.
  - **Limite** : la version réglée est choisie sur la validation (section
    24) ; la version par défaut et la courbe ne le sont pas.

## 26. Limite pour l'oral : l'écart entre LightGBM et LOF est peut-être surestimé

- **Constat** : LightGBM fait environ 12 fois mieux que LOF sans étiquette
  (section 25). Une partie de cet écart peut venir des **artefacts du
  générateur** plutôt que d'un vrai comportement de blanchiment. Un modèle
  supervisé apprend tout ce qui sépare les classes, artefacts compris ; une
  méthode non supervisée ne cherche que ce qui est rare.
- **Artefacts concernés** (section 17) :
  - **plafond des montants** : aucune transaction normale au-dessus de
    999 946,47 ; tous les cas `Over-Invoicing` au-dessus de 1,26 M.
    `log_montant` permet à LightGBM d'apprendre ce seuil exact ; LOF ne voit
    qu'un montant rare ;
  - **rythme mensuel** : aucune variable « jour du mois » n'est donnée aux
    modèles. LightGBM ne peut le capter qu'indirectement, par exemple à
    travers les compteurs sur 30 jours ; l'effet attendu est donc plus faible
    que pour le plafond ;
  - **montants à nombreuses décimales** : aucune variable n'en est tirée ;
    seul le montant arrondi entre dans les modèles.
- **Vérification prévue à l'étape 5** : analyse par typologie. Pour chaque
  typologie, rappel de LightGBM et de LOF au même budget d'alertes. Si
  l'écart vient surtout d'`Over-Invoicing` (ou d'une typologie liée à un
  artefact), on recalcule l'écart sans elle. Les cas `Over-Invoicing` sont
  peu nombreux (10 dans le test, section 15) : leur poids dans la PR-AUC
  globale sera mesuré, pas supposé.
- **Résultat de la vérification** (validation,
  `reports/mesures/evaluation.json`, PR-AUC de LightGBM par défaut et de LOF
  sans étiquette) :

  | Cas gardés | Cas | LightGBM | LOF | Rapport |
  |---|---|---|---|---|
  | Tous | 1 988 | 0,411 | 0,033 | 12,4 |
  | Sans `Over-Invoicing` | 1 970 | 0,411 | 0,031 | 13,2 |
  | Sans montants à nombreuses décimales | 1 776 | 0,420 | 0,031 | 13,5 |
  | Sans les deux | 1 758 | 0,420 | 0,029 | 14,5 |

  **Pour la PR-AUC, l'hypothèse est réfutée** : sans les cas liés aux
  artefacts, l'écart **augmente**. Ce sont ces cas qui aident LOF (16
  `Over-Invoicing` sur 18 trouvés au budget), pas LightGBM (9 sur 18). **Pour
  le coût, l'effet est inverse** : `Over-Invoicing` pèse 75 % du montant et
  avantage LOF (section 9). Le rythme mensuel n'est pas vérifié : la
  validation ne couvre que 64 jours, soit 2 ou 3 fois chaque jour du mois,
  trop peu pour une mesure fiable.
- **Conclusion pour l'oral : l'hypothèse sur les artefacts est réfutée pour
  la PR-AUC.** L'écart entre LightGBM et LOF ne vient pas des artefacts ;
  il augmente même sans eux.

## 27. Perspective : variables comparant un compte à son propre passé

- **Constat** (étape 5, validation, seuil retenu) : les typologies de
  changement de comportement échappent. LOF ne trouve aucun cas
  `Behavioural_Change_1` ni `Behavioural_Change_2` ; LightGBM est à son plus
  faible sur `Behavioural_Change_1` (15 % par défaut, 38 % réglé).
- **Cause probable** : peu de variables comparent un compte à son propre
  passé. Seul le rapport du montant à la moyenne sur 30 jours le fait.
- **Perspective** : des rapports d'activité, par exemple le nombre de
  transactions sur 24 h rapporté à la moyenne journalière du mois, ou le
  nombre de contreparties nouvelles par rapport au mois précédent.
- **Décision : on ne l'ajoute pas maintenant.** Les variables, la matrice et
  tous les choix des étapes 3 à 5 ont été faits sans elles ; les ajouter
  après avoir vu les typologies manquées reviendrait à régler le projet sur
  la validation. C'est une piste de suite, à présenter comme telle.

## 28. Explications SHAP et codes motifs (étape 6)

Règles écrites avant toute mesure de l'étape 6.

- **Modèles expliqués** : LightGBM réglé, par TreeSHAP (exact) ; LOF sans
  étiquette, par KernelSHAP (approché). SHAP répartit le score d'une
  transaction entre ses variables : valeur de base + somme des contributions
  = score.
- **Sens des contributions, le même pour les deux modèles** : une
  contribution positive pousse toujours vers « suspect ».
  - LOF : dans scikit-learn, `score_samples` est plus haut pour les
    transactions normales. On explique donc le **score inversé**
    (`−score_samples`), le même que celui de l'évaluation (plus haut = plus
    suspect). Un test vérifie le sens sur une transaction aberrante.
  - LightGBM : on explique le **score brut (log-odds)**, pas la probabilité,
    qui sature à 1,0 (section 25). TreeSHAP est calculé par LightGBM lui-même
    (`pred_contrib`), sans la bibliothèque SHAP. Dans l'application et l'API,
    on parlera de « score de risque ».
- **KernelSHAP sur LOF, réglages fixés avant les mesures** : données de
  référence = 20 centres de k-moyennes de 10 000 lignes d'entraînement tirées
  au hasard (graine 42) ; 500 évaluations par alerte ; 100 alertes de LOF
  tirées au hasard (graine 42) parmi celles du seuil retenu. La durée par
  alerte est mesurée ; son usage dans l'API sera décidé ensuite.
- **Codes motifs** (`config/codes_motifs.json`) : 13 groupes de variables ;
  la contribution d'un groupe est la somme de celles de ses variables. Une
  alerte affiche **au plus 3 codes**, ceux de plus forte contribution, et
  **seulement des groupes de contribution positive** : s'il y en a moins de
  3, on en affiche moins. Chaque code est accompagné d'une phrase avec les
  vraies valeurs de la transaction.
- **Fidélité, règles fixées avant les mesures** :
  1. additivité : écart entre valeur de base + somme des contributions et
     score du modèle (nul attendu pour TreeSHAP, mesuré pour KernelSHAP) ;
  2. suppression : pour chaque alerte, les variables de ses codes motifs
     sont remplacées par leur médiane de l'entraînement, et la baisse du
     score est comparée à celle obtenue en remplaçant autant de groupes tirés
     au hasard (graine 42). Explications jugées fidèles si la baisse due aux
     codes motifs est plus forte dans **au moins 90 %** des alertes. Sur
     500 alertes de LightGBM tirées au hasard, et sur les 100 alertes de LOF ;
  3. stabilité (KernelSHAP) : 50 de ces alertes recalculées avec une autre
     référence (10 000 autres lignes, graine 1) ; part des alertes dont les
     codes affichés restent les mêmes.
- **Limites** : SHAP explique le modèle, pas la réalité ; les variables
  corrélées se partagent le mérite (le regroupement limite l'effet) ;
  KernelSHAP remplace les variables comme si elles étaient indépendantes et
  peut évaluer des transactions impossibles.
- **Phrases** (décidé après avoir lu les premiers exemples, affichage seulement) :
  chaque phrase donne le sens et la valeur habituelle, médiane de
  l'entraînement retenu (`models/valeurs_habituelles.json`), par exemple
  « seulement 1 émetteur différent en 30 jours (habituellement 6) » ; la
  devise dit « même devise » ou « devise différente ». Raison : une
  contribution positive peut venir d'une valeur basse, et la phrase sans sens
  se lisait comme « beaucoup ».
  Les deux durées d'une phrase sont écrites dans la même unité, celle de la
  plus courte : « il y a 29 h (habituellement 14 h) ».
- **Reproductibilité corrigée** : KernelSHAP ignore `random_state` et tire avec
  le générateur global de numpy ; une première exécution n'était donc pas
  reproductible. La graine est maintenant fixée avant l'appel (test avec
  12 variables, pour qu'il y ait vraiment un tirage). Deux exécutions donnent
  le même fichier de mesures, hors durées.
- **Mesures** (validation, `reports/mesures/explications.json`) :

  | Mesure | LightGBM réglé (TreeSHAP) | LOF sans étiquette (KernelSHAP) |
  |---|---|---|
  | Contrôle du modèle | réappris : scores identiques à l'étape 4 (29 arbres) | scores identiques à la grille |
  | Alertes expliquées | 8 947 (toutes, au seuil retenu) | 100 (tirées au hasard) |
  | Durée par alerte | 0,1 ms | 2,2 à 3,5 s selon la charge du PC |
  | Additivité, écart maximal | 7 × 10⁻¹³ | 2 × 10⁻¹⁵ |
  | **Suppression, règle (≥ 90 %)** | **75,8 % : non fidèle** | **75 % : non fidèle** |
  | Baisse médiane : codes / hasard | 2,96 / 0,37 | 0,34 / 0,003 |
  | Stabilité (autre référence) | — | mêmes codes 92 %, même ordre 84 % |

  **Verdict : selon la règle fixée avant les mesures, les explications ne
  sont pas jugées fidèles.** C'est une limite à présenter à l'oral. En
  médiane, retirer les codes motifs fait pourtant baisser le score 8 fois
  (LightGBM) et 100 fois (LOF) plus que retirer des groupes au hasard.
- **Diagnostic de l'échec** :
  1. défaut de la règle elle-même : les groupes « au hasard » étaient tirés
     parmi les 13, codes motifs compris ; un tirage de 3 en contient au moins
     un dans 58 % des cas (277 alertes sur 500 pour LightGBM, réussite 68 %
     dans ces alertes, 85,2 % dans les autres) ;
  2. vrai défaut de fidélité : dans 15 % des alertes de LightGBM et 17 % de
     LOF, remplacer les variables des codes par leur médiane ne fait pas
     baisser le score. SHAP mesure une contribution par rapport à une
     transaction **moyenne**, le test remplace par la **médiane**, et les
     arbres combinent les variables : la transaction modifiée peut devenir
     suspecte autrement.
- **Second test, conçu après les résultats, à titre d'information** (ne
  change pas le verdict) : groupes tirés au hasard parmi ceux qui ne sont pas
  des codes motifs ; remplacement par la référence de SHAP (moyenne de
  l'entraînement pour LightGBM, moyenne pondérée des 20 centres de référence
  pour LOF). Les codes font baisser le score plus que le hasard dans
  **87,4 %** des alertes (LightGBM) et **84 %** (LOF) ; ils ne le font pas
  baisser dans 10,6 % et 22 % des alertes.
- **Ce que disent les explications** : LightGBM s'appuie d'abord sur les
  contreparties distinctes de l'émetteur, les sources distinctes du
  destinataire et le type de paiement ; les rafales (`em_nb_1h`,
  `em_nb_24h`) ne servent presque pas. LOF s'appuie sur les devises, les pays
  et les rafales (combinaisons rares). Les codes par typologie sont
  cohérents : `Smurfing` → type de paiement (99 %) ; `Over-Invoicing` →
  montant inhabituel pour l'émetteur (92 %) ; `Structuring` et `Fan_In` →
  sources multiples du destinataire (90 % et 82 %).
- **Mémoire** : pic de 1 585 Mo pour le modèle et d'environ 1 700 Mo pour les
  explications, au-dessus de l'objectif de 1,5 Go malgré la séparation en
  deux processus et la lecture des seules colonnes utiles.
- **Note pour l'application (étape 9)** : un score de risque négatif (par
  exemple −4,56, en log-odds) peut faire croire à un analyste que la
  transaction n'est pas risquée. L'application affichera aussi le **rang** de
  l'alerte (par exemple « parmi les 0,5 % les plus suspectes »).
- **Décision pour l'API (étape 8)** : TreeSHAP en direct pour LightGBM ;
  explications de LOF calculées à l'avance pour les exemples (KernelSHAP met
  plusieurs secondes par alerte sur le PC, davantage sur Render).

## 29. Choix figés avant l'ouverture du test (étape 7)

Cette section est commitée et poussée **avant** toute lecture des étiquettes,
des scores ou des variables du test. L'historique Git prouve ainsi qu'aucun
choix n'a changé après. Si une erreur était découverte après l'ouverture du
test, elle serait corrigée visiblement, avec les deux résultats conservés et
expliqués, jamais remplacée en silence.

### Données

- **Test** : transactions du 19/06/2023 22:08:52 au 23/08/2023 10:57:12
  (1 900 971 lignes, `reports/effectifs_periodes.json`). Lu **une seule
  fois**, par un seul script, qui écrit `reports/mesures/test.json`.
- **Variables** : celles de `data/processed/features.parquet`, calculées sur
  l'historique complet par la règle `[t − fenêtre ; t[` (section 18) ; codage
  par fréquence (`models/frequences.json`) et transformation
  (`models/transformation.json`) appris sur l'entraînement retenu, appliqués
  tels quels au test. Aucun paramètre n'est réappris sur le test ni sur la
  validation.

### Modèles et réglages

Tous appris sur l'entraînement retenu seulement (à partir du 06/11/2022,
4 836 832 lignes), **sans réapprentissage sur entraînement + validation** :
ce sont exactement les modèles évalués sur la validation.

| Modèle | Réglage figé | Seuil retenu (score de validation) |
|---|---|---|
| Isolation Forest, sans étiquette | 100 arbres, 256 lignes par arbre, 100 % des variables, graine 42 | 0.6359087562758536 |
| Isolation Forest, réglé | 100 arbres, 4 096 lignes par arbre, 50 % des variables, graine 42 | — (PR-AUC seulement) |
| LOF, sans étiquette | 20 voisines, échantillon de 100 000 lignes (graine 42) | 1.5251219272613525 |
| LOF, réglé | 10 voisines, échantillon de 100 000 lignes (graine 42) | 1.507451057434082 |
| Autoencodeur | goulot de 8, échantillon de 100 000 lignes, graine 42 | — (PR-AUC seulement) |
| LightGBM, par défaut | 100 arbres, 31 feuilles, vitesse 0,1, sans traitement du déséquilibre, graine 42 | -5.200432465566857 |
| LightGBM, réglé | 63 feuilles, vitesse 0,05, **29 arbres**, sans traitement du déséquilibre, graine 42 | -5.034005813871007 |
| Régression logistique (contrôle) | `max_iter=1000`, sans traitement | — (PR-AUC seulement) |

- **Scores** : plus haut = plus suspect ; score brut (log-odds) pour
  LightGBM, `decision_function` pour la régression logistique, score inversé
  pour LOF et Isolation Forest, erreur de recopie pour l'autoencodeur.
- **Une alerte** = une transaction dont le score atteint le seuil ci-dessus,
  appliqué tel quel au test (il n'est pas recalculé).
- **Contrôle avant d'ouvrir le test** : chaque modèle est réappris (ou relu)
  et doit redonner **exactement** ses scores de validation, dont les
  empreintes SHA-256 sont :

  | Fichier de scores de validation (`data/processed/`) | SHA-256 |
  |---|---|
  | `scores_grilles/isolation_forest__n_estimators-100__max_samples-256__max_features-1.0__graine-42.npy` | `5c03b5b9b9fb91dc4dce541dde80a90213e82d57992b9767886fd33780e36b22` |
  | `scores_grilles/isolation_forest__n_estimators-100__max_samples-4096__max_features-0.5__graine-42.npy` | `ce8970f1c7c0bf581485b3a964ebc8c7047bc091ccc276760858f61573df9eff` |
  | `scores_grilles/lof__n_neighbors-20__echantillon-100000__graine-42.npy` | `36fe100569493a74cfabb3450043f1c0d1669697587bd220fa3bce3f2cd625cc` |
  | `scores_grilles/lof__n_neighbors-10__echantillon-100000__graine-42.npy` | `c4bb2fdbe9932c301d60d15aa5381a33ad8c1d743d56d4f1ccebdd29d60ac2e8` |
  | `scores_grilles/autoencodeur__goulot-8__graine-42.npy` | `21688cc6b1524bce80c08fe3fc0bd91a515a2529218b788285648836c0b2d860` |
  | `scores_supervise/desequilibre__option-aucun__graine-42.npy` | `bcb2ac5700db468e8ca7a5236b75e2445863fdf75dd279fdea319768f1624d02` |
  | `scores_supervise/grille__option-aucun__num_leaves-63__learning_rate-0.05.npy` | `b134b25d8d4666f4e043c974cbc9c40a0f4e34e7636a757d64bfdd2f4606a470` |
  | `scores_supervise/logistique__option-aucun.npy` | `48d3f2a704310b2ba97b7140787d3a0235c6bc3fb6bcee7ee4b0ee250abebed7` |

  Matrices : `matrice_entrainement.npy`
  `298000e532c51f684ccf048eae40f9d1a0bd1e7477a3b604f1015ca78fcf6997`,
  `matrice_validation.npy`
  `bbdc59e74280b55a48f1acfd535a76b076e7258ef43157fabac0d88baf7a5700`.

### Fichiers de configuration figés

| Fichier | SHA-256 |
|---|---|
| `config/hypotheses.json` | `6ec83b91f1b3ec4ab48bf759fb9f548bed03d13f2c112e94c82cd61998c79219` |
| `config/codes_motifs.json` | `49f13bd62c486c160af1421a4fc7443d5e8c31cffa283031a67b982c85256db6` |
| `models/frequences.json` | `7342314390547a5e9962efc436a4ecf1f248de22ecba3d296c5386e939a553bb` |
| `models/transformation.json` | `63bbada1066dc83d1b85829a8495ba2f0c02a5317a939c4796353f5f2624c6d9` |
| `models/valeurs_habituelles.json` | `30c9a18193cf65e5e86b12a3b939d2f00263ecc8302a4ed964d15b6ef9ecd8f7` |

- **Hypothèses de coût** : 5 analystes, 7 h par jour, 15 min par alerte,
  30 £ de l'heure ; 140 alertes par jour ; 7,50 £ par alerte. Coût = somme
  des montants des blanchiments non détectés + nombre d'alertes × 7,50 £
  (section 9). Capacité du test = 140 × durée du test en jours.
- **Codes motifs** : les 13 codes de `config/codes_motifs.json`, au plus 3
  par alerte, seulement de contribution positive ; phrases avec la valeur
  habituelle de `models/valeurs_habituelles.json` (section 28).

### Mesures à calculer sur le test

1. **Pour les huit modèles** : PR-AUC, ROC-AUC, rapport au hasard (taux de
   blanchiment du test).
2. **Pour les cinq modèles à seuil**, au seuil figé : alertes par jour
   (comparées aux 140 de la capacité), précision, rappel, F2, MCC, coût
   total, coût sans `Over-Invoicing`, part du montant détecté ; en
   complément, les 140 premières transactions de chaque jour. À titre
   d'information seulement : le seuil qui aurait été de coût minimal sous la
   capacité sur le test, pour mesurer l'écart avec le seuil figé.
3. **Écart validation → test** pour chaque mesure, et pour la PR-AUC
   l'intervalle à 95 % d'un bootstrap par comptes émetteurs (200 tirages,
   graine 42), pour dire si la baisse dépasse le hasard.
4. **Typologies** : rappel par typologie au seuil figé, avec l'intervalle de
   Wilson ; fausses alertes par type normal.
5. **Nouveauté** : part des alertes et précision pour les émetteurs et
   destinataires sans historique sur 30 jours ou jamais vus en entraînement.
6. **Artefacts** : rapport des PR-AUC LightGBM par défaut / LOF sans
   étiquette, avec et sans `Over-Invoicing` et montants à nombreuses
   décimales.
7. **Explications**, mêmes règles qu'en section 28 : additivité ; test de
   suppression (règle à 90 %, verdict) et second test d'information ;
   TreeSHAP sur les alertes de LightGBM réglé (suppression sur 500 tirées au
   hasard, graine 42), KernelSHAP sur 100 alertes de LOF sans étiquette
   (graine 42).
