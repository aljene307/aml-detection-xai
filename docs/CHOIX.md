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
