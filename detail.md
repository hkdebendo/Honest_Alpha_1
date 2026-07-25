# HonestAlpha *(titre provisoire)*

**Deep learning vs modèles simples pour la prédiction de rendements : une étude walk-forward rigoureuse**

> Ce projet ne cherche pas à trouver un alpha. Il cherche à répondre honnêtement à une question : *un réseau de neurones fait-il mieux que des modèles simples pour prédire les rendements d'ETF, une fois la validation et les coûts traités correctement ?* La réponse peut très bien être « non », et ce serait un résultat valide.

Auteur : [Votre nom] · Licence : MIT · Statut : en cours (voir la feuille de route)

---

## Sommaire

1. [Question de recherche](#1-question-de-recherche)
2. [Ce que le projet n'est pas](#2-ce-que-le-projet-nest-pas)
3. [Données](#3-données)
4. [Cible et variables](#4-cible-et-variables)
5. [Modèles](#5-modèles)
6. [Validation walk-forward](#6-validation-walk-forward)
7. [Mesures de performance](#7-mesures-de-performance)
8. [Analyses et tests de robustesse](#8-analyses-et-tests-de-robustesse)
9. [Structure du dépôt](#9-structure-du-dépôt)
10. [Installation et reproduction](#10-installation-et-reproduction)
11. [Feuille de route et livrables](#11-feuille-de-route-et-livrables)
12. [Résultats](#12-résultats)
13. [Limites et échecs](#13-limites-et-échecs)
14. [Article scientifique](#14-article-scientifique)
15. [Outils et transparence](#15-outils-et-transparence)
16. [Références](#16-références)
17. [Avertissement](#17-avertissement)

---

## 1. Question de recherche

**Question** : sur un univers d'ETF liquides, un perceptron multicouche (MLP) écrit en PyTorch prédit-il les rendements à 5 jours mieux que des modèles de référence (régression Ridge, CatBoost), de façon statistiquement significative et stable dans le temps ?

**Hypothèses**
- **H0** : le MLP n'apporte pas de gain d'information par rapport aux modèles de référence (différence d'IC moyen nulle).
- **H1** : le MLP obtient un IC moyen significativement supérieur à celui des modèles de référence.

Le projet est conçu pour pouvoir **ne pas rejeter H0** et le dire clairement.

**Pourquoi ce sujet** : sur des données financières, le rapport signal/bruit est très faible et les erreurs de méthode (fuite d'information du futur, validation mal découpée, coûts ignorés) produisent facilement de faux résultats flatteurs. Le but est de montrer une démarche de recherche propre, pas une performance.

## 2. Ce que le projet n'est pas

- Ce n'est pas une stratégie de trading prête à l'emploi.
- Ce n'est pas une promesse de rentabilité : aucun résultat ici ne doit être lu comme un conseil d'investissement.
- Ce n'est pas un projet de haute fréquence : l'horizon est de 5 jours, avec des données journalières gratuites.

## 3. Données

- **Source** : Yahoo Finance via `yfinance` (prix ajustés et volumes journaliers).
- **Univers** : 30 à 40 ETF liquides couvrant plusieurs classes d'actifs (actions américaines et internationales, secteurs, obligations, matières premières, immobilier coté). La liste exacte est dans `configs/universe.yaml` et sera figée avant toute expérience.
- **Période** : environ 10 ans d'historique journalier. Les ETF qui n'ont pas assez d'historique sont exclus plutôt que complétés.
- **Choix des ETF** : un univers d'ETF limite le biais du survivant par rapport à des actions individuelles (pas de faillites ni de sorties d'indice), mais ne l'élimine pas.

**Limites connues des données (à répéter dans le rapport)**
1. *Biais du survivant résiduel* : seuls les ETF encore cotés aujourd'hui sont présents ; les fonds liquidés ou fusionnés sont absents.
2. *Qualité des données gratuites* : trous, ajustements de dividendes et de splits parfois imparfaits, aucune garantie de fiabilité.
3. *Pas de données de carnet d'ordres* : les coûts de transaction sont modélisés par une hypothèse simple, pas mesurés.
4. *Univers choisi a posteriori* : les ETF populaires aujourd'hui sont ceux qui ont survécu et attiré les capitaux.

Contrôles appliqués : suppression des jours sans cotation, vérification des valeurs aberrantes, alignement des calendriers, journalisation de chaque ligne supprimée.

## 4. Cible et variables

### Cible
- **Rendement futur à 5 jours** (logarithmique), puis **transformation en rang entre actifs à chaque date** (rang transversal). Prédire le rang plutôt que le niveau réduit l'effet des régimes de marché globaux et colle à l'usage en gestion systématique.
- **Alignement temporel** : le signal est calculé avec les données jusqu'à la clôture du jour *t* ; le rendement cible commence à la clôture de *t+1* (on ne peut pas trader au prix qui a servi à calculer le signal) et couvre 5 jours.

### Variables (6 à 8, calculées uniquement avec le passé)
| Groupe | Exemples |
|---|---|
| Momentum | rendements passés à 1, 5, 20 et 60 jours |
| Volatilité | écart-type des rendements sur 20 jours, rapport volatilité courte/longue |
| Volume | volume relatif (volume / moyenne glissante sur 20 jours) |

Toutes les variables sont standardisées **par date, entre actifs** (z-score transversal), ce qui n'utilise aucune information du futur.

### Règles anti-fuite d'information (non négociables)
- Aucune statistique (moyenne, écart-type, seuil) calculée sur toute la période : elles sont soit transversales, soit estimées sur la fenêtre d'entraînement uniquement.
- Les hyperparamètres sont choisis sur une partie validation située **dans** la fenêtre d'entraînement, jamais sur le test.
- Tests automatisés dans `tests/` : décaler une variable vers le futur doit faire exploser l'IC (preuve que le pipeline détecterait une fuite), et mélanger les cibles doit ramener l'IC à zéro.

## 5. Modèles

| Modèle | Rôle | Remarques |
|---|---|---|
| Prédicteur naïf (momentum 20 jours) | plancher de comparaison | aucune estimation |
| **Ridge** | référence linéaire | pénalité choisie sur la validation interne |
| **CatBoost** | référence non linéaire | profondeur faible, arrêt anticipé sur la validation interne |
| **MLP (PyTorch)** | modèle étudié | 2 couches cachées, dropout, `weight decay`, arrêt anticipé, boucle d'entraînement écrite à la main |

Le MLP est volontairement petit : avec peu d'actifs et un signal faible, un gros réseau surapprend. Tous les modèles reçoivent les **mêmes variables** et sont évalués **sur les mêmes fenêtres**. Le MLP est entraîné avec **plusieurs graines aléatoires** (5 prévues) et on rapporte la moyenne et la dispersion.

*Extension possible (hors version 1)* : LSTM ou petit Transformer, univers plus large.

## 6. Validation walk-forward

Principe : on entraîne sur le passé, on teste sur le futur, puis on avance la fenêtre. Aucune validation croisée aléatoire.

```
|----- entraînement -----|-- validation --|gap|--- test ---|
                                  |----- entraînement ------|-- validation --|gap|--- test ---|
```

- **Fenêtre d'entraînement** : extensible (l'historique s'allonge à chaque pas), avec une dernière portion réservée à la validation interne.
- **Intervalle de séparation (gap / purge)** : au moins la durée de l'horizon de la cible (5 jours) plus une marge, entre la fin de l'entraînement et le début du test, pour éviter que des cibles qui se chevauchent ne fassent fuiter l'information.
- **Blocs de test** : successifs, sans recouvrement ; les prédictions de tous les blocs sont concaténées pour former une série hors échantillon continue.
- Les paramètres exacts (tailles de fenêtres, gap) sont dans `configs/walkforward.yaml`, figés avant de regarder les résultats.

## 7. Mesures de performance

**Qualité prédictive**
- **IC** : corrélation de Pearson, à chaque date, entre prédictions et rendements réalisés.
- **IC de rang** : même chose avec la corrélation de Spearman.
- Moyenne de l'IC, écart-type, **ICIR** (moyenne / écart-type) et statistique t.
- Comme les cibles à 5 jours se chevauchent d'un jour à l'autre, les erreurs standard sont corrigées (Newey-West) ou l'IC est échantillonné tous les 5 jours.

**Portefeuille simple après coûts**
- Chaque date de rebalancement : long sur le quintile le mieux classé, short sur le moins bien classé, pondération égale.
- Coûts de transaction : hypothèse paramétrable (par exemple quelques points de base par aller), appliquée au turnover réel. L'analyse est refaite avec plusieurs niveaux de coûts.
- Indicateurs : rendement annualisé, volatilité, ratio de Sharpe, perte maximale, turnover.

Le portefeuille sert de contrôle de réalisme : un IC faible peut disparaître après coûts, et c'est une information en soi.

## 8. Analyses et tests de robustesse

1. **Test statistique de l'écart entre modèles** : test apparié sur les différences d'IC par date (erreurs corrigées de l'autocorrélation) et bootstrap par blocs.
2. **Ablations** : retrait de chaque groupe de variables (momentum, volatilité, volume) et mesure de la perte d'IC.
3. **Stabilité dans le temps** : IC par sous-période (années, régimes de marché).
4. **Graines aléatoires** : dispersion du MLP sur 5 graines.
5. **Sensibilité aux coûts** : performance du portefeuille selon le niveau de coûts.
6. **Contrôles négatifs** : cibles mélangées (IC attendu ≈ 0) et variable volontairement fuitée (IC attendu anormalement élevé).

## 9. Structure du dépôt

```
honest-alpha/
├── README.md
├── pyproject.toml / requirements.txt
├── configs/
│   ├── universe.yaml
│   ├── features.yaml
│   └── walkforward.yaml
├── data/
│   ├── raw/                 # prix téléchargés (non versionnés)
│   └── processed/
├── src/
│   ├── data.py              # téléchargement, nettoyage, journal des suppressions
│   ├── features.py          # variables et cible, sans fuite
│   ├── walkforward.py       # découpage temporel avec gap
│   ├── models/
│   │   ├── baselines.py     # naïf, Ridge, CatBoost
│   │   └── mlp.py           # MLP PyTorch, boucle d'entraînement
│   ├── metrics.py           # IC, IC de rang, ICIR
│   ├── portfolio.py         # portefeuille long/short, coûts
│   └── stats.py             # tests, bootstrap, Newey-West
├── notebooks/               # exploration et figures
├── tests/                   # tests anti-fuite et contrôles négatifs
├── results/                 # tableaux et figures générés
├── report/                  # rapport et article
└── Makefile
```

## 10. Installation et reproduction

```bash
git clone https://github.com/<utilisateur>/honest-alpha.git
cd honest-alpha
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make data        # télécharge et nettoie les données
make features    # construit variables et cible
make run         # walk-forward pour tous les modèles et toutes les graines
make report      # génère tableaux et figures dans results/
make test        # tests anti-fuite
```

Reproductibilité : graines fixées, versions des bibliothèques verrouillées, configurations versionnées, données téléchargées à une date indiquée dans les journaux. Les données Yahoo pouvant être révisées, de petits écarts sont possibles d'un téléchargement à l'autre.

## 11. Feuille de route et livrables

**Version 1 (périmètre actuel)**
- [ ] Univers d'ETF figé et données téléchargées, nettoyées, documentées
- [ ] Cible et variables sans fuite, avec tests
- [ ] Découpage walk-forward avec gap
- [ ] Modèles de référence : naïf, Ridge, CatBoost
- [ ] MLP PyTorch (boucle d'entraînement, régularisation, arrêt anticipé)
- [ ] IC, IC de rang, ICIR
- [ ] Portefeuille long/short après coûts
- [ ] Test statistique, ablations, graines multiples, stabilité temporelle
- [ ] Contrôles négatifs
- [ ] Tableau de résultats et section « limites et échecs »
- [ ] README final et dépôt reproductible

**Version 2 (si le temps le permet)**
- [ ] LSTM ou petit Transformer
- [ ] Univers élargi (environ 100 actifs) et comparaison actions / ETF
- [ ] Variables supplémentaires

**Livrables finaux** : dépôt GitHub reproductible, tableau de résultats, rapport technique, article scientifique (section 14).

## 12. Résultats

*À remplir après les expériences. Aucun chiffre n'est annoncé à l'avance.*

| Modèle | IC moyen | IC de rang moyen | ICIR | Sharpe après coûts |
|---|---|---|---|---|
| Naïf (momentum 20 j) | – | – | – | – |
| Ridge | – | – | – | – |
| CatBoost | – | – | – | – |
| MLP (moyenne sur 5 graines) | – | – | – | – |

Figures prévues : IC cumulé dans le temps par modèle, IC par sous-période, effet des ablations, performance du portefeuille selon les coûts.

## 13. Limites et échecs

*Section à tenir à jour tout au long du projet, y compris pour ce qui ne marche pas.* Points déjà identifiés :

- Univers restreint : les résultats peuvent ne pas se généraliser à d'autres classes d'actifs.
- Biais du survivant résiduel et qualité limitée des données gratuites.
- Horizon de 5 jours et données journalières seulement : pas de conclusion possible sur d'autres fréquences.
- Coûts de transaction modélisés par hypothèse, sans impact de marché.
- Nombre de dates indépendantes limité : la puissance statistique est faible, et un test non significatif ne prouve pas l'absence d'effet.
- Un seul réseau de neurones étudié : on ne peut pas conclure sur « le deep learning » en général.
- Risque de surajustement par les choix de conception : toutes les décisions sont figées avant l'évaluation finale et listées dans le dépôt.

## 14. Article scientifique

Objectif : transformer le projet en un court article de type *preprint* (par exemple sur arXiv, section q-fin), présenté comme une **étude de méthode avec résultats éventuellement négatifs**. Aucune publication n'est garantie : un résultat honnête et bien documenté a de la valeur même sans alpha.

**Titre provisoire** : *Does Deep Learning Beat Simple Baselines for Cross-Sectional Return Prediction? A Rigorous Walk-Forward Study on Liquid ETFs*

**Plan**
1. **Résumé** : question, méthode, résultat principal, limites.
2. **Introduction** : pourquoi la prédiction de rendements est difficile ; erreurs de méthode fréquentes ; contribution (protocole rigoureux, reproductible).
3. **Travaux liés** : prédiction de rendements par apprentissage automatique, validation en finance (purge et embargo), évaluation par IC.
4. **Données** : univers, période, limites.
5. **Méthodologie** : cible, variables, modèles, walk-forward avec gap, métriques, test statistique.
6. **Expériences** : résultats principaux, ablations, stabilité, graines, coûts, contrôles négatifs.
7. **Discussion** : ce que le résultat permet et ne permet pas de conclure.
8. **Limites et pistes** : extensions (LSTM, Transformer, univers plus large).
9. **Conclusion**.
10. **Annexes** : hyperparamètres, liste des ETF, détails d'implémentation, code et données disponibles.

Format : LaTeX (gabarit simple), figures générées par le dépôt, code référencé par un tag de version.

## 15. Outils et transparence

Ce projet est développé avec l'aide d'un assistant d'IA générative pour l'écriture et la relecture du code. L'auteur s'engage à comprendre, tester et pouvoir expliquer chaque composant (données, variables, validation, modèles, métriques), et les choix de conception sont documentés dans le dépôt.

## 16. Références

À compléter avec les sources effectivement lues. Pistes de lecture :
- López de Prado, *Advances in Financial Machine Learning* (purge, embargo, validation).
- Gu, Kelly, Xiu, *Empirical Asset Pricing via Machine Learning* (Review of Financial Studies, 2020).
- Documentation officielle de PyTorch, CatBoost et scikit-learn.

## 17. Avertissement

Ce dépôt est un travail de recherche et d'apprentissage. Il ne constitue ni un conseil en investissement ni une incitation à acheter ou vendre un actif. Les performances passées, simulées ou non, ne préjugent pas des performances futures.
