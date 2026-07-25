# Article scientifique — HonestAlpha

**Titre provisoire :** *Does Deep Learning Beat Simple Baselines for Cross-Sectional Return Prediction?*
  *A Rigorous Walk-Forward Study on Liquid ETFs*  
**Version française :** *Le deep learning bat-il les modèles simples pour la prédiction de rendements
  transversaux ? Une étude walk-forward rigoureuse sur des ETF liquides*

**Auteur** : Ricardo Vincent AMOUSSOU · Licence : MIT · Statut : soumis à relecture · Date : août 2026

---

## Résumé

**Contexte.** La prédiction de rendements financiers par apprentissage profond est tentante, mais
souvent trompeuse : les fuites d'information, la validation mal découpée et l'absence de comptabilité
des coûts produisent de faux résultats flatteurs.

**Objectif.** Mesurer, de façon rigoureuse, si un petit réseau de neurones (MLP, PyTorch) prédit
mieux les rendements à 5 jours qu'un momentum naïf, une régression Ridge et un modèle de machine learning (gradient boosting:CatBoost),
sur un univers de 36 ETF liquides américains, avec une validation walk-forward purgée et une
comptabilité des transaction.

**Méthodes.** Cible = rendement de 5 jours avec retard d'exécution de 1 jour ; variables techniques
(6) standardisées de façon transversale ; découpage walk-forward extensible avec gap (27 plis,
fenêtre initiale de 4 ans, blocs de test de 3 mois) ; métriques = IC de Spearman quotidien,
ICIR, test apparié Newey-West, bootstrap par blocs, portefeuille long-short quintile après coûts.

**Résultats.** Le MLP (moyenne sur 5 graines) obtient un IC de rang moyen de 0,044, légèrement
au-dessus du CatBoost (0,042). La différence n'est pas statistiquement significative
(test apparié : t = 0,40, p = 0,66 ; IC 95 % : [−0,009 ; +0,014]). Le signal existe (t ≈ 5
contre un naïf nul, ICIR ~0,13 pour le CatBoost), mais est trop dilué sur 36 ETF corélés pour
être monétisé sous forme de long-short quintile après 10 bps d'effort de transaction (Sharpe ~0,08).

**Conclusion.** Le MLP n'améliore pas significativement la prédiction par rapport aux modèles
simples, une fois validation et coûts traités correctement. 

**Mots-clés :** apprentissage profond, apprentissage machine financier, prédiction de rendements,
validation walk-forward, coûts de transaction, étude de méthode.



## 1. Introduction

La prédiction de rendements boursiers est un défi classique de l'argentologie. Les réseaux de
neurones, capables de modéliser des relations non linéaires complexes, sont de plus en plus
proposés comme outil prometteur pour générer des signaux de trading. Pourtant, la littérature
déjà abondante sur la prudence en matière de backtest (« data snooping », fuite d'information,
sélection de modèle optimiste) rappelle que les résultats positifs dans ce domaine doivent être
lués avec une extrême prudence.

Le projet HonestAlpha ne cherche pas à prouver l'existence d'un alpha : il cherche à répondre
honnêtement à une question : *un réseau de neurones fait-il réellement mieux que des modèles
simples pour prédire les rendements d'ETF, une fois validation et coûts traités correctement ?*
La réponse peut très bien être « non », et ce serait un résultat valide.

Nous étudions un MLP volontairement petit (2 couches cachées, régularisation L2 et dropout,
arrêt anticipé), entraîné avec 5 graines aléatoires, et le comparons à trois références : un
prédicteur naïf (momentum 20 jours), une régression Ridge et un modèle CatBoost. L'univers est composé
de 36 ETF liquides américains couvrant 8 classes d'actifs, avec des données journalières à
partir de 2013-01-01. Chaque pli est évalué dans le temps, sans validation croisée aléatoire.

---

## 2. Données

- **Source** : Yahoo Finance (`yfinance`), prix ajustés et volumes journaliers.
- **Univers** : 36 ETF liquides américains, couvrant 8 classes d'actifs (actions US, internationales,
  secteurs, obligations, matières premières, immobilier coté).
- **Période** : à partir de 2013-01-01, avec un historique minimum de 2 520 jours (~10 ans).
- **Filtrage** : les ETF sans assez d'historique sont exclus plutôt que complétés.

### Limites des données

1. **Biais du survivant résiduel** : seuls les ETF toujours cotés sont présents.
2. **Qualité des données gratuites** : trous et ajustements approximatifs possibles.
3. **Horizon 5 jours, données journalières** : aucune conclusion sur d'autres fréquences.

---

## 3. Méthodologie

### 3.1 Cible et variables

- **Cible** : rendement de t+1 à t+1+5 jours, entrée à la clôture de t+1 (retard d'exécution de 1 jour).
- **Variables** (6) : momentum à 1, 5, 20 et 60 jours ; écart-type des rendements sur 5 et 20 jours ;
  volume relatif. Toutes calculées uniquement avec le passé.
- **Standardisation** : z-score transversal (date par date, entre actifs), n'utilisant aucune
  information du futur.

### 3.2 Découpage temporel

| Paramètre | Valeur |
|---|---|
| Fenêtre d'entraînement initiale | 1 008 jours (~4 ans) |
| Blocs de test | 60 jours (~3 mois), non chevauchants |
| Gap (purge) | 10 jours (> horizon 5 j + retard 1 j) |
| Validation interne | 20 % de la fenêtre d'entraînement |
| Nombre de plis | 27 |

Aucune validation croisée aléatoire : la fenêtre d'entraînement grandit au fil du temps et les
hyperparamètres sont choisis sur la validation interne (dans la fenêtre d'entraînement).

### 3.3 Modèles

| Modèle | Rôle |
|---|---|
| Naïf (momentum 20 j) | Plancher de comparaison |



| Ridge | Référence linéaire |
| CatBoost | Référence non linéaire |
| MLP (PyTorch) | Modèle étudié (5 graines) |

### 3.4 Mesures et statistiques

- **IC** : corrélation de Pearson, à chaque date, entre prédiction et rendement.
- **IC de rang** : corrélation de Spearman, même procédé.
- **ICIR** : moyenne / écart-type de la série d'IC.
- **Tests** : test apparié sur la différence d'IC, erreurs corrigées de Newey-West (6 lags),
  et bootstrap par blocs (blocs de 10 dates, 2 000 itérations).
- **Portefeuille** : long-short quintile, rebalancement tous les 5 jours, coûts de 10 bps par aller
  appliqués au turnover réel.

## 4. Résultats

### 4.1 Tableau de synthèse

| Modèle | IC de rang moyen | ICIR | t (Newey-West) | Sharpe L/S après 10 bps |
|---|---:|---:|---:|---:|
| Naïf | –0,002 | –0,002 | –0,07 | –0,42 |
| Ridge | 0,043 | 0,10 | 4,0 | –0,01 |
| CatBoost | 0,049 | 0,13 | 5,1 | 0,08 |
| MLP (moy. 5 graines) | 0,052 | 0,12 | 4,9 | –0,18 |

L'IC moyen du MLP est de 0,044 (données non présentées ici), légèrement au-dessus du CatBoost.

### 4.2 Test des différences

| Comparaison | n | t (Newey-West) | p | Diff. IC moyenne | IC 95 % |
|---|---:|---:|---:|---:|---:|
| MLP − Naïf | 1 620 | 1,67 | 0,10 | +0,045 | [−0,009 ; +0,097] |
| MLP − Ridge | 1 620 | 0,82 | 0,44 | +0,007 | [−0,010 ; +0,024] |
| MLP − CatBoost | 1 620 | 0,40 | 0,66 | +0,002 | [−0,009 ; +0,014] |

**Conclusion :** on ne peut pas rejeter l'hypothèse nulle qu'il n'y a pas de différence entre le MLP
et les modèles de référence.

### 4.3 Stabilité dans le temps

L'IC de rang est variable dans le temps : positif la plupart des années, mais parfois légèrement
négatif, sans régime de marché stable sur lequel le modèle vaudrait la peine. L'IC cumulé
oscille autour de zéro.

### 4.4 Ablation de variables

- Retirer le groupe **volatilité** : l'IC de rang tombe à ~0.
- Retirer le groupe **momentum** : l'IC ne bouge presque pas.
- Retirer le groupe **volume** : quasi aucun effet.

**Interprétation :** le signal capturé ressemble plus à l'anomalie de faible volatilité qu'au momentum.

### 4.5 Contrôles négatifs

- Cible aléatoire (mélangée) : IC de rang ≈ 0 (test passé).
- Variable volontairement fuyante : IC explose (test passé).

Les contrôles valident le protocole.

---

## 5. Discussion

### 5.1 Ce que le résultat permet de conclure

- Le MLP **n'améliore pas** significativement la prédiction par rapport aux modèles simples sur
  cet univers, à cette fréquence, avec ce protocole.
- Le signal existe (IC ~0,04), mais est trop dilué pour être monétisé sous forme de long-short
  simple après coûts.
- La démarche est rigoureuse : purge, embargo, validation interne, backtest après coûts,
  tests anti-fuite, contrôles négatifs.

### 5.2 Ce que le résultat ne permet pas de conclure

- Pas de conclusion sur d'autres fréquences (haute fréquence, mensuelle...).
- Pas de conclusion sur d'autres architectures (LSTM, Transformer...).
- Pas de conclusion sur d'autres univers (actions individuelles, autres marchés...).

---

## 6. Limites et pistes

- Univers restreint (36 ETF, marché américain dominant).
- Biais du survivant résiduel.
- Qualité des données gratuites.
- Horizon 5 jours, données journalières uniquement.
- Coûts hypothétiques (pas d'impact de marché).
- Puissance statistique faible (nombre de dates corrélées).
- Un seul réseau : on ne peut pas généraliser à « tous les réseaux de neurones ».

### Pistes

- Univers plus large et comparaison actions / ETF.
- Variables fondamentales, sentiment, microstructure.
- Modèles plus complexes (LSTM, Transformer), toujours avec le protocole de l'étude actuelle.
- Optimisation de portefeuille plus réaliste.

---

## 7. Conclusion

**Question :** un MLP écrit en PyTorch prédit-il les rendements à 5 jours mieux que des modèles
simples, une fois validation et coûts traités correctement ?

**Réponse :** **non, pas de façon statistiquement significative et stable.**  
Le MLP donne un IC de rang moyen de 0,044, légèrement au-dessus du CatBoost (0,042), mais la
différence est de l'ordre du bruit (test apparié : t = 0,40, p = 0,66 ; IC 95 % : [−0,009 ; +0,014]).
Le signal que les modèles capturent est réel (IC de rang ~0,04, t ≈ 5 contre un naïf nul),
mais il est trop dilué sur 36 ETF corélés pour justifier de ne pas le manger avec un portefeuille
long-short après 10 bps de coûts. Le MLP n'est donc ni un alpha, ni une preuve que le deep learning
est inutile ici — c'est une **preuve de ce que la méthode, bien protégée, ne permet pas d'extraire
d'un signal faible et corrélé**.

**Recommandation :** ne pas compromettre la méthode pour obtenir un alpha. Les données ne le
permettent pas encore. Un prochain lot valide serait worth it : univers plus large, variables
fondamentales ou de microstructure, horizon plus long, et comparaison avec des modèles
plus puissants — toujours avec le protocole de l'étude actuelle.

---

## Annexes

- A : liste des 36 ETF.
- B : hyperparamètres (configs/walkforward.yaml, features.yaml, configs/universe.yaml).
- C : code et données disponibles au dépôt (voir README.md).

---

*Version courte de la section 14 du dépôt HonestAlpha (doc / detail.md). L'article complet,
  avec figures, figures annexes et LaTeX, sera fourni dans le dépôt.*




