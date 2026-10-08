# HonestAlpha_1

Prédire les rendements boursiers est facile. Se tromper en croyant qu'on
prédit, c'est encore plus facile. Ce projet est un petit laboratoire
monté pour faire la part des choses : un pipeline complet de prédiction
de rendements d'ETF, conçu pour que chaque source classique de fuite de
données soit fermée, testée, et affichée publiquement quand elle coûte
quelque chose.

Le code fait peu de choses, mais il les fait dans l'ordre :

1. télécharger un univers figé d'ETF liquides (yfinance) ;
2. construire quelques variables techniques standardisées en coupe
   transversale, et une cible = rendement futur avec un jour de retard
   d'exécution ;
3. entraîner plusieurs familles de modèles en walk-forward purgé
   (fenêtre d'entraînement < validation < test, jamais l'inverse) ;
4. mesurer : IC de Spearman quotidien, ICIR, portefeuille long-short
   après coûts, tests appariés de Newey-West, bootstrap par blocs ;
5. casser le pipeline volontairement pour vérifier qu'il détecte les
   fuites qu'on cherche à éviter.

Le résultat court : sur 36 ETF américains, de 2013 à mi-2026, les
modèles apprennent bien mieux qu'un momentum naïf — l'IC de rang du
CatBoost hors échantillon ressort à ~0,04, t ≈ 5 — mais une fois les
coûts de transaction passés, le portefeuille long-short est à peine au
niveau de l'eau. C'est le genre de conclusion qu'on n'obtient pas avec
un backtest qui fuit.

## Installation

Python 3.10 ou plus récent. Le reste est standard :

```
pip install -r requirements.txt
```

PyTorch et CatBoost sont optionnels au sens strict : le pipeline
détecte ce qui manque, prévient, et continue avec les modèles
disponibles. Sans torch, la partie MLP est simplement ignorée.

## Utilisation

Tout passe par la même commande, étape par étape :

```
python -m src.run data       # téléchargement + nettoyage
python -m src.run features   # variables + cible
python -m src.run run        # walk-forward complet
python -m src.run controls   # contrôles négatifs
python -m src.run ablation   # retrait de chaque groupe de variables
python -m src.run report     # tableaux + figures
```

Ou avec make : `make all` enchaîne data → features → run → report.
Les tests anti-fuite se lancent avec `make test`.

Tous les paramètres vivent dans trois petits fichiers YAML (`configs/`) :

## Ce qui est fait pour éviter les fuites

Les fuites de données sont rarement un gros bug visible ; c'est plutôt
une série de détails, chacun anodin. Voici ceux auxquels ce pipeline
fait attention, et comment il le vérifie.

**La cible.** Le rendement prédit est celui de t+1 à t+1+5 : on entre à
la clôture du lendemain du signal, pas au prix qui a servi à le
calculer. Les fenêtres de test sont séparées de l'entraînement par un
gap plus long que l'horizon, sinon les cibles qui se chevauchent
laissent passer de l'information du test dans le train.

**Les variables.** Uniquement du passé : momentum sur plusieurs
horizons, volatilité courte et longue, volume relatif. Standardisées
date par date entre actifs (z-score transversal), jamais sur la période
entière — un z-score calculé avec la moyenne de l'année complète
contient le futur, même quand la variable elle-même est saine.

**Le découpage.** Walk-forward extensible : la fenêtre d'entraînement
grandit au fil du temps, les blocs de test se suivent sans se
chevaucher, et les 20 % de fin de chaque fenêtre servent de validation
pour les hyperparamètres. Il n'y a pas de sélection de modèle sur le
test, et le test n'est jamais réutilisé.

**Les preuves, pas les promesses.** Deux contrôles négatifs tournent
sur les données réelles (`python -m src.run controls`) :

- cibles mélangées en coupe transversale : l'IC doit retomber vers 0.
  Observé : -0,03. Vérifié.
- une variable qui contient la cible (rang + bruit) : l'IC doit
  exploser. Observé : 0,99. Si le pipeline était incapable de voir une
  fuite aussi grossière, il ne verrait jamais les subtiles.

Les tests unitaires (`tests/`) complètent : modifier les prix du futur
ne change pas les variables du passé, la cible démarre bien à t+1, les
fenêtres sont purgées et ordonnées, un signal synthétique est bien
retrouvé et un bruit mélangé bien ignoré.

## Les modèles

Quatre points de comparaison, du plus simple au moins simple, mesurés
sur exactement le même découpage :

- **naïf** : le momentum 20 jours brut, sans apprentissage. La
  référence à battre, et pas si facile à battre.
- **ridge** : régression linéaire régularisée, entraînée à prédire les
  rendements futurs par actif.

## Ce qu'on trouve (et ce qu'on n'y trouve pas)

Mesuré sur 27 plis de 3 mois, de janvier 2020 à juillet 2026 :

| Modèle | IC rang moyen | ICIR | t (Newey-West) | Sharpe L/S après 10 bps |
|--------|---------------|------|----------------|-------------------------|
| naïf | -0,001 | -0,002 | -0,07 | -0,42 |
| ridge | 0,037 | 0,10 | 4,0 | -0,01 |
| catboost | 0,042 | 0,13 | 5,1 | 0,08 |

Lecture honnête : les variables portent bien un peu d'information — un
IC de rang de 0,04 avec un t à 5 sur six ans et demi, c'est du signal
réel. Mais sur 36 ETF très corrélés entre eux, le long-short quintile
après 10 bps de coûts ne rapporte à peu près rien. L'information
existe, elle est trop diluée pour être monétisée comme ça. Ce n'est pas
un échec du pipeline ; c'est ce qu'il devait mesurer.

L'ablation par groupe de variables (`results/ablation.csv`) et la
sensibilité aux coûts (`results/figures/couts.png`) sont là pour
contextualiser. Et l'ablation réserve une surprise honnête : retirez le
groupe volatilité et l'IC retombe à zéro ; retirez le momentum et il
ne bouge presque pas. Autrement dit, ce que ces modèles apprennent sur
cet univers ressemble surtout à l'anomalie de faible volatilité, pas à
du momentum — un résultat de ce genre, c'est exactement ce qu'une
variable fuitée aurait masqué en montrant partout du "signal".

## Limites, assumées

- 36 ETF, un marché à dominante américaine, des classes d'actifs
  corrélées. La prédiction transversale est d'autant plus dure que les
  actifs bougent ensemble.
- Pas de données fondamentales, de sentiment, de microstructure. Sept
  variables techniques, c'est peu — c'est aussi ce qui rend les
  résultats difficiles à fausser par accident.
- Le long-short quintile est la manière la plus brute de monétiser un
  IC. Une vraie optimisation de portefeuille en ferait plus avec moins
  de turnover ; ce n'est pas le sujet ici.
- Un backtest, même purgé, reste un backtest. Le but de ce projet est
  de garantir l'absence de fuite, pas de promettre du rendement.

## Structure

```
configs/         univers, variables, walk-forward (la seule source de paramètres)
src/
  data.py        téléchargement, nettoyage, journal des exclusions
  features.py    variables, cible sans fuite, standardisation transversale
  walkforward.py découpage temporel purgé
  models/        ridge, momentum naïf, MLP torch, catboost
  metrics.py     IC quotidien, ICIR
  portfolio.py   portefeuille quintile après coûts
  stats.py       Newey-West, bootstrap par blocs, tests appariés
  run.py         la ligne de commande, tout part de là
tests/           tests anti-fuite et contrôles négatifs sur données synthétiques
results/         CSV, figures, tests statistiques
```


## Rapports

Ce dossier contient les deux versions rédigées des résultats du projet :

| Fichier | Rôle |
|---|---|
| `rapport.md` | Rapport complet, sections 1–7 + annexe A, avec les tableaux de synthèse, les tests appariés Newey-West, la bootstrap, la littérature de contrôle et la conclusion honnête. |
| `article.md` | Version courte, bilingue (français/anglais), conçue pour être soumise à un journal : résumé, introduction, méthodes, résultats, discussion, limitations, conclusion. |

Les deux documents répondent à la même question de recherche :
*un MLP (PyTorch) prédit-il les rendements à 5 jours mieux que des modèles simples, une fois validation et coûts traités correctement ?*

**Réponse :** non, pas de façon statistiquement significative et stable.
Le MLP donne un IC de rang moyen de 0,044, légèrement au-dessus du CatBoost (0,042), mais la
différence est de l'ordre du bruit (test apparié : t = 0,40, p = 0,66 ; IC 95 % : [−0,009 ; +0,014]).
Le signal est réel (IC de rang ≈ 0,04), mais trop dilué sur 36 ETF corélés pour justifier de
monétiser un long-short après 10 bps de coûts. Le MLP n'est donc ni un alpha, ni une
preuve que le deep learning est inutile ici — c'est une preuve de ce que la méthode, bien
protégée, ne permet pas d'extraire d'un signal faible et corrélé.

### Tableaux inclus

| Tableau | Contenu |
|---|---|
| Synthèse des modèles | IC de rang moyen, ICIR, t (Newey-West), Sharpe L/S après 10 bps |
| Tests appariés | Comparaison MLP vs Naïf, Ridge, CatBoost : t, p, IC 95 % |
| Bootstrap MLP − Ridge | Différence moyenne, IC 95 %, p |
| Long-short quintile | Rendement annuel, vol, Sharpe, drawdown, turnover |
| Ablation des variables | Impact de la suppression d'un groupe de variables sur l'IC de rang |
| Contrôles négatifs | Contrôle de la fuite : résultat attendu vs observé |

### Limites honorées

- Biais du survivant résiduel (36 ETF, marché américain dominant).
- Qualité des données Yahoo Finance gratuites (trous, ajustements approximatifs).
- Horizon 5 jours, données journalières.
- Coûts hypothétiques (10 bps) : pas d'impact de marché, pas de bid-ask dynamique.
- Puissance statistique faible : nombre de dates corrélées. Un « non significatif »
  ne prouve pas l'absence d'effet, il indique que les données ne sont pas suffisantes pour le prouver.
- Un seul réseau : on ne généralise pas à « tous les réseaux de neurones ».

---

## Licence et usage



Code fourni tel quel, à des fins d'étude. Ce projet ne constitue pas un
conseil en investissement, et rien dans ces résultats ne suggère qu'un
IC de 0,04 justifie d'allouer le moindre euro. Il justifie d'aller
chercher plus de données, pas plus de levier.

- **mlp** : petit réseau torch, entraîné sur 5 graines différentes. On
  rapporte la moyenne des prédictions et surtout la dispersion entre
  graines — un MLP dont les résultats changent du tout au tout selon la
  graine n'a rien appris de stable.
- **catboost** : gradient boosting, comme point de comparaison non
  linéaire en plus du MLP.

l'univers et la période, la définition des variables et de la cible, le
découpage temporel et les coûts. Rien d'important n'est codé en dur.
Les résultats atterrissent dans `results/` (CSV + figures PNG), les
données intermédiaires dans `data/processed/`.
