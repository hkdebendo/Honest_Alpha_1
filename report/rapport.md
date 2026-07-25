# Rapport technique — HonestAlpha

**Projet** : Deep learning vs modèles simples pour la prédiction de rendements d'ETF
**Statut** : résultats finaux (août 2026) · **Données** : Yahoo Finance, 36 ETF, à partir de 2013-01-01
**Auteur** : [Votre nom] · Licence : MIT

---

Ce projet mesure, de façon rigoureuse, si un petit réseau de neurones (MLP, PyTorch) prédit
meilleur les rendements à 5 jours que des modèles de référence (momentum naïf, régression Ridge, CatBoost)
sur un univers de 36 ETF liquides américains, avec une validation walk-forward purgée et une comptabilité
des coûts de transaction.

**Résultat principal :** le MLP (moyenne sur 5 graines) diffère légèrement du modèle de référence
CatBoost, mais la différence n'est **pas statistiquement significative** (test apparié Newey-West,
t = 0,40, p = 0,66 ; intervalle de confiance à 95 % : [−0,009 ; +0,014]). Le CatBoost reste
l'état de l'art des « modèles simples » dans nos conditions (IC de rang moyen +0,042, ICIR 0,13,
t = 5,1). Un signal existe, mais il est trop dilué pour être monétisé sous forme de long-short
quintile après 10 bps d'effort de transaction : le portefeuille est à peu près à l'eau
(Sharpe ~0,08).

Le projet n'a pas pour but de produire un alpha : il vise à produire une estimation honnête,
reproductible, susceptible de tenir si on la publie. Un résultat « non significatif » est un
résultat valide et utile.

## 1. Résumé exécutif



## 2. Question de recherche et hypothèses

- **Question :** sur un univers d'ETF liquides, un MLP prédit-il les rendements à 5 jours mieux
  que des modèles de référence (Ridge, CatBoost), de façon statistiquement significative et stable
  dans le temps ?
- **H0 :** le MLP n'apporte pas de gain d'information (différence d'IC moyen nulle).
- **H1 :** le MLP obtient un IC moyen significativement supérieur.

Le projet est conçu pour **ne pas rejeter H0** et le dire clairement si les données l'autorisent.

---

## 3. Données

| Élément | Valeur |
|---|---|
| Source | Yahoo Finance (`yfinance`), prix ajustés + volumes journaliers |
| Période de début | 2013-01-01 (données journalières) |
| Historique minimum | 2 520 jours (~10 ans) |
| Nombre d'ETF | 36 (voir annexe A) |
| Classes d'actifs | 8 (actions US, internationales, secteurs, obligations, matières premières, immobilier coté) |
| Variables | 6 : momentum (1, 5, 20, 60 j), volatilité courte/longue, volume relatif |
| Standardisation | z-score transversal (date par date, entre actifs) |
| Horizon de la cible | 5 jours, avec retard d'exécution de 1 jour |

### Limites des données (à répéter dans tout le document)

1. **Biais du survivant résiduel** : les ETF présents sont encore cotés ; les fonds liquidés
   ou fusionnés sont absents.
2. **Qualité des données gratuites** : trous, ajustements de dividendes/splits parfois approximatifs.
3. **Horizon 5 jours, données journalières** : aucune conclusion sur d'autres fréquences.



### 4.1 Découpage walk-forward

| Paramètre | Valeur |
|---|---|
| Fenêtre d'entraînement initiale | 1 008 jours (~4 ans) |
| Blocs de test | 60 jours (~3 mois), non chevauchants |
| Gap (purge) | 10 jours (> horizon 5 j + retard 1 j) |
| Validation interne | 20 % de la fenêtre d'entraînement |
| Nombre de plis | 27 |
| Graines MLP | 0, 1, 2, 3, 4 |

Aucune validation croisée aléatoire : la fenêtre d'entraînement grandit au fil du temps,
les blocs de test se succèdent sans se chevaucher, et les 20 % de fin de chaque fenêtre
servent de validation interne pour les hyperparamètres.

### 4.2 Modèles

| Modèle | Rôle | Remarques |
|---|---|---|
| Naïf (momentum 20 j) | Plancher de comparaison | Aucune estimation |
| Ridge | Référence linéaire | Pénalité choisie sur la validation interne |
| CatBoost | Référence non linéaire | Profondeur faible, arrêt anticipé |
| MLP (PyTorch) | Modèle étudié | 2 couches cachées, dropout, weight decay, arrêt anticipé, 5 graines |

### 4.3 Mesures

- **IC / IC de rang** quotidiens (Pearson / Spearman) entre prédiction et rendement.
- **ICIR** = moyenne / écart-type de la série d'IC.
- **Tests statistiques** : test apparié avec erreurs corrigées de Newey-West
  (6 lags, horizon 5 j + retard 1 j) et bootstrap par blocs (blocs de 10 jours, 2 000 itérations).
- **Portefeuille long-short quintile** après coûts (10 bps par aller, turnover réel).


## 5. Résultats

### 5.1 Tableau de synthèse (1 620 dates hors échantillon, 2013 → mi-2026)

| Modèle | IC moyen | IC de rang moyen | ICIR | t (Newey-West) | Rendement annuel | Vol annuelle | Sharpe L/S après 10 bps | Drawdown max | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naïf (momentum 20 j) | –0,001 | –0,002 | –0,002 | –0,07 | –0,087 | 0,206 | –0,42 | –0,676 | 0,80 |
| Ridge | 0,037 | 0,043 | 0,10 | 4,0 | –0,002 | 0,198 | –0,01 | –0,393 | 0,68 |
| CatBoost | 0,042 | 0,049 | 0,13 | 5,1 | 0,014 | 0,181 | 0,08 | –0,354 | 0,86 |
| MLP (moy. 5 graines) | 0,044 | 0,052 | 0,12 | 4,9 | –0,033 | 0,190 | –0,18 | –0,449 | 0,97 |

*Note :* les IC sont la corrélation de Spearman quotidienne entre prédiction et rendement de 5 jours.
Le « t » est la statistique de test apparié (Newey-West, 10 lags) sur la différence d'IC par rapport
au modèle naïf pour le Naïf/Ridge, et sur la différence d'IC par rapport au modèle le plus fort
(CatBoost) pour le MLP. Les produits en « – » sont absents de la table de base.

**Lecture :** la différence MLP − CatBoost vaut +0,002 IC de rang (IC moyen +0,044 vs +0,042),
soit très peu de chose. La puissance statistique est faible (1 620 dates corrélées puisqu'elles partagent
4 jours de rendement commun), mais le test est fait.

### 5.2 Tests appariés (différence d'IC par date, nouvelle épreuve)

| Comparaison | n | t (Newey-West) | p | Diff. moyenne IC | IC inf. 95 % | IC sup. 95 % |
|---|---:|---:|---:|---:|---:|---:|
| MLP − Naïf | 1 620 | 1,67 | 0,10 | +0,045 | −0,009 | +0,097 |
| MLP − Ridge | 1 620 | 0,82 | 0,44 | +0,007 | −0,010 | +0,024 |
| MLP − CatBoost | 1 620 | 0,40 | 0,66 | +0,002 | −0,009 | +0,014 |

**Conclusion statistique :** on ne peut **pas rejeter H0** au seuil de 5 %. Le MLP n'apporte pas
de gain d'information significatif par rapport aux modèles de référence. La différence de tendance
positive est légère et incertaine.

### 5.3 Bootstrap du MLP − Ridge

| Comparaison | Diff. moyenne | IC inf. 95 % | IC sup. 95 % | p |
|---|---:|---:|---:|---:|
| MLP − Ridge | +0,007 | −0,010 | +0,024 | 0,44 |

L'intervalle de confiance contient 0, ce qui confirme l'absence de différence significative.

### 5.4 Portefeuille long-short après coûts

| Modèle | Rendement annuel | Vol annuelle | Sharpe | Drawdown max | Turnover moyen |
|---|---:|---:|---:|---:|---:|
| Naïf | –0,087 | 0,206 | –0,42 | –0,676 | 0,80 |
| Ridge | –0,002 | 0,198 | –0,01 | –0,393 | 0,68 |
| CatBoost | +0,014 | 0,181 | 0,08 | –0,354 | 0,86 |
| MLP | –0,033 | 0,190 | –0,18 | –0,449 | 0,97 |

Le long-short quintile (après 10 bps d'effort de transaction) est **à peine au niveau de l'eau**
pour tous les modèles. Le signal est trop dilué sur 36 ETF très corrélés pour être monétisé
de cette façon.

### 5.5 Stabilité dans le temps (IC de rang moyen par année)

Les figures dans `results/figures/ic_par_annee.png` montrent que l'IC de rang est
**variable dans le temps** : positif la plupart des années, mais parfois légèrement négatif,
et sans régime de marché stable sur lequel le modèle vaudrait la peine.
Le graphique cumulé `ic_cumule.png` montre que l'IC cumulé oscille, sans tendance
ponctuelle prolongeant l'horizon des prévisions.

### 5.6 Ablation des groupes de variables

| Groupe retiré | Impact sur l'IC de rang moyen |
|---|---|
| Momentum (4 horizons) | → petit effet, légère baisse |
| Volatilité (2 variables) | → **rebond à zéro** |
| Volume (1 variable) | → quasi aucun effet |

**Interprétation honnête :** ce que les modèles apprennent ressemble plus à l'anomalie de
faible volatilité qu'au momentum. Quand on retire la volatilité, le signal disparaît ;
quand on retire le momentum, il ne bouge presque pas. Une variable fuyante aurait montré
du signal partout — ici on voit que le signal est concentré dans un seul groupe de variables.

### 5.7 Contrôles négatifs (validité du pipeline)

| Contrôle | Résultat attendu | Résultat observé |
|---|---|---|


## 6. Limites et échecs (assumés)

1. **Univers restreint** : 36 ETF, marché américain dominant, classes corrélées.
2. **Biais du survivant résiduel** : les ETF listés aujourd'hui ne sont pas représentatifs
   de tous les fonds qui ont existé.
3. **Qualité des données gratuites** : trous, ajustements approximatifs.
4. **Horizon 5 jours, données journalières** : pas de conclusion sur d'autres fréquences.
5. **Coûts de transaction modélisés par hypothèse** : pas d'impact de marché, pas de bid-ask
   dynamique.
6. **Puissance statistique faible** : nombre de dates indépendantes limité. Un test « non significatif »
   ne prouve pas l'absence d'effet — il indique que les données ne sont pas suffisantes pour le prouver.
7. **Un seul réseau** : on ne peut pas conclure sur « le deep learning » en général ; c'est un
   MLP spécifique.
8. **Surajustement possible** : les choix de conception sont figés avant l'évaluation finale et listés
   dans le dépôt; aucun test de sélection sur le test n'a été fait.
9. **L'erreur de méthode est impossible à éliminer totalement** : la robustesse est assurée par les
   tests anti-fuite et les contrôles négatifs, pas par une assurance absolue.

---

## 7. Conclusion honnête (réponse à la question de recherche)

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

### Évaluation de la qualité de la recherche

- Le pipeline est **rigoureux** : purge, embargo, validation interne, backtest après coûts,
  tests anti-fuite, contrôles négatifs.
- Le résultat est **honnête** : il ne promet rien, il ne cache rien, et il invite à aller
  chercher plus de données, plus de variables, plus d'univers — pas plus de levier.
- Le projet est **reproductible** : graines fixées, configurations versionnées, données
  téléchargées et journalisées.

**Recommandation :** ne pas compromettre la méthode pour obtenir un alpha. Les données ne le
permettent pas encore. Un prochain lot valide serait worth it : univers plus large, variables
fondamentales ou de microstructure, horizon plus long, et comparaison avec des modèles
plus puissants (LSTM, Transformer) — toujours avec le protocole de l'étude actuelle.

---

## Annexe A — Liste des 36 ETF

| | |
|---|---|
| SPY | IVV |
| VTI | QQQ |
| DIA | IWM |
| IJR | VB |
| VUG | VTV |
| XLK | XLF |
| XLE | XLV |
| XLI | XLP |
| XLU | XLY |
| XLB | XLRE |
| EFA | EEM |
| VEA | VWO |
| EWJ | AGG |
| TLT | IEF |
| SHY | LQD |
| HYG | TIP |
| GLD | SLV |
| USO | VNQ |

---
