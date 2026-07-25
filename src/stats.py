"""Tests statistiques pour séries d'IC autocorrélées.

Les IC calculés à deux dates voisines partagent 4 jours de rendement
commun (horizon 5 jours, prédiction quotidienne), donc l'hypothèse
d'indépendance du t-test classique est fausse. Deux traitements :

- erreurs standard corrigées de l'autocorrélation (Newey-West) ;
- bootstrap par blocs, qui rééchantillonne des blocs entiers de dates
  pour préserver la structure de dépendance.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as scistats


def hac_std(x, lags):
    """Écart-type HAC (Newey-West) de la moyenne de x."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s2 = (e @ e) / n
    for l in range(1, min(lags, n - 1) + 1):
        gamma = (e[l:] @ e[:-l]) / n
        s2 += 2.0 * (1.0 - l / (lags + 1)) * gamma
    return np.sqrt(max(s2, 0.0) / n)


def ttest_hac(x, lags):
    """T-stat et p-value (bilatéral) sur la moyenne de x, erreurs Newey-West."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    se = hac_std(x, lags)
    if not np.isfinite(se) or se == 0:
        return np.nan, np.nan
    t = x.mean() / se
    p = 2.0 * scistats.t.sf(abs(t), df=n - 1)
    return t, p


def paired_ic_test(ic_a, ic_b, lags):
    """Test apparié sur la différence d'IC par date entre deux modèles."""
    diff = (ic_a - ic_b).dropna()
    if len(diff) < 10:
        return {"n": len(diff), "t": np.nan, "p": np.nan}
    t, p = ttest_hac(diff.values, lags)
    return {"n": len(diff), "t": t, "p": p}


def block_bootstrap_mean(x, taille_bloc=10, n_boot=2000, seed=0):
    """Bootstrap par blocs (circulaire) de la moyenne de x.

    Retourne l'intervalle à 95 % et une p-value bilatérale approximative
    pour H0 : moyenne nulle.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < taille_bloc:
        return {"ci_bas": np.nan, "ci_haut": np.nan, "p": np.nan}

    rng = np.random.default_rng(seed)
    n_blocs = int(np.ceil(n / taille_bloc))
    offsets = np.arange(taille_bloc)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocs)
        idx = (starts[:, None] + offsets) % n
        sample = x[idx].ravel()[:n]
        means[i] = sample.mean()

    ci_bas, ci_haut = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {"ci_bas": ci_bas, "ci_haut": ci_haut, "p": min(p, 1.0)}
