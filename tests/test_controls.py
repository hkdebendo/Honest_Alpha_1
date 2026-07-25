"""Contrôles négatifs sur données synthétiques, avec Ridge.

Si ces tests échouent, le pipeline a un problème de méthode, pas les
modèles :

- entraîné sur des cibles mélangées, un modèle ne doit rien trouver ;
- si on lui tend une variable qui contient la cible, il doit la trouver
  tout de suite -- sinon on ne détecterait jamais une vraie fuite.
"""
import numpy as np
import pandas as pd

from src.features import make_dataset
from src.metrics import daily_corr, to_wide
from src.models.baselines import RidgeBaseline
from src.walkforward import make_folds


def marche_avec_momentum(n_jours=800, n_actifs=8, phi=0.6, graine=0):
    """Rendements AR(1) par actif : hier aide un peu à prédire demain.

    phi < 1 garantit un processus stationnaire (la première version de ce
    générateur explosait et remplissait le panel de NaN -- laissé ici pour
    mémoire).
    """
    rng = np.random.default_rng(graine)
    eps = rng.normal(0.0, 0.01, (n_jours, n_actifs))
    rets = np.empty_like(eps)
    rets[0] = eps[0]
    for t in range(1, n_jours):
        rets[t] = phi * rets[t - 1] + eps[t]
    index = pd.bdate_range("2015-01-01", periods=n_jours)
    colonnes = [f"A{i}" for i in range(n_actifs)]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                          index=index, columns=colonnes)
    # volumes avec variation par actif, sinon le z-score transversal de
    # rel_volume est indéfini (écart-type nul -> NaN -> panel vide)
    volumes = pd.DataFrame(rng.normal(1e6, 1e5, (n_jours, n_actifs)),
                           index=index, columns=colonnes)
    return prices, volumes


def cfg_variables():
    return {
        "cible": {"horizon": 5, "retard": 1},
        "variables": {"momentum": [1, 5, 20, 60], "vol_longue": 20,
                      "vol_courte": 5, "volume_mm": 20},
    }


def split_train_test(X, y):
    dates = np.array(sorted(X.index.get_level_values("date").unique()))
    n_train = int(len(dates) * 0.7)
    masque = X.index.get_level_values("date").isin(set(dates[:n_train]))
    return X[masque], y[masque], X[~masque], y[~masque]


def test_cible_melangee_ic_proche_de_zero():
    prices, volumes = marche_avec_momentum()
    ds = make_dataset(prices, volumes, cfg_variables())
    X, y, fwd = ds["X"], ds["y"], ds["fwd"]

    # mélange transversal des cibles, date par date
    rng = np.random.default_rng(7)
    y_large = to_wide(y)
    melange = y_large.apply(
        lambda row: pd.Series(rng.permutation(row.dropna().values), index=row.dropna().index),
        axis=1,
    )
    y_melange = melange.stack().reindex(X.index)

    Xtr, ytr, Xte, _ = split_train_test(X, y_melange)
    # validation interne : les 20 % de fin du train
    n_fit = int(len(Xtr) * 0.8)
    modele = RidgeBaseline().fit(Xtr.iloc[:n_fit], ytr.iloc[:n_fit],
                                Xtr.iloc[n_fit:], ytr.iloc[n_fit:])
    pred = modele.predict(Xte)

    ic = daily_corr(to_wide(pred), fwd, "spearman").mean()
    assert abs(ic) < 0.15, f"IC de {ic:.3f} sur cible mélangée : fuite probable"


def test_variable_fuitee_l_ic_explose():
    prices, volumes = marche_avec_momentum()
    ds = make_dataset(prices, volumes, cfg_variables())
    X, y, fwd = ds["X"], ds["y"], ds["fwd"]

    # la variable fuitée est la cible elle-même + un peu de bruit
    rng = np.random.default_rng(11)
    X_fuite = X.assign(fuite=y + rng.normal(0, 0.05, len(X)))

    Xtr, ytr, Xte, _ = split_train_test(X_fuite, y)
    n_fit = int(len(Xtr) * 0.8)
    modele = RidgeBaseline().fit(Xtr.iloc[:n_fit], ytr.iloc[:n_fit],
                                Xtr.iloc[n_fit:], ytr.iloc[n_fit:])
    pred = modele.predict(Xte)

    ic = daily_corr(to_wide(pred), fwd, "spearman").mean()
    assert ic > 0.5, f"IC de {ic:.3f} : le pipeline ne détecte pas les fuites"


def test_le_signal_est_detectable():
    """Le pipeline doit retrouver le momentum synthétique ; sinon les
    contrôles négatifs ne veulent rien dire."""
    prices, volumes = marche_avec_momentum()
    ds = make_dataset(prices, volumes, cfg_variables())
    X, y, fwd = ds["X"], ds["y"], ds["fwd"]

    Xtr, ytr, Xte, _ = split_train_test(X, y)
    n_fit = int(len(Xtr) * 0.8)
    modele = RidgeBaseline().fit(Xtr.iloc[:n_fit], ytr.iloc[:n_fit],
                                Xtr.iloc[n_fit:], ytr.iloc[n_fit:])
    pred = modele.predict(Xte)

    ic = daily_corr(to_wide(pred), fwd, "spearman").mean()
    assert ic > 0.1, f"IC de {ic:.3f} : signal synthétique non détecté"
