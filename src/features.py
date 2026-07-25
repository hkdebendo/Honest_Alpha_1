"""Variables et cible, sans fuite d'information.

Deux règles appliquées partout, et testées dans tests/ :

- une variable à la date t ne regarde que les données <= t ;
- la cible à la date t est le rendement logarithmique de la clôture de
  t+retard à la clôture de t+retard+horizon (on ne peut pas acheter au
  prix qui a servi à calculer le signal), puis transformée en rang
  transversal entre actifs.

La standardisation est transversale (par date, entre actifs), donc
n'utilise aucune information du futur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def xs_zscore(df):
    """Z-score entre actifs, date par date. Rien n'est calculé sur la période entière."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def forward_returns(prices, horizon, retard):
    """Rendement log de la clôture de t+retard à la clôture de t+retard+horizon."""
    p_entree = prices.shift(-retard)
    p_sortie = prices.shift(-(retard + horizon))
    return np.log(p_sortie / p_entree)


def groups_for(cfg):
    """Mapping groupe -> colonnes, pour les ablations."""
    mom = [f"mom_{h}" for h in cfg["variables"]["momentum"]]
    return {
        "momentum": mom,
        "volatilite": ["vol_20", "vol_ratio"],
        "volume": ["rel_volume"],
    }


def build_features(prices, volumes, cfg):
    """Construit les variables en format large (dates x actifs), standardisées."""
    rets = np.log(prices).diff()
    v = cfg["variables"]

    feats = {}
    for h in v["momentum"]:
        feats[f"mom_{h}"] = rets.rolling(h).sum()

    vol_longue = rets.rolling(v["vol_longue"]).std()
    vol_courte = rets.rolling(v["vol_courte"]).std()
    feats["vol_20"] = vol_longue
    feats["vol_ratio"] = vol_courte / vol_longue

    vol_moy = volumes.rolling(v["volume_mm"]).mean()
    feats["rel_volume"] = volumes / vol_moy - 1.0

    return {k: xs_zscore(f) for k, f in feats.items()}


def make_dataset(prices, volumes, cfg):
    """Retourne le panel prêt pour les modèles.

    - X : DataFrame long, index (date, ticker), colonnes = variables ;
    - y : rang transversal de la cible (la chose qu'on prédit) ;
    - fwd : rendements futurs bruts en large, pour les métriques et le
      portefeuille.
    """
    feats = build_features(prices, volumes, cfg)
    horizon = cfg["cible"]["horizon"]
    retard = cfg["cible"]["retard"]

    fwd = forward_returns(prices, horizon, retard)
    ranks = fwd.rank(axis=1, pct=True)

    # passage en long ; concat fait une jointure externe, puis on vire les
    # lignes incomplètes (débuts de période où les moyennes glissantes
    # ne sont pas encore définies)
    X = pd.concat({k: f.stack() for k, f in feats.items()}, axis=1)
    X.index.names = ["date", "ticker"]
    X = X.dropna()

    y = ranks.stack().reindex(X.index).dropna()
    X = X.loc[y.index]

    return {"X": X, "y": y, "fwd": fwd}
