"""IC, IC de rang, ICIR.

L'IC est la corrélation, à chaque date, entre les prédictions et les
rendements réalisés (Pearson) ; l'IC de rang fait la même chose en
Spearman. L'ICIR est le rapport moyenne / écart-type de la série d'IC.

Les cibles à 5 jours se chevauchent d'un jour sur l'autre, donc les IC
successifs sont autocorrélés : la correction est dans stats.py
(Newey-West, bootstrap par blocs), pas ici.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ACTIFS = 3  # en dessous, une corrélation transversale ne veut rien dire


def to_wide(series):
    """Série avec index (date, ticker) -> DataFrame dates x tickers."""
    return series.unstack("ticker")


def daily_corr(pred, realized, method="pearson"):
    """Série d'IC par date entre deux panels dates x actifs."""
    cols = pred.columns.intersection(realized.columns)
    idx = pred.index.intersection(realized.index)
    out = {}
    for d in idx:
        a = pred.loc[d, cols]
        b = realized.loc[d, cols]
        m = a.notna() & b.notna()
        if m.sum() < MIN_ACTIFS:
            continue
        c = a[m].corr(b[m], method=method)
        if pd.notna(c):
            out[d] = c
    return pd.Series(out, name=method)


def ic_summary(ic):
    """Moyenne, écart-type, ICIR et t naïf d'une série d'IC par date."""
    ic = pd.Series(ic).dropna()
    n = len(ic)
    if n == 0:
        return {"n_dates": 0, "ic_moyen": np.nan, "ic_std": np.nan,
                "icir": np.nan, "t_naif": np.nan}
    mean, std = ic.mean(), ic.std()
    return {
        "n_dates": n,
        "ic_moyen": mean,
        "ic_std": std,
        "icir": mean / std if std > 0 else np.nan,
        "t_naif": mean / std * np.sqrt(n) if std > 0 else np.nan,
    }
