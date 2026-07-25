"""Portefeuille long/short simple, après coûts.

Contrôle de réalisme plutôt que stratégie : à chaque date de
rebalancement (une sur `rebalancement` parmi les dates hors échantillon),
on est long le quintile le mieux prédit et short le moins bien prédit,
à pondération égale. Les coûts sont une hypothèse paramétrable
(`couts_bps` par aller) appliquée au turnover réel, et on refait
l'analyse à plusieurs niveaux de coûts dans le rapport.

Un IC faible peut disparaître après coûts : c'est une information en soi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(pred_wide, daily_rets, rebalancement=5, quantiles=5, couts_bps=10.0):
    """Backtest du portefeuille. Retourne (rendements journaliers, turnover moyen)."""
    dates = [d for d in pred_wide.index if d in daily_rets.index]
    rebal_dates = dates[::rebalancement]

    poids_precedents = None
    segments, turnovers = [], []

    for t in rebal_dates:
        scores = pred_wide.loc[t].dropna()
        if len(scores) < 2 * quantiles:
            continue

        groupes = pd.qcut(scores.rank(method="first"), quantiles, labels=False)
        w = pd.Series(0.0, index=scores.index)
        haut = groupes == quantiles - 1
        bas = groupes == 0
        w[haut] = 1.0 / haut.sum()
        w[bas] = -1.0 / bas.sum()

        # les `rebalancement` jours de bourse qui suivent t
        hold = daily_rets.index[daily_rets.index > t][:rebalancement]
        seg = daily_rets.loc[hold, w.index].fillna(0.0).mul(w, axis=1).sum(axis=1)

        # turnover unilatéral, puis coût prélevé au premier jour du segment
        if poids_precedents is None:
            turn = 0.5 * w.abs().sum()
        else:
            turn = 0.5 * w.sub(poids_precedents, fill_value=0.0).abs().sum()
        turnovers.append(turn)
        if len(seg) > 0:
            seg.iloc[0] -= turn * 2.0 * couts_bps / 1e4
            segments.append(seg)

        poids_precedents = w

    if not segments:
        return pd.Series(dtype=float), 0.0
    strat = pd.concat(segments).sort_index()
    return strat, float(np.mean(turnovers))


def summarize(strat, turnover):
    """Rendement annualisé, volatilité, Sharpe, drawdown, turnover."""
    if len(strat) == 0:
        return {"rendement_ann": np.nan, "vol_ann": np.nan, "sharpe": np.nan,
                "drawdown_max": np.nan, "turnover_moyen": np.nan}
    ann_ret = strat.mean() * 252
    ann_vol = strat.std() * np.sqrt(252)
    cum = (1.0 + strat).cumprod()
    dd = (cum / cum.cummax() - 1.0).min()
    return {
        "rendement_ann": ann_ret,
        "vol_ann": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "drawdown_max": dd,
        "turnover_moyen": turnover,
    }
