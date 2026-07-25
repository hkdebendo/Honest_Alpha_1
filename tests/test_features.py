"""Tests anti-fuite sur les variables et la cible.

L'idée simple : si on modifie les données du futur, rien de ce qui est
calculé avant ne doit bouger. Si un de ces tests casse, c'est une fuite.
"""
import numpy as np
import pandas as pd

from src.features import build_features, forward_returns, xs_zscore


def marche_fictive(n_jours=400, n_actifs=6, graine=0):
    rng = np.random.default_rng(graine)
    rets = rng.normal(0.0, 0.01, (n_jours, n_actifs))
    index = pd.bdate_range("2020-01-01", periods=n_jours)
    colonnes = [f"A{i}" for i in range(n_actifs)]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                          index=index, columns=colonnes)
    volumes = pd.DataFrame(rng.integers(1e5, 1e6, (n_jours, n_actifs)).astype(float),
                           index=index, columns=colonnes)
    return prices, volumes


def cfg_variables():
    return {
        "cible": {"horizon": 5, "retard": 1},
        "variables": {"momentum": [1, 5, 20, 60], "vol_longue": 20,
                      "vol_courte": 5, "volume_mm": 20},
    }


def test_modifier_le_futur_ne_change_pas_les_variables():
    prices, volumes = marche_fictive()
    cfg = cfg_variables()

    feats = build_features(prices, volumes, cfg)

    # on fausse les 10 derniers jours de prix
    prices_mod = prices.copy()
    prices_mod.iloc[-10:] = prices_mod.iloc[-10:] * 1.5
    feats_mod = build_features(prices_mod, volumes, cfg)

    for nom in feats:
        # tout ce qui précède les jours modifiés (avec les fenêtres
        # glissantes les plus longues = 60 jours de marge) est intact
        assert feats[nom].iloc[:-70].equals(feats_mod[nom].iloc[:-70]), nom


def test_la_cible_commence_a_t_plus_1():
    prices, _ = marche_fictive()
    horizon, retard = 5, 1

    fwd = forward_returns(prices, horizon, retard)

    # la cible à la date t utilise les prix de t+retard à t+retard+horizon :
    # changer le prix en ligne i+horizon+retard+1 ne doit toucher que les
    # cibles des dates >= i+1
    i = 100
    prices_mod = prices.copy()
    prices_mod.iloc[i + retard + horizon + 1] *= 1.3
    fwd_mod = forward_returns(prices_mod, horizon, retard)

    assert fwd.iloc[:i + 1].equals(fwd_mod.iloc[:i + 1])
    assert not fwd.iloc[i + 1].equals(fwd_mod.iloc[i + 1])


def test_standardisation_transversale():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 1, (50, 8)),
                      index=pd.bdate_range("2020-01-01", periods=50),
                      columns=list("ABCDEFGH"))
    z = xs_zscore(df)
    # par date : moyenne nulle et écart-type unité
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=1), 1.0, atol=1e-12)


def test_panel_sans_nan():
    from src.features import make_dataset
    prices, volumes = marche_fictive()
    ds = make_dataset(prices, volumes, cfg_variables())
    assert not ds["X"].isna().any().any()
    assert not ds["y"].isna().any()
    assert ds["X"].index.names == ["date", "ticker"]
    assert len(ds["X"]) == len(ds["y"])
