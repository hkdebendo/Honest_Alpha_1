"""Téléchargement et nettoyage des données journalières (yfinance).

Politique de nettoyage, volontairement simple et auditable :

1. on écarte les tickers sans assez d'historique, sans rien compléter ;
2. on supprime les jours où toute la ligne est vide (jours non cotés) ;
3. on supprime les jours où il manque au moins un actif plutôt que de
   reporter la dernière valeur -- pour des ETF liquides ces trous sont
   rares, et on préfère un historique propre à un historique inventé ;
4. les rendements journaliers extrêmes sont signalés dans le journal
   mais pas modifiés : à nous de vérifier qu'il s'agit de vrais
   mouvements et pas d'un ajustement de dividende raté.

Tout ce qui est supprimé finit dans data/processed/cleaning_log.csv.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_universe(config_path=ROOT / "configs" / "universe.yaml"):
    """Télécharge prix ajustés et volumes, sauve le brut dans data/raw."""
    import yfinance as yf  # import tardif : inutile pour les tests

    cfg = load_yaml(config_path)
    tickers = list(cfg["tickers"])
    print(f"Téléchargement de {len(tickers)} ETF depuis {cfg['debut']}...")

    raw = yf.download(tickers, start=cfg["debut"], auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
        volumes = raw["Volume"]
    else:  # un seul ticker passé à yf.download
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volumes = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    prices = prices.sort_index()
    volumes = volumes.sort_index()
    prices.to_csv(RAW_DIR / "prices.csv")
    volumes.to_csv(RAW_DIR / "volumes.csv")
    print(f"Brut sauvegardé : {prices.shape[0]} jours x {prices.shape[1]} actifs.")
    return prices, volumes


def clean_data(prices, volumes, min_history=2520):
    """Applique la politique de nettoyage et retourne le journal."""
    log = []

    # 1. historique trop court -> le ticker est exclu, pas complété
    counts = prices.notna().sum().sort_index()
    for t in counts.index:
        if counts[t] < min_history:
            log.append(["ticker_exclu_historique", t, int(counts[t])])
    keep = counts[counts >= min_history].index
    prices, volumes = prices[keep], volumes[keep]

    # 2. jours sans aucune cotation
    for d in prices.index[prices.isna().all(axis=1)]:
        log.append(["jour_sans_cotation", str(d.date()), ""])
    ok = prices.notna().any(axis=1)
    prices, volumes = prices[ok], volumes[ok]

    # 3. jours incomplets : on jette la journée entière
    incomplete = prices.index[prices.isna().any(axis=1)]
    for d in incomplete:
        log.append(["jour_incomplet_supprime", str(d.date()), ""])
    prices, volumes = prices.drop(index=incomplete), volumes.drop(index=incomplete)

    # 4. rendements extrêmes : signalés, pas touchés
    rets = np.log(prices).diff()
    extremes = rets.where(rets.abs() > 0.35)
    for d, row in extremes.iterrows():
        for t in row.dropna().index:
            log.append(["rendement_extreme_flagge", str(d.date()), t])

    return prices, volumes, log


def step_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    p_path, v_path = RAW_DIR / "prices.csv", RAW_DIR / "volumes.csv"
    if p_path.exists() and v_path.exists():
        print("Données brutes déjà présentes, pas de nouveau téléchargement.")
        prices = pd.read_csv(p_path, index_col=0, parse_dates=True)
        volumes = pd.read_csv(v_path, index_col=0, parse_dates=True)
    else:
        prices, volumes = download_universe()

    prices, volumes, log = clean_data(prices, volumes, min_history=2520)
    print(f"Après nettoyage : {prices.shape[0]} jours x {prices.shape[1]} actifs, "
          f"{len(log)} lignes dans le journal.")

    prices.to_csv(PROC_DIR / "prices.csv")
    volumes.to_csv(PROC_DIR / "volumes.csv")
    pd.DataFrame(log, columns=["action", "date", "ticker"]).to_csv(
        PROC_DIR / "cleaning_log.csv", index=False
    )


if __name__ == "__main__":
    step_data()
