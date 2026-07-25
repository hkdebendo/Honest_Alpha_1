"""Ligne de commande du projet.

    python -m src.run data       # téléchargement + nettoyage
    python -m src.run features  # variables + cible, sans fuite
    python -m src.run run       # walk-forward, tous les modèles, toutes les graines
    python -m src.run ablation  # retrait de chaque groupe de variables
    python -m src.run controls  # contrôles négatifs (cible mélangée, variable fuitée)
    python -m src.run report    # tableaux, figures et tests statistiques

Chaque étape écrit dans data/processed/ ou results/ et peut être relancée
indépendamment. Les configurations (configs/*.yaml) sont la seule source
de paramètres : rien n'est codé en dur ici.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .data import PROC_DIR, RAW_DIR, ROOT, load_yaml, step_data
from .features import groups_for, make_dataset
from .metrics import daily_corr, ic_summary, to_wide
from .models.baselines import CatBoostBaseline, NaiveMomentum, RidgeBaseline
from .portfolio import backtest, summarize
from .stats import block_bootstrap_mean, paired_ic_test
from .walkforward import make_folds

RES = ROOT / "results"
FIG = RES / "figures"


# ----------------------------------------------------------------- utilitaires

def charger_dataset():
    X = pd.read_parquet(PROC_DIR / "X.parquet")
    y = pd.read_parquet(PROC_DIR / "y.parquet").iloc[:, 0]
    fwd = pd.read_parquet(PROC_DIR / "fwd.parquet")
    return X, y, fwd


def rendements_journaliers():
    prices = pd.read_csv(PROC_DIR / "prices.csv", index_col=0, parse_dates=True)
    return prices.pct_change()


def build_models(wf, avec_naif=True, avec_mlp=True, colonnes=None,
                 naive_col="mom_20"):
    """Construit le dictionnaire modèle -> instance.

    catboost et torch sont optionnels : s'ils manquent, on continue avec
    ce qui est installé et on prévient.
    """
    models = {}
    if avec_naif and (colonnes is None or naive_col in colonnes):
        models["naif"] = NaiveMomentum(naive_col)
    models["ridge"] = RidgeBaseline()
    try:
        import catboost  # noqa: F401
        models["catboost"] = CatBoostBaseline()
    except ImportError:
        print("avertissement : catboost absent, la référence non linéaire est ignorée")
    if avec_mlp:
        try:
            from .models.mlp import MLPModel
            for graine in wf["graines_mlp"]:
                models[f"mlp_seed{graine}"] = MLPModel(graine=graine)
        except ImportError:
            print("avertissement : torch absent, le MLP est ignoré")
    return models


def walk_forward(X, y, wf, models, colonnes=None):
    """Boucle walk-forward. Retourne {modele: prédictions hors échantillon en large}."""
    if colonnes is not None:
        X = X[colonnes]

    dates = np.array(sorted(X.index.get_level_values("date").unique()))
    folds = make_folds(dates, wf["train_min"], wf["bloc_test"], wf["gap"],
                       wf["part_validation"])
    print(f"{len(folds)} plis, fenêtre initiale {wf['train_min']} jours, "
          f"gap {wf['gap']} jours, blocs de test {wf['bloc_test']} jours.")

    date_level = X.index.get_level_values("date")

    def tranches(liste_dates):
        masque = date_level.isin(set(liste_dates))
        return X[masque], y[masque]

    preds = {nom: [] for nom in models}
    for i, fold in enumerate(folds, start=1):
        Xf, yf = tranches(fold.fit)
        Xv, yv = tranches(fold.val)
        Xt, _ = tranches(fold.test)
        if len(Xt) == 0 or len(Xf) == 0:
            continue
        for nom, modele in models.items():
            modele.fit(Xf, yf, Xv, yv)
            preds[nom].append(modele.predict(Xt))
        print(f"  pli {i}/{len(folds)} : test du {fold.test[0].date()} "
              f"au {fold.test[-1].date()}")

    return {nom: to_wide(pd.concat(s)) for nom, s in preds.items() if s}


# --------------------------------------------------------------------- étapes

def step_features():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(ROOT / "configs" / "features.yaml")
    prices = pd.read_csv(PROC_DIR / "prices.csv", index_col=0, parse_dates=True)
    volumes = pd.read_csv(PROC_DIR / "volumes.csv", index_col=0, parse_dates=True)

    ds = make_dataset(prices, volumes, cfg)
    ds["X"].to_parquet(PROC_DIR / "X.parquet")
    ds["y"].to_frame("cible").to_parquet(PROC_DIR / "y.parquet")
    ds["fwd"].to_parquet(PROC_DIR / "fwd.parquet")

    meta = {"colonnes": list(ds["X"].columns), "groupes": groups_for(cfg)}
    with open(PROC_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Panel : {ds['X'].shape[0]} lignes, {ds['X'].shape[1]} variables, "
          f"{ds['X'].index.get_level_values('date').nunique()} dates.")


def step_run():
    RES.mkdir(parents=True, exist_ok=True)
    X, y, fwd = charger_dataset()
    wf = load_yaml(ROOT / "configs" / "walkforward.yaml")

    models = build_models(wf, colonnes=list(X.columns))
    preds = walk_forward(X, y, wf, models)
    if not preds:
        raise SystemExit("Aucun modèle n'a tourné.")

    # prédiction MLP "ensemble" : moyenne des graines
    seeds = {k: v for k, v in preds.items() if k.startswith("mlp_seed")}
    if seeds:
        preds["mlp"] = pd.concat(seeds.values()).groupby(level=0).mean()

    # prédictions et séries d'IC sauvegardées modèle par modèle
    for nom, pw in preds.items():
        pw.to_csv(RES / f"pred_{nom}.csv")
        daily_corr(pw, fwd, "pearson").rename("ic").to_csv(RES / f"ic_{nom}.csv")
        daily_corr(pw, fwd, "spearman").rename("ic_rang").to_csv(RES / f"icrang_{nom}.csv")

    # synthèse des IC (et dispersion du MLP entre graines)
    lignes = []
    for nom, pw in preds.items():
        lignes.append({"modele": nom, **ic_summary(daily_corr(pw, fwd, "spearman"))})
    if seeds:
        ics_seeds = pd.concat(
            {nom: daily_corr(pw, fwd, "spearman") for nom, pw in seeds.items()},
            axis=1,
        )
        moyenne = ics_seeds.mean(axis=1)
        lignes.append({"modele": "mlp_graines_moyenne", "n_dates": len(moyenne),
                       "ic_moyen": moyenne.mean(), "ic_std": moyenne.std(),
                       "icir": np.nan, "t_naif": np.nan,
                       "ecart_entre_graines": ics_seeds.mean(axis=0).std()})
    pd.DataFrame(lignes).to_csv(RES / "ic_summary.csv", index=False)

    # portefeuille après coûts
    pf = wf["portefeuille"]
    rets = rendements_journaliers()
    lignes_pf = []
    for nom, pw in preds.items():
        strat, turn = backtest(pw, rets, pf["rebalancement"], pf["quantiles"],
                               pf["couts_bps"])
        lignes_pf.append({"modele": nom, **summarize(strat, turn)})
    pd.DataFrame(lignes_pf).to_csv(RES / "portfolio.csv", index=False)

    # test apparié MLP vs chaque référence + bootstrap par blocs
    ic_par_modele = {nom: daily_corr(pw, fwd, "spearman")
                     for nom, pw in preds.items()}
    if "mlp" in ic_par_modele:
        st = wf["stats"]
        tests = []
        for base in ["naif", "ridge", "catboost"]:
            if base not in ic_par_modele:
                continue
            diff = (ic_par_modele["mlp"] - ic_par_modele[base]).dropna()
            test = paired_ic_test(ic_par_modele["mlp"], ic_par_modele[base],
                                  st["lags_newey_west"])
            boot = block_bootstrap_mean(diff.values, st["taille_bloc_bootstrap"],
                                        st["n_bootstrap"])
            tests.append({"comparaison": f"mlp - {base}", **test,
                          "diff_moyenne": diff.mean(), **boot})
        if tests:
            pd.DataFrame(tests).to_csv(RES / "paired_tests.csv", index=False)

    print("Résultats écrits dans results/. Lancez `python -m src.run report` "
          "pour les tableaux et figures.")


def step_ablation():
    RES.mkdir(parents=True, exist_ok=True)
    X, y, fwd = charger_dataset()
    wf = load_yaml(ROOT / "configs" / "walkforward.yaml")
    cfg_feat = load_yaml(ROOT / "configs" / "features.yaml")

    toutes = list(X.columns)
    lignes = []

    def evalue(nom_abl, colonnes, avec_naif):
        models = build_models(wf, avec_naif=avec_naif, avec_mlp=True,
                              colonnes=colonnes)
        preds = walk_forward(X, y, wf, models, colonnes=colonnes)
        seeds = {k: v for k, v in preds.items() if k.startswith("mlp_seed")}
        if seeds:
            preds["mlp"] = pd.concat(seeds.values()).groupby(level=0).mean()
        for nom, pw in preds.items():
            lignes.append({"ablation": nom_abl, "modele": nom,
                           **ic_summary(daily_corr(pw, fwd, "spearman"))})

    evalue("toutes_variables", toutes, avec_naif=True)
    for groupe, cols_g in groups_for(cfg_feat).items():
        restantes = [c for c in toutes if c not in cols_g]
        evalue(f"sans_{groupe}", restantes, avec_naif=("momentum" not in groupe))

    pd.DataFrame(lignes).to_csv(RES / "ablation.csv", index=False)


def step_controls():
    """Deux contrôles négatifs, sur données réelles :

    1. cibles mélangées -> l'IC doit retomber vers 0, sinon quelque chose
       fuit dans le pipeline ;
    2. une variable fuitée (le rang de la cible + bruit) -> l'IC doit
       exploser, sinon le pipeline ne détecterait pas une fuite réelle.
    """
    RES.mkdir(parents=True, exist_ok=True)
    X, y, fwd = charger_dataset()
    wf = load_yaml(ROOT / "configs" / "walkforward.yaml")

    # 1. mélange transversal des cibles, date par date
    rng = np.random.default_rng(42)
    y_large = to_wide(y)
    melange = y_large.apply(
        lambda row: pd.Series(rng.permutation(row.dropna().values),
                              index=row.dropna().index), axis=1
    )
    y_melange = melange.stack().reindex(X.index)
    preds = walk_forward(X, y_melange, wf, {"ridge": RidgeBaseline()})
    ic_melange = daily_corr(preds["ridge"], fwd, "spearman").mean()

    # 2. variable volontairement fuitée
    bruit = pd.Series(rng.normal(0.0, 0.05, len(X)), index=X.index)
    X_fuite = X.assign(fuite=y + bruit)
    preds_f = walk_forward(X_fuite, y, wf, {"ridge": RidgeBaseline()})
    ic_fuite = daily_corr(preds_f["ridge"], fwd, "spearman").mean()

    tableau = pd.DataFrame([
        {"controle": "cible_melangee", "attendu": "~0", "observe": ic_melange},
        {"controle": "variable_fuitee", "attendu": "tres eleve", "observe": ic_fuite},
    ])
    tableau.to_csv(RES / "controls.csv", index=False)
    print(tableau.to_string(index=False))
    if abs(ic_melange) > 0.05:
        print("ATTENTION : IC anormalement eleve sur cible melangee, "
              "chercher une fuite dans le pipeline.")
    if ic_fuite < 0.5:
        print("ATTENTION : la variable fuitee ne donne pas un IC eleve, "
              "le test de detection de fuite est casse.")


def step_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    wf = load_yaml(ROOT / "configs" / "walkforward.yaml")

    # modèles disponibles = ceux qui ont une série d'IC de rang
    modeles = sorted(p.stem.replace("icrang_", "")
                     for p in RES.glob("icrang_*.csv"))
    if not modeles:
        raise SystemExit("Rien à rapporter : lancez d'abord `python -m src.run run`.")

    series_ic = {
        m: pd.read_csv(RES / f"icrang_{m}.csv", index_col=0, parse_dates=True).iloc[:, 0]
        for m in modeles
    }

    # tableau de synthèse
    tableau = [{"modele": m, **ic_summary(series_ic[m])} for m in modeles]
    table_ic = pd.DataFrame(tableau)
    try:
        table_pf = pd.read_csv(RES / "portfolio.csv")
        table_ic = table_ic.merge(table_pf, on="modele", how="left")
    except FileNotFoundError:
        pass
    table_ic.to_csv(RES / "table.csv", index=False)

    # figure 1 : IC de rang cumulé
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, s in series_ic.items():
        ax.plot(s.cumsum(), label=m, lw=1.4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("IC de rang cumulé, hors échantillon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "ic_cumule.png", dpi=150)
    plt.close(fig)

    # figure 2 : IC de rang moyen par année
    fig, ax = plt.subplots(figsize=(9, 5))
    annees = sorted({d.year for s in series_ic.values() for d in s.index})
    largeur = 0.8 / max(len(modeles), 1)
    for i, m in enumerate(modeles):
        par_annee = series_ic[m].groupby(series_ic[m].index.year).mean()
        ax.bar([a + i * largeur for a in annees],
               [par_annee.get(a, np.nan) for a in annees],
               width=largeur, label=m)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(annees)
    ax.set_title("IC de rang moyen par année")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "ic_par_annee.png", dpi=150)
    plt.close(fig)

    # figure 3 : sensibilité aux coûts
    rets = rendements_journaliers()
    pf = wf["portefeuille"]
    niveaux = [0, 5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in modeles:
        pw = pd.read_csv(RES / f"pred_{m}.csv", index_col=0, parse_dates=True)
        sharpe = []
        for bps in niveaux:
            strat, _ = backtest(pw, rets, pf["rebalancement"], pf["quantiles"], bps)
            sharpe.append(summarize(strat, 0.0)["sharpe"])
        ax.plot(niveaux, sharpe, marker="o", label=m)
    ax.set_xlabel("coûts par aller (bps)")
    ax.set_ylabel("Sharpe")
    ax.set_title("Sensibilité du portefeuille aux coûts de transaction")
    ax.axhline(0, color="grey", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "couts.png", dpi=150)
    plt.close(fig)

    # bootstrap de la différence MLP - Ridge (persisté pour le rapport)
    if "mlp" in series_ic and "ridge" in series_ic:
        st = wf["stats"]
        diff = (series_ic["mlp"] - series_ic["ridge"]).dropna()
        boot = block_bootstrap_mean(diff.values, st["taille_bloc_bootstrap"],
                                    st["n_bootstrap"])
        pd.DataFrame([{"comparaison": "mlp - ridge",
                       "diff_moyenne": diff.mean(), **boot}]
                     ).to_csv(RES / "bootstrap.csv", index=False)

    print(table_ic.to_string(index=False))
    print(f"\nFigures dans {FIG}.")


def main():
    parser = argparse.ArgumentParser(description="HonestAlpha : pipeline complet.")
    parser.add_argument(
        "etape",
        choices=["data", "features", "run", "ablation", "controls", "report"],
        help="étape à exécuter",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    if args.etape == "data":
        step_data()
    elif args.etape == "features":
        step_features()
    elif args.etape == "run":
        step_run()
    elif args.etape == "ablation":
        step_ablation()
    elif args.etape == "controls":
        step_controls()
    elif args.etape == "report":
        step_report()


if __name__ == "__main__":
    main()
