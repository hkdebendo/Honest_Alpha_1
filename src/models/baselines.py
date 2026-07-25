"""Prédicteur naïf, Ridge et CatBoost.

Tous les modèles reçoivent les mêmes variables, sont entraînés sur les
mêmes fenêtres et prédisent la même cible (le rang transversal du
rendement futur). Les hyperparamètres sont choisis sur la validation
interne (fin de la fenêtre d'entraînement), jamais sur le test.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge

from ..metrics import daily_corr, to_wide


class NaiveMomentum:
    """Plancher de comparaison : la prédiction est le momentum 20 jours, tel quel."""

    def __init__(self, colonne="mom_20"):
        self.colonne = colonne

    def fit(self, X, y, X_val=None, y_val=None):
        return self

    def predict(self, X):
        return X[self.colonne].copy()


class RidgeBaseline:
    """Régression Ridge ; la pénalité est choisie par IC de rang moyen sur la validation."""

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0)):
        self.alphas = alphas
        self.alpha_ = None
        self.model_ = None

    def fit(self, X, y, X_val, y_val):
        meilleur, meilleure_alpha = None, None
        for a in self.alphas:
            m = Ridge(alpha=a)
            m.fit(X.values, y.values)
            pred = pd.Series(m.predict(X_val.values), index=X_val.index)
            ic = daily_corr(to_wide(pred), to_wide(y_val), "spearman").mean()
            if pd.notna(ic) and (meilleur is None or ic > meilleur):
                meilleur, meilleure_alpha = ic, a
        self.alpha_ = meilleure_alpha if meilleure_alpha is not None else self.alphas[0]

        # réentraînement sur entraînement + validation avec l'alpha retenu :
        # la validation ne sert qu'à choisir, pas à estimer
        X_all = pd.concat([X, X_val])
        y_all = pd.concat([y, y_val])
        self.model_ = Ridge(alpha=self.alpha_).fit(X_all.values, y_all.values)
        return self

    def predict(self, X):
        return pd.Series(self.model_.predict(X.values), index=X.index)


class CatBoostBaseline:
    """Référence non linéaire : profondeur faible, arrêt anticipé sur la validation."""

    def __init__(self, profondeur=4, iterations=600, lr=0.05, graine=0):
        self.profondeur = profondeur
        self.iterations = iterations
        self.lr = lr
        self.graine = graine
        self.model_ = None

    def fit(self, X, y, X_val, y_val):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError(
                "catboost n'est pas installé : pip install catboost"
            ) from e

        self.model_ = CatBoostRegressor(
            depth=self.profondeur,
            learning_rate=self.lr,
            iterations=self.iterations,
            random_seed=self.graine,
            verbose=0,
        )
        self.model_.fit(X.values, y.values, eval_set=(X_val.values, y_val.values),
                       early_stopping_rounds=50)
        return self

    def predict(self, X):
        return pd.Series(self.model_.predict(X.values), index=X.index)
