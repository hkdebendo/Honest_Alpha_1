"""MLP PyTorch avec boucle d'entraînement écrite à la main.

Réseau volontairement petit (2 couches cachées) : avec ~35 actifs et un
signal faible, un gros réseau surapprend. Régularisation par dropout et
weight decay, arrêt anticipé sur la perte de validation interne.

Détail anti-fuite important : les variables sont standardisées avec les
moyennes / écart-types de la fenêtre d'entraînement uniquement (les
z-scores transversaux ne suffisent pas pour un réseau de neurones).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim_entree, cachees=(64, 32), dropout=0.2):
        super().__init__()
        couches = []
        d = dim_entree
        for h in cachees:
            couches += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        couches.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*couches)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPModel:
    def __init__(self, cachees=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 batch_size=512, max_epochs=100, patience=12, graine=0):
        self.cachees = cachees
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.graine = graine
        self.model_ = None
        self.mu_ = None
        self.sd_ = None

    def _standardise(self, X):
        return (X - self.mu_) / self.sd_

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(self.graine)
        np.random.seed(self.graine)
        torch.set_num_threads(1)

        # moyenne / écart-type calculés sur l'entraînement uniquement
        self.mu_ = X.mean()
        self.sd_ = X.std().replace(0.0, 1.0)

        Xt = torch.tensor(self._standardise(X).values, dtype=torch.float32)
        yt = torch.tensor(y.values, dtype=torch.float32)
        Xv = torch.tensor(self._standardise(X_val).values, dtype=torch.float32)
        yv = torch.tensor(y_val.values, dtype=torch.float32)

        modele = MLP(X.shape[1], self.cachees, self.dropout)
        opt = torch.optim.Adam(modele.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        meilleure_perte, meilleur_etat, sans_amelioration = np.inf, None, 0
        n = len(Xt)

        for _ in range(self.max_epochs):
            modele.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                perte = nn.functional.mse_loss(modele(Xt[idx]), yt[idx])
                perte.backward()
                opt.step()

            modele.eval()
            with torch.no_grad():
                perte_val = nn.functional.mse_loss(modele(Xv), yv).item()
            if perte_val < meilleure_perte - 1e-5:
                meilleure_perte = perte_val
                meilleur_etat = copy.deepcopy(modele.state_dict())
                sans_amelioration = 0
            else:
                sans_amelioration += 1
                if sans_amelioration >= self.patience:
                    break

        if meilleur_etat is not None:
            modele.load_state_dict(meilleur_etat)
        self.model_ = modele
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            x = torch.tensor(self._standardise(X).values, dtype=torch.float32)
            pred = self.model_(x).numpy()
        return pd.Series(pred, index=X.index)
