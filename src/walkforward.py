"""Découpage walk-forward avec purge (gap).

|---- entraînement ----| validation |-- gap --|-- test --|
                        |---- entraînement ----| validation |-- gap --|-- test --|

- fenêtre d'entraînement extensible : l'historique s'allonge à chaque pli ;
- la validation interne est la fin de la fenêtre d'entraînement, jamais
  dans le test ;
- entre la fin de l'entraînement et le début du test, on saute `gap`
  dates : les cibles à 5 jours se chevauchent, et sans purge l'information
  fuiterait du test vers l'entraînement ;
- les blocs de test se suivent sans recouvrement : mis bout à bout, ils
  forment une série hors échantillon continue.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fold:
    fit: list    # dates d'entraînement strictes
    val: list    # validation interne (choix des hyperparamètres)
    test: list   # dates hors échantillon


def make_folds(dates, train_min, bloc_test, gap, part_validation):
    dates = list(dates)
    if len(dates) <= train_min + gap + bloc_test:
        raise ValueError(
            f"Pas assez de dates ({len(dates)}) pour train_min={train_min}, "
            f"gap={gap}, bloc_test={bloc_test}."
        )

    folds = []
    pos = train_min
    while pos + gap + bloc_test <= len(dates):
        train = dates[:pos]                       # fenêtre extensible
        n_val = int(len(train) * part_validation)
        val = train[-n_val:]
        fit = train[:-n_val]
        test = dates[pos + gap: pos + gap + bloc_test]
        folds.append(Fold(fit=fit, val=val, test=test))
        pos += bloc_test                           # le test d'hier rejoint l'entraînement
    return folds
