"""Le découpage walk-forward doit être dans le bon ordre, purgé, sans recouvrement."""
import pandas as pd

from src.walkforward import make_folds


def dates_test(n=400):
    return list(pd.bdate_range("2015-01-01", periods=n))


def test_ordre_des_fenêtres():
    folds = make_folds(dates_test(), train_min=150, bloc_test=40, gap=10,
                       part_validation=0.2)
    assert len(folds) >= 3
    for f in folds:
        assert len(f.fit) > 0 and len(f.val) > 0 and len(f.test) == 40
        # entraînement < validation < test, dans cet ordre
        assert max(f.fit) < min(f.val)
        assert max(f.val) < min(f.test)


def test_le_gap_est_respecté():
    dates = dates_test()
    folds = make_folds(dates, train_min=150, bloc_test=40, gap=10,
                       part_validation=0.2)
    for f in folds:
        # entre la dernière date d'entraînement (fit + validation) et la
        # première date de test, il doit manquer exactement `gap` dates
        pos_fin_train = dates.index(max(f.val))
        pos_debut_test = dates.index(min(f.test))
        assert pos_debut_test - pos_fin_train == 11  # gap=10 dates sautées


def test_les_blocs_de_test_ne_se_chevauchent_pas():
    dates = dates_test()
    folds = make_folds(dates, train_min=150, bloc_test=40, gap=10,
                       part_validation=0.2)
    vus = []
    for f in folds:
        vus.extend(f.test)
    assert len(vus) == len(set(vus))
    # et ils se suivent dans l'ordre chronologique
    assert vus == sorted(vus)


def test_la_fenêtre_d_entrainement_s_etend():
    dates = dates_test()
    folds = make_folds(dates, train_min=150, bloc_test=40, gap=10,
                       part_validation=0.2)
    tailles = [len(f.fit) + len(f.val) for f in folds]
    assert all(b > a for a, b in zip(tailles, tailles[1:]))
