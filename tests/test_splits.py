from __future__ import annotations

import numpy as np

from credit_risk.splits import holdout_split, outer_folds


def _y(n=500, rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < rate).astype(int)


def test_holdout_is_disjoint_and_complete():
    y = _y()
    tr, te = holdout_split(y, 0.2, seed=1)
    assert set(tr).isdisjoint(te)
    assert len(tr) + len(te) == len(y)
    assert len(te) == 100


def test_holdout_is_stratified():
    y = _y()
    tr, te = holdout_split(y, 0.2, seed=1)
    assert abs(y[tr].mean() - y[te].mean()) < 0.01


def test_holdout_deterministic_and_seed_sensitive():
    y = _y()
    a = holdout_split(y, 0.2, seed=3)
    b = holdout_split(y, 0.2, seed=3)
    c = holdout_split(y, 0.2, seed=4)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_outer_folds_cover_each_row_once_per_repeat():
    y = _y(200)
    folds = outer_folds(y, n_splits=5, n_repeats=3, seed=0)
    assert len(folds) == 15
    for r in range(3):
        te = np.concatenate([f[2] for f in folds if f[0] == r])
        assert sorted(te) == list(range(200))
    for _, tr, te in folds:
        assert set(tr).isdisjoint(te)
        assert abs(y[te].mean() - y.mean()) < 0.06


def test_outer_folds_repeats_differ():
    y = _y(200)
    folds = outer_folds(y, 5, 2, seed=0)
    first = [f for f in folds if f[0] == 0][0][2]
    second = [f for f in folds if f[0] == 1][0][2]
    assert not np.array_equal(first, second)
