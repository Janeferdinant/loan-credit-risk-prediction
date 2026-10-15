"""Leakage tests: nothing fitted may see rows from the fold it is evaluated on."""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk import data, tuning
from credit_risk.metrics import auc
from credit_risk.models import make_estimator
from credit_risk.splits import outer_folds


@pytest.fixture()
def xy(df):
    num, cat = data.select_features("all")
    return df[num + cat].reset_index(drop=True), df["bad"].to_numpy(), num, cat


def test_features_exclude_target(xy):
    X, _, num, cat = xy
    assert "bad" not in X.columns
    assert "bad" not in num + cat


def test_tuning_never_sees_outer_test_rows(xy, monkeypatch):
    X, y, num, cat = xy
    seen = []
    real = tuning.tune_params

    def spy(name, Xtr, ytr, *a, **k):
        seen.append(set(Xtr.index))
        return real(name, Xtr, ytr, *a, **k)

    monkeypatch.setattr(tuning, "tune_params", spy)
    res = tuning.nested_cv(
        X,
        y,
        num,
        cat,
        ["logreg"],
        n_splits=3,
        n_repeats=1,
        n_trials=2,
        inner_splits=2,
        seed=0,
        n_jobs=1,
    )
    folds = outer_folds(y, 3, 1, seed=0)
    assert len(seen) == len(folds)
    for used, (_, _, te) in zip(seen, folds, strict=True):
        assert used.isdisjoint(set(te))
    assert res.oof[("logreg", "raw")].shape[0] == 1


def test_each_row_predicted_by_a_model_that_did_not_train_on_it(xy):
    X, y, num, cat = xy
    res = tuning.nested_cv(
        X,
        y,
        num,
        cat,
        ["logreg"],
        n_splits=4,
        n_repeats=2,
        n_trials=2,
        inner_splits=2,
        seed=1,
        n_jobs=1,
    )
    oof = res.oof[("logreg", "raw")]
    assert oof.shape == (2, len(y))
    assert not np.isnan(oof).any()  # every row has an out-of-fold prediction in each repeat


def test_label_permutation_gives_chance_level_auc(xy):
    """If the full pipeline leaked labels through preprocessing, shuffled labels would still score > 0.5."""
    X, y, num, cat = xy
    rng = np.random.default_rng(0)
    y_perm = rng.permutation(y)
    res = tuning.nested_cv(
        X,
        y_perm,
        num,
        cat,
        ["logreg", "hgb"],
        n_splits=5,
        n_repeats=1,
        n_trials=2,
        inner_splits=2,
        seed=0,
        n_jobs=1,
    )
    for key in (("logreg", "raw"), ("hgb", "raw")):
        a = auc(y_perm, res.oof[key][0])
        assert 0.40 < a < 0.60, (key, a)


def test_real_signal_survives_the_same_check(xy):
    X, y, num, cat = xy
    res = tuning.nested_cv(
        X,
        y,
        num,
        cat,
        ["logreg"],
        n_splits=5,
        n_repeats=1,
        n_trials=2,
        inner_splits=2,
        seed=0,
        n_jobs=1,
    )
    assert auc(y, res.oof[("logreg", "raw")][0]) > 0.62


def test_preprocessing_fitted_on_train_fold_only(xy):
    X, y, num, cat = xy
    tr, te = np.arange(0, 300), np.arange(300, len(y))
    X = X.copy()
    X.loc[te, "credit_amount"] = 10**7
    est = make_estimator("logreg", num, cat, seed=0).fit(X.iloc[tr], y[tr])
    scaler = est.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    j = num.index("credit_amount")
    expected = np.log1p(X.iloc[tr]["credit_amount"]).mean()
    assert scaler.mean_[j] == pytest.approx(expected)  # unchanged by the 10^7 values in the test rows
