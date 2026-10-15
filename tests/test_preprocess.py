from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from credit_risk import data
from credit_risk.models import make_estimator
from credit_risk.preprocess import build_preprocessor


@pytest.fixture()
def xy(df):
    num, cat = data.select_features("all")
    return df[num + cat], df["bad"].to_numpy(), num, cat


@pytest.mark.parametrize("kind", ["linear", "onehot", "ordinal"])
def test_preprocessors_handle_missing_and_unseen(xy, kind):
    X, _, num, cat = xy
    train, test = X.iloc[:300].copy(), X.iloc[300:].copy()
    test.iloc[0, test.columns.get_loc("duration_months")] = np.nan
    test.iloc[1, test.columns.get_loc("purpose")] = "never seen"
    pre = build_preprocessor(kind, num, cat).fit(train)
    out = pre.transform(test)
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out, dtype=float)
    if kind == "ordinal":
        assert np.isnan(out).sum() >= 1  # unseen category passed on as missing for HGB
    else:
        assert not np.isnan(out).any()


def test_linear_scaler_uses_training_statistics_only(xy):
    X, _, num, cat = xy
    train, test = X.iloc[:300], X.iloc[300:].copy()
    test["duration_months"] = 10_000  # extreme values only in the test rows
    pre = build_preprocessor("linear", num, cat).fit(train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    j = num.index("duration_months")
    assert scaler.mean_[j] == pytest.approx(train["duration_months"].mean())
    assert scaler.mean_[j] < 100  # not contaminated by the 10,000s
    out = pre.transform(test)
    assert out[:, j].mean() > 100  # test rows are transformed with train stats


def test_linear_log_transform_is_applied_to_amount(xy):
    X, _, num, cat = xy
    pre = build_preprocessor("linear", num, cat).fit(X)
    out = pre.transform(X)
    j = num.index("credit_amount")
    assert abs(out[:, j].mean()) < 1e-8  # standardised after log1p
    assert abs(np.corrcoef(out[:, j], np.log1p(X["credit_amount"]))[0, 1]) > 0.999


def test_one_hot_columns_match_categories(xy):
    X, _, num, cat = xy
    pre = build_preprocessor("onehot", num, cat).fit(X)
    names = pre.get_feature_names_out()
    assert len(names) == len(num) + sum(X[c].nunique() for c in cat)


@pytest.mark.parametrize("name", ["logreg", "hgb", "xgb"])
def test_full_pipeline_predicts_on_unseen_category(xy, name):
    X, y, num, cat = xy
    est = make_estimator(name, num, cat, seed=0).fit(X.iloc[:300], y[:300])
    test = X.iloc[300:].copy()
    test["purpose"] = test["purpose"].astype(object)
    test.iloc[0, test.columns.get_loc("purpose")] = "never seen"
    p = est.predict_proba(test)[:, 1]
    assert p.shape == (len(test),)
    assert np.all((p >= 0) & (p <= 1))


def test_pipeline_is_a_sklearn_pipeline(xy):
    _, _, num, cat = xy
    assert isinstance(make_estimator("logreg", num, cat), Pipeline)
    assert isinstance(pd.DataFrame(), pd.DataFrame)
