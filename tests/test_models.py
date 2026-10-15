from __future__ import annotations

import numpy as np
import optuna
import pytest

from credit_risk import data
from credit_risk.models import MODEL_NAMES, make_estimator, suggest_params


@pytest.fixture()
def xy(df):
    num, cat = data.select_features("all")
    return df[num + cat], df["bad"].to_numpy(), num, cat


def test_majority_predicts_training_prior(xy):
    X, y, num, cat = xy
    est = make_estimator("majority", num, cat).fit(X, y)
    p = est.predict_proba(X)[:, 1]
    assert np.allclose(p, y.mean())


@pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n != "majority"])
@pytest.mark.parametrize("calibration", ["raw", "sigmoid", "isotonic"])
def test_every_model_and_calibration_fits(xy, name, calibration):
    X, y, num, cat = xy
    est = make_estimator(name, num, cat, seed=0, calibration=calibration).fit(X, y)
    p = est.predict_proba(X)[:, 1]
    assert p.shape == (len(X),) and p.min() >= 0 and p.max() <= 1


def test_models_beat_majority_on_training_signal(xy):
    from credit_risk.metrics import auc

    X, y, num, cat = xy
    for name in ("logreg", "hgb", "xgb"):
        est = make_estimator(name, num, cat, seed=0).fit(X, y)
        assert auc(y, est.predict_proba(X)[:, 1]) > 0.6


def test_same_seed_same_predictions(xy):
    X, y, num, cat = xy
    a = make_estimator("xgb", num, cat, seed=3).fit(X, y).predict_proba(X)
    b = make_estimator("xgb", num, cat, seed=3).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)


@pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n != "majority"])
def test_suggested_params_are_accepted_by_the_model(xy, name):
    X, y, num, cat = xy
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    trial = study.ask()
    params = suggest_params(name, trial)
    assert params
    make_estimator(name, num, cat, params=params, seed=0).fit(X, y)


def test_unknown_model_or_calibration_raises(xy):
    _, _, num, cat = xy
    with pytest.raises(ValueError):
        make_estimator("nope", num, cat)
    with pytest.raises(ValueError):
        make_estimator("logreg", num, cat, calibration="nope")
