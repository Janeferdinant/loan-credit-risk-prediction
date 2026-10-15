"""Model factory: baselines, logistic regression, gradient boosting, optional calibration."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_risk.preprocess import build_preprocessor

MODEL_NAMES = ("majority", "logreg", "hgb", "xgb")
CALIBRATIONS = ("raw", "sigmoid", "isotonic")
MODEL_LABELS = {
    "majority": "Majority class (prior)",
    "logreg": "Logistic regression",
    "hgb": "HistGradientBoosting",
    "xgb": "XGBoost",
}


def calibrations_for(name: str) -> tuple[str, ...]:
    return ("raw",) if name == "majority" else CALIBRATIONS


def default_params(name: str) -> dict[str, Any]:
    return {
        "majority": {},
        "logreg": {"C": 1.0},
        "hgb": {
            "learning_rate": 0.05,
            "max_leaf_nodes": 8,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "max_iter": 150,
        },
        "xgb": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
    }[name]


def suggest_params(name: str, trial) -> dict[str, Any]:
    """Optuna search spaces (documented in the README)."""
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "hgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 3, 31),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 400),
        }
    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    raise ValueError(f"no search space for model {name!r}")


def _pipeline(
    name: str, numeric: list[str], categorical: list[str], params: dict[str, Any], seed: int
) -> Pipeline:
    if name == "majority":
        return Pipeline(
            [
                ("prep", build_preprocessor("onehot", numeric, categorical)),
                ("clf", DummyClassifier(strategy="prior")),
            ]
        )
    if name == "logreg":
        clf = LogisticRegression(max_iter=5000, random_state=seed, **params)
        return Pipeline([("prep", build_preprocessor("linear", numeric, categorical)), ("clf", clf)])
    if name == "hgb":
        mask = [False] * len(numeric) + [True] * len(categorical)
        clf = HistGradientBoostingClassifier(
            categorical_features=mask, early_stopping=False, random_state=seed, **params
        )
        return Pipeline([("prep", build_preprocessor("ordinal", numeric, categorical)), ("clf", clf)])
    if name == "xgb":
        clf = XGBClassifier(
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
            **params,
        )
        return Pipeline([("prep", build_preprocessor("onehot", numeric, categorical)), ("clf", clf)])
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def make_estimator(
    name: str,
    numeric: list[str],
    categorical: list[str],
    params: dict[str, Any] | None = None,
    seed: int = 0,
    calibration: str = "raw",
):
    """Unfitted estimator. Preprocessing lives inside the pipeline (and inside any calibrator's
    internal folds), so it is always fitted on training rows only."""
    if calibration not in CALIBRATIONS:
        raise ValueError(f"unknown calibration {calibration!r}; choose from {CALIBRATIONS}")
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    pipe = _pipeline(name, numeric, categorical, {**default_params(name), **(params or {})}, seed)
    if calibration == "raw":
        return pipe
    if name == "majority":
        raise ValueError("calibration is meaningless for the majority baseline")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(pipe, method=calibration, cv=cv)
