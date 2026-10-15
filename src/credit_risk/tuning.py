"""Optuna tuning (inner loop) and nested cross-validation (outer loop)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from credit_risk import metrics
from credit_risk.models import calibrations_for, make_estimator, suggest_params
from credit_risk.splits import outer_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuneResult:
    params: dict[str, Any]
    value: float  # best mean inner-CV log loss
    n_trials: int


def tune_params(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    n_trials: int,
    inner_splits: int,
) -> TuneResult:
    """Minimise mean inner-CV log loss with a seeded TPE sampler. Uses only (X, y) it is given."""
    if name == "majority":
        return TuneResult({}, float("nan"), 0)
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(name, trial)
        losses = []
        for tr, va in folds:
            est = make_estimator(name, numeric, categorical, params=params, seed=seed)
            est.fit(X.iloc[tr], y[tr])
            losses.append(metrics.logloss(y[va], est.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return TuneResult(dict(study.best_params), float(study.best_value), n_trials)


@dataclass
class NestedCVResult:
    oof: dict[tuple[str, str], np.ndarray]  # (model, calibration) -> (n_repeats, n_rows)
    folds: list[dict[str, Any]] = field(default_factory=list)
    params: list[dict[str, Any]] = field(default_factory=list)
    n_repeats: int = 1


def _outer_fold(
    repeat: int,
    fold: int,
    tr: np.ndarray,
    te: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    model_names: list[str],
    n_trials: int,
    inner_splits: int,
    seed: int,
) -> dict[str, Any]:
    Xtr, ytr = X.iloc[tr], y[tr]
    Xte, yte = X.iloc[te], y[te]
    fold_seed = seed + 1000 * repeat + fold
    preds: dict[tuple[str, str], np.ndarray] = {}
    records, param_records = [], []
    for name in model_names:
        tuned = tune_params(
            name, Xtr, ytr, numeric, categorical,
            seed=fold_seed, n_trials=n_trials, inner_splits=inner_splits,
        )  # fmt: skip
        param_records.append(
            {
                "repeat": repeat,
                "fold": fold,
                "model": name,
                "params": tuned.params,
                "inner_logloss": tuned.value,
            }
        )
        for cal in calibrations_for(name):
            est = make_estimator(name, numeric, categorical, tuned.params, fold_seed, cal)
            est.fit(Xtr, ytr)
            p = est.predict_proba(Xte)[:, 1]
            preds[(name, cal)] = p
            thr = metrics.theoretical_threshold()
            records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": name,
                    "calibration": cal,
                    "auc": metrics.auc(yte, p),
                    "logloss": metrics.logloss(yte, p),
                    "brier": metrics.brier(yte, p),
                    "ece": metrics.ece(yte, p),
                    "cost_at_theoretical_threshold": metrics.expected_cost(
                        yte, metrics.decide_approve(p, thr)
                    ),
                }
            )
    return {"repeat": repeat, "te": te, "preds": preds, "records": records, "params": param_records}


def nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    model_names: list[str],
    n_splits: int = 5,
    n_repeats: int = 3,
    n_trials: int = 20,
    inner_splits: int = 3,
    seed: int = 42,
    n_jobs: int = 1,
) -> NestedCVResult:
    """Repeated stratified outer CV; tuning happens only on each outer training part.

    Returns pooled out-of-fold probabilities (one row per repeat) for every model/calibration
    plus per-fold metric records and the tuned parameters of each outer fold.
    """
    y = np.asarray(y)
    folds = outer_folds(y, n_splits, n_repeats, seed)
    jobs = [
        delayed(_outer_fold)(
            r,
            k % n_splits,
            tr,
            te,
            X,
            y,
            numeric,
            categorical,
            model_names,
            n_trials,
            inner_splits,
            seed,
        )
        for k, (r, tr, te) in enumerate(folds)
    ]
    results = Parallel(n_jobs=n_jobs)(jobs)
    oof: dict[tuple[str, str], np.ndarray] = {}
    out = NestedCVResult(oof=oof, n_repeats=n_repeats)
    for res in results:
        for key, p in res["preds"].items():
            if key not in oof:
                oof[key] = np.full((n_repeats, len(y)), np.nan)
            oof[key][res["repeat"], res["te"]] = p
        out.folds.extend(res["records"])
        out.params.extend(res["params"])
    return out
