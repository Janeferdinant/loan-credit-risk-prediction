"""SHAP attributions (XGBoost) aggregated to the original features, and manual partial dependence."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def feature_owners(out_names: list[str], categorical: list[str]) -> list[str]:
    """Map each transformed column name (``num__x`` / ``cat__col_level``) to its source feature."""
    owners = []
    by_len = sorted(categorical, key=len, reverse=True)
    for name in out_names:
        kind, _, rest = name.partition("__")
        if kind == "num":
            owners.append(rest)
            continue
        match = next((c for c in by_len if rest.startswith(c + "_")), None)
        if match is None:
            raise ValueError(f"cannot map transformed column {name!r} to a source feature")
        owners.append(match)
    return owners


def shap_by_feature(
    fitted_pipeline, X: pd.DataFrame, categorical: list[str]
) -> dict[str, Any]:
    """SHAP values (log-odds of *bad*) of a fitted preprocessing + XGBoost pipeline, summed over
    one-hot columns so each original feature gets one value per row."""
    import shap

    prep = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]
    Xt = np.asarray(prep.transform(X), dtype=float)
    names = list(prep.get_feature_names_out())
    owners = feature_owners(names, categorical)
    explainer = shap.TreeExplainer(clf)
    sv = np.asarray(explainer.shap_values(Xt))
    agg = pd.DataFrame(sv, columns=names).T.groupby(owners, sort=False).sum().T
    agg.index = X.index
    return {
        "values": agg,
        "base_value": float(np.ravel(explainer.expected_value)[0]),
        "raw_values": sv,
        "raw_names": names,
        "matrix": Xt,
    }


def mean_abs_importance(values: pd.DataFrame) -> pd.Series:
    return values.abs().mean().sort_values(ascending=False)


def partial_dependence_1d(
    estimator,
    X: pd.DataFrame,
    feature: str,
    categorical: bool,
    grid_size: int = 20,
    n_ice: int = 30,
    seed: int = 0,
) -> dict[str, Any]:
    """Average (and ICE) predicted P(bad) as one feature is set to each grid value."""
    if categorical:
        grid: list[Any] = sorted(X[feature].dropna().unique().tolist())
    else:
        grid = sorted(set(np.quantile(X[feature], np.linspace(0.02, 0.98, grid_size)).tolist()))
    rng = np.random.default_rng(seed)
    ice_rows = rng.choice(len(X), size=min(n_ice, len(X)), replace=False)
    avg, ice = [], []
    for v in grid:
        Xv = X.copy()
        Xv[feature] = v
        p = estimator.predict_proba(Xv)[:, 1]
        avg.append(float(p.mean()))
        ice.append(p[ice_rows].tolist())
    return {
        "feature": feature,
        "categorical": categorical,
        "grid": grid,
        "average": avg,
        "ice": np.asarray(ice).T.tolist(),  # one list per sampled row
    }
