"""SHAP attributions (XGBoost) aggregated to the original features, LIME local explanations,
and manual partial dependence."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


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


def shap_by_feature(fitted_pipeline: Any, X: pd.DataFrame, categorical: list[str]) -> dict[str, Any]:
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
    estimator: Any,
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


def lime_explain_applicant(
    fitted_pipeline: Any,
    X_train: pd.DataFrame,
    applicant: pd.DataFrame | pd.Series,
    categorical: list[str],
    seed: int = 42,
    num_samples: int = 1000,
    explainer: Any | None = None,
) -> pd.Series:
    """Generate LIME tabular attributions for a single applicant (class 1: bad credit),
    summed over one-hot columns so each original feature receives a single score."""
    from lime.lime_tabular import LimeTabularExplainer

    prep = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    owners = feature_owners(names, categorical)
    all_owners = sorted(set(owners))

    if explainer is None:
        Xt_tr = np.asarray(prep.transform(X_train), dtype=float)
        explainer = LimeTabularExplainer(
            Xt_tr,
            feature_names=names,
            class_names=["good", "bad"],
            mode="classification",
            random_state=seed,
        )

    app_df = applicant.to_frame().T if isinstance(applicant, pd.Series) else applicant
    xt_app = np.asarray(prep.transform(app_df), dtype=float)[0]

    exp = explainer.explain_instance(
        xt_app,
        clf.predict_proba,
        num_features=len(names),
        num_samples=num_samples,
        labels=(1,),
    )
    weights = dict.fromkeys(all_owners, 0.0)
    for feat_idx, w in exp.as_map()[1]:
        o = owners[feat_idx]
        weights[o] += float(w)

    return pd.Series(weights).sort_values(ascending=False)


def lime_stability_kendall_tau(
    fitted_pipeline: Any,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    categorical: list[str],
    seeds: Sequence[int] = tuple(range(10)),
    num_samples: int = 1000,
) -> float:
    """Measure Kendall's tau rank correlation across multiple LIME random seeds for a cohort of applicants.
    Returns the mean pairwise Kendall tau across all seeds and applicants."""
    from lime.lime_tabular import LimeTabularExplainer

    prep = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    owners = feature_owners(names, categorical)
    all_owners = sorted(set(owners))

    Xt_tr = np.asarray(prep.transform(X_train), dtype=float)
    Xt_eval = np.asarray(prep.transform(X_eval), dtype=float)

    explainers = [
        LimeTabularExplainer(
            Xt_tr,
            feature_names=names,
            class_names=["good", "bad"],
            mode="classification",
            random_state=s,
        )
        for s in seeds
    ]

    applicant_taus: list[float] = []
    for i in range(len(Xt_eval)):
        xt_app = Xt_eval[i]
        seed_ranks: list[np.ndarray] = []
        for exp_obj in explainers:
            exp = exp_obj.explain_instance(
                xt_app,
                clf.predict_proba,
                num_features=len(names),
                num_samples=num_samples,
                labels=(1,),
            )
            weights = dict.fromkeys(all_owners, 0.0)
            for feat_idx, w in exp.as_map()[1]:
                o = owners[feat_idx]
                weights[o] += float(w)
            s_ser = pd.Series(weights)
            seed_ranks.append(s_ser.abs().rank(ascending=False).values)

        pair_taus: list[float] = []
        for a in range(len(seeds)):
            for b in range(a + 1, len(seeds)):
                kt, _ = kendalltau(seed_ranks[a], seed_ranks[b])
                if not np.isnan(kt):
                    pair_taus.append(float(kt))
        if pair_taus:
            applicant_taus.append(float(np.mean(pair_taus)))

    return float(np.mean(applicant_taus)) if applicant_taus else 0.0


def shap_lime_jaccard_agreement(
    shap_rankings: Sequence[Any] | pd.DataFrame,
    lime_rankings: Sequence[Any] | pd.DataFrame,
    k: int = 5,
) -> float:
    """Compute the mean Jaccard overlap between top-k SHAP and top-k LIME features.
    Accepts either lists/tuples/sets of top features, or DataFrames of attribution values."""
    if isinstance(shap_rankings, pd.DataFrame) and isinstance(lime_rankings, pd.DataFrame):
        jaccards: list[float] = []
        for i in range(len(shap_rankings)):
            s_top = set(shap_rankings.iloc[i].abs().nlargest(k).index)
            l_top = set(lime_rankings.iloc[i].abs().nlargest(k).index)
            union = len(s_top | l_top)
            jaccards.append(len(s_top & l_top) / union if union > 0 else 1.0)
        return float(np.mean(jaccards)) if jaccards else 0.0

    if (
        len(shap_rankings) > 0
        and isinstance(shap_rankings[0], (list, tuple, set))
        and len(lime_rankings) > 0
        and isinstance(lime_rankings[0], (list, tuple, set))
    ):
        jaccards = []
        for s_list, l_list in zip(shap_rankings, lime_rankings, strict=False):
            s_set = set(list(s_list)[:k])
            l_set = set(list(l_list)[:k])
            union = len(s_set | l_set)
            jaccards.append(len(s_set & l_set) / union if union > 0 else 1.0)
        return float(np.mean(jaccards)) if jaccards else 0.0

    s_set = set(list(shap_rankings)[:k])
    l_set = set(list(lime_rankings)[:k])
    union = len(s_set | l_set)
    return float(len(s_set & l_set) / union if union > 0 else 1.0)


def measure_explanation_runtime(
    fitted_pipeline: Any,
    X_train: pd.DataFrame,
    X_sample: pd.DataFrame,
    categorical: list[str],
    n_runs: int = 3,
    num_samples: int = 1000,
) -> dict[str, float]:
    """Benchmark per-applicant explanation runtime for TreeSHAP versus LIME."""
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    prep = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    Xt_tr = np.asarray(prep.transform(X_train), dtype=float)
    Xt_sample = np.asarray(prep.transform(X_sample), dtype=float)

    # Benchmark SHAP
    explainer = shap.TreeExplainer(clf)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = explainer.shap_values(Xt_sample)
    t_shap = (time.perf_counter() - t0) / (n_runs * max(1, len(X_sample)))

    # Benchmark LIME
    lime_exp = LimeTabularExplainer(
        Xt_tr,
        feature_names=names,
        class_names=["good", "bad"],
        mode="classification",
        random_state=42,
    )
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for i in range(len(Xt_sample)):
            _ = lime_exp.explain_instance(
                Xt_sample[i],
                clf.predict_proba,
                num_features=len(names),
                num_samples=num_samples,
                labels=(1,),
            )
    t_lime = (time.perf_counter() - t0) / (n_runs * max(1, len(X_sample)))

    return {
        "shap_ms_per_applicant": round(t_shap * 1000, 3),
        "lime_ms_per_applicant": round(t_lime * 1000, 3),
        "speedup_ratio": round(t_lime / max(t_shap, 1e-9), 1),
    }
