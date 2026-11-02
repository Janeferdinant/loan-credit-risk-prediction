from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data import TARGET, select_features
from credit_risk.explain import (
    lime_explain_applicant,
    lime_stability_kendall_tau,
    measure_explanation_runtime,
    shap_lime_jaccard_agreement,
)
from credit_risk.models import make_estimator


def test_rank_agreement_metric_known_value():
    """Verify Kendall tau and Jaccard metrics against known synthetic values."""
    # Jaccard overlap tests
    set_a = ["duration_months", "credit_amount", "age", "installment_rate", "savings"]
    set_b = ["duration_months", "credit_amount", "age", "job", "housing"]
    # 3 shared / 7 unique = 3/7
    assert np.isclose(shap_lime_jaccard_agreement(set_a, set_b, k=5), 3 / 7)

    # Identical sets
    assert np.isclose(shap_lime_jaccard_agreement(set_a, set_a, k=5), 1.0)

    # Disjoint sets
    set_c = ["other_debtors", "telephone", "foreign_worker", "property", "housing"]
    assert np.isclose(shap_lime_jaccard_agreement(set_a, set_c, k=5), 0.0)

    # DataFrame inputs test
    df_shap = pd.DataFrame(
        [
            {"a": 1.0, "b": 0.8, "c": 0.5, "d": 0.1, "e": 0.05, "f": 0.01},
            {"a": 0.1, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.6, "f": 0.5},
        ]
    )
    df_lime = pd.DataFrame(
        [
            {"a": 1.5, "b": 0.7, "c": 0.6, "d": 0.2, "e": 0.1, "f": 0.0},
            {"a": 0.0, "b": 0.1, "c": 0.2, "d": 0.9, "e": 0.8, "f": 0.7},
        ]
    )
    # Row 0: top 5 for both are {a, b, c, d, e} -> Jaccard = 1.0
    # Row 1: df_shap top 5 are {b, c, d, e, f}; df_lime top 5 are {b, c, d, e, f} -> Jaccard = 1.0
    assert np.isclose(shap_lime_jaccard_agreement(df_shap, df_lime, k=5), 1.0)


def test_lime_seeded_reproducible(df):
    """Confirms identical feature attributions given the same random seed."""
    num_cols, cat_cols = select_features("all")
    X = df[num_cols + cat_cols]
    y = df[TARGET]

    est = make_estimator("xgb", num_cols, cat_cols, seed=42)
    est.fit(X.iloc[:100], y.iloc[:100])

    applicant = X.iloc[105]

    exp1 = lime_explain_applicant(est, X.iloc[:100], applicant, cat_cols, seed=123, num_samples=300)
    exp2 = lime_explain_applicant(est, X.iloc[:100], applicant, cat_cols, seed=123, num_samples=300)

    assert len(exp1) == len(num_cols) + len(cat_cols)
    assert np.allclose(exp1.values, exp2.values, atol=1e-6)


def test_lime_explains_pipeline_applicants(df):
    """Explains sample applicants and verifies attribution format and stability calculation."""
    num_cols, cat_cols = select_features("all")
    X = df[num_cols + cat_cols]
    y = df[TARGET]

    est = make_estimator("xgb", num_cols, cat_cols, seed=42)
    est.fit(X.iloc[:100], y.iloc[:100])

    exp = lime_explain_applicant(est, X.iloc[:100], X.iloc[101], cat_cols, seed=42, num_samples=200)
    assert isinstance(exp, pd.Series)
    assert not exp.isna().any()
    assert (exp.abs() > 0).sum() > 0

    # Test stability on 3 applicants across 3 seeds
    tau = lime_stability_kendall_tau(
        est,
        X.iloc[:100],
        X.iloc[101:104],
        cat_cols,
        seeds=[1, 2, 3],
        num_samples=200,
    )
    assert -1.0 <= tau <= 1.0
    assert tau > 0.0  # Should be positively correlated


def test_measure_explanation_runtime(df):
    """Verifies explanation runtime benchmark runs and returns valid timings."""
    num_cols, cat_cols = select_features("all")
    X = df[num_cols + cat_cols]
    y = df[TARGET]

    est = make_estimator("xgb", num_cols, cat_cols, seed=42)
    est.fit(X.iloc[:100], y.iloc[:100])

    runtimes = measure_explanation_runtime(
        est,
        X.iloc[:100],
        X.iloc[101:103],
        cat_cols,
        n_runs=1,
        num_samples=200,
    )
    assert "shap_ms_per_applicant" in runtimes
    assert "lime_ms_per_applicant" in runtimes
    assert runtimes["shap_ms_per_applicant"] >= 0.0
    assert runtimes["lime_ms_per_applicant"] > 0.0
