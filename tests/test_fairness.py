from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk import fairness


def _example():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0])  # 1 = bad
    p = np.array([0.1, 0.2, 0.9, 0.3, 0.05, 0.8, 0.5, 0.1])
    g = pd.Series(["a", "a", "a", "a", "b", "b", "b", "b"])
    return y, p, g


def test_group_report_values():
    y, p, g = _example()
    rep = fairness.group_report(y, p, threshold=0.4, groups=g).set_index("group")
    a, b = rep.loc["a"], rep.loc["b"]
    assert a["n"] == 4 and b["n"] == 4
    assert a["bad_rate"] == 0.5 and b["bad_rate"] == 0.25
    # group a approves p < 0.4 -> rows 0,1,3 -> 0.75 ; group b -> rows 4,7 -> 0.5
    assert a["approval_rate"] == 0.75 and b["approval_rate"] == 0.5
    # of actual-good applicants, a approves both (1.0); b approves 2 of 3
    assert a["approve_good_rate"] == 1.0 and b["approve_good_rate"] == pytest.approx(2 / 3)
    # of actual-bad applicants, a approves 1 of 2 ; b approves 0 of 1
    assert a["approve_bad_rate"] == 0.5 and b["approve_bad_rate"] == 0.0


def test_single_class_group_has_nan_auc():
    y = np.array([0, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.9, 0.3])
    g = pd.Series(["a", "a", "a", "b"])
    rep = fairness.group_report(y, p, 0.5, g).set_index("group")
    assert np.isnan(rep.loc["b", "auc"])


def test_gap_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    n = 600
    g = pd.Series(rng.choice(["x", "y"], n, p=[0.7, 0.3]))
    y = (rng.random(n) < np.where(g == "x", 0.25, 0.4)).astype(int)
    p = np.clip(0.2 + 0.4 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
    out = fairness.gap_with_ci(y, p, 1 / 6, g, group="y", reference="x", n_boot=300, seed=0)
    for key in ("approval_rate_gap", "approve_good_rate_gap", "approve_bad_rate_gap"):
        est, lo, hi = out[key]
        assert lo <= est <= hi


def test_gap_sign_convention():
    y, p, g = _example()
    out = fairness.gap_with_ci(y, p, 0.4, g, group="b", reference="a", n_boot=50, seed=0)
    assert out["approval_rate_gap"][0] == pytest.approx(0.5 - 0.75)
