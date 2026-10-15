from __future__ import annotations

import numpy as np
import pytest

from credit_risk import metrics


def test_theoretical_threshold_matches_cost_matrix():
    # approve a bad applicant costs 5, reject a good applicant costs 1 -> approve if p_bad < 1/6
    assert metrics.theoretical_threshold(5, 1) == pytest.approx(1 / 6)
    assert metrics.theoretical_threshold(1, 1) == pytest.approx(0.5)


def test_expected_cost_orientation():
    y = np.array([1, 0, 1, 0])  # 1 = bad
    approve = np.array([True, True, False, False])
    # row0: bad approved -> 5 ; row1: good approved -> 0 ; row2: bad rejected -> 0 ; row3: good rejected -> 1
    assert metrics.expected_cost(y, approve, cost_false_approval=5, cost_false_rejection=1) == 1.5


def test_decide_is_strict():
    p = np.array([0.1, 1 / 6, 0.5])
    assert list(metrics.decide_approve(p, 1 / 6)) == [True, False, False]


def test_baselines_cost():
    y = np.array([1] * 3 + [0] * 7)
    assert metrics.approve_all_cost(y, 5, 1) == pytest.approx(1.5)
    assert metrics.reject_all_cost(y, 5, 1) == pytest.approx(0.7)


def test_best_threshold_minimises_cost():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(int)
    p = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.15, 2000), 0.001, 0.999)
    t, c = metrics.best_threshold(y, p, 5, 1)
    grid = np.linspace(0, 1, 201)
    costs = [metrics.expected_cost(y, metrics.decide_approve(p, g), 5, 1) for g in grid]
    assert c <= min(costs) + 1e-12
    assert 0 <= t <= 1


def test_ece_perfectly_calibrated_is_zero():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    p = np.array([0.0] * 4 + [1.0] * 6)
    assert metrics.ece(y, p, n_bins=10) == pytest.approx(0.0)


def test_ece_constant_overconfident():
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.9)
    assert metrics.ece(y, p, n_bins=10) == pytest.approx(0.4)


def test_brier_and_logloss_known():
    y = np.array([1, 0])
    p = np.array([0.8, 0.3])
    assert metrics.brier(y, p) == pytest.approx((0.04 + 0.09) / 2)
    assert metrics.logloss(y, p) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)


def test_auc_perfect_and_random():
    assert metrics.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert np.isnan(metrics.auc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_bootstrap_ci_properties():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.3).astype(int)
    p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 300), 0.01, 0.99)
    lo, hi = metrics.bootstrap_ci(y, p, metrics.auc, n_boot=300, seed=1)
    pt = metrics.auc(y, p)
    assert lo < pt < hi
    again = metrics.bootstrap_ci(y, p, metrics.auc, n_boot=300, seed=1)
    assert (lo, hi) == again
    # more data -> narrower interval
    y2 = np.tile(y, 4)
    p2 = np.tile(p, 4)
    lo2, hi2 = metrics.bootstrap_ci(y2, p2, metrics.auc, n_boot=300, seed=1)
    assert (hi2 - lo2) < (hi - lo)


def test_bootstrap_is_stratified_keeps_both_classes():
    y = np.array([1] + [0] * 49)
    p = np.linspace(0.9, 0.1, 50)
    lo, hi = metrics.bootstrap_ci(y, p, metrics.auc, n_boot=100, seed=0)
    assert not np.isnan(lo) and not np.isnan(hi)


def test_paired_bootstrap_difference():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.3).astype(int)
    good = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.1, 400), 0.01, 0.99)
    bad = rng.random(400)
    diff, lo, hi = metrics.paired_bootstrap_diff(y, good, bad, metrics.auc, n_boot=300, seed=0)
    assert diff > 0 and lo > 0 and hi >= lo
