"""Metrics, the cost matrix and bootstrap intervals.

Convention: ``y`` is 1 for a bad credit risk. ``p`` is the predicted probability of *bad*. An
applicant is approved when ``p < threshold``.

Cost matrix (from the German Credit documentation): approving an applicant who turns out bad
costs 5; rejecting an applicant who would have repaid costs 1; correct decisions cost 0.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

COST_FALSE_APPROVAL = 5.0
COST_FALSE_REJECTION = 1.0

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_pred": float(p[m].mean()),
                    "observed": float(y[m].mean()),
                }
            )
    return out


def theoretical_threshold(
    cost_false_approval: float = COST_FALSE_APPROVAL,
    cost_false_rejection: float = COST_FALSE_REJECTION,
) -> float:
    """Approve when p_bad < c_rej / (c_app + c_rej): the Bayes-optimal rule for calibrated p."""
    return cost_false_rejection / (cost_false_approval + cost_false_rejection)


def decide_approve(p: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(p) < threshold


def expected_cost(
    y: np.ndarray,
    approve: np.ndarray,
    cost_false_approval: float = COST_FALSE_APPROVAL,
    cost_false_rejection: float = COST_FALSE_REJECTION,
) -> float:
    """Mean cost per applicant."""
    y = np.asarray(y).astype(bool)
    approve = np.asarray(approve).astype(bool)
    false_approval = y & approve
    false_rejection = ~y & ~approve
    return float(
        (cost_false_approval * false_approval.sum() + cost_false_rejection * false_rejection.sum()) / len(y)
    )


def approve_all_cost(y: np.ndarray, cost_false_approval: float, cost_false_rejection: float) -> float:
    return expected_cost(y, np.ones(len(y), bool), cost_false_approval, cost_false_rejection)


def reject_all_cost(y: np.ndarray, cost_false_approval: float, cost_false_rejection: float) -> float:
    return expected_cost(y, np.zeros(len(y), bool), cost_false_approval, cost_false_rejection)


def cost_curve(
    y: np.ndarray,
    p: np.ndarray,
    cost_false_approval: float = COST_FALSE_APPROVAL,
    cost_false_rejection: float = COST_FALSE_REJECTION,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0, 1, 201) if grid is None else np.asarray(grid)
    costs = np.array(
        [expected_cost(y, decide_approve(p, t), cost_false_approval, cost_false_rejection) for t in grid]
    )
    return grid, costs


def best_threshold(
    y: np.ndarray,
    p: np.ndarray,
    cost_false_approval: float = COST_FALSE_APPROVAL,
    cost_false_rejection: float = COST_FALSE_REJECTION,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Threshold minimising mean cost on (y, p); ties resolved by the middle of the plateau."""
    grid, costs = cost_curve(y, p, cost_false_approval, cost_false_rejection, grid)
    best = costs.min()
    ties = grid[np.isclose(costs, best, atol=1e-12)]
    return float(np.median(ties)), float(best)


def _strata(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(len(y))
    parts = []
    for cls in np.unique(y):
        members = idx[y == cls]
        parts.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(parts)


def bootstrap_indices(y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) stratified bootstrap row indices, shared across metrics and models."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    return np.stack([_strata(y, rng) for _ in range(n_boot)])


def ci_from_indices(
    y: np.ndarray, p: np.ndarray, fn: MetricFn, idx: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    y, p = np.asarray(y), np.asarray(p)
    vals = np.array([fn(y[i], p[i]) for i in idx])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    fn: MetricFn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling within each class so both classes stay present."""
    return ci_from_indices(y, p, fn, bootstrap_indices(y, n_boot, seed), alpha)


def paired_bootstrap_diff(
    y: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    fn: MetricFn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """fn(y,p1) - fn(y,p2) with a paired percentile bootstrap CI (same resamples for both)."""
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    idx = bootstrap_indices(y, n_boot, seed)
    vals = np.array([fn(y[i], p1[i]) - fn(y[i], p2[i]) for i in idx])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(y, p1) - fn(y, p2)), float(lo), float(hi)
