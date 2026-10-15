"""Group-wise decision metrics for a documented sensitive attribute.

All rates are defined for the *approve* decision at a fixed threshold:

* approval_rate: share of the group approved (demographic-parity view).
* approve_good_rate: share of truly good applicants approved (equal-opportunity view).
* approve_bad_rate: share of truly bad applicants approved (the costly error).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.metrics import auc

RATE_KEYS = ("approval_rate", "approve_good_rate", "approve_bad_rate")


def _rates(y: np.ndarray, approve: np.ndarray) -> dict[str, float]:
    good = y == 0
    bad = y == 1
    return {
        "approval_rate": float(approve.mean()) if len(y) else float("nan"),
        "approve_good_rate": float(approve[good].mean()) if good.any() else float("nan"),
        "approve_bad_rate": float(approve[bad].mean()) if bad.any() else float("nan"),
    }


def group_report(
    y: np.ndarray, p: np.ndarray, threshold: float, groups: pd.Series | np.ndarray
) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    g = np.asarray(groups)
    rows = []
    for level in pd.unique(g):
        m = g == level
        approve = p[m] < threshold
        rows.append(
            {
                "group": level,
                "n": int(m.sum()),
                "bad_rate": float(y[m].mean()),
                **_rates(y[m], approve),
                "auc": auc(y[m], p[m]),
                "mean_pred_bad": float(p[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def gap_with_ci(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    groups: pd.Series | np.ndarray,
    group: str,
    reference: str,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float, float]]:
    """(estimate, lo, hi) of rate(group) - rate(reference); bootstrap resamples each group."""
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(groups)
    ia, ib = np.where(g == group)[0], np.where(g == reference)[0]
    rng = np.random.default_rng(seed)

    def gaps(ja: np.ndarray, jb: np.ndarray) -> dict[str, float]:
        ra = _rates(y[ja], p[ja] < threshold)
        rb = _rates(y[jb], p[jb] < threshold)
        return {f"{k}_gap": ra[k] - rb[k] for k in RATE_KEYS}

    est = gaps(ia, ib)
    boots: dict[str, list[float]] = {k: [] for k in est}
    for _ in range(n_boot):
        ja = rng.choice(ia, size=len(ia), replace=True)
        jb = rng.choice(ib, size=len(ib), replace=True)
        for k, v in gaps(ja, jb).items():
            boots[k].append(v)
    out = {}
    for k, v in est.items():
        arr = np.asarray(boots[k], float)
        if np.isnan(arr).all():
            lo = hi = float("nan")
        else:
            lo, hi = np.nanpercentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = (float(v), float(lo), float(hi))
    return out
