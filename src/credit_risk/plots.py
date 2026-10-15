"""Static figures for the README (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILION, PURPLE, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#6b7280",
)
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    }
)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def model_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    """Forest plot of test AUC with bootstrap CI and nested-CV mean."""
    rows = [r for r in rows if r["model"] != "majority"]
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(rows) + 1.2))
    for i, r in enumerate(rows):
        ax.errorbar(
            r["test_auc"], i, xerr=[[r["test_auc"] - r["test_auc_lo"]], [r["test_auc_hi"] - r["test_auc"]]],
            fmt="o", color=BLUE, capsize=3, label="test (95% bootstrap CI)" if i == 0 else None,
        )  # fmt: skip
        ax.plot(r["cv_auc"], i, "D", color=ORANGE, label="nested CV mean" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("ROC AUC")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, path)


def reliability(curves: dict[str, list[dict[str, float]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="perfect")
    for (label, bins), colour in zip(curves.items(), [BLUE, ORANGE, GREEN, VERMILION, PURPLE], strict=False):
        ax.plot(
            [b["mean_pred"] for b in bins],
            [b["observed"] for b in bins],
            "o-",
            ms=4,
            color=colour,
            label=label,
        )
    ax.set_xlabel("predicted P(bad)")
    ax.set_ylabel("observed share bad")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reliability on the test split (10 bins)", fontsize=9)
    _save(fig, path)


def cost_vs_threshold(
    grid: np.ndarray,
    test_cost: np.ndarray,
    oof_cost: np.ndarray,
    marks: dict[str, float],
    baselines: dict[str, float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(grid, oof_cost, color=ORANGE, label="out-of-fold (train rows)")
    ax.plot(grid, test_cost, color=BLUE, label="test split")
    for (name, t), colour in zip(marks.items(), [GREEN, VERMILION, PURPLE], strict=False):
        ax.axvline(t, color=colour, ls="--", lw=1, label=f"{name} ({t:.3f})")
    for name, c in baselines.items():
        ax.axhline(c, color=GREY, ls=":", lw=1)
        ax.text(0.99, c, name, ha="right", va="bottom", fontsize=7, color=GREY)
    ax.set_xlabel("approve if P(bad) < threshold")
    ax.set_ylabel("mean cost per applicant")
    ax.legend(frameon=False, loc="center right", fontsize=7)
    _save(fig, path)


def shap_summary(values: pd.DataFrame, X: pd.DataFrame, path: Path, top: int = 12) -> None:
    order = values.abs().mean().sort_values(ascending=False).index[:top]
    fig, ax = plt.subplots(figsize=(6.6, 0.38 * top + 1.2))
    rng = np.random.default_rng(0)
    for i, f in enumerate(order):
        v = values[f].to_numpy()
        y = i + rng.uniform(-0.28, 0.28, len(v))
        col = X[f]
        if pd.api.types.is_numeric_dtype(col):
            rank = col.rank(pct=True).to_numpy()
            ax.scatter(v, y, c=rank, cmap="coolwarm", s=7, alpha=0.8, linewidths=0)
        else:
            ax.scatter(v, y, color=GREY, s=7, alpha=0.6, linewidths=0)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (log-odds of bad credit); numeric features coloured low (blue) to high (red)")
    ax.xaxis.label.set_size(8)
    _save(fig, path)


def shap_bar(importance: pd.Series, path: Path, top: int = 12) -> None:
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(imp) + 1.0))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("mean |SHAP| (log-odds), test split")
    _save(fig, path)


def partial_dependence(pds: list[dict[str, Any]], path: Path) -> None:
    n = len(pds)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), sharey=True)
    for ax, pd_ in zip(np.atleast_1d(axes), pds, strict=True):
        if pd_["categorical"]:
            ax.bar(range(len(pd_["grid"])), pd_["average"], color=BLUE)
            ax.set_xticks(range(len(pd_["grid"])))
            ax.set_xticklabels(pd_["grid"], rotation=60, ha="right", fontsize=7)
        else:
            for ice in pd_["ice"]:
                ax.plot(pd_["grid"], ice, color=GREY, alpha=0.18, lw=0.7)
            ax.plot(pd_["grid"], pd_["average"], color=BLUE, lw=2)
        ax.set_xlabel(pd_["feature"])
    np.atleast_1d(axes)[0].set_ylabel("P(bad)")
    _save(fig, path)


def fairness_bars(groups: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 3.0), sharey=True)
    for ax, (attr, rep) in zip(np.atleast_1d(axes), groups.items(), strict=True):
        x = np.arange(len(rep))
        w = 0.27
        ax.bar(x - w, rep["approval_rate"], w, color=BLUE, label="approval rate")
        ax.bar(x, rep["approve_good_rate"], w, color=GREEN, label="good applicants approved")
        ax.bar(x + w, rep["approve_bad_rate"], w, color=VERMILION, label="bad applicants approved")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(rep["group"], rep["n"], strict=True)], fontsize=8)
        ax.set_title(attr, fontsize=9)
        ax.set_ylim(0, 1)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
