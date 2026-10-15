"""Train / evaluate / sweep orchestration. Everything written here is reproducible from a seed."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_risk import explain, fairness, metrics, plots, runinfo
from credit_risk.data import (
    TARGET,
    add_sensitive,
    load_german_credit,
    select_features,
)
from credit_risk.models import MODEL_LABELS, MODEL_NAMES, calibrations_for, make_estimator
from credit_risk.splits import holdout_split
from credit_risk.tuning import nested_cv, tune_params

COST_FA = metrics.COST_FALSE_APPROVAL
COST_FR = metrics.COST_FALSE_REJECTION
CV_METRICS = ("auc", "logloss", "brier", "ece", "cost_at_theoretical_threshold")


@dataclass
class Config:
    out: Path = Path(".")
    data_file: Path | None = None
    data_dir: Path | None = None
    seed: int = 42
    test_size: float = 0.2
    trials: int = 20
    splits: int = 5
    repeats: int = 3
    inner_splits: int = 3
    jobs: int = 4
    models: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    bootstrap: int = 2000

    @property
    def artifacts(self) -> Path:
        return self.out / "artifacts"

    @property
    def results(self) -> Path:
        return self.out / "results"

    @property
    def img(self) -> Path:
        return self.out / "docs" / "img"

    @property
    def runs(self) -> Path:
        return self.out / "docs" / "runs"


def key_of(name: str, cal: str) -> str:
    return f"{name}|{cal}"


def label_of(name: str, cal: str) -> str:
    base = MODEL_LABELS[name]
    return base if cal == "raw" else f"{base} + {cal} calibration"


def load_frame(cfg: Config) -> pd.DataFrame:
    return add_sensitive(load_german_credit(cfg.data_file, cfg.data_dir)).reset_index(drop=True)


def _aggregate_nested(res, y_train: np.ndarray) -> list[dict[str, Any]]:
    folds = pd.DataFrame(res.folds)
    rows = []
    for (name, cal), oof in res.oof.items():
        sub = folds[(folds["model"] == name) & (folds["calibration"] == cal)]
        row: dict[str, Any] = {"model": name, "calibration": cal, "label": label_of(name, cal)}
        for m in CV_METRICS:
            row[m] = {"mean": float(sub[m].mean()), "sd": float(sub[m].std(ddof=1))}
        pooled = {
            "auc": np.mean([metrics.auc(y_train, oof[r]) for r in range(oof.shape[0])]),
            "logloss": np.mean([metrics.logloss(y_train, oof[r]) for r in range(oof.shape[0])]),
        }
        row["pooled_oof"] = {k: float(v) for k, v in pooled.items()}
        rows.append(row)
    return rows


def select_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-declared rule: lowest mean nested-CV log loss among non-baseline variants."""
    cands = [r for r in rows if r["model"] != "majority"]
    return min(cands, key=lambda r: r["logloss"]["mean"])


def run_train(cfg: Config) -> dict[str, Any]:
    t0 = time.time()
    cfg.results.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    df = load_frame(cfg)
    numeric, categorical = select_features("all")
    X = df[numeric + categorical]
    y = df[TARGET].to_numpy()
    tr, te = holdout_split(y, cfg.test_size, cfg.seed)
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]

    res = nested_cv(
        Xtr, ytr, numeric, categorical, cfg.models,
        n_splits=cfg.splits, n_repeats=cfg.repeats, n_trials=cfg.trials,
        inner_splits=cfg.inner_splits, seed=cfg.seed, n_jobs=cfg.jobs,
    )  # fmt: skip
    t_nested = time.time() - t0
    rows = _aggregate_nested(res, ytr)
    selected = select_variant(rows)

    # pooled out-of-fold probabilities (train rows only) for threshold choice and fairness
    oof_tbl = {"row": tr.tolist(), "y": ytr.tolist()}
    for (name, cal), arr in res.oof.items():
        oof_tbl[key_of(name, cal)] = np.round(np.nanmean(arr, axis=0), 6).tolist()
    pd.DataFrame(oof_tbl).to_csv(cfg.results / "oof_predictions.csv", index=False)
    stacked = {
        key_of(*k): {"p": arr.tolist()} for k, arr in res.oof.items()
    }  # per-repeat OOF kept for the threshold analysis
    joblib.dump(stacked, cfg.artifacts / "oof_by_repeat.joblib")

    # final models: tune on the full training split, fit every calibration variant
    t1 = time.time()
    final_params: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for name in cfg.models:
        tuned = tune_params(
            name, Xtr, ytr, numeric, categorical,
            seed=cfg.seed, n_trials=cfg.trials, inner_splits=cfg.inner_splits,
        )  # fmt: skip
        final_params[name] = {"params": tuned.params, "inner_logloss": tuned.value}
        for cal in calibrations_for(name):
            est = make_estimator(name, numeric, categorical, tuned.params, cfg.seed, cal)
            models[key_of(name, cal)] = est.fit(Xtr, ytr)
    t_final = time.time() - t1

    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, cfg.artifacts / "models.joblib")
    meta = {
        "seed": cfg.seed,
        "numeric": numeric,
        "categorical": categorical,
        "train_rows": tr.tolist(),
        "test_rows": te.tolist(),
        "selected": key_of(selected["model"], selected["calibration"]),
        "final_params": final_params,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
    }
    runinfo.dump_json(meta, cfg.artifacts / "meta.json", indent=None)
    nested = {
        "protocol": {
            "holdout_test_size": cfg.test_size,
            "outer_splits": cfg.splits,
            "outer_repeats": cfg.repeats,
            "inner_splits": cfg.inner_splits,
            "optuna_trials_per_model": cfg.trials,
            "tuning_objective": "mean inner-CV log loss, TPE sampler seeded per fold",
            "seed": cfg.seed,
            "n_train": int(len(tr)),
        },
        "selected": meta["selected"],
        "summary": rows,
        "folds": res.folds,
        "tuned_params_per_fold": res.params,
    }
    runinfo.dump_json(nested, cfg.results / "nested_cv.json")
    train_info = {
        "env": runinfo.environment(),
        "seconds_nested_cv": round(t_nested, 1),
        "seconds_final_fit": round(t_final, 1),
        "config": meta["config"],
        "final_params": final_params,
    }
    runinfo.dump_json(train_info, cfg.results / "train_run.json")
    return {"selected": meta["selected"], "rows": rows, "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------------------------


def _ci_row(y, p, fn, idx):
    lo, hi = metrics.ci_from_indices(y, p, fn, idx)
    return {"value": float(fn(y, p)), "lo": lo, "hi": hi}


def _fairness_block(
    y: np.ndarray, p: np.ndarray, groups: pd.DataFrame, threshold: float, n_boot: int, seed: int
) -> dict[str, Any]:
    spec = {
        "sex": ("female", "male"),
        "age_group": ("<25", ">=25"),
        "foreign_worker": ("no", "yes"),
    }
    out: dict[str, Any] = {}
    for attr, (grp, ref) in spec.items():
        rep = fairness.group_report(y, p, threshold, groups[attr].to_numpy())
        gaps = fairness.gap_with_ci(
            y, p, threshold, groups[attr].to_numpy(), grp, ref, n_boot=n_boot, seed=seed
        )
        out[attr] = {
            "group": grp,
            "reference": ref,
            "table": rep.to_dict(orient="records"),
            "gaps": {k: {"est": v[0], "lo": v[1], "hi": v[2]} for k, v in gaps.items()},
        }
    return out


def run_evaluate(cfg: Config, ablation: bool = True) -> dict[str, Any]:
    t0 = time.time()
    meta = runinfo.load_json(cfg.artifacts / "meta.json")
    for k in ("seed", "test_size", "trials", "splits", "repeats", "inner_splits"):
        setattr(cfg, k, meta["config"][k])  # evaluate must reuse the protocol that train used
    nested = runinfo.load_json(cfg.results / "nested_cv.json")
    train_info = runinfo.load_json(cfg.results / "train_run.json")
    models = joblib.load(cfg.artifacts / "models.joblib")
    oof_by_repeat = joblib.load(cfg.artifacts / "oof_by_repeat.joblib")
    df = load_frame(cfg)
    numeric, categorical = meta["numeric"], meta["categorical"]
    X = df[numeric + categorical]
    y = df[TARGET].to_numpy()
    tr, te = np.array(meta["train_rows"]), np.array(meta["test_rows"])
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
    Xte, yte = X.iloc[te].reset_index(drop=True), y[te]
    groups_tr = df.iloc[tr][["sex", "age_group", "foreign_worker"]].reset_index(drop=True)
    groups_te = df.iloc[te][["sex", "age_group", "foreign_worker"]].reset_index(drop=True)
    selected_key = meta["selected"]
    cv_rows = {key_of(r["model"], r["calibration"]): r for r in nested["summary"]}
    theo = metrics.theoretical_threshold(COST_FA, COST_FR)
    idx = metrics.bootstrap_indices(yte, cfg.bootstrap, cfg.seed)

    # --- test metrics with bootstrap intervals ------------------------------------------------
    test_p: dict[str, np.ndarray] = {}
    table: list[dict[str, Any]] = []
    oof_thr: dict[str, float] = {}
    for key, est in models.items():
        name, cal = key.split("|")
        p = est.predict_proba(Xte)[:, 1]
        test_p[key] = p
        # threshold from this variant's own out-of-fold predictions on the training rows
        runs = np.asarray(oof_by_repeat[key]["p"])  # (repeats, n_train)
        thr, _ = metrics.best_threshold(np.tile(ytr, runs.shape[0]), runs.ravel(), COST_FA, COST_FR)
        oof_thr[key] = thr

        def cost_fn(t: float):
            return lambda yy, pp: metrics.expected_cost(yy, metrics.decide_approve(pp, t), COST_FA, COST_FR)

        row = {
            "key": key,
            "model": name,
            "calibration": cal,
            "label": label_of(name, cal),
            "oof_threshold": thr,
            "auc": _ci_row(yte, p, metrics.auc, idx),
            "pr_auc": _ci_row(yte, p, metrics.pr_auc, idx),
            "logloss": _ci_row(yte, p, metrics.logloss, idx),
            "brier": _ci_row(yte, p, metrics.brier, idx),
            "ece": _ci_row(yte, p, metrics.ece, idx),
            "cost_at_theoretical": _ci_row(yte, p, cost_fn(theo), idx),
            "cost_at_oof_threshold": _ci_row(yte, p, cost_fn(thr), idx),
            "cost_at_0.5": _ci_row(yte, p, cost_fn(0.5), idx),
        }
        table.append(row)
    by_key = {r["key"]: r for r in table}

    # --- threshold analysis for the selected variant ------------------------------------------
    sel = by_key[selected_key]
    thr_op = sel["oof_threshold"]
    p_sel = test_p[selected_key]
    all_p_oof = np.asarray(oof_by_repeat[selected_key]["p"])
    grid, oof_curve = metrics.cost_curve(
        np.tile(ytr, all_p_oof.shape[0]), all_p_oof.ravel(), COST_FA, COST_FR
    )
    _, test_curve = metrics.cost_curve(yte, p_sel, COST_FA, COST_FR, grid)
    approve_all = metrics.approve_all_cost(yte, COST_FA, COST_FR)
    reject_all = metrics.reject_all_cost(yte, COST_FA, COST_FR)
    cost_op = lambda yy, pp: metrics.expected_cost(yy, metrics.decide_approve(pp, thr_op), COST_FA, COST_FR)  # noqa: E731
    # paired bootstrap: policy cost minus the trivial policies' cost on the same resamples
    diffs = np.array(
        [cost_op(yte[j], p_sel[j]) - metrics.reject_all_cost(yte[j], COST_FA, COST_FR) for j in idx]
    )
    diffs_all = np.array(
        [cost_op(yte[j], p_sel[j]) - metrics.approve_all_cost(yte[j], COST_FA, COST_FR) for j in idx]
    )
    threshold_block = {
        "selected": selected_key,
        "theoretical": theo,
        "oof_optimal": thr_op,
        "cost_matrix": {"false_approval": COST_FA, "false_rejection": COST_FR},
        "test_cost_at_operating_threshold": sel["cost_at_oof_threshold"],
        "test_cost_at_theoretical": sel["cost_at_theoretical"],
        "test_cost_at_0.5": sel["cost_at_0.5"],
        "test_cost_approve_all": approve_all,
        "test_cost_reject_all": reject_all,
        "diff_vs_reject_all": {
            "est": float(cost_op(yte, p_sel) - reject_all),
            "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
        },
        "diff_vs_approve_all": {
            "est": float(cost_op(yte, p_sel) - approve_all),
            "lo": float(np.percentile(diffs_all, 2.5)),
            "hi": float(np.percentile(diffs_all, 97.5)),
        },
        "curve": {"grid": grid, "oof_cost": oof_curve, "test_cost": test_curve},
    }

    # --- paired comparison of the headline models (AUC) ---------------------------------------
    paired = {}
    for a, b in (("xgb|raw", "logreg|raw"), ("hgb|raw", "logreg|raw"), ("xgb|raw", "hgb|raw")):
        if a in test_p and b in test_p:
            d, lo, hi = metrics.paired_bootstrap_diff(
                yte, test_p[a], test_p[b], metrics.auc, n_boot=cfg.bootstrap, seed=cfg.seed
            )
            paired[f"{a} minus {b}"] = {"auc_diff": d, "lo": lo, "hi": hi}

    # --- fairness -------------------------------------------------------------------------------
    oof_mean = pd.read_csv(cfg.results / "oof_predictions.csv")[selected_key].to_numpy()
    fair_oof = _fairness_block(ytr, oof_mean, groups_tr, thr_op, cfg.bootstrap, cfg.seed)
    fair_test = _fairness_block(yte, p_sel, groups_te, thr_op, cfg.bootstrap, cfg.seed)
    fairness_block = {
        "threshold": thr_op,
        "model": selected_key,
        "pooled_out_of_fold_train_rows": fair_oof,
        "test_split": fair_test,
    }

    # --- ablation: drop the sensitive attributes -------------------------------------------------
    ablation_block: dict[str, Any] | None = None
    if ablation:
        sel_name = selected_key.split("|")[0]
        num2, cat2 = select_features("no_sensitive")
        res2 = nested_cv(
            Xtr[num2 + cat2], ytr, num2, cat2, [sel_name],
            n_splits=cfg.splits, n_repeats=cfg.repeats, n_trials=cfg.trials,
            inner_splits=cfg.inner_splits, seed=cfg.seed, n_jobs=cfg.jobs,
        )  # fmt: skip
        cal = selected_key.split("|")[1]
        oof2 = np.nanmean(res2.oof[(sel_name, cal)], axis=0)
        fair2 = _fairness_block(ytr, oof2, groups_tr, thr_op, cfg.bootstrap, cfg.seed)
        ablation_block = {
            "model": selected_key,
            "features_dropped": ["personal_status", "age", "foreign_worker"],
            "auc_oof_all_features": float(np.mean([metrics.auc(ytr, r) for r in all_p_oof])),
            "auc_oof_no_sensitive": float(np.mean([metrics.auc(ytr, r) for r in res2.oof[(sel_name, cal)]])),
            "cost_oof_all_features": float(
                metrics.expected_cost(ytr, metrics.decide_approve(oof_mean, thr_op), COST_FA, COST_FR)
            ),
            "cost_oof_no_sensitive": float(
                metrics.expected_cost(ytr, metrics.decide_approve(oof2, thr_op), COST_FA, COST_FR)
            ),
            "fairness_no_sensitive": fair2,
        }

    # --- label-permutation leakage check on the real data ---------------------------------------
    rng = np.random.default_rng(cfg.seed)
    y_perm = rng.permutation(ytr)
    perm = nested_cv(
        Xtr, y_perm, numeric, categorical, ["logreg", "hgb"],
        n_splits=5, n_repeats=1, n_trials=min(cfg.trials, 5),
        inner_splits=cfg.inner_splits, seed=cfg.seed, n_jobs=cfg.jobs,
    )  # fmt: skip
    leakage = {
        "description": "full nested pipeline run on permuted training labels; honest AUC is about 0.5",
        "auc_logreg_permuted": metrics.auc(y_perm, perm.oof[("logreg", "raw")][0]),
        "auc_hgb_permuted": metrics.auc(y_perm, perm.oof[("hgb", "raw")][0]),
    }

    # --- SHAP (XGBoost) and partial dependence (HGB) --------------------------------------------
    sh = explain.shap_by_feature(models["xgb|raw"], Xte, categorical)
    imp_test = explain.mean_abs_importance(sh["values"])
    sh_tr = explain.shap_by_feature(models["xgb|raw"], Xtr, categorical)
    imp_train = explain.mean_abs_importance(sh_tr["values"])
    pdp_feats = [
        ("duration_months", False),
        ("credit_amount", False),
        ("age", False),
        ("checking_status", True),
    ]
    pdps = [explain.partial_dependence_1d(models["hgb|raw"], Xtr, f, c, seed=cfg.seed) for f, c in pdp_feats]

    # --- figures --------------------------------------------------------------------------------
    cmp_rows = []
    for r in table:
        cv = cv_rows[r["key"]]
        cmp_rows.append(
            {
                "model": r["model"],
                "label": r["label"],
                "test_auc": r["auc"]["value"],
                "test_auc_lo": r["auc"]["lo"],
                "test_auc_hi": r["auc"]["hi"],
                "cv_auc": cv["auc"]["mean"],
            }
        )
    plots.model_comparison(cmp_rows, cfg.img / "model_comparison.png")
    sel_name = selected_key.split("|")[0]
    rel = {
        label_of(sel_name, c): metrics.reliability_bins(yte, test_p[key_of(sel_name, c)], 10)
        for c in calibrations_for(sel_name)
    }
    plots.reliability(rel, cfg.img / "reliability.png")
    plots.cost_vs_threshold(
        grid, test_curve, oof_curve,
        {"theoretical 1/6": theo, "OOF optimum": thr_op, "default 0.5": 0.5},
        {"approve all": approve_all, "reject all": reject_all},
        cfg.img / "cost_vs_threshold.png",
    )  # fmt: skip
    plots.shap_summary(sh["values"], Xte, cfg.img / "shap_summary.png")
    plots.shap_bar(imp_test, cfg.img / "shap_importance.png")
    plots.partial_dependence(pdps, cfg.img / "partial_dependence.png")
    plots.fairness_bars({a: pd.DataFrame(fair_oof[a]["table"]) for a in fair_oof}, cfg.img / "fairness.png")

    # --- outputs --------------------------------------------------------------------------------
    cv_tbl = nested["summary"]
    test_summary = {
        "seed": cfg.seed,
        "date": runinfo.environment()["date"],
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "bad_rate_train": float(ytr.mean()),
        "bad_rate_test": float(yte.mean()),
        "selected": selected_key,
        "selection_rule": "lowest mean nested-CV log loss among non-baseline variants (test split not used)",
        "threshold": {"theoretical": theo, "oof_optimal": thr_op},
        "bootstrap_resamples": cfg.bootstrap,
        "test": table,
        "nested_cv": cv_tbl,
        "paired_auc_differences": paired,
        "threshold_analysis": threshold_block,
        "fairness_test_n": {a: {g["group"]: g["n"] for g in fair_test[a]["table"]} for a in fair_test},
        "leakage_check": leakage,
    }
    runinfo.dump_json(test_summary, cfg.results / "summary.json")
    runinfo.dump_json(fairness_block, cfg.results / "fairness.json")
    if ablation_block:
        runinfo.dump_json(ablation_block, cfg.results / "ablation_no_sensitive.json")
    runinfo.dump_json(
        {
            "model": "xgb|raw",
            "units": "log-odds of bad credit, summed over one-hot columns",
            "importance_test": imp_test.to_dict(),
            "importance_train": imp_train.to_dict(),
        },
        cfg.results / "shap_importance.json",
    )
    runinfo.dump_json(pdps, cfg.results / "partial_dependence.json", indent=None)
    explorer = {
        "meta": {
            "dataset": "Statlog (German Credit Data), UCI, CC BY 4.0",
            "seed": cfg.seed,
            "date": test_summary["date"],
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "cost_false_approval": COST_FA,
            "cost_false_rejection": COST_FR,
            "theoretical_threshold": theo,
            "selected": selected_key,
            "selected_label": label_of(*selected_key.split("|")),
            "oof_threshold": dict(oof_thr),
        },
        "models": [
            {
                "key": r["key"],
                "label": r["label"],
                "cv": cv_rows[r["key"]],
                "test": {m: r[m] for m in ("auc", "pr_auc", "logloss", "brier", "ece")},
                "cost_at_theoretical": r["cost_at_theoretical"],
                "cost_at_oof_threshold": r["cost_at_oof_threshold"],
            }
            for r in table
        ],
        "importance": [{"feature": f, "mean_abs_shap": float(v)} for f, v in imp_test.items()],
        "threshold_data": {
            "y": yte.tolist(),
            "p": {k: np.round(v, 4).tolist() for k, v in test_p.items() if not k.startswith("majority")},
        },
    }
    runinfo.dump_json(explorer, cfg.results / "explorer_data.json", indent=None)
    runinfo.dump_json({"seconds_evaluate": round(time.time() - t0, 1)}, cfg.results / "evaluate_run.json")
    from credit_risk.report import write_run_log

    write_run_log(cfg, train_info, test_summary, fairness_block, ablation_block, round(time.time() - t0, 1))
    return test_summary


# ----------------------------------------------------------------------------------------------
# seed sweep
# ----------------------------------------------------------------------------------------------


def run_sweep(cfg: Config, seeds: list[int]) -> dict[str, Any]:
    """Repeat tuning + hold-out evaluation for several split/tuning seeds (raw and sigmoid-calibrated)."""
    t0 = time.time()
    df = load_frame(cfg)
    numeric, categorical = select_features("all")
    X = df[numeric + categorical]
    y = df[TARGET].to_numpy()
    theo = metrics.theoretical_threshold(COST_FA, COST_FR)
    rows = []
    for s in seeds:
        tr, te = holdout_split(y, cfg.test_size, s)
        Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
        Xte, yte = X.iloc[te].reset_index(drop=True), y[te]
        for name in cfg.models:
            tuned = tune_params(
                name, Xtr, ytr, numeric, categorical,
                seed=s, n_trials=cfg.trials, inner_splits=cfg.inner_splits,
            )  # fmt: skip
            for cal in ("raw", "sigmoid") if name != "majority" else ("raw",):
                est = make_estimator(name, numeric, categorical, tuned.params, s, cal).fit(Xtr, ytr)
                p = est.predict_proba(Xte)[:, 1]
                rows.append(
                    {
                        "seed": s,
                        "model": name,
                        "calibration": cal,
                        "auc": metrics.auc(yte, p),
                        "logloss": metrics.logloss(yte, p),
                        "brier": metrics.brier(yte, p),
                        "ece": metrics.ece(yte, p),
                        "cost_at_theoretical": metrics.expected_cost(
                            yte, metrics.decide_approve(p, theo), COST_FA, COST_FR
                        ),
                    }
                )
    sweep = pd.DataFrame(rows)
    agg = (
        sweep.groupby(["model", "calibration"])[["auc", "logloss", "brier", "ece", "cost_at_theoretical"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    out = {
        "seeds": seeds,
        "trials": cfg.trials,
        "env": runinfo.environment(),
        "per_seed": rows,
        "aggregate": agg.to_dict(orient="records"),
        "seconds": round(time.time() - t0, 1),
    }
    runinfo.dump_json(out, cfg.results / "seed_sweep.json")
    from credit_risk.report import write_sweep_log

    write_sweep_log(cfg, out)
    return out
