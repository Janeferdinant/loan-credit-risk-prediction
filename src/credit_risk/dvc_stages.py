"""DVC pipeline stages: prepare, train, evaluate.

Driven by params.yaml to provide fully reproducible command-line execution under DVC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from credit_risk import metrics
from credit_risk.data import TARGET, add_sensitive, load_german_credit, select_features
from credit_risk.models import make_estimator
from credit_risk.splits import holdout_split


def load_params(path: Path | str = "params.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_prepare(
    params_path: Path | str = "params.yaml", out_csv: Path | str = "results/prepared.csv"
) -> Path:
    params = load_params(params_path)
    prep_cfg = params.get("prepare", {})
    seed = prep_cfg.get("seed", 42)

    df = load_german_credit()
    df = add_sensitive(df)

    out_file = Path(out_csv)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Prepared dataset: {len(df)} rows written to {out_file} (seed: {seed})")
    return out_file


def stage_train(
    params_path: Path | str = "params.yaml",
    in_csv: Path | str = "results/prepared.csv",
    out_model: Path | str = "results/model.joblib",
) -> Path:
    params = load_params(params_path)
    prep_cfg = params.get("prepare", {})
    train_cfg = params.get("train", {})

    test_size = float(prep_cfg.get("test_size", 0.2))
    prep_seed = int(prep_cfg.get("seed", 42))
    train_seed = int(train_cfg.get("seed", 42))
    model_name = train_cfg.get("model_name", "xgb")

    df = pd.read_csv(in_csv)
    num_cols, cat_cols = select_features("all")

    tr_idx, _ = holdout_split(df[TARGET].values, test_size=test_size, seed=prep_seed)
    tr = df.iloc[tr_idx]

    # Extract model hyperparameters
    hyperparams = {
        k: v
        for k, v in train_cfg.items()
        if k not in ("model_name", "seed") and not isinstance(v, (dict, list))
    }

    est = make_estimator(
        name=model_name,
        numeric=num_cols,
        categorical=cat_cols,
        params=hyperparams,
        seed=train_seed,
    )
    est.fit(tr[num_cols + cat_cols], tr[TARGET])

    out_file = Path(out_model)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(est, out_file)
    print(f"Fitted {model_name} model on {len(tr)} training rows; saved to {out_file}")
    return out_file


def stage_evaluate(
    params_path: Path | str = "params.yaml",
    in_csv: Path | str = "results/prepared.csv",
    in_model: Path | str = "results/model.joblib",
    out_metrics: Path | str = "results/metrics.json",
) -> Path:
    params = load_params(params_path)
    prep_cfg = params.get("prepare", {})
    eval_cfg = params.get("evaluate", {})

    test_size = float(prep_cfg.get("test_size", 0.2))
    prep_seed = int(prep_cfg.get("seed", 42))

    cost_fa = float(eval_cfg.get("cost_false_approval", 5.0))
    cost_fr = float(eval_cfg.get("cost_false_rejection", 1.0))
    thr_op = float(eval_cfg.get("operating_threshold", 0.14))
    theo_thr = metrics.theoretical_threshold(cost_fa, cost_fr)

    df = pd.read_csv(in_csv)
    num_cols, cat_cols = select_features("all")
    _, te_idx = holdout_split(df[TARGET].values, test_size=test_size, seed=prep_seed)
    te = df.iloc[te_idx]

    est = joblib.load(in_model)
    X_te = te[num_cols + cat_cols]
    y_te = te[TARGET].values
    p_te = est.predict_proba(X_te)[:, 1]

    decisions_op = metrics.decide_approve(p_te, thr_op)
    decisions_theo = metrics.decide_approve(p_te, theo_thr)

    cost_op = float(metrics.expected_cost(y_te, decisions_op, cost_fa, cost_fr))
    cost_theo = float(metrics.expected_cost(y_te, decisions_theo, cost_fa, cost_fr))
    approve_rate = float(decisions_op.mean())

    res = {
        "auc": round(float(metrics.auc(y_te, p_te)), 4),
        "pr_auc": round(float(metrics.pr_auc(y_te, p_te)), 4),
        "logloss": round(float(metrics.logloss(y_te, p_te)), 4),
        "brier": round(float(metrics.brier(y_te, p_te)), 4),
        "cost_at_operating_threshold": round(cost_op, 4),
        "cost_at_theoretical_threshold": round(cost_theo, 4),
        "approval_rate_operating": round(approve_rate, 4),
        "n_test": len(te),
    }

    out_file = Path(out_metrics)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"Evaluation metrics written to {out_file}:")
    print(json.dumps(res, indent=2))
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="DVC Pipeline Stages for Loan Credit Risk")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    p_prep = subparsers.add_parser("prepare")
    p_prep.add_argument("--params", default="params.yaml")
    p_prep.add_argument("--out", default="results/prepared.csv")

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--params", default="params.yaml")
    p_train.add_argument("--in-csv", default="results/prepared.csv")
    p_train.add_argument("--out-model", default="results/model.joblib")

    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--params", default="params.yaml")
    p_eval.add_argument("--in-csv", default="results/prepared.csv")
    p_eval.add_argument("--in-model", default="results/model.joblib")
    p_eval.add_argument("--out-metrics", default="results/metrics.json")

    args = parser.parse_args()
    if args.stage == "prepare":
        stage_prepare(args.params, args.out)
    elif args.stage == "train":
        stage_train(args.params, args.in_csv, args.out_model)
    elif args.stage == "evaluate":
        stage_evaluate(args.params, args.in_csv, args.in_model, args.out_metrics)


if __name__ == "__main__":
    main()
