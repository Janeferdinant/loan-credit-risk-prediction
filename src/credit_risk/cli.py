"""Command line interface: ``credit-risk train | evaluate | predict | sweep``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from credit_risk import metrics, runinfo
from credit_risk.data import ALL_FEATURES, CODES, NUMERIC_FEATURES
from credit_risk.models import MODEL_NAMES
from credit_risk.pipeline import Config, run_evaluate, run_sweep, run_train

DISCLAIMER = (
    "NOTE: research demo trained on 1,000 German credit records from 1994. Not for real lending decisions."
)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, default=Path("."), help="project root for artifacts/ results/ docs/")
    p.add_argument("--data-file", type=Path, default=None, help="local german.data (skips the download)")
    p.add_argument("--data-dir", type=Path, default=None, help="cache directory for the verified download")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)


def _tuning(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trials", type=int, default=20, help="Optuna trials per model and fold")
    p.add_argument("--splits", type=int, default=5, help="outer CV folds")
    p.add_argument("--repeats", type=int, default=3, help="outer CV repeats")
    p.add_argument("--inner-splits", type=int, default=3, help="inner CV folds used by Optuna")
    p.add_argument("--jobs", type=int, default=4, help="parallel outer folds")
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="credit-risk", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="nested CV, tuning and final fits")
    _common(t)
    _tuning(t)
    t.add_argument("--bootstrap", type=int, default=2000)

    e = sub.add_parser("evaluate", help="test metrics, thresholds, fairness, SHAP, figures")
    _common(e)
    e.add_argument("--bootstrap", type=int, default=2000)
    e.add_argument("--jobs", type=int, default=4, help="parallel outer folds for the ablation")
    e.add_argument("--no-ablation", action="store_true", help="skip the no-sensitive-features run")

    s = sub.add_parser("sweep", help="repeat tuning and test evaluation over several seeds")
    _common(s)
    _tuning(s)
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    pr = sub.add_parser("predict", help="score applicants from a CSV with the trained model")
    pr.add_argument("--out", type=Path, default=Path("."))
    pr.add_argument("--input", type=Path, required=True, help="CSV with the 20 raw feature columns")
    pr.add_argument("--output", type=Path, default=None, help="write predictions here (default: stdout)")
    pr.add_argument("--model", default=None, help="variant key such as xgb|sigmoid (default: selected)")
    pr.add_argument("--threshold", type=float, default=None, help="approve if P(bad) < threshold")
    return ap


def _config(a: argparse.Namespace) -> Config:
    cfg = Config(out=a.out, data_file=a.data_file, data_dir=a.data_dir, seed=a.seed, test_size=a.test_size)
    for name in ("trials", "splits", "repeats", "inner_splits", "jobs", "models", "bootstrap"):
        if hasattr(a, name):
            setattr(cfg, name, getattr(a, name))
    return cfg


def _predict(a: argparse.Namespace) -> None:
    arts = a.out / "artifacts"
    meta = runinfo.load_json(arts / "meta.json")
    models = joblib.load(arts / "models.joblib")
    summary = runinfo.load_json(a.out / "results" / "summary.json")
    key = a.model or meta["selected"]
    if key not in models:
        raise SystemExit(f"unknown model {key!r}; available: {sorted(models)}")
    df = pd.read_csv(a.input)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"input is missing columns: {missing}")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col, mapping in CODES.items():
        unknown = sorted(set(df[col].astype(str)) - set(mapping.values()))
        if unknown:
            print(f"warning: unseen values in {col}: {unknown} (treated as unknown)", file=sys.stderr)
    X = df[meta["numeric"] + meta["categorical"]]
    p = models[key].predict_proba(X)[:, 1]
    threshold = (
        a.threshold if a.threshold is not None else summary["test"][_index(summary, key)]["oof_threshold"]
    )
    out = df.copy()
    out["p_bad"] = p.round(4)
    out["decision"] = ["approve" if d else "reject" for d in metrics.decide_approve(p, threshold)]
    out["threshold"] = round(threshold, 4)
    out["model"] = key
    if a.output:
        out.to_csv(a.output, index=False)
        print(f"wrote {len(out)} predictions to {a.output}")
    else:
        print(out[["p_bad", "decision", "threshold", "model"]].to_string(index=False))
    print(DISCLAIMER, file=sys.stderr)


def _index(summary: dict, key: str) -> int:
    for i, r in enumerate(summary["test"]):
        if r["key"] == key:
            return i
    raise SystemExit(f"model {key!r} not in results/summary.json")


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if a.command == "predict":
        _predict(a)
        return
    cfg = _config(a)
    if a.command == "train":
        r = run_train(cfg)
        print(f"selected variant: {r['selected']}  ({r['seconds']} s)")
    elif a.command == "evaluate":
        s = run_evaluate(cfg, ablation=not a.no_ablation)
        print(f"selected: {s['selected']}; thresholds {s['threshold']}")
    elif a.command == "sweep":
        out = run_sweep(cfg, a.seeds)
        print(f"sweep over seeds {a.seeds} done in {out['seconds']} s")


if __name__ == "__main__":
    main()
