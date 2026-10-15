"""Markdown run logs written next to the results (docs/runs/)."""

from __future__ import annotations

from typing import Any

from credit_risk.runinfo import environment


def _fmt(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _ci(d: dict[str, float], nd: int = 3) -> str:
    return f"{d['value']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def _env_block(env: dict[str, Any]) -> list[str]:
    pk = ", ".join(f"{k} {v}" for k, v in env["packages"].items())
    return [
        f"- Date: {env['date']}",
        f"- Hardware: {env['cpu']}, {env['cpu_count']} logical CPUs, {env['gpu']}; {env['os']}",
        f"- Python {env['python']}; {pk}",
    ]


def write_run_log(cfg, train_info, summary, fairness, ablation, seconds_eval) -> None:
    env = environment()
    lines = [
        f"# Main run, {env['date']}",
        "",
        "Produced by `credit-risk train` followed by `credit-risk evaluate` "
        "(commands and flags below). Every number in the README and the explorer is read from the "
        "JSON files this run wrote to `results/`.",
        "",
        "## Setup",
        *_env_block(train_info["env"]),
        f"- Seed: {summary['seed']} (hold-out split, tuning samplers, model seeds, bootstrap). "
        "Per-fold tuning seeds are derived as `seed + 1000*repeat + fold`.",
        f"- Hold-out: {summary['n_train']} train / {summary['n_test']} test rows, stratified; "
        f"bad rate {summary['bad_rate_train']:.3f} train, {summary['bad_rate_test']:.3f} test.",
        f"- Nested CV: {cfg.repeats} repeats x {cfg.splits} outer folds, {cfg.inner_splits} inner folds, "
        f"{cfg.trials} Optuna TPE trials per model, objective inner-CV log loss.",
        f"- Bootstrap: {summary['bootstrap_resamples']} stratified resamples of the test split.",
        f"- Wall clock: nested CV {train_info['seconds_nested_cv']} s, final fit "
        f"{train_info['seconds_final_fit']} s, evaluate {seconds_eval} s.",
        "",
        "```",
        f"credit-risk train --seed {summary['seed']} --trials {cfg.trials} --splits {cfg.splits} "
        f"--repeats {cfg.repeats} --inner-splits {cfg.inner_splits}",
        f"credit-risk evaluate --seed {summary['seed']} --bootstrap {cfg.bootstrap}",
        "```",
        "",
        f"Selected variant (rule: {summary['selection_rule']}): `{summary['selected']}`.",
        "",
        "## Nested cross-validation (training rows only, mean over outer folds)",
        "",
        "| Model | AUC | log loss | Brier | ECE | cost @ 1/6 |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["nested_cv"]:
        c = r["cost_at_theoretical_threshold"]
        lines.append(
            f"| {r['label']} | {r['auc']['mean']:.3f} +/- {r['auc']['sd']:.3f} | "
            f"{r['logloss']['mean']:.3f} | {r['brier']['mean']:.3f} | {r['ece']['mean']:.3f} | "
            f"{c['mean']:.3f} |"
        )
    lines += [
        "",
        "## Hold-out test split (95% stratified bootstrap intervals)",
        "",
        "| Model | AUC | PR-AUC | log loss | Brier | ECE | cost @ 1/6 | cost @ OOF thr. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["test"]:
        lines.append(
            f"| {r['label']} | {_ci(r['auc'])} | {_ci(r['pr_auc'])} | {_ci(r['logloss'])} | "
            f"{_ci(r['brier'])} | {_ci(r['ece'])} | {_ci(r['cost_at_theoretical'])} | "
            f"{_ci(r['cost_at_oof_threshold'])} (t={r['oof_threshold']:.3f}) |"
        )
    t = summary["threshold_analysis"]
    lines += [
        "",
        "## Cost-sensitive threshold (cost matrix 5 for a false approval, 1 for a false rejection)",
        "",
        f"- Theoretical threshold for calibrated probabilities: 1/6 = {t['theoretical']:.4f}.",
        f"- Threshold minimising out-of-fold cost on the training rows: {t['oof_optimal']:.3f}.",
        f"- Test cost per applicant at that threshold: {_ci(t['test_cost_at_operating_threshold'])}",
        f"- At the theoretical threshold: {_ci(t['test_cost_at_theoretical'])}",
        f"- At the default 0.5: {_ci(t['test_cost_at_0.5'])}",
        f"- Approve everyone: {t['test_cost_approve_all']:.3f}; "
        f"reject everyone: {t['test_cost_reject_all']:.3f}",
        f"- Policy minus reject-all, paired bootstrap: {t['diff_vs_reject_all']['est']:+.3f} "
        f"[{t['diff_vs_reject_all']['lo']:+.3f}, {t['diff_vs_reject_all']['hi']:+.3f}]",
        f"- Policy minus approve-all: {t['diff_vs_approve_all']['est']:+.3f} "
        f"[{t['diff_vs_approve_all']['lo']:+.3f}, {t['diff_vs_approve_all']['hi']:+.3f}]",
        "",
        "## Paired AUC differences on the test split",
        "",
    ]
    for k, v in summary["paired_auc_differences"].items():
        lines.append(f"- {k}: {v['auc_diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    lines += ["", "## Leakage check on real data (permuted training labels)", ""]
    lk = summary["leakage_check"]
    lines += [
        f"- Logistic regression AUC on permuted labels: {lk['auc_logreg_permuted']:.3f}",
        f"- HistGradientBoosting AUC on permuted labels: {lk['auc_hgb_permuted']:.3f}",
        "",
        f"## Fairness (operating threshold {fairness['threshold']:.3f}, model `{fairness['model']}`)",
        "",
    ]
    for scope, name in (
        ("pooled_out_of_fold_train_rows", "Pooled out-of-fold predictions on the training rows"),
        ("test_split", "Test split (small groups, wide intervals)"),
    ):
        lines += [f"### {name}", ""]
        for attr, blk in fairness[scope].items():
            lines += [
                f"**{attr}** (gaps are {blk['group']} minus {blk['reference']})",
                "",
                "| group | n | bad rate | approval | good approved | bad approved | AUC |",
                "|---|---|---|---|---|---|---|",
            ]
            for g in blk["table"]:
                lines.append(
                    f"| {g['group']} | {g['n']} | {g['bad_rate']:.3f} | {g['approval_rate']:.3f} | "
                    f"{_fmt(g['approve_good_rate'])} | {_fmt(g['approve_bad_rate'])} | {_fmt(g['auc'])} |"
                )
            gp = blk["gaps"]
            lines += [
                "",
                "Gaps (95% bootstrap): "
                + "; ".join(
                    f"{k.replace('_gap', '')} {v['est']:+.3f} [{_fmt(v['lo'])}, {_fmt(v['hi'])}]"
                    for k, v in gp.items()
                ),
                "",
            ]
    if ablation:
        lines += [
            "## Ablation: drop personal_status, age and foreign_worker",
            "",
            f"- Out-of-fold AUC: {ablation['auc_oof_all_features']:.3f} with all features, "
            f"{ablation['auc_oof_no_sensitive']:.3f} without.",
            f"- Out-of-fold cost per applicant at the operating threshold: "
            f"{ablation['cost_oof_all_features']:.3f} vs {ablation['cost_oof_no_sensitive']:.3f}.",
            "",
        ]
        for attr, blk in ablation["fairness_no_sensitive"].items():
            gp = blk["gaps"]
            lines.append(
                f"- {attr} gaps without the sensitive columns: "
                + "; ".join(
                    f"{k.replace('_gap', '')} {v['est']:+.3f} [{_fmt(v['lo'])}, {_fmt(v['hi'])}]"
                    for k, v in gp.items()
                )
            )
        lines.append("")
    path = cfg.runs / f"{env['date']}-main-run.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sweep_log(cfg, out: dict[str, Any]) -> None:
    env = out["env"]
    lines = [
        f"# Seed sweep, {env['date']}",
        "",
        "Repeats the whole tune-then-evaluate procedure for several hold-out/tuning seeds "
        "(`credit-risk sweep`), to show how much a single split can move the numbers.",
        "",
        *_env_block(env),
        f"- Seeds: {out['seeds']}; Optuna trials per model: {out['trials']}; wall clock {out['seconds']} s.",
        "",
        "| Model | calibration | AUC mean (sd) | log loss | Brier | ECE | cost @ 1/6 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in out["aggregate"]:
        lines.append(
            f"| {r['model']} | {r['calibration']} | {r['auc_mean']:.3f} ({_fmt(r.get('auc_std'))}) | "
            f"{r['logloss_mean']:.3f} ({_fmt(r.get('logloss_std'))}) | {r['brier_mean']:.3f} | "
            f"{r['ece_mean']:.3f} | {r['cost_at_theoretical_mean']:.3f} "
            f"({_fmt(r.get('cost_at_theoretical_std'))}) |"
        )
    lines += ["", "Per-seed rows are in `results/seed_sweep.json`.", ""]
    path = cfg.runs / f"{env['date']}-seed-sweep.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
