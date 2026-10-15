from __future__ import annotations

import json
import subprocess
import sys

import pytest

from credit_risk import cli


def test_cli_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("train", "evaluate", "predict"):
        assert cmd in out


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        cli.main([])


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path):
    from synthetic import german_like_text

    data_file = tmp_path / "german.data"
    data_file.write_text(german_like_text(400, seed=2))
    out = tmp_path / "out"
    common = ["--data-file", str(data_file), "--out", str(out)]
    cli.main(["train", *common, "--trials", "2", "--splits", "3", "--repeats", "1", "--bootstrap", "50"])
    cli.main(["evaluate", *common, "--bootstrap", "50"])
    assert (out / "results" / "nested_cv.json").exists()
    summary = json.loads((out / "results" / "summary.json").read_text())
    assert summary["seed"] == 42
    assert "test" in summary and "threshold" in summary

    sample = tmp_path / "apps.csv"
    sample.write_text(
        "checking_status,duration_months,credit_history,purpose,credit_amount,savings,employment_since,"
        "installment_rate,personal_status,other_debtors,residence_since,property,age,other_installment_plans,"
        "housing,existing_credits,job,dependents,telephone,foreign_worker\n"
        "<0 DM,24,existing credits paid back duly till now,car (new),4000,<100 DM,1-4 years,3,"
        "male: single,none,2,real estate,30,none,own,1,skilled employee / official,1,none,yes\n"
    )
    pred = tmp_path / "pred.csv"
    cli.main(["predict", "--out", str(out), "--input", str(sample), "--output", str(pred)])
    text = pred.read_text()
    assert "p_bad" in text and "decision" in text


@pytest.mark.slow
def test_module_entry_point_runs():
    r = subprocess.run([sys.executable, "-m", "credit_risk", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "train" in r.stdout
