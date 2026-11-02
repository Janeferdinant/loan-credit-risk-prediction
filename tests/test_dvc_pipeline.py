from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from credit_risk.dvc_stages import stage_evaluate, stage_train


def test_dvc_yaml_structure():
    """Verify dvc.yaml and params.yaml define the full 3-stage reproducible pipeline."""
    root = Path(__file__).resolve().parent.parent
    dvc_yaml_path = root / "dvc.yaml"
    params_yaml_path = root / "params.yaml"

    assert dvc_yaml_path.exists(), "dvc.yaml must exist"
    assert params_yaml_path.exists(), "params.yaml must exist"

    with open(dvc_yaml_path, encoding="utf-8") as f:
        dvc_cfg = yaml.safe_load(f)
    with open(params_yaml_path, encoding="utf-8") as f:
        params_cfg = yaml.safe_load(f)

    stages = dvc_cfg.get("stages", {})
    assert "prepare" in stages
    assert "train" in stages
    assert "evaluate" in stages

    # Check stage dependencies and outputs
    assert any("prepared.csv" in out for out in stages["prepare"]["outs"])
    assert any("model.joblib" in out for out in stages["train"]["outs"])
    assert "results/metrics.json" in stages["evaluate"]["metrics"][0]

    # Check params are declared
    assert "prepare" in params_cfg
    assert "train" in params_cfg
    assert "evaluate" in params_cfg


def test_params_drive_training(tmp_path: Path, df: pd.DataFrame):
    """Verify that changing hyperparameters in params.yaml changes the trained model output."""
    prep_csv = tmp_path / "prepared.csv"
    df.to_csv(prep_csv, index=False)

    params_a = {
        "prepare": {"seed": 42, "test_size": 0.2},
        "train": {
            "model_name": "xgb",
            "seed": 42,
            "learning_rate": 0.01,
            "max_depth": 2,
            "n_estimators": 10,
        },
        "evaluate": {
            "cost_false_approval": 5.0,
            "cost_false_rejection": 1.0,
            "operating_threshold": 0.14,
        },
    }
    params_b = {
        "prepare": {"seed": 42, "test_size": 0.2},
        "train": {
            "model_name": "xgb",
            "seed": 42,
            "learning_rate": 0.30,
            "max_depth": 6,
            "n_estimators": 100,
        },
        "evaluate": {
            "cost_false_approval": 5.0,
            "cost_false_rejection": 1.0,
            "operating_threshold": 0.14,
        },
    }

    yaml_a = tmp_path / "params_a.yaml"
    yaml_b = tmp_path / "params_b.yaml"
    with open(yaml_a, "w", encoding="utf-8") as f:
        yaml.safe_dump(params_a, f)
    with open(yaml_b, "w", encoding="utf-8") as f:
        yaml.safe_dump(params_b, f)

    model_a = tmp_path / "model_a.joblib"
    model_b = tmp_path / "model_b.joblib"

    stage_train(yaml_a, prep_csv, model_a)
    stage_train(yaml_b, prep_csv, model_b)

    hash_a = hashlib.sha256(model_a.read_bytes()).hexdigest()
    hash_b = hashlib.sha256(model_b.read_bytes()).hexdigest()
    assert hash_a != hash_b, "Changing training parameters must alter model artifact hash"

    # Verify metrics evaluate
    metrics_a = tmp_path / "metrics_a.json"
    metrics_b = tmp_path / "metrics_b.json"
    stage_evaluate(yaml_a, prep_csv, model_a, metrics_a)
    stage_evaluate(yaml_b, prep_csv, model_b, metrics_b)

    res_a = json.loads(metrics_a.read_text(encoding="utf-8"))
    res_b = json.loads(metrics_b.read_text(encoding="utf-8"))

    assert "auc" in res_a and "auc" in res_b
    assert res_a["auc"] > 0.5
    assert res_b["auc"] > 0.5
    # The models should produce different predictions
    assert res_a["logloss"] != res_b["logloss"]
