"""Environment description and JSON helpers shared by the run logs."""

from __future__ import annotations

import json
import os
import platform
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "optuna", "shap", "matplotlib")


def environment() -> dict[str, Any]:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return {
        "date": date.today().isoformat(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": "GPU: none (CPU only)",
        "packages": versions,
    }


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _clean(o: Any) -> Any:
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def dump_json(obj: Any, path: Path, indent: int | None = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), default=_default, indent=indent), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
