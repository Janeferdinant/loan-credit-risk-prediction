"""Download, verify and parse the Statlog (German Credit Data) set from the UCI repository.

Source: Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository,
https://doi.org/10.24432/C5NC77, licensed CC BY 4.0. The archive is fetched at run time and
checked against a pinned SHA-256; the raw data is never committed.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
UCI_SHA256 = "e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b"
ARCHIVE_NAME = "statlog_german_credit.zip"
DATA_NAME = "german.data"

TARGET = "bad"  # 1 = bad credit risk (class 2 in the raw file), 0 = good

NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "dependents",
]
CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings",
    "employment_since",
    "personal_status",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]
# Raw file order (attributes 1..20)
RAW_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment_since",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
]
ALL_FEATURES = RAW_COLUMNS
SENSITIVE_FEATURES = ["personal_status", "age", "foreign_worker"]

# Code book transcribed from german.doc in the UCI archive.
CODES: dict[str, dict[str, str]] = {
    "checking_status": {
        "A11": "<0 DM",
        "A12": "0-200 DM",
        "A13": ">=200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings": {
        "A61": "<100 DM",
        "A62": "100-500 DM",
        "A63": "500-1000 DM",
        "A64": ">=1000 DM",
        "A65": "unknown / no savings",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "<1 year",
        "A73": "1-4 years",
        "A74": "4-7 years",
        "A75": ">=7 years",
    },
    "personal_status": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed / unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


class ChecksumError(RuntimeError):
    """Raised when a downloaded file does not match its pinned SHA-256."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_data_dir() -> Path:
    return Path(os.environ.get("CREDIT_RISK_DATA_DIR", "data"))


def _urlopen_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "credit-risk-study/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        return resp.read()


def download_german_credit(
    dest_dir: Path | str | None = None,
    url: str = UCI_URL,
    sha256: str = UCI_SHA256,
    opener: Callable[[str], bytes] | None = None,
) -> Path:
    """Fetch the UCI archive (if not cached), verify its SHA-256 and extract ``german.data``."""
    dest = Path(dest_dir) if dest_dir is not None else default_data_dir()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ARCHIVE_NAME
    if archive.exists() and sha256_file(archive) != sha256:
        archive.unlink()  # corrupt or stale cache: fetch again
    if not archive.exists():
        payload = (opener or _urlopen_bytes)(url)
        got = sha256_bytes(payload)
        if got != sha256:
            raise ChecksumError(f"SHA-256 mismatch for {url}: expected {sha256}, got {got}")
        archive.write_bytes(payload)
    target = dest / DATA_NAME
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as z:
        target.write_bytes(z.read(DATA_NAME))
    return target


def parse_german_data(text: str) -> pd.DataFrame:
    """Parse raw ``german.data`` text into a DataFrame with readable labels and a 0/1 target."""
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 21:
            raise ValueError(f"line {lineno}: expected 21 fields, found {len(parts)}")
        rows.append(parts)
    raw = pd.DataFrame(rows, columns=[*RAW_COLUMNS, "class"])
    out = pd.DataFrame(index=raw.index)
    for col in RAW_COLUMNS:
        if col in CODES:
            bad = sorted(set(raw[col]) - set(CODES[col]))
            if bad:
                raise ValueError(f"unknown code(s) {bad} in column {col}")
            out[col] = raw[col].map(CODES[col])
        else:
            out[col] = pd.to_numeric(raw[col]).astype(int)
    out[TARGET] = (pd.to_numeric(raw["class"]) == 2).astype(int)
    return out


def load_german_credit(
    data_file: Path | str | None = None, data_dir: Path | str | None = None
) -> pd.DataFrame:
    """Load from a local ``german.data`` if given, otherwise download (verified) and parse."""
    path = Path(data_file) if data_file else download_german_credit(data_dir)
    return parse_german_data(path.read_text(encoding="latin-1"))


def add_sensitive(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived sensitive columns used for the fairness check.

    ``sex`` comes from the combined personal-status/sex attribute; ``age_group`` splits at 25
    (applicants under 25 vs 25 and over, a common convention for this dataset).
    """
    out = df.copy()
    out["sex"] = out["personal_status"].str.startswith("female").map({True: "female", False: "male"})
    out["age_group"] = (out["age"] < 25).map({True: "<25", False: ">=25"})
    return out


def select_features(feature_set: str = "all") -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names for a named feature set."""
    if feature_set == "all":
        drop: set[str] = set()
    elif feature_set == "no_sensitive":
        drop = set(SENSITIVE_FEATURES)
    else:
        raise ValueError(f"unknown feature set {feature_set!r}; use 'all' or 'no_sensitive'")
    return (
        [c for c in NUMERIC_FEATURES if c not in drop],
        [c for c in CATEGORICAL_FEATURES if c not in drop],
    )
