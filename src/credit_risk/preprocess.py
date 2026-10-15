"""Preprocessing pipelines. Every step is fitted inside a sklearn Pipeline, so in any
cross-validation or hold-out the statistics come from the training rows only."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

LOG_COLUMNS = ("credit_amount",)


class LogColumns(BaseEstimator, TransformerMixin):
    """log1p on selected columns (by name, resolved against the numeric column list)."""

    def __init__(self, columns: tuple[str, ...] = (), numeric: tuple[str, ...] = ()):
        self.columns = columns
        self.numeric = numeric

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        for c in self.columns:
            if c in self.numeric:
                j = list(self.numeric).index(c)
                X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
        return X


def build_preprocessor(kind: str, numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer.

    kind: ``linear`` (impute, log1p amount, standardise, one-hot), ``onehot`` (impute, one-hot, for
    XGBoost) or ``ordinal`` (impute, ordinal codes with unseen categories mapped to NaN, for
    HistGradientBoosting's native categorical support).
    """
    if kind == "linear":
        num = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("log", LogColumns(LOG_COLUMNS, tuple(numeric))),
                ("scale", StandardScaler()),
            ]
        )
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    elif kind == "onehot":
        num = Pipeline([("impute", SimpleImputer(strategy="median"))])
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    elif kind == "ordinal":
        num = Pipeline([("impute", SimpleImputer(strategy="median"))])
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                ),
            ]
        )
    else:
        raise ValueError(f"unknown preprocessor kind {kind!r}")
    return ColumnTransformer(
        [("num", num, list(numeric)), ("cat", cat, list(categorical))],
        verbose_feature_names_out=True,
    )
