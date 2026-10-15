"""Stratified hold-out and repeated outer cross-validation splits."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


def holdout_split(y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test row positions."""
    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    return np.sort(train), np.sort(test)


def outer_folds(
    y: np.ndarray, n_splits: int, n_repeats: int, seed: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Repeated stratified K-fold as (repeat, train_positions, test_positions)."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    X = np.zeros((len(y), 1))
    return [(i // n_splits, tr, te) for i, (tr, te) in enumerate(cv.split(X, y))]
