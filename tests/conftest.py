from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import german_like_text  # noqa: E402

from credit_risk.data import parse_german_data  # noqa: E402


@pytest.fixture(scope="session")
def raw_text() -> str:
    return german_like_text(400, seed=1)


@pytest.fixture(scope="session")
def df(raw_text):
    return parse_german_data(raw_text)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs excluded from CI")
