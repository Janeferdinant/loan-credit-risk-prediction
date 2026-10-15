"""Synthetic stand-in for ``german.data`` used only by the tests.

It reproduces the raw file format (space separated symbolic codes, last column is the
class 1 = good / 2 = bad) so the parser, the pipelines and the CLI can be exercised
without any network access. It is *not* the real data and carries no real signal beyond
a few hand-made effects.
"""

from __future__ import annotations

import numpy as np

CATS = {
    1: ["A11", "A12", "A13", "A14"],
    3: ["A30", "A31", "A32", "A33", "A34"],
    4: ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"],
    6: ["A61", "A62", "A63", "A64", "A65"],
    7: ["A71", "A72", "A73", "A74", "A75"],
    9: ["A91", "A92", "A93", "A94"],
    10: ["A101", "A102", "A103"],
    12: ["A121", "A122", "A123", "A124"],
    14: ["A141", "A142", "A143"],
    15: ["A151", "A152", "A153"],
    17: ["A171", "A172", "A173", "A174"],
    19: ["A191", "A192"],
    20: ["A201", "A202"],
}


def german_like_rows(n: int = 300, seed: int = 0) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        row: dict[int, str] = {}
        for k, codes in CATS.items():
            row[k] = str(rng.choice(codes))
        duration = int(rng.integers(4, 61))
        amount = int(np.exp(rng.normal(7.8, 0.7)))
        age = int(rng.integers(19, 75))
        score = (
            0.9 * (row[1] == "A11")
            - 0.8 * (row[1] == "A14")
            + 0.03 * duration
            + 0.0002 * amount
            - 0.01 * age
            - 1.6
        )
        bad = rng.random() < 1 / (1 + np.exp(-2.5 * (score + 0.4)))
        row[2] = str(duration)
        row[5] = str(amount)
        row[8] = str(int(rng.integers(1, 5)))
        row[11] = str(int(rng.integers(1, 5)))
        row[13] = str(age)
        row[16] = str(int(rng.integers(1, 4)))
        row[18] = str(int(rng.integers(1, 3)))
        rows.append([row[i] for i in range(1, 21)] + ["2" if bad else "1"])
    return rows


def german_like_text(n: int = 300, seed: int = 0) -> str:
    return "\n".join(" ".join(r) for r in german_like_rows(n, seed)) + "\n"
