from __future__ import annotations

import io
import zipfile

import pytest

from credit_risk import data


def test_parse_shapes_and_target(df):
    assert len(df) == 400
    assert set(df["bad"].unique()) <= {0, 1}
    for col in data.ALL_FEATURES:
        assert col in df.columns
    assert len(data.ALL_FEATURES) == 20


def test_parse_decodes_known_row():
    # first row of german.data (code A11 ... class 1 = good)
    line = "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1\n"
    row = data.parse_german_data(line).iloc[0]
    assert row["checking_status"] == "<0 DM"
    assert row["duration_months"] == 6
    assert row["credit_amount"] == 1169
    assert row["age"] == 67
    assert row["personal_status"] == "male: single"
    assert row["bad"] == 0


def test_class_two_is_bad():
    line = "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2\n"
    assert data.parse_german_data(line).iloc[0]["bad"] == 1


def test_unknown_code_raises():
    line = "A99 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1\n"
    with pytest.raises(ValueError, match="A99"):
        data.parse_german_data(line)


def test_wrong_column_count_raises():
    with pytest.raises(ValueError, match="21"):
        data.parse_german_data("A11 6 A34\n")


def test_sensitive_attributes(df):
    out = data.add_sensitive(df)
    female_codes = out.loc[out["personal_status"].str.startswith("female"), "sex"]
    assert (female_codes == "female").all()
    male = out.loc[out["personal_status"].str.startswith("male"), "sex"]
    assert (male == "male").all()
    assert (out.loc[out["age"] < 25, "age_group"] == "<25").all()
    assert (out.loc[out["age"] >= 25, "age_group"] == ">=25").all()


def test_age_boundary():
    import pandas as pd

    out = data.add_sensitive(
        pd.DataFrame({"age": [24, 25], "personal_status": ["male: single", "female: single"], "bad": [0, 1]})
    )
    assert list(out["age_group"]) == ["<25", ">=25"]


def test_feature_sets():
    num, cat = data.select_features("all")
    assert len(num) + len(cat) == 20
    num2, cat2 = data.select_features("no_sensitive")
    dropped = set(num + cat) - set(num2 + cat2)
    assert dropped == {"personal_status", "age", "foreign_worker"}
    assert "bad" not in num + cat
    with pytest.raises(ValueError):
        data.select_features("bogus")


def _zip_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("german.data", text)
    return buf.getvalue()


def test_download_verifies_checksum_and_extracts(tmp_path, raw_text):
    payload = _zip_bytes(raw_text)
    calls = []

    def opener(url):
        calls.append(url)
        return payload

    path = data.download_german_credit(
        tmp_path, url="http://x/y.zip", sha256=data.sha256_bytes(payload), opener=opener
    )
    assert path.read_text() == raw_text
    # second call uses the verified cache, no new request
    data.download_german_credit(
        tmp_path, url="http://x/y.zip", sha256=data.sha256_bytes(payload), opener=opener
    )
    assert len(calls) == 1


def test_download_rejects_wrong_checksum(tmp_path, raw_text):
    payload = _zip_bytes(raw_text)
    with pytest.raises(data.ChecksumError):
        data.download_german_credit(tmp_path, url="http://x/y.zip", sha256="0" * 64, opener=lambda u: payload)
    assert not (tmp_path / "german.data").exists()
    assert not (tmp_path / data.ARCHIVE_NAME).exists()


def test_download_replaces_corrupt_cache(tmp_path, raw_text):
    payload = _zip_bytes(raw_text)
    (tmp_path / data.ARCHIVE_NAME).write_bytes(b"corrupt")
    path = data.download_german_credit(
        tmp_path,
        url="http://x/y.zip",
        sha256=data.sha256_bytes(payload),
        opener=lambda u: payload,
    )
    assert path.exists()


def test_pinned_checksum_is_well_formed():
    assert len(data.UCI_SHA256) == 64
    assert data.UCI_URL.startswith("https://archive.ics.uci.edu/")
