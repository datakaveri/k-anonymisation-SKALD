import os
import json
import pandas as pd
import pytest

from SKALD.preprocess import (
    suppress,
    pseudonymize,
    encrypt_columns
)

# ---------- helpers ----------
def base_df():
    return pd.DataFrame(
        {
            "FULLNAMEENGLISH": ["Alice", "Bob"],
            "HEAD_MEMBERNAME": ["Carol", "Dave"],
            "MEMEBERSLNO": [1, 2],
            "MEMBERID": [100, 200],
            "UIDREFERENCENUMBER": ["X1", "Y2"],
            "HEAD_MEMBERID": ["H1", "H2"],
            "Other": ["foo", "bar"],
        }
    )


# ---------- suppress ----------
def test_suppress_drops_columns_and_ignores_missing():
    df = base_df()
    out = suppress(df.copy(), ["FULLNAMEENGLISH", "HEAD_MEMBERNAME", "NOT_PRESENT"])
    assert "FULLNAMEENGLISH" not in out.columns
    assert "HEAD_MEMBERNAME" not in out.columns
    # rest untouched
    for c in ["MEMEBERSLNO", "MEMBERID", "UIDREFERENCENUMBER", "HEAD_MEMBERID", "Other"]:
        assert c in out.columns


# ---------- pseudonymize ----------
def test_pseudonymize_hashes_and_drops_sources():
    df = base_df()
    out = pseudonymize(df.copy(), ["FULLNAMEENGLISH", "HEAD_MEMBERNAME"])

    # sources dropped
    assert "FULLNAMEENGLISH" not in out.columns
    assert "HEAD_MEMBERNAME" not in out.columns

    # outputs present
    assert "UID" not in out.columns
    assert "Hashed Value" in out.columns
    assert out["Hashed Value"].dtype == object

    # two unique combinations -> two unique hashes
    assert out["Hashed Value"].nunique() == 2

    # deterministic hashing for same inputs
    out2 = pseudonymize(df.copy(), ["FULLNAMEENGLISH", "HEAD_MEMBERNAME"])
    assert list(out["Hashed Value"]) == list(out2["Hashed Value"])


def test_pseudonymize_with_insufficient_columns_is_noop():
    df = base_df()
    out = pseudonymize(df.copy(), ["FULLNAMEENGLISH"])  # only one column -> skip
    # unchanged structure
    assert set(out.columns) == set(df.columns)
    assert "Hashed Value" not in out.columns


# ---------- encrypt ----------
def test_encrypt_columns_generates_and_reuses_keys(tmp_path):
    df = base_df()
    key_dir = tmp_path / "keys"

    cols = ["MEMEBERSLNO", "MEMBERID"]

    # first run -> creates keys, encrypts
    enc1 = encrypt_columns(df.copy(), cols, key_dir=str(key_dir))
    key_file = key_dir / "column_keys.json"
    assert key_file.exists()

    with key_file.open() as f:
        keys1 = json.load(f)

    # values should be encrypted (not equal to originals)
    for col in cols:
        assert all(isinstance(v, str) for v in enc1[col])  # ciphertext strings
        assert str(df[col].iloc[0]) != enc1[col].iloc[0]

    # second run -> reuse same keys (not necessarily same ciphertext due to Fernet nonce)
    enc2 = encrypt_columns(df.copy(), cols, key_dir=str(key_dir))
    with key_file.open() as f:
        keys2 = json.load(f)
    assert keys1 == keys2  # key reuse verified


def test_encrypt_columns_skips_missing_and_warns(tmp_path, capsys):
    df = base_df()
    key_dir = tmp_path / "keys"

    enc = encrypt_columns(df.copy(), ["MEMBERID", "NOT_EXIST"], key_dir=str(key_dir))
    captured = capsys.readouterr().out
    assert "[WARN] Column 'NOT_EXIST' not found in dataset — skipping." in captured
    assert "MEMBERID" in enc.columns  # still processed
    # keys file exists for columns that were encrypted
    assert (key_dir / "column_keys.json").exists()


def test_encrypt_handles_nan_without_crashing(tmp_path):
    df = pd.DataFrame(
        {
            "MEMBERID": [100, None, 300],
            "Other": ["a", "b", "c"],
        }
    )
    key_dir = tmp_path / "keys"

    # Should not raise; NaN becomes "nan" string and remains unencrypted by design
    out = encrypt_columns(df.copy(), ["MEMBERID"], key_dir=str(key_dir))
    # at least one literal "nan" string should be present
    assert "nan" in set(out["MEMBERID"].astype(str))
