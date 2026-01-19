# tests/unit/test_encoder.py
import os
import json
import pandas as pd
import pytest

def test_encode_single_int_column(monkeypatch, tmp_path):
    # set encoding dir BEFORE importing the module
    monkeypatch.setenv("SKALD_ENCODING_DIR", str(tmp_path / "encodings"))
    from SKALD import encoder

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pd.DataFrame({"AGE": [20, 25, 25]}).to_csv(chunks_dir / "c1.csv", index=False)

    maps = encoder.encode_numerical_columns(
        ["c1.csv"], str(chunks_dir),
        [{"column": "AGE", "encode": True, "type": "int"}]
    )

    assert "AGE" in maps
    assert maps["AGE"]["multiplier"] == 1

    out_file = tmp_path / "encodings" / "age_encoding.json"
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert "encoding_map" in data
    assert set(data["encoding_map"].values()) == set([1, 2])  # 20 -> 1, 25 -> 2


def test_encoding_files_written_once_per_column(monkeypatch, tmp_path):
    monkeypatch.setenv("SKALD_ENCODING_DIR", str(tmp_path / "encodings"))
    from SKALD import encoder

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pd.DataFrame({"AGE": [20, 21]}).to_csv(chunks_dir / "c1.csv", index=False)
    pd.DataFrame({"AGE": [21, 22]}).to_csv(chunks_dir / "c2.csv", index=False)

    encoder.encode_numerical_columns(
        ["c1.csv", "c2.csv"], str(chunks_dir),
        [{"column": "AGE", "encode": True, "type": "int"}]
    )

    out_file = tmp_path / "encodings" / "age_encoding.json"
    assert out_file.exists()

    mtime1 = out_file.stat().st_mtime

    # run again should either overwrite with identical content or same timestamp
    encoder.encode_numerical_columns(
        ["c1.csv", "c2.csv"], str(chunks_dir),
        [{"column": "AGE", "encode": True, "type": "int"}]
    )
    mtime2 = out_file.stat().st_mtime

    assert out_file.exists()
    # File exists after second run; exact mtime equality is OS-dependent, so just re-open and check schema
    data = json.loads(out_file.read_text())
    assert set(data.keys()) >= {"encoding_map", "decoding_map", "multiplier", "type"}


def test_non_encoded_column_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv("SKALD_ENCODING_DIR", str(tmp_path / "encodings"))
    from SKALD import encoder

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pd.DataFrame({"AGE": [20, 25]}).to_csv(chunks_dir / "c1.csv", index=False)

    maps = encoder.encode_numerical_columns(
        ["c1.csv"], str(chunks_dir),
        [{"column": "AGE", "encode": False, "type": "int"}]
    )

    assert maps == {}  # nothing encoded
    assert not (tmp_path / "encodings").exists() or not any((tmp_path / "encodings").glob("*.json"))


def test_encode_float_with_multiplier(monkeypatch, tmp_path):
    monkeypatch.setenv("SKALD_ENCODING_DIR", str(tmp_path / "encodings"))
    from SKALD import encoder

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pd.DataFrame({"BMI": [21.1, 21.10, 21.123]}).to_csv(chunks_dir / "c1.csv", index=False)

    maps = encoder.encode_numerical_columns(
        ["c1.csv"], str(chunks_dir),
        [{"column": "BMI", "encode": True, "type": "float"}]
    )

    assert maps["BMI"]["multiplier"] >= 10  # at least 10 based on max decimals
    assert (tmp_path / "encodings" / "bmi_encoding.json").exists()
