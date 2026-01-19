import os
import json
import yaml
import pytest
import pandas as pd

from unittest.mock import patch, MagicMock

from SKALD.core import run_pipeline
from SKALD.preprocess import suppress, pseudonymize, encrypt_columns
from SKALD.chunking import split_csv_by_ram
from SKALD.encoder import encode_numerical_columns
from SKALD.build_QI import build_quasi_identifiers
from SKALD.chunk_processing import process_chunks_for_histograms        
from SKALD.config_validation import Config
from SKALD.quasi_identifier import QuasiIdentifier


@pytest.fixture
def minimal_yaml(tmp_path):
    """
    Creates a minimal valid config.yaml file for testing.
    """
    cfg = {
        "enable_k_anonymity": True,
        "output_path": "gen_chunk1.csv",
        "output_directory": str(tmp_path / "output"),
        "key_directory": str(tmp_path / "keys"),
        "log_file": "log.txt",
        "suppress": [],
        "pseudonymize": [],
        "encrypt": [],
        "quasi_identifiers": {
            "numerical": [{"column": "AGE", "encode": False, "type": "int"}],
            "categorical": [{"column": "GENDER"}]
        },
        "k": 2,
        "l": 1,
        "sensitive_parameter": "STATE",
        "bin_width_multiplication_factor": {"AGE": 2},
        "hardcoded_min_max": {"AGE": [20, 40]},
        "suppression_limit": 0,
        "number_of_chunks": 1,
    }

    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


@pytest.fixture
def mock_chunks(tmp_path):
    """
    Creates a fake chunks directory with one CSV.
    """
    chunks = tmp_path / "chunks"
    chunks.mkdir()

    df = pd.DataFrame({
        "AGE": [20, 30],
        "GENDER": ["Male", "Female"],
        "STATE": ["KA", "KA"]
    })
    df.to_csv(chunks / "chunk1.csv", index=False)

    return str(chunks)


# ============================================================
# TEST 1: Missing chunk directory
# ============================================================
def test_missing_chunk_dir(minimal_yaml):
    with patch("SKALD.core.os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            run_pipeline(config_path=minimal_yaml)


# ============================================================
# TEST 2: No CSV files in chunk directory
# ============================================================
def test_empty_chunk_dir(minimal_yaml, mock_chunks):
    empty_dir = mock_chunks
    # delete CSVs
    for f in os.listdir(empty_dir):
        os.remove(os.path.join(empty_dir, f))

    with patch("SKALD.core.os.listdir", return_value=[]):
        with pytest.raises(ValueError):
            run_pipeline(config_path=minimal_yaml)


# ============================================================
# TEST 3: Full pipeline with mocks
# ============================================================
def test_full_pipeline(monkeypatch, minimal_yaml, mock_chunks, tmp_path):
    from SKALD.config_validation import Config
    import shutil

    # Switch CWD so core.py uses tmp_path
    monkeypatch.chdir(tmp_path)

    # mock_chunks fixture already created tmp_path/chunks
    chunks_dir = tmp_path / "chunks"

    # Create data/source.csv
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    src_file = data_dir / "source.csv"
    src_file.write_text("AGE,GENDER,STATE\n20,Male,KA\n")

    # Mock load_config
    monkeypatch.setattr(
        "SKALD.core.load_config",
        lambda _: Config(**yaml.safe_load(open(minimal_yaml)))
    )

    # Fake split_csv_by_ram → write into existing chunks_dir
    def fake_split_csv_by_ram(_unused):
        shutil.copy(src_file, chunks_dir / "chunk1.csv")

    monkeypatch.setattr("SKALD.core.split_csv_by_ram", fake_split_csv_by_ram)

    # Mock filesystem lookups
    monkeypatch.setattr("SKALD.core.os.listdir", lambda p: ["chunk1.csv"])
    monkeypatch.setattr("SKALD.core.os.path.exists", lambda p: True)
    monkeypatch.setattr("SKALD.core.os.path.getsize", lambda p: 10)

    # Mock encoder
    monkeypatch.setattr(
        "SKALD.core.encode_numerical_columns",
        lambda *a, **k: {"AGE": {"encoding_map": {}, "multiplier": 1}}
    )

    # Mock QIs
    mock_qis = [
    QuasiIdentifier("AGE", is_categorical=False, min_value=20, max_value=40),
    QuasiIdentifier("GENDER", is_categorical=True)
]

    monkeypatch.setattr("SKALD.core.build_quasi_identifiers",
                        lambda *a: (mock_qis, ["AGE", "GENDER"]))

    # Mock OLA_2
    mock_ola2 = MagicMock()
    mock_ola2.merge_histograms.return_value = [[1]]
    mock_ola2.get_final_binwidths.return_value = [1]
    mock_ola2.get_equivalence_class_stats.return_value = {1: 1}
    mock_ola2.get_suppressed_percent.return_value = 0
    mock_ola2.lowest_dm_star = 10
    mock_ola2.best_num_eq_classes = 1

    core_mod = __import__("SKALD.core", fromlist=["run_pipeline"])
    monkeypatch.setattr(core_mod, "OLA_2", lambda *a, **k: mock_ola2)

    # Mock histogram and generalizer
    monkeypatch.setattr("SKALD.core.process_chunks_for_histograms", lambda *a, **k: [[[1]]])
    monkeypatch.setattr("SKALD.core.generalize_first_chunk", lambda *a, **k: None)

    rf, elapsed, dm, eq, stats = run_pipeline(config_path=minimal_yaml)

    assert rf == [1]
    assert dm == 10
    assert eq == 1
    assert stats == {1: 1}


# ============================================================
# TEST 4: Suppression/pseudonymization/encryption-only mode
# ============================================================
def test_preprocess_only(minimal_yaml, monkeypatch, mock_chunks, tmp_path):
    from SKALD.config_validation import Config
    import shutil
    monkeypatch.chdir(tmp_path)

    cfg = yaml.safe_load(open(minimal_yaml))
    cfg["enable_k_anonymity"] = False
    with open(minimal_yaml, "w") as f:
        yaml.dump(cfg, f)

    # Fake data directory + source file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "source.csv").write_text("AGE,GENDER,STATE\n20,Male,KA\n")

    # Fake chunk generator
    def fake_split_csv_by_ram(_):
        shutil.copy(data_dir / "source.csv", tmp_path / "chunks" / "chunk1.csv")

    monkeypatch.setattr("SKALD.core.load_config", lambda _: Config(**cfg))
    monkeypatch.setattr("SKALD.core.split_csv_by_ram", fake_split_csv_by_ram)

    monkeypatch.setattr("SKALD.core.os.listdir", lambda p: ["chunk1.csv"])
    monkeypatch.setattr("SKALD.core.os.path.exists", lambda p: True)
    monkeypatch.setattr("SKALD.core.os.path.getsize", lambda p: 10)

    # Because core imports these inside run_pipeline, patch REAL preprocess module
    monkeypatch.setattr("SKALD.preprocess.suppress", lambda df, cols: df)
    monkeypatch.setattr("SKALD.preprocess.pseudonymize", lambda df, cols: df)
    monkeypatch.setattr("SKALD.preprocess.encrypt_columns", lambda df, cols: df)

    out, _, _, _, _ = run_pipeline(config_path=minimal_yaml)
    assert isinstance(out, str)

# ============================================================
# TEST 5: _entry_main function
# ============================================================
def test_entry_main(monkeypatch, tmp_path):
    import json
    from SKALD.core import _entry_main

    # create fake config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "test.json"

    config_data = {
        "operations": ["chunking"],
        "data_type": "medical",
        "medical": {
            "enable_k_anonymity": True,
            "suppress": ["A"],
            "pseudonymize": ["B"],
            "encrypt": ["C"],
            "quasi_identifiers": {
                "numerical": [{"column": "AGE", "encode": False, "type": "int"}],
                "categorical": [{"column": "GENDER"}]
            },
            "k_anonymize": {"k": 2},
            "l_diversity": {"l": 1},
            "sensitive_parameter": "STATE",
            "bin_width_multiplication_factor": {"AGE": 2},
            "hardcoded_min_max": {"AGE": [20, 50]},
            "suppression_limit": 0
        }
    }

    config_file.write_text(json.dumps(config_data))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("SKALD.core.os.listdir", lambda p: ["test.json"])



    # mock run_pipeline so we don't execute it
    called = {}

    def fake_run_pipeline(config_path):
        called["config_path"] = config_path
        return "OK"

    monkeypatch.setattr("SKALD.core.run_pipeline", fake_run_pipeline)

    result = _entry_main()

    # verify pipeline was called
    assert result == "OK"

    # verify a YAML file was generated
    assert "config_path" in called
    assert os.path.exists(called["config_path"])

    yaml_data = yaml.safe_load(open(called["config_path"]))

    assert yaml_data["k"] == 2
    assert yaml_data["l"] == 1
    assert yaml_data["suppress"] == ["A"]
    assert yaml_data["sensitive_parameter"] == "STATE"


