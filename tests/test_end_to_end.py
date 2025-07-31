import os
import pandas as pd
import tempfile
import shutil
import yaml
import pytest
from unittest.mock import patch
from SKALD.core import run_pipeline


def test_run_pipeline_minimal():
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_dir = os.path.join(tmp_dir, "datachunks")
        os.makedirs(chunk_dir)

        data = pd.DataFrame({
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
            "PIN": [560001 + i for i in range(12)],
            "Blood Group": ["A+", "B+", "O+", "A+", "AB+", "B+", "O-", "A-", "B-", "AB-", "O+", "A+"],
        })

        for i in range(2):
            chunk = data.iloc[i * 6:(i + 1) * 6]
            chunk.to_csv(os.path.join(chunk_dir, f"KanonMedicalData_chunk{i + 1}.csv"), index=False)

        config = {
            "number_of_chunks": 2,
            "k": 2,
            "max_number_of_eq_classes": 100,
            "suppression_limit": 0.1,
            "chunk_directory": chunk_dir,
            "output_path": os.path.join(tmp_dir, "generalized_chunk1.csv"),
            "log_file": os.path.join(tmp_dir, "log.txt"),
            "save_output": True,
            "quasi_identifiers": {
                "categorical": [{"column": "Blood Group"}],
                "numerical": [
                    {"column": "PIN", "encode": True, "type": "int"},
                    {"column": "Age", "encode": False, "type": "int"}
                ]
            },
            "hardcoded_min_max": {
                "Age": [25, 60]
            },
            "bin_width_multiplication_factor": {
                "Age": 2,
                "PIN": 2
            }
        }
        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        final_rf, elapsed = run_pipeline(config_path=config_path)
        assert isinstance(final_rf, list)
        assert os.path.exists(os.path.join(tmp_dir, "generalized_chunk1.csv"))


def test_run_pipeline_with_missing_min_max():
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_dir = os.path.join(tmp_dir, "datachunks")
        os.makedirs(chunk_dir)

        data = pd.DataFrame({
            "PIN": [101.1, 102.2, 103.3],
            "Profession": ["Doctor", "Engineer", "Nurse"]
        })
        data.to_csv(os.path.join(chunk_dir, "chunk.csv"), index=False)

        config = {
            "number_of_chunks": 1,
            "k": 2,
            "max_number_of_eq_classes": 10,
            "suppression_limit": 0.1,
            "chunk_directory": chunk_dir,
            "output_path": os.path.join(tmp_dir, "output.csv"),
            "log_file": os.path.join(tmp_dir, "log.txt"),
            "save_output": True,
            "quasi_identifiers": {
                "categorical": [{"column": "Profession"}],
                "numerical": [{"column": "PIN", "encode": True, "type": "float"}]
            },
            "bin_width_multiplication_factor": {"PIN": 2},
            "hardcoded_min_max": {}
        }
        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        final_rf, elapsed = run_pipeline(config_path=config_path)
        assert isinstance(final_rf, list)
        assert os.path.exists(config["output_path"])


def test_invalid_column_encoding():
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_dir = os.path.join(tmp_dir, "datachunks")
        os.makedirs(chunk_dir)

        pd.DataFrame({"Age": [1, 2, 3]}).to_csv(os.path.join(chunk_dir, "chunk.csv"), index=False)

        config = {
            "number_of_chunks": 1,
            "k": 2,
            "max_number_of_eq_classes": 10,
            "suppression_limit": 0.1,
            "chunk_directory": chunk_dir,
            "output_path": os.path.join(tmp_dir, "out.csv"),
            "log_file": os.path.join(tmp_dir, "log.txt"),
            "save_output": True,
            "quasi_identifiers": {
                "categorical": [],
                "numerical": [{"column": "InvalidColumn", "encode": True, "type": "int"}]
            },
            "bin_width_multiplication_factor": {"InvalidColumn": 2},
            "hardcoded_min_max": {}
        }

        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(Exception):
            run_pipeline(config_path=config_path)



def test_save_encoding_map_failure():
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_dir = os.path.join(tmp_dir, "datachunks")
        os.makedirs(chunk_dir)

        df = pd.DataFrame({"PIN": [100, 101, 102]})
        df.to_csv(os.path.join(chunk_dir, "chunk.csv"), index=False)

        config = {
            "number_of_chunks": 1,
            "k": 2,
            "max_number_of_eq_classes": 100,
            "suppression_limit": 0.1,
            "chunk_directory": chunk_dir,
            "output_path": os.path.join(tmp_dir, "out.csv"),
            "log_file": os.path.join(tmp_dir, "log.txt"),
            "save_output": True,
            "quasi_identifiers": {
                "categorical": [],
                "numerical": [{"column": "PIN", "encode": True, "type": "int"}]
            },
            "bin_width_multiplication_factor": {"PIN": 2},
            "hardcoded_min_max": {}
        }

        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Patch open to simulate IOError when saving encoding map
        with patch("builtins.open", side_effect=IOError("Failed to write")):
            try:
                run_pipeline(config_path=config_path)
            except IOError:
                pass
