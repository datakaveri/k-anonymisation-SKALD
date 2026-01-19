import pytest
from decimal import Decimal
from SKALD.config_validation import load_config, NumericalQuasiIdentifier, Config
from pydantic import ValidationError
import yaml
import tempfile
import os


# -------------------------------------------------------------------------
# ✅ 1. BASE VALID CONFIG (YOUR ORIGINAL TEST)
# -------------------------------------------------------------------------
def test_load_valid_yaml_config():
    cfg_path = "tests/data/config_valid.yaml"
    config = load_config(cfg_path)

    assert config.enable_k_anonymity is True
    assert config.suppression_limit == Decimal("0.01")
    assert config.key_directory == "keys"
    assert config.log_file == "log.txt"

    assert "FULLNAMEENGLISH" in config.suppress
    assert "URN" in config.encrypt

    categorical_columns = [cat_qi.column for cat_qi in config.quasi_identifiers.categorical]
    numerical_columns_info = [
        {"column": num_qi.column, "encode": num_qi.encode, "type": num_qi.type}
        for num_qi in config.quasi_identifiers.numerical
    ]

    num_qi = numerical_columns_info[0]
    assert num_qi["column"] == "AGE"
    assert num_qi["encode"] is False
    assert num_qi["type"] == "int"

    assert categorical_columns[0] == "GENDER"
    assert config.k == 20
    assert config.l == 2
    assert config.hardcoded_min_max["AGE"] == [19, 85]


# -------------------------------------------------------------------------
# ✅ 2. INVALID YAML → RAISES
# -------------------------------------------------------------------------
def test_invalid_yaml_config_raises():
    with pytest.raises(Exception):
        load_config("tests/data/config_invalid.yaml")


# -------------------------------------------------------------------------
# ✅ 3. MISSING FILE → FileNotFoundError
# -------------------------------------------------------------------------
def test_missing_yaml_file():
    with pytest.raises(FileNotFoundError):
        load_config("tests/data/nope.yaml")


# -------------------------------------------------------------------------
# ✅ 4. NumericalQID type must be "int" or "float"
# -------------------------------------------------------------------------
def test_invalid_qi_type_raises():
    bad_yaml = {
        "output_path": "x.csv",
        "output_directory": "out",
        "key_directory": "keys",
        "log_file": "log.txt",
        "k": 5,
        "l": 2,
        "enable_k_anonymity": True,
        "quasi_identifiers": {
            "numerical": [{"column": "AGE", "encode": False, "type": "wrong"}],
            "categorical": []
        }
    }

    with pytest.raises(ValidationError):
        Config(**bad_yaml)


# -------------------------------------------------------------------------
# ✅ 5. Float QI must have encode=True
# -------------------------------------------------------------------------
def test_float_type_without_encode_fails():
    bad_yaml = {
        "output_path": "x.csv",
        "output_directory": "out",
        "key_directory": "keys",
        "log_file": "log.txt",
        "k": 5,
        "l": 2,
        "enable_k_anonymity": True,
        "quasi_identifiers": {
            "numerical": [{"column": "BMI", "encode": False, "type": "float"}],
            "categorical": []
        }
    }

    with pytest.raises(ValidationError):
        Config(**bad_yaml)


# -------------------------------------------------------------------------
# ✅ 6. k-anonymity disabled → No k/l/QI required
# -------------------------------------------------------------------------
def test_disable_k_anonymity_allows_missing_k_l():
    yaml_cfg = {
        "enable_k_anonymity": False,
        "output_path": "out.csv",
        "output_directory": "out",
        "key_directory": "keys",
        "log_file": "log.txt",
        "suppression_limit": 0,
        "suppress": [],
        "pseudonymize": [],
        "encrypt": [],
    }

    cfg = Config(**yaml_cfg)
    assert cfg.enable_k_anonymity is False
    assert cfg.k is None
    assert cfg.l is None
    assert cfg.quasi_identifiers is None


# -------------------------------------------------------------------------
# ✅ 7. Missing k/l when K-anonymity enabled → Validation error
# -------------------------------------------------------------------------
def test_missing_k_or_l_when_k_anonymity_enabled():
    bad_yaml = {
        "enable_k_anonymity": True,
        "output_path": "out.csv",
        "output_directory": "out",
        "key_directory": "keys",
        "log_file": "log.txt",
        "quasi_identifiers": {
            "numerical": [{"column": "AGE", "encode": False, "type": "int"}],
            "categorical": []
        },
        # missing k and l
    }

    with pytest.raises(ValidationError):
        Config(**bad_yaml)


# -------------------------------------------------------------------------
# ✅ 8. Multiplication factor must be >1
# -------------------------------------------------------------------------
def test_invalid_multiplication_factor():
    bad_yaml = {
        "enable_k_anonymity": True,
        "output_path": "out.csv",
        "output_directory": "out",
        "key_directory": "keys",
        "log_file": "log.txt",
        "k": 5,
        "l": 2,
        "quasi_identifiers": {
            "numerical": [{"column": "AGE", "encode": False, "type": "int"}],
            "categorical": []
        },
        "bin_width_multiplication_factor": {"AGE": 1},  # invalid
    }

    with pytest.raises(ValidationError):
        Config(**bad_yaml)


# -------------------------------------------------------------------------
# ✅ 9. Minimal valid config (K disabled)
# -------------------------------------------------------------------------
def test_minimal_valid_config_no_k():
    yaml_cfg = {
        "enable_k_anonymity": False,
        "output_path": "out.csv",
        "output_directory": "out",
        "key_directory": "keys",
        "log_file": "log.txt",
    }

    cfg = Config(**yaml_cfg)
    assert cfg.enable_k_anonymity is False


# -------------------------------------------------------------------------
# ✅ 10. Load config using temporary YAML file
# -------------------------------------------------------------------------
def test_temp_yaml_loading():
    temp = {
        "enable_k_anonymity": False,
        "output_path": "x.csv",
        "output_directory": "out",
        "key_directory": "keys",
        "log_file": "log.txt",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(temp, f)
        path = f.name

    cfg = load_config(path)
    assert cfg.output_path == "x.csv"
    os.remove(path)
