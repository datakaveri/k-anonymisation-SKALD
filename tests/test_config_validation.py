import unittest
import os
import tempfile
import yaml
from chunkanon.config_validation import load_config
from pydantic import ValidationError


class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        # Create temporary directories and files to simulate real paths
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        self.chunk_dir = os.path.join(self.temp_dir.name, "datachunks")
        os.mkdir(self.chunk_dir)

        # Valid config dictionary
        self.valid_config = {
            "number_of_chunks": 1,
            "k": 10,
            "max_number_of_eq_classes": 15000000,
            "suppression_limit": 0.001,
            "chunk_directory": self.chunk_dir,
            "output_path": os.path.join(self.temp_dir.name, "output.csv"),
            "log_file": os.path.join(self.temp_dir.name, "log.txt"),
            "save_output": True,
            "quasi_identifiers": {
                "numerical": [
                    {"column": "Age", "encode": True, "type": "int"},
                    {"column": "BMI", "encode": True, "type": "float"},
                    {"column": "PIN Code", "encode": True, "type": "int"}
                ],
                "categorical": [
                    {"column": "Blood Group"},
                    {"column": "Profession"}
                ]
            },
            "bin_width_multiplication_factor": {
                "Age": 2,
                "BMI": 2,
                "PIN Code": 2
            },
            "hardcoded_min_max": {
                "Age": [19, 85],
                "BMI": [12.7, 35.8],
                "PIN Code": [560001, 591346]
            }
        }

    def write_config(self, config_dict):
        with open(self.config_path, "w") as f:
            yaml.dump(config_dict, f)

    def test_valid_config(self):
        self.write_config(self.valid_config)
        config = load_config(self.config_path)
        self.assertEqual(config.k, 10)
        self.assertEqual(config.chunk_directory, self.chunk_dir)

    def test_invalid_suppression_limit(self):
        self.valid_config["suppression_limit"] = 1.5  # Invalid
        self.write_config(self.valid_config)
        with self.assertRaises(ValidationError):
            load_config(self.config_path)

    def test_missing_chunk_directory(self):
        self.valid_config["chunk_directory"] = "/nonexistent/path"
        self.write_config(self.valid_config)
        with self.assertRaises(ValidationError):
            load_config(self.config_path)

    def test_invalid_bin_width_factor(self):
        self.valid_config["bin_width_multiplication_factor"]["Age"] = 1  # Invalid (should be >1)
        self.write_config(self.valid_config)
        with self.assertRaises(ValidationError):
            load_config(self.config_path)

    def test_invalid_encode_for_float(self):
        # Set 'encode' to False for a 'float' type, which should raise an error
        self.valid_config["quasi_identifiers"]["numerical"][1]["encode"] = False  # BMI with type 'float' and encode False
        self.write_config(self.valid_config)
        with self.assertRaises(ValidationError):
            load_config(self.config_path)

    def tearDown(self):
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
