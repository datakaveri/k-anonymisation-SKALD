import pytest
import pandas as pd
from chunkanon.generalization_rf import OLA_2
from types import SimpleNamespace
import os
import json

@pytest.fixture
def mock_quasi_identifiers(tmp_path):
    # Simulate categorical and numerical QIs
    return [
        SimpleNamespace(column_name="Blood Group", is_categorical=True),
        SimpleNamespace(column_name="Profession", is_categorical=True),
        SimpleNamespace(column_name="Age", is_categorical=False, min_value=0, get_range=lambda: 100, is_encoded=False),
        SimpleNamespace(column_name="PIN_encoded", is_categorical=False, min_value=0, get_range=lambda: 128, is_encoded=True)
    ]

@pytest.fixture
def multiplication_factors():
    return {
        "Age": 2,
        "PIN": 2  # for encoded column, key is "PIN"
    }

@pytest.fixture
def sample_chunk():
    return pd.DataFrame({
        "Blood Group": ["A+", "B+", "O+", "AB+"],
        "Profession": ["Doctor", "Engineer", "Artist", "Lawyer"],
        "Age": [25, 35, 45, 55],
        "PIN_encoded": [0, 32, 64, 96]
    })

@pytest.fixture
def encoded_decoding_file(tmp_path):
    encoding_data = {str(i): i+560000 for i in range(0, 129)}
    enc_dir = tmp_path / "encodings"
    enc_dir.mkdir()
    file_path = enc_dir / "pin_encoded_encoding.json"
    with open(file_path, "w") as f:
        json.dump(encoding_data, f)
    os.makedirs("encodings", exist_ok=True)
    with open("encodings/pin_encoded_encoding.json", "w") as f:
        json.dump(encoding_data, f)
    yield file_path
    os.remove("encodings/pin_encoded_encoding.json")

def test_tree_building(mock_quasi_identifiers, multiplication_factors):
    ola = OLA_2(mock_quasi_identifiers, 100, 0.1, multiplication_factors)
    initial_ri = [1, 1, 5, 2]
    tree = ola.build_tree(initial_ri)
    assert isinstance(tree, list)
    assert len(tree) > 0

def test_process_chunk(mock_quasi_identifiers, multiplication_factors, sample_chunk):
    ola = OLA_2(mock_quasi_identifiers, 4, 0.2, multiplication_factors)
    bin_widths = [1, 1, 10, 32]
    histogram = ola.process_chunk(sample_chunk, bin_widths)
    assert isinstance(histogram, dict)
    assert all(isinstance(k, tuple) for k in histogram.keys())

def test_merge_equivalence_classes(mock_quasi_identifiers, multiplication_factors, sample_chunk):
    ola = OLA_2(mock_quasi_identifiers, 4, 0.2, multiplication_factors)
    bin_widths = [1, 1, 10, 32]
    histogram = ola.process_chunk(sample_chunk, bin_widths)
    merged = ola.merge_equivalence_classes(histogram, [2, 1, 20, 64])
    assert isinstance(merged, dict)
    assert all(isinstance(k, tuple) for k in merged)

def test_generalize_chunk(mock_quasi_identifiers, multiplication_factors, sample_chunk, encoded_decoding_file):
    ola = OLA_2(mock_quasi_identifiers, 4, 0.2, multiplication_factors)
    bin_widths = [1, 1, 10, 32]
    generalized = ola.generalize_chunk(sample_chunk, bin_widths)
    assert isinstance(generalized, pd.DataFrame)
    assert "Age" in generalized.columns
    assert "PIN_encoded" in generalized.columns

def test_check_k_anonymity(mock_quasi_identifiers, multiplication_factors, sample_chunk):
    ola = OLA_2(mock_quasi_identifiers, 4, 0.2, multiplication_factors)
    histogram = ola.process_chunk(sample_chunk, [1, 1, 10, 32])
    assert not ola.check_k_anonymity(histogram, 2)

def test_merge_histograms(mock_quasi_identifiers, multiplication_factors, sample_chunk):
    ola = OLA_2(mock_quasi_identifiers, 4, 0.2, multiplication_factors)
    h1 = ola.process_chunk(sample_chunk, [1, 1, 10, 32])
    h2 = ola.process_chunk(sample_chunk, [1, 1, 10, 32])
    merged = ola.merge_histograms([h1, h2])
    assert isinstance(merged, dict)
    assert all(isinstance(v, int) for v in merged.values())

def test_get_suppressed_percent(mock_quasi_identifiers, multiplication_factors, sample_chunk):
    ola = OLA_2(mock_quasi_identifiers, 4, 0.5, multiplication_factors)
    histogram = ola.process_chunk(sample_chunk, [1, 1, 10, 32])
    percent = ola.get_suppressed_percent([1, 1, 10, 32], histogram, 2)
    assert isinstance(percent, float)
    assert 0 <= percent <= 100
