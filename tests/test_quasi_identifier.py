import pandas as pd
import pytest
from chunkanon.quasi_identifier import QuasiIdentifier  # Adjust the import if needed

def test_numerical_qi_initialization_and_range():
    qi = QuasiIdentifier(column_name="Age", is_categorical=False, min_value=20, max_value=50)
    assert qi.column_name == "Age"
    assert not qi.is_categorical
    assert qi.min_value == 20
    assert qi.max_value == 50
    assert qi.get_range() == 30

def test_categorical_qi_initialization_and_range():
    qi = QuasiIdentifier(column_name="Blood Group", is_categorical=True)
    assert qi.column_name == "Blood Group"
    assert qi.is_categorical
    assert qi.min_value == 1
    assert qi.max_value == 4  # CATEGORICAL_RANGES + 1
    assert qi.get_range() == 3  # Value from CATEGORICAL_RANGES

def test_unknown_categorical_qi():
    qi = QuasiIdentifier(column_name="City", is_categorical=True)
    assert qi.column_name == "City"
    assert qi.is_categorical
    assert qi.get_range() == 0

def test_update_min_max_on_numeric_chunk():
    qi = QuasiIdentifier(column_name="PIN", is_categorical=False)
    chunk = pd.DataFrame({"PIN": [1000, 2000, 3000]})
    qi.update_min_max(chunk)
    assert qi.min_value == 1000
    assert qi.max_value == 3000
    assert qi.get_range() == 2000

    # Update with more data
    new_chunk = pd.DataFrame({"PIN": [500, 4000]})
    qi.update_min_max(new_chunk)
    assert qi.min_value == 500
    assert qi.max_value == 4000
    assert qi.get_range() == 3500

def test_update_min_max_on_categorical_chunk():
    qi = QuasiIdentifier(column_name="Blood Group", is_categorical=True)
    chunk = pd.DataFrame({"Blood Group": ["A+", "O+", "B+"]})
    old_min = qi.min_value
    old_max = qi.max_value
    qi.update_min_max(chunk)
    # Min and max should not change
    assert qi.min_value == old_min
    assert qi.max_value == old_max
